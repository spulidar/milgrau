"""Group-level Level 0 processing helpers."""

from __future__ import annotations

from copy import deepcopy
import logging
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from milgrau.config.station import resolve_station_context, select_lidar_channels
from milgrau.io.filesystem import ensure_directories
from milgrau.io.licel import parse_licel_group
from milgrau.io.logging_utils import bind_log_context
from milgrau.io.paths import level0_output_path, level0_scc_output_path
from milgrau.io.weather import fetch_surface_weather
from milgrau.level0.config import resolve_level0_config, station_coordinates
from milgrau.level0.netcdf import build_level0_netcdf
from milgrau.physics.solar import (
    SOLAR_POSITION_ALGORITHM,
    build_solar_segments,
    solar_elevation_deg,
    solar_regime,
)
from milgrau.operations import ExecutionResult
from milgrau.provenance import write_netcdf_provenance


_SURFACE_WEATHER_FIELDS = (
    "temperature_c",
    "pressure_hpa",
    "relative_humidity_percent",
    "cloud_cover_percent",
    "wind_speed_kmh",
)


def _surface_weather_hours(group_df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return hourly UTC source times bracketing the complete measurement session."""
    measurement_rows = group_df[group_df["meas_type"] == "measurements"]
    if measurement_rows.empty:
        raise ValueError("Surface weather requires at least one measurement row.")
    start = pd.to_datetime(measurement_rows["start_time_utc"], utc=True).min()
    if "stop_time" in measurement_rows and measurement_rows["stop_time"].notna().any():
        stop = pd.to_datetime(measurement_rows["stop_time"], utc=True).max()
    else:
        stop = pd.to_datetime(measurement_rows["start_time_utc"], utc=True).max()
    first_hour = start.floor("h")
    last_hour = stop.ceil("h")
    if last_hour < first_hour:
        last_hour = first_hour
    return pd.date_range(first_hour, last_hour, freq="1h", tz="UTC")


def fetch_group_weather(group_df: pd.DataFrame, config: Mapping[str, Any], logger: logging.Logger) -> dict[str, Any]:
    """Fetch the hourly surface-weather series covering one continuous session."""
    level0 = resolve_level0_config(config)
    lat, lon = station_coordinates(config)
    weather_logger = bind_log_context(logger, stage="weather")
    times = _surface_weather_hours(group_df)
    values = {field: [] for field in _SURFACE_WEATHER_FIELDS}
    missing_times: list[pd.Timestamp] = []

    for timestamp in times:
        weather = fetch_surface_weather(
            timestamp.to_pydatetime(),
            lat,
            lon,
            logger=weather_logger,
            config=config,
        )
        if weather is None:
            missing_times.append(timestamp)
            for field in _SURFACE_WEATHER_FIELDS:
                values[field].append(np.nan)
            continue
        for field in _SURFACE_WEATHER_FIELDS:
            try:
                values[field].append(float(weather[field]))
            except (KeyError, TypeError, ValueError):
                values[field].append(np.nan)

    if missing_times and level0.surface_weather.missing_policy == "fail":
        rendered = ", ".join(item.strftime("%Y-%m-%dT%H:%MZ") for item in missing_times)
        raise RuntimeError(
            "Surface weather is unavailable for required hourly source time(s) "
            f"{rendered} and level0.surface_weather.missing_policy='fail'."
        )

    finite_temperature = np.asarray(values["temperature_c"], dtype=np.float64)
    finite_pressure = np.asarray(values["pressure_hpa"], dtype=np.float64)
    valid = np.isfinite(finite_temperature) & np.isfinite(finite_pressure)
    weather_logger.info(
        "%d/%d hourly records | coverage %s -> %s",
        int(np.count_nonzero(valid)),
        len(times),
        times[0].strftime("%Y-%m-%d %H:%MZ"),
        times[-1].strftime("%Y-%m-%d %H:%MZ"),
    )
    if missing_times:
        weather_logger.warning(
            "%d hourly record(s) unavailable | NaN preserved (policy=nan)",
            len(missing_times),
        )

    return {
        "weather_time": times.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]"),
        "source": "Open-Meteo Archive API",
        "cadence": "hourly",
        **{
            field: np.asarray(field_values, dtype=np.float64)
            for field, field_values in values.items()
        },
    }


def _annotate_solar_context(
    group_df: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Attach per-profile solar elevation, regime, and segment identity."""
    result = group_df.copy()
    measurement_mask = result["meas_type"] == "measurements"
    measurement_rows = result.loc[measurement_mask].copy()
    if measurement_rows.empty:
        raise ValueError("Solar context requires at least one measurement row.")

    measurement_rows = measurement_rows.sort_values("start_time_utc")
    starts = pd.to_datetime(measurement_rows["start_time_utc"], utc=True)
    stops = pd.to_datetime(measurement_rows["stop_time"], utc=True)
    midpoint = starts + (stops - starts) / 2
    latitude, longitude = station_coordinates(config)
    level0 = resolve_level0_config(config)
    elevation = solar_elevation_deg(midpoint, latitude, longitude)
    regimes = solar_regime(
        elevation,
        day_night_threshold_deg=level0.solar_regime.day_night_threshold_deg,
    )
    segment_ids, _segments = build_solar_segments(starts, stops, regimes)

    result["solar_elevation_deg"] = np.nan
    result["solar_regime"] = None
    result["segment_id"] = None
    result["_profile_index"] = np.nan
    result.loc[measurement_rows.index, "solar_elevation_deg"] = elevation
    result.loc[measurement_rows.index, "solar_regime"] = regimes
    result.loc[measurement_rows.index, "segment_id"] = segment_ids
    result.loc[measurement_rows.index, "_profile_index"] = np.arange(
        len(measurement_rows), dtype=np.int64
    )
    result.attrs["solar_position_algorithm"] = SOLAR_POSITION_ALGORITHM
    result.attrs["solar_day_night_threshold_deg"] = float(
        level0.solar_regime.day_night_threshold_deg
    )
    return result


def _resolve_group_station_config(
    group_df: pd.DataFrame,
    lidar_data: Mapping[str, Any],
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve station/profile context without imposing one SCC mode on a session."""
    if not isinstance(config.get("_station_catalog"), Mapping):
        raise KeyError("Level 0 processing requires a loaded station catalog.")
    measurement_rows = group_df[group_df["meas_type"] == "measurements"]
    if measurement_rows.empty:
        raise ValueError("Cannot resolve station profile without measurement rows.")
    measurement_time = pd.to_datetime(
        measurement_rows["start_time_utc"], utc=True
    ).min().to_pydatetime()
    channels = lidar_data.get("channels", [])
    context = deepcopy(
        dict(
            resolve_station_context(
                config,
                measurement_time=measurement_time,
                available_channels=channels,
                mode=None,
            )
        )
    )
    regimes = sorted(
        {
            str(value)
            for value in measurement_rows["solar_regime"].dropna().unique()
        }
    )
    segments = [
        str(value)
        for value in measurement_rows["segment_id"].dropna().unique()
    ]
    context["solar_regimes_present"] = regimes
    context["solar_segments_present"] = segments

    effective_config = deepcopy(dict(config))
    effective_config["_resolved_station"] = deepcopy(dict(context))
    station_logger = bind_log_context(logger, stage="station")
    station_logger.info(
        "%s | calibration=%s | solar=%s | segments=%s",
        context["profile_id"],
        context["calibration_id"],
        ",".join(regimes) or "none",
        ",".join(segments) or "none",
    )
    return effective_config, dict(lidar_data), context

def _internal_level0_config(effective_config: Mapping[str, Any]) -> dict[str, Any]:
    """Disable SCC-only variables for the full-channel primary Level 0 product."""
    internal = deepcopy(dict(effective_config))
    resolved = internal.get("_resolved_station")
    if isinstance(resolved, Mapping):
        resolved_copy = deepcopy(dict(resolved))
        resolved_copy["scc_available"] = False
        resolved_copy["lr_input"] = {}
        internal["_resolved_station"] = resolved_copy
    return internal


def _select_lidar_profiles(
    lidar_data: Mapping[str, Any],
    profile_indices: np.ndarray,
) -> dict[str, Any]:
    """Subset parsed lidar tensors and per-profile metadata by time index."""
    indices = np.asarray(profile_indices, dtype=np.int64)
    result = deepcopy(dict(lidar_data))
    tensors = lidar_data.get("tensors", {})
    result["tensors"] = {
        str(channel): np.asarray(values)[indices, :]
        for channel, values in tensors.items()
    }
    if "laser_shots" in lidar_data:
        shots = np.asarray(lidar_data["laser_shots"])
        result["laser_shots"] = shots[indices, :]
    return result


def _segment_group_df(
    group_df: pd.DataFrame,
    segment_id: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return one solar-segment measurement subset plus shared dark-current rows."""
    measurement_rows = group_df[
        (group_df["meas_type"] == "measurements")
        & (group_df["segment_id"].astype(str) == str(segment_id))
    ].copy()
    if measurement_rows.empty:
        raise ValueError(f"Solar segment {segment_id!r} has no measurement rows.")
    measurement_rows = measurement_rows.sort_values("start_time_utc")
    indices = pd.to_numeric(
        measurement_rows["_profile_index"], errors="raise"
    ).astype(np.int64).to_numpy()
    dark_rows = group_df[group_df["meas_type"] == "dark_current"].copy()
    return pd.concat([measurement_rows, dark_rows], ignore_index=True), indices


def _weather_for_interval(
    weather_data: Mapping[str, Any],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> dict[str, Any]:
    """Subset hourly weather to source times bracketing one SCC solar segment."""
    times = pd.to_datetime(
        np.asarray(weather_data.get("weather_time", [])),
        utc=True,
    )
    if len(times) == 0:
        return dict(weather_data)
    lower = pd.Timestamp(start_utc).tz_convert("UTC").floor("h")
    upper = pd.Timestamp(end_utc).tz_convert("UTC").ceil("h")
    mask = (times >= lower) & (times <= upper)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        nearest = int(
            np.argmin(
                np.abs(
                    times.asi8
                    - pd.Timestamp(start_utc).tz_convert("UTC").value
                )
            )
        )
        indices = np.asarray([nearest], dtype=np.int64)

    result = dict(weather_data)
    result["weather_time"] = (
        times[indices]
        .tz_convert("UTC")
        .tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    for field in _SURFACE_WEATHER_FIELDS:
        values = np.asarray(weather_data.get(field, []), dtype=np.float64).reshape(-1)
        if values.size != len(times):
            raise ValueError(
                f"Surface weather field {field!r} has {values.size} values for "
                f"{len(times)} weather_time entries."
            )
        result[field] = values[indices]
    return result


def _write_scc_exports(
    session_id: str,
    lidar_data: Mapping[str, Any],
    group_df: pd.DataFrame,
    weather_data: Mapping[str, Any],
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> list[Path]:
    """Write one SCC-compatible Level 0 derivative per solar segment."""
    measurement_rows = group_df[group_df["meas_type"] == "measurements"].copy()
    if measurement_rows.empty:
        return []

    channels = [str(channel) for channel in lidar_data.get("channels", [])]
    outputs: list[Path] = []
    for segment_id in measurement_rows["segment_id"].dropna().astype(str).unique():
        segment_df, profile_indices = _segment_group_df(group_df, segment_id)
        regimes = segment_df.loc[
            segment_df["meas_type"] == "measurements", "solar_regime"
        ].dropna().astype(str).unique()
        if len(regimes) != 1:
            raise ValueError(
                f"Solar segment {segment_id!r} must contain exactly one regime; got {regimes.tolist()}."
            )
        regime = str(regimes[0])
        segment_start = pd.to_datetime(
            segment_df.loc[
                segment_df["meas_type"] == "measurements", "start_time_utc"
            ],
            utc=True,
        ).min()
        context = deepcopy(
            dict(
                resolve_station_context(
                    config,
                    measurement_time=segment_start.to_pydatetime(),
                    available_channels=channels,
                    mode=regime,
                )
            )
        )
        context["segment_id"] = str(segment_id)
        context["solar_regime"] = regime
        context["solar_day_night_threshold_deg"] = float(
            resolve_level0_config(config).solar_regime.day_night_threshold_deg
        )

        scc_logger = bind_log_context(
            logger,
            stage="scc",
            segment_id=str(segment_id),
        )
        if not context.get("scc_available", False):
            scc_logger.info("%s | no SCC mapping for station profile", regime)
            continue
        if not context.get("scc_export_ready", False):
            scc_logger.warning(
                "%s | SCC export skipped | missing=%s",
                regime,
                ",".join(context.get("missing_scc_channels", [])) or "unknown",
            )
            continue

        segment_config = deepcopy(dict(config))
        segment_config["_resolved_station"] = context
        scc_channels = [str(channel) for channel in context.get("scc_channels", [])]
        if not scc_channels:
            scc_logger.warning("%s | no SCC channels present", regime)
            continue

        segment_lidar = _select_lidar_profiles(lidar_data, profile_indices)
        scc_lidar = select_lidar_channels(segment_lidar, scc_channels)
        segment_stop = pd.to_datetime(
            segment_df.loc[
                segment_df["meas_type"] == "measurements", "stop_time"
            ],
            utc=True,
        ).max()
        segment_weather = _weather_for_interval(
            weather_data,
            segment_start,
            segment_stop,
        )
        scc_path = level0_scc_output_path(
            session_id,
            segment_config,
            segment_id=str(segment_id),
        )
        ensure_directories(scc_path.parent)
        build_level0_netcdf(
            netcdf_path=str(scc_path),
            session_id=session_id,
            lidar_data=scc_lidar,
            group_df=segment_df,
            weather_data=segment_weather,
            config=segment_config,
            logger=scc_logger,
        )
        write_netcdf_provenance(scc_path, segment_config)
        outputs.append(scc_path)
        scc_logger.info(
            "%s | %s | %d/%d channels | config %s",
            regime,
            scc_path.name,
            len(scc_channels),
            len(channels),
            context["scc_configuration_id"],
        )
    return outputs

def process_session_group(
    session_id: str,
    group_df: pd.DataFrame,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> ExecutionResult:
    """Process one continuous session into full-channel and optional SCC Level 0 products."""
    started_at = time.perf_counter()
    netcdf_path = level0_output_path(session_id, config)
    out_dir = netcdf_path.parent
    stage = "level0.measurements"
    files_meas: list[str] = []
    try:
        df_meas = group_df[group_df["meas_type"] == "measurements"].sort_values(
            "start_time_utc"
        )
        files_meas = df_meas["filepath"].tolist()
        if not files_meas:
            return ExecutionResult.skipped(
                stage,
                "No measurement files found",
                output_path=netcdf_path,
                metadata={"pipeline": "L0", "session_id": session_id},
            )
        stage = "level0.parse"
        parse_logger = bind_log_context(logger, stage="parse")
        lidar_data_tensors = parse_licel_group(files_meas, parse_logger)
        if not lidar_data_tensors.get("tensors"):
            return ExecutionResult.skipped(
                stage,
                "No valid lidar tensors parsed",
                input_path=files_meas[0],
                output_path=netcdf_path,
                metadata={"pipeline": "L0", "session_id": session_id},
            )
        parse_logger.debug(
            "files=%d | channels=%d", len(files_meas), len(lidar_data_tensors.get("channels", []))
        )
        stage = "level0.solar"
        group_df = _annotate_solar_context(group_df, config)
        stage = "level0.station"
        effective_config, lidar_data_tensors, station_context = _resolve_group_station_config(
            group_df, lidar_data_tensors, config, logger
        )
        stage = "level0.weather"
        weather_data = fetch_group_weather(group_df, effective_config, logger)
        stage = "level0.write"
        ensure_directories(out_dir)
        primary_config = _internal_level0_config(effective_config)
        build_level0_netcdf(
            netcdf_path=str(netcdf_path),
            session_id=session_id,
            lidar_data=lidar_data_tensors,
            group_df=group_df,
            weather_data=weather_data,
            config=primary_config,
            logger=bind_log_context(logger, stage="write"),
        )
        provenance_attrs = write_netcdf_provenance(netcdf_path, primary_config)
        bind_log_context(logger, stage="provenance").debug(
            "MILGRAU=%s | config=%s | station=%s | profile=%s | calibration=%s",
            provenance_attrs.get("software_version", "-"),
            provenance_attrs.get("processing_configuration_file", "-"),
            provenance_attrs.get("station_configuration_file", "-"),
            provenance_attrs.get("station_profile_id", "-"),
            provenance_attrs.get("instrument_calibration_id", "-"),
        )
        stage = "level0.scc_export"
        scc_paths = _write_scc_exports(
            session_id=session_id,
            lidar_data=lidar_data_tensors,
            group_df=group_df,
            weather_data=weather_data,
            config=config,
            logger=logger,
        )
        result_metadata = {
            "pipeline": "L0",
            "session_id": session_id,
            "file_count": len(files_meas),
            "level0_channel_count": len(lidar_data_tensors.get("channels", [])),
        }
        resolved_station = effective_config.get("_resolved_station")
        if isinstance(resolved_station, Mapping):
            result_metadata["station_profile"] = resolved_station["profile_id"]
            result_metadata["instrument_calibration"] = resolved_station["calibration_id"]
            result_metadata["solar_regimes"] = list(
                station_context.get("solar_regimes_present", [])
            )
            result_metadata["solar_segments"] = list(
                station_context.get("solar_segments_present", [])
            )
            result_metadata["scc_export_count"] = len(scc_paths)
            if scc_paths:
                result_metadata["scc_export_paths"] = [str(path) for path in scc_paths]
        return ExecutionResult.success(
            "level0.complete",
            "Level 0 NetCDF generated",
            input_path=files_meas[0],
            output_path=netcdf_path,
            duration_seconds=time.perf_counter() - started_at,
            metadata=result_metadata,
        )
    except Exception as exc:
        return ExecutionResult.failure(
            stage,
            "Level 0 conversion failed",
            input_path=None if not files_meas else files_meas[0],
            output_path=netcdf_path,
            cause=exc,
            include_traceback=True,
            duration_seconds=time.perf_counter() - started_at,
            metadata={"pipeline": "L0", "session_id": session_id},
        )
