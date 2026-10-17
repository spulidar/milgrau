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
from milgrau.io.paths import level0_output_path
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


def _regime_group_df(
    group_df: pd.DataFrame,
    regime: str,
    month_utc: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...]]:
    """Return all measurement profiles from one solar regime plus shared dark rows."""
    regime_value = str(regime).strip().lower()
    if regime_value not in {"day", "night"}:
        raise ValueError("SCC solar regime must be 'day' or 'night'.")

    measurement_rows = group_df[
        (group_df["meas_type"] == "measurements")
        & (group_df["solar_regime"].astype(str).str.lower() == regime_value)
    ].copy()
    if month_utc is not None:
        starts = pd.to_datetime(measurement_rows["start_time_utc"], utc=True)
        measurement_rows = measurement_rows.loc[
            starts.dt.strftime("%Y%m") == month_utc
        ].copy()
    if measurement_rows.empty:
        raise ValueError(f"Solar regime {regime_value!r} has no measurement rows.")

    measurement_rows = measurement_rows.sort_values("start_time_utc")
    indices = pd.to_numeric(
        measurement_rows["_profile_index"], errors="raise"
    ).astype(np.int64).to_numpy()
    source_segments = tuple(
        dict.fromkeys(
            measurement_rows["segment_id"].dropna().astype(str).tolist()
        )
    )
    if not source_segments:
        raise ValueError(
            f"Solar regime {regime_value!r} has no source segment identity."
        )

    dark_rows = group_df[group_df["meas_type"] == "dark_current"].copy()
    return (
        pd.concat([measurement_rows, dark_rows], ignore_index=True),
        indices,
        source_segments,
    )


def _weather_for_measurement_rows(
    weather_data: Mapping[str, Any],
    measurement_rows: pd.DataFrame,
) -> dict[str, Any]:
    """Subset hourly weather to the union of actual SCC source segments."""
    times = pd.to_datetime(
        np.asarray(weather_data.get("weather_time", [])),
        utc=True,
    )
    if len(times) == 0:
        return dict(weather_data)
    if measurement_rows.empty:
        raise ValueError("SCC weather subsetting requires measurement rows.")

    mask = np.zeros(len(times), dtype=bool)
    ordered_segments = tuple(
        dict.fromkeys(
            measurement_rows["segment_id"].dropna().astype(str).tolist()
        )
    )
    for segment_id in ordered_segments:
        rows = measurement_rows[
            measurement_rows["segment_id"].astype(str) == segment_id
        ]
        if rows.empty:
            continue
        lower = pd.to_datetime(rows["start_time_utc"], utc=True).min().floor("h")
        upper = pd.to_datetime(rows["stop_time"], utc=True).max().ceil("h")
        mask |= np.asarray((times >= lower) & (times <= upper), dtype=bool)

    indices = np.flatnonzero(mask)
    if indices.size == 0:
        target = pd.to_datetime(
            measurement_rows["start_time_utc"], utc=True
        ).min()
        nearest = int(np.argmin(np.abs(times.asi8 - target.value)))
        indices = np.asarray([nearest], dtype=np.int64)

    result = dict(weather_data)
    result["weather_time"] = (
        times[indices]
        .tz_convert("UTC")
        .tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    for field in _SURFACE_WEATHER_FIELDS:
        values = np.asarray(
            weather_data.get(field, []), dtype=np.float64
        ).reshape(-1)
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
    """Stage SCC-only day/night × UTC-month submissions atomically in cache.

    A SCIENTIFIC session always remains continuous. One SCC Measurement_ID
    identifies precisely one staged submission, never a canonical output.
    """
    if not isinstance(config.get("_scc_catalog"), Mapping):
        return []

    from milgrau.level0.netcdf import scc_measurement_id
    from milgrau.level0.scc_lidar_ratio import write_scc_lidar_ratio_ancillary
    from milgrau.level0.scc_staging import (
        atomic_scc_package,
        pending_scc_package,
        validate_pending_scc_package,
        write_scc_manifest,
    )

    measurement_rows = group_df[group_df["meas_type"] == "measurements"].copy()
    if measurement_rows.empty:
        return []
    starts = pd.to_datetime(measurement_rows["start_time_utc"], utc=True)
    ends = pd.to_datetime(measurement_rows["stop_time"], utc=True)
    if (ends <= starts).any():
        raise ValueError("SCC export requires positive profile durations.")
    # A profile itself crossing a month cannot be associated with a single
    # constant monthly LR. Do not silently assign it to one month.
    if (
        starts.dt.strftime("%Y%m")
        != (ends - pd.Timedelta(nanoseconds=1)).dt.strftime("%Y%m")
    ).any():
        raise ValueError(
            "One raw SCC measurement profile crosses a UTC month boundary; "
            "split that raw measurement before SCC staging."
        )
    measurement_rows["scc_month_utc"] = starts.dt.strftime("%Y%m")
    regime_values = measurement_rows["solar_regime"].astype(str).str.lower()
    partitions = sorted({
        (str(regime), str(month))
        for regime, month in zip(
            regime_values, measurement_rows["scc_month_utc"], strict=True,
        )
    })
    outputs: list[Path] = []
    used_ids: set[str] = set()
    channels = [str(channel) for channel in lidar_data.get("channels", [])]
    canonical = level0_output_path(session_id, config)

    for regime, month in partitions:
        if regime not in {"day", "night"}:
            raise ValueError(f"Unsupported SCC solar regime: {regime!r}.")
        regime_df, profile_indices, source_segments = _regime_group_df(
            group_df, regime, month,
        )
        regime_measurements = regime_df[
            regime_df["meas_type"] == "measurements"
        ].copy()
        first = pd.to_datetime(regime_measurements["start_time_utc"], utc=True).min()
        last = pd.to_datetime(regime_measurements["stop_time"], utc=True).max()

        context = deepcopy(dict(resolve_station_context(
            config, measurement_time=first.to_pydatetime(),
            available_channels=channels, mode=regime,
        )))
        context["solar_regime"] = regime
        context["source_segments"] = ",".join(source_segments)
        context["source_segment_count"] = len(source_segments)
        context["contains_time_gaps"] = len(source_segments) > 1
        context["solar_day_night_threshold_deg"] = float(
            resolve_level0_config(config).solar_regime.day_night_threshold_deg
        )
        scc_logger = bind_log_context(logger, stage=f"scc.{regime}.{month}")
        if not context.get("scc_available", False):
            scc_logger.info("no SCC mapping for station profile")
            continue
        if not context.get("scc_export_ready", False):
            scc_logger.warning(
                "SCC export skipped, missing channels: %s",
                ",".join(context.get("missing_scc_channels", [])) or "unknown",
            )
            continue
        scc_channels = list(context.get("scc_channels", []))
        if not scc_channels:
            continue

        measurement_id = scc_measurement_id(first, str(context["station_id"]))
        if measurement_id in used_ids:
            raise ValueError(
                f"SCC Measurement_ID {measurement_id} collision across "
                "regime/month partitions of one scientific session."
            )
        used_ids.add(measurement_id)
        regime_config = deepcopy(dict(config))
        regime_config["_resolved_station"] = context
        regime_lidar = _select_lidar_profiles(lidar_data, profile_indices)
        scc_lidar = select_lidar_channels(regime_lidar, scc_channels)
        regime_weather = _weather_for_measurement_rows(
            weather_data, regime_measurements,
        )
        with atomic_scc_package(config, measurement_id) as staging:
            scc_path = staging / "raw.nc"
            lr_path = write_scc_lidar_ratio_ancillary(
                scc_path, regime_measurements, scc_lidar, regime_config, context,
            )
            if lr_path is not None:
                regime_config["_scc_lr_file_name"] = lr_path.name
            build_level0_netcdf(
                netcdf_path=str(scc_path),
                session_id=session_id,
                lidar_data=scc_lidar,
                group_df=regime_df,
                weather_data=regime_weather,
                config=regime_config,
                logger=scc_logger,
                scc_export=True,
            )
            write_netcdf_provenance(scc_path, regime_config)
            write_scc_manifest(
                staging,
                measurement_id=measurement_id,
                session_id=session_id,
                regime=regime,
                month_utc=month,
                config_id=context["scc_configuration_id"],
                product_ids=context["scc_product_ids"],
                source_segments=list(source_segments),
                profile_start_utc=first.isoformat(),
                profile_stop_utc=last.isoformat(),
                source_canonical_l0=canonical,
                config=config,
                ancillary_name=None if lr_path is None else lr_path.name,
                source_profile_indices=profile_indices.tolist(),
                source_measurement_files=regime_measurements["filepath"].astype(str).tolist(),
            )
        package = pending_scc_package(config, measurement_id)
        validate_pending_scc_package(package)
        outputs.append(package)
        scc_logger.info(
            "staged | %s | %d channels | config %s | segments %s",
            package, len(scc_channels), context["scc_configuration_id"],
            ",".join(source_segments),
        )
    return outputs

def process_session_group(
    session_id: str,
    group_df: pd.DataFrame,
    config: Mapping[str, Any],
    logger: logging.Logger,
    *,
    scc_only: bool = False,
) -> ExecutionResult:
    """Write a canonical scientific session, or re-export only its SCC upload bytes."""
    started_at = time.perf_counter()
    netcdf_path = level0_output_path(session_id, config)
    out_dir = netcdf_path.parent
    stage = "level0.measurements"
    files_meas: list[str] = []
    try:
        if scc_only:
            from milgrau.level0.libids import _primary_output_satisfies_contract

            if not isinstance(config.get("_scc_catalog"), Mapping):
                raise ValueError("SCC-only re-export requires scc.enabled: true.")
            if not _primary_output_satisfies_contract(netcdf_path):
                raise ValueError(
                    "Cannot SCC-only re-export: no valid canonical L0 exists "
                    f"at {netcdf_path}."
                )
        df_meas = group_df[group_df["meas_type"] == "measurements"]
        if not df_meas.empty:
            df_meas = df_meas.sort_values("start_time_utc")
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
        if not scc_only:
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
        scc_error = None
        try:
            scc_paths = _write_scc_exports(
                session_id=session_id,
                lidar_data=lidar_data_tensors,
                group_df=group_df,
                weather_data=weather_data,
                config=config,
                logger=logger,
            )
        except Exception as exc:
            if scc_only:
                raise
            scc_paths = []
            scc_error = str(exc)
            bind_log_context(logger, stage="scc.error").error(
                "canonical L0 saved, SCC staging failed: %s", exc,
            )
        result_metadata = {
            "pipeline": "L0",
            "session_id": session_id,
            "file_count": len(files_meas),
            "level0_channel_count": len(lidar_data_tensors.get("channels", [])),
        }
        if scc_error is not None:
            result_metadata["scc_staging_error"] = scc_error
        resolved_station = effective_config.get("_resolved_station")
        if isinstance(resolved_station, Mapping):
            result_metadata["station_profile"] = resolved_station["profile_id"]
            result_metadata["instrument_calibration"] = resolved_station["calibration_id"]
            result_metadata["solar_regimes"] = ",".join(
                str(value)
                for value in station_context.get("solar_regimes_present", [])
            )
            result_metadata["solar_segments"] = ",".join(
                str(value)
                for value in station_context.get("solar_segments_present", [])
            )
            result_metadata["scc_export_count"] = len(scc_paths)
            if scc_paths:
                result_metadata["scc_export_paths"] = ";".join(
                    str(path) for path in scc_paths
                )
        return ExecutionResult.success(
            "level0.scc_reexport" if scc_only else "level0.complete",
            "SCC-only packages staged" if scc_only else "Level 0 NetCDF generated",
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
