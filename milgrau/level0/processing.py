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


def _resolve_group_station_config(
    group_df: pd.DataFrame,
    lidar_data: Mapping[str, Any],
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve station metadata while preserving every valid Licel channel."""
    if not isinstance(config.get("_station_catalog"), Mapping):
        raise KeyError("Level 0 processing requires a loaded station catalog.")
    measurement_rows = group_df[group_df["meas_type"] == "measurements"]
    if measurement_rows.empty:
        raise ValueError("Cannot resolve station profile without measurement rows.")
    measurement_time = pd.to_datetime(measurement_rows["start_time_utc"], utc=True).min().to_pydatetime()
    channels = lidar_data.get("channels", [])
    context = resolve_station_context(
        config,
        measurement_time=measurement_time,
        available_channels=channels,
    )

    # The primary MILGRAU session remains continuous across day/night. Until the
    # solar-regime refactor owns SCC mode selection, never export one SCC file
    # using a mode that is not homogeneous over the complete session.
    mode_samples = list(pd.to_datetime(measurement_rows["start_time_utc"], utc=True))
    if "stop_time" in measurement_rows and measurement_rows["stop_time"].notna().any():
        final_stop = pd.to_datetime(measurement_rows["stop_time"], utc=True).max()
        mode_samples.append(final_stop - pd.Timedelta(microseconds=1))
    scc_modes = {
        str(
            resolve_station_context(
                config,
                measurement_time=pd.Timestamp(timestamp).to_pydatetime(),
                available_channels=channels,
            )["mode"]
        )
        for timestamp in mode_samples
    }
    context = deepcopy(dict(context))
    context["scc_modes_present"] = sorted(scc_modes)
    context["scc_session_mode_homogeneous"] = len(scc_modes) <= 1
    if len(scc_modes) > 1:
        context["scc_export_ready"] = False

    effective_config = deepcopy(dict(config))
    effective_config["_resolved_station"] = deepcopy(dict(context))
    station_logger = bind_log_context(logger, stage="station")
    if context.get("scc_available", False):
        station_logger.info("%s | SCC %s", context["profile_id"], context["scc_configuration_id"])
        station_logger.debug(
            "mode=%s | calibration=%s | selected=%d | SCC=%d | extra=%s | missing=%s",
            context["mode"],
            context["calibration_id"],
            len(context["selected_channels"]),
            len(context.get("scc_channels", [])),
            ",".join(context["extra_channels"]) or "none",
            ",".join(context["missing_scc_channels"]) or "none",
        )
        if not context.get("scc_session_mode_homogeneous", True):
            station_logger.warning(
                "SCC export disabled | session spans modes=%s",
                ",".join(context.get("scc_modes_present", [])),
            )
        elif context["missing_scc_channels"]:
            station_logger.warning("SCC export disabled | missing=%s", ",".join(context["missing_scc_channels"]))
    else:
        station_logger.info("%s | SCC none", context["profile_id"])
        station_logger.debug("calibration=%s | selected=%d", context["calibration_id"], len(context["selected_channels"]))
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


def _write_scc_export(
    session_id: str,
    lidar_data: Mapping[str, Any],
    group_df: pd.DataFrame,
    weather_data: Mapping[str, Any],
    effective_config: Mapping[str, Any],
    context: Mapping[str, Any],
    logger: logging.Logger,
) -> Path | None:
    """Write an SCC-compatible channel subset derived from the full Licel Level 0."""
    if not context.get("scc_available", False) or not context.get("scc_export_ready", False):
        return None
    scc_logger = bind_log_context(logger, stage="scc")
    scc_channels = [str(channel) for channel in context.get("scc_channels", [])]
    if not scc_channels:
        scc_logger.warning("mapping configured but no SCC channels present")
        return None
    scc_lidar = select_lidar_channels(lidar_data, scc_channels)
    scc_path = level0_scc_output_path(session_id, effective_config)
    ensure_directories(scc_path.parent)
    build_level0_netcdf(
        netcdf_path=str(scc_path),
        session_id=session_id,
        lidar_data=scc_lidar,
        group_df=group_df,
        weather_data=dict(weather_data),
        config=dict(effective_config),
        logger=scc_logger,
    )
    write_netcdf_provenance(scc_path, effective_config)
    scc_logger.info(
        "%s | %d/%d channels | config %s",
        scc_path.name,
        len(scc_channels),
        len(lidar_data.get("channels", [])),
        context["scc_configuration_id"],
    )
    return scc_path


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
        df_meas = group_df[group_df["meas_type"] == "measurements"]
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
        scc_path = _write_scc_export(
            session_id=session_id,
            lidar_data=lidar_data_tensors,
            group_df=group_df,
            weather_data=weather_data,
            effective_config=effective_config,
            context=station_context,
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
            result_metadata["scc_available"] = bool(resolved_station.get("scc_available", False))
            result_metadata["scc_export_ready"] = bool(station_context.get("scc_export_ready", False))
            if resolved_station.get("scc_configuration_id") is not None:
                result_metadata["scc_configuration_id"] = resolved_station["scc_configuration_id"]
            if scc_path is not None:
                result_metadata["scc_export_path"] = str(scc_path)
                result_metadata["scc_channel_count"] = len(station_context.get("scc_channels", []))
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
