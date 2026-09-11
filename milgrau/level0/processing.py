"""Group-level Level 0 processing helpers."""

from __future__ import annotations

from copy import deepcopy
import logging
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from milgrau.config.station import apply_station_context, resolve_station_context, select_lidar_channels
from milgrau.io.filesystem import ensure_directories
from milgrau.io.licel import parse_licel_group
from milgrau.io.logging_utils import bind_log_context
from milgrau.io.paths import level0_output_path, level0_scc_output_path, measurement_save_id
from milgrau.io.weather import fetch_surface_weather
from milgrau.level0.config import resolve_level0_config, station_coordinates
from milgrau.level0.netcdf import build_level0_netcdf
from milgrau.operations import ExecutionResult


def fetch_group_weather(group_df: pd.DataFrame, config: Mapping[str, Any], logger: logging.Logger) -> dict[str, Any]:
    """Fetch surface weather according to the explicit Level 0 missing-data policy."""
    level0 = resolve_level0_config(config)
    lat, lon = station_coordinates(config)
    dt_utc_mean = group_df["start_time_utc"].iloc[len(group_df) // 2].to_pydatetime()
    weather_logger = bind_log_context(logger, stage="weather")
    weather_data = fetch_surface_weather(dt_utc_mean, lat, lon, logger=weather_logger, config=config)
    if weather_data:
        weather_logger.info(
            "%.1f C | %.1f hPa",
            float(weather_data["temperature_c"]),
            float(weather_data["pressure_hpa"]),
        )
        return weather_data

    if level0.surface_weather.missing_policy == "fail":
        raise RuntimeError(
            "Surface weather is unavailable and level0.surface_weather.missing_policy='fail'."
        )

    weather_logger.warning("unavailable -> NaN (policy=nan)")
    return {
        "temperature_c": np.nan,
        "pressure_hpa": np.nan,
        "relative_humidity_percent": np.nan,
        "cloud_cover_percent": np.nan,
        "wind_speed_kmh": np.nan,
    }


def _resolve_group_station_config(
    group_df: pd.DataFrame,
    period: str,
    lidar_data: Mapping[str, Any],
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve station metadata while preserving every valid Licel channel."""
    if not isinstance(config.get("_station_catalog"), Mapping):
        return dict(config), dict(lidar_data), {}

    measurement_rows = group_df[group_df["meas_type"] == "measurements"]
    if measurement_rows.empty:
        raise ValueError("Cannot resolve station profile without measurement rows.")
    measurement_time = pd.to_datetime(measurement_rows["start_time_utc"], utc=True).min().to_pydatetime()
    context = resolve_station_context(
        config,
        measurement_time=measurement_time,
        period=period,
        available_channels=lidar_data.get("channels", []),
    )
    effective_config = apply_station_context(config, context)
    station_logger = bind_log_context(logger, stage="station")

    if context.get("scc_available", False):
        station_logger.info(
            "profile=%s | SCC=%s",
            context["profile_id"],
            context["scc_configuration_id"],
        )
        station_logger.debug(
            "mode=%s selected_channels=%d SCC_channels=%d extra_channels=%s missing_SCC_channels=%s",
            context["mode"],
            len(context["selected_channels"]),
            len(context.get("scc_channels", [])),
            ",".join(context["extra_channels"]) or "none",
            ",".join(context["missing_scc_channels"]) or "none",
        )
        if context["missing_scc_channels"]:
            station_logger.warning(
                "SCC export disabled | missing=%s",
                ",".join(context["missing_scc_channels"]),
            )
    else:
        station_logger.info("profile=%s | SCC=none", context["profile_id"])
        station_logger.debug("selected_channels=%d", len(context["selected_channels"]))
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
    meas_id: str,
    save_id: str,
    period: str,
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
    scc_path = level0_scc_output_path(meas_id, effective_config)
    ensure_directories(scc_path.parent)
    build_level0_netcdf(
        netcdf_path=str(scc_path),
        save_id=save_id,
        period=period,
        lidar_data=scc_lidar,
        group_df=group_df,
        weather_data=dict(weather_data),
        config=dict(effective_config),
        logger=logger,
    )
    scc_logger.info(
        "%s | %d/%d channels | config=%s",
        scc_path.name,
        len(scc_channels),
        len(lidar_data.get("channels", [])),
        context["scc_configuration_id"],
    )
    return scc_path


def process_measurement_group(
    meas_id: str,
    group_df: pd.DataFrame,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> ExecutionResult:
    """Process one measurement group into full-channel and optional SCC Level 0 products."""
    started_at = time.perf_counter()
    save_id = measurement_save_id(meas_id)
    netcdf_path = level0_output_path(meas_id, config)
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
                metadata={"pipeline": "L0", "save_id": save_id},
            )

        stage = "level0.parse"
        lidar_data_tensors = parse_licel_group(files_meas, logger)
        if not lidar_data_tensors.get("tensors"):
            return ExecutionResult.skipped(
                stage,
                "No valid lidar tensors parsed",
                input_path=files_meas[0],
                output_path=netcdf_path,
                metadata={"pipeline": "L0", "save_id": save_id},
            )
        bind_log_context(logger, stage="parse").debug(
            "files=%d channels=%d",
            len(files_meas),
            len(lidar_data_tensors.get("channels", [])),
        )

        stage = "level0.station"
        period = meas_id[8:]
        effective_config, lidar_data_tensors, station_context = _resolve_group_station_config(
            group_df, period, lidar_data_tensors, config, logger
        )

        stage = "level0.weather"
        weather_data = fetch_group_weather(group_df, effective_config, logger)

        stage = "level0.write"
        ensure_directories(out_dir)
        primary_config = _internal_level0_config(effective_config)
        build_level0_netcdf(
            netcdf_path=str(netcdf_path),
            save_id=save_id,
            period=period,
            lidar_data=lidar_data_tensors,
            group_df=group_df,
            weather_data=weather_data,
            config=primary_config,
            logger=logger,
        )

        stage = "level0.scc_export"
        scc_path = _write_scc_export(
            meas_id=meas_id,
            save_id=save_id,
            period=period,
            lidar_data=lidar_data_tensors,
            group_df=group_df,
            weather_data=weather_data,
            effective_config=effective_config,
            context=station_context,
            logger=logger,
        )

        result_metadata = {
            "pipeline": "L0",
            "save_id": save_id,
            "file_count": len(files_meas),
            "level0_channel_count": len(lidar_data_tensors.get("channels", [])),
        }
        resolved_station = effective_config.get("_resolved_station")
        if isinstance(resolved_station, Mapping):
            result_metadata["station_profile"] = resolved_station["profile_id"]
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
            metadata={"pipeline": "L0", "save_id": save_id},
        )
