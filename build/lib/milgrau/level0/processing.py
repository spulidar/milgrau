"""Group-level Level 0 processing helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from milgrau.io.filesystem import ensure_directories
from milgrau.io.licel import parse_licel_group
from milgrau.io.paths import level0_output_path, measurement_save_id
from milgrau.io.weather import fetch_surface_weather
from milgrau.level0.netcdf import build_level0_netcdf


def fetch_group_weather(group_df: pd.DataFrame, config: Mapping[str, Any], logger: logging.Logger) -> dict[str, Any]:
    """Fetch surface weather for one measurement group with config fallback."""
    lat = float(config["physics"].get("latitude", -23.561))
    lon = float(config["physics"].get("longitude", -46.735))
    dt_utc_mean = group_df["start_time_utc"].iloc[len(group_df) // 2].to_pydatetime()
    weather_data = fetch_surface_weather(dt_utc_mean, lat, lon, logger=logger, config=config)
    if weather_data:
        return weather_data

    logger.warning("  -> Weather API/cache failed. Using fallback standard surface values.")
    return {
        "temperature_c": float(config["physics"].get("default_surface_temp_c", 25.0)),
        "pressure_hpa": float(config["physics"].get("default_surface_pressure_hpa", 940.0)),
        "relative_humidity_percent": np.nan,
        "cloud_cover_percent": np.nan,
        "wind_speed_kmh": np.nan,
    }


def process_measurement_group(
    meas_id: str,
    group_df: pd.DataFrame,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> tuple[bool, str]:
    """Process one measurement group into a Level 0 product."""
    save_id = measurement_save_id(meas_id)
    netcdf_path = level0_output_path(meas_id, config)
    out_dir = netcdf_path.parent

    df_meas = group_df[group_df["meas_type"] == "measurements"]
    files_meas = df_meas["filepath"].tolist()
    if not files_meas:
        return False, f"  -> [{save_id}] No measurement files found. Skipping."

    weather_data = fetch_group_weather(group_df, config, logger)
    lidar_data_tensors = parse_licel_group(files_meas, logger)
    if not lidar_data_tensors.get("tensors"):
        return False, f"  -> [{save_id}] No valid lidar tensors parsed. Skipping."

    ensure_directories(out_dir)
    build_level0_netcdf(
        netcdf_path=str(netcdf_path),
        save_id=save_id,
        period=meas_id[8:],
        lidar_data=lidar_data_tensors,
        group_df=group_df,
        weather_data=weather_data,
        config=config,
        logger=logger,
    )
    return True, f"  -> [OK] NetCDF successfully generated: {netcdf_path}"
