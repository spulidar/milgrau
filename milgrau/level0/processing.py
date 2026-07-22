"""Group-level Level 0 processing helpers."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from milgrau.io.filesystem import ensure_directories
from milgrau.io.licel import parse_licel_group
from milgrau.io.paths import level0_output_path, measurement_save_id
from milgrau.io.weather import fetch_surface_weather
from milgrau.level0.netcdf import build_level0_netcdf
from milgrau.operations import ExecutionResult
from milgrau.provenance import ProductProvenance, build_product_provenance, write_provenance_manifest


def measurement_group_provenance(
    meas_id: str,
    group_df: pd.DataFrame,
    config: Mapping[str, Any],
) -> ProductProvenance:
    """Build Level 0 provenance from every measurement and associated dark file."""
    input_paths = [Path(path) for path in group_df["filepath"].tolist()]
    provenance_columns = [
        column
        for column in (
            "filepath",
            "meas_type",
            "association_method",
            "dark_current_association_delta_hours",
        )
        if column in group_df
    ]
    source_records = group_df[provenance_columns].copy()
    if "filepath" in source_records:
        source_records["filepath"] = source_records["filepath"].map(lambda value: Path(value).name)
        source_records = source_records.sort_values("filepath")
    return build_product_provenance(
        "level0",
        input_paths,
        config,
        variant={"measurement_id": meas_id, "sources": source_records.to_dict(orient="records")},
    )


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
) -> ExecutionResult:
    """Process one measurement group into a Level 0 product."""
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
                f"No measurement files found for {save_id}",
                output_path=netcdf_path,
                metadata={"pipeline": "LIBIDS", "save_id": save_id},
            )

        stage = "level0.provenance"
        provenance = measurement_group_provenance(meas_id, group_df, config)
        stage = "level0.weather"
        weather_data = fetch_group_weather(group_df, config, logger)
        stage = "level0.parse"
        lidar_data_tensors = parse_licel_group(files_meas, logger)
        if not lidar_data_tensors.get("tensors"):
            return ExecutionResult.skipped(
                stage,
                f"No valid lidar tensors parsed for {save_id}",
                input_path=files_meas[0],
                output_path=netcdf_path,
                metadata={"pipeline": "LIBIDS", "save_id": save_id},
            )

        stage = "level0.write"
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
        stage = "level0.provenance.write"
        write_provenance_manifest(netcdf_path, provenance)
        return ExecutionResult.success(
            "level0.complete",
            f"NetCDF successfully generated for {save_id}",
            input_path=files_meas[0],
            output_path=netcdf_path,
            duration_seconds=time.perf_counter() - started_at,
            metadata={"pipeline": "LIBIDS", "save_id": save_id, "file_count": len(files_meas)},
        )
    except Exception as exc:
        return ExecutionResult.failure(
            stage,
            f"Level 0 conversion failed for {save_id}",
            input_path=None if not files_meas else files_meas[0],
            output_path=netcdf_path,
            cause=exc,
            include_traceback=True,
            duration_seconds=time.perf_counter() - started_at,
            metadata={"pipeline": "LIBIDS", "save_id": save_id},
        )
