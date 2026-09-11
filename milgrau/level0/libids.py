"""LIBIDS Level 0 orchestration."""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.config.station import resolve_station_context
from milgrau.incremental import output_is_current
from milgrau.io.contracts import netcdf_satisfies_contract, validate_level0_contract
from milgrau.io.logging_utils import bind_log_context
from milgrau.io.paths import level0_output_path, level0_scc_output_path, measurement_save_id, raw_data_root
from milgrau.level0.common import incremental_enabled
from milgrau.level0.config import resolve_level0_config, validate_level0_config
from milgrau.level0.inventory import build_measurement_inventory
from milgrau.level0.processing import process_measurement_group
from milgrau.level0.quality import filter_laser_shots
from milgrau.operations import ExecutionResult, ExecutionStatus, ExecutionSummary


def _with_station_geometry(config: Mapping) -> dict:
    """Materialize station-owned pointing geometry into the legacy writer view."""
    catalog = config.get("_station_catalog")
    if not isinstance(catalog, Mapping):
        raise KeyError("No station catalog is loaded; Level 0 pointing geometry is station-owned.")
    station = catalog.get("station")
    if not isinstance(station, Mapping):
        raise KeyError("station is required in the station catalog.")
    geometry = station.get("lidar_geometry")
    if not isinstance(geometry, Mapping) or "pointing_angle_deg_from_zenith" not in geometry:
        raise KeyError("station.lidar_geometry.pointing_angle_deg_from_zenith is required.")
    angle = float(geometry["pointing_angle_deg_from_zenith"])
    if not np.isfinite(angle) or angle < 0.0 or angle > 180.0:
        raise ValueError("station.lidar_geometry.pointing_angle_deg_from_zenith must be finite and within 0..180 degrees.")
    resolved = deepcopy(dict(config))
    physics = resolved.setdefault("physics", {})
    if not isinstance(physics, dict):
        raise ValueError("Configuration physics compatibility view must be a mapping.")
    physics["laser_pointing_angle_deg"] = angle
    return resolved


def _raw_input_paths(group_df) -> list[Path]:
    return [Path(path) for path in group_df["filepath"].tolist()]


def _resolve_expected_scc_context(meas_id: str, group_df, config: Mapping, output_path: Path) -> dict | None:
    """Resolve SCC expectations from the full-channel primary Level 0."""
    if not isinstance(config.get("_station_catalog"), Mapping):
        return None
    measurement_rows = group_df[group_df["meas_type"] == "measurements"]
    if measurement_rows.empty:
        return None
    try:
        with xr.open_dataset(output_path) as ds:
            channels = [str(value) for value in ds["channel_string"].values]
        measurement_time = pd.to_datetime(measurement_rows["start_time_utc"], utc=True).min().to_pydatetime()
        context = resolve_station_context(
            config,
            measurement_time=measurement_time,
            period=meas_id[8:],
            available_channels=channels,
        )
    except Exception:
        return None
    return context


def _scc_output_satisfies_context(path: Path, context: Mapping) -> bool:
    """Validate Level 0 contract plus SCC-specific variables required by context."""
    if not netcdf_satisfies_contract(path, validate_level0_contract):
        return False
    try:
        with xr.open_dataset(path, mask_and_scale=False) as ds:
            if "channel_ID" not in ds:
                return False
            expected_channels = [str(value) for value in context.get("scc_channels", [])]
            actual_channels = [str(value) for value in ds["channel_string"].values]
            if actual_channels != expected_channels:
                return False
            expected_ids = np.asarray([int(context["channel_ids"][channel]) for channel in expected_channels], dtype=np.int64)
            actual_ids = np.asarray(ds["channel_ID"].values, dtype=np.int64)
            if actual_ids.shape != expected_ids.shape or not np.array_equal(actual_ids, expected_ids):
                return False
            lr_input = context.get("lr_input", {})
            if isinstance(lr_input, Mapping) and lr_input:
                if "LR_Input" not in ds:
                    return False
                values = np.ma.asarray(ds["LR_Input"].values)
                for index, channel in enumerate(actual_channels):
                    if channel in lr_input and (np.ma.is_masked(values[index]) or int(values[index]) != int(lr_input[channel])):
                        return False
        return True
    except Exception:
        return False


def _level0_is_current(meas_id: str, group_df, config: dict, output_path) -> bool:
    output = Path(output_path)
    inputs = _raw_input_paths(group_df)
    primary_current = output_is_current(
        output,
        inputs,
        config=config,
        integrity_check=lambda path: netcdf_satisfies_contract(path, validate_level0_contract),
    )
    if not primary_current:
        return False
    context = _resolve_expected_scc_context(meas_id, group_df, config, output)
    if not context or not (context.get("scc_available", False) and context.get("scc_export_ready", False)):
        return True
    scc_path = level0_scc_output_path(meas_id, config)
    return output_is_current(
        scc_path,
        inputs,
        config=config,
        integrity_check=lambda path: _scc_output_satisfies_context(path, context),
    )


def _normalize_requested_measurements(values: Sequence[str] | None) -> set[str] | None:
    if not values:
        return None
    normalized: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if len(value) == 12 and value[8:10] == "sa":
            value = value[:8] + value[10:]
        if len(value) != 10 or value[8:] not in {"am", "pm", "nt"} or not value[:8].isdigit():
            raise ValueError(f"LIBIDS input must be YYYYMMDDam/pm/nt or YYYYMMDDsaam/sapm/sant; got {raw!r}.")
        normalized.add(value)
    return normalized


def process_level_0(
    config: dict,
    logger: logging.Logger,
    *,
    inputs: Sequence[str] | None = None,
    force: bool = False,
) -> ExecutionSummary:
    """Run LIBIDS, optionally restricting processing to selected measurement IDs."""
    validate_level0_config(config)
    config = _with_station_geometry(config)
    level0_config = resolve_level0_config(config)
    pipeline_logger = bind_log_context(logger, pipeline="L0")
    requested = _normalize_requested_measurements(inputs)

    raw_dir = raw_data_root(config)
    df_raw = build_measurement_inventory(str(raw_dir), config, pipeline_logger)
    if requested is not None and not df_raw.empty:
        df_raw = df_raw[df_raw["meas_id"].astype(str).isin(requested)].copy()
        missing = sorted(requested - set(df_raw["meas_id"].astype(str).unique()))
        if missing:
            raise FileNotFoundError(f"Requested LIBIDS measurement group(s) not found: {', '.join(missing)}")
    if df_raw.empty:
        bind_log_context(pipeline_logger, stage="discovery").info("no raw measurements found")
        return ExecutionSummary.from_results(
            [ExecutionResult.skipped("level0.discovery", "No new data to process", input_path=raw_dir, metadata={"pipeline": "L0"})]
        )

    df_good = filter_laser_shots(
        df_raw,
        pipeline_logger,
        tolerance_fraction=level0_config.acquisition_qa.laser_shot_tolerance_fraction,
        header_time_jitter_s=level0_config.acquisition_qa.licel_header_time_jitter_s,
    )
    if df_good.empty:
        bind_log_context(pipeline_logger, stage="qa").warning("no data survived acquisition QA")
        return ExecutionSummary.from_results(
            [ExecutionResult.skipped("level0.quality", "No data survived quality control", input_path=raw_dir, metadata={"pipeline": "L0"})]
        )

    incremental = incremental_enabled(config)
    total_groups = len(df_good["meas_id"].unique())
    results: list[ExecutionResult] = []
    for meas_id, group_df in df_good.groupby("meas_id"):
        save_id = measurement_save_id(meas_id)
        group_logger = bind_log_context(pipeline_logger, save_id=save_id)
        netcdf_path = level0_output_path(meas_id, config)
        if not force and incremental and _level0_is_current(meas_id, group_df, config, netcdf_path):
            bind_log_context(group_logger, stage="skip").info("up to date | %s", netcdf_path.name)
            results.append(ExecutionResult.skipped("level0.incremental", "Level 0 is up to date", output_path=netcdf_path, metadata={"pipeline": "L0", "save_id": save_id}))
            continue

        measurement_count = int((group_df["meas_type"] == "measurements").sum())
        bind_log_context(group_logger, stage="start").info("%d raw files", measurement_count)
        try:
            result = process_measurement_group(meas_id, group_df, config, group_logger)
            if not isinstance(result, ExecutionResult):
                raise TypeError(f"process_measurement_group returned {type(result).__name__}; expected ExecutionResult.")
        except Exception as exc:
            result = ExecutionResult.failure("level0.group", "unexpected group conversion error", output_path=netcdf_path, cause=exc, include_traceback=True, metadata={"pipeline": "L0", "save_id": save_id})
        if result.status is ExecutionStatus.OK:
            duration = 0.0 if result.duration_seconds is None else result.duration_seconds
            bind_log_context(group_logger, stage="done").info("%s | %.1f s", netcdf_path.name, duration)
        elif result.status is ExecutionStatus.ERROR:
            bind_log_context(group_logger, stage=result.stage.removeprefix("level0.")).error("%s | %s", result.message, result.cause or "unknown failure")
            if result.traceback:
                group_logger.debug("failure traceback\n%s", result.traceback)
        results.append(result)

    summary = ExecutionSummary.from_results(results)
    counts = summary.counts
    bind_log_context(pipeline_logger, stage="summary").info(
        "groups=%d | processed=%d | skipped=%d | errors=%d",
        total_groups,
        counts[ExecutionStatus.OK],
        counts[ExecutionStatus.SKIPPED],
        counts[ExecutionStatus.ERROR],
    )
    return summary
