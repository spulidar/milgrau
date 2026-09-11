"""LIPANCORA Level 1 pipeline orchestration."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.incremental import output_is_current
from milgrau.io.contracts import netcdf_satisfies_contract, validate_level1_contract
from milgrau.io.filesystem import ensure_directories
from milgrau.io.logging_utils import bind_log_context
from milgrau.io.paths import processed_data_root, product_save_id
from milgrau.operations import ExecutionResult, ExecutionStatus, ExecutionSummary
from milgrau.level1.common import (
    diagnostic_vector,
    incremental_enabled,
    level0_dark_current_available,
    level1_output_path,
)
from milgrau.level1.config import (
    Level1Config,
    resolve_channel_calibration,
    resolve_level1_config,
    validate_level1_config,
)
from milgrau.level1.corrections import apply_instrumental_corrections
from milgrau.level1.diagnostics import finalize_correction_dataset
from milgrau.level1.ingestion import load_and_prepare_level0
from milgrau.level1.pbl import estimate_pbl_timeseries
from milgrau.level1.thermodynamics import integrate_thermodynamics

SPEED_OF_LIGHT_M_S: Final[float] = 299_792_458.0


def _bin_time_us(z_arr: np.ndarray) -> float:
    if len(z_arr) < 2:
        raise ValueError("Altitude grid must contain at least two bins.")
    dz = float(z_arr[1] - z_arr[0])
    if dz <= 0.0 or not np.isfinite(dz):
        raise ValueError(f"Invalid altitude step: {dz}")
    return (2.0 * dz / SPEED_OF_LIGHT_M_S) * 1e6


def _channel_laser_shots(ds: xr.Dataset, channel_index: int) -> xr.DataArray:
    shots = ds["Laser_Shots"].isel(channel=channel_index).astype(np.float64)
    if shots.dims != ("time",):
        raise ValueError(f"Laser_Shots channel slice must have dimensions ('time',); got {shots.dims}.")
    values = np.asarray(shots.values, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Laser_Shots contains non-finite or non-positive values for the selected channel.")
    return shots.assign_coords(time=ds.time)


def _native_channel_grid(ds: xr.Dataset, channel_index: int) -> np.ndarray:
    dz = float(ds["Raw_Data_Range_Resolution"].isel(channel=channel_index).values)
    if not np.isfinite(dz) or dz <= 0.0:
        raise ValueError(f"Invalid native range resolution for channel index {channel_index}: {dz}")
    return (np.arange(ds.sizes["altitude"], dtype=np.float64) + 0.5) * dz


def _background_mask(channel_name: str, z_da: xr.DataArray, level1_config: Level1Config) -> xr.DataArray:
    bg_low = level1_config.background.start_altitude_m
    bg_high = level1_config.background.stop_altitude_m
    bg_mask = (z_da >= bg_low) & (z_da <= bg_high)
    if int(bg_mask.sum().values) < 2:
        raise ValueError(
            f"Channel {channel_name}: configured Level 1 background window "
            f"{bg_low:.1f}-{bg_high:.1f} m contains fewer than 2 native bins."
        )
    return bg_mask


def _dark_current_profile(
    ds: xr.Dataset,
    channel_index: int,
    source_z_arr: np.ndarray,
) -> tuple[xr.DataArray | None, xr.DataArray | None, bool]:
    if not level0_dark_current_available(ds, channel_index):
        return None, None, False
    dc_data = ds["Background_Profile"].isel(channel=channel_index)
    if dc_data.sizes.get("time_bck", 0) <= 0:
        return None, None, False
    dc_mean = dc_data.mean(dim="time_bck", skipna=True)
    if not np.isfinite(dc_mean.values).any():
        return None, None, False
    count = max(ds.sizes.get("time_bck", 1), 1)
    dc_err = dc_data.std(dim="time_bck", skipna=True) / np.sqrt(count)
    dc_mean = dc_mean.rename({"altitude": "range"}).assign_coords(range=source_z_arr)
    dc_err = dc_err.rename({"altitude": "range"}).assign_coords(range=source_z_arr)
    return dc_mean, dc_err, True


def _same_grid(source: xr.DataArray, target_z_arr: np.ndarray) -> bool:
    source_range = np.asarray(source["range"].values, dtype=np.float64)
    return source_range.shape == target_z_arr.shape and np.allclose(source_range, target_z_arr, rtol=0.0, atol=1e-9)


def _interpolate_numeric(data: xr.DataArray, target_z_arr: np.ndarray) -> xr.DataArray:
    if _same_grid(data, target_z_arr):
        return data
    return data.interp(range=target_z_arr)


def _interpolate_mask(data: xr.DataArray, target_z_arr: np.ndarray) -> xr.DataArray:
    if _same_grid(data, target_z_arr):
        return data.astype(bool)
    return data.astype(np.float32).interp(range=target_z_arr, method="nearest").fillna(0.0) >= 0.5


def _channel_result_dataset(
    channel_name: str,
    corrected: xr.DataArray,
    corrected_error: xr.DataArray,
    rcs: xr.DataArray,
    rcs_error: xr.DataArray,
    diagnostics: Mapping[str, Any],
    target_z_arr: np.ndarray,
) -> xr.Dataset:
    corrected = _interpolate_numeric(corrected, target_z_arr)
    corrected_error = _interpolate_numeric(corrected_error, target_z_arr)
    rcs = _interpolate_numeric(rcs, target_z_arr)
    rcs_error = _interpolate_numeric(rcs_error, target_z_arr)
    pc_saturation_mask = _interpolate_mask(diagnostics["pc_saturation_mask"], target_z_arr)
    return xr.Dataset(
        {
            "corrected_signal": corrected.rename({"range": "altitude"}).assign_coords(channel=channel_name).astype(np.float32),
            "corrected_signal_error": corrected_error.rename({"range": "altitude"}).assign_coords(channel=channel_name).astype(np.float32),
            "range_corrected_signal": rcs.rename({"range": "altitude"}).assign_coords(channel=channel_name).astype(np.float32),
            "range_corrected_signal_error": rcs_error.rename({"range": "altitude"}).assign_coords(channel=channel_name).astype(np.float32),
            "pc_saturation_mask": pc_saturation_mask.rename({"range": "altitude"}).assign_coords(channel=channel_name).astype(np.int8),
        }
    )


def _channel_diagnostic_record(ds: xr.Dataset, channel_name: str, diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "channel": channel_name,
        "deadtime_correction_applied": int(diagnostics["deadtime_correction_applied"]),
        "deadtime_clipping_fraction": diagnostic_vector(diagnostics, "deadtime_clipping_fraction", ds.time),
        "pc_saturation_fraction": diagnostic_vector(diagnostics, "pc_saturation_fraction", ds.time),
        "deadtime_min_denominator_observed": float(diagnostics["deadtime_min_denominator_observed"]),
        "deadtime_min_denominator_allowed": float(diagnostics["deadtime_min_denominator_allowed"]),
        "pc_saturation_characterized": int(diagnostics["pc_saturation_characterized"]),
        "pc_saturation_rate_limit_mhz": float(diagnostics["pc_saturation_rate_limit_mhz"]),
        "bin_shift_bins": int(diagnostics["bin_shift_bins"]),
        "bin_shift_invalid_fraction": diagnostic_vector(diagnostics, "bin_shift_invalid_fraction", ds.time),
    }


def _correct_single_channel(
    ds: xr.Dataset,
    source_z_arr: np.ndarray,
    target_z_arr: np.ndarray,
    channel_index: int,
    channel_name: str,
    shots: xr.DataArray,
    bin_time_us: float,
    config: Mapping[str, Any],
    level1_config: Level1Config,
) -> tuple[xr.Dataset, dict[str, Any], bool]:
    calibration = resolve_channel_calibration(config, ds, channel_name)
    z_da = xr.DataArray(source_z_arr, dims=["range"], coords={"range": source_z_arr}, attrs={"units": "m"})
    sig = ds["Raw_Lidar_Data"].isel(channel=channel_index).rename({"altitude": "range"}).assign_coords(range=source_z_arr)
    bg_mask = _background_mask(channel_name, z_da, level1_config)
    is_photon = calibration.detector_mode == "photon_counting"
    dc_prof, dc_err, dark_current_used = _dark_current_profile(ds, channel_index, source_z_arr)
    corrected, corrected_error, rcs, rcs_error, diagnostics = apply_instrumental_corrections(
        sig=sig,
        z_da=z_da,
        shots=shots,
        bin_time_us=bin_time_us,
        deadtime=calibration.deadtime_us,
        shift=calibration.bin_shift_bins,
        bg_offset=calibration.background_offset,
        is_photon=is_photon,
        bg_mask=bg_mask,
        dc_prof=dc_prof,
        dc_err=dc_err,
        deadtime_min_denominator=level1_config.photon_counting.deadtime_min_denominator,
        pc_saturation_max_rate_mhz=calibration.saturation_max_rate_mhz,
        return_diagnostics=True,
    )
    channel_dataset = _channel_result_dataset(channel_name, corrected, corrected_error, rcs, rcs_error, diagnostics, target_z_arr)
    diagnostic_record = _channel_diagnostic_record(ds, channel_name, diagnostics)
    diagnostic_record["calibration_assumed_neutral"] = int(calibration.assumed_neutral)
    return channel_dataset, diagnostic_record, dark_current_used


def _processing_metadata(input_file: Path) -> dict[str, str]:
    return {
        "Processing_level": (
            "Level 1: PC counts->MHz using Laser_Shots(time,channel), calibrated DeadTime, explicit numerical clipping QA, "
            "physical PC saturation only when characterized, Dark Current, Bin Shift, configured Background subtraction, "
            "native-grid correction and common-grid interpolation, corrected signal, Range Corrected Signal, uncertainty "
            "propagation, PBL, canonical thermodynamic atmosphere, Tropopause"
        ),
        "Pipeline": "MILGRAU/LIPANCORA",
        "Input_Level0_File": input_file.name,
        "Altitude_units": "m",
        "Altitude_grid_convention": "range-bin centers on finest native Raw_Data_Range_Resolution",
    }


def _level1_encoding(ds: xr.Dataset) -> dict[str, dict[str, int | bool]]:
    return {var: {"zlib": True, "complevel": 4} for var in ds.data_vars if ds[var].ndim > 0}


def _make_level1_netcdf_safe(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds
    time_index = pd.to_datetime(ds["time"].values)
    if getattr(time_index, "tz", None) is not None:
        time_index = time_index.tz_convert("UTC").tz_localize(None)
    return ds.assign_coords(time=time_index.to_numpy())


def _discover_level0_files(config: Mapping[str, Any]) -> list[Path]:
    in_dir = processed_data_root(config)
    discovered: list[Path] = []
    for path in sorted(in_dir.rglob("*.nc")):
        if "level" in path.name:
            continue
        if any(part in {"quicklooks", "level2_qa"} for part in path.parts):
            continue
        if path.parent.name != path.stem:
            continue
        discovered.append(path)
    return discovered


def _files_requiring_level1(
    files: list[Path],
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> tuple[list[Path], list[ExecutionResult]]:
    incremental = incremental_enabled(config)
    files_to_process: list[Path] = []
    skipped_results: list[ExecutionResult] = []
    for file_path in files:
        save_id = product_save_id(file_path)
        output_path = level1_output_path(file_path, config)
        is_current = False
        if incremental and output_path.exists():
            is_current = output_is_current(
                output_path,
                [file_path],
                config=config,
                integrity_check=lambda path: netcdf_satisfies_contract(path, validate_level1_contract),
            )
        if is_current:
            bind_log_context(logger, save_id=save_id, stage="skip").info("up to date | %s", output_path.name)
            skipped_results.append(
                ExecutionResult.skipped(
                    "level1.incremental",
                    "Level 1 is up to date",
                    input_path=file_path,
                    output_path=output_path,
                    metadata={"pipeline": "L1", "save_id": save_id},
                )
            )
            continue
        files_to_process.append(file_path)
    return files_to_process, skipped_results


def apply_all_physical_corrections(
    ds: xr.Dataset,
    z_arr: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> xr.Dataset:
    """Apply corrections on native channel grids and log only aggregate operator events."""
    level1_config = resolve_level1_config(config)
    channel_datasets = []
    status_records = []
    diagnostic_records = []
    failed_channels: list[str] = []
    uncharacterized_pc: list[str] = []
    neutral_channels: list[str] = []
    clipped_channels: list[str] = []

    for ch_idx, ch_name in enumerate(ds.channel.values.astype(str)):
        channel_logger = bind_log_context(logger, stage=ch_name)
        try:
            source_z_arr = _native_channel_grid(ds, ch_idx)
            bin_time_us = _bin_time_us(source_z_arr)
            shots = _channel_laser_shots(ds, ch_idx)
            if not np.allclose(source_z_arr, z_arr, rtol=0.0, atol=1e-9):
                channel_logger.debug(
                    "native dz=%.6f m -> common dz=%.6f m",
                    source_z_arr[1] - source_z_arr[0],
                    z_arr[1] - z_arr[0],
                )
            channel_dataset, diagnostic_record, dark_current_used = _correct_single_channel(
                ds=ds,
                source_z_arr=source_z_arr,
                target_z_arr=z_arr,
                channel_index=ch_idx,
                channel_name=ch_name,
                shots=shots,
                bin_time_us=bin_time_us,
                config=config,
                level1_config=level1_config,
            )
            channel_datasets.append(channel_dataset)
            status_records.append((ch_name, 1, int(dark_current_used)))
            diagnostic_records.append(diagnostic_record)
            clip_fraction = float(diagnostic_record["deadtime_clipping_fraction"].max(skipna=True).values)
            if clip_fraction > 0.0:
                clipped_channels.append(f"{ch_name}({100.0 * clip_fraction:.2f}%)")
            if not bool(diagnostic_record["pc_saturation_characterized"]) and ch_name.upper().endswith(".PC"):
                uncharacterized_pc.append(ch_name)
            if bool(diagnostic_record["calibration_assumed_neutral"]):
                neutral_channels.append(ch_name)
            channel_logger.debug("corrected successfully | dark_current=%s", dark_current_used)
        except Exception as exc:
            status_records.append((ch_name, 0, 0))
            failed_channels.append(ch_name)
            channel_logger.debug("correction failed: %s", exc, exc_info=True)

    if not channel_datasets:
        raise RuntimeError("All channels failed during instrumental correction.")

    corrections_logger = bind_log_context(logger, stage="corrections")
    corrections_logger.info("%d/%d channels", len(channel_datasets), ds.sizes.get("channel", 0))
    if neutral_channels:
        bind_log_context(logger, stage="calibration").warning("neutral assumed: %s", ", ".join(neutral_channels))
    if clipped_channels:
        bind_log_context(logger, stage="deadtime").warning("clipped: %s", ", ".join(clipped_channels))
    if uncharacterized_pc:
        bind_log_context(logger, stage="saturation").warning("uncharacterized PC: %s", ", ".join(uncharacterized_pc))
    if failed_channels:
        corrections_logger.warning("failed: %s", ", ".join(failed_channels))

    final_ds = xr.concat(channel_datasets, dim="channel")
    return finalize_correction_dataset(final_ds, status_records, diagnostic_records)


def process_single_file(args: tuple[str | Path, Mapping[str, Any], logging.Logger]) -> ExecutionResult:
    nc_path, config, logger = args
    started_at = time.perf_counter()
    nc_file = Path(nc_path)
    save_id = product_save_id(nc_file)
    file_logger = bind_log_context(logger, save_id=save_id)
    save_path: Path | None = None
    stage = "level1.initialize"
    try:
        stage = "level1.configuration"
        validate_level1_config(config)
        stage = "level1.output_path"
        save_path = level1_output_path(nc_file, config)
        stage = "level1.ingestion"
        ds_raw, z_arr = load_and_prepare_level0(nc_file, bind_log_context(file_logger, stage="ingestion"))
        bind_log_context(file_logger, stage="start").info("%d channels", ds_raw.sizes.get("channel", 0))
        stage = "level1.corrections"
        final_ds = apply_all_physical_corrections(ds_raw, z_arr, config, file_logger)
        stage = "level1.pbl"
        final_ds = estimate_pbl_timeseries(final_ds, z_arr, config, bind_log_context(file_logger, stage="pbl"))
        stage = "level1.thermodynamics"
        final_ds = integrate_thermodynamics(final_ds, config, bind_log_context(file_logger, stage="atmosphere"))
        final_ds.attrs.update(ds_raw.attrs)
        final_ds.attrs.update(_processing_metadata(nc_file))
        final_ds = _make_level1_netcdf_safe(final_ds)
        stage = "level1.validation"
        validate_level1_contract(final_ds)
        stage = "level1.write"
        ensure_directories(save_path.parent)
        final_ds.to_netcdf(save_path, encoding=_level1_encoding(final_ds))
        return ExecutionResult.success(
            "level1.complete",
            "Level 1 generated",
            input_path=nc_file,
            output_path=save_path,
            duration_seconds=time.perf_counter() - started_at,
            metadata={"pipeline": "L1", "save_id": save_id, "channel_count": final_ds.sizes.get("channel", 0)},
        )
    except Exception as exc:
        return ExecutionResult.failure(
            stage,
            "Level 1 processing failed",
            input_path=nc_file,
            output_path=save_path,
            cause=exc,
            include_traceback=True,
            duration_seconds=time.perf_counter() - started_at,
            metadata={"pipeline": "L1", "save_id": save_id},
        )


def process_level_1(config: Mapping[str, Any], logger: logging.Logger) -> ExecutionSummary:
    validate_level1_config(config)
    in_dir = processed_data_root(config)
    files = _discover_level0_files(config)
    if not files:
        bind_log_context(logger, stage="discovery").warning("no Level 0 files found | %s", in_dir)
        return ExecutionSummary.from_results(
            [ExecutionResult.skipped("level1.discovery", "No Level 0 files found", input_path=in_dir, metadata={"pipeline": "L1"})]
        )
    files_to_process, skipped_results = _files_requiring_level1(files, config, logger)
    if not files_to_process:
        bind_log_context(logger, stage="summary").info("all Level 1 products are current")
        return ExecutionSummary.from_results(skipped_results)
    bind_log_context(logger, stage="queue").info(
        "%d files to process | %d skipped", len(files_to_process), len(skipped_results)
    )
    results = list(skipped_results)
    for file_path in files_to_process:
        save_id = product_save_id(file_path)
        file_logger = bind_log_context(logger, save_id=save_id)
        result = process_single_file((str(file_path), config, file_logger))
        if result.status is ExecutionStatus.SUCCESS:
            duration = 0.0 if result.duration_seconds is None else result.duration_seconds
            channel_count = int(result.metadata.get("channel_count", 0))
            bind_log_context(file_logger, stage="done").info(
                "%d channels | %s | %.1f s", channel_count, result.output_path.name if result.output_path else "no output", duration
            )
        elif result.status.is_failure:
            bind_log_context(file_logger, stage=result.stage.removeprefix("level1.")).error(
                "%s | %s", result.message, result.cause or "unknown failure"
            )
            if result.traceback:
                file_logger.debug("failure traceback\n%s", result.traceback)
        results.append(result)
    return ExecutionSummary.from_results(results)
