"""LIPANCORA Level 1 pipeline orchestration."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr

from milgrau.io.contracts import validate_level1_contract
from milgrau.io.filesystem import ensure_directories
from milgrau.io.paths import processed_data_root
from milgrau.level1.common import (
    diagnostic_vector,
    get_channel_constant,
    incremental_enabled,
    level0_dark_current_available,
    level1_output_path,
)
from milgrau.level1.corrections import apply_instrumental_corrections
from milgrau.level1.diagnostics import finalize_correction_dataset
from milgrau.level1.ingestion import load_and_prepare_level0
from milgrau.level1.thermodynamics import estimate_pbl_timeseries, integrate_thermodynamics

DEFAULT_SPEED_OF_LIGHT_M_S = 299792458.0


def _physics_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the physics configuration section."""
    physics = config.get("physics", {})
    return physics if isinstance(physics, Mapping) else {}


def _speed_of_light_m_s(config: Mapping[str, Any]) -> float:
    """Return the configured speed of light constant."""
    physics = _physics_config(config)
    return float(physics.get("speed_of_light", physics.get("speed_of_light_m_s", DEFAULT_SPEED_OF_LIGHT_M_S)))


def _bin_time_us(z_arr: np.ndarray, config: Mapping[str, Any]) -> float:
    """Return the bin integration time derived from the altitude grid."""
    if len(z_arr) < 2:
        raise ValueError("Altitude grid must contain at least two bins.")
    dz = float(z_arr[1] - z_arr[0])
    if dz <= 0.0 or not np.isfinite(dz):
        raise ValueError(f"Invalid altitude step: {dz}")
    return (2.0 * dz / _speed_of_light_m_s(config)) * 1e6


def _accumulated_shots(ds: xr.Dataset) -> float:
    """Return the accumulated laser shots stored in the Level 0 product."""
    shots = float(ds.attrs.get("Accumulated_Shots", np.nan))
    if not np.isfinite(shots) or shots <= 0.0:
        raise ValueError(f"Invalid Accumulated_Shots attribute: {shots}")
    return shots


def _background_mask(ds: xr.Dataset, channel_index: int, channel_name: str, logger: logging.Logger) -> xr.DataArray:
    """Return the background-selection mask for one channel."""
    bg_low = float(ds["Background_Low"].isel(channel=channel_index))
    bg_high = float(ds["Background_High"].isel(channel=channel_index))
    bg_mask = (ds["altitude"] >= bg_low) & (ds["altitude"] <= bg_high)
    if int(bg_mask.sum().values) < 2:
        logger.warning(f"  -> Channel {channel_name}: background mask has fewer than 2 bins ({bg_low:.1f}-{bg_high:.1f} m).")
    return bg_mask


def _dark_current_profile(
    ds: xr.Dataset,
    channel_index: int,
) -> tuple[xr.DataArray | None, xr.DataArray | None, bool]:
    """Return dark-current mean profile and uncertainty for one channel when available."""
    if not level0_dark_current_available(ds, channel_index):
        return None, None, False

    dc_data = ds["Background_Profile"].isel(channel=channel_index)
    if dc_data.sizes.get("time_bck", 0) <= 0:
        return None, None, False

    dc_mean = dc_data.mean(dim="time_bck", skipna=True)
    if not np.isfinite(dc_mean.values).any():
        return None, None, False

    dc_err = dc_data.std(dim="time_bck", skipna=True).rename({"altitude": "range"}) / np.sqrt(max(ds.sizes.get("time_bck", 1), 1))
    return dc_mean.rename({"altitude": "range"}), dc_err, True


def _channel_result_dataset(
    channel_name: str,
    corrected: xr.DataArray,
    corrected_error: xr.DataArray,
    rcs: xr.DataArray,
    rcs_error: xr.DataArray,
    diagnostics: Mapping[str, Any],
) -> xr.Dataset:
    """Build the per-channel Level 1 dataset returned by the correction kernel."""
    return xr.Dataset(
        {
            "corrected_signal": corrected.rename({"range": "altitude"}).assign_coords(channel=channel_name).astype(np.float32),
            "corrected_signal_error": corrected_error.rename({"range": "altitude"}).assign_coords(channel=channel_name).astype(np.float32),
            "range_corrected_signal": rcs.rename({"range": "altitude"}).assign_coords(channel=channel_name).astype(np.float32),
            "range_corrected_signal_error": rcs_error.rename({"range": "altitude"}).assign_coords(channel=channel_name).astype(np.float32),
            "pc_saturation_mask": diagnostics["pc_saturation_mask"].rename({"range": "altitude"}).assign_coords(channel=channel_name).astype(np.int8),
        }
    )


def _channel_diagnostic_record(
    ds: xr.Dataset,
    channel_name: str,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the channel-level diagnostic record consumed by the finalizer."""
    return {
        "channel": channel_name,
        "deadtime_correction_applied": int(diagnostics["deadtime_correction_applied"]),
        "deadtime_clipping_fraction": diagnostic_vector(diagnostics, "deadtime_clipping_fraction", ds.time),
        "pc_saturation_fraction": diagnostic_vector(diagnostics, "pc_saturation_fraction", ds.time),
        "deadtime_min_denominator_observed": float(diagnostics["deadtime_min_denominator_observed"]),
        "deadtime_min_denominator_allowed": float(diagnostics["deadtime_min_denominator_allowed"]),
        "bin_shift_bins": int(diagnostics["bin_shift_bins"]),
        "bin_shift_invalid_fraction": diagnostic_vector(diagnostics, "bin_shift_invalid_fraction", ds.time),
    }


def _correct_single_channel(
    ds: xr.Dataset,
    z_da: xr.DataArray,
    channel_index: int,
    channel_name: str,
    shots: float,
    bin_time_us: float,
    channels_config: Mapping[str, Any],
    logger: logging.Logger,
) -> tuple[xr.Dataset, dict[str, Any], bool]:
    """Apply Level 1 corrections to one channel and return result, diagnostics and DC usage."""
    sig = ds["Raw_Lidar_Data"].isel(channel=channel_index)
    bg_mask = _background_mask(ds, channel_index, channel_name, logger)
    deadtime, shift, bg_offset = get_channel_constant(channels_config, channel_name, logger)
    is_photon = "pc" in channel_name.lower() or "ph" in channel_name.lower()
    dc_prof, dc_err, dark_current_used = _dark_current_profile(ds, channel_index)

    corrected, corrected_error, rcs, rcs_error, diagnostics = apply_instrumental_corrections(
        sig=sig.rename({"altitude": "range"}),
        z_da=z_da,
        shots=shots,
        bin_time_us=bin_time_us,
        deadtime=deadtime,
        shift=shift,
        bg_offset=bg_offset,
        is_photon=is_photon,
        bg_mask=bg_mask.rename({"altitude": "range"}),
        dc_prof=dc_prof,
        dc_err=dc_err,
        return_diagnostics=True,
    )
    channel_dataset = _channel_result_dataset(channel_name, corrected, corrected_error, rcs, rcs_error, diagnostics)
    diagnostic_record = _channel_diagnostic_record(ds, channel_name, diagnostics)
    return channel_dataset, diagnostic_record, dark_current_used


def _processing_metadata(input_file: Path) -> dict[str, str]:
    """Return standardized Level 1 processing metadata."""
    return {
        "Processing_level": (
            "Level 1: PC counts->MHz, DeadTime, PC saturation mask, Dark Current, "
            "Bin Shift, Background subtraction, corrected signal, Range Corrected Signal, "
            "uncertainty propagation, PBL, Radiosonde, Tropopause"
        ),
        "Pipeline": "MILGRAU/LIPANCORA",
        "Input_Level0_File": input_file.name,
        "Altitude_units": "m",
    }


def _level1_encoding(ds: xr.Dataset) -> dict[str, dict[str, int | bool]]:
    """Return the NetCDF encoding used for Level 1 outputs."""
    return {var: {"zlib": True, "complevel": 4} for var in ds.data_vars if ds[var].ndim > 0}


def _discover_level0_files(config: Mapping[str, Any]) -> list[Path]:
    """Return all candidate Level 0 NetCDF files under processed_data."""
    in_dir = processed_data_root(config)
    return [path for path in sorted(in_dir.rglob("*.nc")) if "level" not in path.name]


def _files_requiring_level1(files: list[Path], config: Mapping[str, Any], logger: logging.Logger) -> tuple[list[Path], int]:
    """Filter candidate Level 0 files according to incremental mode."""
    incremental = incremental_enabled(config)
    files_to_process: list[Path] = []
    skipped_count = 0
    for file_path in files:
        output_path = level1_output_path(file_path, config)
        if incremental and output_path.exists():
            logger.info(f"[SKIPPED] Level 1 already exists for {file_path.name}: {output_path}")
            skipped_count += 1
            continue
        files_to_process.append(file_path)
    return files_to_process, skipped_count


def apply_all_physical_corrections(
    ds: xr.Dataset,
    z_arr: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> xr.Dataset:
    """Apply Level 1 instrumental corrections to all available channels."""
    channels_config = _physics_config(config).get("channels", {})
    bin_time_us = _bin_time_us(z_arr, config)
    shots = _accumulated_shots(ds)
    z_da = xr.DataArray(z_arr, dims=["range"], attrs={"units": "m"})
    channel_datasets = []
    status_records = []
    diagnostic_records = []
    logger.info("  -> Running instrumental corrections channel-by-channel...")

    for ch_idx, ch_name in enumerate(ds.channel.values.astype(str)):
        try:
            channel_dataset, diagnostic_record, dark_current_used = _correct_single_channel(
                ds=ds,
                z_da=z_da,
                channel_index=ch_idx,
                channel_name=ch_name,
                shots=shots,
                bin_time_us=bin_time_us,
                channels_config=channels_config,
                logger=logger,
            )
            channel_datasets.append(channel_dataset)
            status_records.append((ch_name, 1, int(dark_current_used)))
            diagnostic_records.append(diagnostic_record)
            clip_fraction = float(diagnostic_record["deadtime_clipping_fraction"].max(skipna=True).values)
            if clip_fraction > 0.0:
                logger.warning(f"  -> Channel {ch_name}: dead-time denominator clipped in up to {100.0 * clip_fraction:.2f}% of bins.")
            logger.info(f"  -> Channel {ch_name}: corrected successfully.")
        except Exception as exc:
            status_records.append((ch_name, 0, 0))
            logger.warning(f"  -> Channel {ch_name} failed during correction: {exc}")

    if not channel_datasets:
        raise RuntimeError("All channels failed during instrumental correction.")

    final_ds = xr.concat(channel_datasets, dim="channel")
    return finalize_correction_dataset(final_ds, status_records, diagnostic_records)


def process_single_file(args: tuple[str | Path, Mapping[str, Any], logging.Logger]) -> str:
    """Process one Level 0 NetCDF into a Level 1 RCS NetCDF product."""
    nc_path, config, logger = args
    try:
        nc_file = Path(nc_path)
        stem = nc_file.stem
        save_path = level1_output_path(nc_file, config)
        logger.info(f"[{stem}] Initializing Level 1 processing...")
        ds_raw, z_arr = load_and_prepare_level0(nc_file, logger)
        final_ds = apply_all_physical_corrections(ds_raw, z_arr, config, logger)
        final_ds = estimate_pbl_timeseries(final_ds, z_arr, config, logger)
        final_ds = integrate_thermodynamics(final_ds, config, logger)
        final_ds.attrs.update(ds_raw.attrs)
        final_ds.attrs.update(_processing_metadata(nc_file))
        validate_level1_contract(final_ds)
        ensure_directories(save_path.parent)
        final_ds.to_netcdf(save_path, encoding=_level1_encoding(final_ds))
        return f"[OK] {stem} Level 1 generated successfully: {save_path}"
    except Exception:
        return f"[FAILED] {nc_path} execution halted:\n{traceback.format_exc()}"


def process_level_1(config: Mapping[str, Any], logger: logging.Logger) -> None:
    """Discover and process every Level 0 NetCDF file into Level 1."""
    in_dir = processed_data_root(config)
    files = _discover_level0_files(config)
    if not files:
        logger.warning(f"No Level 0 files found in {in_dir}. Exiting.")
        return

    files_to_process, skipped_count = _files_requiring_level1(files, config, logger)

    if not files_to_process:
        logger.info(f"No Level 0 files require Level 1 processing. Skipped {skipped_count} existing products.")
        return

    logger.info(f"Found {len(files_to_process)} Level 0 files to process ({skipped_count} skipped).")
    for file_path in files_to_process:
        logger.info(process_single_file((str(file_path), config, logger)))
