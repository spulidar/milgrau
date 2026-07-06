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
    diagnostic_vector as _diagnostic_vector,
    get_channel_constant as _get_channel_constant,
    incremental_enabled as _incremental_enabled,
    level0_dark_current_available as _level0_dark_current_available,
    level1_output_path,
)
from milgrau.level1.diagnostics import finalize_correction_dataset
from milgrau.level1.ingestion import load_and_prepare_level0
from milgrau.level1.thermodynamics import estimate_pbl_timeseries, integrate_thermodynamics
from milgrau.level1.corrections import apply_instrumental_corrections


def apply_all_physical_corrections(
    ds: xr.Dataset,
    z_arr: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> xr.Dataset:
    """Apply Level 1 instrumental corrections to all available channels."""
    channels_config = config.get("physics", {}).get("channels", {})
    c_speed = float(config.get("physics", {}).get("speed_of_light", config.get("physics", {}).get("speed_of_light_m_s", 299792458.0)))
    if len(z_arr) < 2:
        raise ValueError("Altitude grid must contain at least two bins.")
    dz = float(z_arr[1] - z_arr[0])
    if dz <= 0.0 or not np.isfinite(dz):
        raise ValueError(f"Invalid altitude step: {dz}")

    bin_time_us = (2.0 * dz / c_speed) * 1e6
    shots = float(ds.attrs.get("Accumulated_Shots", np.nan))
    if not np.isfinite(shots) or shots <= 0.0:
        raise ValueError(f"Invalid Accumulated_Shots attribute: {shots}")

    z_da = xr.DataArray(z_arr, dims=["range"], attrs={"units": "m"})
    channel_datasets = []
    status_records = []
    diagnostic_records = []
    logger.info("  -> Running instrumental corrections channel-by-channel...")

    for ch_idx, ch_name in enumerate(ds.channel.values.astype(str)):
        dark_current_used = False
        try:
            sig = ds["Raw_Lidar_Data"].isel(channel=ch_idx)
            bg_low = float(ds["Background_Low"].isel(channel=ch_idx))
            bg_high = float(ds["Background_High"].isel(channel=ch_idx))
            bg_mask = (ds["altitude"] >= bg_low) & (ds["altitude"] <= bg_high)
            if int(bg_mask.sum().values) < 2:
                logger.warning(f"  -> Channel {ch_name}: background mask has fewer than 2 bins ({bg_low:.1f}-{bg_high:.1f} m).")

            deadtime, shift, bg_offset = _get_channel_constant(channels_config, ch_name, logger)
            is_photon = "pc" in ch_name.lower() or "ph" in ch_name.lower()
            dc_prof, dc_err = None, None
            if _level0_dark_current_available(ds, ch_idx):
                dc_data = ds["Background_Profile"].isel(channel=ch_idx)
                if dc_data.sizes.get("time_bck", 0) > 0:
                    dc_mean = dc_data.mean(dim="time_bck", skipna=True)
                    if np.isfinite(dc_mean.values).any():
                        dc_prof = dc_mean.rename({"altitude": "range"})
                        dc_err = dc_data.std(dim="time_bck", skipna=True).rename({"altitude": "range"}) / np.sqrt(max(ds.sizes.get("time_bck", 1), 1))
                        dark_current_used = True

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

            ch_ds = xr.Dataset(
                {
                    "corrected_signal": corrected.rename({"range": "altitude"}).assign_coords(channel=ch_name).astype(np.float32),
                    "corrected_signal_error": corrected_error.rename({"range": "altitude"}).assign_coords(channel=ch_name).astype(np.float32),
                    "range_corrected_signal": rcs.rename({"range": "altitude"}).assign_coords(channel=ch_name).astype(np.float32),
                    "range_corrected_signal_error": rcs_error.rename({"range": "altitude"}).assign_coords(channel=ch_name).astype(np.float32),
                    "pc_saturation_mask": diagnostics["pc_saturation_mask"].rename({"range": "altitude"}).assign_coords(channel=ch_name).astype(np.int8),
                }
            )
            channel_datasets.append(ch_ds)
            status_records.append((ch_name, 1, int(dark_current_used)))
            diagnostic_records.append(
                {
                    "channel": ch_name,
                    "deadtime_correction_applied": int(diagnostics["deadtime_correction_applied"]),
                    "deadtime_clipping_fraction": _diagnostic_vector(diagnostics, "deadtime_clipping_fraction", ds.time),
                    "pc_saturation_fraction": _diagnostic_vector(diagnostics, "pc_saturation_fraction", ds.time),
                    "deadtime_min_denominator_observed": float(diagnostics["deadtime_min_denominator_observed"]),
                    "deadtime_min_denominator_allowed": float(diagnostics["deadtime_min_denominator_allowed"]),
                    "bin_shift_bins": int(diagnostics["bin_shift_bins"]),
                    "bin_shift_invalid_fraction": _diagnostic_vector(diagnostics, "bin_shift_invalid_fraction", ds.time),
                }
            )
            clip_fraction = float(diagnostics["deadtime_clipping_fraction"].max(skipna=True).values)
            if clip_fraction > 0.0:
                logger.warning(f"  -> Channel {ch_name}: dead-time denominator clipped in up to {100.0 * clip_fraction:.2f}% of bins.")
            logger.info(f"  -> Channel {ch_name}: corrected successfully.")
        except Exception as exc:
            status_records.append((ch_name, 0, int(dark_current_used)))
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
        final_ds.attrs.update(
            {
                "Processing_level": "Level 1: PC counts->MHz, DeadTime, PC saturation mask, Dark Current, Bin Shift, Background subtraction, corrected signal, Range Corrected Signal, uncertainty propagation, PBL, Radiosonde, Tropopause",
                "Pipeline": "MILGRAU/LIPANCORA",
                "Input_Level0_File": str(nc_file.name),
                "Altitude_units": "m",
            }
        )
        validate_level1_contract(final_ds)
        ensure_directories(save_path.parent)
        encoding = {var: {"zlib": True, "complevel": 4} for var in final_ds.data_vars if final_ds[var].ndim > 0}
        final_ds.to_netcdf(save_path, encoding=encoding)
        return f"[OK] {stem} Level 1 generated successfully: {save_path}"
    except Exception:
        return f"[FAILED] {nc_path} execution halted:\n{traceback.format_exc()}"


def process_level_1(config: Mapping[str, Any], logger: logging.Logger) -> None:
    """Discover and process every Level 0 NetCDF file into Level 1."""
    in_dir = processed_data_root(config)
    files = [f for f in sorted(in_dir.rglob("*.nc")) if "level" not in f.name]
    if not files:
        logger.warning(f"No Level 0 files found in {in_dir}. Exiting.")
        return

    incremental = _incremental_enabled(config)
    files_to_process = []
    skipped_count = 0
    for file_path in files:
        output_path = level1_output_path(file_path, config)
        if incremental and output_path.exists():
            logger.info(f"[SKIPPED] Level 1 already exists for {file_path.name}: {output_path}")
            skipped_count += 1
            continue
        files_to_process.append(file_path)

    if not files_to_process:
        logger.info(f"No Level 0 files require Level 1 processing. Skipped {skipped_count} existing products.")
        return

    logger.info(f"Found {len(files_to_process)} Level 0 files to process ({skipped_count} skipped).")
    for file_path in files_to_process:
        logger.info(process_single_file((str(file_path), config, logger)))
