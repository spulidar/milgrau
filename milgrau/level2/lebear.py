"""LEBEAR Level 2 optical inversion orchestration."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Mapping, Any

import numpy as np
import xarray as xr

from milgrau.io.contracts import validate_level1_contract, validate_level2_contract
from milgrau.io.filesystem import ensure_directories
from milgrau.io.paths import LEVEL2_SUFFIX, level2_output_path
from milgrau.level2.config import (
    get_wavelengths_to_process,
    incremental_enabled,
)
from milgrau.level2.dataset import build_level2_dataset
from milgrau.level2.discovery import discover_level1_files
from milgrau.level2.retrieval import (
    evaluate_rayleigh_reference as _evaluate_rayleigh_reference,
    process_wavelength,
    propagate_glued_error as _propagate_glued_error,
)
from milgrau.level2.time_window import subset_level1_time_window
from milgrau.viz.level2_qa import plot_all_level2_qa


def process_single_level1_file(
    nc_file: str | Path,
    config: Mapping[str, Any],
    logger: logging.Logger,
    start_utc: str | None = None,
    stop_utc: str | None = None,
    output_tag: str | None = None,
) -> str:
    """Process one Level 1 file into a Level 2 optical product."""
    nc_path = Path(nc_file)
    try:
        with xr.open_dataset(nc_path) as ds_l1:
            ds_l1.load()
            validate_level1_contract(ds_l1)
            ds_l1, inferred_output_tag = subset_level1_time_window(ds_l1, start_utc, stop_utc)
            if output_tag is None:
                output_tag = inferred_output_tag
            altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
            if np.nanmax(altitude_m) <= 100.0:
                altitude_m = altitude_m * 1000.0
            wavelengths = get_wavelengths_to_process(config)
            results = []
            for wavelength in wavelengths:
                try:
                    results.append(process_wavelength(ds_l1, wavelength, altitude_m, config, logger))
                except Exception as exc:
                    logger.warning(f"  -> Skipping {wavelength} nm in {nc_path.name}: {exc}")
            if not results:
                raise RuntimeError("No wavelength could be processed by LEBEAR.")
            ds_l2 = build_level2_dataset(ds_l1, results, altitude_m, nc_path, config)
            validate_level2_contract(ds_l2)

        output_path = level2_output_path(nc_path, variant_tag=output_tag)
        ensure_directories(output_path.parent)
        encoding = {var: {"zlib": True, "complevel": 4} for var in ds_l2.data_vars if ds_l2[var].ndim > 0}
        ds_l2.to_netcdf(output_path, encoding=encoding)
        logger.info(f"  -> [OK] Level 2 NetCDF generated: {output_path}")

        qa_cfg = config.get("visualization", {}).get("level2_qa", {}) or {}
        if bool(qa_cfg.get("enabled", True)):
            qa_dir = output_path.parent / "level2_qa"
            with xr.open_dataset(output_path) as ds_saved, xr.open_dataset(nc_path) as ds_l1_saved:
                ds_saved.load()
                ds_l1_saved.load()
                generated = plot_all_level2_qa(
                    ds_l2=ds_saved,
                    output_folder=qa_dir,
                    file_name_prefix=output_path.name.replace(LEVEL2_SUFFIX, ""),
                    config=dict(config),
                    root_dir=Path.cwd(),
                    ds_l1=ds_l1_saved,
                )
            logger.info(f"  -> Generated {len(generated)} Level 2 QA plot(s).")
        return f"[OK] {nc_path.name} Level 2 generated successfully: {output_path}"
    except Exception:
        return f"[FAILED] {nc_path}:\n{traceback.format_exc()}"


def process_level_2(config: Mapping[str, Any], logger: logging.Logger) -> None:
    """Discover Level 1 files and process them into Level 2 products."""
    files = discover_level1_files(config)
    if not files:
        logger.warning("No Level 1 files found for LEBEAR processing.")
        return

    incremental = incremental_enabled(config)
    files_to_process = []
    skipped_count = 0
    for file_path in files:
        output_path = level2_output_path(file_path)
        if incremental and output_path.exists():
            logger.info(f"[SKIPPED] Level 2 already exists for {file_path.name}: {output_path}")
            skipped_count += 1
            continue
        files_to_process.append(file_path)

    if not files_to_process:
        logger.info(f"No Level 1 files require Level 2 processing. Skipped {skipped_count} existing products.")
        return

    logger.info(f"Found {len(files_to_process)} Level 1 files for LEBEAR ({skipped_count} skipped).")
    for file_path in files_to_process:
        logger.info(process_single_level1_file(file_path, config, logger))
