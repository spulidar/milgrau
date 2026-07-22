"""LEBEAR Level 2 optical inversion orchestration."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Mapping, Any

import numpy as np
import xarray as xr

from milgrau.io.contracts import netcdf_satisfies_contract, validate_level1_contract, validate_level2_contract
from milgrau.io.filesystem import ensure_directories
from milgrau.io.paths import level2_output_path
from milgrau.operations import ExecutionResult, ExecutionSummary
from milgrau.provenance import (
    ProductProvenance,
    build_product_provenance,
    build_product_provenance_from_signatures,
    file_signature,
    output_is_current,
    write_provenance_manifest,
)
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
from milgrau.level2.qa import generate_level2_qa, level2_qa_enabled
from milgrau.level2.time_window import subset_level1_time_window


def level2_product_provenance(
    nc_file: str | Path,
    config: Mapping[str, Any],
    *,
    start_utc: str | None = None,
    stop_utc: str | None = None,
    output_tag: str | None = None,
    input_signature: Mapping[str, Any] | None = None,
) -> ProductProvenance:
    """Build provenance for a complete or time-windowed Level 2 product."""
    variant = {"start_utc": start_utc, "stop_utc": stop_utc, "output_tag": output_tag}
    if input_signature is not None:
        return build_product_provenance_from_signatures(
            "level2",
            [input_signature],
            config,
            variant=variant,
        )
    return build_product_provenance("level2", [nc_file], config, variant=variant)


def level2_output_is_current(
    nc_file: str | Path,
    output_path: str | Path,
    config: Mapping[str, Any],
    *,
    start_utc: str | None = None,
    stop_utc: str | None = None,
    output_tag: str | None = None,
) -> bool:
    """Return whether one Level 2 output is intact and has matching provenance."""
    if not Path(output_path).exists():
        return False
    expected = level2_product_provenance(
        nc_file,
        config,
        start_utc=start_utc,
        stop_utc=stop_utc,
        output_tag=output_tag,
    )
    return output_is_current(
        output_path,
        expected,
        integrity_check=lambda path: netcdf_satisfies_contract(path, validate_level2_contract),
    )


def _write_level2_atomically(ds: xr.Dataset, output_path: Path, encoding: Mapping[str, Mapping[str, int | bool]]) -> None:
    """Write beside the destination and atomically replace it only after success."""
    ensure_directories(output_path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        ds.to_netcdf(temporary_path, encoding=dict(encoding))
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def process_single_level1_file(
    nc_file: str | Path,
    config: Mapping[str, Any],
    logger: logging.Logger,
    start_utc: str | None = None,
    stop_utc: str | None = None,
    output_tag: str | None = None,
) -> ExecutionSummary:
    """Generate one Level 2 product and report optional QA separately."""
    nc_path = Path(nc_file)
    started_at = time.perf_counter()
    output_path: Path | None = None
    stage = "level2.ingestion"
    try:
        source_signature = file_signature(nc_path)
        with xr.open_dataset(nc_path) as ds_l1:
            ds_l1.load()
            stage = "level2.validation.input"
            validate_level1_contract(ds_l1)
            stage = "level2.time_window"
            ds_l1, inferred_output_tag = subset_level1_time_window(ds_l1, start_utc, stop_utc)
            if output_tag is None:
                output_tag = inferred_output_tag
            altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
            if np.nanmax(altitude_m) <= 100.0:
                altitude_m = altitude_m * 1000.0
            wavelengths = get_wavelengths_to_process(config)
            results = []
            stage = "level2.retrieval"
            for wavelength in wavelengths:
                try:
                    results.append(process_wavelength(ds_l1, wavelength, altitude_m, config, logger))
                except Exception as exc:
                    logger.warning(f"  -> Skipping {wavelength} nm in {nc_path.name}: {exc}")
            if not results:
                raise RuntimeError("No wavelength could be processed by LEBEAR.")
            stage = "level2.dataset"
            ds_l2 = build_level2_dataset(ds_l1, results, altitude_m, nc_path, config)
            stage = "level2.validation.output"
            validate_level2_contract(ds_l2)

        stage = "level2.write"
        output_path = level2_output_path(nc_path, variant_tag=output_tag)
        encoding = {var: {"zlib": True, "complevel": 4} for var in ds_l2.data_vars if ds_l2[var].ndim > 0}
        _write_level2_atomically(ds_l2, output_path, encoding)
        stage = "level2.provenance.write"
        provenance = level2_product_provenance(
            nc_path,
            config,
            start_utc=start_utc,
            stop_utc=stop_utc,
            output_tag=output_tag,
            input_signature=source_signature,
        )
        write_provenance_manifest(output_path, provenance)
        logger.info(f"  -> [OK] Level 2 NetCDF generated: {output_path}")

        product_result = ExecutionResult.success(
            "level2.complete",
            f"{nc_path.name} Level 2 generated successfully",
            input_path=nc_path,
            output_path=output_path,
            duration_seconds=time.perf_counter() - started_at,
            metadata={"pipeline": "LEBEAR"},
        )
        if level2_qa_enabled(config):
            qa_result = generate_level2_qa(nc_path, output_path, config, logger)
        else:
            qa_result = ExecutionResult.skipped(
                "level2.qa",
                "Level 2 QA disabled by configuration",
                input_path=output_path,
                output_path=output_path.parent / "level2_qa",
                metadata={"pipeline": "LEBEAR"},
            )
        qa_result.log(logger)
        return ExecutionSummary.from_results([product_result, qa_result])
    except Exception as exc:
        return ExecutionSummary.from_results(
            [
                ExecutionResult.failure(
                    stage,
                    f"Level 2 processing failed for {nc_path.name}",
                    input_path=nc_path,
                    output_path=output_path,
                    cause=exc,
                    include_traceback=True,
                    duration_seconds=time.perf_counter() - started_at,
                    metadata={"pipeline": "LEBEAR"},
                )
            ]
        )


def process_level_2(config: Mapping[str, Any], logger: logging.Logger) -> ExecutionSummary:
    """Discover Level 1 files and process them into Level 2 products."""
    files = discover_level1_files(config)
    if not files:
        logger.warning("No Level 1 files found for LEBEAR processing.")
        return ExecutionSummary.from_results(
            [ExecutionResult.skipped("level2.discovery", "No Level 1 files found")]
        )

    incremental = incremental_enabled(config)
    files_to_process = []
    skipped_results: list[ExecutionResult] = []
    for file_path in files:
        output_path = level2_output_path(file_path)
        if incremental and level2_output_is_current(file_path, output_path, config):
            result = ExecutionResult.skipped(
                "level2.incremental",
                f"Level 2 provenance is current for {file_path.name}",
                input_path=file_path,
                output_path=output_path,
            )
            result.log(logger)
            skipped_results.append(result)
            if level2_qa_enabled(config):
                qa_result = generate_level2_qa(file_path, output_path, config, logger)
                qa_result.log(logger)
                skipped_results.append(qa_result)
            continue
        files_to_process.append(file_path)

    if not files_to_process:
        logger.info(f"No Level 1 files require Level 2 processing. Skipped {len(skipped_results)} existing products.")
        return ExecutionSummary.from_results(skipped_results)

    logger.info(f"Found {len(files_to_process)} Level 1 files for LEBEAR ({len(skipped_results)} skipped).")
    results = list(skipped_results)
    for file_path in files_to_process:
        file_summary = process_single_level1_file(file_path, config, logger)
        for result in file_summary.results:
            if result.stage != "level2.qa":
                result.log(logger)
        results.extend(file_summary.results)
    return ExecutionSummary.from_results(results)
