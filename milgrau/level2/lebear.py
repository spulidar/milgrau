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

from milgrau.incremental import output_is_current
from milgrau.io.contracts import netcdf_satisfies_contract, validate_level1_contract, validate_level2_contract
from milgrau.io.filesystem import ensure_directories
from milgrau.io.paths import level2_output_path
from milgrau.operations import ExecutionResult, ExecutionSummary
from milgrau.level2.completeness import (
    Level2ProductContract,
    ProductCompleteness,
    WavelengthAttempt,
    WavelengthAttemptStatus,
    WavelengthFailureCode,
    WavelengthFailureDiagnostic,
    WavelengthFailureStage,
    canonical_wavelengths,
    diagnostic_from_exception,
)
from milgrau.level2.config import (
    get_gluing_config,
    get_kfs_mode,
    get_lidar_ratio,
    get_molecular_fit_config,
    get_wavelengths_to_process,
    incremental_enabled,
)
from milgrau.level2.dataset import build_level2_dataset
from milgrau.level2.discovery import discover_level1_files
from milgrau.level2.retrieval import (
    RetrievalStageError,
    evaluate_rayleigh_reference as _evaluate_rayleigh_reference,
    process_wavelength,
    propagate_glued_error as _propagate_glued_error,
)
from milgrau.level2.qa import generate_level2_qa, level2_qa_enabled
from milgrau.level2.time_window import subset_level1_time_window


def level2_output_is_current(
    nc_file: str | Path,
    output_path: str | Path,
    config: Mapping[str, Any],
    *,
    start_utc: str | None = None,
    stop_utc: str | None = None,
    output_tag: str | None = None,
) -> bool:
    """Return whether one complete Level 2 output is intact and up to date."""
    output = Path(output_path)
    if not output.is_file():
        return False
    requested = list(canonical_wavelengths(get_wavelengths_to_process(config)))
    try:
        with xr.open_dataset(output) as ds:
            validate_level2_contract(ds)
            if (
                str(ds.attrs.get("product_completeness", "")) != "complete"
                or str(ds.attrs.get("product_status", "")) != "success"
                or "requested_wavelengths" not in ds
                or "processed_wavelengths" not in ds
                or "failed_wavelengths" not in ds
            ):
                return False
            requested_written = [int(value) for value in np.asarray(ds["requested_wavelengths"].values).tolist()]
            processed_written = [int(value) for value in np.asarray(ds["processed_wavelengths"].values).tolist()]
            failed_written = [int(value) for value in np.asarray(ds["failed_wavelengths"].values).tolist()]
            if requested_written != requested or processed_written != requested or failed_written != []:
                return False
    except Exception:
        return False
    return output_is_current(
        output,
        [nc_file],
        config=config,
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


class FatalWavelengthProcessingError(RuntimeError):
    """Signal that one wavelength exposed a global resource/runtime failure."""


def attempt_wavelength(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> WavelengthAttempt:
    """Isolate one requested wavelength and classify its scientific outcome."""
    try:
        result = process_wavelength(ds_l1, wavelength_nm, altitude_m, config, logger)
    except (MemoryError, SystemError) as exc:
        diagnostic = diagnostic_from_exception(wavelength_nm, exc)
        return WavelengthAttempt.fatal_failure(diagnostic)
    except Exception as exc:
        retrieval_stage = exc.stage if isinstance(exc, RetrievalStageError) else None
        diagnostic = diagnostic_from_exception(
            wavelength_nm,
            exc,
            retrieval_stage=retrieval_stage,
        )
        logger.warning(
            f"  -> {wavelength_nm} nm failed at {diagnostic.stage.name.lower()} "
            f"({diagnostic.code.name.lower()}): {diagnostic.message}"
        )
        return WavelengthAttempt.recoverable_failure(diagnostic)

    if not bool(result.optical.retrieval_success_flag.astype(bool).any()):
        diagnostic = WavelengthFailureDiagnostic(
            wavelength_nm=wavelength_nm,
            stage=WavelengthFailureStage.RETRIEVAL_VALIDATION,
            code=WavelengthFailureCode.NO_VALID_RETRIEVAL_BLOCK,
            message="No block produced a valid Rayleigh plus two-sided KFS optical retrieval.",
            cause_summary="retrieval_success_flag contains no successful block",
        )
        logger.warning(
            f"  -> {wavelength_nm} nm failed: {diagnostic.code.name.lower()}."
        )
        return WavelengthAttempt.recoverable_failure(diagnostic)
    return WavelengthAttempt.success(result)


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
        with xr.open_dataset(nc_path) as ds_l1:
            ds_l1.load()
            stage = "level2.validation.input"
            validate_level1_contract(ds_l1)
            stage = "level2.time_window"
            ds_l1, inferred_output_tag = subset_level1_time_window(ds_l1, start_utc, stop_utc)
            if output_tag is None:
                output_tag = inferred_output_tag
            stage = "level2.configuration"
            wavelengths = canonical_wavelengths(get_wavelengths_to_process(config))
            get_kfs_mode(config)
            get_gluing_config(config)
            get_molecular_fit_config(config)
            for wavelength in wavelengths:
                get_lidar_ratio(config, wavelength, ds_l1["time"].values[0])
            altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
            if np.nanmax(altitude_m) <= 100.0:
                altitude_m = altitude_m * 1000.0
            attempts: list[WavelengthAttempt] = []
            stage = "level2.retrieval"
            for wavelength in wavelengths:
                attempt = attempt_wavelength(ds_l1, wavelength, altitude_m, config, logger)
                attempts.append(attempt)
                if attempt.status is WavelengthAttemptStatus.FATAL_FAILURE:
                    assert attempt.diagnostic is not None
                    raise FatalWavelengthProcessingError(attempt.diagnostic.message)
            product_contract = Level2ProductContract.from_attempts(wavelengths, attempts)
            if product_contract.completeness is ProductCompleteness.FAILED:
                diagnostics = "; ".join(
                    f"{item.wavelength_nm} nm: {item.code.name.lower()}"
                    for item in product_contract.failure_diagnostics
                )
                return ExecutionSummary.from_results(
                    [
                        ExecutionResult.failure(
                            "level2.retrieval",
                            f"No requested wavelength produced a usable optical product ({diagnostics}).",
                            input_path=nc_path,
                            duration_seconds=time.perf_counter() - started_at,
                            metadata={"pipeline": "LEBEAR", **product_contract.execution_metadata()},
                        )
                    ]
                )
            results_by_wavelength = {
                attempt.wavelength_nm: attempt.result
                for attempt in attempts
                if attempt.status is WavelengthAttemptStatus.SUCCESS
                and attempt.result is not None
            }
            results = [
                results_by_wavelength[wavelength]
                for wavelength in product_contract.processed_wavelengths
            ]
            stage = "level2.dataset"
            ds_l2 = build_level2_dataset(
                ds_l1,
                results,
                altitude_m,
                nc_path,
                config,
                product_contract,
            )
            stage = "level2.validation.output"
            validate_level2_contract(ds_l2)

        stage = "level2.write"
        output_path = level2_output_path(nc_path, variant_tag=output_tag)
        encoding = {
            var: {"zlib": True, "complevel": 4}
            for var in ds_l2.data_vars
            if ds_l2[var].ndim > 0 and ds_l2[var].dtype.kind not in {"O", "S", "U"}
        }
        _write_level2_atomically(ds_l2, output_path, encoding)
        if product_contract.completeness is ProductCompleteness.COMPLETE:
            logger.info(f"  -> [OK] Complete Level 2 NetCDF generated: {output_path}")
            product_results = [
                ExecutionResult.success(
                    "level2.complete",
                    f"{nc_path.name} complete Level 2 generated successfully",
                    input_path=nc_path,
                    output_path=output_path,
                    duration_seconds=time.perf_counter() - started_at,
                    metadata={"pipeline": "LEBEAR", **product_contract.execution_metadata()},
                )
            ]
        else:
            logger.warning(f"  -> [PARTIAL] Incomplete Level 2 NetCDF generated: {output_path}")
            product_results = [
                ExecutionResult.success(
                    "level2.write",
                    f"{nc_path.name} partial Level 2 file written atomically",
                    input_path=nc_path,
                    output_path=output_path,
                    duration_seconds=time.perf_counter() - started_at,
                    metadata={"pipeline": "LEBEAR", **product_contract.execution_metadata()},
                ),
                ExecutionResult.failure(
                    "level2.partial",
                    f"{nc_path.name} Level 2 was written but is incomplete; failed wavelengths: "
                    f"{', '.join(str(value) for value in product_contract.failed_wavelengths)} nm",
                    input_path=nc_path,
                    output_path=output_path,
                    duration_seconds=time.perf_counter() - started_at,
                    metadata={"pipeline": "LEBEAR", **product_contract.execution_metadata()},
                ),
            ]
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
        return ExecutionSummary.from_results([*product_results, qa_result])
    except Exception as exc:
        fatal_stages = {
            "level2.ingestion",
            "level2.validation.input",
            "level2.time_window",
            "level2.configuration",
            "level2.dataset",
            "level2.validation.output",
            "level2.write",
        }
        return ExecutionSummary.from_results(
            [
                ExecutionResult.failure(
                    stage,
                    f"Level 2 processing failed for {nc_path.name}",
                    fatal=isinstance(exc, FatalWavelengthProcessingError) or stage in fatal_stages,
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
                f"Level 2 is up to date for {file_path.name}",
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
