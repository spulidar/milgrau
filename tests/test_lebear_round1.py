"""Current LEBEAR orchestration and traceability regression tests.

The original round-1 characterization suite duplicated kernel-level science tests
and froze pre-strict configuration/status contracts. Scientific kernels now live
in dedicated test modules; this file guards the current LEBEAR orchestration
boundary only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from milgrau.level2 import lebear
from milgrau.level2.completeness import WavelengthAttemptStatus, WavelengthFailureCode
from milgrau.operations import ExecutionResult, ExecutionStatus, ExecutionSummary, ExitCode


def _logger(name: str = "test.lebear.orchestration") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def _write_completeness_shell(
    path: Path,
    *,
    requested: list[int],
    processed: list[int],
    failed: list[int],
    completeness: str,
    status: str,
) -> Path:
    xr.Dataset(
        data_vars={
            "requested_wavelengths": (("requested_wavelength",), np.asarray(requested, dtype=np.int32)),
            "processed_wavelengths": (("processed_wavelength",), np.asarray(processed, dtype=np.int32)),
            "failed_wavelengths": (("failed_wavelength",), np.asarray(failed, dtype=np.int32)),
        },
        attrs={"product_completeness": completeness, "product_status": status},
    ).to_netcdf(path)
    return path


def test_attempt_wavelength_maps_local_exception_to_recoverable_diagnostic(monkeypatch) -> None:
    def fail_retrieval(*_args, **_kwargs):
        raise ValueError("synthetic local retrieval failure")

    monkeypatch.setattr(lebear, "process_wavelength", fail_retrieval)

    attempt = lebear.attempt_wavelength(
        xr.Dataset(),
        532,
        np.array([0.0, 7.5]),
        {},
        _logger("test.lebear.local"),
    )

    assert attempt.status is WavelengthAttemptStatus.RECOVERABLE_FAILURE
    assert attempt.diagnostic is not None
    assert attempt.diagnostic.wavelength_nm == 532
    assert attempt.diagnostic.code is WavelengthFailureCode.INTERNAL_ERROR
    assert "synthetic local retrieval failure" in attempt.diagnostic.message


def test_attempt_wavelength_keeps_system_failure_fatal(monkeypatch) -> None:
    def fail_retrieval(*_args, **_kwargs):
        raise MemoryError("synthetic allocation failure")

    monkeypatch.setattr(lebear, "process_wavelength", fail_retrieval)

    attempt = lebear.attempt_wavelength(
        xr.Dataset(),
        355,
        np.array([0.0, 7.5]),
        {},
        _logger("test.lebear.fatal"),
    )

    assert attempt.status is WavelengthAttemptStatus.FATAL_FAILURE
    assert attempt.diagnostic is not None
    assert attempt.diagnostic.wavelength_nm == 355


def test_level2_currentness_requires_complete_requested_wavelength_set(tmp_path: Path, monkeypatch) -> None:
    level1 = tmp_path / "20240101sant_level1_rcs.nc"
    level1.write_text("synthetic upstream", encoding="utf-8")
    output = tmp_path / "20240101sant_level2_optical.nc"
    _write_completeness_shell(
        output,
        requested=[355, 532],
        processed=[355, 532],
        failed=[],
        completeness="complete",
        status="success",
    )

    monkeypatch.setattr(lebear, "get_wavelengths_to_process", lambda _config: [355, 532])
    monkeypatch.setattr(lebear, "validate_level2_contract", lambda _ds: None)
    monkeypatch.setattr(lebear, "output_is_current", lambda *_args, **_kwargs: True)

    assert lebear.level2_output_is_current(level1, output, {})

    with xr.open_dataset(output) as opened:
        partial = opened.load()
    partial.attrs["product_completeness"] = "partial"
    partial.attrs["product_status"] = "partial_failure"
    partial["processed_wavelengths"] = (("processed_wavelength",), np.asarray([532], dtype=np.int32))
    partial["failed_wavelengths"] = (("failed_wavelength",), np.asarray([355], dtype=np.int32))
    partial.to_netcdf(output, mode="w")

    assert not lebear.level2_output_is_current(level1, output, {})


def test_process_level2_skips_only_current_product(tmp_path: Path, monkeypatch) -> None:
    files = [
        tmp_path / "20240101saam_level1_rcs.nc",
        tmp_path / "20240101sapm_level1_rcs.nc",
    ]
    for path in files:
        path.write_text("synthetic", encoding="utf-8")

    monkeypatch.setattr(lebear, "discover_level1_files", lambda _config: files)
    monkeypatch.setattr(lebear, "incremental_enabled", lambda _config: True)
    monkeypatch.setattr(lebear, "level2_output_is_current", lambda path, *_args, **_kwargs: Path(path) == files[0])
    monkeypatch.setattr(lebear, "level2_qa_enabled", lambda _config: False)

    calls: list[Path] = []

    def fake_process(path, *_args, **_kwargs) -> ExecutionSummary:
        path = Path(path)
        calls.append(path)
        return ExecutionSummary.from_results(
            [ExecutionResult.success("level2.complete", "processed", input_path=path)]
        )

    monkeypatch.setattr(lebear, "process_single_level1_file", fake_process)

    summary = lebear.process_level_2({"processing": {"incremental": True}}, _logger("test.lebear.batch"))

    assert calls == [files[1]]
    assert [result.status for result in summary.results] == [ExecutionStatus.SKIPPED, ExecutionStatus.OK]
    assert summary.exit_code is ExitCode.OK


def test_process_level2_continues_after_processing_error(tmp_path: Path, monkeypatch) -> None:
    files = [
        tmp_path / "20240101saam_level1_rcs.nc",
        tmp_path / "20240101sapm_level1_rcs.nc",
    ]
    for path in files:
        path.write_text("synthetic", encoding="utf-8")

    monkeypatch.setattr(lebear, "discover_level1_files", lambda _config: files)
    monkeypatch.setattr(lebear, "incremental_enabled", lambda _config: False)

    calls: list[Path] = []

    def fake_process(path, *_args, **_kwargs) -> ExecutionSummary:
        path = Path(path)
        calls.append(path)
        result = (
            ExecutionResult.failure("level2.retrieval", "first failed", input_path=path)
            if path == files[0]
            else ExecutionResult.success("level2.complete", "second succeeded", input_path=path)
        )
        return ExecutionSummary.from_results([result])

    monkeypatch.setattr(lebear, "process_single_level1_file", fake_process)

    summary = lebear.process_level_2({"processing": {"incremental": False}}, _logger("test.lebear.continue"))

    assert calls == files
    assert [result.status for result in summary.results] == [ExecutionStatus.ERROR, ExecutionStatus.OK]
    assert summary.exit_code is ExitCode.ERROR


def test_atomic_level2_write_preserves_existing_product_and_removes_temporary_file(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "product_level2_optical.nc"
    output_path.write_text("stable product", encoding="utf-8")
    dataset = xr.Dataset({"value": (("x",), np.array([1.0]))})

    def fail_write(_self, path, **_kwargs):
        Path(path).write_text("partial product", encoding="utf-8")
        raise OSError("synthetic write failure")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fail_write)

    with pytest.raises(OSError, match="synthetic write failure"):
        lebear._write_level2_atomically(dataset, output_path, {})

    assert output_path.read_text(encoding="utf-8") == "stable product"
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))
