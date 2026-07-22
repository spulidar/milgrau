"""Tests for the common operational execution-result contract."""

from __future__ import annotations

import json
import logging
import operator
from pathlib import Path
from collections.abc import Callable

import pytest

from milgrau.operations import ExecutionResult, ExecutionStatus, ExecutionSummary, ExitCode


def test_success_result_normalizes_paths_and_freezes_metadata(tmp_path: Path) -> None:
    result = ExecutionResult.success(
        " level1.save ",
        " product generated ",
        input_path=str(tmp_path / "input.nc"),
        output_path=tmp_path / "output.nc",
        duration_seconds=0.125,
        metadata={"wavelength_nm": 532, "incremental": False},
    )

    assert result.status is ExecutionStatus.SUCCESS
    assert result.stage == "level1.save"
    assert result.message == "product generated"
    assert result.input_path == tmp_path / "input.nc"
    assert result.output_path == tmp_path / "output.nc"
    assert result.duration_seconds == pytest.approx(0.125)
    with pytest.raises(TypeError):
        operator.setitem(result.metadata, "wavelength_nm", 355)


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (lambda: ExecutionResult.success("", "ok"), ValueError, "stage"),
        (lambda: ExecutionResult.success("save", ""), ValueError, "message"),
        (lambda: ExecutionResult.success("save", "ok", duration_seconds=-0.1), ValueError, "duration_seconds"),
        (lambda: ExecutionResult.success("save", "ok", metadata={"array": [1, 2]}), TypeError, "JSON scalar"),
        (lambda: ExecutionResult.success("save", "ok", metadata={"value": float("nan")}), ValueError, "finite"),
    ],
)
def test_result_invariants_reject_ambiguous_or_unsafe_values(
    factory: Callable[[], ExecutionResult], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        factory()


def test_nonfailure_cannot_carry_exception_or_traceback() -> None:
    with pytest.raises(ValueError, match="cannot carry"):
        ExecutionResult(
            status=ExecutionStatus.SKIPPED,
            stage="discovery",
            message="already exists",
            cause=RuntimeError("not a failure"),
        )


def test_failure_preserves_original_cause_but_serializes_only_safe_details(tmp_path: Path) -> None:
    try:
        raise OSError("disk unavailable")
    except OSError as cause:
        original_cause = cause
        result = ExecutionResult.failure(
            "level2.write",
            "could not save product",
            fatal=True,
            output_path=tmp_path / "level2.nc",
            cause=cause,
            include_traceback=True,
            metadata={"attempt": 2},
        )

    assert result.status is ExecutionStatus.FATAL_FAILURE
    assert result.cause is original_cause
    assert "OSError: disk unavailable" in result.traceback
    payload = result.to_dict()
    assert payload["cause"] == {"type": "builtins.OSError", "message": "disk unavailable"}
    serialized = json.dumps(payload, allow_nan=False)
    assert "OSError('disk unavailable')" not in serialized


def test_explicit_statuses_do_not_depend_on_message_parsing() -> None:
    skipped = ExecutionResult.skipped("level1.incremental", "ordinary text")
    failure = ExecutionResult.failure("level1.ingestion", "ordinary text")

    assert skipped.status is ExecutionStatus.SKIPPED
    assert failure.status is ExecutionStatus.RECOVERABLE_FAILURE
    assert not skipped.status.is_failure
    assert failure.status.is_failure


def test_log_uses_status_level_and_keeps_human_readable_tags(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("milgrau.tests.execution")
    result = ExecutionResult.failure(
        "level0.group",
        "conversion failed",
        input_path="raw/20240101",
        cause=ValueError("invalid header"),
    )

    with caplog.at_level(logging.WARNING, logger=logger.name):
        result.log(logger)

    assert caplog.records[-1].levelno == logging.WARNING
    assert caplog.records[-1].getMessage() == (
        "[FAILED] level0.group: conversion failed | input=raw/20240101 | cause=ValueError: invalid header"
    )


def test_summary_aggregates_counts_and_partial_failure_exit_code() -> None:
    summary = ExecutionSummary.from_results(
        [
            ExecutionResult.success("level1", "generated"),
            ExecutionResult.skipped("level1", "already current"),
            ExecutionResult.failure("level1", "one input failed"),
        ]
    )

    assert summary.counts == {
        ExecutionStatus.SUCCESS: 1,
        ExecutionStatus.SKIPPED: 1,
        ExecutionStatus.RECOVERABLE_FAILURE: 1,
        ExecutionStatus.FATAL_FAILURE: 0,
    }
    assert summary.overall_status is ExecutionStatus.RECOVERABLE_FAILURE
    assert summary.exit_code is ExitCode.PARTIAL_FAILURE
    assert summary.to_dict()["exit_code"] == 1
    json.dumps(summary.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ([], ExitCode.SUCCESS),
        ([ExecutionResult.skipped("batch", "no work")], ExitCode.SUCCESS),
        ([ExecutionResult.failure("batch", "all failed")], ExitCode.FAILURE),
        ([ExecutionResult.failure("batch", "fatal", fatal=True), ExecutionResult.success("batch", "done")], ExitCode.FAILURE),
    ],
)
def test_summary_exit_policy_for_clean_total_and_fatal_batches(results: list[ExecutionResult], expected: ExitCode) -> None:
    assert ExecutionSummary.from_results(results).exit_code is expected
