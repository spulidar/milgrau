"""Shared CLI guards and exit-code reporting."""

from __future__ import annotations

import logging
from collections.abc import Callable

from milgrau.io.logging_utils import bind_log_context
from milgrau.operations import ExecutionResult, ExecutionStatus, ExecutionSummary


def run_guarded(stage: str, logger: logging.Logger, operation: Callable[[], ExecutionSummary]) -> ExecutionSummary:
    """Run one CLI operation and convert unexpected exceptions to fatal results."""
    try:
        summary = operation()
        if not isinstance(summary, ExecutionSummary):
            raise TypeError(f"{stage} returned {type(summary).__name__}; expected ExecutionSummary.")
        return summary
    except Exception as exc:
        failure_logger = bind_log_context(logger, stage="fatal")
        failure_logger.error("unexpected CLI failure: %s", exc)
        failure_logger.debug("CLI failure traceback", exc_info=True)
        result = ExecutionResult.failure(
            stage,
            "Unexpected CLI failure",
            fatal=True,
            cause=exc,
            include_traceback=True,
            metadata={"component": "cli"},
        )
        return ExecutionSummary.from_results([result])


def finish_cli(name: str, summary: ExecutionSummary, logger: logging.Logger) -> int:
    """Log aggregate counts and return the ADR-002 process exit code."""
    counts = summary.counts
    bind_log_context(logger, stage="summary").info(
        "success=%d | skipped=%d | recoverable=%d | fatal=%d | exit=%d",
        counts[ExecutionStatus.SUCCESS],
        counts[ExecutionStatus.SKIPPED],
        counts[ExecutionStatus.RECOVERABLE_FAILURE],
        counts[ExecutionStatus.FATAL_FAILURE],
        int(summary.exit_code),
    )
    return int(summary.exit_code)
