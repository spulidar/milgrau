"""Shared CLI guards and exit-code reporting."""

from __future__ import annotations

import logging
from collections.abc import Callable

from milgrau.operations import ExecutionResult, ExecutionStatus, ExecutionSummary


def run_guarded(stage: str, logger: logging.Logger, operation: Callable[[], ExecutionSummary]) -> ExecutionSummary:
    """Run one CLI operation and convert unexpected exceptions to fatal results."""
    try:
        summary = operation()
        if not isinstance(summary, ExecutionSummary):
            raise TypeError(f"{stage} returned {type(summary).__name__}; expected ExecutionSummary.")
        return summary
    except Exception as exc:
        result = ExecutionResult.failure(
            stage,
            "Unexpected CLI failure",
            fatal=True,
            cause=exc,
            include_traceback=True,
            metadata={"component": "cli"},
        )
        result.log(logger)
        return ExecutionSummary.from_results([result])


def finish_cli(name: str, summary: ExecutionSummary, logger: logging.Logger) -> int:
    """Log aggregate counts and return the ADR-002 process exit code."""
    counts = summary.counts
    logger.info(
        f"=== {name} finished: success {counts[ExecutionStatus.SUCCESS]}, "
        f"skipped {counts[ExecutionStatus.SKIPPED]}, "
        f"recoverable failures {counts[ExecutionStatus.RECOVERABLE_FAILURE]}, "
        f"fatal failures {counts[ExecutionStatus.FATAL_FAILURE]}, "
        f"exit code {int(summary.exit_code)}. ==="
    )
    return int(summary.exit_code)
