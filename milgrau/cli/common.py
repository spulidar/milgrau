"""Shared CLI guards and concise exit-code reporting."""

from __future__ import annotations

import logging
from collections.abc import Callable

from milgrau.io.logging_utils import bind_log_context
from milgrau.operations import ExecutionResult, ExecutionStatus, ExecutionSummary


def run_guarded(stage: str, logger: logging.Logger, operation: Callable[[], ExecutionSummary]) -> ExecutionSummary:
    """Run one CLI operation and reserve exit code 2 for structural/unexpected failure."""
    try:
        summary = operation()
        if not isinstance(summary, ExecutionSummary):
            raise TypeError(f"{stage} returned {type(summary).__name__}; expected ExecutionSummary.")
        return summary
    except Exception as exc:
        failure_logger = bind_log_context(logger, stage="fatal")
        failure_logger.error("cannot run command: %s", exc)
        failure_logger.debug("CLI failure traceback", exc_info=True)
        return ExecutionSummary.from_results(
            [
                ExecutionResult.failure(
                    stage,
                    "Command could not run",
                    fatal=True,
                    cause=exc,
                    include_traceback=True,
                    metadata={"component": "cli"},
                )
            ]
        )


def finish_cli(name: str, summary: ExecutionSummary, logger: logging.Logger) -> int:
    """Log an operator-friendly summary and return the shell exit code."""
    counts = summary.counts
    bind_log_context(logger, stage="summary").info(
        "%d processed | %d skipped | %d with errors",
        counts[ExecutionStatus.OK],
        counts[ExecutionStatus.SKIPPED],
        counts[ExecutionStatus.ERROR],
    )
    return int(summary.exit_code)
