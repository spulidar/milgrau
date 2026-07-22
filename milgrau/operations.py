"""Typed operational results shared by MILGRAU pipeline orchestration.

The contract deliberately separates operational outcome from scientific product
validity.  Results have one explicit status, non-empty stage and message fields,
optional paths, and JSON-safe scalar metadata.  Successful or skipped results
cannot carry exceptions or tracebacks; failure results may retain the original
exception in memory while serialization exposes only its type and message.
"""

from __future__ import annotations

import logging
import math
import traceback as traceback_module
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
from types import MappingProxyType

type MetadataValue = str | int | float | bool | None


class ExecutionStatus(StrEnum):
    """Explicit outcome states for one operational stage."""

    SUCCESS = "success"
    SKIPPED = "skipped"
    RECOVERABLE_FAILURE = "recoverable_failure"
    FATAL_FAILURE = "fatal_failure"

    @property
    def is_failure(self) -> bool:
        """Return whether this status represents a failure."""
        return self in {self.RECOVERABLE_FAILURE, self.FATAL_FAILURE}


class ExitCode(IntEnum):
    """Process exit codes defined by ADR-002."""

    SUCCESS = 0
    PARTIAL_FAILURE = 1
    FAILURE = 2


def _freeze_metadata(metadata: Mapping[str, MetadataValue]) -> Mapping[str, MetadataValue]:
    """Validate and copy metadata into an immutable, JSON-safe mapping."""
    copied: dict[str, MetadataValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Metadata keys must be non-empty strings.")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"Metadata value for {key!r} must be a JSON scalar; got {type(value).__name__}.")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Metadata value for {key!r} must be finite.")
        copied[key] = value
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Result of one pipeline stage, with invariants enforced at construction."""

    status: ExecutionStatus
    stage: str
    message: str
    input_path: Path | None = None
    output_path: Path | None = None
    cause: BaseException | None = field(default=None, repr=False, compare=False)
    traceback: str | None = field(default=None, repr=False, compare=False)
    duration_seconds: float | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, ExecutionStatus):
            raise TypeError("status must be an ExecutionStatus.")
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError("stage must be a non-empty string.")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be a non-empty string.")
        object.__setattr__(self, "stage", self.stage.strip())
        object.__setattr__(self, "message", self.message.strip())

        for attribute in ("input_path", "output_path"):
            value = getattr(self, attribute)
            if value is not None:
                try:
                    object.__setattr__(self, attribute, Path(value))
                except TypeError as exc:
                    raise TypeError(f"{attribute} must be path-like or None.") from exc

        if self.duration_seconds is not None:
            duration = float(self.duration_seconds)
            if not math.isfinite(duration) or duration < 0.0:
                raise ValueError("duration_seconds must be finite and non-negative.")
            object.__setattr__(self, "duration_seconds", duration)

        if not self.status.is_failure and (self.cause is not None or self.traceback is not None):
            raise ValueError("Successful and skipped results cannot carry a cause or traceback.")
        if self.cause is not None and not isinstance(self.cause, BaseException):
            raise TypeError("cause must be an exception or None.")
        if self.traceback is not None and (not isinstance(self.traceback, str) or not self.traceback.strip()):
            raise ValueError("traceback must be a non-empty string or None.")

        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @classmethod
    def success(
        cls,
        stage: str,
        message: str,
        *,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        duration_seconds: float | None = None,
        metadata: Mapping[str, MetadataValue] | None = None,
    ) -> ExecutionResult:
        """Build a successful result."""
        return cls(
            status=ExecutionStatus.SUCCESS,
            stage=stage,
            message=message,
            input_path=None if input_path is None else Path(input_path),
            output_path=None if output_path is None else Path(output_path),
            duration_seconds=duration_seconds,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def skipped(
        cls,
        stage: str,
        message: str,
        *,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        metadata: Mapping[str, MetadataValue] | None = None,
    ) -> ExecutionResult:
        """Build a skipped result."""
        return cls(
            status=ExecutionStatus.SKIPPED,
            stage=stage,
            message=message,
            input_path=None if input_path is None else Path(input_path),
            output_path=None if output_path is None else Path(output_path),
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def failure(
        cls,
        stage: str,
        message: str,
        *,
        fatal: bool = False,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        cause: BaseException | None = None,
        include_traceback: bool = False,
        duration_seconds: float | None = None,
        metadata: Mapping[str, MetadataValue] | None = None,
    ) -> ExecutionResult:
        """Build a recoverable or fatal failure, optionally capturing traceback text."""
        traceback_text = None
        if include_traceback and cause is not None:
            traceback_text = "".join(traceback_module.format_exception(cause)).rstrip()
        return cls(
            status=ExecutionStatus.FATAL_FAILURE if fatal else ExecutionStatus.RECOVERABLE_FAILURE,
            stage=stage,
            message=message,
            input_path=None if input_path is None else Path(input_path),
            output_path=None if output_path is None else Path(output_path),
            cause=cause,
            traceback=traceback_text,
            duration_seconds=duration_seconds,
            metadata={} if metadata is None else metadata,
        )

    @property
    def log_level(self) -> int:
        """Return the standard logging level associated with the status."""
        return {
            ExecutionStatus.SUCCESS: logging.INFO,
            ExecutionStatus.SKIPPED: logging.INFO,
            ExecutionStatus.RECOVERABLE_FAILURE: logging.WARNING,
            ExecutionStatus.FATAL_FAILURE: logging.ERROR,
        }[self.status]

    def to_log_message(self) -> str:
        """Format a compact human-readable message without parsing status text."""
        tag = {
            ExecutionStatus.SUCCESS: "OK",
            ExecutionStatus.SKIPPED: "SKIPPED",
            ExecutionStatus.RECOVERABLE_FAILURE: "FAILED",
            ExecutionStatus.FATAL_FAILURE: "FATAL",
        }[self.status]
        details = [f"[{tag}] {self.stage}: {self.message}"]
        if self.input_path is not None:
            details.append(f"input={self.input_path}")
        if self.output_path is not None:
            details.append(f"output={self.output_path}")
        if self.duration_seconds is not None:
            details.append(f"duration={self.duration_seconds:.3f}s")
        if self.cause is not None:
            details.append(f"cause={type(self.cause).__name__}: {self.cause}")
        return " | ".join(details)

    def __str__(self) -> str:
        """Keep legacy logger calls readable during incremental pipeline migration."""
        return self.to_log_message()

    def log(self, logger: logging.Logger) -> None:
        """Emit this result through a logger at its status-derived level."""
        message = self.to_log_message()
        if self.traceback is not None:
            message = f"{message}\n{self.traceback}"
        if self.log_level == logging.ERROR:
            logger.error(message)
        elif self.log_level == logging.WARNING:
            logger.warning(message)
        else:
            logger.info(message)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation that never serializes the exception object."""
        cause_payload: dict[str, str] | None = None
        if self.cause is not None:
            cause_type = type(self.cause)
            cause_payload = {
                "type": f"{cause_type.__module__}.{cause_type.__qualname__}",
                "message": str(self.cause),
            }
        return {
            "status": self.status.value,
            "stage": self.stage,
            "input_path": None if self.input_path is None else str(self.input_path),
            "output_path": None if self.output_path is None else str(self.output_path),
            "message": self.message,
            "cause": cause_payload,
            "traceback": self.traceback,
            "duration_seconds": self.duration_seconds,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    """Aggregate results and apply the ADR-002 partial-failure exit policy."""

    results: tuple[ExecutionResult, ...]

    def __post_init__(self) -> None:
        results = tuple(self.results)
        if not all(isinstance(result, ExecutionResult) for result in results):
            raise TypeError("results must contain only ExecutionResult instances.")
        object.__setattr__(self, "results", results)

    @classmethod
    def from_results(cls, results: Iterable[ExecutionResult]) -> ExecutionSummary:
        """Build a summary from any finite iterable of results."""
        return cls(tuple(results))

    @property
    def counts(self) -> dict[ExecutionStatus, int]:
        """Return a count for every explicit status."""
        observed = Counter(result.status for result in self.results)
        return {status: observed[status] for status in ExecutionStatus}

    @property
    def overall_status(self) -> ExecutionStatus:
        """Return the highest-severity status represented in the batch."""
        if any(result.status is ExecutionStatus.FATAL_FAILURE for result in self.results):
            return ExecutionStatus.FATAL_FAILURE
        if any(result.status is ExecutionStatus.RECOVERABLE_FAILURE for result in self.results):
            return ExecutionStatus.RECOVERABLE_FAILURE
        if any(result.status is ExecutionStatus.SUCCESS for result in self.results):
            return ExecutionStatus.SUCCESS
        return ExecutionStatus.SKIPPED

    @property
    def exit_code(self) -> ExitCode:
        """Return 0 for clean batches, 1 for partial failure, or 2 for total/fatal failure."""
        counts = self.counts
        if counts[ExecutionStatus.FATAL_FAILURE]:
            return ExitCode.FAILURE
        if counts[ExecutionStatus.RECOVERABLE_FAILURE]:
            completed = counts[ExecutionStatus.SUCCESS] + counts[ExecutionStatus.SKIPPED]
            return ExitCode.PARTIAL_FAILURE if completed else ExitCode.FAILURE
        return ExitCode.SUCCESS

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe summary payload."""
        return {
            "overall_status": self.overall_status.value,
            "exit_code": int(self.exit_code),
            "counts": {status.value: count for status, count in self.counts.items()},
            "results": [result.to_dict() for result in self.results],
        }
