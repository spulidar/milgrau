"""Typed operational results shared by MILGRAU pipeline orchestration."""

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
    """Simple operational outcome states.

    Scientific QA remains in product variables/diagnostics; this enum only says
    whether an operation completed, was intentionally skipped, or errored.
    """

    OK = "ok"
    SKIPPED = "skipped"
    ERROR = "error"

    @property
    def is_failure(self) -> bool:
        return self is self.ERROR


class ExitCode(IntEnum):
    """Shell-facing exit codes, intentionally independent of scientific QA."""

    OK = 0
    ERROR = 1
    FATAL = 2


def _freeze_metadata(metadata: Mapping[str, MetadataValue]) -> Mapping[str, MetadataValue]:
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
    status: ExecutionStatus
    stage: str
    message: str
    input_path: Path | None = None
    output_path: Path | None = None
    cause: BaseException | None = field(default=None, repr=False, compare=False)
    traceback: str | None = field(default=None, repr=False, compare=False)
    duration_seconds: float | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)
    fatal: bool = False

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
                object.__setattr__(self, attribute, Path(value))
        if self.duration_seconds is not None:
            duration = float(self.duration_seconds)
            if not math.isfinite(duration) or duration < 0.0:
                raise ValueError("duration_seconds must be finite and non-negative.")
            object.__setattr__(self, "duration_seconds", duration)
        if self.status is not ExecutionStatus.ERROR and (self.cause is not None or self.traceback is not None or self.fatal):
            raise ValueError("Only ERROR results may carry cause, traceback, or fatal=True.")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @classmethod
    def success(cls, stage: str, message: str, **kwargs) -> "ExecutionResult":
        return cls(status=ExecutionStatus.OK, stage=stage, message=message, **kwargs)

    @classmethod
    def skipped(cls, stage: str, message: str, **kwargs) -> "ExecutionResult":
        return cls(status=ExecutionStatus.SKIPPED, stage=stage, message=message, **kwargs)

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
    ) -> "ExecutionResult":
        traceback_text = None
        if include_traceback and cause is not None:
            traceback_text = "".join(traceback_module.format_exception(cause)).rstrip()
        return cls(
            status=ExecutionStatus.ERROR,
            stage=stage,
            message=message,
            input_path=input_path,
            output_path=output_path,
            cause=cause,
            traceback=traceback_text,
            duration_seconds=duration_seconds,
            metadata={} if metadata is None else metadata,
            fatal=fatal,
        )

    @property
    def log_level(self) -> int:
        if self.status is ExecutionStatus.ERROR:
            return logging.ERROR
        return logging.INFO

    def to_log_message(self) -> str:
        tag = {ExecutionStatus.OK: "OK", ExecutionStatus.SKIPPED: "SKIPPED", ExecutionStatus.ERROR: "ERROR"}[self.status]
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
        return self.to_log_message()

    def log(self, logger: logging.Logger) -> None:
        message = self.to_log_message()
        if self.status is ExecutionStatus.ERROR:
            logger.error(message)
        else:
            logger.info(message)
        if self.traceback is not None:
            logger.debug("%s", self.traceback)

    def to_dict(self) -> dict[str, object]:
        cause_payload = None
        if self.cause is not None:
            cause_type = type(self.cause)
            cause_payload = {"type": f"{cause_type.__module__}.{cause_type.__qualname__}", "message": str(self.cause)}
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
            "fatal": self.fatal,
        }


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    results: tuple[ExecutionResult, ...]

    def __post_init__(self) -> None:
        results = tuple(self.results)
        if not all(isinstance(result, ExecutionResult) for result in results):
            raise TypeError("results must contain only ExecutionResult instances.")
        object.__setattr__(self, "results", results)

    @classmethod
    def from_results(cls, results: Iterable[ExecutionResult]) -> "ExecutionSummary":
        return cls(tuple(results))

    @property
    def counts(self) -> dict[ExecutionStatus, int]:
        observed = Counter(result.status for result in self.results)
        return {status: observed[status] for status in (ExecutionStatus.OK, ExecutionStatus.SKIPPED, ExecutionStatus.ERROR)}

    @property
    def overall_status(self) -> ExecutionStatus:
        if any(result.status is ExecutionStatus.ERROR for result in self.results):
            return ExecutionStatus.ERROR
        if any(result.status is ExecutionStatus.OK for result in self.results):
            return ExecutionStatus.OK
        return ExecutionStatus.SKIPPED

    @property
    def exit_code(self) -> ExitCode:
        errors = [result for result in self.results if result.status is ExecutionStatus.ERROR]
        if not errors:
            return ExitCode.OK
        if any(result.fatal for result in errors):
            return ExitCode.FATAL
        return ExitCode.ERROR

    def to_dict(self) -> dict[str, object]:
        return {
            "overall_status": self.overall_status.value,
            "exit_code": int(self.exit_code),
            "counts": {status.value: count for status, count in self.counts.items()},
            "results": [result.to_dict() for result in self.results],
        }
