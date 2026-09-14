"""Logging helpers for MILGRAU command-line pipelines."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from milgrau.io.paths import log_output_root

_MILGRAU_HANDLER_MARKER = "_milgrau_owned_handler"
_CONTEXT_DEFAULTS = {"pipeline": "--", "save_id": "-", "stage": "-"}
_LEVEL_LABELS = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARN",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "FATAL",
}
_ISO_OFFSET_DATEFMT = "__MILGRAU_ISO_OFFSET__"


def _clean_log_message(message: str) -> str:
    """Normalize legacy presentation noise without changing event semantics."""
    text = str(message).strip()
    text = re.sub(r"^(?:[-=]>\s*)+", "", text)
    text = re.sub(r"^->\s*", "", text)
    text = " | ".join(part.strip() for part in text.splitlines() if part.strip())

    parse_match = re.fullmatch(r"Parsing\s+(\d+)\s+raw binary files\.{0,3}", text, flags=re.IGNORECASE)
    if parse_match:
        return f"{parse_match.group(1)} raw files"

    dark_match = re.fullmatch(
        r"Successfully injected Dark Current matrix\s*\((\d+) profiles\)\.?",
        text,
        flags=re.IGNORECASE,
    )
    if dark_match:
        return f"dark current | {dark_match.group(1)} profiles"

    timing_match = re.fullmatch(
        r"(Measurement|Background) SCC time axis normalized to (\d+) s from upstream QA; "
        r"adjusted (\d+)/(\d+) profiles \(max (\d+) s\)\.?",
        text,
        flags=re.IGNORECASE,
    )
    if timing_match:
        label, cadence, adjusted, total, maximum = timing_match.groups()
        return f"{label.lower()} {cadence} s | {adjusted}/{total} adjusted | max {maximum} s"

    clipped_match = re.fullmatch(r"clipped:\s*(.+)", text, flags=re.IGNORECASE)
    if clipped_match:
        payload = clipped_match.group(1).strip()
        single = re.fullmatch(r"([^,()\s]+)\(([0-9.]+%)\)", payload)
        if single:
            return f"{single.group(1)} | {single.group(2)} clipped"
        return f"{payload} | clipped"

    neutral_match = re.fullmatch(r"neutral assumed:\s*(.+)", text, flags=re.IGNORECASE)
    if neutral_match:
        return f"neutral assumed | {neutral_match.group(1)}"

    saturation_match = re.fullmatch(r"uncharacterized PC:\s*(.+)", text, flags=re.IGNORECASE)
    if saturation_match:
        return f"uncharacterized PC | {saturation_match.group(1)}"

    return text


class MilgrauLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that carries pipeline/save-id/stage context through calls."""

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        extra = dict(self.extra)
        call_extra = kwargs.get("extra")
        if isinstance(call_extra, Mapping):
            extra.update(call_extra)
        kwargs["extra"] = extra
        return msg, kwargs


class _ContextFormatter(logging.Formatter):
    """Formatter that makes missing context explicit and keeps each event on one row."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802
        if datefmt == _ISO_OFFSET_DATEFMT:
            return datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="seconds")
        return super().formatTime(record, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        for key, default in _CONTEXT_DEFAULTS.items():
            value = getattr(record, key, None)
            setattr(record, key, default if value is None or str(value).strip() == "" else str(value))
        record.levelshort = _LEVEL_LABELS.get(record.levelno, record.levelname)

        original_msg = record.msg
        original_args = record.args
        try:
            record.msg = _clean_log_message(record.getMessage())
            record.args = ()
            return super().format(record)
        finally:
            record.msg = original_msg
            record.args = original_args


class _ConsoleDedupFilter(logging.Filter):
    """Keep calibration warnings readable without hiding them from the audit file."""

    _DEDUP_STAGES = {"calibration", "saturation"}

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[tuple[str, str, str]] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        stage = str(getattr(record, "stage", "-"))
        if record.levelno < logging.WARNING or stage not in self._DEDUP_STAGES:
            return True
        key = (
            str(getattr(record, "pipeline", "--")),
            stage,
            _clean_log_message(record.getMessage()),
        )
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


def bind_log_context(
    logger: logging.Logger | logging.LoggerAdapter,
    *,
    pipeline: str | None = None,
    save_id: str | None = None,
    stage: str | None = None,
) -> MilgrauLoggerAdapter:
    """Return a logger carrying stable MILGRAU context without changing the base logger."""
    if isinstance(logger, logging.LoggerAdapter):
        base_logger = logger.logger
        context = dict(logger.extra)
    else:
        base_logger = logger
        context = {}
    if pipeline is not None:
        context["pipeline"] = str(pipeline)
    if save_id is not None:
        context["save_id"] = str(save_id)
    if stage is not None:
        context["stage"] = str(stage)
    return MilgrauLoggerAdapter(base_logger, context)


def _coerce_log_level(value: Any, label: str) -> int:
    """Normalize one explicitly configured logging level or fail clearly."""
    if isinstance(value, bool):
        raise ValueError(f"Configuration {label} must be a valid logging level.")
    if isinstance(value, int):
        if value in _LEVEL_LABELS:
            return int(value)
        raise ValueError(f"Configuration {label} has unsupported numeric logging level {value!r}.")
    if isinstance(value, str):
        normalized = value.strip().upper()
        resolved = logging.getLevelNamesMapping().get(normalized)
        if isinstance(resolved, int):
            return int(resolved)
    raise ValueError(f"Configuration {label} must be a valid logging level name such as INFO or DEBUG.")


def _configured_log_levels(config: Mapping[str, Any]) -> tuple[int, int]:
    processing = config.get("processing")
    if not isinstance(processing, Mapping):
        raise KeyError("Configuration processing section is required for logging.")
    missing = [key for key in ("console_level", "file_level") if key not in processing]
    if missing:
        raise KeyError(
            "Missing required logging configuration: "
            + ", ".join(f"processing.{key}" for key in missing)
        )
    return (
        _coerce_log_level(processing["console_level"], "processing.console_level"),
        _coerce_log_level(processing["file_level"], "processing.file_level"),
    )


def setup_logger(
    module_name: str,
    log_dir: str | Path | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    root_dir: str | Path | None = None,
) -> logging.Logger:
    """Create MILGRAU console and audit-file loggers with independent verbosity."""
    if config is not None:
        resolved_log_dir = log_output_root(config, root_dir=root_dir)
        console_level, file_level = _configured_log_levels(config)
    else:
        # Standalone/library use may opt out of a loaded MILGRAU config. Productive
        # CLIs always pass config and therefore never use these convenience values.
        resolved_log_dir = Path("logs" if log_dir is None else log_dir)
        console_level = logging.INFO
        file_level = logging.INFO

    resolved_log_dir = Path(resolved_log_dir)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_log_dir / f"{module_name.lower()}.log"

    logger = logging.getLogger(module_name)
    logger.setLevel(min(console_level, file_level))

    # Reconfiguration owns only handlers created by this helper. Integrations
    # may attach capture, telemetry, or application handlers that must survive.
    for handler in list(logger.handlers):
        if getattr(handler, _MILGRAU_HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()

    row_format = "%(asctime)s %(levelshort)-5s %(pipeline)-3s %(save_id)-12s %(stage)-11s %(message)s"
    console_formatter = _ContextFormatter(row_format, datefmt="%H:%M:%S")
    file_formatter = _ContextFormatter(row_format, datefmt=_ISO_OFFSET_DATEFMT)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    setattr(file_handler, _MILGRAU_HANDLER_MARKER, True)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    setattr(stream_handler, _MILGRAU_HANDLER_MARKER, True)
    stream_handler.setLevel(console_level)
    stream_handler.setFormatter(console_formatter)
    stream_handler.addFilter(_ConsoleDedupFilter())
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger
