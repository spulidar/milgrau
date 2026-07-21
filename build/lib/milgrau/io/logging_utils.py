"""Logging helpers for MILGRAU command-line pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from milgrau.io.paths import DEFAULT_LOG_DIR, log_output_root


def _coerce_log_level(value: Any, default: int = logging.INFO) -> int:
    """Normalize string or integer logging levels into stdlib constants."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized:
            return int(getattr(logging, normalized, default))
    return default


def setup_logger(
    module_name: str,
    log_dir: str | Path | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    root_dir: str | Path | None = None,
) -> logging.Logger:
    """Create a standardized UTF-8 logger for a MILGRAU module."""
    if config is not None:
        resolved_log_dir = log_output_root(config, root_dir=root_dir)
        processing = config.get("processing", {})
        console_level = _coerce_log_level(processing.get("console_level"), logging.INFO)
        file_level = _coerce_log_level(processing.get("file_level"), logging.INFO)
    else:
        resolved_log_dir = Path(DEFAULT_LOG_DIR if log_dir is None else log_dir)
        console_level = logging.INFO
        file_level = logging.INFO

    resolved_log_dir = Path(resolved_log_dir)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_log_dir / f"{module_name.lower()}.log"

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger
