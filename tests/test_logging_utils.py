"""Tests for MILGRAU logger ownership, levels, context, and reconfiguration."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

from milgrau.io.logging_utils import bind_log_context, setup_logger


def _config(tmp_path: Path, *, console_level: str = "INFO", file_level: str = "DEBUG") -> dict:
    return {
        "directories": {"log_dir": str(tmp_path)},
        "processing": {"console_level": console_level, "file_level": file_level},
    }


def _close_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_debug_messages_reach_console_and_file_when_configured(tmp_path: Path, capsys) -> None:
    logger = setup_logger("TEST_DEBUG_LEVELS", config=_config(tmp_path, console_level="DEBUG", file_level="DEBUG"))
    try:
        logger.debug("diagnostic detail")
        for handler in logger.handlers:
            handler.flush()

        assert "diagnostic detail" in capsys.readouterr().err
        assert "diagnostic detail" in (tmp_path / "test_debug_levels.log").read_text(encoding="utf-8")
        assert logger.level == logging.DEBUG
    finally:
        _close_handlers(logger)


def test_console_context_matches_compact_pipeline_style(tmp_path: Path, capsys) -> None:
    logger = setup_logger("TEST_CONTEXT", config=_config(tmp_path))
    try:
        contextual = bind_log_context(logger, pipeline="L1", save_id="20240101sant", stage="atmosphere")
        contextual.warning("radiosonde | delta=2.0 h | USSA76 extension=14.2%%")
        for handler in logger.handlers:
            handler.flush()

        stderr = capsys.readouterr().err
        assert "WARN" in stderr
        assert "L1" in stderr
        assert "20240101sant" in stderr
        assert "atmosphere" in stderr
        assert "radiosonde | delta=2.0 h | USSA76 extension=14.2%" in stderr
    finally:
        _close_handlers(logger)


def test_info_is_concise_on_console_while_debug_stays_in_audit_file(tmp_path: Path, capsys) -> None:
    logger = setup_logger("TEST_SPLIT", config=_config(tmp_path, console_level="INFO", file_level="DEBUG"))
    try:
        contextual = bind_log_context(logger, pipeline="L2", save_id="20240101sant", stage="355nm")
        contextual.debug("candidate windows=84 best_rmse=0.012")
        contextual.info("retrieval complete")
        for handler in logger.handlers:
            handler.flush()

        stderr = capsys.readouterr().err
        log_text = (tmp_path / "test_split.log").read_text(encoding="utf-8")
        assert "candidate windows" not in stderr
        assert "retrieval complete" in stderr
        assert "candidate windows=84" in log_text
        assert "pipeline=L2" in log_text
        assert "save_id=20240101sant" in log_text
    finally:
        _close_handlers(logger)


def test_loaded_config_requires_explicit_logging_levels(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="processing.file_level"):
        setup_logger(
            "TEST_MISSING_LEVEL",
            config={"directories": {"log_dir": str(tmp_path)}, "processing": {"console_level": "INFO"}},
        )


def test_invalid_logging_level_fails_instead_of_falling_back(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="processing.console_level"):
        setup_logger("TEST_INVALID_LEVEL", config=_config(tmp_path, console_level="LOUD"))


def test_external_handler_is_preserved_across_setup(tmp_path: Path) -> None:
    logger = logging.getLogger("TEST_EXTERNAL_HANDLER")
    external_output = io.StringIO()
    external_handler = logging.StreamHandler(external_output)
    external_handler.setLevel(logging.INFO)
    logger.addHandler(external_handler)
    try:
        configured = setup_logger("TEST_EXTERNAL_HANDLER", config=_config(tmp_path))
        configured.info("preserved handler")
        external_handler.flush()

        assert external_handler in configured.handlers
        assert "preserved handler" in external_output.getvalue()
    finally:
        _close_handlers(logger)


def test_repeated_setup_replaces_only_owned_handlers_without_duplicates(tmp_path: Path) -> None:
    logger = logging.getLogger("TEST_REPEAT_SETUP")
    external_handler = logging.NullHandler()
    logger.addHandler(external_handler)
    try:
        first = setup_logger("TEST_REPEAT_SETUP", config=_config(tmp_path))
        first_owned = [handler for handler in first.handlers if getattr(handler, "_milgrau_owned_handler", False)]
        second = setup_logger("TEST_REPEAT_SETUP", config=_config(tmp_path, console_level="WARNING"))
        second_owned = [handler for handler in second.handlers if getattr(handler, "_milgrau_owned_handler", False)]

        assert len(first_owned) == 2
        assert len(second_owned) == 2
        assert not any(handler in second.handlers for handler in first_owned)
        assert external_handler in second.handlers
        assert len(second.handlers) == 3
    finally:
        _close_handlers(logger)


def test_handler_levels_are_independent_while_logger_uses_most_verbose(tmp_path: Path, capsys) -> None:
    logger = setup_logger("TEST_SPLIT_LEVELS", config=_config(tmp_path, console_level="WARNING", file_level="DEBUG"))
    try:
        logger.debug("file only debug")
        logger.warning("both destinations")
        for handler in logger.handlers:
            handler.flush()

        stderr = capsys.readouterr().err
        log_text = (tmp_path / "test_split_levels.log").read_text(encoding="utf-8")
        assert "file only debug" not in stderr
        assert "both destinations" in stderr
        assert "file only debug" in log_text
        assert "both destinations" in log_text
        assert logger.level == logging.DEBUG
    finally:
        _close_handlers(logger)
