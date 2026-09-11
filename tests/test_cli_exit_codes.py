"""Tests for the common MILGRAU CLI outcome and exit-code policy."""

from __future__ import annotations

import argparse
import logging
import sys
import types
from pathlib import Path

import pytest

from milgrau.cli import explorer as explorer_cli
from milgrau.cli import lebear as lebear_cli
from milgrau.cli import libids as libids_cli
from milgrau.cli import lipancora as lipancora_cli
from milgrau.cli import liracos as liracos_cli
from milgrau.operations import ExecutionResult, ExecutionStatus, ExecutionSummary


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[tuple[int, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append((record.levelno, record.getMessage()))


def _logger(name: str) -> tuple[logging.Logger, _CaptureHandler]:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    handler = _CaptureHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, handler


def _error_summary() -> ExecutionSummary:
    return ExecutionSummary.from_results(
        [ExecutionResult.success("pipeline", "done"), ExecutionResult.failure("pipeline", "one failed")]
    )


@pytest.mark.parametrize(
    ("module", "operation_name"),
    [
        (libids_cli, "process_level_0"),
        (lipancora_cli, "process_level_1"),
        (liracos_cli, "process_all_level1_files"),
    ],
)
def test_pipeline_clis_return_one_when_processing_contains_errors(monkeypatch, module: object, operation_name: str) -> None:
    logger, handler = _logger(f"test.cli.{operation_name}")
    monkeypatch.setattr(sys, "argv", [getattr(module, "__name__", "milgrau")])
    monkeypatch.setattr(module, "load_config", lambda: {"processing": {"incremental": False}})
    monkeypatch.setattr(module, "setup_logger", lambda *_args, **_kwargs: logger)
    monkeypatch.setattr(module, operation_name, lambda *_args, **_kwargs: _error_summary())

    assert module.main() == 1
    assert any("1 processed | 0 skipped | 1 with errors" in message for _, message in handler.messages)


def test_lebear_cli_returns_one_when_all_selected_processing_fails(monkeypatch) -> None:
    logger, handler = _logger("test.cli.lebear.errors")
    parser = types.SimpleNamespace(
        parse_args=lambda: argparse.Namespace(inputs=[], time_window=None, force=False)
    )
    summary = ExecutionSummary.from_results([ExecutionResult.failure("level2", "all failed")])
    monkeypatch.setattr(lebear_cli, "_build_parser", lambda: parser)
    monkeypatch.setattr(lebear_cli, "load_config", lambda: {"processing": {"incremental": False}})
    monkeypatch.setattr(lebear_cli, "setup_logger", lambda *_args, **_kwargs: logger)
    monkeypatch.setattr(lebear_cli, "_process_selected_files", lambda *_args: summary)

    assert lebear_cli.main() == 1
    assert any("0 processed | 0 skipped | 1 with errors" in message for _, message in handler.messages)


def test_lebear_cli_reserves_two_for_command_that_cannot_run(monkeypatch) -> None:
    logger, handler = _logger("test.cli.lebear.fatal")
    parser = types.SimpleNamespace(
        parse_args=lambda: argparse.Namespace(inputs=[], time_window=None, force=False)
    )
    monkeypatch.setattr(lebear_cli, "_build_parser", lambda: parser)
    monkeypatch.setattr(lebear_cli, "load_config", lambda: {"processing": {"incremental": False}})
    monkeypatch.setattr(lebear_cli, "setup_logger", lambda *_args, **_kwargs: logger)

    def fail_before_processing(*_args):
        raise RuntimeError("configuration unavailable")

    monkeypatch.setattr(lebear_cli, "_process_selected_files", fail_before_processing)

    assert lebear_cli.main() == 2
    assert any("cannot run command" in message for _, message in handler.messages)


def test_lebear_selected_batch_continues_and_aggregates_mixed_results(tmp_path: Path, monkeypatch) -> None:
    files = [
        tmp_path / "20240101saam_level1_rcs.nc",
        tmp_path / "20240101sapm_level1_rcs.nc",
    ]
    for path in files:
        path.write_text("synthetic", encoding="utf-8")
    logger, _handler = _logger("test.cli.lebear.selected")
    calls: list[Path] = []
    args = argparse.Namespace(inputs=[str(path) for path in files], time_window=None, force=True)

    def fake_process(path, *_args, **_kwargs) -> ExecutionSummary:
        path = Path(path)
        calls.append(path)
        if path == files[0]:
            result = ExecutionResult.failure("level2.retrieval", "first failed", input_path=path)
        else:
            result = ExecutionResult.success("level2.complete", "second succeeded", input_path=path)
        return ExecutionSummary.from_results([result])

    monkeypatch.setattr(lebear_cli, "process_single_level1_file", fake_process)

    summary = lebear_cli._process_selected_files(args, {"processing": {"incremental": False}}, logger)

    assert calls == files
    assert [result.status for result in summary.results] == [ExecutionStatus.ERROR, ExecutionStatus.OK]
    assert int(summary.exit_code) == 1


@pytest.mark.parametrize(("source_code", "expected_code"), [(0, 0), (9, 2)])
def test_explorer_cli_normalizes_streamlit_exit_codes(monkeypatch, source_code: int, expected_code: int) -> None:
    class FakeStreamlitCLI:
        @staticmethod
        def main() -> None:
            raise SystemExit(source_code)

    streamlit_module = types.ModuleType("streamlit")
    web_module = types.ModuleType("streamlit.web")
    web_module.cli = FakeStreamlitCLI
    streamlit_module.web = web_module
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_module)
    monkeypatch.setitem(sys.modules, "streamlit.web", web_module)

    assert explorer_cli.main() == expected_code
