"""Tests for structured aggregation and ADR-002 exit codes in every CLI."""

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


class _ListLogger:
    """Capture CLI logs without global logging configuration."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        self.records.append(("INFO", message))

    def warning(self, message: str) -> None:
        self.records.append(("WARNING", message))

    def error(self, message: str) -> None:
        self.records.append(("ERROR", message))


def _partial_summary() -> ExecutionSummary:
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
def test_pipeline_clis_return_partial_failure_exit_code(monkeypatch, module: object, operation_name: str) -> None:
    logger = _ListLogger()
    monkeypatch.setattr(module, "load_config", lambda: {})
    monkeypatch.setattr(module, "setup_logger", lambda *_args, **_kwargs: logger)
    monkeypatch.setattr(module, operation_name, lambda *_args, **_kwargs: _partial_summary())

    assert module.main() == 1
    assert any("exit code 1" in message for _, message in logger.records)


def test_lebear_cli_returns_total_failure_exit_code(monkeypatch) -> None:
    logger = _ListLogger()
    parser = types.SimpleNamespace(parse_args=lambda: argparse.Namespace(inputs=[], time_window=None))
    summary = ExecutionSummary.from_results([ExecutionResult.failure("level2", "all failed")])
    monkeypatch.setattr(lebear_cli, "_build_parser", lambda: parser)
    monkeypatch.setattr(lebear_cli, "load_config", lambda: {})
    monkeypatch.setattr(lebear_cli, "setup_logger", lambda *_args, **_kwargs: logger)
    monkeypatch.setattr(lebear_cli, "_process_selected_files", lambda *_args: summary)

    assert lebear_cli.main() == 2
    assert any("exit code 2" in message for _, message in logger.records)


def test_lebear_selected_batch_continues_and_aggregates_mixed_results(tmp_path: Path, monkeypatch) -> None:
    files = [tmp_path / "first_level1_rcs.nc", tmp_path / "second_level1_rcs.nc"]
    logger = _ListLogger()
    calls: list[Path] = []
    args = argparse.Namespace(inputs=[str(path) for path in files], time_window=None)

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
    assert [result.status for result in summary.results] == [
        ExecutionStatus.RECOVERABLE_FAILURE,
        ExecutionStatus.SUCCESS,
    ]
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
