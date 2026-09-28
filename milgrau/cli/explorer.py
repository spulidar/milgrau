"""Command-line entry point for the MILGRAU Streamlit explorer."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Protocol

from milgrau.cli.common import finish_cli, run_guarded
from milgrau.operations import ExecutionResult, ExecutionSummary


class _StreamlitCLI(Protocol):
    def main(self) -> object:
        """Run Streamlit and return or raise its process exit value."""


def _normalize_process_exit_code(value: object) -> int:
    """Normalize a third-party CLI return/SystemExit value."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    return 1


def _launch_streamlit(stcli: _StreamlitCLI, app_path: Path, logger: logging.Logger) -> ExecutionSummary:
    """Launch Streamlit and map its process outcome to the MILGRAU contract."""
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    exit_cause: SystemExit | None = None
    try:
        source_exit_code = _normalize_process_exit_code(stcli.main())
    except SystemExit as exc:
        exit_cause = exc
        source_exit_code = _normalize_process_exit_code(exc.code)

    if source_exit_code == 0:
        result = ExecutionResult.success(
            "explorer.complete",
            "Streamlit explorer finished successfully",
            input_path=app_path,
            metadata={"source_exit_code": source_exit_code},
        )
    else:
        result = ExecutionResult.failure(
            "explorer.streamlit",
            f"Streamlit explorer exited with code {source_exit_code}",
            fatal=True,
            input_path=app_path,
            cause=exit_cause,
            include_traceback=exit_cause is not None,
            metadata={"source_exit_code": source_exit_code},
        )
    result.log(logger)
    return ExecutionSummary.from_results([result])


def main() -> int:
    """Launch the Streamlit app with Streamlit's CLI module."""
    from streamlit.web import cli as stcli

    app_path = Path(__file__).resolve().parents[1] / "explorer" / "streamlit_app.py"
    logger = logging.getLogger("MILGRAU.EXPLORER")
    summary = run_guarded(
        "cli.explorer",
        logger,
        lambda: _launch_streamlit(stcli, app_path, logger),
    )
    return finish_cli("EXPLORER", summary, logger)
