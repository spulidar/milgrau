"""Level 1 figure orchestration."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping

import xarray as xr

from milgrau.io.logging_utils import bind_log_context
from milgrau.io.paths import logging_session_id, product_session_id
from milgrau.operations import ExecutionResult
from milgrau.viz.atmosphere import plot_atmospheric_profile


def _figure_prefix(level1_path: Path) -> str:
    try:
        return product_session_id(level1_path)
    except ValueError:
        return level1_path.stem.removesuffix("_L1")


def generate_level1_atmospheric_figure(
    level1_path: str | Path,
    config: Mapping[str, Any],
    logger: logging.Logger,
    *,
    root_dir: str | Path | None = None,
) -> ExecutionResult:
    """Generate the atmospheric comparison figure without changing L1 validity."""
    started_at = time.perf_counter()
    path = Path(level1_path)
    output_dir = path.parent / "figures"
    root = Path.cwd() if root_dir is None else Path(root_dir)
    session_id = logging_session_id(path)
    figure_logger = bind_log_context(logger, stage="atmospheric_figure", session_id=session_id)
    try:
        with xr.open_dataset(path) as ds:
            ds.load()
            output = plot_atmospheric_profile(
                ds,
                output_folder=output_dir,
                file_name_prefix=_figure_prefix(path),
                config=config,
                root_dir=root,
            )
        figure_logger.info("%s", output.name)
        return ExecutionResult.success(
            "level1.figure.atmosphere",
            "Level 1 atmospheric comparison figure generated",
            input_path=path,
            output_path=output,
            duration_seconds=time.perf_counter() - started_at,
            metadata={"pipeline": "L1", "session_id": session_id, "figure": "atmospheric_profile"},
        )
    except Exception as exc:
        return ExecutionResult.failure(
            "level1.figure.atmosphere",
            "Level 1 atmospheric comparison figure failed",
            fatal=False,
            input_path=path,
            output_path=output_dir,
            cause=exc,
            include_traceback=True,
            duration_seconds=time.perf_counter() - started_at,
            metadata={"pipeline": "L1", "session_id": session_id, "figure": "atmospheric_profile"},
        )


__all__ = ["generate_level1_atmospheric_figure"]
