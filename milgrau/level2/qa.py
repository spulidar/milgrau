"""Optional Level 2 QA orchestration, isolated from the retrieval core."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import xarray as xr

from milgrau.incremental import output_is_current
from milgrau.io.paths import LEVEL2_SUFFIX
from milgrau.operations import ExecutionResult

type Level2QAPlotter = Callable[..., list[Path]]
_QA_LOGO_NAMES = ("CC_BY-NC-ND.png", "lalinet_logo2.png", "logo_leal2.png")


def level2_qa_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether Level 2 QA generation is enabled in configuration."""
    qa_config = config.get("visualization", {}).get("level2_qa", {}) or {}
    return bool(qa_config.get("enabled", True))


def _load_plotter() -> Level2QAPlotter:
    """Import the Matplotlib-backed plotter only when QA is requested."""
    from milgrau.viz.level2_qa import plot_all_level2_qa

    return plot_all_level2_qa


def _qa_dependencies(level1_path: Path, level2_path: Path, root_path: Path) -> tuple[list[Path], list[Path]]:
    inputs = [level1_path, level2_path]
    logos = [
        logo_path
        for logo_name in _QA_LOGO_NAMES
        if (logo_path := root_path / "img" / logo_name).is_file()
    ]
    return inputs, logos


def level2_qa_is_current(
    input_path: str | Path,
    product_path: str | Path,
    config: Mapping[str, Any],
    *,
    root_dir: str | Path | None = None,
) -> bool:
    """Return whether every existing QA plot is newer than its dependencies."""
    level1_path = Path(input_path)
    level2_path = Path(product_path)
    root_path = Path.cwd() if root_dir is None else Path(root_dir)
    qa_dir = level2_path.parent / "level2_qa"
    if not qa_dir.is_dir():
        return False
    outputs = [path for path in qa_dir.iterdir() if path.is_file() and not path.name.startswith(".")]
    if not outputs:
        return False
    inputs, logos = _qa_dependencies(level1_path, level2_path, root_path)
    return all(
        output_is_current(
            output,
            inputs,
            config=config,
            extra_dependencies=logos,
        )
        for output in outputs
    )


def generate_level2_qa(
    input_path: str | Path,
    product_path: str | Path,
    config: Mapping[str, Any],
    logger: logging.Logger,
    *,
    root_dir: str | Path | None = None,
) -> ExecutionResult:
    """Generate QA for an existing product and report failure independently."""
    started_at = time.perf_counter()
    level1_path = Path(input_path)
    level2_path = Path(product_path)
    qa_dir = level2_path.parent / "level2_qa"
    root_path = Path.cwd() if root_dir is None else Path(root_dir)
    try:
        incremental = bool(config.get("processing", {}).get("incremental", False))
        if incremental and level2_qa_is_current(
            level1_path,
            level2_path,
            config,
            root_dir=root_path,
        ):
            return ExecutionResult.skipped(
                "level2.qa",
                "Level 2 QA is up to date",
                input_path=level2_path,
                output_path=qa_dir,
                metadata={"pipeline": "LEBEAR"},
            )

        plotter = _load_plotter()
        with xr.open_dataset(level2_path) as ds_l2, xr.open_dataset(level1_path) as ds_l1:
            ds_l2.load()
            ds_l1.load()
            generated = plotter(
                ds_l2=ds_l2,
                output_folder=qa_dir,
                file_name_prefix=level2_path.name.replace(LEVEL2_SUFFIX, ""),
                config=dict(config),
                root_dir=root_path,
                ds_l1=ds_l1,
            )
        return ExecutionResult.success(
            "level2.qa",
            f"Generated {len(generated)} Level 2 QA plot(s)",
            input_path=level2_path,
            output_path=qa_dir,
            duration_seconds=time.perf_counter() - started_at,
            metadata={"pipeline": "LEBEAR", "generated": len(generated)},
        )
    except Exception as exc:
        return ExecutionResult.failure(
            "level2.qa",
            f"QA generation failed for {level2_path.name}",
            input_path=level2_path,
            output_path=qa_dir,
            cause=exc,
            include_traceback=True,
            duration_seconds=time.perf_counter() - started_at,
            metadata={"pipeline": "LEBEAR"},
        )
