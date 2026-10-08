"""Level 2 figure orchestration owned by LEBEAR."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import xarray as xr

from milgrau.incremental import output_is_current
from milgrau.io.logging_utils import bind_log_context
from milgrau.io.paths import logging_session_id
from milgrau.operations import ExecutionResult

type Level2FigurePlotter = Callable[..., list[Path]]
_FIGURE_LOGO_NAMES = ("CC_BY-NC-ND.png", "lalinet_logo2.png", "logo_leal2.png")


def level2_figures_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether LEBEAR should maintain Level 2 figures."""
    visualization = config.get("visualization")
    if not isinstance(visualization, Mapping):
        raise KeyError("Configuration visualization section is required.")
    section = visualization.get("level2_figures")
    if not isinstance(section, Mapping):
        raise KeyError("Configuration visualization.level2_figures is required.")
    enabled = section.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("Configuration visualization.level2_figures.enabled must be boolean.")
    return enabled


def _load_figure_renderer() -> Level2FigurePlotter:
    from milgrau.viz.level2 import plot_all_level2_figures

    return plot_all_level2_figures


def _figure_file_prefix(level2_path: Path) -> str:
    """Return the source-product prefix used before the Level 2 figure token."""
    stem = level2_path.stem
    if stem.endswith("_L2_scc"):
        return f"{stem.removesuffix('_L2_scc')}_scc"
    if stem.endswith("_L2"):
        return stem.removesuffix("_L2")
    return stem


def _figure_dependencies(
    level1_path: Path,
    level2_path: Path,
    root_path: Path,
) -> tuple[list[Path], list[Path]]:
    inputs = [level1_path, level2_path]
    logos = [
        logo_path
        for logo_name in _FIGURE_LOGO_NAMES
        if (logo_path := root_path / "img" / logo_name).is_file()
    ]
    return inputs, logos


def level2_figures_are_current(
    input_path: str | Path,
    product_path: str | Path,
    config: Mapping[str, Any],
    *,
    root_dir: str | Path | None = None,
) -> bool:
    """Return whether all existing figures for this Level 2 product are current."""
    level1_path = Path(input_path)
    level2_path = Path(product_path)
    root_path = Path.cwd() if root_dir is None else Path(root_dir)
    figures_dir = level2_path.parent / "figures"
    if not figures_dir.is_dir():
        return False

    prefix = _figure_file_prefix(level2_path)
    outputs = sorted(
        path
        for path in figures_dir.iterdir()
        if path.is_file()
        and path.name.startswith(f"{prefix}_L2_")
        and not path.name.endswith(".provenance.json")
    )
    if not outputs:
        return False

    inputs, logos = _figure_dependencies(level1_path, level2_path, root_path)
    return all(
        output_is_current(
            output,
            inputs,
            config=config,
            extra_dependencies=logos,
        )
        for output in outputs
    )


def generate_level2_figures(
    input_path: str | Path,
    product_path: str | Path,
    config: Mapping[str, Any],
    logger: logging.Logger,
    *,
    root_dir: str | Path | None = None,
) -> ExecutionResult:
    """Generate scientific/diagnostic figures without changing Level 2 validity."""
    started_at = time.perf_counter()
    level1_path = Path(input_path)
    level2_path = Path(product_path)
    figures_dir = level2_path.parent / "figures"
    root_path = Path.cwd() if root_dir is None else Path(root_dir)
    session_id = logging_session_id(level2_path)
    figure_logger = bind_log_context(logger, stage="figures", session_id=session_id)

    if not level2_figures_enabled(config):
        return ExecutionResult.skipped(
            "level2.figures",
            "Level 2 figures disabled by configuration",
            input_path=level2_path,
            output_path=figures_dir,
            metadata={"pipeline": "L2", "session_id": session_id},
        )

    try:
        incremental = bool(config.get("processing", {}).get("incremental", False))
        if incremental and level2_figures_are_current(
            level1_path,
            level2_path,
            config,
            root_dir=root_path,
        ):
            return ExecutionResult.skipped(
                "level2.figures",
                "Level 2 figures are up to date",
                input_path=level2_path,
                output_path=figures_dir,
                metadata={"pipeline": "L2", "session_id": session_id},
            )

        figures_dir.mkdir(parents=True, exist_ok=True)
        plotter = _load_figure_renderer()
        with xr.open_dataset(level2_path) as ds_l2, xr.open_dataset(level1_path) as ds_l1:
            ds_l2.load()
            ds_l1.load()
            generated = plotter(
                ds_l2=ds_l2,
                output_folder=figures_dir,
                file_name_prefix=_figure_file_prefix(level2_path),
                config=dict(config),
                root_dir=root_path,
                ds_l1=ds_l1,
            )

        figure_config = config["visualization"]["level2_figures"]
        requested = any(
            bool(figure_config.get(name, False))
            for name in (
                "generate_gluing",
                "generate_molecular_reference",
                "generate_mc_reference",
                "generate_optical_profiles",
            )
        )
        if requested and not generated:
            raise RuntimeError(
                "Level 2 figures are enabled and configured, but no compatible figure was generated."
            )

        figure_logger.info("%d figure(s)", len(generated))
        return ExecutionResult.success(
            "level2.figures",
            f"Generated {len(generated)} Level 2 figure(s)",
            input_path=level2_path,
            output_path=figures_dir,
            duration_seconds=time.perf_counter() - started_at,
            metadata={
                "pipeline": "L2",
                "session_id": session_id,
                "generated": len(generated),
            },
        )
    except Exception as exc:
        return ExecutionResult.failure(
            "level2.figures",
            f"Level 2 figure generation failed for {level2_path.name}",
            fatal=False,
            input_path=level2_path,
            output_path=figures_dir,
            cause=exc,
            include_traceback=True,
            duration_seconds=time.perf_counter() - started_at,
            metadata={"pipeline": "L2", "session_id": session_id},
        )


__all__ = [
    "generate_level2_figures",
    "level2_figures_are_current",
    "level2_figures_enabled",
]
