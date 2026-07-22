"""Optional Level 2 QA orchestration, isolated from the retrieval core."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import xarray as xr

from milgrau.io.paths import LEVEL2_SUFFIX
from milgrau.operations import ExecutionResult
from milgrau.provenance import build_product_provenance, output_is_current, write_provenance_manifest

type Level2QAPlotter = Callable[..., list[Path]]
QA_PROVENANCE_INDEX = ".level2_qa.provenance-index.json"
_QA_LOGO_NAMES = ("CC_BY-NC-ND.png", "lalinet_logo2.png", "logo_leal2.png")


def level2_qa_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether Level 2 QA generation is enabled in configuration."""
    qa_config = config.get("visualization", {}).get("level2_qa", {}) or {}
    return bool(qa_config.get("enabled", True))


def _load_plotter() -> Level2QAPlotter:
    """Import the Matplotlib-backed plotter only when QA is requested."""
    from milgrau.viz.level2_qa import plot_all_level2_qa

    return plot_all_level2_qa


def _qa_provenance(
    level1_path: Path,
    level2_path: Path,
    config: Mapping[str, Any],
    root_path: Path,
):
    """Build shared provenance for the complete set of Level 2 QA plots."""
    input_paths = [level1_path, level2_path]
    input_paths.extend(
        logo_path
        for logo_name in _QA_LOGO_NAMES
        if (logo_path := root_path / "img" / logo_name).is_file()
    )
    return build_product_provenance("lebear.qa", input_paths, config)


def _qa_index_path(level2_path: Path) -> Path:
    return level2_path.parent / "level2_qa" / QA_PROVENANCE_INDEX


def _write_qa_index(index_path: Path, generated: list[Path]) -> None:
    """Atomically persist the exact QA output set governed by provenance."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=index_path.parent,
        prefix=f".{index_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"outputs": sorted(path.name for path in generated)}, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, index_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def level2_qa_is_current(
    input_path: str | Path,
    product_path: str | Path,
    config: Mapping[str, Any],
    *,
    root_dir: str | Path | None = None,
) -> bool:
    """Return whether every indexed Level 2 QA plot is intact and current."""
    level1_path = Path(input_path)
    level2_path = Path(product_path)
    root_path = Path.cwd() if root_dir is None else Path(root_dir)
    index_path = _qa_index_path(level2_path)
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        output_names = index["outputs"]
        if not isinstance(output_names, list) or not all(isinstance(name, str) for name in output_names):
            return False
        provenance = _qa_provenance(level1_path, level2_path, config, root_path)
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return all(output_is_current(index_path.parent / name, provenance) for name in output_names)


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
                "Level 2 QA provenance is current",
                input_path=level2_path,
                output_path=qa_dir,
                metadata={"pipeline": "LEBEAR"},
            )
        provenance = _qa_provenance(level1_path, level2_path, config, root_path)
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
        for plot_path in generated:
            write_provenance_manifest(plot_path, provenance)
        _write_qa_index(_qa_index_path(level2_path), generated)
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
