"""Command-line entry point for MILGRAU LIRACOS Level 1 visualization."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from milgrau.cli.common import finish_cli, run_guarded
from milgrau.config.loader import load_config
from milgrau.io.logging_utils import bind_log_context, setup_logger
from milgrau.io.paths import LEVEL1_SUFFIX, measurement_product_dir, product_save_id
from milgrau.operations import ExecutionStatus, ExecutionSummary
from milgrau.version import __version__
from milgrau.viz.liracos import process_all_level1_files, process_single_nc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="milgrau-liracos", description="Render MILGRAU Level 1 visual products.")
    parser.add_argument(
        "-i",
        "--input",
        dest="inputs",
        action="append",
        default=[],
        help="Level 1 file, product directory, or save ID (YYYYMMDDsaam/sapm/sant). Repeatable.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate plots even when incremental outputs are current.")
    parser.add_argument("--version", action="version", version=f"MILGRAU {__version__}")
    return parser


def _expand_inputs(inputs: list[str], config: dict) -> list[Path]:
    resolved: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.exists() and path.is_dir():
            resolved.extend(sorted(path.rglob(f"*{LEVEL1_SUFFIX}")))
        elif path.exists() and path.is_file():
            resolved.append(path)
        elif re.fullmatch(r"\d{8}sa(?:am|pm|nt)", raw):
            resolved.append(measurement_product_dir(raw, config) / f"{raw}{LEVEL1_SUFFIX}")
        else:
            raise FileNotFoundError(f"Input {raw!r} is not a Level 1 file, product directory, or save ID.")
    unique = sorted(dict.fromkeys(resolved))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError("Level 1 input(s) not found: " + ", ".join(str(path) for path in missing))
    return unique


def _process_selected(args: argparse.Namespace, config: dict, logger, root_dir: Path) -> ExecutionSummary:
    effective_config = (
        {**config, "processing": {**config["processing"], "incremental": False}}
        if args.force
        else config
    )
    results = []
    for path in _expand_inputs(args.inputs, effective_config):
        save_id = product_save_id(path)
        file_logger = bind_log_context(logger, save_id=save_id)
        result = process_single_nc((path, effective_config, root_dir, file_logger))
        if result.status is ExecutionStatus.OK:
            duration = 0.0 if result.duration_seconds is None else result.duration_seconds
            bind_log_context(file_logger, stage="done").info("plots=%s | %.1f s", result.output_path or "none", duration)
        elif result.status is ExecutionStatus.ERROR:
            bind_log_context(file_logger, stage=result.stage.removeprefix("visualization.")).error("%s", result.message)
        results.append(result)
    return ExecutionSummary.from_results(results)


def main() -> int:
    """Run LIRACOS from the command line."""
    args = _build_parser().parse_args()
    config = load_config()
    logger = bind_log_context(setup_logger("LIRACOS", config=config), pipeline="VIZ")
    bind_log_context(logger, stage="start").info("LIRACOS visualization")
    root_dir = Path.cwd()
    if args.inputs:
        operation = lambda: _process_selected(args, config, logger, root_dir)
    else:
        effective_config = (
            {**config, "processing": {**config["processing"], "incremental": False}}
            if args.force
            else config
        )
        operation = lambda: process_all_level1_files(config=effective_config, logger=logger, root_dir=root_dir)
    summary = run_guarded("cli.liracos", logger, operation)
    return finish_cli("LIRACOS", summary, logger)


if __name__ == "__main__":
    raise SystemExit(main())
