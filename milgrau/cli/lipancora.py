"""Command-line entry point for MILGRAU LIPANCORA Level 1 processing."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from milgrau.cli.common import finish_cli, run_guarded
from milgrau.config.loader import load_config
from milgrau.io.logging_utils import bind_log_context, setup_logger
from milgrau.io.paths import measurement_product_dir, product_save_id
from milgrau.level1.lipancora import _files_requiring_level1, process_level_1, process_single_file
from milgrau.operations import ExecutionStatus, ExecutionSummary
from milgrau.version import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="milgrau-lipancora", description="Run MILGRAU Level 1 processing.")
    parser.add_argument(
        "-i",
        "--input",
        dest="inputs",
        action="append",
        default=[],
        help="Level 0 file, product directory, or save ID (YYYYMMDDsaam/sapm/sant). Repeatable.",
    )
    parser.add_argument("--force", action="store_true", help="Reprocess even when the Level 1 product is current.")
    parser.add_argument("--version", action="version", version=f"MILGRAU {__version__}")
    return parser


def _expand_inputs(inputs: list[str], config: dict) -> list[Path]:
    resolved: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.exists() and path.is_dir():
            resolved.extend(sorted(p for p in path.rglob("*.nc") if "level" not in p.name and p.parent.name == p.stem))
        elif path.exists() and path.is_file():
            resolved.append(path)
        elif re.fullmatch(r"\d{8}sa(?:am|pm|nt)", raw):
            resolved.append(measurement_product_dir(raw, config) / f"{raw}.nc")
        else:
            raise FileNotFoundError(f"Input {raw!r} is not a Level 0 file, product directory, or save ID.")
    unique = sorted(dict.fromkeys(resolved))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError("Level 0 input(s) not found: " + ", ".join(str(path) for path in missing))
    return unique


def _process_selected(args: argparse.Namespace, config: dict, logger) -> ExecutionSummary:
    files = _expand_inputs(args.inputs, config)
    skipped = []
    if not args.force:
        files, skipped = _files_requiring_level1(files, config, logger)
    results = list(skipped)
    for path in files:
        save_id = product_save_id(path)
        file_logger = bind_log_context(logger, save_id=save_id)
        result = process_single_file((path, config, file_logger))
        if result.status is ExecutionStatus.OK:
            duration = 0.0 if result.duration_seconds is None else result.duration_seconds
            bind_log_context(file_logger, stage="done").info(
                "%s | %.1f s", result.output_path.name if result.output_path else "no output", duration
            )
        elif result.status is ExecutionStatus.ERROR:
            bind_log_context(file_logger, stage=result.stage.removeprefix("level1.")).error("%s", result.message)
            if result.traceback:
                file_logger.debug("failure traceback\n%s", result.traceback)
        results.append(result)
    return ExecutionSummary.from_results(results)


def main() -> int:
    """Run LIPANCORA from the command line."""
    args = _build_parser().parse_args()
    config = load_config()
    logger = bind_log_context(setup_logger("LIPANCORA", config=config), pipeline="L1")
    bind_log_context(logger, stage="start").info("LIPANCORA Level 1")
    operation = (
        (lambda: _process_selected(args, config, logger))
        if args.inputs
        else (lambda: process_level_1({**config, "processing": {**config["processing"], "incremental": False}} if args.force else config, logger))
    )
    summary = run_guarded("cli.lipancora", logger, operation)
    return finish_cli("LIPANCORA", summary, logger)


if __name__ == "__main__":
    raise SystemExit(main())
