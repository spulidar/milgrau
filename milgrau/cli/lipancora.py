"""Command-line entry point for MILGRAU LIPANCORA Level 1 processing."""

from __future__ import annotations

import argparse
from pathlib import Path

from milgrau.cli.common import add_input_argument, finish_cli, run_guarded
from milgrau.config.loader import load_config
from milgrau.io.logging_utils import bind_log_context, setup_logger
from milgrau.io.paths import (
    LEVEL0_SUFFIX,
    build_measurement_id,
    level0_output_path,
    logging_measurement_id,
    measurement_day_dir,
    station_id,
)
from milgrau.io.selection import parse_input_selection
from milgrau.level1.lipancora import _files_requiring_level1, process_level_1, process_single_file
from milgrau.operations import ExecutionStatus, ExecutionSummary
from milgrau.version import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="milgrau-lipancora", description="Run MILGRAU Level 1 processing.")
    add_input_argument(parser, source="Level 0 selection")
    parser.add_argument("--force", action="store_true", help="Reprocess even when the Level 1 product is current.")
    parser.add_argument("--version", action="version", version=f"MILGRAU {__version__}")
    return parser


def _expand_inputs(inputs, config: dict) -> list[Path]:
    selection = parse_input_selection(inputs, config)
    resolved: list[Path] = []

    for measurement_id in sorted(selection.measurement_ids):
        resolved.append(level0_output_path(measurement_id, config))

    canonical_station: str | None = None
    for date_text in sorted(selection.dates):
        if canonical_station is None:
            canonical_station = station_id(config)
        anchor = build_measurement_id(date_text, canonical_station, "00")
        day_dir = measurement_day_dir(anchor, config)
        matches = sorted(day_dir.glob(f"{date_text}_{canonical_station}_??{LEVEL0_SUFFIX}"))
        if not matches:
            raise FileNotFoundError(f"No Level 0 products found for date {date_text}.")
        resolved.extend(matches)

    for path in selection.paths:
        if path.is_dir():
            resolved.extend(sorted(path.rglob(f"*{LEVEL0_SUFFIX}")))
        else:
            resolved.append(path)

    unique = sorted(dict.fromkeys(resolved))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError("Level 0 input(s) not found: " + ", ".join(str(path) for path in missing))
    return unique


def _process_selected(args: argparse.Namespace, config: dict, logger) -> ExecutionSummary:
    """Process explicit Level 0 paths without requiring MILGRAU-specific filenames.

    External SCC raw files may use provider-specific names. File naming is not a
    scientific identity source, so logging falls back to ``-`` when a canonical
    MILGRAU measurement ID cannot be parsed. Ingestion resolves channel identity from
    file metadata plus station.yaml instead.
    """
    files = _expand_inputs(args.inputs, config)
    skipped = []
    if not args.force:
        files, skipped = _files_requiring_level1(files, config, logger)
    results = list(skipped)
    for path in files:
        measurement_id = logging_measurement_id(path)
        file_logger = bind_log_context(logger, measurement_id=measurement_id)
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
