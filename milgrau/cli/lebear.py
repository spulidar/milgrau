"""Command-line entry point for MILGRAU LEBEAR Level 2 processing."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Sequence

from milgrau.cli.common import add_input_argument, finish_cli, run_guarded
from milgrau.config.loader import load_config
from milgrau.io.logging_utils import bind_log_context, setup_logger
from milgrau.io.paths import (
    LEVEL1_SUFFIX,
    build_measurement_id,
    level2_output_path,
    logging_measurement_id,
    measurement_day_dir,
    station_id,
)
from milgrau.io.selection import parse_input_selection
from milgrau.level2.lebear import level2_output_is_current, process_single_level1_file
from milgrau.level2.discovery import discover_level1_files
from milgrau.level2.qa import generate_level2_qa, level2_qa_enabled
from milgrau.operations import ExecutionResult, ExecutionSummary
from milgrau.version import __version__


def _incremental_enabled(config: dict) -> bool:
    processing = config.get("processing")
    if not isinstance(processing, dict) or "incremental" not in processing:
        raise KeyError("Missing required configuration: processing.incremental")
    if not isinstance(processing["incremental"], bool):
        raise ValueError("Configuration processing.incremental must be a boolean.")
    return processing["incremental"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="milgrau-lebear",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run MILGRAU Level 2 processing on selected Level 1 files.",
        epilog=(
            "Examples:\n"
            "  milgrau-lebear\n"
            "  milgrau-lebear --input 20250612_spu_00\n"
            "  milgrau-lebear --input 20250612_spu_00 --time-window-utc 4:00 5:00\n"
            "  milgrau-lebear --input 20250612_spu_00 --force\n"
        ),
    )
    add_input_argument(parser, source="Level 1 selection")
    parser.add_argument(
        "--time-window-utc",
        dest="time_window",
        nargs=2,
        metavar=("START_UTC", "STOP_UTC"),
        help="Optional UTC time window applied before Level 2 processing. Use HH:MM or ISO UTC timestamps.",
    )
    parser.add_argument("--force", action="store_true", help="Reprocess even when the Level 2 product is current.")
    parser.add_argument("--version", action="version", version=f"MILGRAU {__version__}")
    return parser


def _expand_level1_inputs(inputs, config: dict) -> list[Path]:
    selection = parse_input_selection(inputs, config)
    resolved: list[Path] = []
    canonical_station: str | None = None

    for measurement_id in sorted(selection.measurement_ids):
        resolved.append(measurement_day_dir(measurement_id, config) / f"{measurement_id}{LEVEL1_SUFFIX}")

    for date_text in sorted(selection.dates):
        if canonical_station is None:
            canonical_station = station_id(config)
        anchor = build_measurement_id(date_text, canonical_station, "00")
        day_dir = measurement_day_dir(anchor, config)
        matches = sorted(day_dir.glob(f"{date_text}_{canonical_station}_??{LEVEL1_SUFFIX}"))
        if not matches:
            raise FileNotFoundError(f"No Level 1 products found for date {date_text}.")
        resolved.extend(matches)

    for path in selection.paths:
        if path.is_dir():
            resolved.extend(sorted(path.rglob(f"*{LEVEL1_SUFFIX}")))
        else:
            resolved.append(path)

    unique = sorted(dict.fromkeys(resolved))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError("Level 1 input(s) not found: " + ", ".join(str(path) for path in missing))
    return unique


def _format_time_window_tag(start_utc: str, stop_utc: str) -> str:
    def normalize(raw: str) -> str:
        value = str(raw).strip()
        if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", value):
            parts = value.split(":")
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2]) if len(parts) > 2 else 0
            return f"{hour:02d}{minute:02d}" if second == 0 else f"{hour:02d}{minute:02d}{second:02d}"
        return value.replace(":", "").replace("-", "").replace(" ", "").replace("Z", "")
    return f"{normalize(start_utc)}-{normalize(stop_utc)}Z"


def _process_selected_files(args: argparse.Namespace, config: dict, logger: logging.Logger) -> ExecutionSummary:
    files = _expand_level1_inputs(args.inputs, config)
    if not files:
        bind_log_context(logger, stage="discovery").warning("no Level 1 files found")
        return ExecutionSummary.from_results(
            [ExecutionResult.skipped("level2.discovery", "No Level 1 files found", metadata={"pipeline": "L2"})]
        )

    incremental = _incremental_enabled(config)
    files_to_process: list[Path] = []
    skipped_results: list[ExecutionResult] = []
    output_tag = None
    if args.time_window is not None:
        output_tag = _format_time_window_tag(args.time_window[0], args.time_window[1])
    for file_path in files:
        measurement_id = logging_measurement_id(file_path)
        file_logger = bind_log_context(logger, measurement_id=measurement_id)
        output_path = level2_output_path(file_path, variant_tag=output_tag)
        if not args.force and incremental and level2_output_is_current(
            file_path,
            output_path,
            config,
            start_utc=args.time_window[0] if args.time_window else None,
            stop_utc=args.time_window[1] if args.time_window else None,
            output_tag=output_tag,
        ):
            bind_log_context(file_logger, stage="skip").info("up to date | %s", output_path.name)
            skipped_results.append(
                ExecutionResult.skipped(
                    "level2.incremental",
                    "Level 2 provenance is current",
                    input_path=file_path,
                    output_path=output_path,
                    metadata={"pipeline": "L2", "measurement_id": measurement_id},
                )
            )
            if level2_qa_enabled(config):
                skipped_results.append(generate_level2_qa(file_path, output_path, config, file_logger))
            continue
        files_to_process.append(file_path)

    if not files_to_process:
        bind_log_context(logger, stage="summary").info("all selected Level 2 products are current")
        return ExecutionSummary.from_results(skipped_results)

    bind_log_context(logger, stage="queue").info("%d files to process | %d skipped", len(files_to_process), len(skipped_results))
    results = list(skipped_results)
    for file_path in files_to_process:
        measurement_id = logging_measurement_id(file_path)
        file_summary = process_single_level1_file(
            file_path,
            config,
            bind_log_context(logger, measurement_id=measurement_id),
            start_utc=args.time_window[0] if args.time_window else None,
            stop_utc=args.time_window[1] if args.time_window else None,
            output_tag=output_tag,
        )
        results.extend(file_summary.results)
    return ExecutionSummary.from_results(results)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    config = load_config()
    logger = bind_log_context(setup_logger("LEBEAR", config=config), pipeline="L2")
    bind_log_context(logger, stage="start").info("LEBEAR Level 2")
    summary = run_guarded("cli.lebear", logger, lambda: _process_selected_files(args, config, logger))
    return finish_cli("LEBEAR", summary, logger)


if __name__ == "__main__":
    raise SystemExit(main())
