"""Command-line entry point for MILGRAU LEBEAR Level 2 processing."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Sequence

from milgrau.cli.common import finish_cli, run_guarded
from milgrau.config.loader import load_config
from milgrau.io.logging_utils import bind_log_context, setup_logger
from milgrau.io.paths import LEVEL1_SUFFIX, level2_output_path, measurement_product_dir, product_save_id
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
            "  milgrau-lebear --input 20250612sant\n"
            "  milgrau-lebear --input 20250612sant --time-window 4:00 5:00\n"
            "  milgrau-lebear --input 20250612sant --force\n"
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="inputs",
        action="append",
        default=[],
        help="Level 1 file, Level 1 directory, or save ID (YYYYMMDDsa<am/pm/nt>). Repeatable.",
    )
    parser.add_argument(
        "--time-window",
        dest="time_window",
        nargs=2,
        metavar=("START_UTC", "STOP_UTC"),
        help="Optional UTC time window applied before Level 2 processing. Use HH:MM or ISO UTC timestamps.",
    )
    parser.add_argument("--force", action="store_true", help="Reprocess even when the Level 2 product is current.")
    parser.add_argument("--version", action="version", version=f"MILGRAU {__version__}")
    return parser


def _expand_level1_inputs(inputs: Sequence[str], config: dict) -> list[Path]:
    resolved: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.exists() and path.is_dir():
            resolved.extend(sorted(path.rglob("*_level1_rcs.nc")))
        elif path.name.endswith(LEVEL1_SUFFIX):
            resolved.append(path)
        elif re.fullmatch(r"\d{8}sa(?:am|pm|nt)", raw):
            stem = raw
            resolved.append(measurement_product_dir(stem, config) / f"{stem}{LEVEL1_SUFFIX}")
        else:
            if path.exists():
                resolved.append(path)
            else:
                raise FileNotFoundError(f"Input {raw!r} is not a directory, Level 1 file, or known save ID.")
    if not resolved:
        return discover_level1_files(config)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return sorted(unique)


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
    return f"{normalize(start_utc)}-{normalize(stop_utc)}"


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
        save_id = product_save_id(file_path)
        file_logger = bind_log_context(logger, save_id=save_id)
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
                    metadata={"pipeline": "L2", "save_id": save_id},
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
        save_id = product_save_id(file_path)
        file_summary = process_single_level1_file(
            file_path,
            config,
            bind_log_context(logger, save_id=save_id),
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
