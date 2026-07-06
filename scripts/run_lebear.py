"""Command-line entry point for MILGRAU LEBEAR Level 2 processing."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

from milgrau.config.loader import load_config
from milgrau.io.paths import LEVEL1_SUFFIX, measurement_product_dir
from milgrau.io.logging_utils import setup_logger
from milgrau.pipeline.lebear import LEVEL2_SUFFIX, discover_level1_files, process_single_level1_file


def _incremental_enabled(config: dict) -> bool:
    """Return whether incremental processing is enabled."""
    return bool(config.get("processing", {}).get("incremental", False))


def _level2_output_path(level1_file: Path, variant_tag: str | None = None) -> Path:
    """Return the expected Level 2 output path for one Level 1 file."""
    stem = level1_file.name.replace("_level1_rcs.nc", "")
    if variant_tag:
        safe_tag = str(variant_tag).strip().replace(" ", "_")
        if safe_tag:
            stem = f"{stem}_{safe_tag}"
    return level1_file.parent / f"{stem}{LEVEL2_SUFFIX}"


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for LEBEAR."""
    parser = argparse.ArgumentParser(
        prog="milgrau-lebear",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run MILGRAU Level 2 processing on selected Level 1 files.",
        epilog=(
            "Examples:\n"
            "  milgrau-lebear\n"
            "  milgrau-lebear --input 20250612sant\n"
            "  milgrau-lebear --input 20250612sant --time-window 4:00 5:00\n"
            "  milgrau-lebear --input 02-processed_data/2025/06/20250612sant --time-window 04:00 05:00\n"
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="inputs",
        action="append",
        default=[],
        help="Level 1 file, Level 1 directory, or save ID (for example 20250612sant). Repeatable.",
    )
    parser.add_argument(
        "--time-window",
        dest="time_window",
        nargs=2,
        metavar=("START_UTC", "STOP_UTC"),
        help="Optional UTC time window applied before Level 2 processing. Use HH:MM or ISO UTC timestamps.",
    )
    return parser


def _expand_level1_inputs(inputs: Sequence[str], config: dict) -> list[Path]:
    """Expand CLI inputs into a sorted list of Level 1 NetCDF files."""
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
    """Return a filename-safe tag for one UTC time window."""

    def normalize(raw: str) -> str:
        value = str(raw).strip()
        if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", value):
            parts = value.split(":")
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2]) if len(parts) > 2 else 0
            return f"{hour:02d}{minute:02d}" if second == 0 else f"{hour:02d}{minute:02d}{second:02d}"
        return value.replace(":", "").replace("-", "").replace(" ", "").replace("Z", "")

    start_tag = normalize(start_utc)
    stop_tag = normalize(stop_utc)
    return f"{start_tag}-{stop_tag}"


def main() -> None:
    """Run LEBEAR from the command line."""
    parser = _build_parser()
    args = parser.parse_args()

    config = load_config()
    logger = setup_logger("LEBEAR", config["directories"]["log_dir"])
    logger.info("=== Starting MILGRAU LEBEAR processing (Level 2) ===")

    files = _expand_level1_inputs(args.inputs, config)
    if not files:
        logger.warning("No Level 1 files found for LEBEAR processing.")
        logger.info("=== LEBEAR finished. ===")
        return

    incremental = _incremental_enabled(config)
    files_to_process: list[Path] = []
    skipped_count = 0
    output_tag = None
    if args.time_window is not None:
        output_tag = _format_time_window_tag(args.time_window[0], args.time_window[1])
    for file_path in files:
        output_path = _level2_output_path(file_path, variant_tag=output_tag)
        if incremental and output_path.exists():
            logger.info(f"[SKIPPED] Level 2 already exists for {file_path.name}: {output_path}")
            skipped_count += 1
            continue
        files_to_process.append(file_path)

    if not files_to_process:
        logger.info(f"No Level 1 files require Level 2 processing. Skipped {skipped_count} existing products.")
        logger.info("=== LEBEAR finished. ===")
        return

    logger.info(f"Found {len(files_to_process)} Level 1 files to process ({skipped_count} skipped).")
    for file_path in files_to_process:
        logger.info(
            process_single_level1_file(
                file_path,
                config,
                logger,
                start_utc=args.time_window[0] if args.time_window else None,
                stop_utc=args.time_window[1] if args.time_window else None,
                output_tag=output_tag,
            )
        )

    logger.info("=== LEBEAR finished. ===")


if __name__ == "__main__":
    main()
