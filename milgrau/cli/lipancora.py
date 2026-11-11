"""Command-line entry point for MILGRAU LIPANCORA Level 1 processing."""

from __future__ import annotations

import argparse
from pathlib import Path

from milgrau.cli.common import add_input_argument, finish_cli, run_guarded
from milgrau.config.loader import load_config
from milgrau.io.cds_queue import read_cds_queue
from milgrau.io.logging_utils import bind_log_context, setup_logger
from milgrau.io.paths import LEVEL0_SUFFIX, LEVEL1_SUFFIX, logging_session_id
from milgrau.io.selection import parse_input_selection, resolve_product_selection
from milgrau.level1.figures import generate_level1_figures
from milgrau.level1.lipancora import _files_requiring_level1, process_level_1, process_single_file
from milgrau.operations import ExecutionResult, ExecutionStatus, ExecutionSummary
from milgrau.version import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="milgrau-lipancora", description="Run MILGRAU Level 1 processing.")
    add_input_argument(parser, source="Level 0 selection")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Generate/repair Level 1 figures from existing L1 products without recomputing Level 1.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess Level 1, or regenerate all figures when used with --figures-only.",
    )
    parser.add_argument(
        "--radiosonde-only",
        action="store_true",
        help=(
            "Use radiosonde as the productive hourly atmosphere without ERA5/CDS. "
            "Use a 12h window for historical daily 12Z soundings; "
            "fall back to explicit USSA76 if neither Wyoming nor IGRA2 is available. "
            "Write the normal *_L1.nc with source provenance; LEBEAR accepts it."
        ),
    )
    parser.add_argument(
        "--cds-queue-threshold", type=int, default=8000,
        help="Heuristic threshold for public CDS queued requests (default: 8000). "
             "Above it, read cached ERA5 only and use radiosonde/USSA76 for missing hours.",
    )
    parser.add_argument(
        "--no-cds-queue-check", action="store_true",
        help="Do not inspect the experimental public CDS dashboard; use normal ERA5.",
    )
    parser.add_argument("--version", action="version", version=f"MILGRAU {__version__}")
    return parser


def _expand_inputs(inputs, config: dict) -> list[Path]:
    selection = parse_input_selection(inputs, config)
    return resolve_product_selection(selection, config, suffix=LEVEL0_SUFFIX)


def _expand_level1_inputs(inputs, config: dict) -> list[Path]:
    selection = parse_input_selection(inputs, config)
    return resolve_product_selection(selection, config, suffix=LEVEL1_SUFFIX)


def _figure_config(config: dict, *, force: bool) -> dict:
    if not force:
        return config
    return {
        **config,
        "processing": {
            **config["processing"],
            "incremental": False,
        },
    }


def _process_figures_only(args: argparse.Namespace, config: dict, logger) -> ExecutionSummary:
    """Generate missing/outdated Level 1 figures without recomputing Level 1."""
    files = _expand_level1_inputs(args.inputs, config)
    if not files:
        bind_log_context(logger, stage="discovery").warning("no Level 1 files found")
        return ExecutionSummary.from_results(
            [
                ExecutionResult.skipped(
                    "level1.figures.discovery",
                    "No Level 1 NetCDF files found",
                    metadata={"pipeline": "L1"},
                )
            ]
        )

    figure_config = _figure_config(config, force=bool(args.force))
    config_file = config.get("_config_file")
    root_dir = (
        Path(str(config_file)).expanduser().resolve().parent
        if isinstance(config_file, str) and config_file.strip()
        else Path.cwd()
    )

    results = []
    bind_log_context(logger, stage="queue").info("%d Level 1 file(s) for figures", len(files))
    for path in files:
        session_id = logging_session_id(path)
        file_logger = bind_log_context(logger, session_id=session_id)
        result = generate_level1_figures(
            path,
            figure_config,
            file_logger,
            root_dir=root_dir,
        )
        if result.status is ExecutionStatus.OK:
            bind_log_context(file_logger, stage="figures").info("%s", result.message)
        elif result.status is ExecutionStatus.SKIPPED:
            bind_log_context(file_logger, stage="figures").info("%s", result.message)
        results.append(result)
    return ExecutionSummary.from_results(results)


def _process_selected(args: argparse.Namespace, config: dict, logger) -> ExecutionSummary:
    """Process explicit Level 0 paths without requiring MILGRAU-specific filenames.

    External SCC raw files may use provider-specific names. File naming is not a
    scientific identity source, so logging falls back to ``-`` when a canonical
    MILGRAU session ID cannot be parsed. Ingestion resolves channel identity from
    file metadata plus station.yaml instead.
    """
    files = _expand_inputs(args.inputs, config)
    skipped = []
    if not args.force:
        files, skipped = _files_requiring_level1(files, config, logger)
    results = list(skipped)
    for path in files:
        session_id = logging_session_id(path)
        file_logger = bind_log_context(logger, session_id=session_id)
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


def _radiosonde_only_config(config: dict) -> dict:
    """CLI-only observational alternative, leaving config.yaml untouched."""
    atmosphere = dict(config["level1"]["atmosphere"])
    if not isinstance(atmosphere.get("radiosonde"), dict):
        raise ValueError("Radiosonde-only mode requires level1.atmosphere.radiosonde configuration.")
    atmosphere["source_priority"] = ["radiosonde", "ussa76"]
    atmosphere.pop("era5", None)
    # Historic SPU radiosondes are often 12Z-only. Seek observations within
    # 12h, falling back to identified USSA76 where neither source has coverage.
    # The ordinary ERA5/QA configuration retains its six-hour tolerance.
    radiosonde = dict(atmosphere["radiosonde"])
    radiosonde["max_time_delta_hours"] = 12.0
    atmosphere["radiosonde"] = radiosonde
    return {
        **config,
        "level1": {**config["level1"], "atmosphere": atmosphere},
        "_radiosonde_only_mode": True,
        "_era5_network_allowed": False,
        "_atmosphere_selection_reason": "explicit_radiosonde_only",
    }


def _apply_cds_queue_policy(config: dict, args: argparse.Namespace, logger) -> dict:
    """Only disable new CDS downloads; validated cached ERA5 remains first."""
    if args.figures_only or args.radiosonde_only or args.no_cds_queue_check:
        return config
    if args.cds_queue_threshold < 0:
        raise ValueError("--cds-queue-threshold must be nonnegative.")
    level1 = config.get("level1")
    if not isinstance(level1, dict) or not isinstance(level1.get("atmosphere"), dict):
        return config
    atmosphere = level1["atmosphere"]
    if "era5" not in atmosphere["source_priority"]:
        return config
    try:
        snapshot = read_cds_queue()
    except (OSError, ValueError, TimeoutError) as exc:
        logger.warning("CDS global queue unknown (%s): keep standard ERA5 policy", exc)
        return {**config, "_cds_queue_state": "unknown"}
    high = snapshot.queued_requests > args.cds_queue_threshold
    logger.info(
        "CDS global queued requests=%d | threshold=%d | %s",
        snapshot.queued_requests, args.cds_queue_threshold,
        "cache-only ERA5" if high else "standard ERA5",
    )
    selected = {
        **config,
        "_cds_queue_state": "high" if high else "normal",
        "_cds_queue_requests": snapshot.queued_requests,
        "_cds_queue_threshold": args.cds_queue_threshold,
        "_era5_network_allowed": not high,
        "_atmosphere_selection_reason": "cds_queue_high" if high else "cds_queue_normal",
    }
    if high:
        # 12Z-only historical SPU soundings can cover evening observations.
        updated = dict(atmosphere)
        radio = dict(updated["radiosonde"])
        radio["max_time_delta_hours"] = max(12.0, float(radio["max_time_delta_hours"]))
        updated["radiosonde"] = radio
        selected["level1"] = {**config["level1"], "atmosphere": updated}
    return selected


def main() -> int:
    """Run LIPANCORA from the command line."""
    parser = _build_parser()
    args = parser.parse_args()
    if args.radiosonde_only and args.figures_only:
        parser.error("--radiosonde-only cannot be combined with --figures-only.")
    config = load_config()
    if args.radiosonde_only:
        config = _radiosonde_only_config(config)
    logger = bind_log_context(setup_logger("LIPANCORA", config=config), pipeline="L1")
    bind_log_context(logger, stage="start").info("LIPANCORA Level 1")
    if args.radiosonde_only:
        bind_log_context(logger, stage="atmosphere").info(
            "radiosonde-only | 12h observed window | Wyoming -> IGRA2 -> USSA76 | no CDS"
        )
    else:
        config = _apply_cds_queue_policy(config, args, logger)
    operation = (
        (lambda: _process_figures_only(args, config, logger))
        if args.figures_only
        else (
            (lambda: _process_selected(args, config, logger))
            if args.inputs
            else (
                lambda: process_level_1(
                    {
                        **config,
                        "processing": {**config["processing"], "incremental": False},
                    }
                    if args.force
                    else config,
                    logger,
                )
            )
        )
    )
    summary = run_guarded("cli.lipancora", logger, operation)
    return finish_cli("LIPANCORA", summary, logger)


if __name__ == "__main__":
    raise SystemExit(main())
