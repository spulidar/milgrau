"""Command-line entry point for MILGRAU LIPANCORA Level 1 processing."""

from __future__ import annotations

import argparse
from pathlib import Path

from milgrau.cli.common import add_input_argument, finish_cli, run_guarded
from milgrau.config.loader import load_config
from milgrau.io.logging_utils import bind_log_context, setup_logger
from milgrau.io.paths import LEVEL0_SUFFIX, LEVEL1_SUFFIX, logging_session_id
from milgrau.io.selection import parse_input_selection, resolve_product_selection
from milgrau.level1.figures import generate_level1_figures
from milgrau.level1.lipancora import _files_requiring_level1, process_level_1, process_single_file
from milgrau.operations import ExecutionStatus, ExecutionSummary
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
        else:
            bind_log_context(file_logger, stage="figures").warning(
                "%s | %s",
                result.message,
                result.cause or "unknown figure failure",
            )
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


def main() -> int:
    """Run LIPANCORA from the command line."""
    args = _build_parser().parse_args()
    config = load_config()
    logger = bind_log_context(setup_logger("LIPANCORA", config=config), pipeline="L1")
    bind_log_context(logger, stage="start").info("LIPANCORA Level 1")
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
