"""Command-line entry point for MILGRAU LIBIDS Level 0 processing."""

from __future__ import annotations

import argparse

from milgrau.cli.common import add_input_argument, finish_cli, run_guarded
from milgrau.config.loader import load_config
from milgrau.io.logging_utils import bind_log_context, setup_logger
from milgrau.level0.libids import process_level_0
from milgrau.version import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="milgrau-libids", description="Run MILGRAU Level 0 processing.")
    add_input_argument(parser, source="Raw measurement selection")
    parser.add_argument("--force", action="store_true", help="Reprocess selected data even when incremental outputs are current.")
    parser.add_argument("--version", action="version", version=f"MILGRAU {__version__}")
    return parser


def main() -> int:
    """Run LIBIDS from the command line."""
    args = _build_parser().parse_args()
    config = load_config()
    logger = bind_log_context(setup_logger("LIBIDS", config=config), pipeline="L0")
    bind_log_context(logger, stage="start").info("LIBIDS Level 0")
    summary = run_guarded(
        "cli.libids",
        logger,
        lambda: process_level_0(config, logger, inputs=args.inputs, force=args.force),
    )
    return finish_cli("LIBIDS", summary, logger)


if __name__ == "__main__":
    raise SystemExit(main())
