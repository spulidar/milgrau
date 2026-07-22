"""Command-line entry point for MILGRAU LIBIDS Level 0 processing."""

from __future__ import annotations

from milgrau.cli.common import finish_cli, run_guarded
from milgrau.config.loader import load_config
from milgrau.io.logging_utils import setup_logger
from milgrau.level0.libids import process_level_0


def main() -> int:
    """Run LIBIDS from the command line."""
    config = load_config()
    logger = setup_logger("LIBIDS", config=config)
    logger.info("=== Starting MILGRAU LIBIDS processing (Level 0) ===")
    summary = run_guarded("cli.libids", logger, lambda: process_level_0(config, logger))
    return finish_cli("LIBIDS", summary, logger)


if __name__ == "__main__":
    raise SystemExit(main())
