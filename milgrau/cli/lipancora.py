"""Command-line entry point for MILGRAU LIPANCORA Level 1 processing."""

from __future__ import annotations

from milgrau.cli.common import finish_cli, run_guarded
from milgrau.config.loader import load_config
from milgrau.io.logging_utils import setup_logger
from milgrau.level1.lipancora import process_level_1


def main() -> int:
    """Run LIPANCORA from the command line."""
    config = load_config()
    logger = setup_logger("LIPANCORA", config=config)
    logger.info("=== Starting MILGRAU LIPANCORA processing (Level 1) ===")
    summary = run_guarded("cli.lipancora", logger, lambda: process_level_1(config, logger))
    return finish_cli("LIPANCORA", summary, logger)


if __name__ == "__main__":
    raise SystemExit(main())
