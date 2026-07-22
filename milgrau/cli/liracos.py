"""Command-line entry point for MILGRAU LIRACOS Level 1 visualization."""

from __future__ import annotations

from pathlib import Path

from milgrau.cli.common import finish_cli, run_guarded
from milgrau.config.loader import load_config
from milgrau.io.logging_utils import setup_logger
from milgrau.viz.liracos import process_all_level1_files


def main() -> int:
    """Run LIRACOS from the command line."""
    config = load_config()
    logger = setup_logger("LIRACOS", config=config)
    logger.info("=== Starting MILGRAU LIRACOS rendering (Level 1 Visualization) ===")
    summary = run_guarded(
        "cli.liracos",
        logger,
        lambda: process_all_level1_files(config=config, logger=logger, root_dir=Path.cwd()),
    )
    return finish_cli("LIRACOS", summary, logger)


if __name__ == "__main__":
    raise SystemExit(main())
