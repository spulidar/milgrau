"""Filesystem helpers for MILGRAU raw-data discovery and safe sanitization."""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

from milgrau.io.paths import radiosonde_cache_dir, surface_weather_cache_dir


def ensure_directories(*directories: str | Path) -> None:
    """Create one or more directories if they do not already exist."""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def _config_section(config: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    """Return one configuration section or an empty mapping."""
    if not config:
        return {}
    section = config.get(key, {})
    return section if isinstance(section, Mapping) else {}


def _processing_option(config: Mapping[str, Any] | None, key: str, default):
    """Return an optional processing setting from config."""
    return _config_section(config, "processing").get(key, default)


def quarantine_file(path: Path, quarantine_root: Path, logger: Optional[logging.Logger]) -> None:
    """Move a spurious file to a quarantine folder instead of deleting it."""
    ensure_directories(quarantine_root)
    destination = quarantine_root / path.name
    if destination.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = quarantine_root / f"{path.stem}_{stamp}{path.suffix}"
    shutil.move(str(path), str(destination))
    if logger:
        logger.info(f"  -> Spurious file quarantined: {path.name} -> {destination}")


def _ignored_raw_scan_dirs(config: Mapping[str, Any] | None, quarantine_dir: Path) -> set[str]:
    """Return directory basenames that should not be scanned as raw Licel data."""
    ignored = {
        "_quarantine",
        ".git",
        "__pycache__",
        surface_weather_cache_dir(config).name,
        radiosonde_cache_dir(config).name,
    }
    if config:
        ignored.update(str(name) for name in _processing_option(config, "raw_scan_ignore_dirs", []))
        ignored.add(quarantine_dir.name)
    return ignored


def scan_raw_files(
    datadir_name: str,
    logger: Optional[logging.Logger] = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """Scan the raw-data tree for Licel files and classify dark-current files."""
    filepath: list[str] = []
    meas_type: list[str] = []

    raw_root = Path(datadir_name)
    if not raw_root.exists():
        if logger:
            logger.error(f"Raw data directory not found: {datadir_name}")
        return filepath, meas_type

    spurious_extensions = tuple(
        ext.lower()
        for ext in _processing_option(config, "spurious_extensions", [".dat", ".dpp", ".zip"])
    )
    quarantine_spurious = bool(_processing_option(config, "quarantine_spurious_files", True))
    delete_spurious = bool(_processing_option(config, "delete_spurious_files", False))
    quarantine_dir = Path(_processing_option(config, "quarantine_dir", str(raw_root / "_quarantine")))
    ignored_dirs = _ignored_raw_scan_dirs(config, quarantine_dir)

    for dirpath, dirnames, files in os.walk(raw_root):
        dirnames.sort()
        files.sort()

        try:
            quarantine_resolved = quarantine_dir.resolve()
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if dirname not in ignored_dirs and Path(dirpath, dirname).resolve() != quarantine_resolved
            ]
        except Exception:
            dirnames[:] = [dirname for dirname in dirnames if dirname not in ignored_dirs]

        for file_name in files:
            full_path = Path(dirpath) / file_name
            suffix = full_path.suffix.lower()

            if suffix in spurious_extensions:
                try:
                    if delete_spurious:
                        full_path.unlink()
                        if logger:
                            logger.info(f"  -> Spurious file deleted: {full_path.name}")
                    elif quarantine_spurious:
                        quarantine_file(full_path, quarantine_dir, logger)
                    else:
                        if logger:
                            logger.debug(f"  -> Spurious file ignored: {full_path.name}")
                except Exception as exc:
                    if logger:
                        logger.warning(f"Could not handle spurious file {full_path}: {exc}")
                continue

            filepath.append(str(full_path))
            meas_type.append("dark_current" if "dark" in str(full_path).lower() else "measurements")

    return filepath, meas_type
