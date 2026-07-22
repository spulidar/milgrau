"""Filesystem helpers for MILGRAU raw-data discovery and safe sanitization."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Optional

from milgrau.io.paths import radiosonde_cache_dir, surface_weather_cache_dir
from milgrau.operations import ExecutionResult, ExecutionSummary


class RawFileKind(StrEnum):
    """Read-only classification assigned during raw-data discovery."""

    MEASUREMENT = "measurements"
    DARK_CURRENT = "dark_current"
    SPURIOUS = "spurious"


@dataclass(frozen=True, slots=True)
class RawFileCandidate:
    """One path and its validation/classification result."""

    path: Path
    kind: RawFileKind
    reason: str


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


def _quarantine_destination(path: Path, quarantine_root: Path) -> Path:
    """Return an auditable collision-safe destination for one source path."""
    destination = quarantine_root / path.name
    if not destination.exists():
        return destination
    source_digest = sha256(str(path.absolute()).encode("utf-8")).hexdigest()[:12]
    destination = quarantine_root / f"{path.stem}_{source_digest}{path.suffix}"
    collision_index = 1
    while destination.exists():
        destination = quarantine_root / f"{path.stem}_{source_digest}_{collision_index}{path.suffix}"
        collision_index += 1
    return destination


def quarantine_file(path: str | Path, quarantine_root: str | Path, logger: Optional[logging.Logger] = None) -> ExecutionResult:
    """Explicitly move one file to quarantine; repeated calls are safe skips."""
    source = Path(path)
    quarantine = Path(quarantine_root)
    if not source.exists():
        result = ExecutionResult.skipped("filesystem.quarantine", "Source file is already absent", input_path=source)
    elif not source.is_file():
        result = ExecutionResult.failure(
            "filesystem.quarantine",
            "Only regular files can be quarantined",
            input_path=source,
            cause=IsADirectoryError(source),
        )
    else:
        destination = _quarantine_destination(source, quarantine)
        try:
            ensure_directories(quarantine)
            shutil.move(str(source), str(destination))
            result = ExecutionResult.success(
                "filesystem.quarantine",
                "File quarantined",
                input_path=source,
                output_path=destination,
            )
        except Exception as exc:
            result = ExecutionResult.failure(
                "filesystem.quarantine",
                "Could not quarantine file",
                input_path=source,
                output_path=destination,
                cause=exc,
                include_traceback=True,
            )
    if logger:
        result.log(logger)
    return result


def delete_file(path: str | Path, logger: Optional[logging.Logger] = None) -> ExecutionResult:
    """Explicitly delete one regular file; repeated calls are safe skips."""
    target = Path(path)
    if not target.exists():
        result = ExecutionResult.skipped("filesystem.delete", "File is already absent", input_path=target)
    elif not target.is_file():
        result = ExecutionResult.failure(
            "filesystem.delete",
            "Only regular files can be deleted",
            input_path=target,
            cause=IsADirectoryError(target),
        )
    else:
        try:
            target.unlink()
            result = ExecutionResult.success("filesystem.delete", "File deleted", input_path=target)
        except Exception as exc:
            result = ExecutionResult.failure(
                "filesystem.delete",
                "Could not delete file",
                input_path=target,
                cause=exc,
                include_traceback=True,
            )
    if logger:
        result.log(logger)
    return result


def quarantine_files(
    paths: Iterable[str | Path], quarantine_root: str | Path, logger: Optional[logging.Logger] = None
) -> ExecutionSummary:
    """Explicitly quarantine a finite collection of files."""
    return ExecutionSummary.from_results(quarantine_file(path, quarantine_root, logger) for path in paths)


def delete_files(paths: Iterable[str | Path], logger: Optional[logging.Logger] = None) -> ExecutionSummary:
    """Explicitly delete a finite collection of regular files."""
    return ExecutionSummary.from_results(delete_file(path, logger) for path in paths)


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


def classify_raw_file(path: str | Path, spurious_extensions: Iterable[str]) -> RawFileCandidate:
    """Validate/classify one path without modifying it."""
    candidate = Path(path)
    normalized_extensions = {str(extension).lower() for extension in spurious_extensions}
    if candidate.suffix.lower() in normalized_extensions:
        return RawFileCandidate(candidate, RawFileKind.SPURIOUS, f"extension {candidate.suffix.lower()} is configured as spurious")
    if "dark" in str(candidate).lower():
        return RawFileCandidate(candidate, RawFileKind.DARK_CURRENT, "path contains dark-current marker")
    return RawFileCandidate(candidate, RawFileKind.MEASUREMENT, "candidate measurement file")


def discover_raw_files(
    datadir_name: str,
    logger: Optional[logging.Logger] = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[RawFileCandidate, ...]:
    """Discover and classify raw-tree files without moving or deleting anything."""
    candidates: list[RawFileCandidate] = []

    raw_root = Path(datadir_name)
    if not raw_root.exists():
        if logger:
            logger.error(f"Raw data directory not found: {datadir_name}")
        return ()

    spurious_extensions = tuple(
        ext.lower()
        for ext in _processing_option(config, "spurious_extensions", [".dat", ".dpp", ".zip"])
    )
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
            candidate = classify_raw_file(full_path, spurious_extensions)
            candidates.append(candidate)
            if logger and candidate.kind is RawFileKind.SPURIOUS:
                logger.info(f"  -> Spurious file detected; no action taken: {full_path}")

    return tuple(candidates)


def scan_raw_files(
    datadir_name: str,
    logger: Optional[logging.Logger] = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """Return Licel candidates from a strictly read-only raw-data scan."""
    candidates = discover_raw_files(datadir_name, logger=logger, config=config)
    licel_candidates = [candidate for candidate in candidates if candidate.kind is not RawFileKind.SPURIOUS]
    filepath = [str(candidate.path) for candidate in licel_candidates]
    meas_type = [candidate.kind.value for candidate in licel_candidates]

    return filepath, meas_type
