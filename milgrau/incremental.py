"""Simple timestamp-based incremental product reuse.

MILGRAU deliberately keeps incremental processing lightweight: an output may be
reused when it exists, is non-empty, passes its optional integrity check, and is
not older than any of its inputs, configuration files, extra dependencies, or
installed MILGRAU Python sources.  No hashes or sidecar manifests are written.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

_PACKAGE_ROOT = Path(__file__).resolve().parent


def _as_paths(values: Iterable[str | Path]) -> list[Path]:
    return [Path(value).expanduser() for value in values]


def config_dependency_paths(config: Mapping[str, Any] | None) -> list[Path]:
    """Return config/station files recorded by ``load_config`` when available."""
    if not isinstance(config, Mapping):
        return []
    paths: list[Path] = []
    for key in ("_config_file", "_station_config_path"):
        value = config.get(key)
        if value:
            paths.append(Path(str(value)).expanduser())
    return paths


@lru_cache(maxsize=1)
def newest_package_source_mtime_ns() -> int:
    """Return the newest shipped Python-source mtime without hashing source code."""
    newest = 0
    try:
        for path in _PACKAGE_ROOT.rglob("*.py"):
            if path.is_file():
                newest = max(newest, path.stat().st_mtime_ns)
    except OSError:
        return 0
    return newest


def output_is_current(
    output_path: str | Path,
    input_paths: Iterable[str | Path],
    *,
    config: Mapping[str, Any] | None = None,
    extra_dependencies: Iterable[str | Path] = (),
    integrity_check: Callable[[Path], bool] | None = None,
    include_code: bool = True,
) -> bool:
    """Return whether an existing output is safe to reuse incrementally.

    This intentionally uses only filesystem mtimes plus an optional product
    contract.  Any missing dependency makes the output stale.
    """
    output = Path(output_path).expanduser()
    try:
        if not output.is_file() or output.stat().st_size <= 0:
            return False
        output_mtime = output.stat().st_mtime_ns

        dependencies = [
            *_as_paths(input_paths),
            *config_dependency_paths(config),
            *_as_paths(extra_dependencies),
        ]
        for dependency in dependencies:
            if not dependency.is_file():
                return False
            if dependency.stat().st_mtime_ns > output_mtime:
                return False

        if include_code:
            source_mtime = newest_package_source_mtime_ns()
            if source_mtime <= 0 or source_mtime > output_mtime:
                return False

        if integrity_check is not None and not integrity_check(output):
            return False
        return True
    except OSError:
        return False
