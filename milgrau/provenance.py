"""FAIR provenance helpers shared by MILGRAU product stages."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

PROVENANCE_ATTRS: tuple[str, ...] = (
    "processing_config_sha256",
    "station_config_sha256",
    "station_profile_id",
    "instrument_calibration_id",
)


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest of one immutable file byte stream."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_provenance(config: Mapping[str, Any]) -> dict[str, str]:
    """Resolve hashes and station/calibration IDs already known at runtime.

    Missing runtime path metadata is omitted rather than represented by an
    invented placeholder. Productive configs loaded by ``load_config`` carry the
    source paths; focused low-level tests may intentionally construct mappings
    without them.
    """
    attrs: dict[str, str] = {}
    config_path = config.get("_config_file")
    if isinstance(config_path, str) and config_path.strip():
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Resolved processing config file no longer exists: {path}")
        attrs["processing_config_sha256"] = file_sha256(path)

    station_path = config.get("_station_config_path")
    if isinstance(station_path, str) and station_path.strip():
        path = Path(station_path)
        if not path.is_file():
            raise FileNotFoundError(f"Resolved station config file no longer exists: {path}")
        attrs["station_config_sha256"] = file_sha256(path)

    resolved = config.get("_resolved_station")
    if isinstance(resolved, Mapping):
        profile_id = resolved.get("profile_id")
        calibration_id = resolved.get("calibration_id")
        if profile_id is not None:
            attrs["station_profile_id"] = str(profile_id)
        if calibration_id is not None:
            attrs["instrument_calibration_id"] = str(calibration_id)
    return attrs


def inherited_provenance(source_attrs: Mapping[str, Any]) -> dict[str, str]:
    """Copy only stable MILGRAU provenance attributes into a derived product."""
    result: dict[str, str] = {}
    for name in PROVENANCE_ATTRS:
        value = source_attrs.get(name)
        if value is not None and str(value).strip():
            result[name] = str(value)
    return result
