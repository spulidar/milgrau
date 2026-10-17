"""Disposable, atomic SCC submission staging.

Do not record remote acceptance here: .cache may be deleted at any time.
Durable submission state and upload/cleanup commands belong to later phases.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping, Any

from milgrau.io.paths import resolve_project_path
from milgrau.provenance import file_sha256

_MEASUREMENT_ID = re.compile(r"^\d{8}[a-z0-9]{3}\d{4}$")


def scc_staging_root(config: Mapping[str, Any]) -> Path:
    """Return the configured SCC cache, never a scientific output directory."""
    scc = config.get("scc")
    if not isinstance(scc, Mapping) or scc.get("enabled") is not True:
        raise ValueError("SCC staging requires config.scc.enabled: true.")
    setting = scc.get("staging_dir", ".cache/scc")
    if not isinstance(setting, str) or not setting.strip():
        raise ValueError("config.scc.staging_dir must be a non-empty directory.")
    return resolve_project_path(setting)


def pending_scc_package(config: Mapping[str, Any], measurement_id: str) -> Path:
    if not _MEASUREMENT_ID.fullmatch(measurement_id):
        raise ValueError(f"Unsafe SCC Measurement_ID: {measurement_id!r}.")
    return scc_staging_root(config) / "pending" / measurement_id


@contextmanager
def atomic_scc_package(
    config: Mapping[str, Any],
    measurement_id: str,
) -> Iterator[Path]:
    """Stage in a sibling temporary directory; expose only complete packages.

    Existing pending submissions are not overwritten, even by a different
    historical session. Re-export must be explicit and cannot silently alter
    already queued upload bytes.
    """
    target = pending_scc_package(config, measurement_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(
            f"SCC Measurement_ID {measurement_id} is already staged at {target}; "
            "remove/reconcile that pending package before re-export."
        )
    staging = Path(tempfile.mkdtemp(prefix=f".{measurement_id}.", dir=target.parent))
    try:
        yield staging
        manifest = staging / "manifest.json"
        if not manifest.is_file() or not (staging / "raw.nc").is_file():
            raise ValueError("SCC staging is incomplete: raw.nc and manifest.json are required.")
        if target.exists():
            raise FileExistsError(f"SCC Measurement_ID collision: {measurement_id}.")
        # Single-filesystem directory rename presents the complete package at once.
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def write_scc_manifest(
    staging: Path,
    *,
    measurement_id: str,
    session_id: str,
    regime: str,
    month_utc: str,
    config_id: int,
    product_ids: list[int],
    source_segments: list[str],
    profile_start_utc: str,
    profile_stop_utc: str,
    source_canonical_l0: Path,
    config: Mapping[str, Any],
    ancillary_name: str | None,
    source_profile_indices: list[int] | None = None,
    source_measurement_files: list[str] | None = None,
) -> dict[str, Any]:
    """Write a local handoff manifest with checksums of the actual upload bytes."""
    names = ["raw.nc"] + ([] if ancillary_name is None else [ancillary_name])
    checksums = {}
    for name in names:
        path = staging / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"SCC staged file is missing/empty: {path}.")
        checksums[name] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    catalog_path = config.get("_scc_config_path")
    catalog_hash = file_sha256(catalog_path) if catalog_path else None
    manifest = {
        "schema_version": 1,
        "measurement_id": measurement_id,
        "scientific_session_id": session_id,
        "solar_regime": regime,
        "climatology_month_utc": month_utc,
        "scc_configuration_id": int(config_id),
        "scc_product_ids": [int(value) for value in product_ids],
        "source_segments": source_segments,
        "source_profile_indices": [] if source_profile_indices is None else [
            int(index) for index in source_profile_indices
        ],
        "source_measurement_files": [] if source_measurement_files is None else [
            Path(name).name for name in source_measurement_files
        ],
        "measurement_start_utc": profile_start_utc,
        "measurement_stop_utc": profile_stop_utc,
        "canonical_level0_path": str(source_canonical_l0),
        "scc_config_sha256": catalog_hash,
        "staged_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": "pending",  # local file state only; not an assertion of SCC acceptance
        "files": checksums,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


def validate_pending_scc_package(path: Path) -> dict[str, Any]:
    """Verify local metadata + checksums, independently of any remote API."""
    manifest_file = path / "manifest.json"
    if not manifest_file.is_file():
        raise ValueError(f"Missing SCC staging manifest: {manifest_file}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported SCC package manifest.")
    if path.name != manifest.get("measurement_id"):
        raise ValueError("SCC directory name and Measurement_ID disagree.")
    if manifest.get("state") != "pending":
        raise ValueError("SCC package has invalid local staging state.")
    files = manifest.get("files")
    if not isinstance(files, dict) or "raw.nc" not in files:
        raise ValueError("SCC package has no raw.nc entry.")
    for name, entry in files.items():
        if Path(name).name != name or name not in {"raw.nc", f"lr_{path.name}.nc"}:
            raise ValueError(f"Unsafe/unexpected SCC staged filename: {name!r}.")
        file_path = path / name
        if not file_path.is_file() or file_sha256(file_path) != entry["sha256"]:
            raise ValueError(f"SCC staged checksum mismatch: {name}.")
        if file_path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"SCC staged size mismatch: {name}.")
    return manifest
