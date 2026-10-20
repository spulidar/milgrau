"""Atomic SCC submission inputs located beside each scientific session.

Input packages are ancillary to scientific L0/L1/L2; SCC upload state is
tracked independently of their filesystem location.
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

from milgrau.io.paths import processed_data_root, session_dir
from milgrau.provenance import file_sha256

_MEASUREMENT_ID = re.compile(r"^\d{8}[a-z0-9]{3}\d{4}$")


def scc_raw_filename(measurement_id: str) -> str:
    """SCC-defined NetCDF basename, both on disk and as multipart filename."""
    if not _MEASUREMENT_ID.fullmatch(measurement_id):
        raise ValueError(f"Unsafe SCC Measurement_ID: {measurement_id!r}.")
    return f"{measurement_id}.nc"


def manifest_raw_name(manifest: Mapping[str, Any]) -> str:
    """Require the canonical session-local SCC raw NetCDF filename."""
    name = scc_raw_filename(str(manifest["measurement_id"]))
    if name not in manifest["files"]:
        raise ValueError(f"SCC package must reference its canonical raw NetCDF {name}.")
    return name


def scc_staging_root(config: Mapping[str, Any], session_id: str) -> Path:
    """Session-scoped SCC inputs: <session>/scc/scc-inputs/."""
    scc = config.get("scc")
    if not isinstance(scc, Mapping) or scc.get("enabled") is not True:
        raise ValueError("SCC staging requires config.scc.enabled: true.")
    return session_dir(session_id, config) / "scc" / "scc-inputs"


def _staged_matches(config: Mapping[str, Any], measurement_id: str) -> list[Path]:
    """Find exact IDs without relying on a machine-specific SQLite journal."""
    root = processed_data_root(config)
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.glob(f"*/*/*/*/scc/scc-inputs/{measurement_id}")
        if path.is_dir() and not path.is_symlink()
    )


def pending_scc_package(
    config: Mapping[str, Any], measurement_id: str, *,
    session_id: str | None = None,
) -> Path:
    if not _MEASUREMENT_ID.fullmatch(measurement_id):
        raise ValueError(f"Unsafe SCC Measurement_ID: {measurement_id!r}.")
    if session_id is not None:
        return scc_staging_root(config, session_id) / measurement_id
    matches = _staged_matches(config, measurement_id)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one session/scc/scc-inputs/{measurement_id}, "
            f"found {len(matches)}; specify a scientific session when staging."
        )
    return matches[0]


@contextmanager
def atomic_scc_package(
    config: Mapping[str, Any],
    measurement_id: str,
    *,
    session_id: str,
) -> Iterator[Path]:
    """Stage in a sibling temporary directory; expose only complete packages.

    Existing pending submissions are not overwritten, even by a different
    historical session. Re-export must be explicit and cannot silently alter
    already queued upload bytes.
    """
    target = pending_scc_package(config, measurement_id, session_id=session_id)
    elsewhere = [p for p in _staged_matches(config, measurement_id) if p != target]
    if elsewhere:
        raise FileExistsError(
            f"SCC Measurement_ID {measurement_id} already belongs to {elsewhere[0]}."
        )
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
        if not manifest.is_file() or not (staging / scc_raw_filename(measurement_id)).is_file():
            raise ValueError("SCC staging requires <Measurement_ID>.nc and manifest.json.")
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
    sounding_name: str | None = None,
    source_profile_indices: list[int] | None = None,
    source_measurement_files: list[str] | None = None,
) -> dict[str, Any]:
    """Write a local handoff manifest with checksums of the actual upload bytes."""
    if sounding_name is not None:
        from milgrau.level0.scc_sounding import scc_sounding_name

        expected_sounding = scc_sounding_name(measurement_id)
        if sounding_name != expected_sounding:
            raise ValueError(
                f"SCC sounding filename must be {expected_sounding}, got {sounding_name!r}."
            )
    names = [scc_raw_filename(measurement_id)] + (
        [] if ancillary_name is None else [ancillary_name]
    ) + ([] if sounding_name is None else [sounding_name])
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
        "sounding_stage": (
            "awaiting_lipancora" if config.get("_scc_catalog", {})
            .get("scc_policy", {}).get("sounding", {}).get("enabled", False)
            else "not_requested"
        ),
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
    if not isinstance(files, dict):
        raise ValueError("SCC package has no files mapping.")
    raw_name = manifest_raw_name(manifest)
    from milgrau.level0.scc_sounding import scc_sounding_name

    for name, entry in files.items():
        if Path(name).name != name or name not in {
            raw_name, f"lr_{path.name[:13]}.nc",
            scc_sounding_name(path.name),
        }:
            raise ValueError(f"Unsafe/unexpected SCC staged filename: {name!r}.")
        file_path = path / name
        if not file_path.is_file() or file_sha256(file_path) != entry["sha256"]:
            raise ValueError(f"SCC staged checksum mismatch: {name}.")
        if file_path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"SCC staged size mismatch: {name}.")
    return manifest
