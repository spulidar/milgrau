"""Deterministic immutable meteorology cache with canonical manifests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from milgrau.meteorology.request import MeteorologyProvider, MeteorologyRequest

MANIFEST_SCHEMA = "milgrau-meteorology-cache-v1"


class Era5Release(StrEnum):
    FINAL = "final"
    ERA5T_PROVISIONAL = "era5t_provisional"

    @property
    def provisional(self) -> bool:
        return self is Era5Release.ERA5T_PROVISIONAL


@dataclass(frozen=True, slots=True)
class CachePaths:
    artifact: Path
    manifest: Path
    identity: str


@dataclass(frozen=True, slots=True)
class CacheValidation:
    valid: bool
    reason: str
    manifest: dict[str, Any] | None = None


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(payload: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def _time_tag(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%dT%HZ")


def radiosonde_cache_paths(
    request: MeteorologyRequest,
    nominal_time: datetime,
    *,
    normalized: bool,
) -> CachePaths:
    request_payload = request.artifact_request_payload(
        provider=MeteorologyProvider.RADIOSONDE,
        timestamps=(nominal_time,),
    )
    identity_payload = {
        "provider": "wyoming_siphon",
        "dataset": request.radiosonde_station_id,
        "request": request_payload,
        "artifact": "normalized" if normalized else "raw",
        "manifest_schema": MANIFEST_SCHEMA,
    }
    identity = _identity(identity_payload)
    base = (
        request.cache_directory
        / "radiosonde"
        / "wyoming"
        / request.radiosonde_station_id
        / f"{nominal_time.year:04d}"
        / f"{nominal_time.month:02d}"
    )
    stem = f"{_time_tag(nominal_time)}_{identity[:20]}"
    artifact_dir = base / ("normalized" if normalized else "raw")
    suffix = ".nc" if normalized else ".payload"
    return CachePaths(
        artifact=artifact_dir / f"{stem}{suffix}",
        manifest=base / "manifests" / f"{stem}.{'normalized' if normalized else 'raw'}.json",
        identity=identity,
    )


def era5_cache_paths(
    request: MeteorologyRequest,
    hours: tuple[datetime, ...],
    release: Era5Release,
    *,
    normalized: bool,
) -> CachePaths:
    if not hours:
        raise ValueError("ERA5 cache paths require at least one analysis hour.")
    if len({(value.year, value.month) for value in hours}) != 1:
        raise ValueError("One ERA5 cache artifact cannot cross a month boundary.")
    request_payload = request.artifact_request_payload(
        provider=MeteorologyProvider.ERA5,
        timestamps=hours,
    )
    identity_payload = {
        "provider": "ecmwf_cds",
        "dataset": "reanalysis-era5-complete",
        "release": release.value,
        "request": request_payload,
        "artifact": "normalized" if normalized else "raw_grib",
        "manifest_schema": MANIFEST_SCHEMA,
    }
    identity = _identity(identity_payload)
    first = hours[0]
    base = (
        request.cache_directory
        / "era5"
        / "model_levels"
        / request.site_id
        / f"{first.year:04d}"
        / f"{first.month:02d}"
        / release.value
    )
    stem = f"{_time_tag(hours[0])}-{_time_tag(hours[-1])}_{identity[:20]}"
    artifact_dir = base / ("normalized" if normalized else "raw_grib")
    suffix = ".nc" if normalized else ".grib"
    return CachePaths(
        artifact=artifact_dir / f"{stem}{suffix}",
        manifest=base / "manifests" / f"{stem}.{'normalized' if normalized else 'raw'}.json",
        identity=identity,
    )


def _write_temporary(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _backup_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".bak",
    )
    os.close(descriptor)
    backup = Path(name)
    backup.unlink()
    return backup


def _assert_secret_free(value: object, path: str = "manifest") -> None:
    secret_terms = ("token", "secret", "password", "api_key", "authorization")
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if any(term in key_text for term in secret_terms):
                raise ValueError(f"{path} contains a forbidden credential field: {key}")
            _assert_secret_free(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_secret_free(child, f"{path}[{index}]")


def build_manifest(
    *,
    paths: CachePaths,
    artifact_bytes: bytes,
    provider: str,
    dataset: str,
    request_payload: Mapping[str, object],
    timestamps: tuple[datetime, ...],
    area: tuple[float, float, float, float] | None,
    variables: tuple[str, ...],
    levels: tuple[int, ...],
    release: Era5Release | None,
    normalizer: str | None,
    normalizer_version: str | None,
    source_files: tuple[Mapping[str, object], ...] = (),
    raw_payload_kind: str | None = None,
    dependency_versions: Mapping[str, str] | None = None,
    acquisition_time: datetime | None = None,
) -> dict[str, object]:
    acquired = datetime.now(UTC) if acquisition_time is None else acquisition_time.astimezone(UTC)
    manifest: dict[str, object] = {
        "manifest_schema": MANIFEST_SCHEMA,
        "identity": paths.identity,
        "provider": provider,
        "dataset": dataset,
        "era5_release": release.value if release is not None else None,
        "meteorology_provisional": release.provisional if release is not None else False,
        "request": dict(request_payload),
        "timestamps_utc": [value.astimezone(UTC).isoformat() for value in timestamps],
        "area_north_west_south_east": list(area) if area is not None else None,
        "variables": list(variables),
        "model_levels": list(levels),
        "acquisition_time_utc": acquired.isoformat(),
        "artifact_name": paths.artifact.name,
        "size_bytes": len(artifact_bytes),
        "sha256": sha256_bytes(artifact_bytes),
        "normalizer": normalizer,
        "normalizer_version": normalizer_version,
        "source_files": [dict(value) for value in source_files],
        "raw_payload_kind": raw_payload_kind,
        "dependency_versions": dict(sorted((dependency_versions or {}).items())),
        "validation_status": "valid",
    }
    _assert_secret_free(manifest)
    return manifest


def publish_artifact(paths: CachePaths, artifact_bytes: bytes, manifest: Mapping[str, object]) -> None:
    """Publish an artifact/manifest pair, rolling back either failed replacement."""
    if manifest.get("identity") != paths.identity:
        raise ValueError("Manifest identity does not match cache path identity.")
    if manifest.get("sha256") != sha256_bytes(artifact_bytes):
        raise ValueError("Manifest hash does not match artifact bytes.")
    _assert_secret_free(manifest)
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    targets = (
        (paths.artifact, artifact_bytes),
        (paths.manifest, manifest_bytes),
    )
    temporaries: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    backup_paths: list[Path] = []
    published: set[Path] = set()
    try:
        for target, payload in targets:
            temporaries[target] = _write_temporary(target, payload)
        for target, _ in targets:
            if target.exists():
                backup = _backup_path(target)
                backup_paths.append(backup)
                os.replace(target, backup)
                backups[target] = backup
        for target, _ in targets:
            os.replace(temporaries[target], target)
            published.add(target)
    except Exception:
        rollback_error: OSError | None = None
        for target, _ in reversed(targets):
            backup = backups.get(target)
            try:
                if backup is not None and backup.exists():
                    os.rename(backup, target)
                elif target in published and target.exists():
                    target.unlink()
            except OSError as exc:
                rollback_error = exc
        if rollback_error is not None:
            raise OSError(
                "Cache publication failed and the previous pair could not be restored."
            ) from rollback_error
        raise
    finally:
        for path in (*temporaries.values(), *backup_paths):
            path.unlink(missing_ok=True)


def validate_cached_artifact(
    paths: CachePaths,
    *,
    expected_request: Mapping[str, object],
    expected_release: Era5Release | None,
) -> CacheValidation:
    """Validate exact identity, canonical request, manifest, size and SHA-256."""
    try:
        if not paths.artifact.is_file():
            return CacheValidation(False, "artifact_missing")
        if not paths.manifest.is_file():
            return CacheValidation(False, "manifest_missing")
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return CacheValidation(False, "manifest_not_object")
        if manifest.get("manifest_schema") != MANIFEST_SCHEMA:
            return CacheValidation(False, "manifest_schema_mismatch", manifest)
        if manifest.get("identity") != paths.identity:
            return CacheValidation(False, "identity_mismatch", manifest)
        if manifest.get("request") != dict(expected_request):
            return CacheValidation(False, "request_mismatch", manifest)
        expected_release_value = expected_release.value if expected_release is not None else None
        if manifest.get("era5_release") != expected_release_value:
            return CacheValidation(False, "release_mismatch", manifest)
        if manifest.get("validation_status") != "valid":
            return CacheValidation(False, "validation_status_invalid", manifest)
        payload = paths.artifact.read_bytes()
        if not payload:
            return CacheValidation(False, "artifact_empty", manifest)
        if manifest.get("size_bytes") != len(payload):
            return CacheValidation(False, "size_mismatch", manifest)
        if manifest.get("sha256") != sha256_bytes(payload):
            return CacheValidation(False, "sha256_mismatch", manifest)
        return CacheValidation(True, "valid", manifest)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return CacheValidation(False, "manifest_or_artifact_unreadable")


def source_reference(paths: CachePaths, manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "name": paths.artifact.name,
        "sha256": manifest["sha256"],
        "size_bytes": manifest["size_bytes"],
        "identity": paths.identity,
    }
