"""Non-destructive remote SCC snapshot retained outside disposable staging cache.

Fail closed: never allow a remote delete if an advertised archive download is
missing, corrupt, cross-origin or unexpectedly HTML. A prior external LR
reference must also be retrievable, not silently forgotten.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import netCDF4 as nc

from milgrau.provenance import file_sha256
from milgrau.scc.client import SCCRemoteError

MAX_BACKUP_BYTES = 1_500_000_000
URL_FIELDS = {
    "raw": "raw_file_download_url",
    "elpp": "elpp_products_download_url",
    "elda": "elda_products_download_url",
    "cloudmask": "cloudmask_products_download_url",
    "elic": "elic_products_download_url",
    "elquick": "elquick_products_download_url",
    "hirelpp": "hirelpp_products_download_url",
    "eldec": "eldec_products_download_url",
}
STATE_FIELDS = (
    "id", "creation_date", "start", "stop", "last_processing_end",
    "upload", "elpp", "elda", "cloudmask", "elic", "elquick",
    "hirelpp", "eldec", "is_queued", "is_being_processed",
)


def remote_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """Stable subset for remote-state race detection and provenance."""
    data = {field: record.get(field) for field in STATE_FIELDS}
    system = record.get("system")
    if not isinstance(system, Mapping) or not isinstance(system.get("id"), int):
        raise SCCRemoteError("SCC record lacks authoritative system ID.")
    data["configuration_id"] = int(system["id"])
    return data


def remote_fingerprint(record: Mapping[str, Any]) -> str:
    data = json.dumps(remote_summary(record), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _download(client: Any, url: str, path: Path) -> dict[str, Any]:
    """Stream bounded same-origin download and verify ZIP or NetCDF container."""
    if urlsplit(url).scheme != "https":
        raise SCCRemoteError("Remote archive download URL is not HTTPS.")
    client._require_origin(url)
    relative = url.removeprefix(client.base_url)
    if relative == url:
        raise SCCRemoteError("Remote archive URL does not match configured SCC origin.")
    response = client._read(relative, stream=True)
    if response.status_code != 200:
        raise SCCRemoteError(f"Remote archive download returned HTTP {response.status_code}.")
    try:
        content_type = str(response.headers.get("Content-Type", "")).lower()
        if "text/html" in content_type or "text/plain" in content_type:
            raise SCCRemoteError("Remote archive endpoint returned HTML/text instead of data.")
        digest = hashlib.sha256()
        total = 0
        with path.open("xb") as dest:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_BACKUP_BYTES:
                    raise SCCRemoteError("Remote archive exceeds permitted size.")
                digest.update(chunk)
                dest.write(chunk)
        if total < 16:
            raise SCCRemoteError("Downloaded SCC archive is empty or truncated.")
        with path.open("rb") as handle:
            signature = handle.read(8)
        if signature[:2] == b"PK":
            with zipfile.ZipFile(path) as zipped:
                members = zipped.infolist()
                if not members or len(members) > 10000:
                    raise SCCRemoteError("SCC archive ZIP has invalid member count.")
                for member in members:
                    if member.file_size > MAX_BACKUP_BYTES:
                        raise SCCRemoteError("SCC ZIP member is unexpectedly large.")
                if zipped.testzip() is not None:
                    raise SCCRemoteError("SCC ZIP checksum verification failed.")
            kind = "zip"
        elif signature.startswith(b"CDF") or signature == b"\x89HDF\r\n\x1a\n":
            with nc.Dataset(path) as ds:
                if not ds.dimensions:
                    raise SCCRemoteError("SCC NetCDF backup has no dimensions.")
            kind = "netcdf"
        else:
            raise SCCRemoteError("Remote SCC backup is neither ZIP nor NetCDF.")
        return {"bytes": total, "sha256": digest.hexdigest(), "format": kind}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise SCCRemoteError("Could not verify the SCC archive download.") from exc
    finally:
        close = getattr(response, "close", None)
        if close is not None:
            close()


def _raw_lr_reference(path: Path, kind: str) -> str | None:
    """Find whether the archived SCC raw NetCDF references external LR."""
    if kind == "netcdf":
        with nc.Dataset(path) as ds:
            return str(ds.getncattr("LR_File_Name")) if "LR_File_Name" in ds.ncattrs() else None
    with zipfile.ZipFile(path) as zf:
        candidates = [m for m in zf.infolist() if m.filename.lower().endswith(".nc")]
        if not candidates:
            raise SCCRemoteError("SCC raw ZIP contains no NetCDF.")
        found: set[str] = set()
        for member in candidates:
            if member.file_size > MAX_BACKUP_BYTES:
                raise SCCRemoteError("SCC raw NetCDF member is too large.")
            with zf.open(member) as src:
                raw_data = src.read(MAX_BACKUP_BYTES + 1)
            if len(raw_data) > MAX_BACKUP_BYTES:
                raise SCCRemoteError("SCC raw NetCDF member exceeded backup limit.")
            try:
                with nc.Dataset("memory", memory=raw_data) as ds:
                    if "LR_File_Name" in ds.ncattrs():
                        found.add(str(ds.getncattr("LR_File_Name")))
            except (OSError, RuntimeError) as exc:
                raise SCCRemoteError("Could not inspect downloaded remote NetCDF.") from exc
        if len(found) > 1:
            raise SCCRemoteError("Remote raw backup references multiple different LR files.")
        return next(iter(found), None)


def _back_up_external_lr(
    client: Any, filename: str, destination: Path,
) -> dict[str, Any]:
    """Resolve an SCC-reported downloadable LR file; never guess its URL."""
    if Path(filename).name != filename or not filename.endswith(".nc"):
        raise SCCRemoteError("Remote LR filename is unsafe or malformed.")
    response = client._read("api/v1/lidarratio_files/", params={"filename": filename})
    if response.status_code != 200:
        raise SCCRemoteError("Cannot verify prior SCC LR through ancillary API.")
    try:
        objects = response.json()["objects"]
    except (KeyError, TypeError, ValueError) as exc:
        raise SCCRemoteError("SCC LR ancillary API returned invalid metadata.") from exc
    if not isinstance(objects, list) or len(objects) != 1:
        raise SCCRemoteError("Cannot uniquely resolve original SCC LR auxiliary file.")
    item = objects[0]
    if item.get("filename") != filename:
        raise SCCRemoteError("SCC LR API filename mismatch.")
    # The SCC version may not expose download URLs for the ancillary data.
    # In that case, DO NOT delete the measurement until a verified retrieval
    # endpoint is documented/implemented.
    address = item.get("download_url")
    if not isinstance(address, str):
        raise SCCRemoteError(
            "Original SCC measurement uses external LR, but no verified "
            "LR download URL is exposed. Replacement is blocked."
        )
    return _download(client, address, destination)


def archive_remote_measurement(
    client: Any, record: Mapping[str, Any], root: Path,
) -> tuple[Path, str]:
    """Create an immutable verified snapshot or fail without touching the SCC."""
    summary = remote_summary(record)
    identity = summary["id"]
    if not isinstance(identity, str):
        raise SCCRemoteError("Remote SCC Measurement_ID is missing.")
    from milgrau.level0.scc_staging import _MEASUREMENT_ID

    if not _MEASUREMENT_ID.fullmatch(identity):
        raise SCCRemoteError("Remote SCC Measurement_ID invalid.")
    if summary["is_queued"] or summary["is_being_processed"]:
        raise SCCRemoteError("Cannot replace SCC measurement that is queued/running.")
    raw_url = record.get(URL_FIELDS["raw"])
    if not isinstance(raw_url, str):
        raise SCCRemoteError("Original SCC raw download URL is missing; no deletion allowed.")
    folder = root / "archives" / identity
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=folder))
    try:
        files: dict[str, dict[str, Any]] = {}
        endpoints = {"raw": raw_url}
        for name, field in URL_FIELDS.items():
            if name == "raw":
                continue
            address = record.get(field)
            if isinstance(address, str) and address.startswith("https://"):
                endpoints[name] = address
            elif record.get(name) == 127:
                raise SCCRemoteError(
                    f"SCC reports finished {name} products but no downloadable archive."
                )
        for name, address in endpoints.items():
            files[name + ".archive"] = _download(client, address, staging / (name + ".archive"))
        lr_name = _raw_lr_reference(staging / "raw.archive", files["raw.archive"]["format"])
        if lr_name:
            files["original_lr.archive"] = _back_up_external_lr(
                client, lr_name, staging / "original_lr.archive",
            )
        snapshot = {
            "schema_version": 1,
            "measurement_id": identity,
            "remote_snapshot": summary,
            "remote_fingerprint": remote_fingerprint(record),
            "remote_original_lr_file_name": lr_name,
            "download_urls": endpoints,
            "files": files,
            "archived_utc": datetime.now(timezone.utc).isoformat(),
        }
        snapshot_file = staging / "archive.json"
        snapshot_file.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
        manifest_sha = file_sha256(snapshot_file)
        final = folder / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        staging.rename(final)
        return final, manifest_sha
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_archive(directory: Path, manifest_hash: str) -> dict[str, Any]:
    """Check every archived artifact byte against its own SHA-256 before delete."""
    snapshot = directory / "archive.json"
    if not snapshot.is_file() or file_sha256(snapshot) != manifest_hash:
        raise SCCRemoteError("Missing or altered remote archive manifest.")
    data = json.loads(snapshot.read_text(encoding="utf-8"))
    if not (directory / "raw.archive").is_file():
        raise SCCRemoteError("Remote raw backup is missing.")
    for name, record in data["files"].items():
        path = directory / name
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise SCCRemoteError(f"Remote archive checksum failed for {name}.")
        if path.stat().st_size != record["bytes"]:
            raise SCCRemoteError(f"Remote archive size failed for {name}.")
    return data
