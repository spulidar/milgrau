"""SCC results are external, persistent artifacts, never canonical MILGRAU L1/L2.

Download is read-only. The directory is atomically published only after
verifying every advertised SCC archive, including its SHA-256. Neither a
remote record nor an existing results directory establishes upload ownership.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from milgrau.io.paths import level0_output_path
from milgrau.provenance import file_sha256
from milgrau.scc.backup import (
    URL_FIELDS, _download, remote_fingerprint, remote_summary,
)
from milgrau.scc.client import SCCRemoteError
from milgrau.scc.state import SubmissionStore, operation_lock, state_root

RESULT_TYPES = ("elpp", "elda", "cloudmask", "elic", "elquick", "hirelpp", "eldec")
REQUIRED_TYPES = ("elpp", "elda")


def result_location(config: Mapping[str, Any], session_id: str, identity: str, remote_creation_date: str) -> Path:
    """Return a verified scientific session's *external* SCC results subdirectory."""
    from milgrau.level0.scc_staging import _MEASUREMENT_ID

    if not _MEASUREMENT_ID.fullmatch(identity):
        raise ValueError("Invalid SCC Measurement_ID.")
    level0 = level0_output_path(session_id, config)
    # Do not create a new scientific session for a remote-only SCC record.
    if not level0.is_file():
        raise SCCRemoteError(
            f"Canonical Level 0 {level0} is missing; cannot associate SCC results "
            "with an unverified scientific session."
        )
    if not isinstance(remote_creation_date, str) or not remote_creation_date:
        raise SCCRemoteError("SCC remote creation timestamp is required.")
    digest = hashlib.sha256(
        (identity + "|" + remote_creation_date).encode("utf-8")
    ).hexdigest()[:20]
    return level0.parent / "scc-results" / identity / "generations" / ("remote-" + digest)


def _confirmed_outputs(remote: Mapping[str, Any]) -> dict[str, str]:
    """Require finished ELPP/ELDA and URLs for all advertised completed results."""
    if remote.get("is_queued") or remote.get("is_being_processed"):
        raise SCCRemoteError("SCC measurement is queued or still processing.")
    urls: dict[str, str] = {}
    for product in RESULT_TYPES:
        status = remote.get(product)
        address = remote.get(URL_FIELDS[product])
        if product in REQUIRED_TYPES and status != 127:
            raise SCCRemoteError(
                f"SCC {product.upper()} is not confirmed completed (code={status!r})."
            )
        if status == 127:
            if not isinstance(address, str) or not address.startswith("https://"):
                raise SCCRemoteError(
                    f"SCC {product} completed but has no verifiable download URL."
                )
            urls[product] = address
    return urls


def validate_results(directory: Path) -> dict[str, Any]:
    """Check a finalized, trusted results package before a cache cleanup."""
    manifest_file = directory / "results.json"
    if directory.is_symlink() or not manifest_file.is_file() or manifest_file.is_symlink():
        raise SCCRemoteError("SCC results manifest missing or unsafe.")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if (
            manifest["schema_version"] != 1
            or manifest["measurement_id"] != directory.parent.parent.name
            or not manifest.get("remote_creation_date")
            or not set(REQUIRED_TYPES).issubset(manifest["files"])
        ):
            raise SCCRemoteError("SCC results manifest has invalid identity/types.")
        for name, record in manifest["files"].items():
            if name not in RESULT_TYPES:
                raise SCCRemoteError("Unknown SCC results archive type.")
            path = directory / f"{name}.archive"
            if path.is_symlink() or not path.is_file():
                raise SCCRemoteError(f"Missing SCC {name} archive.")
            if file_sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise SCCRemoteError(f"Corrupted SCC {name} archive.")
        if set(manifest["files"]) != {
            file.name.removesuffix(".archive")
            for file in directory.glob("*.archive") if file.is_file()
        }:
            raise SCCRemoteError("Untracked SCC results archives present.")
        return manifest
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise SCCRemoteError("SCC results package is invalid.") from exc


def download_results(
    config: Mapping[str, Any], identity: str, client: Any,
    *, source_session: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Fetch results for a locally known session; never imply remote upload ownership.

    Durable accepted/processing/completed submissions are promoted to downloaded
    only when the same exact remote creation date and configuration are present.
    Existing, externally uploaded measurements can be downloaded for scientific
    comparison, but do not become eligible for cache cleanup.
    """
    root = state_root(config)
    with operation_lock(root, identity), SubmissionStore(root) as journal:
        row = journal.get(identity)
        if row is None and not source_session:
            raise SCCRemoteError(
                "No submission in local SCC journal. Specify --session SESSION_ID "
                "to download an existing, externally uploaded SCC measurement."
            )
        session_id = source_session or row["scientific_session_id"]
        if row and source_session and source_session != row["scientific_session_id"]:
            raise SCCRemoteError("Session does not match the existing durable SCC journal.")
        remote = client.measurement(identity)
        if remote is None or remote.get("id") != identity:
            raise SCCRemoteError(f"SCC record {identity} is absent or inconsistent.")
        target = result_location(
            config, session_id, identity, str(remote.get("creation_date") or ""),
        )
        system = remote.get("system", {})
        configuration_id = system.get("id") if isinstance(system, Mapping) else None
        if row is not None and configuration_id != row["configuration_id"]:
            raise SCCRemoteError("Remote SCC configuration differs from local submission.")
        endpoints = _confirmed_outputs(remote)
        provenance = {
            "schema_version": 1,
            "measurement_id": identity,
            "scientific_session_id": session_id,
            "scc_configuration_id": configuration_id,
            "remote_creation_date": remote.get("creation_date"),
            "remote_fingerprint": remote_fingerprint(remote),
            "remote_processing": remote_summary(remote),
            "download_urls": endpoints,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "uploaded_by_milgrau": bool(
                row is not None
                and row["state"] in {"accepted", "processing", "completed", "downloaded", "cleaned"}
                and row["remote_creation_date"]
                and row["remote_creation_date"] == remote.get("creation_date")
            ),
            "files": {},
        }
        if row is not None and row["state"] in {
            "uncertain", "deleting", "remote_deleted", "uploading",
        }:
            raise SCCRemoteError(
                "Destructive/upload outcome is uncertain. Resolve the journal "
                "before downloading or cleaning this submission."
            )
        if target.is_dir():
            previous = validate_results(target)
            if (
                previous["remote_fingerprint"] != provenance["remote_fingerprint"]
                or previous["scientific_session_id"] != session_id
            ):
                raise SCCRemoteError(
                    "Existing persistent SCC result snapshot differs from the remote "
                    "measurement; will not overwrite it. Review/retain the prior result."
                )
            # On journal recovery, a verified snapshot can be registered again.
            _record_download(journal, row, target, previous)
            return target, previous
        if target.exists():
            raise SCCRemoteError("SCC results target is not a directory.")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.parent.is_symlink():
            raise SCCRemoteError("SCC results parent is a symlink.")
        staging = Path(tempfile.mkdtemp(prefix=f".{identity}.", dir=target.parent))
        try:
            for kind, url in endpoints.items():
                provenance["files"][kind] = _download(client, url, staging / f"{kind}.archive")
            (staging / "results.json").write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            # Verify the staging file bytes before publishing their directory.
            for kind, info in provenance["files"].items():
                archive = staging / f"{kind}.archive"
                if file_sha256(archive) != info["sha256"]:
                    raise SCCRemoteError("SCC result changed during local staging.")
            # No overwrite of any earlier result, including a racing external client.
            if target.exists():
                raise SCCRemoteError("SCC results target appeared during download.")
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        verified = validate_results(target)
        _record_download(journal, row, target, verified)
        return target, verified


def _record_download(
    store: SubmissionStore,
    row: Mapping[str, Any] | None,
    target: Path,
    result: Mapping[str, Any],
) -> None:
    if row is None:
        return
    # Existing portal records / status=remote_exists / failed submissions
    # may be downloaded for research, but are NOT confirmed MILGRAU uploads.
    if not result["uploaded_by_milgrau"]:
        return
    if row["state"] not in {"accepted", "processing", "completed", "downloaded", "cleaned"}:
        return
    digest = file_sha256(target / "results.json")
    if row["state"] != "cleaned" and (
        row["state"] != "downloaded"
        or row["results_manifest_sha256"] != digest
    ):
        store.transition(
            result["measurement_id"], "downloaded",
            action="verified_remote_results_download",
            results_path=str(target),
            results_manifest_sha256=digest,
            detail={
                "remote_creation_date": result["remote_creation_date"],
                "scc_configuration_id": result["scc_configuration_id"],
                "archive_sha256": {
                    key: item["sha256"] for key, item in result["files"].items()
                },
            },
        )
