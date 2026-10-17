"""Guarded one-at-a-time SCC submission/replacement with durable transitions.

Replacement is not atomic on the SCC. The journal must reflect uncertainty
if the server may have deleted the previous version or accepted new bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from milgrau.scc.backup import (
    archive_remote_measurement, compare_staged_to_archive, remote_fingerprint,
    verify_archive,
)
from milgrau.scc.client import SCCAlreadyExists, SCCRemoteError
from milgrau.scc.packages import validate_upload_package
from milgrau.scc.state import SubmissionStore, operation_lock, state_root


def submit(
    config: Mapping[str, Any], package: Path, client: Any,
    *, replace_existing: bool = False,
) -> dict[str, Any]:
    """Submit with stable SCC ID; all destructive actions explicit and journalled."""
    manifest = validate_upload_package(package)
    identity = str(manifest["measurement_id"])
    root = state_root(config)
    with operation_lock(root, identity), SubmissionStore(root) as store:
        store.begin(manifest, allow_new_revision=replace_existing)
        existing = client.measurement(identity)
        if existing is not None:
            store.transition(
                identity, "remote_exists", action="remote_preflight", detail={
                    "remote_fingerprint": remote_fingerprint(existing),
                    "remote_creation_date": existing.get("creation_date"),
                }, remote_creation_date=existing.get("creation_date"),
            )
            if not replace_existing:
                raise SCCAlreadyExists(
                    f"SCC already contains {identity}. To replace, use "
                    "--id ID --replace-existing --confirm after reviewing the "
                    "backup and deletion risk."
                )
            original_fingerprint = remote_fingerprint(existing)
            store.transition(identity, "archiving", action="archive_started")
            try:
                archive, archive_hash = archive_remote_measurement(client, existing, root)
                copy = verify_archive(archive, archive_hash)
                if copy["remote_fingerprint"] != original_fingerprint:
                    raise SCCRemoteError("Remote backup fingerprint does not match preflight.")
                # Validate original remote state has not changed while downloading.
                current = client.measurement(identity)
                if current is None or remote_fingerprint(current) != original_fingerprint:
                    raise SCCRemoteError(
                        "Remote SCC measurement changed or disappeared during backup."
                    )
            except Exception as exc:
                store.transition(
                    identity, "failed", action="archive_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            store.transition(
                identity, "backed_up", action="backup_verified",
                archive_path=str(archive), archive_manifest_sha256=archive_hash,
                detail={"archive_path": str(archive), "archive_sha256": archive_hash},
            )
            # This is the point of no automatic retry: deletion POST is not
            # atomic with the later upload. Even a timeout can mean deletion.
            store.transition(identity, "deleting", action="delete_requested")
            try:
                client.delete_measurement(
                    identity, expected_fingerprint=original_fingerprint,
                )
            except Exception as exc:
                store.transition(
                    identity, "uncertain", action="delete_result_uncertain",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            store.transition(identity, "remote_deleted", action="deletion_confirmed")
        elif replace_existing:
            # A deliberate replacement must not unexpectedly create a new
            # measurement when the old one is already missing.
            store.transition(
                identity, "failed", action="replacement_target_missing",
                error="Existing remote SCC record is absent.",
            )
            raise SCCRemoteError(
                f"SCC record {identity} not present; --replace-existing requires "
                "an existing remote record. Use upload --confirm for new IDs."
            )

        store.transition(identity, "uploading", action="upload_requested")
        try:
            result = client.upload_package(package)
        except Exception as exc:
            store.transition(
                identity, "uncertain", action="upload_result_uncertain",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        store.transition(
            identity, "accepted", action="upload_verified",
            remote_creation_date=None,
            detail={
                "remote_page": str(result["remote_page"]),
                "replacement": replace_existing,
            },
        )
        return result


def reconcile_status(
    store: SubmissionStore,
    identity: str,
    remote: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Read-only classification; never clear uncertain without human action."""
    row = store.get(identity)
    if row is None:
        return None
    state = row["state"]
    if state in {"uncertain", "deleting", "remote_deleted", "uploading"}:
        return row
    if remote is None:
        # Remote deletion may have occurred externally. Never silently reset.
        return row
    if state not in {"accepted", "processing", "completed"}:
        return row
    if remote.get("is_being_processed") or remote.get("is_queued"):
        next_state = "processing"
    else:
        finished = all(
            remote.get(name) == 127
            for name in ("elpp", "elda")
        )
        next_state = "completed" if finished else "accepted"
    if state != next_state:
        store.transition(identity, next_state, action="remote_status_observed")
    return store.get(identity)
