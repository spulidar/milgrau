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
            comparison = compare_staged_to_archive(manifest, copy)
            if comparison["identical"]:
                store.transition(
                    identity, "remote_exists", action="identical_noop",
                    detail=comparison,
                )
                return {
                    "measurement_id": identity,
                    "remote_page": client._url(
                        f"data_processing/measurements/{identity}/"
                    ),
                    "status": "already_identical",
                }
            store.transition(
                identity, "backed_up", action="replacement_difference_verified",
                detail=comparison,
            )
            # The remote archive and staged upload bytes must still match
            # the approved preflight immediately before deleting anything.
            verify_archive(archive, archive_hash)
            refreshed = validate_upload_package(package)
            if refreshed["files"] != manifest["files"] or (
                refreshed["scc_configuration_id"] != manifest["scc_configuration_id"]
                or refreshed["measurement_id"] != identity
            ):
                store.transition(
                    identity, "failed", action="local_source_changed_before_delete",
                    error="Pending SCC raw/LR changed after preflight; no deletion attempted.",
                )
                raise SCCRemoteError(
                    "Pending SCC upload package changed during remote backup; "
                    "the prior SCC measurement was not deleted."
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
        # Read the new authoritative record so the journal cannot retain
        # the pre-replacement creation timestamp as if it were the new one.
        try:
            confirmed = client.measurement(identity)
            if confirmed is None:
                raise SCCRemoteError(
                    "New SCC measurement disappeared after upload confirmation."
                )
        except Exception as exc:
            store.transition(
                identity, "uncertain", action="post_upload_recheck_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        store.transition(
            identity, "accepted", action="upload_verified",
            remote_creation_date=confirmed.get("creation_date"),
            detail={
                "remote_page": str(result["remote_page"]),
                "replacement": replace_existing,
                "remote_fingerprint": remote_fingerprint(confirmed),
            },
        )
        return result



def resume_after_uncertain_delete(
    config: Mapping[str, Any], package: Path, client: Any,
) -> dict[str, Any]:
    """Operator-confirmed repair of a DELETE-POST uncertainty, never a retry.

    Only a prior *specific* delete_result_uncertain journal event qualifies.
    The server must now prove the ID absent and the original remote archive
    must still verify. This submits the already-approved staged replacement
    under the SAME Measurement_ID; it does not send any further DELETE.
    Failures are never automatically retried, and all evidence stays intact.
    """
    from milgrau.level0.scc_staging import pending_scc_package

    manifest = validate_upload_package(package)
    identity = str(manifest["measurement_id"])
    expected_path = pending_scc_package(config, identity)
    if package.resolve() != expected_path.resolve() or package.is_symlink():
        raise SCCRemoteError("Recovery requires the original pending SCC package.")

    root = state_root(config)
    with operation_lock(root, identity), SubmissionStore(root) as store:
        row = store.get(identity)
        if row is None or row["state"] != "uncertain":
            raise SCCRemoteError(
                "This ID is not in an uncertain state; recovery is not applicable."
            )
        events = store.events(identity)
        if not events or events[-1]["action"] != "delete_result_uncertain":
            raise SCCRemoteError(
                "Recovery is limited to a verified, interrupted DELETE POST. "
                "Do not use it for uncertain uploads or other states."
            )
        required_events = {"backup_verified", "delete_requested"}
        if not required_events.issubset({event["action"] for event in events}):
            raise SCCRemoteError("Replacement backup/delete audit trail incomplete.")

        files = manifest["files"]
        lr_name = next((name for name in files if name != "raw.nc"), None)
        if (
            row["scientific_session_id"] != manifest["scientific_session_id"]
            or row["configuration_id"] != manifest["scc_configuration_id"]
            or row["raw_sha256"] != files["raw.nc"]["sha256"]
            or row["lr_sha256"] != (
                files[lr_name]["sha256"] if lr_name else None
            )
            or row["start_utc"] != manifest["measurement_start_utc"]
            or row["stop_utc"] != manifest["measurement_stop_utc"]
        ):
            raise SCCRemoteError(
                "Local SCC staged data/period diverges from the immutable "
                "submission journal. Recovery blocked."
            )
        if not row["archive_path"] or not row["archive_manifest_sha256"]:
            raise SCCRemoteError("Original remote backup is missing from journal.")
        archive = Path(row["archive_path"])
        archive_base = (root / "archives" / identity).resolve()
        if (
            archive.is_symlink()
            or archive.resolve().parent != archive_base
            or not archive.is_dir()
        ):
            raise SCCRemoteError("Original SCC backup location is unsafe or missing.")
        snapshot = verify_archive(archive, row["archive_manifest_sha256"])
        if snapshot["measurement_id"] != identity:
            raise SCCRemoteError("Original SCC archive has the wrong Measurement_ID.")
        if snapshot["remote_snapshot"]["configuration_id"] != row["configuration_id"]:
            raise SCCRemoteError("Original SCC backup configuration has changed.")

        # No destructive operation here. Require a fresh explicit 404 from
        # the actual SCC API *before* moving out of the blocked journal state.
        if client.measurement(identity) is not None:
            raise SCCRemoteError(
                "SCC ID is present again; DO NOT upload over another record. "
                "Investigate with the operator."
            )
        refreshed = validate_upload_package(package)
        if refreshed["files"] != manifest["files"]:
            raise SCCRemoteError("Local SCC input changed during recovery preflight.")

        store.transition(
            identity, "remote_deleted",
            action="operator_confirmed_remote_absence_after_uncertain_delete",
            detail={
                "archive_path": str(archive),
                "archive_manifest_sha256": row["archive_manifest_sha256"],
                "prior_event": events[-1]["action"],
            },
        )
        store.transition(identity, "uploading", action="recovery_upload_requested")
        try:
            # The client performs a second read-only API absence check,
            # validates NetCDF/LR, and sends exactly ONE new-upload POST.
            result = client.upload_package(package)
            confirmed = client.measurement(identity)
            if confirmed is None:
                raise SCCRemoteError("Recovery upload has no SCC API confirmation.")
            system = confirmed.get("system", {})
            if not isinstance(system, Mapping) or (
                system.get("id") != manifest["scc_configuration_id"]
            ):
                raise SCCRemoteError(
                    "Recovered SCC record has unexpected configuration."
                )
            if not confirmed.get("creation_date"):
                raise SCCRemoteError(
                    "Recovered SCC record has no authoritative creation timestamp."
                )
        except Exception as exc:
            store.transition(
                identity, "uncertain", action="recovery_upload_uncertain",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        store.transition(
            identity, "accepted", action="recovery_upload_verified",
            remote_creation_date=str(confirmed["creation_date"]),
            detail={
                "remote_page": str(result["remote_page"]),
                "remote_fingerprint": remote_fingerprint(confirmed),
                "archive_path": str(archive),
                "same_measurement_id": True,
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
