"""Guarded one-at-a-time SCC submission/replacement with durable transitions.

Replacement is not atomic on the SCC. The journal must reflect uncertainty
if the server may have deleted the previous version or accepted new bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from milgrau.scc.backup import (
    archive_remote_measurement, compare_staged_to_archive, remote_fingerprint,
    verify_archive,
)
from milgrau.scc.client import SCCAlreadyExists, SCCRemoteError
from milgrau.scc.packages import validate_upload_package
from milgrau.scc.state import SubmissionStore, operation_lock, state_root


def _verify_recovery_preimage(
    row: Mapping[str, Any], root: Path, identity: str,
    events: list[dict[str, Any]],
) -> Path | None:
    """Verify the archived preimage OR the durable no-backup waiver.

    A waiver is valid only for the current revision with a recorded remote
    fingerprint and a recorded delete request. Never infer operator consent
    merely from a missing archive or an SCC 404.
    """
    actions = {event["action"] for event in events}
    archive_path = row.get("archive_path")
    archive_sha = row.get("archive_manifest_sha256")
    if archive_path or archive_sha:
        if not archive_path or not archive_sha or "backup_verified" not in actions:
            raise SCCRemoteError("Original remote backup audit trail is incomplete.")
        archive = Path(archive_path)
        archive_base = (root / "archives" / identity).resolve()
        if (
            archive.is_symlink()
            or archive.resolve().parent != archive_base
            or not archive.is_dir()
        ):
            raise SCCRemoteError("Original SCC backup location is unsafe or missing.")
        snapshot = verify_archive(archive, archive_sha)
        if (
            snapshot["measurement_id"] != identity
            or snapshot["remote_snapshot"]["configuration_id"] != row["configuration_id"]
        ):
            raise SCCRemoteError("Original SCC archive identity/configuration mismatch.")
        return archive

    waiver = next(
        (event for event in reversed(events)
         if event["action"] == "operator_confirmed_no_backup"), None,
    )
    if waiver is None or "delete_requested_without_backup" not in actions:
        raise SCCRemoteError(
            "Original remote backup is missing and no explicit no-backup "
            "replacement waiver was recorded for this revision."
        )
    try:
        detail = json.loads(waiver["payload_json"])
    except (ValueError, TypeError) as exc:
        raise SCCRemoteError("Invalid no-backup waiver audit evidence.") from exc
    fingerprint = detail.get("remote_fingerprint")
    if (
        not detail.get("irreversible")
        or not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(c not in "0123456789abcdef" for c in fingerprint)
    ):
        raise SCCRemoteError("No-backup waiver lacks an authoritative remote fingerprint.")
    return None


def _recovery_post(
    store: SubmissionStore, identity: str, package: Path, client: Any,
    manifest: Mapping[str, Any], *,
    requested_action: str, uncertain_action: str, verified_action: str,
    requested_detail: Mapping[str, Any] | None = None,
    verified_detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One journalled POST; failure/recheck ambiguity never causes a retry."""
    store.transition(identity, "uploading", action=requested_action, detail=requested_detail)
    try:
        result = client.upload_package(package)
        confirmed = client.measurement(identity)
        if confirmed is None:
            raise SCCRemoteError("Recovery upload has no SCC API confirmation.")
        system = confirmed.get("system")
        if (
            not isinstance(system, Mapping)
            or system.get("id") != manifest["scc_configuration_id"]
            or not confirmed.get("creation_date")
        ):
            raise SCCRemoteError(
                "Recovered SCC record has unexpected configuration or creation timestamp."
            )
    except Exception as exc:
        store.transition(
            identity, "uncertain", action=uncertain_action,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    store.transition(
        identity, "accepted", action=verified_action,
        remote_creation_date=str(confirmed["creation_date"]),
        detail={
            "remote_page": str(result["remote_page"]),
            "remote_fingerprint": remote_fingerprint(confirmed),
            **(verified_detail or {}),
        },
    )
    return result


def submit(
    config: Mapping[str, Any], package: Path, client: Any,
    *, replace_existing: bool = False, skip_backup: bool = False,
) -> dict[str, Any]:
    """Submit with stable SCC ID; all destructive actions explicit and journalled."""
    if skip_backup and not replace_existing:
        raise ValueError("Backup waiver only applies to explicitly requested SCC replacements.")
    manifest = validate_upload_package(package)
    identity = str(manifest["measurement_id"])
    root = state_root(config)
    with operation_lock(root, identity), SubmissionStore(root) as store:
        previous = store.get(identity)
        from milgrau.scc.state import BLOCKED_RESTART

        if previous is not None and previous["state"] in BLOCKED_RESTART:
            raise RuntimeError(
                f"SCC {identity} recorded as {previous['state']}; "
                "manual remote reconciliation required before any other POST or DELETE."
            )
        if replace_existing:
            # Do not reset an accepted/completed journal revision before
            # downloading and verifying the remote preimage. The legacy flow
            # did so before the backup had even started.
            existing = client.measurement(identity)
            if existing is None:
                raise SCCRemoteError(
                    f"SCC {identity} not present: replacement requires an existing "
                    "remote record. Use an ordinary upload for new IDs."
                )
            original_fingerprint = remote_fingerprint(existing)
            if skip_backup:
                # All read-only checks happen BEFORE beginning a new journal
                # revision. A changed remote or staged package must not demote
                # an accepted historical revision to failed.
                current = client.measurement(identity)
                if current is None or remote_fingerprint(current) != original_fingerprint:
                    raise SCCRemoteError("Remote SCC measurement changed before replacement.")
                refreshed = validate_upload_package(package)
                if refreshed != manifest:
                    raise SCCRemoteError("Staged SCC bytes changed before replacement.")
                store.begin(
                    manifest, allow_new_revision=True, allow_same_revision=True,
                )
                store.transition(
                    identity, "preflight", action="operator_confirmed_no_backup",
                    detail={
                        "remote_fingerprint": original_fingerprint,
                        "remote_creation_date": existing.get("creation_date"),
                        "irreversible": True,
                    },
                )
            else:
                archive, archive_hash = archive_remote_measurement(
                    client, existing, root,
                )
                copy = verify_archive(archive, archive_hash)
                if copy["remote_fingerprint"] != original_fingerprint:
                    raise SCCRemoteError("Remote backup fingerprint does not match preflight.")
                current = client.measurement(identity)
                if current is None or remote_fingerprint(current) != original_fingerprint:
                    raise SCCRemoteError(
                        "Remote SCC measurement changed or disappeared during backup."
                    )
                comparison = compare_staged_to_archive(manifest, copy)
                if comparison["identical"]:
                    if previous is None:
                        store.begin(manifest)
                        store.transition(
                            identity, "remote_exists", action="identical_noop",
                            remote_creation_date=existing.get("creation_date"),
                            detail=comparison,
                        )
                    else:
                        # A verified read-only no-op must not demote a prior
                        # accepted/downloaded journal to remote_exists.
                        store.transition(
                            identity, previous["state"], action="identical_noop",
                            detail=comparison,
                        )
                    return {
                        "measurement_id": identity,
                        "remote_page": client._url(
                            f"data_processing/measurements/{identity}/"
                        ),
                        "status": "already_identical",
                    }
                # Remote preimage and local candidate are known: only now begin
                # the new journal revision and authorize the destructive phase.
                store.begin(
                    manifest, allow_new_revision=True, allow_same_revision=True,
                )
                store.transition(
                    identity, "archiving", action="archive_started",
                    detail={"remote_fingerprint": original_fingerprint},
                )
                store.transition(
                    identity, "backed_up", action="backup_verified",
                    archive_path=str(archive), archive_manifest_sha256=archive_hash,
                    detail={"archive_path": str(archive), "archive_sha256": archive_hash},
                )
                store.transition(
                    identity, "backed_up", action="replacement_difference_verified",
                    detail=comparison,
                )
                verify_archive(archive, archive_hash)
                refreshed = validate_upload_package(package)
                if refreshed["files"] != manifest["files"] or (
                    refreshed["scc_configuration_id"] != manifest["scc_configuration_id"]
                    or refreshed["measurement_id"] != identity
                ):
                    store.transition(
                        identity, "failed", action="local_source_changed_before_delete",
                        error="Pending SCC input changed after preflight; no deletion attempted.",
                    )
                    raise SCCRemoteError(
                        "Pending SCC upload package changed during remote backup; "
                        "the prior SCC measurement was not deleted."
                    )
            store.transition(
                identity, "deleting",
                action="delete_requested_without_backup" if skip_backup else "delete_requested",
            )
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
        else:
            # Ordinary new upload: fail closed if there was any prior remote
            # submission locally or remotely. No implicit overwrites.
            if previous is not None and previous["state"] in {
                "accepted", "processing", "completed", "downloaded", "cleaned",
            }:
                raise RuntimeError(
                    f"SCC {identity} is already journalled; use explicit "
                    "--replace-existing only if its current remote record exists."
                )
            existing = client.measurement(identity)
            if existing is not None:
                store.begin(manifest)
                store.transition(
                    identity, "remote_exists", action="remote_preflight",
                    remote_creation_date=existing.get("creation_date"),
                    detail={
                        "remote_fingerprint": remote_fingerprint(existing),
                        "remote_creation_date": existing.get("creation_date"),
                    },
                )
                raise SCCAlreadyExists(
                    f"SCC already contains {identity}. Use --replace-existing "
                    "only after reviewing the archived remote measurement."
                )
            store.begin(manifest)

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
    """Explicit recovery of an uncertain DELETE with no further DELETE.

    Requires this revision's immutable staged inputs, a verified archive OR
    a recorded no-backup waiver, a specific uncertain-delete event and a new
    SCC API 404. Any ambiguous upload stays blocked for operator review.
    """
    from milgrau.level0.scc_staging import pending_scc_package

    manifest = validate_upload_package(package)
    identity = str(manifest["measurement_id"])
    expected_path = pending_scc_package(config, identity)
    if package.is_symlink() or package.resolve() != expected_path.resolve():
        raise SCCRemoteError("Recovery requires the original pending SCC package.")

    root = state_root(config)
    with operation_lock(root, identity), SubmissionStore(root) as store:
        row = store.get(identity)
        if row is None or row["state"] != "uncertain":
            raise SCCRemoteError(
                "This ID is not in an uncertain state; recovery is not applicable."
            )
        events = store.current_revision_events(identity)
        if not events or events[-1]["action"] != "delete_result_uncertain":
            raise SCCRemoteError(
                "Recovery is limited to a verified, interrupted DELETE POST. "
                "Do not use it for uncertain uploads or other states."
            )
        if not store.matches_manifest(identity, manifest):
            raise SCCRemoteError(
                "Local SCC staged data/period diverges from the immutable "
                "submission journal. Recovery blocked."
            )
        archive = _verify_recovery_preimage(row, root, identity, events)
        required_delete = (
            "delete_requested" if archive is not None
            else "delete_requested_without_backup"
        )
        if required_delete not in {event["action"] for event in events}:
            raise SCCRemoteError("Replacement DELETE audit trail incomplete.")

        if client.measurement(identity) is not None:
            raise SCCRemoteError(
                "SCC ID is present again; DO NOT upload over another record. "
                "Investigate with the operator."
            )
        if validate_upload_package(package) != manifest:
            raise SCCRemoteError("Local SCC input changed during recovery preflight.")

        store.transition(
            identity, "remote_deleted",
            action="operator_confirmed_remote_absence_after_uncertain_delete",
            detail={
                "archive_path": str(archive) if archive is not None else None,
                "archive_manifest_sha256": row["archive_manifest_sha256"],
                "no_backup_waiver": archive is None,
                "prior_event": events[-1]["action"],
            },
        )
        return _recovery_post(
            store, identity, package, client, manifest,
            requested_action="recovery_upload_requested",
            uncertain_action="recovery_upload_uncertain",
            verified_action="recovery_upload_verified",
            verified_detail={
                "archive_path": str(archive) if archive is not None else None,
                "same_measurement_id": True,
            },
        )


def retry_rejected_recovery_upload(
    config: Mapping[str, Any], package: Path, client: Any,
    *, after_lr_fix: bool = False,
) -> dict[str, Any]:
    """One explicit upload-only retry after prior *recovery* POST was uncertain.

    The operator must independently confirm the ID absent in SCC's website.
    The code proves continued API absence, unchanged scientific source bytes,
    a valid original remote backup and the exact previous failure type.
    It NEVER sends DELETE and NEVER retries its own uncertain POST.
    """
    from milgrau.level0.scc_staging import pending_scc_package, scc_raw_filename

    manifest = validate_upload_package(package)
    identity = str(manifest["measurement_id"])
    raw_name = scc_raw_filename(identity)
    if raw_name not in manifest["files"]:
        raise SCCRemoteError(
            "First migrate local SCC cache to Measurement_ID.nc "
            "using upload --migrate-cache."
        )
    expected_path = pending_scc_package(config, identity)
    if package.is_symlink() or package.resolve() != expected_path.resolve():
        raise SCCRemoteError("Recovery requires the exact pending package.")
    root = state_root(config)
    with operation_lock(root, identity), SubmissionStore(root) as store:
        row = store.get(identity)
        if row is None or row["state"] != "uncertain":
            raise SCCRemoteError("SCC journal is not in the required uncertain state.")
        events = store.current_revision_events(identity)
        expected_action = (
            "local_lr_filename_repair" if after_lr_fix
            else "recovery_upload_uncertain"
        )
        if not events or events[-1]["action"] != expected_action:
            raise SCCRemoteError(
                f"Retry requires exact latest journal event {expected_action}; "
                "never repeat a previous uncertain POST."
            )
        required = {
            "backup_verified", "delete_result_uncertain",
            "operator_confirmed_remote_absence_after_uncertain_delete",
            "recovery_upload_requested",
        }
        if after_lr_fix:
            required.update({
                "recovery_upload_uncertain",
                "manual_corrected_upload_uncertain",
            })
        if not required.issubset({event["action"] for event in events}):
            raise SCCRemoteError("Incomplete original SCC recovery audit trail.")
        files = manifest["files"]
        if not store.matches_manifest(identity, manifest):
            raise SCCRemoteError(
                "Pending raw/LR, period, configuration or scientific provenance "
                "changed since original upload. Recovery blocked."
            )
        archive = _verify_recovery_preimage(row, root, identity, events)

        if client.measurement(identity) is not None:
            raise SCCRemoteError(
                "SCC ID reappeared since operator verified portal absence. "
                "Do not overwrite the current remote measurement."
            )
        if validate_upload_package(package) != manifest:
            raise SCCRemoteError("Pending package changed during reconciliation.")
        if after_lr_fix:
            from milgrau.level0.scc_lidar_ratio import scc_lr_file_name

            repaired_name = scc_lr_file_name(identity)
            old_name = f"lr_{identity}.nc"
            if repaired_name not in files or old_name in files:
                raise SCCRemoteError("SCC LR filename repair is incomplete.")
            # The newly recorded revision must be exactly the fix described
            # by the last durable event. No other input changes are accepted.
            detail = json.loads(events[-1]["payload_json"])
            if (
                detail.get("new_lr_filename") != repaired_name
                or detail.get("new_raw_sha256") != row["raw_sha256"]
                or detail.get("unchanged_lr_sha256") != row["lr_sha256"]
                or detail.get("previous_action") != "manual_corrected_upload_uncertain"
            ):
                raise SCCRemoteError("Local LR repair journal revision disagrees with staged inputs.")

        return _recovery_post(
            store, identity, package, client, manifest,
            requested_action=(
                "manual_lr_fixed_upload_requested" if after_lr_fix
                else "manual_corrected_filename_upload_requested"
            ),
            uncertain_action=(
                "manual_lr_fixed_upload_uncertain" if after_lr_fix
                else "manual_corrected_upload_uncertain"
            ),
            verified_action=(
                "manual_lr_fixed_upload_verified" if after_lr_fix
                else "manual_corrected_upload_verified"
            ),
            requested_detail={
                "multipart_filename": raw_name,
                "remote_delete": False,
                "prior_failed_action": events[-1]["action"],
                "original_archive": str(archive) if archive is not None else None,
            },
            verified_detail={
                "original_archive": str(archive) if archive is not None else None,
                "multipart_filename": raw_name,
            },
        )


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
