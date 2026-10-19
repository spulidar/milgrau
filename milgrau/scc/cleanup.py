"""Conservative cleanup of disposable SCC packets after verified downloads.

No remote delete, no scientific file removal, no journal or archive deletion.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

from milgrau.level0.scc_staging import manifest_raw_name, pending_scc_package
from milgrau.provenance import file_sha256
from milgrau.scc.client import SCCRemoteError
from milgrau.scc.packages import validate_upload_package
from milgrau.scc.results import validate_results, result_location
from milgrau.scc.state import SubmissionStore, operation_lock, state_root


def verify_cleanup(config: Mapping[str, Any], identity: str) -> dict[str, Any]:
    """Offline proof that a confirmed upload and matching downloads are durable."""
    root = state_root(config)
    with SubmissionStore(root) as store:
        row = store.get(identity)
        if row is None:
            raise SCCRemoteError(
                "No MILGRAU submission journal for this SCC ID; "
                "an existing remote measurement is not evidence of our upload."
            )
        if row["state"] not in {"downloaded", "cleaned"}:
            raise SCCRemoteError(
                f"SCC {identity} state={row['state']} is not eligible for cleanup; "
                "must first confirm upload, completion and verified results download."
            )
        if not row["results_path"] or not row["results_manifest_sha256"]:
            raise SCCRemoteError("SCC results have no durable verified journal record.")
        path = result_location(
            config, row["scientific_session_id"], identity,
            row["remote_creation_date"],
        )
        if path.resolve() != Path(row["results_path"]).resolve():
            raise SCCRemoteError("SCC results path does not match its canonical session.")
        manifest = validate_results(path)
        if file_sha256(path / "results.json") != row["results_manifest_sha256"]:
            raise SCCRemoteError("SCC results manifest changed after verification.")
        if not manifest.get("uploaded_by_milgrau"):
            raise SCCRemoteError("SCC results were not from a journal-confirmed upload.")
        if manifest["remote_creation_date"] != row["remote_creation_date"]:
            raise SCCRemoteError("Remote SCC generation is not the one accepted by MILGRAU.")
        if manifest["scc_configuration_id"] != row["configuration_id"]:
            raise SCCRemoteError("SCC results configuration differs from submission.")
        staged = pending_scc_package(config, identity, session_id=row["scientific_session_id"])
        if row["state"] == "cleaned":
            if staged.exists():
                raise SCCRemoteError(
                    "SCC journal says cache was cleaned, but pending directory reappeared. "
                    "Manual reconciliation is required."
                )
            return {"identity": identity, "staged": None, "already_cleaned": True}
        if not staged.is_dir() or staged.is_symlink():
            raise SCCRemoteError(
                "Pending SCC staging missing/unsafe. Refusing to mark cleaned."
            )
        pending = validate_upload_package(staged)
        if (
            pending["scientific_session_id"] != row["scientific_session_id"]
            or pending["scc_configuration_id"] != row["configuration_id"]
            or pending["files"][manifest_raw_name(pending)]["sha256"] != row["raw_sha256"]
        ):
            raise SCCRemoteError("SCC pending packet changed from the confirmed upload.")
        lr_sha = next(
            (item["sha256"] for name, item in pending["files"].items() if name.startswith("lr_")),
            None,
        )
        if lr_sha != row["lr_sha256"]:
            raise SCCRemoteError("SCC pending LR differs from confirmed upload.")
        sounding_sha = next(
            (item["sha256"] for name, item in pending["files"].items()
             if name.startswith("rs_")),
            None,
        )
        if sounding_sha != row["sounding_sha256"]:
            raise SCCRemoteError("SCC pending sounding differs from confirmed upload.")
        return {"identity": identity, "staged": staged, "already_cleaned": False}


def cleanup_pending(
    config: Mapping[str, Any], identity: str, *, dry_run: bool = True,
) -> dict[str, Any]:
    """Delete only session/scc/scc-inputs/ID after durable verifications.

    A crash between rmtree and the final SQLite transition leaves 'downloaded'
    state, which needs manual inspection; no automatic blind replay or deletion.
    """
    root = state_root(config)
    with operation_lock(root, identity):
        result = verify_cleanup(config, identity)
        if result["already_cleaned"] or dry_run:
            return result
        staged = Path(result["staged"])
        with SubmissionStore(root) as store:
            row = store.get(identity)
        if row is None:
            raise SCCRemoteError("No SCC journal record for input cleanup.")
        expected = pending_scc_package(config, identity, session_id=row["scientific_session_id"])
        if staged.resolve() != expected.resolve() or staged.is_symlink():
            raise SCCRemoteError("Unsafe SCC staging path; cleanup aborted.")
        shutil.rmtree(staged)
        with SubmissionStore(root) as store:
            store.transition(
                identity, "cleaned", action="verified_pending_cache_cleanup",
                detail={"deleted_only": str(staged)},
            )
        return {"identity": identity, "staged": None, "already_cleaned": False}
