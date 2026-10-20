"""Phase 12.5 durable state, collision locks and upload uncertainty."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from milgrau.scc.state import (
    SubmissionStore, operation_lock, state_root,
)


IDENTITY = "20240913spu1549"


def _config(tmp_path: Path):
    return {
        "scc": {
            "enabled": True,
            "state_dir": str(tmp_path / ".state/scc"),
        }
    }


def _manifest(raw_hash: str = "a" * 64):
    return {
        "measurement_id": IDENTITY,
        "scientific_session_id": "spu_20240913-1549Z_20240913-1750Z",
        "scc_configuration_id": 1047,
        "scc_product_ids": [3683, 3684, 3685],
        "source_segments": ["seg00", "seg02"],
        "scc_config_sha256": "c" * 64,
        "measurement_start_utc": "2024-09-13T15:49:32+00:00",
        "measurement_stop_utc": "2024-09-13T17:50:42+00:00",
        "files": {
            "raw.nc": {"sha256": raw_hash},
            f"lr_{IDENTITY}.nc": {"sha256": "b" * 64},
        },
    }


def test_persistent_journal_survives_deleted_cache_and_records_provenance(
    tmp_path: Path,
) -> None:
    root = state_root(_config(tmp_path))
    with SubmissionStore(root) as journal:
        row = journal.begin(_manifest())
        assert row["state"] == "pending"
        assert json.loads(row["product_ids_json"]) == [3683, 3684, 3685]
        assert json.loads(row["source_segments_json"]) == ["seg00", "seg02"]
        journal.transition(
            IDENTITY, "backed_up", action="archive_verified",
            archive_path=str(root / "archives" / IDENTITY),
            archive_manifest_sha256="d" * 64,
        )
    cache = tmp_path / ".cache"
    cache.mkdir()
    (cache / "fake.nc").write_bytes(b"fake")
    (cache / "fake.nc").unlink()
    cache.rmdir()
    with SubmissionStore(root) as journal:
        record = journal.get(IDENTITY)
        assert record["state"] == "backed_up"
        assert record["archive_manifest_sha256"] == "d" * 64
        assert [event["action"] for event in journal.events(IDENTITY)] == [
            "initial_stage", "archive_verified",
        ]


@pytest.mark.parametrize("state", ["deleting", "remote_deleted", "uploading", "uncertain"])
def test_ambiguous_or_destructive_state_blocks_repeat(
    tmp_path: Path, state: str,
) -> None:
    with SubmissionStore(state_root(_config(tmp_path))) as journal:
        journal.begin(_manifest())
        journal.transition(IDENTITY, state, action="simulated_interruption")
        with pytest.raises(RuntimeError, match="manual remote reconciliation"):
            journal.begin(_manifest(), allow_new_revision=True)
    with SubmissionStore(state_root(_config(tmp_path))) as journal:
        assert journal.get(IDENTITY)["state"] == state


def test_no_implicit_revision_even_after_success(tmp_path: Path) -> None:
    with SubmissionStore(state_root(_config(tmp_path))) as journal:
        journal.begin(_manifest())
        journal.transition(IDENTITY, "accepted", action="upload_confirmed")
        with pytest.raises(RuntimeError, match="already submitted"):
            journal.begin(_manifest())
        with pytest.raises(RuntimeError, match="no duplicate upload"):
            journal.begin(_manifest(), allow_new_revision=True)
        with pytest.raises(RuntimeError, match="explicit replacement"):
            journal.begin(_manifest("c" * 64))
        row = journal.begin(_manifest("c" * 64), allow_new_revision=True)
        assert row["state"] == "pending"
        assert row["raw_sha256"] == "c" * 64
        assert len(journal.events(IDENTITY)) == 3


def test_cross_process_exclusive_lock_is_safe(tmp_path: Path) -> None:
    root = state_root(_config(tmp_path))
    with operation_lock(root, IDENTITY):
        with pytest.raises(RuntimeError, match="active/stale lock"):
            with operation_lock(root, IDENTITY):
                pass
    with operation_lock(root, IDENTITY):
        pass
    assert not list((root / "locks").glob("*.lock"))


def test_state_cannot_be_stored_under_scientific_session_products(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["directories"] = {"processed_data": str(tmp_path / "02-processed_data")}
    config["scc"]["state_dir"] = str(tmp_path / "02-processed_data/spu/journal")
    with pytest.raises(ValueError, match="outside processed session"):
        state_root(config)


def test_manifest_identity_matches_durable_journal_and_rejects_changed_inputs(
    tmp_path: Path,
) -> None:
    from copy import deepcopy

    with SubmissionStore(state_root(_config(tmp_path))) as store:
        store.begin(_manifest())
        assert store.matches_manifest(IDENTITY, _manifest())
        changed = deepcopy(_manifest())
        changed["scc_product_ids"] = [3683]
        assert not store.matches_manifest(IDENTITY, changed)
        changed = deepcopy(_manifest())
        changed["files"]["raw.nc"]["sha256"] = "f" * 64
        assert not store.matches_manifest(IDENTITY, changed)
        changed = deepcopy(_manifest())
        changed["measurement_start_utc"] = "2024-09-13T15:49:33+00:00"
        assert not store.matches_manifest(IDENTITY, changed)


def test_current_revision_events_never_reuses_older_replacement_waiver(
    tmp_path: Path,
) -> None:
    with SubmissionStore(state_root(_config(tmp_path))) as store:
        store.begin(_manifest())
        store.transition(
            IDENTITY, "preflight", action="operator_confirmed_no_backup",
            detail={"irreversible": True, "remote_fingerprint": "a" * 64},
        )
        store.transition(IDENTITY, "accepted", action="old_submission_accepted")
        store.begin(_manifest(), allow_new_revision=True, allow_same_revision=True)
        store.transition(IDENTITY, "uncertain", action="delete_result_uncertain")
        actions = [event["action"] for event in store.current_revision_events(IDENTITY)]
        assert actions == ["new_explicit_revision", "delete_result_uncertain"]
        assert len(store.events(IDENTITY)) > len(actions)


def test_legacy_cleaned_journal_remains_readable_after_cleanup_removal(
    tmp_path: Path,
) -> None:
    root = state_root(_config(tmp_path))
    with SubmissionStore(root) as store:
        store.begin(_manifest())
        store.transition(IDENTITY, "cleaned", action="historical_cache_cleanup")
    with SubmissionStore(root) as store:
        assert store.get(IDENTITY)["state"] == "cleaned"
        assert store.events(IDENTITY)[-1]["action"] == "historical_cache_cleanup"
        with pytest.raises(RuntimeError, match="already submitted"):
            store.begin(_manifest())
