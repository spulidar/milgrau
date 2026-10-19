"""SCC replacement: verified backup by default or explicit --confirm waiver."""

from __future__ import annotations

from pathlib import Path

import pytest

from milgrau.scc import workflow
from milgrau.scc.client import SCCAlreadyExists, SCCRemoteError
from milgrau.scc.state import SubmissionStore, state_root


ID = "20240913spu1549"


def _config(tmp_path: Path) -> dict:
    return {
        "scc": {
            "enabled": True,
            "state_dir": str(tmp_path / ".state/scc"),
        },
        "directories": {"processed_data": str(tmp_path / "02-processed_data")},
        "_station_catalog": {"station": {"id": "spu"}},
    }


def _manifest():
    return {
        "measurement_id": ID,
        "scientific_session_id": "spu_20240913-1549Z_20240913-1750Z",
        "scc_configuration_id": 1047,
        "scc_product_ids": [3683, 3684, 3685],
        "scc_config_sha256": None,
        "source_segments": ["seg00"],
        "measurement_start_utc": "2024-09-13T15:49:32+00:00",
        "measurement_stop_utc": "2024-09-13T17:50:42+00:00",
        "files": {"raw.nc": {"sha256": "a" * 64}},
    }


def _remote():
    return {
        "id": ID, "creation_date": "2026-10-17T14:29:53",
        "start": "2024-09-13T15:49:32",
        "stop": "2024-09-13T17:50:42",
        "last_processing_end": "2026-10-17T14:34:31",
        "is_queued": False, "is_being_processed": False,
        "system": {"id": 1047}, "upload": 127,
        "elpp": 127, "elda": 127,
    }


class FakeClient:
    def __init__(self, *, exists=True, fail_delete=False, fail_upload=False):
        self.record = _remote() if exists else None
        self.delete_calls = 0
        self.upload_calls = 0
        self.fail_delete = fail_delete
        self.fail_upload = fail_upload
        self.expected_fingerprint = None

    def measurement(self, identity):
        assert identity == ID
        return self.record

    def delete_measurement(self, identity, *, expected_fingerprint):
        from milgrau.scc.backup import remote_fingerprint

        assert identity == ID
        assert remote_fingerprint(self.record) == expected_fingerprint
        self.delete_calls += 1
        if self.fail_delete:
            raise SCCRemoteError("deletion response lost")
        self.record = None

    def upload_package(self, path):
        assert self.record is None
        self.upload_calls += 1
        if self.fail_upload:
            raise SCCRemoteError("upload response lost")
        self.record = _remote()
        return {"measurement_id": ID, "remote_page": "https://scc.example.test/detail/"}


def _patch(monkeypatch: pytest.MonkeyPatch, *, backup_failure=False, changed=False):
    from milgrau.scc.backup import remote_fingerprint

    monkeypatch.setattr(workflow, "validate_upload_package", lambda _: _manifest())

    def archive(_client, record, root):
        if backup_failure:
            raise SCCRemoteError("remote raw archive unavailable")
        target = root / "archives" / ID / "20261017"
        target.mkdir(parents=True)
        return target, "b" * 64

    def verify(_directory, _sha):
        if changed:
            return {"remote_fingerprint": "wrong"}
        return {"remote_fingerprint": remote_fingerprint(_remote())}

    monkeypatch.setattr(workflow, "archive_remote_measurement", archive)
    monkeypatch.setattr(workflow, "verify_archive", verify)
    monkeypatch.setattr(
        workflow, "compare_staged_to_archive",
        lambda *_: {
            "identical": False, "period_differs": True,
            "raw_identical": False, "lr_identical": False,
        },
    )


def test_confirmed_replacement_skips_unavailable_backup_but_records_waiver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, backup_failure=True)
    client = FakeClient()
    config = _config(tmp_path)
    result = workflow.submit(
        config, tmp_path, client, replace_existing=True, skip_backup=True,
    )
    assert result["measurement_id"] == ID
    assert client.delete_calls == client.upload_calls == 1
    with SubmissionStore(state_root(config)) as journal:
        assert journal.get(ID)["state"] == "accepted"
        assert journal.get(ID)["archive_path"] is None
        events = journal.events(ID)
        assert "operator_confirmed_no_backup" in [e["action"] for e in events]
        assert "delete_requested_without_backup" in [e["action"] for e in events]


def test_confirmed_unbacked_delete_failure_still_blocks_automatic_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, backup_failure=True)
    config = _config(tmp_path)
    client = FakeClient(fail_delete=True)
    with pytest.raises(SCCRemoteError, match="deletion response lost"):
        workflow.submit(
            config, tmp_path, client, replace_existing=True, skip_backup=True,
        )
    with SubmissionStore(state_root(config)) as journal:
        assert journal.get(ID)["state"] == "uncertain"
    with pytest.raises(RuntimeError, match="manual remote reconciliation"):
        workflow.submit(
            config, tmp_path, client, replace_existing=True, skip_backup=True,
        )
    assert client.delete_calls == 1 and client.upload_calls == 0


def test_new_measurement_is_journalled_and_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)
    client = FakeClient(exists=False)
    result = workflow.submit(_config(tmp_path), tmp_path, client)
    assert result["measurement_id"] == ID
    with SubmissionStore(state_root(_config(tmp_path))) as journal:
        assert journal.get(ID)["state"] == "accepted"
        assert journal.get(ID)["remote_creation_date"] == "2026-10-17T14:29:53"
        assert journal.events(ID)[-1]["action"] == "upload_verified"
    assert client.delete_calls == 0 and client.upload_calls == 1


def test_existing_without_replace_never_deletes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)
    client = FakeClient()
    with pytest.raises(SCCAlreadyExists):
        workflow.submit(_config(tmp_path), tmp_path, client)
    with SubmissionStore(state_root(_config(tmp_path))) as journal:
        assert journal.get(ID)["state"] == "remote_exists"
    assert client.delete_calls == client.upload_calls == 0


def test_identical_old_remote_measurement_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)
    monkeypatch.setattr(
        workflow, "compare_staged_to_archive",
        lambda *_: {
            "identical": True, "period_differs": False,
            "raw_identical": True, "lr_identical": True,
        },
    )
    client = FakeClient()
    client._url = lambda path: "https://scc.example.test/" + path
    result = workflow.submit(
        _config(tmp_path), tmp_path, client, replace_existing=True,
    )
    assert result["status"] == "already_identical"
    assert client.delete_calls == client.upload_calls == 0
    with SubmissionStore(state_root(_config(tmp_path))) as store:
        assert store.get(ID)["state"] == "remote_exists"


def test_explicit_replacement_preserves_id_and_verified_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)
    client = FakeClient()
    result = workflow.submit(
        _config(tmp_path), tmp_path, client, replace_existing=True,
    )
    assert result["measurement_id"] == ID
    assert client.delete_calls == client.upload_calls == 1
    with SubmissionStore(state_root(_config(tmp_path))) as journal:
        row = journal.get(ID)
        assert row["state"] == "accepted"
        assert row["archive_manifest_sha256"] == "b" * 64
        states = [entry["state"] for entry in journal.events(ID)]
        assert states[-5:] == [
            "backed_up", "deleting", "remote_deleted", "uploading", "accepted",
        ]


@pytest.mark.parametrize("failure", ["archive", "different_record"])
def test_backup_unavailable_or_changed_fingerprint_blocks_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    _patch(monkeypatch, backup_failure=failure == "archive",
           changed=failure == "different_record")
    client = FakeClient()
    with pytest.raises(SCCRemoteError):
        workflow.submit(_config(tmp_path), tmp_path, client, replace_existing=True)
    assert client.delete_calls == client.upload_calls == 0
    with SubmissionStore(state_root(_config(tmp_path))) as journal:
        # Failed remote backup never reassigns this ID's journal revision.
        assert journal.get(ID) is None


def test_previous_accepted_revision_survives_failed_remote_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, backup_failure=True)
    config = _config(tmp_path)
    with SubmissionStore(state_root(config)) as store:
        store.begin(_manifest())
        store.transition(ID, "accepted", action="previous_accepted")
        before = store.get(ID)
        events = store.events(ID)
    client = FakeClient()
    with pytest.raises(SCCRemoteError, match="archive unavailable"):
        workflow.submit(config, tmp_path, client, replace_existing=True)
    with SubmissionStore(state_root(config)) as store:
        assert store.get(ID) == before
        assert store.events(ID) == events
    assert client.delete_calls == client.upload_calls == 0


def test_explicit_replacement_with_same_local_hash_still_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)
    config = _config(tmp_path)
    with SubmissionStore(state_root(config)) as store:
        store.begin(_manifest())
        store.transition(ID, "accepted", action="previous_accepted")
    client = FakeClient()
    workflow.submit(config, tmp_path, client, replace_existing=True)
    assert client.delete_calls == client.upload_calls == 1
    with SubmissionStore(state_root(config)) as store:
        assert store.get(ID)["state"] == "accepted"
        assert "new_explicit_revision" in {e["action"] for e in store.events(ID)}


def test_remote_identical_preserves_previous_accepted_journal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)
    monkeypatch.setattr(
        workflow, "compare_staged_to_archive",
        lambda *_: {"identical": True, "period_differs": False,
                    "raw_identical": True, "lr_identical": True},
    )
    config = _config(tmp_path)
    with SubmissionStore(state_root(config)) as store:
        store.begin(_manifest())
        store.transition(ID, "accepted", action="previous_accepted")
    client = FakeClient()
    client._url = lambda path: "https://scc.example.test/" + path
    result = workflow.submit(config, tmp_path, client, replace_existing=True)
    assert result["status"] == "already_identical"
    assert client.delete_calls == client.upload_calls == 0
    with SubmissionStore(state_root(config)) as store:
        assert store.get(ID)["state"] == "accepted"
        assert store.events(ID)[-1]["action"] == "identical_noop"


def test_local_staged_bytes_change_during_backup_blocks_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from copy import deepcopy

    _patch(monkeypatch)
    counter = {"calls": 0}

    def varying_manifest(_):
        counter["calls"] += 1
        result = deepcopy(_manifest())
        if counter["calls"] > 1:
            result["files"]["raw.nc"]["sha256"] = "f" * 64
        return result

    monkeypatch.setattr(workflow, "validate_upload_package", varying_manifest)
    client = FakeClient()
    with pytest.raises(SCCRemoteError, match="changed during remote backup"):
        workflow.submit(_config(tmp_path), tmp_path, client, replace_existing=True)
    assert client.delete_calls == client.upload_calls == 0
    with SubmissionStore(state_root(_config(tmp_path))) as store:
        assert store.get(ID)["state"] == "failed"


def test_delete_uncertain_prevents_any_upload_and_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)
    client = FakeClient(fail_delete=True)
    with pytest.raises(SCCRemoteError, match="deletion response"):
        workflow.submit(_config(tmp_path), tmp_path, client, replace_existing=True)
    assert client.delete_calls == 1
    assert client.upload_calls == 0
    with SubmissionStore(state_root(_config(tmp_path))) as journal:
        row = journal.get(ID)
        assert row["state"] == "uncertain"
        assert row["archive_path"]
    with pytest.raises(RuntimeError, match="manual remote reconciliation"):
        workflow.submit(_config(tmp_path), tmp_path, client, replace_existing=True)
    assert client.delete_calls == 1


def test_upload_ambiguous_keeps_original_archive_and_blocks_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)
    client = FakeClient(fail_upload=True)
    with pytest.raises(SCCRemoteError, match="upload response"):
        workflow.submit(_config(tmp_path), tmp_path, client, replace_existing=True)
    with SubmissionStore(state_root(_config(tmp_path))) as journal:
        assert journal.get(ID)["state"] == "uncertain"
        assert journal.get(ID)["archive_path"]
    assert client.delete_calls == 1 and client.upload_calls == 1
    with pytest.raises(RuntimeError, match="manual remote reconciliation"):
        workflow.submit(_config(tmp_path), tmp_path, client, replace_existing=True)


def _interrupted_delete_journal(tmp_path: Path, *, action="delete_result_uncertain"):
    config = _config(tmp_path)
    root = state_root(config)
    backup = root / "archives" / ID / "20261017T2130"
    backup.mkdir(parents=True, exist_ok=True)
    with SubmissionStore(root) as store:
        store.begin(_manifest())
        store.transition(
            ID, "backed_up", action="backup_verified",
            archive_path=str(backup), archive_manifest_sha256="f" * 64,
        )
        store.transition(ID, "deleting", action="delete_requested")
        store.transition(ID, "uncertain", action=action)
    from milgrau.level0.scc_staging import pending_scc_package

    staged = pending_scc_package(config, ID, session_id="spu_20240913-1549Z_20240913-1750Z")
    staged.mkdir(parents=True)
    return config, backup, staged


def test_recover_after_delete_unknown_but_remote_absent_sends_only_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)
    config, archive, staged = _interrupted_delete_journal(tmp_path)
    from milgrau.scc import workflow as module

    monkeypatch.setattr(
        module, "verify_archive",
        lambda path, digest: {
            "measurement_id": ID,
            "remote_snapshot": {"configuration_id": 1047},
        },
    )
    client = FakeClient(exists=False)
    result = workflow.resume_after_uncertain_delete(config, staged, client)
    assert result["measurement_id"] == ID
    assert client.delete_calls == 0
    assert client.upload_calls == 1
    with SubmissionStore(state_root(config)) as store:
        row = store.get(ID)
        assert row["state"] == "accepted"
        assert row["archive_path"] == str(archive)
        assert row["archive_manifest_sha256"] == "f" * 64
        actions = [entry["action"] for entry in store.events(ID)]
        assert actions[-3:] == [
            "operator_confirmed_remote_absence_after_uncertain_delete",
            "recovery_upload_requested", "recovery_upload_verified",
        ]


def test_recovery_does_not_upload_if_id_reappeared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)
    config, _, staged = _interrupted_delete_journal(tmp_path)
    monkeypatch.setattr(
        workflow, "verify_archive",
        lambda *_: {
            "measurement_id": ID,
            "remote_snapshot": {"configuration_id": 1047},
        },
    )
    client = FakeClient(exists=True)
    with pytest.raises(SCCRemoteError, match="present again"):
        workflow.resume_after_uncertain_delete(config, staged, client)
    assert client.upload_calls == client.delete_calls == 0
    with SubmissionStore(state_root(config)) as store:
        assert store.get(ID)["state"] == "uncertain"


def test_recovery_rejects_other_uncertain_causes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)
    config, _, staged = _interrupted_delete_journal(
        tmp_path, action="upload_result_uncertain",
    )
    client = FakeClient(exists=False)
    with pytest.raises(SCCRemoteError, match="limited to"):
        workflow.resume_after_uncertain_delete(config, staged, client)
    assert client.upload_calls == client.delete_calls == 0


def test_recovery_fails_closed_if_archive_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)
    config, _, staged = _interrupted_delete_journal(tmp_path)
    monkeypatch.setattr(
        workflow, "verify_archive",
        lambda *_: (_ for _ in ()).throw(SCCRemoteError("backup corrupted")),
    )
    client = FakeClient(exists=False)
    with pytest.raises(SCCRemoteError, match="backup corrupted"):
        workflow.resume_after_uncertain_delete(config, staged, client)
    assert client.upload_calls == client.delete_calls == 0
    with SubmissionStore(state_root(config)) as store:
        assert store.get(ID)["state"] == "uncertain"


def test_recovery_upload_unknown_fails_closed_and_is_not_repeatable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)
    config, _, staged = _interrupted_delete_journal(tmp_path)
    monkeypatch.setattr(
        workflow, "verify_archive",
        lambda *_: {
            "measurement_id": ID,
            "remote_snapshot": {"configuration_id": 1047},
        },
    )
    client = FakeClient(exists=False, fail_upload=True)
    with pytest.raises(SCCRemoteError, match="upload response"):
        workflow.resume_after_uncertain_delete(config, staged, client)
    assert client.delete_calls == 0 and client.upload_calls == 1
    with SubmissionStore(state_root(config)) as store:
        assert store.get(ID)["state"] == "uncertain"
        assert store.events(ID)[-1]["action"] == "recovery_upload_uncertain"
    with pytest.raises(SCCRemoteError, match="limited to"):
        workflow.resume_after_uncertain_delete(config, staged, client)
    assert client.upload_calls == 1


def _prior_unknown_recovery_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from copy import deepcopy

    config, _, staged = _interrupted_delete_journal(tmp_path)
    with SubmissionStore(state_root(config)) as store:
        store.transition(
            ID, "remote_deleted",
            action="operator_confirmed_remote_absence_after_uncertain_delete",
        )
        store.transition(ID, "uploading", action="recovery_upload_requested")
        store.transition(ID, "uncertain", action="recovery_upload_uncertain")
    manifest = deepcopy(_manifest())
    manifest["files"][f"{ID}.nc"] = manifest["files"].pop("raw.nc")
    monkeypatch.setattr(workflow, "validate_upload_package", lambda _: manifest)
    monkeypatch.setattr(
        workflow, "verify_archive",
        lambda *_: {
            "measurement_id": ID,
            "remote_snapshot": {"configuration_id": 1047},
        },
    )
    return config, staged


def _prior_lr_filename_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from copy import deepcopy

    config, staged = _prior_unknown_recovery_upload(tmp_path, monkeypatch)
    new_manifest = deepcopy(_manifest())
    new_manifest["files"] = {
        f"{ID}.nc": {"sha256": "c" * 64},
        f"lr_{ID[:13]}.nc": {"sha256": "b" * 64},
    }
    monkeypatch.setattr(workflow, "validate_upload_package", lambda _: new_manifest)
    with SubmissionStore(state_root(config)) as store:
        store.transition(ID, "uncertain", action="manual_corrected_upload_uncertain")
        with store._conn:
            store._conn.execute(
                "UPDATE submissions SET raw_sha256=?, lr_sha256=? WHERE measurement_id=?",
                ("c" * 64, "b" * 64, ID),
            )
        store.transition(
            ID, "uncertain", action="local_lr_filename_repair",
            detail={
                "previous_action": "manual_corrected_upload_uncertain",
                "new_lr_filename": f"lr_{ID[:13]}.nc",
                "new_raw_sha256": "c" * 64,
                "unchanged_lr_sha256": "b" * 64,
            },
        )
    return config, staged


def test_lr_fixed_recovery_submits_once_and_never_deletes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, staged = _prior_lr_filename_repair(tmp_path, monkeypatch)
    client = FakeClient(exists=False)
    result = workflow.retry_rejected_recovery_upload(
        config, staged, client, after_lr_fix=True,
    )
    assert result["measurement_id"] == ID
    assert client.upload_calls == 1 and client.delete_calls == 0
    with SubmissionStore(state_root(config)) as store:
        assert store.get(ID)["state"] == "accepted"
        assert store.events(ID)[-1]["action"] == "manual_lr_fixed_upload_verified"


def test_lr_fixed_recovery_rejects_present_remote_and_never_posts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, staged = _prior_lr_filename_repair(tmp_path, monkeypatch)
    client = FakeClient(exists=True)
    with pytest.raises(SCCRemoteError, match="reappeared"):
        workflow.retry_rejected_recovery_upload(
            config, staged, client, after_lr_fix=True,
        )
    assert client.upload_calls == client.delete_calls == 0


def test_lr_fixed_recovery_uncertain_is_not_repeatable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, staged = _prior_lr_filename_repair(tmp_path, monkeypatch)
    client = FakeClient(exists=False, fail_upload=True)
    with pytest.raises(SCCRemoteError, match="upload response lost"):
        workflow.retry_rejected_recovery_upload(
            config, staged, client, after_lr_fix=True,
        )
    with SubmissionStore(state_root(config)) as store:
        assert store.get(ID)["state"] == "uncertain"
        assert store.events(ID)[-1]["action"] == "manual_lr_fixed_upload_uncertain"
    with pytest.raises(SCCRemoteError, match="latest journal event"):
        workflow.retry_rejected_recovery_upload(
            config, staged, client, after_lr_fix=True,
        )
    assert client.upload_calls == 1 and client.delete_calls == 0


def test_second_upload_only_after_uncertain_recovery_and_scc_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, staged = _prior_unknown_recovery_upload(tmp_path, monkeypatch)
    client = FakeClient(exists=False)
    result = workflow.retry_rejected_recovery_upload(config, staged, client)
    assert result["measurement_id"] == ID
    assert client.delete_calls == 0 and client.upload_calls == 1
    with SubmissionStore(state_root(config)) as store:
        assert store.get(ID)["state"] == "accepted"
        assert store.events(ID)[-1]["action"] == "manual_corrected_upload_verified"


def test_second_upload_blocks_if_scc_id_reappeared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, staged = _prior_unknown_recovery_upload(tmp_path, monkeypatch)
    client = FakeClient(exists=True)
    with pytest.raises(SCCRemoteError, match="reappeared"):
        workflow.retry_rejected_recovery_upload(config, staged, client)
    assert client.delete_calls == client.upload_calls == 0


def test_second_upload_failure_refuses_any_further_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, staged = _prior_unknown_recovery_upload(tmp_path, monkeypatch)
    client = FakeClient(exists=False, fail_upload=True)
    with pytest.raises(SCCRemoteError, match="upload response lost"):
        workflow.retry_rejected_recovery_upload(config, staged, client)
    assert client.delete_calls == 0 and client.upload_calls == 1
    with SubmissionStore(state_root(config)) as store:
        assert store.get(ID)["state"] == "uncertain"
        assert store.events(ID)[-1]["action"] == "manual_corrected_upload_uncertain"
    with pytest.raises(SCCRemoteError, match="latest journal event"):
        workflow.retry_rejected_recovery_upload(config, staged, client)
    assert client.upload_calls == 1


def test_second_upload_requires_original_verified_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, staged = _prior_unknown_recovery_upload(tmp_path, monkeypatch)
    monkeypatch.setattr(
        workflow, "verify_archive",
        lambda *_: (_ for _ in ()).throw(SCCRemoteError("original corrupt")),
    )
    client = FakeClient(exists=False)
    with pytest.raises(SCCRemoteError, match="original corrupt"):
        workflow.retry_rejected_recovery_upload(config, staged, client)
    assert client.upload_calls == client.delete_calls == 0


def test_replacement_rejects_missing_remote_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)
    client = FakeClient(exists=False)
    with pytest.raises(SCCRemoteError, match="not present"):
        workflow.submit(_config(tmp_path), tmp_path, client, replace_existing=True)
    assert client.upload_calls == client.delete_calls == 0
