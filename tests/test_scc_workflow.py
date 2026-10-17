"""Phase 12.5 same-ID replacement must never delete without verified archive."""

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
            "staging_dir": str(tmp_path / ".cache/scc"),
        },
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
        assert journal.get(ID)["state"] == "failed"


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


def test_replacement_rejects_missing_remote_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)
    client = FakeClient(exists=False)
    with pytest.raises(SCCRemoteError, match="not present"):
        workflow.submit(_config(tmp_path), tmp_path, client, replace_existing=True)
    assert client.upload_calls == client.delete_calls == 0
