"""Remote status and verified SCC downloads can run over all locally known IDs."""

from __future__ import annotations

from pathlib import Path

import pytest

from milgrau.cli import scc as cli
from milgrau.level0.scc_staging import pending_scc_package
from milgrau.scc import results
from milgrau.scc.state import SubmissionStore, state_root

from test_scc_results import (
    ID, SESSION, _config, _mock_stream, _record, _stage,
)

ID2 = "20241220spu0414"
ID3 = "20250114spu0419"


class ManyClient:
    def __init__(self, records: dict):
        self.records = records
        self.reads: list[str] = []

    def measurement(self, identity: str):
        self.reads.append(identity)
        return self.records.get(identity)


def _journal(config: dict, identity: str, *, state: str = "pending") -> None:
    manifest = {
        "measurement_id": identity,
        "scientific_session_id": SESSION,
        "scc_configuration_id": 1047,
        "scc_product_ids": [3683],
        "source_segments": ["seg00"],
        "measurement_start_utc": "2024-09-13T15:49:32+00:00",
        "measurement_stop_utc": "2024-09-13T17:50:42+00:00",
        "files": {f"{identity}.nc": {"sha256": "a" * 64}},
    }
    with SubmissionStore(state_root(config)) as journal:
        journal.begin(manifest)
        if state != "pending":
            journal.transition(identity, state, action="prior_remote_operation")


def test_bulk_remote_status_unions_pending_and_journal_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    cfg = _config(tmp_path)
    _stage(cfg)
    _journal(cfg, ID2, state="uncertain")
    client = ManyClient({ID: _record()})
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "_client", lambda _: client)
    assert cli.main(["status", "--remote"]) == 0
    output = capsys.readouterr().out
    assert f"remote {ID}: present" in output
    assert f"remote {ID2}: NOT FOUND" in output
    assert "total=2; present=1; absent=1; errors=0" in output
    assert client.reads == [ID, ID2]
    # No implicit unlock/retry of an uncertain record.
    with SubmissionStore(state_root(cfg)) as journal:
        assert journal.get(ID2)["state"] == "uncertain"


def test_bulk_remote_status_continues_after_one_get_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    cfg = _config(tmp_path)
    _journal(cfg, ID2)
    _journal(cfg, ID3)
    class ErrorClient:
        def __init__(self):
            self.reads = []
        def measurement(self, identity):
            self.reads.append(identity)
            if identity == ID2:
                raise RuntimeError("temporary API error")
            return None
    client = ErrorClient()
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "_client", lambda _: client)
    assert cli.main(["status", "--remote"]) == 1
    out = capsys.readouterr()
    assert f"remote {ID2}: ERROR" in out.err
    assert f"remote {ID3}: NOT FOUND" in out.out
    assert client.reads == [ID2, ID3]


def test_bulk_download_captures_complete_external_packet_and_retains_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    cfg = _config(tmp_path)  # all session inputs are retained
    _stage(cfg)  # No local submission journal: externally uploaded SCC ID.
    _journal(cfg, ID2)
    client = ManyClient({ID: _record()})
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "_client", lambda _: client)
    _mock_stream(monkeypatch)
    assert cli.main(["download"]) == 0
    output = capsys.readouterr().out
    assert f"{ID}: VERIFIED" in output
    assert f"{ID2}: SKIPPED" in output
    assert "verified=1; waiting=0; absent=1; errors=0" in output
    assert pending_scc_package(cfg, ID).is_dir()
    saved = results.result_location(cfg, SESSION, ID, _record()["creation_date"])
    assert results.validate_results(saved)["uploaded_by_milgrau"] is False
    # One status GET and another inside validated download for the complete ID.
    assert client.reads.count(ID) == 2 and client.reads.count(ID2) == 1


def test_bulk_download_defers_processing_and_isolates_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    cfg = _config(tmp_path)
    _journal(cfg, ID2)
    _journal(cfg, ID3)
    waiting = _record(finished=False)
    waiting["id"] = ID2
    success = _record()
    success["id"] = ID3
    client = ManyClient({ID2: waiting, ID3: success})
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "_client", lambda _: client)
    def failed_download(config, identity, client, *, source_session):
        raise RuntimeError("one archive failed")
    monkeypatch.setattr(cli, "_download_one", failed_download)
    assert cli.main(["download"]) == 1
    out = capsys.readouterr()
    assert f"{ID2}: WAITING" in out.out
    assert f"{ID3}: ERROR" in out.err
    assert "waiting=1; absent=0; errors=1" in out.out


def test_bulk_download_rejects_unscoped_session_and_allows_single_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    cfg = _config(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    assert cli.main(["download", "--session", SESSION]) == 1
    assert "--session requires --id" in capsys.readouterr().err
    assert cli._parser().parse_args(["download", "--id", ID]).identity == ID


def test_remote_status_reports_ambiguous_post_even_after_api_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    cfg = _config(tmp_path)
    _journal(cfg, ID2, state="uncertain")
    client = ManyClient({})
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "_client", lambda _: client)

    assert cli.main(["status", "--id", ID2, "--remote"]) == 0
    output = capsys.readouterr().out
    assert f"remote {ID2}: NOT FOUND" in output
    assert f"ATTENTION {ID2}:" in output
    assert "absence alone does not prove a POST was rejected" in output
    with SubmissionStore(state_root(cfg)) as journal:
        assert journal.get(ID2)["state"] == "uncertain"
