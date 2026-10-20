"""Sequential SCC batch upload: preflight-all, skip, replace and fail-stop."""

from __future__ import annotations

from pathlib import Path

import pytest

from milgrau.cli import scc as cli
from milgrau.scc import batch
from milgrau.scc.state import SubmissionStore, state_root

ID1 = "20240913spu1549"
ID2 = "20241220spu0414"
SESSION = "spu_20240913-1549Z_20240913-1750Z"


def _config(tmp_path: Path) -> dict:
    return {
        "scc": {
            "enabled": True,
            "state_dir": str(tmp_path / ".state/scc"),
        },
        "_scc_catalog": {"server": {"base_url": "https://scc.example.test/"}},
    }


def _paths(tmp_path: Path) -> list[Path]:
    return [tmp_path / ID1, tmp_path / ID2]


def _manifest(identity: str) -> dict:
    return {
        "measurement_id": identity,
        "scientific_session_id": SESSION,
        "scc_configuration_id": 1047,
        "scc_product_ids": [3683],
        "source_segments": ["seg00"],
        "measurement_start_utc": "2024-09-13T15:49:32+00:00",
        "measurement_stop_utc": "2024-09-13T17:50:42+00:00",
        "files": {f"{identity}.nc": {"sha256": "a" * 64}},
    }


class Client:
    def __init__(self, *, existing=(), fail_lookup=None):
        self.existing = set(existing)
        self.fail_lookup = fail_lookup
        self.reads = []

    def measurement(self, identity):
        self.reads.append(identity)
        if identity == self.fail_lookup:
            raise RuntimeError("network status unknown")
        return {"id": identity} if identity in self.existing else None


def _stubs(monkeypatch: pytest.MonkeyPatch, *, fail_submit=None):
    submitted = []

    def submit(config, path, client, *, replace_existing, skip_backup=False):
        submitted.append((path.name, replace_existing))
        if path.name == fail_submit:
            raise RuntimeError("remote POST unknown")
        return {"measurement_id": path.name, "remote_page": "https://scc.example.test/record/"}

    monkeypatch.setattr(batch, "validate_upload_package", lambda path: _manifest(path.name))
    monkeypatch.setattr(batch, "submit", submit)
    return submitted


def test_batch_preflights_all_before_one_remote_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = _stubs(monkeypatch)
    def broken(path):
        if path.name == ID2:
            raise ValueError("corrupt raw NetCDF")
        return _manifest(path.name)

    monkeypatch.setattr(batch, "validate_upload_package", broken)
    client = Client()
    lines = []
    report = batch.submit_batch(
        _config(tmp_path), _paths(tmp_path), client, report_line=lines.append,
    )
    assert report.failed == 1 and report.invalid == 1 and report.remaining == 0
    assert client.reads == [ID1]
    assert submitted == [(ID1, False)]
    assert "PRECHECK INVALID" in " ".join(lines)


def test_batch_uploads_missing_and_skips_existing_without_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = _stubs(monkeypatch)
    client = Client(existing={ID1})
    lines = []
    report = batch.submit_batch(
        _config(tmp_path), _paths(tmp_path), client, report_line=lines.append,
    )
    assert report.successful
    assert (report.submitted, report.skipped_existing, report.replaced) == (1, 1, 0)
    assert submitted == [(ID2, False)]
    assert client.reads == [ID1, ID2]
    assert any("SKIPPED" in line for line in lines)
    assert any("SUBMITTED" in line for line in lines)


def test_explicit_batch_replacement_classifies_new_and_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = _stubs(monkeypatch)
    client = Client(existing={ID1})
    report = batch.submit_batch(
        _config(tmp_path), _paths(tmp_path), client, replace_existing=True,
        report_line=lambda _: None,
    )
    assert report.successful
    assert report.submitted == 1 and report.replaced == 1
    assert submitted == [(ID1, True), (ID2, False)]


def test_batch_network_unknown_stops_remaining_and_avoids_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = _stubs(monkeypatch, fail_submit=ID1)
    client = Client()
    report = batch.submit_batch(
        _config(tmp_path), _paths(tmp_path), client, report_line=lambda _: None,
    )
    assert not report.successful and report.failed == 1
    assert report.stopped_at == ID1
    assert report.remaining == 1
    assert submitted == [(ID1, False)]
    assert client.reads == [ID1]


def test_batch_remote_lookup_failure_never_submits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = _stubs(monkeypatch)
    client = Client(fail_lookup=ID1)
    report = batch.submit_batch(
        _config(tmp_path), _paths(tmp_path), client, report_line=lambda _: None,
    )
    assert report.failed == 1 and report.stopped_at == ID1
    assert submitted == []


def test_uncertain_journal_blocks_batch_even_if_remote_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = _stubs(monkeypatch)
    config = _config(tmp_path)
    with SubmissionStore(state_root(config)) as journal:
        journal.begin(_manifest(ID1))
        journal.transition(ID1, "uncertain", action="simulated_unknown_upload")
    client = Client(existing={ID1})
    report = batch.submit_batch(
        config, _paths(tmp_path), client, report_line=lambda _: None,
    )
    assert report.successful and report.skipped_existing == 1
    assert client.reads == [ID1, ID2]
    assert submitted == [(ID2, False)]


def test_accepted_local_id_but_missing_remotely_halts_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = _stubs(monkeypatch)
    config = _config(tmp_path)
    with SubmissionStore(state_root(config)) as journal:
        journal.begin(_manifest(ID1))
        journal.transition(ID1, "accepted", action="previous_upload")
    report = batch.submit_batch(
        config, _paths(tmp_path), Client(), report_line=lambda _: None,
    )
    assert report.failed == 1 and report.stopped_at == ID1
    assert submitted == []


def test_nineteen_pending_are_sequential_and_cache_is_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = _stubs(monkeypatch)
    identities = [f"20240913spu{1500 + offset:04d}" for offset in range(19)]
    paths = [tmp_path / identity for identity in reversed(identities)]
    client = Client(existing={identities[0], identities[10]})
    lines = []
    report = batch.submit_batch(
        _config(tmp_path), paths, client, report_line=lines.append,
    )
    assert report.successful
    assert report.total == 19
    assert report.submitted == 17 and report.skipped_existing == 2
    assert client.reads == identities
    assert [identity for identity, _ in submitted] == [
        identity for identity in identities
        if identity not in client.existing
    ]
    assert lines[0].startswith("Preflight: 19/19")
    assert lines[-1].startswith("[19/19]")


def test_operational_error_never_continues_to_another_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = _stubs(monkeypatch, fail_submit=ID1)
    client = Client()
    messages = []
    report = batch.submit_batch(
        _config(tmp_path), _paths(tmp_path), client,
        report_line=messages.append,
    )
    assert report.failed == 1 and report.submitted == 0
    assert report.remaining == 1
    assert submitted == [(ID1, False)]
    assert any("No further IDs will be processed" in message for message in messages)
    assert not any("other IDs may continue" in message for message in messages)


def test_one_invalid_package_no_longer_blocks_unrelated_valid_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = _stubs(monkeypatch)
    def validate(path):
        if path.name == ID1:
            raise ValueError("corrupt ancillary")
        return _manifest(path.name)

    monkeypatch.setattr(batch, "validate_upload_package", validate)
    report = batch.submit_batch(
        _config(tmp_path), _paths(tmp_path), Client(),
        report_line=lambda _: None,
    )
    assert report.invalid == 1 and report.submitted == 1
    assert submitted == [(ID2, False)]


def test_cli_routes_batch_without_id_and_prints_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "discover_pending", lambda *_: _paths(tmp_path))
    monkeypatch.setattr(cli, "_client", lambda _: Client(existing={ID1}))
    _stubs(monkeypatch)
    assert cli.main(["upload", "--confirm"]) == 0
    out = capsys.readouterr().out
    assert "submitted=1" in out and "skipped_existing=1" in out
    assert "SCC inputs, scientific data and durable SCC archives were retained." in out


def test_cli_routes_replace_existing_batch_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "discover_pending", lambda *_: _paths(tmp_path))
    monkeypatch.setattr(cli, "_client", lambda _: Client(existing={ID1}))
    _stubs(monkeypatch)
    assert cli.main(["upload", "--confirm", "--replace-existing"]) == 0
    assert "replaced=1" in capsys.readouterr().out


def test_cli_batch_stops_and_returns_failure_on_unknown_remote_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "discover_pending", lambda *_: _paths(tmp_path))
    monkeypatch.setattr(cli, "_client", lambda _: Client())
    _stubs(monkeypatch, fail_submit=ID1)
    assert cli.main(["upload", "--confirm"]) == 1
    output = capsys.readouterr().out
    assert "failed=1" in output and "unprocessed=1" in output
    assert "STOPPED" in output
