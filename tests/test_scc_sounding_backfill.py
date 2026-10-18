"""Historical SCC pending sounding backfill only from existing verified L1 data."""

from __future__ import annotations

import json

import netCDF4 as nc
import pytest

from milgrau.cli import scc as cli
from milgrau.provenance import file_sha256
from milgrau.scc import sounding_backfill
from milgrau.scc.packages import validate_upload_package
from milgrau.scc.state import SubmissionStore, state_root

from test_scc_sounding_finalize import (
    ID, _cached, _config, _create_stage, _l1,
)


def _legacy_stage(cfg: dict, tmp_path: Path):
    path, l0, l1 = _create_stage(cfg, tmp_path)
    mf = path / "manifest.json"
    manifest = json.loads(mf.read_text())
    manifest.pop("sounding_stage", None)  # real historical manifests can lack this field
    mf.write_text(json.dumps(manifest) + "\n")
    return path, l0, l1


def test_backfill_preview_does_not_change_pending_bytes_or_journal(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    path, l0, l1 = _legacy_stage(cfg, tmp_path)
    _cached(cfg)
    _l1(l1, available=True)
    monkeypatch.setattr(sounding_backfill, "level1_output_path", lambda _, __: l1)
    initial_files = sorted(p.name for p in path.iterdir())
    raw_sha = file_sha256(path / f"{ID}.nc")
    old_manifest = (path / "manifest.json").read_bytes()
    report, code = sounding_backfill.backfill_pending_soundings(cfg)
    assert code == 0
    assert len(report) == 1 and report[0].status == "READY"
    assert sorted(p.name for p in path.iterdir()) == initial_files
    assert file_sha256(path / f"{ID}.nc") == raw_sha
    assert (path / "manifest.json").read_bytes() == old_manifest
    with SubmissionStore(state_root(cfg)) as journal:
        assert journal.get(ID) is None


def test_backfill_historical_ready_and_preserves_l1_lr_original_digest(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    path, l0, l1 = _legacy_stage(cfg, tmp_path)
    _cached(cfg)
    _l1(l1, available=True)
    monkeypatch.setattr(sounding_backfill, "level1_output_path", lambda _, __: l1)
    first_l1_sha = file_sha256(l1)
    first_lr_sha = file_sha256(path / f"lr_{ID[:13]}.nc")
    old = json.loads((path / "manifest.json").read_text())
    report, code = sounding_backfill.backfill_pending_soundings(cfg, confirm=True)
    assert code == 0 and report[0].status == "UPDATED"
    current = validate_upload_package(path)
    assert current["sounding_stage"] == "ready"
    assert current["sounding_backfilled"]
    assert current["sounding_original_scc_config_sha256"] == old["scc_config_sha256"]
    assert file_sha256(path / f"lr_{ID[:13]}.nc") == first_lr_sha
    assert file_sha256(l1) == first_l1_sha
    assert (path / f"rs_{ID}.nc").exists()
    with nc.Dataset(path / f"{ID}.nc") as raw:
        assert raw.getncattr("Sounding_File_Name") == f"rs_{ID}.nc"
        assert raw.variables["Molecular_Calc"].getValue() == 1
    again, code = sounding_backfill.backfill_pending_soundings(cfg, confirm=True)
    assert code == 0 and again[0].status == "SKIP"
    with SubmissionStore(state_root(cfg)) as journal:
        assert journal.get(ID) is None


def test_absent_l1_observation_leaves_historical_cache_unchanged(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    path, l0, l1 = _legacy_stage(cfg, tmp_path)
    _l1(l1, available=False)
    monkeypatch.setattr(sounding_backfill, "level1_output_path", lambda _, __: l1)
    initial = (path / "manifest.json").read_bytes()
    raw_sha = file_sha256(path / f"{ID}.nc")
    report, code = sounding_backfill.backfill_pending_soundings(cfg, confirm=True)
    assert code == 0 and report[0].status == "SKIP"
    assert (path / "manifest.json").read_bytes() == initial
    assert file_sha256(path / f"{ID}.nc") == raw_sha
    assert not (path / f"rs_{ID}.nc").exists()


def test_existing_journal_blocks_backfill_and_leaves_everything_untouched(
    tmp_path, monkeypatch,
):
    cfg = _config(tmp_path)
    path, l0, l1 = _legacy_stage(cfg, tmp_path)
    _cached(cfg)
    _l1(l1, available=True)
    monkeypatch.setattr(sounding_backfill, "level1_output_path", lambda _, __: l1)
    initial = (path / "manifest.json").read_bytes()
    raw_sha = file_sha256(path / f"{ID}.nc")
    with SubmissionStore(state_root(cfg)) as journal:
        journal.begin(json.loads(initial))
        original = journal.get(ID)
        events = journal.events(ID)
    report, code = sounding_backfill.backfill_pending_soundings(cfg, confirm=True)
    assert code == 1 and report[0].status == "BLOCKED"
    assert (path / "manifest.json").read_bytes() == initial
    assert file_sha256(path / f"{ID}.nc") == raw_sha
    with SubmissionStore(state_root(cfg)) as journal:
        assert journal.get(ID) == original
        assert journal.events(ID) == events


def test_backfill_cli_one_id_dry_run_and_confirm_without_network(
    tmp_path, monkeypatch, capsys,
):
    cfg = _config(tmp_path)
    path, l0, l1 = _legacy_stage(cfg, tmp_path)
    _cached(cfg)
    _l1(l1, available=True)
    monkeypatch.setattr(sounding_backfill, "level1_output_path", lambda _, __: l1)
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "_client", lambda _: pytest.fail("No SCC network expected"))
    assert cli.main(["sounding", "--id", ID, "--dry-run"]) == 0
    assert "READY" in capsys.readouterr().out
    assert cli.main(["sounding", "--id", ID, "--confirm"]) == 0
    assert "UPDATED" in capsys.readouterr().out
    assert validate_upload_package(path)["sounding_stage"] == "ready"
