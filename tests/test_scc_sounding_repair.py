"""Offline repair of a stale full-ID sounding filename in historical SCC cache."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import netCDF4 as nc
import pytest

from milgrau.cli import scc as cli
from milgrau.provenance import file_sha256
from milgrau.scc.packages import validate_upload_package
from milgrau.scc.sounding_finalize import finalize_session_soundings
from milgrau.scc.sounding_repair import repair_sounding_name
from milgrau.scc.state import SubmissionStore, state_root

from test_scc_sounding_finalize import ID, _cached, _config, _create_stage, _l1


def _old_stage(tmp_path: Path, *, manually_renamed: bool = True):
    cfg = _config(tmp_path)
    path, l0, l1 = _create_stage(cfg, tmp_path)
    _cached(cfg)
    _l1(l1, available=True)
    finalize_session_soundings(l0, l1, cfg, logging.getLogger(__name__))
    old = f"rs_{ID}.nc"
    new = f"rs_{ID[:13]}.nc"
    (path / new).rename(path / old)
    with nc.Dataset(path / f"{ID}.nc", "r+") as ds:
        ds.setncattr("Sounding_File_Name", old)
    manifest_path = path / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["files"][old] = m["files"].pop(new)
    raw = path / f"{ID}.nc"
    m["files"][raw.name] = {
        "sha256": file_sha256(raw), "size_bytes": raw.stat().st_size,
    }
    manifest_path.write_text(json.dumps(m))
    if manually_renamed:
        (path / old).rename(path / new)  # operator renamed but forgot manifest + attr
    return cfg, path, old, new


def test_repair_preview_is_non_mutating_and_repairs_manual_rename(tmp_path, monkeypatch):
    cfg, path, old, new = _old_stage(tmp_path)
    original_manifest = (path / "manifest.json").read_bytes()
    old_raw = file_sha256(path / f"{ID}.nc")
    lr_name = f"lr_{ID[:13]}.nc"
    old_lr = file_sha256(path / lr_name)
    old_sounding = file_sha256(path / new)
    monkeypatch.setattr(
        cli, "_client",
        lambda _: pytest.fail("SCC network forbidden during offline repair"),
    )
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    assert cli.main(["sounding-repair", "--id", ID, "--dry-run"]) == 0
    assert (path / "manifest.json").read_bytes() == original_manifest
    assert file_sha256(path / f"{ID}.nc") == old_raw
    assert cli.main(["sounding-repair", "--id", ID, "--confirm"]) == 0
    revised = validate_upload_package(path)
    assert old not in revised["files"] and new in revised["files"]
    assert file_sha256(path / new) == old_sounding
    assert file_sha256(path / lr_name) == old_lr
    assert revised["files"][f"{ID}.nc"]["sha256"] != old_raw
    with nc.Dataset(path / f"{ID}.nc") as ds:
        assert ds.getncattr("Sounding_File_Name") == new
        assert ds.variables["Molecular_Calc"].getValue() == 1
    backup = path.parent.parent / "legacy-sounding" / ID
    assert backup.is_dir()
    assert (backup / "manifest.json").read_bytes() == original_manifest
    assert cli.main(["sounding-repair", "--id", ID, "--dry-run"]) == 0
    with SubmissionStore(state_root(cfg)) as journal:
        assert journal.get(ID) is None


def test_repair_old_physical_file_not_renamed_yet(tmp_path):
    cfg, path, old, new = _old_stage(tmp_path, manually_renamed=False)
    assert old in json.loads((path / "manifest.json").read_text())["files"]
    assert "REPAIRED" in repair_sounding_name(cfg, ID, confirm=True)
    assert validate_upload_package(path)["files"][new]["sha256"]
    assert not (path / old).exists()


def test_repair_rejects_uncertain_or_accepted_journal(tmp_path):
    cfg, path, _, _ = _old_stage(tmp_path)
    preimage = (path / "manifest.json").read_bytes()
    with SubmissionStore(state_root(cfg)) as journal:
        # Use the raw and LR hashes recorded before the operator's rename.
        journal.begin(json.loads(preimage))
        journal.transition(ID, "uncertain", action="upload_result_uncertain")
    with pytest.raises(ValueError, match="submission history"):
        repair_sounding_name(cfg, ID, confirm=True)
    assert (path / "manifest.json").read_bytes() == preimage
    assert not (path.parent.parent / "legacy-sounding" / ID).exists()


def test_repair_refuses_tampered_raw_not_covered_by_manifest(tmp_path):
    cfg, path, _, _ = _old_stage(tmp_path)
    with nc.Dataset(path / f"{ID}.nc", "r+") as ds:
        ds.setncattr("some_other_change", "unverified")
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        repair_sounding_name(cfg, ID, confirm=True)
    assert not (path.parent.parent / "legacy-sounding" / ID).exists()
