"""SCC local cache migration never modifies SCC, NetCDF bytes, or upload journal."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from milgrau.cli import scc as cli
from milgrau.level0.scc_staging import pending_scc_package
from milgrau.scc.migrate import migrate_pending
from milgrau.scc.packages import validate_upload_package
from milgrau.scc.state import SubmissionStore, state_root

ID = "20240913spu1549"
SESSION = "spu_20240913-1549Z_20240913-1750Z"


def _config(tmp_path: Path) -> dict:
    return {
        "scc": {
            "enabled": True,
            "staging_dir": str(tmp_path / ".cache/scc"),
            "state_dir": str(tmp_path / ".state/scc"),
        },
        "_scc_catalog": {"server": {"base_url": "https://scc.example.test/"}},
        "_station_catalog": {"station": {"id": "spu"}},
        "directories": {"processed_data": str(tmp_path / "02-processed_data")},
    }


def _stage_old(config: dict, identity: str = ID) -> Path:
    path = pending_scc_package(config, identity, session_id=SESSION)
    path.mkdir(parents=True)
    with nc.Dataset(path / "raw.nc", "w") as raw:
        raw.setncattr("Measurement_ID", identity)
        raw.setncattr("SCC_Configuration_ID", 1047)
        raw.setncattr("LR_File_Name", f"lr_{identity}.nc")
        raw.createDimension("channels", 1)
        raw.createVariable("channel_ID", "i4", ("channels",))[:] = [4071]
    with nc.Dataset(path / f"lr_{identity}.nc", "w") as lr:
        lr.createDimension("products", 1)
        lr.createDimension("points", 3)
        lr.createVariable("Altitude", "f8", ("points",))[:] = [0, 10, 20]
        lr.createVariable("product_ID", "i4", ("products",))[:] = [3683]
        lr.createVariable("Lidar_Ratio", "f8", ("products", "points"))[:] = np.full(
            (1, 3), 29.74,
        )
    file_names = ["raw.nc", f"lr_{identity}.nc"]
    manifest = {
        "schema_version": 1,
        "measurement_id": identity,
        "scientific_session_id": "spu_20240913-1549Z_20240913-1750Z",
        "solar_regime": "day",
        "climatology_month_utc": "202409",
        "scc_configuration_id": 1047,
        "scc_product_ids": [3683],
        "source_segments": ["seg00"],
        "measurement_start_utc": "2024-09-13T15:49:32+00:00",
        "measurement_stop_utc": "2024-09-13T17:50:42+00:00",
        "scc_config_sha256": None,
        "state": "pending",
        "files": {
            name: {
                "sha256": hashlib.sha256((path / name).read_bytes()).hexdigest(),
                "size_bytes": (path / name).stat().st_size,
            } for name in file_names
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest) + "\n")
    assert validate_upload_package(path, allow_legacy_lr=True)["measurement_id"] == identity
    return path


def test_legacy_19_packages_migrate_without_changing_raw_bytes_or_lr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    config = _config(tmp_path)
    identities = [f"20240913spu{1500 + i:04d}" for i in range(19)]
    originals = {}
    for identity in identities:
        path = _stage_old(config, identity)
        originals[identity] = {
            name: (path / name).read_bytes()
            for name in ("raw.nc", f"lr_{identity}.nc")
        }
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "_client", lambda _: pytest.fail("Unexpected SCC network!"))
    assert cli.main(["upload", "--migrate-cache"]) == 0
    assert "19 migrated" in capsys.readouterr().out
    for identity in identities:
        path = pending_scc_package(config, identity)
        assert not (path / "raw.nc").exists()
        assert (path / f"{identity}.nc").read_bytes() == originals[identity]["raw.nc"]
        assert (path / f"lr_{identity}.nc").read_bytes() == originals[identity][
            f"lr_{identity}.nc"
        ]
        assert set(validate_upload_package(path, allow_legacy_lr=True)["files"]) == {
            f"{identity}.nc", f"lr_{identity}.nc",
        }
    assert cli.main(["upload", "--migrate-cache"]) == 0
    assert "19 already current" in capsys.readouterr().out


def test_uncertain_journal_is_unchanged_by_local_filename_migration(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    path = _stage_old(config)
    manifest = json.loads((path / "manifest.json").read_text())
    with SubmissionStore(state_root(config)) as store:
        store.begin(manifest)
        store.transition(ID, "uncertain", action="recovery_upload_uncertain")
        before = store.get(ID)
        original_events = store.events(ID)
    _, changed = migrate_pending(config, ID)
    assert changed
    with SubmissionStore(state_root(config)) as store:
        after = store.get(ID)
        assert after == before
        assert store.events(ID) == original_events
    assert not (path / "raw.nc").exists()
    assert (path / f"{ID}.nc").exists()


def test_crash_recovery_uses_preexisting_matching_copy(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    path = _stage_old(config)
    shutil.copyfile(path / "raw.nc", path / f"{ID}.nc")
    _, changed = migrate_pending(config, ID)
    assert changed
    assert not (path / "raw.nc").exists()
    assert validate_upload_package(path, allow_legacy_lr=True)["files"][f"{ID}.nc"]["sha256"]


def test_preexisting_different_id_named_file_is_never_overwritten(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    path = _stage_old(config)
    (path / f"{ID}.nc").write_bytes(b"different source bytes")
    with pytest.raises(ValueError, match="Conflicting SCC"):
        migrate_pending(config, ID)
    assert (path / "raw.nc").exists()
    assert (path / f"{ID}.nc").read_bytes() == b"different source bytes"
    assert "raw.nc" in json.loads((path / "manifest.json").read_text())["files"]


def test_tampered_original_refuses_migration_without_remote_request(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    path = _stage_old(config)
    (path / "raw.nc").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Invalid SCC package"):
        migrate_pending(config, ID)
    assert not (path / f"{ID}.nc").exists()

def test_lr_repair_changes_only_reference_and_keeps_original_snapshot(
    tmp_path: Path,
) -> None:
    from milgrau.scc.lr_repair import repair_lr_cache

    config = _config(tmp_path)
    path = _stage_old(config)
    migrate_pending(config, ID)
    old_manifest = validate_upload_package(path, allow_legacy_lr=True)
    old_raw = (path / f"{ID}.nc").read_bytes()
    old_lr = (path / f"lr_{ID}.nc").read_bytes()
    corrected, changed = repair_lr_cache(config, ID)
    assert changed and corrected == path
    fixed = validate_upload_package(path)
    new_lr = f"lr_{ID[:13]}.nc"
    assert set(fixed["files"]) == {f"{ID}.nc", new_lr}
    with nc.Dataset(path / f"{ID}.nc") as ds:
        assert ds.getncattr("LR_File_Name") == new_lr
        assert ds.getncattr("Measurement_ID") == ID
    assert (path / new_lr).read_bytes() == old_lr
    assert fixed["files"][new_lr]["sha256"] == old_manifest["files"][
        f"lr_{ID}.nc"
    ]["sha256"]
    assert fixed["files"][f"{ID}.nc"]["sha256"] != old_manifest[
        "files"
    ][f"{ID}.nc"]["sha256"]
    legacy = path.parent.parent / "legacy-lr" / ID
    assert (legacy / f"{ID}.nc").read_bytes() == old_raw
    assert (legacy / f"lr_{ID}.nc").read_bytes() == old_lr
    assert repair_lr_cache(config, ID)[1] is False


def test_lr_repair_audits_uncertain_journal_without_resetting_upload_state(
    tmp_path: Path,
) -> None:
    from milgrau.scc.lr_repair import repair_lr_cache

    config = _config(tmp_path)
    path = _stage_old(config)
    migrate_pending(config, ID)
    before = validate_upload_package(path, allow_legacy_lr=True)
    with SubmissionStore(state_root(config)) as store:
        store.begin(before)
        store.transition(
            ID, "uncertain", action="manual_corrected_upload_uncertain",
            error="SCC Quick upload form rejection",
        )
        old = store.get(ID)
    repair_lr_cache(config, ID)
    with SubmissionStore(state_root(config)) as store:
        new = store.get(ID)
        assert new["state"] == "uncertain"
        assert new["archive_path"] == old["archive_path"]
        assert new["last_error"] == old["last_error"]
        assert new["raw_sha256"] != old["raw_sha256"]
        assert new["lr_sha256"] == old["lr_sha256"]
        assert store.events(ID)[-1]["action"] == "local_lr_filename_repair"


def test_lr_repair_cli_is_offline_and_keeps_pending_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    config = _config(tmp_path)
    path = _stage_old(config)
    migrate_pending(config, ID)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "_client", lambda _: pytest.fail("Network forbidden!"))
    assert cli.main(["upload", "--id", ID, "--repair-lr-cache"]) == 0
    assert "1 repaired" in capsys.readouterr().out
    assert path.exists()
    assert cli.main(["upload", "--dry-run"]) == 0


def test_lr_repair_blocks_same_hour_ancillary_collisions(
    tmp_path: Path,
) -> None:
    from milgrau.scc.lr_repair import repair_lr_cache

    config = _config(tmp_path)
    first = _stage_old(config, ID)
    second_id = "20240913spu1515"
    second = _stage_old(config, second_id)
    migrate_pending(config, ID)
    migrate_pending(config, second_id)
    with pytest.raises(ValueError, match="collide"):
        repair_lr_cache(config, ID)
    assert (first / f"lr_{ID}.nc").is_file()
    assert (second / f"lr_{second_id}.nc").is_file()


def test_lr_repair_refuses_revision_of_unrecognized_uncertain_upload(
    tmp_path: Path,
) -> None:
    from milgrau.scc.lr_repair import repair_lr_cache

    config = _config(tmp_path)
    path = _stage_old(config)
    migrate_pending(config, ID)
    manifest = validate_upload_package(path, allow_legacy_lr=True)
    with SubmissionStore(state_root(config)) as store:
        store.begin(manifest)
        store.transition(ID, "uncertain", action="unknown_destruction")
    with pytest.raises(ValueError, match="unrecognized"):
        repair_lr_cache(config, ID)
    assert validate_upload_package(path, allow_legacy_lr=True)


