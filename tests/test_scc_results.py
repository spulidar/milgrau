"""Phase 12.6: verified SCC downloads and conservative disposable-cache cleanup."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from milgrau.cli import scc as cli
from milgrau.io.paths import level0_output_path
from milgrau.level0.scc_staging import (
    atomic_scc_package, pending_scc_package, write_scc_manifest,
)
from milgrau.scc import results
from milgrau.scc.cleanup import cleanup_pending, verify_cleanup
from milgrau.scc.client import SCCRemoteError
from milgrau.scc.results import download_results, validate_results
from milgrau.scc.state import SubmissionStore, state_root

ID = "20240913spu1549"
SESSION = "spu_20240913-1549Z_20240913-1750Z"
CREATION = "2026-10-17T14:29:53"


def _config(tmp_path: Path) -> dict:
    return {
        "directories": {
            "processed_data": str(tmp_path / "02-processed_data"),
        },
        "_station_catalog": {"station": {"id": "spu"}},
        "_scc_catalog": {"server": {"base_url": "https://scc.example.test/"}},
        "scc": {
            "enabled": True,
            "state_dir": str(tmp_path / ".state/scc"),
            "staging_dir": str(tmp_path / ".cache/scc"),
            "cleanup": {"policy": "after_download"},
        },
    }


def _zip_bytes() -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("profile.txt", "SCC external optical products")
    return data.getvalue()


def _record(*, created: str = CREATION, finished: bool = True) -> dict:
    status = 127 if finished else 4
    return {
        "id": ID,
        "system": {"id": 1047},
        "creation_date": created,
        "start": "2024-09-13T15:49:32",
        "stop": "2024-09-13T17:50:42",
        "last_processing_end": "2026-10-17T14:34:31",
        "is_queued": False, "is_being_processed": False,
        "elpp": status, "elda": status,
        "elpp_products_download_url":
            f"https://scc.example.test/data_processing/measurements/{ID}/download-preprocessed/",
        "elda_products_download_url":
            f"https://scc.example.test/data_processing/measurements/{ID}/download-optical/",
    }


class FakeClient:
    def __init__(self, record: dict | None = None):
        self.record = _record() if record is None else record
        self.queried = 0

    def measurement(self, identity):
        assert identity == ID
        self.queried += 1
        return self.record


def _stage(config: dict) -> dict:
    scientific = level0_output_path(SESSION, config)
    scientific.parent.mkdir(parents=True, exist_ok=True)
    scientific.write_bytes(b"canonical scientific Level 0 is NEVER modified")
    with atomic_scc_package(config, ID) as directory:
        with nc.Dataset(directory / f"{ID}.nc", "w") as raw:
            raw.setncattr("Measurement_ID", ID)
            raw.setncattr("SCC_Configuration_ID", 1047)
            raw.setncattr("LR_File_Name", f"lr_{ID[:13]}.nc")
            raw.createDimension("channels", 1)
            raw.createVariable("channel_ID", "i4", ("channels",))[:] = [4071]
        with nc.Dataset(directory / f"lr_{ID[:13]}.nc", "w") as lr:
            lr.createDimension("products", 1)
            lr.createDimension("points", 3)
            lr.createVariable("Altitude", "f8", ("points",))[:] = [0, 15, 30]
            lr.createVariable("product_ID", "i4", ("products",))[:] = [3683]
            lr.createVariable("Lidar_Ratio", "f8", ("products", "points"))[:] = np.full(
                (1, 3), 29.74,
            )
        manifest = write_scc_manifest(
            directory, measurement_id=ID, session_id=SESSION,
            regime="day", month_utc="202409", config_id=1047,
            product_ids=[3683], source_segments=["seg00"],
            profile_start_utc="2024-09-13T15:49:32+00:00",
            profile_stop_utc="2024-09-13T17:50:42+00:00",
            source_canonical_l0=scientific,
            config=config, ancillary_name=f"lr_{ID[:13]}.nc",
        )
    return manifest


def _owned(config: dict, manifest: dict, *, state: str = "accepted") -> None:
    with SubmissionStore(state_root(config)) as store:
        store.begin(manifest)
        store.transition(
            ID, state, action="confirmed_upload",
            remote_creation_date=CREATION,
        )


def _mock_stream(monkeypatch: pytest.MonkeyPatch, *, fail: str | None = None):
    payload = _zip_bytes()
    calls = []

    def stream(client, url, destination):
        kind = "elpp" if "preprocessed" in url else "elda"
        calls.append(kind)
        if kind == fail:
            raise SCCRemoteError("simulated remote result transfer interruption")
        destination.write_bytes(payload)
        return {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "format": "zip",
        }

    monkeypatch.setattr(results, "_download", stream)
    return calls


def test_journal_owned_download_auto_cleans_only_pending_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    manifest = _stage(config)
    _owned(config, manifest)
    calls = _mock_stream(monkeypatch)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "_client", lambda _: FakeClient())

    assert cli.main(["download", "--id", ID]) == 0
    assert sorted(calls) == ["elda", "elpp"]
    target = results.result_location(config, SESSION, ID, CREATION)
    saved = validate_results(target)
    assert saved["uploaded_by_milgrau"] is True
    assert saved["scientific_session_id"] == SESSION
    assert set(saved["files"]) == {"elpp", "elda"}
    assert not pending_scc_package(config, ID).exists()
    assert level0_output_path(SESSION, config).read_bytes() == (
        b"canonical scientific Level 0 is NEVER modified"
    )
    assert not (target.parent.parent / f"{SESSION}_L1.nc").exists()
    with SubmissionStore(state_root(config)) as journal:
        assert journal.get(ID)["state"] == "cleaned"
        assert journal.get(ID)["results_manifest_sha256"]
        assert journal.events(ID)[-1]["action"] == "verified_pending_cache_cleanup"
        assert len(journal.events(ID)) >= 3
    assert (target / "results.json").is_file()
    assert (state_root(config) / "submissions.sqlite").exists()


def test_download_keep_cache_then_explicit_cleanup_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    manifest = _stage(config)
    _owned(config, manifest)
    _mock_stream(monkeypatch)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "_client", lambda _: FakeClient())
    assert cli.main(["download", "--id", ID, "--keep-cache"]) == 0
    assert pending_scc_package(config, ID).is_dir()
    with SubmissionStore(state_root(config)) as store:
        assert store.get(ID)["state"] == "downloaded"
    assert cli.main(["cleanup", "--id", ID, "--dry-run"]) == 0
    assert pending_scc_package(config, ID).is_dir()
    assert cli.main(["cleanup", "--id", ID, "--confirm"]) == 0
    assert not pending_scc_package(config, ID).exists()
    assert cli.main(["cleanup", "--id", ID, "--confirm"]) == 0


def test_preexisting_external_scc_can_download_but_must_keep_pending_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _stage(config)
    _mock_stream(monkeypatch)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "_client", lambda _: FakeClient())
    assert cli.main(["download", "--id", ID, "--session", SESSION]) == 0
    assert pending_scc_package(config, ID).is_dir()
    target = results.result_location(config, SESSION, ID, CREATION)
    assert validate_results(target)["uploaded_by_milgrau"] is False
    with pytest.raises(SCCRemoteError, match="No MILGRAU submission journal"):
        verify_cleanup(config, ID)


def test_missing_elda_completion_never_publishes_or_cleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    manifest = _stage(config)
    _owned(config, manifest)
    _mock_stream(monkeypatch)
    with pytest.raises(SCCRemoteError, match="ELPP|ELDA"):
        download_results(config, ID, FakeClient(_record(finished=False)))
    assert pending_scc_package(config, ID).is_dir()
    assert not (level0_output_path(SESSION, config).parent / "scc-results" / ID).exists()


def test_partial_network_failure_leaves_no_visible_results_and_retains_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    manifest = _stage(config)
    _owned(config, manifest)
    _mock_stream(monkeypatch, fail="elda")
    with pytest.raises(SCCRemoteError, match="transfer interruption"):
        download_results(config, ID, FakeClient())
    assert not results.result_location(config, SESSION, ID, CREATION).exists()
    assert pending_scc_package(config, ID).exists()
    with SubmissionStore(state_root(config)) as store:
        assert store.get(ID)["state"] == "accepted"


def test_remote_generation_mismatch_downloads_but_refuses_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    manifest = _stage(config)
    _owned(config, manifest)
    _mock_stream(monkeypatch)
    target, saved = download_results(config, ID, FakeClient(_record(created="different")))
    assert target.is_dir()
    assert saved["uploaded_by_milgrau"] is False
    with SubmissionStore(state_root(config)) as store:
        assert store.get(ID)["state"] == "accepted"
    with pytest.raises(SCCRemoteError, match="not eligible"):
        verify_cleanup(config, ID)


def test_tampered_result_blocks_cleanup_without_touching_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    manifest = _stage(config)
    _owned(config, manifest)
    _mock_stream(monkeypatch)
    download_results(config, ID, FakeClient())
    target = results.result_location(config, SESSION, ID, CREATION)
    (target / "elpp.archive").write_bytes(b"tampered")
    with pytest.raises(SCCRemoteError, match="Corrupted"):
        cleanup_pending(config, ID, dry_run=False)
    assert pending_scc_package(config, ID).is_dir()


def test_download_without_scientific_session_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _mock_stream(monkeypatch)
    with pytest.raises(SCCRemoteError, match="Specify --session"):
        download_results(config, ID, FakeClient())
    with pytest.raises(SCCRemoteError, match="Canonical Level 0"):
        download_results(config, ID, FakeClient(), source_session=SESSION)


def test_failed_legacy_record_cleanup_never_assumes_upload_ownership(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = _stage(config)
    with SubmissionStore(state_root(config)) as store:
        store.begin(manifest)
        store.transition(ID, "remote_exists", action="read_only_remote_check")
    with pytest.raises(SCCRemoteError, match="not eligible"):
        cleanup_pending(config, ID, dry_run=False)


def test_result_snapshot_is_immutable_across_remote_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _stage(config)
    _mock_stream(monkeypatch)
    download_results(config, ID, FakeClient(), source_session=SESSION)
    newer, newer_manifest = download_results(
        config, ID, FakeClient(_record(created="new-generation")),
        source_session=SESSION,
    )
    previous = results.result_location(config, SESSION, ID, CREATION)
    assert newer != previous
    assert previous.is_dir() and newer.is_dir()
    assert newer_manifest["remote_creation_date"] == "new-generation"
    assert pending_scc_package(config, ID).exists()


def test_legacy_state_db_migrates_without_losing_upload_journal(tmp_path: Path) -> None:
    import sqlite3

    root = state_root(_config(tmp_path))
    root.mkdir(parents=True)
    conn = sqlite3.connect(root / "submissions.sqlite")
    conn.executescript(
        """
        CREATE TABLE submissions (
            measurement_id TEXT PRIMARY KEY,
            scientific_session_id TEXT NOT NULL,
            configuration_id INTEGER NOT NULL,
            raw_sha256 TEXT NOT NULL,
            lr_sha256 TEXT, product_ids_json TEXT NOT NULL,
            source_segments_json TEXT NOT NULL, scc_config_sha256 TEXT,
            start_utc TEXT NOT NULL, stop_utc TEXT NOT NULL, state TEXT NOT NULL,
            remote_creation_date TEXT, archive_path TEXT,
            archive_manifest_sha256 TEXT, last_error TEXT,
            updated_utc TEXT NOT NULL
        );
        CREATE TABLE events(
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            measurement_id TEXT NOT NULL, timestamp_utc TEXT NOT NULL,
            action TEXT NOT NULL, state TEXT NOT NULL, payload_json TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()
    with SubmissionStore(root) as store:
        assert store.list() == []
        columns = {
            row[1] for row in store._conn.execute("PRAGMA table_info(submissions)")
        }
        assert {"results_path", "results_manifest_sha256"}.issubset(columns)
