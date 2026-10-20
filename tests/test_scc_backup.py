"""SCC old-data snapshots must be complete before any destructive replacement."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import netCDF4 as nc
import pytest

from milgrau.scc.backup import (
    archive_remote_measurement, compare_staged_to_archive,
    verify_archive, remote_fingerprint,
)
from milgrau.scc.client import SCCRemoteError


ID = "20240913spu1549"


def _record(*, processed=False):
    return {
        "id": ID, "creation_date": "2026-10-17T14:29:53",
        "start": "2024-09-13T15:49:32", "stop": "2024-09-13T17:50:42",
        "last_processing_end": "2026-10-17T14:34:31",
        "is_being_processed": False, "is_queued": False,
        "system": {"id": 1047},
        "upload": 127, "elpp": 127 if processed else 0,
        "elda": 127 if processed else 0,
        "raw_file_download_url": f"https://scc.example.test/data_processing/measurements/{ID}/download-raw/",
        **({
            "elpp_products_download_url":
                f"https://scc.example.test/data_processing/measurements/{ID}/download-preprocessed/",
            "elda_products_download_url":
                f"https://scc.example.test/data_processing/measurements/{ID}/download-optical/",
        } if processed else {}),
    }


def _netcdf_bytes(tmp_path: Path, lr_name: str | None = None) -> bytes:
    path = tmp_path / "existing.nc"
    with nc.Dataset(path, "w") as ds:
        ds.setncattr("Measurement_ID", ID)
        if lr_name:
            ds.setncattr("LR_File_Name", lr_name)
        ds.createDimension("time", 3)
        ds.createVariable("signals", "f4", ("time",))[:] = [1, 2, 3]
    return path.read_bytes()


class BinaryResponse:
    def __init__(self, raw: bytes, url: str, *, content_type: str = "application/zip"):
        self.status_code = 200
        self.url = url
        self.headers = {"Content-Type": content_type}
        self.raw = raw

    def iter_content(self, chunk_size):
        for i in range(0, len(self.raw), chunk_size):
            yield self.raw[i : i + chunk_size]

    def close(self):
        pass


class BinaryClient:
    def __init__(self, data: dict[str, bytes]):
        self.base_url = "https://scc.example.test/"
        self.data = data
        self.calls = []

    def _require_origin(self, url):
        if not url.startswith(self.base_url):
            raise SCCRemoteError("Cross-origin remote download rejected.")

    def _read(self, relative, **kwargs):
        self.calls.append(relative)
        if relative == "api/v1/lidarratio_files/":
            class JsonResponse:
                status_code = 200
                def json(self):
                    return {"objects": []}
            return JsonResponse()
        path = self.base_url + relative
        return BinaryResponse(self.data[relative], path)


def _zip(raw: bytes) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("original_remote.nc", raw)
    return stream.getvalue()


def test_complete_remote_snapshot_is_retained_with_verified_hashes(tmp_path: Path):
    record = _record(processed=True)
    raw_zip = _zip(_netcdf_bytes(tmp_path))
    client = BinaryClient({
        f"data_processing/measurements/{ID}/download-raw/": raw_zip,
        f"data_processing/measurements/{ID}/download-preprocessed/": raw_zip,
        f"data_processing/measurements/{ID}/download-optical/": raw_zip,
    })
    destination, digest = archive_remote_measurement(
        client, record, tmp_path / ".state/scc",
    )
    backup = verify_archive(destination, digest)
    assert backup["remote_fingerprint"] == remote_fingerprint(record)
    assert backup["remote_original_lr_file_name"] is None
    assert sorted(backup["files"]) == ["elda.archive", "elpp.archive", "raw.archive"]
    assert all(info["format"] == "zip" for info in backup["files"].values())
    assert json.loads((destination / "archive.json").read_text())["measurement_id"] == ID


def test_previous_external_lr_is_blocked_without_provable_retrieval(tmp_path: Path):
    record = _record()
    raw_zip = _zip(_netcdf_bytes(tmp_path, "lr_existing.nc"))
    client = BinaryClient({
        f"data_processing/measurements/{ID}/download-raw/": raw_zip,
    })
    with pytest.raises(SCCRemoteError, match="uniquely resolve original SCC LR"):
        archive_remote_measurement(client, record, tmp_path / ".state/scc")
    assert not list((tmp_path / ".state/scc/archives" / ID).glob("20*"))


def test_completed_product_missing_url_blocks_any_snapshot(tmp_path: Path):
    record = _record(processed=True)
    record.pop("elda_products_download_url")
    client = BinaryClient({
        f"data_processing/measurements/{ID}/download-raw/":
            _zip(_netcdf_bytes(tmp_path)),
    })
    with pytest.raises(SCCRemoteError, match="no downloadable archive"):
        archive_remote_measurement(client, record, tmp_path / ".state/scc")


def test_backup_manifest_and_data_tampering_is_detected(tmp_path: Path):
    record = _record()
    client = BinaryClient({
        f"data_processing/measurements/{ID}/download-raw/":
            _zip(_netcdf_bytes(tmp_path)),
    })
    destination, digest = archive_remote_measurement(
        client, record, tmp_path / ".state/scc",
    )
    (destination / "raw.archive").write_bytes(b"altered")
    with pytest.raises(SCCRemoteError, match="checksum"):
        verify_archive(destination, digest)


def test_period_and_lidar_ratio_changes_trigger_replacement(tmp_path: Path):
    import hashlib

    record = _record()
    raw = _netcdf_bytes(tmp_path)
    client = BinaryClient({
        f"data_processing/measurements/{ID}/download-raw/": _zip(raw),
    })
    archive, sha = archive_remote_measurement(
        client, record, tmp_path / ".state/scc",
    )
    snapshot = verify_archive(archive, sha)
    local = {
        "measurement_id": ID,
        "measurement_start_utc": "2024-09-13T15:49:32+00:00",
        "measurement_stop_utc": "2024-09-13T17:50:42+00:00",
        "files": {f"{ID}.nc": {"sha256": hashlib.sha256(raw).hexdigest()}},
    }
    result = compare_staged_to_archive(local, snapshot)
    assert result["identical"]
    local["measurement_stop_utc"] = "2024-09-13T18:10:00+00:00"
    assert compare_staged_to_archive(local, snapshot)["period_differs"]
    assert not compare_staged_to_archive(local, snapshot)["identical"]
    local["measurement_stop_utc"] = "2024-09-13T17:50:42+00:00"
    local["files"][f"lr_{ID}.nc"] = {"sha256": "a" * 64}
    result = compare_staged_to_archive(local, snapshot)
    assert result["raw_identical"]
    assert not result["lr_identical"]
    assert not result["identical"]


def test_remote_fingerprint_rejects_record_changes():
    original = _record()
    newer = _record()
    newer["last_processing_end"] = "2026-10-17T16:00:00"
    assert remote_fingerprint(original) != remote_fingerprint(newer)
