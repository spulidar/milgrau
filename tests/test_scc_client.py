"""Phase 12.4: native SCC upload protocol, offline validation and safety."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import netCDF4 as nc
import numpy as np
import pytest

from milgrau.cli import scc as cli
from milgrau.level0.scc_staging import (
    atomic_scc_package,
    pending_scc_package,
    write_scc_manifest,
)
from milgrau.scc.client import (
    SCCAlreadyExists, SCCClient, SCCCredentials, SCCRemoteError,
    validated_origin,
)
from milgrau.scc.packages import SCCPackageError, validate_upload_package


IDENTITY = "20240913spu1549"


def _config(tmp_path: Path) -> dict:
    return {
        "scc": {
            "enabled": True, "staging_dir": str(tmp_path / ".cache/scc"),
            "upload": {"retries": 0, "timeout_seconds": 30},
        },
        "_scc_catalog": {"server": {"base_url": "https://scc.example.test/"}},
    }


def _stage(tmp_path: Path, *, lr: bool = True) -> Path:
    config = _config(tmp_path)
    with atomic_scc_package(config, IDENTITY) as directory:
        raw = directory / "raw.nc"
        with nc.Dataset(raw, "w") as ds:
            ds.setncattr("Measurement_ID", IDENTITY)
            ds.setncattr("SCC_Configuration_ID", 1047)
            ds.createDimension("channels", 1)
            ds.createVariable("channel_ID", "i4", ("channels",))[:] = [4071]
            if lr:
                ds.setncattr("LR_File_Name", f"lr_{IDENTITY}.nc")
        ancillary_name = None
        if lr:
            ancillary_name = f"lr_{IDENTITY}.nc"
            with nc.Dataset(directory / ancillary_name, "w") as ds:
                ds.createDimension("products", 1)
                ds.createDimension("points", 3)
                ds.createVariable("Altitude", "f8", ("points",))[:] = [0, 15, 30]
                ds.createVariable("product_ID", "i4", ("products",))[:] = [3683]
                ds.createVariable(
                    "Lidar_Ratio", "f8", ("products", "points"),
                )[:] = np.full((1, 3), 29.74)
        write_scc_manifest(
            directory,
            measurement_id=IDENTITY,
            session_id="spu_20240913-1549Z_20240913-1750Z",
            regime="day", month_utc="202409",
            config_id=1047, product_ids=[3683],
            source_segments=["seg00"],
            profile_start_utc="2024-09-13T15:49:00+00:00",
            profile_stop_utc="2024-09-13T17:50:00+00:00",
            source_canonical_l0=tmp_path / "scientific_L0.nc",
            config=config,
            ancillary_name=ancillary_name,
        )
    return pending_scc_package(config, IDENTITY)


class FakeResponse:
    def __init__(self, status: int, url: str, *, data: dict | None = None,
                 csrf: str | None = None):
        self.status_code = status
        self.url = url
        self._data = data
        self.cookies = {} if csrf is None else {"csrftoken": csrf}

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, *, duplicate: bool = False, fail_post: bool = False):
        self.verify = None
        self.auth = None
        self.cookies = {}
        self.duplicate = duplicate
        self.fail_post = fail_post
        self.uploaded = False
        self.posts = []
        self.gets = []

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        path = urlsplit(url).path
        if path == "/accounts/login/":
            return FakeResponse(200, url, csrf="login-token")
        if path == "/data_processing/measurements/quick/":
            return FakeResponse(200, url, csrf="upload-token")
        if path.startswith("/api/v1/measurements/"):
            return FakeResponse(
                200 if self.duplicate or self.uploaded else 404,
                url,
                data={"id": IDENTITY} if self.duplicate or self.uploaded else None,
            )
        raise AssertionError(f"Unexpected GET {url}")

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if url.endswith("/accounts/login/"):
            assert kwargs["headers"]["X-CSRFToken"] == "login-token"
            assert kwargs["data"]["username"] == "person"
            return FakeResponse(200, "https://scc.example.test/home/")
        if url.endswith("/data_processing/measurements/quick/"):
            assert kwargs["headers"]["X-CSRFToken"] == "upload-token"
            assert kwargs["data"]["system"] == "1047"
            assert kwargs["files"]["data"][0] == "raw.nc"
            assert kwargs["files"]["lidar_ratio_file"][0] == f"lr_{IDENTITY}.nc"
            assert kwargs["files"]["data"][1].read(4)
            if self.fail_post:
                raise OSError("transmission uncertain")
            self.uploaded = True
            return FakeResponse(
                200, f"https://scc.example.test/data_processing/measurements/{IDENTITY}/",
            )
        raise AssertionError(f"Unexpected POST {url}")


def _client(fake: FakeSession) -> SCCClient:
    return SCCClient(
        "https://scc.example.test/",
        SCCCredentials("person", "password", "basic", "secret"),
        timeout=30, retries=0, session=fake,
    )


def test_validated_package_and_dry_run_use_no_credentials_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    path = _stage(tmp_path)
    manifest = validate_upload_package(path)
    assert manifest["files"] == {
        name: item for name, item in manifest["files"].items()
    }
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    monkeypatch.delenv("MILGRAU_SCC_USERNAME", raising=False)
    monkeypatch.delenv("MILGRAU_SCC_PASSWORD", raising=False)
    assert cli.main(["upload", "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "Dry-run: 1/1 valid; no network request." in text
    assert "1047" in text


def test_dry_run_rejects_checksum_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _stage(tmp_path)
    (path / "raw.nc").write_bytes(b"tampered")
    with pytest.raises(SCCPackageError, match="checksum"):
        validate_upload_package(path)
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    assert cli.main(["upload", "--dry-run"]) == 1


def test_native_upload_sends_correct_multipart_only_once(tmp_path: Path) -> None:
    path = _stage(tmp_path)
    fake = FakeSession()
    client = _client(fake)
    result = client.upload_package(path)
    assert result["measurement_id"] == IDENTITY
    assert fake.uploaded
    assert fake.verify is True
    assert fake.auth == ("basic", "secret")
    assert len([url for url, _ in fake.posts if url.endswith("/quick/")]) == 1
    assert any("api/v1/measurements" in url for url, _ in fake.gets)


def test_remote_duplicate_never_triggers_upload_or_delete(tmp_path: Path) -> None:
    path = _stage(tmp_path)
    fake = FakeSession(duplicate=True)
    with pytest.raises(SCCAlreadyExists, match="already contains"):
        _client(fake).upload_package(path)
    assert len(fake.posts) == 1  # Website login only


def test_ambiguous_post_is_never_retried(tmp_path: Path) -> None:
    path = _stage(tmp_path)
    fake = FakeSession(fail_post=True)
    with pytest.raises(SCCRemoteError, match="UNKNOWN remote outcome"):
        _client(fake).upload_package(path)
    assert len([url for url, _ in fake.posts if url.endswith("/quick/")]) == 1


def test_real_cli_send_requires_exact_id_and_confirm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stage(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    monkeypatch.setattr(cli, "_client", lambda _: _client(FakeSession()))
    assert cli.main(["upload", "--confirm"]) == 1
    assert cli.main(["upload", "--id", IDENTITY, "--confirm"]) == 0


def test_download_and_cleanup_are_nondestructive_until_later_phase(
    tmp_path: Path,
) -> None:
    path = _stage(tmp_path)
    assert cli.main(["download"]) == 2
    assert cli.main(["cleanup"]) == 2
    assert path.is_dir()
    assert validate_upload_package(path)["measurement_id"] == IDENTITY


def test_https_origin_and_environment_credentials_are_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for invalid in ["http://scc.example.test/", "https://name:pass@scc.example.test/",
                    "https://scc.example.test/path", "https://scc.example.test/?a=1"]:
        with pytest.raises(ValueError, match="HTTPS origin"):
            validated_origin(invalid)
    monkeypatch.delenv("MILGRAU_SCC_USERNAME", raising=False)
    monkeypatch.delenv("MILGRAU_SCC_PASSWORD", raising=False)
    with pytest.raises(SCCRemoteError, match="MILGRAU_SCC_USERNAME"):
        SCCCredentials.from_environment()
    monkeypatch.setenv("MILGRAU_SCC_USERNAME", "person")
    monkeypatch.setenv("MILGRAU_SCC_PASSWORD", "secret")
    monkeypatch.setenv("MILGRAU_SCC_BASIC_USERNAME", "basic")
    monkeypatch.delenv("MILGRAU_SCC_BASIC_PASSWORD", raising=False)
    with pytest.raises(SCCRemoteError, match="both"):
        SCCCredentials.from_environment()
