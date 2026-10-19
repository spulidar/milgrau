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
SESSION = "spu_20240913-1549Z_20240913-1750Z"


def _config(tmp_path: Path) -> dict:
    return {
        "scc": {
            "enabled": True,
            "state_dir": str(tmp_path / ".state/scc"),
            "upload": {"retries": 0, "timeout_seconds": 30},
        },
        "_scc_catalog": {"server": {"base_url": "https://scc.example.test/"}},
        "_station_catalog": {"station": {"id": "spu"}},
        "directories": {"processed_data": str(tmp_path / "02-processed_data")},
    }


def _stage(tmp_path: Path, *, lr: bool = True) -> Path:
    config = _config(tmp_path)
    with atomic_scc_package(config, IDENTITY, session_id=SESSION) as directory:
        raw = directory / f"{IDENTITY}.nc"
        with nc.Dataset(raw, "w") as ds:
            ds.setncattr("Measurement_ID", IDENTITY)
            ds.setncattr("SCC_Configuration_ID", 1047)
            ds.createDimension("channels", 1)
            ds.createVariable("channel_ID", "i4", ("channels",))[:] = [4071]
            if lr:
                ds.setncattr("LR_File_Name", f"lr_{IDENTITY[:13]}.nc")
        ancillary_name = None
        if lr:
            ancillary_name = f"lr_{IDENTITY[:13]}.nc"
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
    return pending_scc_package(config, IDENTITY, session_id=SESSION)


class FakeResponse:
    def __init__(self, status: int, url: str, *, data: dict | None = None,
                 csrf: str | None = None, location: str | None = None):
        self.status_code = status
        self.url = url
        self._data = data
        self.cookies = {} if csrf is None else {"csrftoken": csrf}
        self.headers = {} if location is None else {"Location": location}

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
                data={
                    "id": IDENTITY,
                    "system": {"id": 1047},
                    "creation_date": "2026-10-17T14:29:53",
                    "start": "2024-09-13T15:49:32",
                    "stop": "2024-09-13T17:50:42",
                } if self.duplicate or self.uploaded else None,
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
            assert kwargs["files"]["data"][0] == f"{IDENTITY}.nc"
            assert kwargs["files"]["lidar_ratio_file"][0] == f"lr_{IDENTITY[:13]}.nc"
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
    (path / f"{IDENTITY}.nc").write_bytes(b"tampered")
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


def test_optional_sounding_in_multipart_and_checked_raw_reference(
    tmp_path: Path,
) -> None:
    import json
    from milgrau.provenance import file_sha256

    folder = _stage(tmp_path)
    sounding_name = f"rs_{IDENTITY[:13]}.nc"
    with nc.Dataset(folder / sounding_name, "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("points", 4)
        for name, values in (
            ("Altitude", [0, 1000, 5000, 10000]),
            ("Temperature", [20, 10, -10, -55]),
            ("Pressure", [931, 840, 540, 260]),
        ):
            ds.createVariable(name, "f8", ("points",))[:] = values
        ds.setncatts({
            "Latitude_degrees_north": -23.5167,
            "Longitude_degrees_east": -46.6333,
            "Altitude_meter_asl": 722.0,
            "Sounding_Start_Date": "20240913",
            "Sounding_Start_Time_UT": "120000",
        })
    raw_file = folder / f"{IDENTITY}.nc"
    with nc.Dataset(raw_file, "r+") as ds:
        ds.setncattr("Sounding_File_Name", sounding_name)
        ds.createVariable("Molecular_Calc", "i4").assignValue(1)
    manifest_file = folder / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    for name in (sounding_name, f"{IDENTITY}.nc"):
        file = folder / name
        manifest["files"][name] = {
            "sha256": file_sha256(file), "size_bytes": file.stat().st_size,
        }
    manifest_file.write_text(json.dumps(manifest))
    assert validate_upload_package(folder)["files"][sounding_name]
    fake = FakeSession()
    assert _client(fake).upload_package(folder)["measurement_id"] == IDENTITY
    sent = next(kwargs["files"] for url, kwargs in fake.posts if url.endswith("/quick/"))
    assert set(sent) == {"data", "lidar_ratio_file", "sounding_file"}
    assert sent["sounding_file"][0] == sounding_name


def test_upload_eventual_api_visibility_never_repeats_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import milgrau.scc.client as client_module

    path = _stage(tmp_path)
    fake = FakeSession()
    original_get = fake.get
    missing_after_submit = {"count": 0}

    def slow_api_get(url, **kwargs):
        if fake.uploaded and "/api/v1/measurements/" in url:
            missing_after_submit["count"] += 1
            if missing_after_submit["count"] <= 2:
                fake.gets.append((url, kwargs))
                return FakeResponse(404, url)
        return original_get(url, **kwargs)

    fake.get = slow_api_get
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    result = _client(fake).upload_package(path)
    assert result["measurement_id"] == IDENTITY
    assert missing_after_submit["count"] == 3
    assert len([
        url for url, _ in fake.posts if url.endswith("/quick/")
    ]) == 1


def test_upload_quick_form_validation_rejection_is_diagnostic_not_resubmitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import milgrau.scc.client as client_module

    path = _stage(tmp_path)
    fake = FakeSession()
    original_post = fake.post

    def reject_form(url, **kwargs):
        if url.endswith("/data_processing/measurements/quick/"):
            fake.posts.append((url, kwargs))
            return FakeResponse(200, url)
        return original_post(url, **kwargs)

    fake.post = reject_form
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    with pytest.raises(SCCRemoteError, match="original form"):
        _client(fake).upload_package(path)
    assert fake.uploaded is False
    assert len([
        url for url, _ in fake.posts if url.endswith("/quick/")
    ]) == 1
    assert len([
        url for url, _ in fake.gets if "/api/v1/measurements/" in url
    ]) == 4  # one prior duplicate-check plus three read-only polls


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


def test_real_cli_send_batch_without_id_and_keep_single_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    _stage(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    monkeypatch.setattr(cli, "_client", lambda _: _client(FakeSession()))
    assert cli.main(["upload", "--confirm"]) == 0
    assert "Batch summary: total=1; submitted=1" in capsys.readouterr().out
    assert pending_scc_package(_config(tmp_path), IDENTITY).is_dir()


def test_explicit_single_id_still_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stage(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    monkeypatch.setattr(cli, "_client", lambda _: _client(FakeSession()))
    assert cli.main(["upload", "--id", IDENTITY, "--confirm"]) == 0


def test_resume_after_delete_cli_requires_one_id_confirm_and_never_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stage(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    assert cli.main(["upload", "--resume-after-delete", "--confirm"]) == 1
    assert cli.main([
        "upload", "--id", IDENTITY, "--resume-after-delete", "--dry-run",
    ]) == 1
    assert cli.main([
        "upload", "--id", IDENTITY, "--resume-after-delete",
        "--replace-existing", "--confirm",
    ]) == 1


def test_resume_after_delete_cli_routes_once_without_deleting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stage(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    import milgrau.scc.workflow as workflow

    calls = []
    def fake_recovery(config, path, client):
        calls.append((path.name, client))
        return {"measurement_id": IDENTITY, "remote_page": "https://scc.example.test/record/"}

    monkeypatch.setattr(workflow, "resume_after_uncertain_delete", fake_recovery)
    fake = object()
    monkeypatch.setattr(cli, "_client", lambda _: fake)
    assert cli.main([
        "upload", "--id", IDENTITY, "--resume-after-delete", "--confirm",
    ]) == 0
    assert calls == [(IDENTITY, fake)]


def test_retry_after_upload_cli_is_single_id_confirm_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stage(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    assert cli.main(["upload", "--retry-after-upload", "--confirm"]) == 1
    assert cli.main([
        "upload", "--id", IDENTITY, "--retry-after-upload", "--dry-run",
    ]) == 1
    assert cli.main([
        "upload", "--id", IDENTITY, "--retry-after-upload",
        "--replace-existing", "--confirm",
    ]) == 1


def test_retry_after_upload_cli_routes_once_without_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stage(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    import milgrau.scc.workflow as workflow

    calls = []
    def replacement(config, path, client):
        calls.append((path.name, client))
        return {
            "measurement_id": IDENTITY,
            "remote_page": "https://scc.example.test/record/",
        }

    monkeypatch.setattr(workflow, "retry_rejected_recovery_upload", replacement)
    fake = object()
    monkeypatch.setattr(cli, "_client", lambda _: fake)
    assert cli.main([
        "upload", "--id", IDENTITY, "--retry-after-upload", "--confirm",
    ]) == 0
    assert calls == [(IDENTITY, fake)]


def test_retry_after_lr_fix_requires_one_explicit_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stage(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    assert cli.main(["upload", "--retry-after-lr-fix", "--confirm"]) == 1
    assert cli.main([
        "upload", "--id", IDENTITY, "--retry-after-lr-fix", "--dry-run",
    ]) == 1
    assert cli.main([
        "upload", "--id", IDENTITY, "--retry-after-lr-fix",
        "--replace-existing", "--confirm",
    ]) == 1


def test_remote_status_can_outlive_deleted_local_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    monkeypatch.setattr(cli, "_client", lambda _: _client(FakeSession(duplicate=True)))
    assert cli.main(["status", "--id", IDENTITY, "--remote"]) == 0
    text = capsys.readouterr().out
    assert "No matching local" in text
    assert f"remote {IDENTITY}" in text


def test_remote_status_survives_corrupted_local_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    pending = _stage(tmp_path)
    # A half-finished manual rename must not prevent remote diagnosis.
    (pending / f"{IDENTITY}.nc").write_bytes(b"locally damaged")
    fake = FakeSession(duplicate=True)
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    monkeypatch.setattr(cli, "_client", lambda _: _client(fake))
    assert cli.main(["status", "--id", IDENTITY, "--remote"]) == 1
    out = capsys.readouterr()
    assert "invalid pending package" in out.err
    assert f"remote {IDENTITY}: present" in out.out
    assert all(not url.endswith("/quick/") and not url.endswith("/delete/")
               for url, _ in fake.posts)


def test_read_only_status_backup_does_not_upload_or_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    import milgrau.scc.backup as remote_backup

    fake = FakeSession(duplicate=True)
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    monkeypatch.setattr(cli, "_client", lambda _: _client(fake))
    archive = tmp_path / "safe_remote_archive"
    archive.mkdir()
    monkeypatch.setattr(
        remote_backup, "archive_remote_measurement",
        lambda *args: (archive, "b" * 64),
    )
    monkeypatch.setattr(remote_backup, "verify_archive", lambda *args: {})
    assert cli.main(["status", "--id", IDENTITY, "--remote", "--backup"]) == 0
    assert "Remote SCC archive VERIFIED:" in capsys.readouterr().out
    assert len([url for url, _ in fake.posts if url.endswith("/quick/")
                or url.endswith("/delete/")]) == 0
    assert cli.main(["status", "--id", IDENTITY, "--backup"]) == 1


def test_download_and_cleanup_are_nondestructive_until_later_phase(
    tmp_path: Path,
) -> None:
    path = _stage(tmp_path)
    # Bulk download/status intentionally accept no --id; cleanup
    # still requires an explicit ID and confirmation.
    assert cli._parser().parse_args(["download"]).identity is None
    assert cli._parser().parse_args(["status", "--remote"]).identity is None
    with pytest.raises(SystemExit) as err_cleanup:
        cli.main(["cleanup"])
    assert err_cleanup.value.code == 2
    assert path.is_dir()
    assert validate_upload_package(path)["measurement_id"] == IDENTITY


def test_login_page_http_401_basic_challenge_is_diagnosed_before_web_login() -> None:
    fake = FakeSession()
    orig_get = fake.get

    def auth_gate_get(url, **kwargs):
        if url.endswith("/accounts/login/"):
            response = FakeResponse(401, url)
            response.headers["WWW-Authenticate"] = 'Basic realm="SCC"'
            return response
        return orig_get(url, **kwargs)

    fake.get = auth_gate_get
    with pytest.raises(SCCRemoteError, match="MILGRAU_SCC_BASIC_USERNAME"):
        _client(fake).login()
    assert fake.posts == []


def test_login_page_http_401_without_challenge_does_not_guess_credentials() -> None:
    fake = FakeSession()
    orig_get = fake.get

    def auth_gate_get(url, **kwargs):
        if url.endswith("/accounts/login/"):
            return FakeResponse(401, url)
        return orig_get(url, **kwargs)

    fake.get = auth_gate_get
    with pytest.raises(SCCRemoteError, match="before website login"):
        _client(fake).login()
    assert fake.posts == []


def test_client_blocks_cross_origin_redirect_without_sending_credentials() -> None:
    fake = FakeSession()
    orig_post = fake.post

    def malicious_login(url, **kwargs):
        if url.endswith("/accounts/login/"):
            return FakeResponse(302, url, location="https://attacker.example/steal")
        return orig_post(url, **kwargs)

    fake.post = malicious_login
    with pytest.raises(SCCRemoteError, match="outside its HTTPS origin"):
        _client(fake).login()
    assert len(fake.posts) == 0


def test_client_does_not_follow_http_307_replay() -> None:
    fake = FakeSession()
    orig_post = fake.post

    def redirect_login(url, **kwargs):
        if url.endswith("/accounts/login/"):
            return FakeResponse(307, url, location="/accounts/login/")
        return orig_post(url, **kwargs)

    fake.post = redirect_login
    with pytest.raises(SCCRemoteError, match="refusing unsafe redirect"):
        _client(fake).login()


def test_client_follows_same_origin_redirect_as_get_after_login() -> None:
    fake = FakeSession()
    orig_get = fake.get
    orig_post = fake.post

    def redirected_get(url, **kwargs):
        if url.endswith("/dashboard/"):
            return FakeResponse(200, url)
        return orig_get(url, **kwargs)

    def redirected_login(url, **kwargs):
        if url.endswith("/accounts/login/"):
            return FakeResponse(302, url, location="/dashboard/")
        return orig_post(url, **kwargs)

    fake.get = redirected_get
    fake.post = redirected_login
    client = _client(fake)
    client.login()
    assert client.measurement(IDENTITY) is None


def test_admin_delete_uses_exact_id_csrf_and_does_not_delete_related() -> None:
    from milgrau.scc.backup import remote_fingerprint

    fake = FakeSession()
    record = {
        "id": IDENTITY, "system": {"id": 1047},
        "creation_date": "2026-10-17T14:29:53",
        "start": "2024-09-13T15:49:32",
        "stop": "2024-09-13T17:50:42",
        "is_queued": False, "is_being_processed": False,
    }
    fake.remote = dict(record)
    original_get = fake.get
    original_post = fake.post

    def fetch(url, **kwargs):
        if "/api/v1/measurements/" in url:
            return FakeResponse(
                200 if fake.remote is not None else 404,
                url, data=fake.remote,
            )
        if url.endswith(f"/admin/database/measurements/{IDENTITY}/delete/"):
            return FakeResponse(200, url, csrf="delete-token")
        return original_get(url, **kwargs)

    def submit(url, **kwargs):
        if url.endswith(f"/admin/database/measurements/{IDENTITY}/delete/"):
            assert kwargs["data"] == {
                "post": "yes",
                "select_delete_related_measurements": "not_delete_related",
            }
            assert kwargs["headers"]["X-CSRFToken"] == "delete-token"
            assert kwargs["allow_redirects"] is False
            fake.remote = None
            return FakeResponse(200, url)
        return original_post(url, **kwargs)

    fake.get = fetch
    fake.post = submit
    _client(fake).delete_measurement(
        IDENTITY, expected_fingerprint=remote_fingerprint(record),
    )
    assert fake.remote is None


def test_admin_delete_rejects_stale_remote_snapshot_before_post() -> None:
    from milgrau.scc.backup import remote_fingerprint

    fake = FakeSession()
    record = {"id": IDENTITY, "system": {"id": 1047}}
    orig_get = fake.get

    def fetch(url, **kwargs):
        if "/api/v1/measurements/" in url:
            return FakeResponse(200, url, data={
                **record, "last_processing_end": "changed",
            })
        return orig_get(url, **kwargs)

    fake.get = fetch
    with pytest.raises(SCCRemoteError, match="changed since"):
        _client(fake).delete_measurement(
            IDENTITY, expected_fingerprint=remote_fingerprint(record),
        )
    assert all(not url.endswith("/delete/") for url, _ in fake.posts)


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
