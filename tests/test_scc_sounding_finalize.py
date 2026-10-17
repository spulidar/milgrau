"""Two-stage LIBIDS→LIPANCORA→SCC handoff, offline and fail-closed."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from milgrau.level0.scc_staging import (
    atomic_scc_package, pending_scc_package, write_scc_manifest,
)
from milgrau.provenance import file_sha256
from milgrau.scc.packages import SCCPackageError, validate_upload_package
from milgrau.scc.sounding_finalize import finalize_session_soundings
from milgrau.scc.state import SubmissionStore, state_root

ID = "20240913spu1549"
SESSION = "spu_20240913-1549Z_20240913-1750Z"


def _config(tmp_path: Path, *, missing="skip") -> dict:
    return {
        "scc": {
            "enabled": True,
            "staging_dir": str(tmp_path / ".cache/scc"),
            "state_dir": str(tmp_path / ".state/scc"),
        },
        "_scc_catalog": {"scc_policy": {"sounding": {
            "enabled": True, "source": "cached_wyoming",
            "missing_policy": missing,
        }}},
        "_station_catalog": {"station": {"radiosonde": {
            "station_id": "83779", "station_name": "SBMT",
            "latitude": -23.5167, "longitude": -46.6333,
            "station_altitude_m": 722.0,
        }}},
        "level1": {"atmosphere": {"radiosonde": {
            "cache_dir": str(tmp_path / "radiosonde"),
            "synoptic_hours_utc": [0, 12],
            "selection": "nearest",
            "max_time_delta_hours": 6.0,
        }}},
    }


def _create_stage(config: dict, tmp_path: Path) -> tuple[Path, Path, Path]:
    l0 = tmp_path / f"{SESSION}_L0.nc"
    l0.write_bytes(b"scientific Level 0 not modified")
    l1 = tmp_path / f"{SESSION}_L1.nc"
    with atomic_scc_package(config, ID) as stage:
        raw = stage / f"{ID}.nc"
        with nc.Dataset(raw, "w") as ds:
            ds.setncattr("Measurement_ID", ID)
            ds.setncattr("SCC_Configuration_ID", 1047)
            ds.setncattr("LR_File_Name", f"lr_{ID[:13]}.nc")
            ds.createDimension("channels", 1)
            ds.createVariable("channel_ID", "i4", ("channels",))[:] = [4071]
            ds.createVariable("Molecular_Calc", "i4").assignValue(0)
        with nc.Dataset(stage / f"lr_{ID[:13]}.nc", "w") as ds:
            ds.createDimension("points", 4)
            ds.createDimension("products", 1)
            ds.createVariable("Altitude", "f8", ("points",))[:] = [0, 10, 20, 30]
            ds.createVariable("Lidar_Ratio", "f8", ("products", "points"))[:] = np.full((1,4), 30.)
            ds.createVariable("product_ID", "i4", ("products",))[:] = [3683]
        write_scc_manifest(
            stage, measurement_id=ID, session_id=SESSION,
            regime="day", month_utc="202409", config_id=1047,
            product_ids=[3683], source_segments=["seg00"],
            profile_start_utc="2024-09-13T15:49:32+00:00",
            profile_stop_utc="2024-09-13T17:50:42+00:00",
            source_canonical_l0=l0, config=config,
            ancillary_name=f"lr_{ID[:13]}.nc",
        )
    return pending_scc_package(config, ID), l0, l1


def _l1(path: Path, *, available: bool, target="2024-09-13T12:00:00+00:00") -> None:
    with nc.Dataset(path, "w") as ds:
        ds.setncattr("Session_ID", SESSION)
        ds.setncattr("radiosonde_available", str(available).lower())
        ds.setncattr("radiosonde_qa_target_datetime_utc", target)


def _cached(cfg: dict) -> None:
    root = Path(cfg["level1"]["atmosphere"]["radiosonde"]["cache_dir"])
    path = root / "2024/09/radiosonde_83779_20240913_12Z.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame({
        "height": [722., 1500., 5000., 10000., 15000.],
        "temperature": [18., 12., -10., -45., -59.],
        "pressure": [930., 850., 540., 265., 110.],
    }).to_csv(path, index=False)
    path.with_suffix(".json").write_text(json.dumps({
        "station_id": "83779",
        "target_datetime_utc": "2024-09-13T12:00:00+00:00",
        "source_type": "radiosonde",
    }))


def test_new_scc_packet_is_not_uploadable_before_lipancora(tmp_path) -> None:
    cfg = _config(tmp_path)
    path, _, _ = _create_stage(cfg, tmp_path)
    with pytest.raises(SCCPackageError, match="LIPANCORA"):
        validate_upload_package(path)


def test_lipancora_finalizes_observed_sounding_then_upload_validation_passes(tmp_path) -> None:
    cfg = _config(tmp_path)
    path, l0, l1 = _create_stage(cfg, tmp_path)
    _cached(cfg)
    _l1(l1, available=True)
    before = json.loads((path / "manifest.json").read_text())
    original_lr = file_sha256(path / f"lr_{ID[:13]}.nc")
    result = finalize_session_soundings(l0, l1, cfg, logging.getLogger(__name__))
    assert result == [f"{ID}: observed sounding attached"]
    manifest = validate_upload_package(path)
    assert manifest["sounding_stage"] == "ready"
    assert manifest["files"][f"rs_{ID}.nc"]["sha256"]
    assert file_sha256(path / f"lr_{ID[:13]}.nc") == original_lr
    assert manifest["files"][f"{ID}.nc"]["sha256"] != before["files"][f"{ID}.nc"]["sha256"]
    with nc.Dataset(path / f"{ID}.nc") as ds:
        assert ds.getncattr("Sounding_File_Name") == f"rs_{ID}.nc"
        assert ds.variables["Molecular_Calc"].getValue() == 1
    assert finalize_session_soundings(l0, l1, cfg, logging.getLogger(__name__)) == []
    assert not (path.parent / f".{ID}.before-sounding").exists()


def test_missing_l1_observation_skips_without_fake_profile(tmp_path) -> None:
    cfg = _config(tmp_path)
    path, l0, l1 = _create_stage(cfg, tmp_path)
    _l1(l1, available=False)
    assert finalize_session_soundings(l0, l1, cfg, logging.getLogger(__name__)) == [
        f"{ID}: no observed sounding (allowed)"
    ]
    assert validate_upload_package(path)["sounding_stage"] == "skipped_missing"
    assert not (path / f"rs_{ID}.nc").exists()
    with nc.Dataset(path / f"{ID}.nc") as ds:
        assert ds.variables["Molecular_Calc"].getValue() == 0
        assert "Sounding_File_Name" not in ds.ncattrs()


def test_strict_missing_keeps_original_awaiting_and_blocks_upload(tmp_path) -> None:
    cfg = _config(tmp_path, missing="fail")
    path, l0, l1 = _create_stage(cfg, tmp_path)
    _l1(l1, available=False)
    before = file_sha256(path / f"{ID}.nc")
    with pytest.raises(ValueError, match="no validated observed sounding"):
        finalize_session_soundings(l0, l1, cfg, logging.getLogger(__name__))
    assert file_sha256(path / f"{ID}.nc") == before
    with pytest.raises(SCCPackageError, match="LIPANCORA"):
        validate_upload_package(path)


def test_submitted_id_is_never_revised_even_if_l1_now_has_sounding(tmp_path) -> None:
    cfg = _config(tmp_path)
    path, l0, l1 = _create_stage(cfg, tmp_path)
    _l1(l1, available=True)
    _cached(cfg)
    with SubmissionStore(state_root(cfg)) as store:
        store.begin(json.loads((path / "manifest.json").read_text()))
    with pytest.raises(ValueError, match="durable submission history"):
        finalize_session_soundings(l0, l1, cfg, logging.getLogger(__name__))
    with pytest.raises(SCCPackageError, match="LIPANCORA"):
        validate_upload_package(path)
