"""Observed SCC sounding is opt-in, cache-only and never model-synthesized."""

from __future__ import annotations

import json
from pathlib import Path

import netCDF4 as nc
import pandas as pd
import pytest

from milgrau.level0.scc_sounding import (
    scc_sounding_name, stage_cached_sounding, validate_sounding,
)

ID = "20240913spu1549"
START = "2024-09-13T15:49:32+00:00"
TARGET = "2024-09-13T12:00:00+00:00"


def config(tmp_path: Path, *, enabled=True, missing="fail") -> dict:
    return {
        "_scc_catalog": {
            "scc_policy": {
                "sounding": {
                    "enabled": enabled, "source": "cached_wyoming",
                    "missing_policy": missing,
                }
            }
        },
        "_station_catalog": {
            "station": {
                "radiosonde": {
                    "station_id": "83779",
                    "station_name": "SBMT Marte Civ",
                    "latitude": -23.5167,
                    "longitude": -46.6333,
                    "station_altitude_m": 722.0,
                }
            }
        },
        "level1": {"atmosphere": {"radiosonde": {
            "cache_dir": str(tmp_path / "radiosonde"),
            "synoptic_hours_utc": [0, 12],
            "selection": "nearest",
            "max_time_delta_hours": 6.0,
        }}},
    }


def cached_csv(config: dict) -> Path:
    root = Path(config["level1"]["atmosphere"]["radiosonde"]["cache_dir"])
    path = root / "2024" / "09" / "radiosonde_83779_20240913_12Z.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame({
        "height": [722., 1500., 5000., 10000., 15000.],
        "temperature": [18., 12., -10., -45., -59.],
        "pressure": [930., 850., 540., 265., 110.],
        "relative_humidity": [75., 65., 40., 12., 10.],
    }).to_csv(path, index=False)
    path.with_suffix(".json").write_text(
        json.dumps({
            "station_id": "83779",
            "target_datetime_utc": TARGET,
            "source_type": "radiosonde",
        }),
        encoding="utf-8",
    )
    return path


def test_observed_wyoming_cache_produces_scc_soundings_exact_schema(tmp_path) -> None:
    cfg = config(tmp_path)
    cached_csv(cfg)
    staging = tmp_path / ID
    staging.mkdir()
    output = stage_cached_sounding(staging, ID, START, cfg)
    assert output == staging / f"rs_{ID}.nc"
    assert validate_sounding(output, identity=ID)["points"] == 5
    with nc.Dataset(output) as ds:
        assert ds.file_format == "NETCDF4_CLASSIC"
        assert ds.variables["Altitude"][:].tolist() == pytest.approx(
            [0, 778, 4278, 9278, 14278]
        )
        assert ds.variables["Temperature"][:].tolist() == [18, 12, -10, -45, -59]
        assert ds.variables["Pressure"][:].tolist() == [930, 850, 540, 265, 110]
        assert ds.getncattr("Sounding_Start_Date") == "20240913"
        assert ds.getncattr("Sounding_Start_Time_UT") == "120000"
        assert ds.getncattr("WMO_Station_Number") == "83779"
        assert ds.getncattr("Altitude_meter_asl") == 722.0


def test_sounding_uses_lipancora_session_midpoint_not_first_laser_time(tmp_path) -> None:
    cfg = config(tmp_path)
    cached_csv(cfg)  # 12Z sounding exists, 00Z sounding does not.
    stage = tmp_path / ID
    stage.mkdir()
    output = stage_cached_sounding(
        stage, ID, "2024-09-13T04:19:00+00:00", cfg,
        observed_target_utc="2024-09-13T12:00:00+00:00",
        measurement_stop_utc="2024-09-13T18:19:00+00:00",
    )
    assert output is not None
    assert validate_sounding(output, identity=ID)["points"] == 5
    # Without the stop time, the nearest launch to 04:19Z is 00Z;
    # a 12Z observed QA must not be silently substituted.
    with pytest.raises(ValueError, match="No matching LIPANCORA"):
        stage_cached_sounding(
            stage, ID, "2024-09-13T04:19:00+00:00", cfg,
            observed_target_utc="2024-09-13T12:00:00+00:00",
        )


def test_sounding_is_off_by_default_and_never_downloads_or_fakes(tmp_path) -> None:
    cfg = config(tmp_path, enabled=False)
    stage = tmp_path / ID
    stage.mkdir()
    assert stage_cached_sounding(stage, ID, START, cfg) is None
    cfg = config(tmp_path, enabled=True, missing="skip")
    assert stage_cached_sounding(stage, ID, START, cfg) is None
    cfg = config(tmp_path, enabled=True, missing="fail")
    with pytest.raises(ValueError, match="missing"):
        stage_cached_sounding(stage, ID, START, cfg)
    assert not list(stage.iterdir())


def test_wrong_sounding_station_metadata_rejected_before_any_new_file(tmp_path) -> None:
    cfg = config(tmp_path)
    csv = cached_csv(cfg)
    csv.with_suffix(".json").write_text(
        json.dumps({
            "station_id": "99999", "target_datetime_utc": TARGET,
            "source_type": "radiosonde",
        }),
        encoding="utf-8",
    )
    stage = tmp_path / ID
    stage.mkdir()
    with pytest.raises(ValueError, match="provenance"):
        stage_cached_sounding(stage, ID, START, cfg)
    assert list(stage.iterdir()) == []


def test_unphysical_pressure_fails_closed(tmp_path) -> None:
    cfg = config(tmp_path)
    csv = cached_csv(cfg)
    df = pd.read_csv(csv)
    df.loc[2, "pressure"] = 999.0
    df.to_csv(csv, index=False)
    stage = tmp_path / ID
    stage.mkdir()
    with pytest.raises(ValueError, match="physical"):
        stage_cached_sounding(stage, ID, START, cfg)
    assert list(stage.iterdir()) == []


def test_sounding_id_filename_is_not_shortened_like_lr(tmp_path) -> None:
    assert scc_sounding_name(ID) == f"rs_{ID}.nc"
    with pytest.raises(ValueError, match="Unsafe"):
        scc_sounding_name("../fake")
