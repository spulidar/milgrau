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
    assert output == staging / f"rs_{ID[:13]}.nc"
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


def test_sounding_filename_follows_scc_two_digit_suffix(tmp_path) -> None:
    assert scc_sounding_name(ID) == "rs_20240913spu15.nc"
    # Regression: this exact ID previously failed Quick Upload in the SCC UI.
    assert scc_sounding_name("20241008spu1650") == "rs_20241008spu16.nc"
    assert scc_sounding_name("20241220spu0414") == "rs_20241220spu04.nc"
    with pytest.raises(ValueError, match="Unsafe"):
        scc_sounding_name("../fake")


def test_igra_uses_actual_validated_launch_even_when_nominal_nearest_differs(tmp_path) -> None:
    cfg = config(tmp_path)
    csv = cached_csv(cfg)
    sidecar = json.loads(csv.with_suffix(".json").read_text())
    sidecar["source"] = "NOAA NCEI IGRA2 historical sounding archive"
    sidecar["historical_archive"] = "BRM00083779-data.txt"
    csv.with_suffix(".json").write_text(json.dumps(sidecar))
    output_dir = tmp_path / "igra_selected"
    output_dir.mkdir()
    result = stage_cached_sounding(
        output_dir, ID, "2024-09-13T19:00:00+00:00", cfg,
        observed_target_utc="2024-09-13T12:00:00+00:00",
        observed_origin="igra2",
        observed_cache_csv_name=csv.name,
        validated_max_time_delta_hours=12.0,
    )
    assert result is not None
    with nc.Dataset(result) as ds:
        assert ds.getncattr("Source_Observation_Provider") == "igra2"
        assert "NOAA" in ds.getncattr("Source_Observation")
        assert ds.getncattr("Sounding_Start_Time_UT") == "120000"
        assert ds.getncattr("Selection_Time_Delta_hours") == 7.0


def test_scc_selects_wyoming_suffix_when_igra_and_wyoming_coexist(tmp_path) -> None:
    cfg = config(tmp_path)
    igra_csv = cached_csv(cfg)
    old_sidecar = json.loads(igra_csv.with_suffix(".json").read_text())
    old_sidecar.update({
        "source": "NOAA NCEI IGRA2 historical sounding archive",
        "historical_archive": "BRM00083779-data.txt",
    })
    igra_csv.with_suffix(".json").write_text(json.dumps(old_sidecar))
    wyoming_csv = igra_csv.with_name(igra_csv.stem + "_wyoming.csv")
    pd.read_csv(igra_csv).to_csv(wyoming_csv, index=False)
    wyoming_sidecar = dict(old_sidecar)
    wyoming_sidecar.pop("historical_archive")
    wyoming_sidecar["source"] = "University of Wyoming Upper Air via Siphon"
    wyoming_sidecar["csv_file"] = wyoming_csv.name
    wyoming_csv.with_suffix(".json").write_text(json.dumps(wyoming_sidecar))

    stage = tmp_path / "choose_wyoming"
    stage.mkdir()
    result = stage_cached_sounding(
        stage, ID, START, cfg,
        observed_target_utc=TARGET, observed_origin="wyoming_siphon",
        observed_cache_csv_name=wyoming_csv.name,
    )
    assert result is not None
    with nc.Dataset(result) as ds:
        assert ds.getncattr("Source_Observation_Provider") == "wyoming_siphon"
        assert ds.getncattr("Source_CSV_SHA256") == __import__(
            "hashlib"
        ).sha256(wyoming_csv.read_bytes()).hexdigest()
    assert old_sidecar["historical_archive"] == "BRM00083779-data.txt"
    assert json.loads(igra_csv.with_suffix(".json").read_text()) == old_sidecar


def test_scc_refuses_mislabeled_igra_cache_as_wyoming(tmp_path) -> None:
    cfg = config(tmp_path)
    csv = cached_csv(cfg)
    sidecar = json.loads(csv.with_suffix(".json").read_text())
    sidecar.update({
        "source": "NOAA NCEI IGRA2 historical sounding archive",
        "historical_archive": "BRM00083779-data.txt",
    })
    csv.with_suffix(".json").write_text(json.dumps(sidecar))
    stage = tmp_path / "reject_mismatch"
    stage.mkdir()
    with pytest.raises(ValueError, match="contradicts LIPANCORA"):
        stage_cached_sounding(
            stage, ID, START, cfg,
            observed_target_utc=TARGET, observed_origin="wyoming_siphon",
            observed_cache_csv_name=csv.name,
        )
    assert list(stage.iterdir()) == []


def test_scc_refuses_cache_bytes_changed_since_lipancora(tmp_path) -> None:
    cfg = config(tmp_path)
    csv = cached_csv(cfg)
    staging = tmp_path / "cache_hash_mismatch"
    staging.mkdir()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        stage_cached_sounding(
            staging, ID, START, cfg,
            observed_target_utc=TARGET,
            observed_cache_csv_name=csv.name,
            expected_cache_sha256="0" * 64,
        )
    assert list(staging.iterdir()) == []
