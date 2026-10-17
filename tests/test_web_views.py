"""Real NetCDF contract-backed tests for all non-RCS Explorer panels."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import yaml
from fastapi.testclient import TestClient
from pathlib import Path

from milgrau.web.app import create_app

SESSION = "spu_20240620-2103Z_20240621-0848Z"


@pytest.fixture
def demo(tmp_path: Path) -> TestClient:
    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repo / "config.yaml").read_text(encoding="utf-8"))
    config["station_config"] = str((repo / "station.yaml").resolve())
    root = tmp_path / "processed"
    config["directories"]["processed_data"] = str(root)
    cp = tmp_path / "config.yaml"
    cp.write_text(yaml.safe_dump(config), encoding="utf-8")
    folder = root / "spu" / "2024" / "06" / SESSION
    folder.mkdir(parents=True)
    times = np.asarray(["2024-06-20T23:40", "2024-06-20T23:45"],
                       dtype="datetime64[m]")
    atmosphere_times = np.asarray(["2024-06-20T23:00", "2024-06-21T00:00"],
                                  dtype="datetime64[m]")
    z = np.asarray([0., 500., 1000., 1500., 2000., 2500., 3000.])
    ds = xr.Dataset(
        {
            "range_corrected_signal": (
                ("time", "channel", "altitude"),
                np.ones((2, 2, 7), dtype=np.float32),
                {"units": "a.u. m2"},
            ),
            "signal_pre_background": (
                ("time", "channel", "altitude"),
                5 * np.ones((2, 2, 7), dtype=np.float32),
                {"units": "a.u."},
            ),
            "Atmospheric_Temperature_K": (
                ("atmosphere_time", "altitude"),
                np.asarray([[290, 285, 280, 275, 270, 265, 260],
                            [291, 286, 281, 276, 271, 266, 261]], dtype=float),
                {"units": "K"},
            ),
            "Atmospheric_Pressure_hPa": (
                ("atmosphere_time", "altitude"),
                np.asarray([[1000, 950, 900, 850, 800, 750, 700],
                            [995, 945, 895, 845, 795, 745, 695]], dtype=float),
                {"units": "hPa"},
            ),
            "Atmospheric_Source_Type": (
                ("atmosphere_time",), ["era5", "ussa76"],
            ),
            "background_outlier_fraction": (
                ("time", "channel"), [[0.01, 0.02], [0.04, 0.06]],
            ),
            "pc_saturation_fraction": (
                ("time", "channel"), [[0.0, 0.1], [0.0, 0.2]],
            ),
            "solar_regime": (("time",), ["night", "night"]),
        },
        coords={
            "time": times, "atmosphere_time": atmosphere_times,
            "altitude": ("altitude", z, {"units": "m"}),
            "channel": ["532.AN", "532.PC"],
        },
        attrs={"Session_ID": SESSION, "software_version": "test-only"},
    )
    ds.to_netcdf(folder / f"{SESSION}_L1.nc")
    l2 = xr.Dataset(
        {
            "aerosol_backscatter_mean": (
                ("wavelength", "altitude"), [[1, 2, 3, 4, 5, 6, 7],
                                             [7, 6, 5, 4, 3, 2, 1]],
                {"units": "m-1 sr-1"},
            ),
            "aerosol_extinction_mean": (
                ("wavelength", "altitude"), [[2, 3, 4, 5, 6, 7, 8],
                                             [8, 7, 6, 5, 4, 3, 2]],
                {"units": "m-1"},
            ),
            "retrieval_success_flag": (("block_time", "wavelength"),
                                       [[1, 0], [1, 1]]),
        },
        coords={
            "wavelength": [355, 532],
            "altitude": ("altitude", z, {"units": "m"}),
            "block_time": times,
        },
        attrs={"Session_ID": SESSION},
    )
    l2.to_netcdf(folder / f"{SESSION}_L2.nc")
    return TestClient(create_app(str(cp)))


def test_overview_has_inspect_style_session_and_levels(demo: TestClient) -> None:
    root = f"/api/sessions/{SESSION}"
    data = demo.get(root + "/overview").json()
    assert data["session_id"] == SESSION
    assert data["duration_minutes"] > 0
    assert data["start_utc"].startswith("2024-06-20")
    assert data["channels"] == ["532.AN", "532.PC"]
    assert [row["level"] for row in data["levels"] if row["available"]] == ["L1", "L2"]
    assert data["figures_count"] == 0
    assert demo.get("/api/sessions/invalid/overview").status_code == 404


def test_metadata_lists_file_structure_without_reading_signal_arrays(demo: TestClient) -> None:
    root = f"/api/sessions/{SESSION}"
    l1 = demo.get(root + "/metadata", params={"level": 1}).json()
    assert l1["available"]
    assert l1["file"] == SESSION + "_L1.nc"
    assert any(d["name"] == "atmosphere_time" for d in l1["dimensions"])
    assert any(v["name"] == "Atmospheric_Temperature_K" for v in l1["variables"])
    assert any(a["name"] == "software_version" for a in l1["attributes"])
    l0 = demo.get(root + "/metadata", params={"level": 0}).json()
    assert not l0["available"]
    assert demo.get(root + "/metadata", params={"level": 8}).status_code == 422


def test_atmosphere_profiles_read_saved_time_and_source(demo: TestClient) -> None:
    root = f"/api/sessions/{SESSION}/profiles/atmosphere"
    options = demo.get(root + "/options").json()
    assert options["available"]
    assert options["choice_axis"] == "atmosphere_time"
    assert len(options["choices"]) == 2
    row = demo.get(root, params={"variable": "Atmospheric_Temperature_K",
                                  "index": 1}).json()
    assert row["values"][0] == 291
    assert row["altitude_km"][-1] == 3.0
    assert row["source"] == "ussa76"
    assert demo.get(root, params={"variable": "bad", "index": 0}).status_code == 422
    assert demo.get(root, params={"variable": "Atmospheric_Temperature_K",
                                  "index": 9}).status_code == 422


def test_l2_profiles_plot_saved_wavelength_profile_only(demo: TestClient) -> None:
    root = f"/api/sessions/{SESSION}/profiles/optical"
    opts = demo.get(root + "/options").json()
    assert opts["available"]
    assert opts["choices"] == ["355", "532"]
    assert "aerosol_backscatter_mean" in [x["id"] for x in opts["variables"]]
    row = demo.get(root, params={"variable": "aerosol_backscatter_mean",
                                  "index": 1}).json()
    assert row["values"][0] == 7.0
    assert row["units"] == "m-1 sr-1"


def test_qa_uses_only_saved_named_diagnostics(demo: TestClient) -> None:
    root = f"/api/sessions/{SESSION}/qa"
    choices = demo.get(root + "/options").json()["metrics"]
    assert any(x["variable"] == "background_outlier_fraction" for x in choices)
    assert any(x["variable"] == "retrieval_success_flag" for x in choices)
    l1 = demo.get(root, params={"level": 1, "variable": "background_outlier_fraction",
                                 "index": 1}).json()
    assert l1["values"] == [0.02, 0.06]
    assert l1["channel"] == "532.PC"
    l2 = demo.get(root, params={"level": 2, "variable": "retrieval_success_flag",
                                 "index": 0}).json()
    assert l2["values"] == [1.0, 1.0]
    assert demo.get(root, params={"level": 2, "variable": "aerosol_backscatter_mc_mean",
                                  "index": 0}).status_code == 422
    assert demo.get(root, params={"level": 1, "variable": "background_outlier_fraction",
                                  "index": 50}).status_code == 422
