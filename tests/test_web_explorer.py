"""Synthetic read-only Web Explorer tests, independent of Streamlit."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml
from fastapi.testclient import TestClient

from milgrau.web.app import create_app
from milgrau.web.catalog import Catalog
from milgrau.web.rcs import rcs_map

SESSION = "spu_20240620-2103Z_20240621-0848Z"


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project / "config.yaml").read_text(encoding="utf-8"))
    config["station_config"] = str((project / "station.yaml").resolve())
    output = tmp_path / "processed"
    config["directories"]["processed_data"] = str(output)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    folder = output / "spu" / "2024" / "06" / SESSION
    folder.mkdir(parents=True)
    times = np.array([
        "2024-06-20T23:40", "2024-06-20T23:45",
        "2024-06-21T03:00", "2024-06-21T03:05",
    ], dtype="datetime64[m]")
    signal = np.ones((4, 2, 7), dtype=np.float32)
    signal[:, 1, :] *= 5.0
    signal[0, 0, 2] = np.nan
    signal[1, 0, 2] = np.nan
    signal[2, 0, 3] = 12.0
    ds = xr.Dataset(
        {"range_corrected_signal": (
            ("time", "channel", "altitude"), signal,
            {"units": "a.u."}
        )},
        coords={
            "time": times,
            "channel": ["532.AN", "355.PC"],
            "altitude": np.arange(7, dtype=float) * 250.0,
        },
        attrs={"Session_ID": SESSION},
    )
    source = folder / f"{SESSION}_L1.nc"
    ds.to_netcdf(source)
    (folder / f"{SESSION}_night_L1_scc.nc").write_bytes(source.read_bytes())
    (folder / f"{SESSION}_L0.nc").touch()
    return config_path, source, folder


def test_catalog_reads_config_and_preserves_cross_midnight_session(prepared) -> None:
    config_path, _source, _folder = prepared
    catalog = Catalog(config_path)
    assert catalog.site["name"] == "SPU-Lidar"
    assert catalog.dates() == ["2024-06-21", "2024-06-20"]
    assert [row["id"] for row in catalog.all_sessions()] == [SESSION]
    from datetime import date
    assert catalog.sessions(date(2024, 6, 20))[0]["id"] == SESSION
    assert catalog.sessions(date(2024, 6, 21))[0]["id"] == SESSION
    products = catalog.products(SESSION)
    assert len(products) == 3
    assert sum(p["level"] == 1 for p in products) == 2
    assert all("_path" not in p for p in products)
    assert any(p["variant"] == "night" and p["scc"] for p in products)


def test_rcs_returns_bounded_values_with_real_gaps_and_nan(prepared) -> None:
    _config, path, _folder = prepared
    result = rcs_map(path, "532.AN", max_time=4, max_altitude=7)
    assert result["samples"]["time"] == 4
    assert len(result["time_utc"]) == 6  # two real columns plus two gap markers
    assert result["rcs"][3][2] is None
    assert result["rcs"][3][3] is None
    assert result["rcs"][2][0] is None
    assert result["rcs"][3][4] == 12.0
    assert result["mean_profile"][3] == pytest.approx(3.75)
    assert result["altitude_km_agl"][-1] == 1.5


def test_rcs_peak_reduction_is_labeled_and_preserves_extrema(prepared) -> None:
    _config, path, _folder = prepared
    result = rcs_map(path, "532.AN", statistic="max", max_time=2, max_altitude=3)
    assert result["statistic"] == "max"
    assert result["samples"]["display_altitude"] <= 3
    assert any(
        value == 12.0 for row in result["rcs"] for value in row
        if value is not None
    )
    with pytest.raises(ValueError, match="Canal"):
        rcs_map(path, "unknown")


def test_api_serves_ui_catalog_and_selected_l1_data(prepared) -> None:
    config_path, _path, _folder = prepared
    client = TestClient(create_app(str(config_path)))
    assert client.get("/").status_code == 200
    assert "Plotly.js" in client.get("/").text
    assert client.get("/static/js/app.js").status_code == 200
    assert client.get("/api/site").json()["id"] == "spu"
    # New header selects full scientific sessions, not a publication date.
    assert [row["id"] for row in client.get("/api/sessions").json()["sessions"]] == [SESSION]
    assert client.get("/api/dates").json()["dates"] == ["2024-06-21", "2024-06-20"]
    sessions = client.get("/api/sessions", params={"date": "2024-06-20"}).json()
    assert [s["id"] for s in sessions["sessions"]] == [SESSION]
    product_id = f"{SESSION}_L1.nc"
    root = f"/api/sessions/{SESSION}"
    opts = client.get(root + "/options", params={"product": product_id})
    assert opts.status_code == 200
    assert opts.json()["channels"] == ["532.AN", "355.PC"]
    result = client.get(root + "/rcs", params={
        "product": product_id, "channel": "355.PC"
    })
    assert result.status_code == 200
    assert result.json()["mean_profile"][0] == 5.0


def test_api_never_accepts_unregistered_product_paths(prepared) -> None:
    config_path, _source, _folder = prepared
    client = TestClient(create_app(str(config_path)))
    root = f"/api/sessions/{SESSION}"
    assert client.get(root + "/rcs", params={
        "product": "../../station.yaml", "channel": "532.AN",
    }).status_code == 404
    assert client.get(root + "/options", params={
        "product": "other_file_L1.nc",
    }).status_code == 404
    assert client.get("/api/sessions/other-session/products").status_code == 404
    assert client.get(root + "/rcs", params={
        "product": f"{SESSION}_L1.nc", "channel": "532.AN", "statistic": "bad",
    }).status_code == 422




def test_site_serves_only_configured_local_logos(prepared) -> None:
    config_path, _path, _folder = prepared
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    logo_dir = config_path.parent / "img"
    logo_dir.mkdir()
    (logo_dir / "SPU_LOGO.png").write_bytes(b"\\x89PNG\\r\\n\\x1a\\n")
    config["visualization"]["logos"] = [{"path": "img/SPU_LOGO.png"}]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    client = TestClient(create_app(str(config_path)))
    logos = client.get("/api/site").json()["logos"]
    assert len(logos) == 1
    assert logos[0]["url"] == "/api/logo/0"
    assert client.get(logos[0]["url"]).status_code == 200
    assert client.get("/api/logo/1").status_code == 404
    assert client.get("/api/logo/-1").status_code == 404


def test_all_session_order_is_newest_first(prepared) -> None:
    config_path, source, folder = prepared
    new_sid = "spu_20240920-1300Z_20240920-1400Z"
    new_folder = folder.parents[1] / "09" / new_sid
    new_folder.mkdir(parents=True)
    (new_folder / f"{new_sid}_L1.nc").write_bytes(source.read_bytes())
    client = TestClient(create_app(str(config_path)))
    sessions = client.get("/api/sessions").json()["sessions"]
    assert [entry["id"] for entry in sessions] == [new_sid, SESSION]

def test_cli_starts_web_server_without_streamlit(monkeypatch) -> None:
    from milgrau.cli import explorer
    monkeypatch.setattr("sys.argv", ["milgrau-explorer"])
    seen = {}
    monkeypatch.setattr(explorer, "create_app", lambda config: (seen.update(config=config), object())[1])
    monkeypatch.setattr(explorer.uvicorn, "run", lambda app, **kw: seen.update(kw))
    assert explorer.main() == 0
    assert seen["config"] == "config.yaml"
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 8765
