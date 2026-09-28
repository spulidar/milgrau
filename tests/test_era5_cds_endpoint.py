"""Regression tests for ERA5 Climate Data Store routing and concise failures."""

from __future__ import annotations

import logging
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from milgrau.io.era5 import CDS_API_URL, fetch_era5_pressure_level_profile


def _settings() -> dict:
    return {
        "cache_dir": "cache/era5",
        "dataset": "reanalysis-era5-pressure-levels",
        "pressure_levels_hpa": [1000, 500, 100],
        "grid_deg": 0.25,
        "area_half_width_deg": 0.25,
    }


def _write_fake_era5(path: str | Path) -> None:
    pressure = np.array([1000.0, 500.0, 100.0])
    temperature = np.array([290.0, 255.0, 215.0])
    geopotential_height = np.array([100.0, 5500.0, 16000.0])
    ds = xr.Dataset(
        data_vars={
            "t": (("valid_time", "pressure_level", "latitude", "longitude"), temperature[None, :, None, None]),
            "z": (("valid_time", "pressure_level", "latitude", "longitude"), (geopotential_height * 9.80665)[None, :, None, None]),
        },
        coords={
            "valid_time": np.array([np.datetime64("2025-05-10T02:00:00")]),
            "pressure_level": pressure,
            "latitude": np.array([-23.5]),
            "longitude": np.array([-46.75]),
        },
    )
    ds.to_netcdf(path)


def test_era5_client_is_pinned_to_climate_data_store_even_if_environment_points_to_ads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, *, url: str, quiet: bool) -> None:
            captured["url"] = url
            captured["quiet"] = quiet

        def retrieve(self, dataset: str, request: dict, target: str) -> None:
            captured["dataset"] = dataset
            captured["request"] = request
            _write_fake_era5(target)

    monkeypatch.setenv("CDSAPI_URL", "https://ads.atmosphere.copernicus.eu/api")
    monkeypatch.setitem(sys.modules, "cdsapi", types.SimpleNamespace(Client=FakeClient))

    frame = fetch_era5_pressure_level_profile(
        datetime(2025, 5, 10, 2, tzinfo=timezone.utc),
        -23.5607,
        -46.7398,
        logging.getLogger("test-era5-cds"),
        settings=_settings(),
        root_dir=tmp_path,
    )

    assert frame is not None and not frame.empty
    assert captured["url"] == CDS_API_URL
    assert captured["quiet"] is True
    assert captured["dataset"] == "reanalysis-era5-pressure-levels"


def test_era5_failure_warning_is_one_operator_line(tmp_path: Path, monkeypatch, caplog) -> None:
    class FailingClient:
        def __init__(self, *, url: str, quiet: bool) -> None:
            assert url == CDS_API_URL
            assert quiet is True

        def retrieve(self, dataset: str, request: dict, target: str) -> None:
            raise RuntimeError("404 Client Error\nprocess not found\ndataset not found")

    monkeypatch.setitem(sys.modules, "cdsapi", types.SimpleNamespace(Client=FailingClient))
    logger = logging.getLogger("test-era5-compact")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        frame = fetch_era5_pressure_level_profile(
            datetime(2025, 5, 10, 2, tzinfo=timezone.utc),
            -23.5607,
            -46.7398,
            logger,
            settings=_settings(),
            root_dir=tmp_path,
        )

    assert frame is None
    warning = next(record.getMessage() for record in caplog.records if record.levelno == logging.WARNING)
    assert warning == "ERA5 unavailable | 404 Client Error"
    assert "\n" not in warning
