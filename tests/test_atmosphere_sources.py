"""Scientific and IO tests for MILGRAU thermodynamic atmosphere sources."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from milgrau.io.era5 import (
    build_era5_request,
    era5_profile_from_dataset,
    fetch_era5_pressure_level_profile,
    nearest_era5_analysis_hour,
)
from milgrau.level1.thermodynamics import integrate_thermodynamics
from milgrau.level2.atmosphere import get_standard_atmosphere


def test_ussa76_fallback_uses_stratified_layers() -> None:
    altitude_m = np.array([0.0, 11_000.0, 15_000.0, 20_000.0, 25_000.0, 32_000.0, 47_000.0])
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_m)

    assert np.isclose(pressure_hpa[0], 1013.25, rtol=0.0, atol=1e-8)
    assert np.isclose(temperature_k[0], 288.15, rtol=0.0, atol=1e-8)
    assert np.all(np.diff(pressure_hpa) < 0.0)
    assert 216.0 < temperature_k[2] < 218.0
    # A clipped-troposphere fallback would remain at 216.65 K. USSA76 warms
    # through the lower stratosphere, which is the behavior needed for L2.
    assert temperature_k[4] > 218.0
    assert temperature_k[-1] > 265.0


def test_ussa76_rejects_silent_extrapolation_above_supported_domain() -> None:
    with pytest.raises(ValueError, match="84"):
        get_standard_atmosphere(np.array([0.0, 100_000.0]))


def test_era5_nearest_analysis_hour_rounds_at_half_hour() -> None:
    before = datetime(2024, 1, 1, 10, 29, tzinfo=timezone.utc)
    after = datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)
    assert nearest_era5_analysis_hour(before).hour == 10
    assert nearest_era5_analysis_hour(after).hour == 11


def test_era5_request_contains_only_molecular_atmosphere_fields() -> None:
    analysis = datetime(2024, 6, 10, 12, tzinfo=timezone.utc)
    dataset, request = build_era5_request(
        analysis,
        -23.56,
        -46.73,
        config={"era5": {"enabled": True, "pressure_levels_hpa": [1000, 500, 100]}},
    )
    assert dataset == "reanalysis-era5-pressure-levels"
    assert request["variable"] == ["temperature", "geopotential"]
    assert request["pressure_level"] == ["1000", "500", "100"]
    assert request["data_format"] == "netcdf"
    assert request["download_format"] == "unarchived"


def test_era5_dataset_is_standardized_to_geometric_height_temperature_pressure() -> None:
    pressure = np.array([1000.0, 500.0, 100.0])
    temperature = np.array([290.0, 255.0, 215.0])
    geopotential_height = np.array([100.0, 5500.0, 16000.0])
    geopotential = geopotential_height * 9.80665
    ds = xr.Dataset(
        data_vars={
            "t": (("valid_time", "pressure_level", "latitude", "longitude"), temperature[None, :, None, None]),
            "z": (("valid_time", "pressure_level", "latitude", "longitude"), geopotential[None, :, None, None]),
        },
        coords={
            "valid_time": np.array([np.datetime64("2024-06-10T12:00:00")]),
            "pressure_level": pressure,
            "latitude": np.array([-23.5]),
            "longitude": np.array([-46.75]),
        },
    )

    profile = era5_profile_from_dataset(ds, -23.56, -46.73)
    assert list(profile.columns) == ["height", "temperature", "temperature_k", "pressure"]
    assert np.all(np.diff(profile["height"].to_numpy()) > 0.0)
    assert np.allclose(profile["temperature_k"].to_numpy(), temperature)
    assert np.allclose(profile["pressure"].to_numpy(), pressure)
    assert profile["height"].iloc[-1] > geopotential_height[-1]


def test_era5_disabled_never_requires_cdsapi(tmp_path) -> None:
    result = fetch_era5_pressure_level_profile(
        datetime(2024, 6, 10, 12, tzinfo=timezone.utc),
        -23.56,
        -46.73,
        logging.getLogger("test-era5-disabled"),
        config={"era5": {"enabled": False, "cache_dir": str(tmp_path)}},
    )
    assert result is None


def _level1_shell() -> xr.Dataset:
    return xr.Dataset(coords={"time": pd.date_range("2024-06-10T12:00:00", periods=3, freq="10min")})


def _profile(source_type: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "height": [800.0, 5000.0, 10000.0, 16000.0],
            "temperature": [20.0, -10.0, -45.0, -58.0],
            "pressure": [930.0, 540.0, 265.0, 105.0],
        }
    )
    frame.attrs.update(
        {
            "source_type": source_type,
            "source": "synthetic test profile",
            "analysis_datetime_utc": "2024-06-10T12:00:00+00:00",
            "target_datetime_utc": "2024-06-10T12:00:00+00:00",
            "time_delta_hours": 0.0,
            "doi": "test-doi" if source_type == "era5" else "",
        }
    )
    return frame


def test_level1_uses_era5_only_after_radiosonde_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "milgrau.level1.thermodynamics.fetch_wyoming_radiosonde",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "milgrau.level1.thermodynamics.fetch_era5_pressure_level_profile",
        lambda *args, **kwargs: _profile("era5"),
    )
    config = {
        "site": {"latitude": -23.56, "longitude": -46.73},
        "era5": {"enabled": True},
        "radiosonde": {"station_id": "83779"},
    }
    result = integrate_thermodynamics(_level1_shell(), config, logging.getLogger("test-era5"))

    assert result.attrs["radiosonde_available"] == "false"
    assert result.attrs["thermodynamic_profile_source_type"] == "era5"
    assert result.attrs["thermodynamic_profile_doi"] == "test-doi"
    assert "Atmospheric_Temperature_K" in result
    assert "Atmospheric_Pressure_hPa" in result
    # Transitional aliases keep the existing L2 reader operational, while their
    # attrs make clear that these values are not from a radiosonde.
    assert "ERA5" in result["Radiosonde_Temperature_K"].attrs["compatibility_note"]


def test_level1_marks_ussa76_when_external_profiles_are_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "milgrau.level1.thermodynamics.fetch_wyoming_radiosonde",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "milgrau.level1.thermodynamics.fetch_era5_pressure_level_profile",
        lambda *args, **kwargs: None,
    )
    config = {
        "site": {"latitude": -23.56, "longitude": -46.73},
        "era5": {"enabled": True},
        "radiosonde": {"station_id": "83779"},
    }
    result = integrate_thermodynamics(_level1_shell(), config, logging.getLogger("test-ussa76"))
    assert result.attrs["thermodynamic_profile_source_type"] == "ussa76"
    assert result.attrs["thermodynamic_profile_available"] == "false"
    assert "Atmospheric_Temperature_K" not in result
