"""Scientific and IO tests for MILGRAU thermodynamic atmosphere sources."""

from __future__ import annotations

from copy import deepcopy
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from milgrau.config.loader import load_config
from milgrau.io.era5 import (
    build_era5_request,
    era5_config,
    era5_profile_from_dataset,
    fetch_era5_pressure_level_profile,
    nearest_era5_analysis_hour,
)
from milgrau.io.radiosonde import select_radiosonde_target_datetime
from milgrau.level1.thermodynamics import integrate_thermodynamics
from milgrau.level2.retrieval import build_thermodynamic_profile
from milgrau.physics.atmosphere import get_standard_atmosphere


def _era5_settings() -> dict:
    return {
        "cache_dir": ".cache/test-era5",
        "dataset": "reanalysis-era5-pressure-levels",
        "pressure_levels_hpa": [1000, 500, 100],
        "grid_deg": 0.25,
        "area_half_width_deg": 0.25,
    }


def _config_with_priority(*sources: str, extension: str = "ussa76") -> dict:
    config = deepcopy(load_config("config.yaml"))
    atmosphere = config["level1"]["atmosphere"]
    atmosphere["source_priority"] = list(sources)
    atmosphere["external_profile_outside_coverage"] = extension
    if "radiosonde" not in sources:
        atmosphere.pop("radiosonde", None)
    if "era5" not in sources:
        atmosphere.pop("era5", None)
    return config


def test_ussa76_fallback_uses_stratified_layers() -> None:
    altitude_m = np.array([0.0, 11_000.0, 15_000.0, 20_000.0, 25_000.0, 32_000.0, 47_000.0])
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_m)
    assert np.isclose(pressure_hpa[0], 1013.25, rtol=0.0, atol=1e-8)
    assert np.isclose(temperature_k[0], 288.15, rtol=0.0, atol=1e-8)
    assert np.all(np.diff(pressure_hpa) < 0.0)
    assert 216.0 < temperature_k[2] < 218.0
    assert temperature_k[4] > 218.0
    assert temperature_k[-1] > 265.0


def test_ussa76_rejects_silent_extrapolation_above_supported_domain() -> None:
    with pytest.raises(ValueError, match="84"):
        get_standard_atmosphere(np.array([0.0, 100_000.0]))


def test_radiosonde_nearest_selection_uses_explicit_synoptic_hours() -> None:
    measurement = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
    target = select_radiosonde_target_datetime(
        measurement,
        synoptic_hours_utc=[0, 12],
        selection="nearest",
        max_time_delta_hours=6.0,
    )
    assert target == datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)


def test_radiosonde_selection_respects_explicit_maximum_time_delta() -> None:
    measurement = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    target = select_radiosonde_target_datetime(
        measurement,
        synoptic_hours_utc=[0],
        selection="nearest",
        max_time_delta_hours=6.0,
    )
    assert target is None


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
        _era5_settings(),
    )
    assert dataset == "reanalysis-era5-pressure-levels"
    assert request["variable"] == ["temperature", "geopotential"]
    assert request["pressure_level"] == ["1000", "500", "100"]
    assert request["data_format"] == "netcdf"
    assert request["download_format"] == "unarchived"


def test_era5_configuration_does_not_replace_invalid_pressure_levels_with_defaults() -> None:
    settings = _era5_settings()
    settings["pressure_levels_hpa"] = ["bad"]
    with pytest.raises(ValueError, match="pressure_levels_hpa"):
        era5_config(settings)

    settings = _era5_settings()
    del settings["pressure_levels_hpa"]
    with pytest.raises(KeyError, match="pressure_levels_hpa"):
        era5_config(settings)


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


def test_era5_fetch_rejects_incomplete_settings_before_network_access(tmp_path) -> None:
    with pytest.raises(KeyError, match="pressure_levels_hpa"):
        fetch_era5_pressure_level_profile(
            datetime(2024, 6, 10, 12, tzinfo=timezone.utc),
            -23.56,
            -46.73,
            logging.getLogger("test-era5-strict"),
            settings={
                "cache_dir": str(tmp_path),
                "dataset": "reanalysis-era5-pressure-levels",
                "grid_deg": 0.25,
                "area_half_width_deg": 0.25,
            },
        )


def _level1_shell() -> xr.Dataset:
    return xr.Dataset(
        coords={
            "time": pd.date_range("2024-06-10T12:00:00", periods=3, freq="10min"),
            "altitude": np.arange(0.0, 20_000.0, 500.0),
        }
    )


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


def test_level1_reads_radiosonde_station_identity_only_from_station_catalog(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_radio(_dt, station_id, _logger, **kwargs):
        captured["station_id"] = station_id
        captured.update(kwargs)
        return _profile("radiosonde")

    monkeypatch.setattr("milgrau.level1.thermodynamics.fetch_wyoming_radiosonde", fake_radio)
    config = _config_with_priority("radiosonde")
    result = integrate_thermodynamics(_level1_shell(), config, logging.getLogger("test-radio-id"))

    assert captured["station_id"] == "83779"
    assert captured["selection"] == "nearest"
    assert captured["synoptic_hours_utc"] == [0, 12]
    assert result.attrs["thermodynamic_profile_source_type"] == "radiosonde"


def test_level1_uses_era5_only_after_radiosonde_failure(monkeypatch) -> None:
    monkeypatch.setattr("milgrau.level1.thermodynamics.fetch_wyoming_radiosonde", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "milgrau.level1.thermodynamics.fetch_era5_pressure_level_profile",
        lambda *args, **kwargs: _profile("era5"),
    )
    config = _config_with_priority("radiosonde", "era5", "ussa76")
    result = integrate_thermodynamics(_level1_shell(), config, logging.getLogger("test-era5"))

    assert result.attrs["radiosonde_available"] == "false"
    assert result.attrs["thermodynamic_profile_source_type"] == "era5"
    assert result.attrs["thermodynamic_profile_doi"] == "test-doi"
    assert result.attrs["thermodynamic_profile_available"] == "true"
    assert result.attrs["thermodynamic_source_attempts"] == "radiosonde,era5"
    assert 0.0 < float(result.attrs["thermodynamic_profile_standard_fallback_fraction"]) < 1.0
    assert result["Atmospheric_Temperature_K"].dims == ("altitude",)
    assert result["Atmospheric_Pressure_hPa"].dims == ("altitude",)


def test_level1_does_not_call_era5_when_source_policy_omits_it(monkeypatch) -> None:
    monkeypatch.setattr("milgrau.level1.thermodynamics.fetch_wyoming_radiosonde", lambda *args, **kwargs: None)

    def forbidden_era5(*args, **kwargs):
        raise AssertionError("ERA5 must not be called when absent from source_priority")

    monkeypatch.setattr("milgrau.level1.thermodynamics.fetch_era5_pressure_level_profile", forbidden_era5)
    config = _config_with_priority("radiosonde", "ussa76")
    result = integrate_thermodynamics(_level1_shell(), config, logging.getLogger("test-policy"))
    assert result.attrs["thermodynamic_profile_source_type"] == "ussa76"
    assert result.attrs["thermodynamic_source_attempts"] == "radiosonde,ussa76"


def test_level1_materializes_ussa76_only_when_explicitly_listed(monkeypatch) -> None:
    monkeypatch.setattr("milgrau.level1.thermodynamics.fetch_wyoming_radiosonde", lambda *args, **kwargs: None)
    config = _config_with_priority("radiosonde", "ussa76")
    result = integrate_thermodynamics(_level1_shell(), config, logging.getLogger("test-ussa76"))

    assert result.attrs["thermodynamic_profile_source_type"] == "ussa76"
    assert result.attrs["thermodynamic_profile_available"] == "true"
    assert float(result.attrs["thermodynamic_profile_standard_fallback_fraction"]) == 1.0
    assert np.all(np.isfinite(result["Atmospheric_Temperature_K"].values))
    assert np.all(np.isfinite(result["Atmospheric_Pressure_hPa"].values))


def test_level1_fails_when_configured_atmosphere_sources_are_exhausted(monkeypatch) -> None:
    monkeypatch.setattr("milgrau.level1.thermodynamics.fetch_wyoming_radiosonde", lambda *args, **kwargs: None)
    config = _config_with_priority("radiosonde")

    with pytest.raises(RuntimeError, match="source policy was exhausted"):
        integrate_thermodynamics(_level1_shell(), config, logging.getLogger("test-exhausted"))


def test_level1_external_profile_extension_can_be_configured_to_fail(monkeypatch) -> None:
    monkeypatch.setattr(
        "milgrau.level1.thermodynamics.fetch_wyoming_radiosonde",
        lambda *args, **kwargs: _profile("radiosonde"),
    )
    config = _config_with_priority("radiosonde", extension="fail")

    with pytest.raises(RuntimeError, match="source policy was exhausted"):
        integrate_thermodynamics(_level1_shell(), config, logging.getLogger("test-extension-fail"))


def test_level2_reads_only_materialized_level1_atmosphere() -> None:
    ds = _level1_shell()
    altitude = np.asarray(ds["altitude"].values, dtype=np.float64)
    pressure, temperature = get_standard_atmosphere(altitude + 760.0)
    ds["Atmospheric_Temperature_K"] = (("altitude",), temperature)
    ds["Atmospheric_Pressure_hPa"] = (("altitude",), pressure)
    ds.attrs["thermodynamic_profile_source_type"] = "ussa76"
    observed_pressure, observed_temperature, source = build_thermodynamic_profile(ds, altitude, {})
    assert source == "ussa76"
    assert np.array_equal(observed_pressure, pressure)
    assert np.array_equal(observed_temperature, temperature)


def test_level2_rejects_old_level1_without_canonical_atmosphere() -> None:
    ds = _level1_shell()
    altitude = np.asarray(ds["altitude"].values, dtype=np.float64)
    with pytest.raises(KeyError, match="reprocess Level 1"):
        build_thermodynamic_profile(ds, altitude, {})
