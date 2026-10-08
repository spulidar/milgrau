"""Tests for strict productive Level 0 configuration and missing-data policies."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from milgrau.config.loader import load_config
from milgrau.level0.config import (
    Level0ConfigurationError,
    resolve_level0_config,
    station_coordinates,
    station_timezone,
)
from milgrau.level0.processing import _weather_for_interval, fetch_group_weather


def _minimal_config() -> dict:
    return {
        "directories": {
            "raw_data": "01-data",
            "processed_data": "02-processed_data",
            "log_dir": "logs",
        },
        "processing": {
            "spurious_extensions": [".zip"],
            "raw_scan_ignore_dirs": [],
            "quarantine_dir": "quarantine",
        },
        "level0": {
            "acquisition_qa": {
                "laser_shot_tolerance_fraction": 0.002,
                "licel_header_time_jitter_s": 1.0,
            },
            "session": {"max_gap_seconds": 1800.0},
            "solar_regime": {"day_night_threshold_deg": -3.0},
            "dark_current": {"max_association_hours": 12.0},
            "surface_weather": {"missing_policy": "nan"},
        },
    }


def test_repository_level0_recipe_is_explicit() -> None:
    config = load_config("config.yaml")
    resolved = resolve_level0_config(config)

    assert resolved.directories.raw_data == "01-data"
    assert resolved.directories.processed_data == "02-processed_data"
    assert resolved.directories.log_dir == "logs"
    assert resolved.discovery.spurious_extensions == (".dpp", ".zip", ".txt", ".json", ".7z", ".tar")
    assert resolved.discovery.raw_scan_ignore_dirs == ()
    assert resolved.discovery.quarantine_dir == "quarantine"
    assert resolved.acquisition_qa.laser_shot_tolerance_fraction == 0.002
    assert resolved.acquisition_qa.licel_header_time_jitter_s == 1.0
    assert resolved.session.max_gap_seconds == 1800.0
    assert resolved.solar_regime.day_night_threshold_deg == -3.0
    assert resolved.dark_current.max_association_hours == 12.0
    assert resolved.surface_weather.missing_policy == "nan"
    assert "physics" not in config


def test_level0_recipe_rejects_missing_and_unknown_fields() -> None:
    missing = _minimal_config()
    del missing["level0"]["dark_current"]
    with pytest.raises(Level0ConfigurationError, match="dark_current"):
        resolve_level0_config(missing)

    unknown = _minimal_config()
    unknown["level0"]["surface_weather"]["fallback_temperature_c"] = 25.0
    with pytest.raises(Level0ConfigurationError, match="unknown"):
        resolve_level0_config(unknown)


def test_level0_requires_all_productive_directories() -> None:
    config = _minimal_config()
    del config["directories"]["log_dir"]
    with pytest.raises(Level0ConfigurationError, match=r"directories\.log_dir"):
        resolve_level0_config(config)


def test_level0_requires_raw_discovery_policy_without_defaults() -> None:
    for key in ("spurious_extensions", "raw_scan_ignore_dirs", "quarantine_dir"):
        config = _minimal_config()
        del config["processing"][key]
        with pytest.raises(Level0ConfigurationError, match=rf"processing\.{key}"):
            resolve_level0_config(config)


def test_level0_rejects_invalid_spurious_extension_and_ignore_path() -> None:
    invalid_extension = _minimal_config()
    invalid_extension["processing"]["spurious_extensions"] = ["zip"]
    with pytest.raises(Level0ConfigurationError, match="file suffixes"):
        resolve_level0_config(invalid_extension)

    invalid_ignore = _minimal_config()
    invalid_ignore["processing"]["raw_scan_ignore_dirs"] = ["nested/cache"]
    with pytest.raises(Level0ConfigurationError, match="basenames"):
        resolve_level0_config(invalid_ignore)


def test_level0_weather_policy_is_explicit() -> None:
    invalid = _minimal_config()
    invalid["level0"]["surface_weather"]["missing_policy"] = "use_defaults"
    with pytest.raises(Level0ConfigurationError, match="nan.*fail"):
        resolve_level0_config(invalid)


def test_station_timezone_and_coordinates_require_station_catalog() -> None:
    with pytest.raises(Level0ConfigurationError, match="_station_catalog"):
        station_timezone(_minimal_config())
    with pytest.raises(Level0ConfigurationError, match="_station_catalog"):
        station_coordinates(_minimal_config())


def test_repository_station_identity_is_used_without_site_fallbacks() -> None:
    config = load_config("config.yaml")
    assert station_timezone(config) == "America/Sao_Paulo"
    latitude, longitude = station_coordinates(config)
    assert np.isfinite(latitude)
    assert np.isfinite(longitude)


def _weather_config(policy: str) -> dict:
    config = _minimal_config()
    config["level0"]["surface_weather"]["missing_policy"] = policy
    config["_station_catalog"] = {
        "station": {
            "site": {"latitude": -23.56, "longitude": -46.73},
        }
    }
    return config


def _group() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "meas_type": ["measurements"],
            "start_time_utc": [pd.Timestamp("2024-01-01T12:05:00Z")],
            "stop_time": [pd.Timestamp("2024-01-01T12:55:00Z")],
        }
    )


def test_missing_surface_weather_nan_policy_does_not_invent_temperature_or_pressure(monkeypatch) -> None:
    monkeypatch.setattr("milgrau.level0.processing.fetch_surface_weather", lambda *args, **kwargs: None)
    result = fetch_group_weather(_group(), _weather_config("nan"), logging.getLogger("test-weather-nan"))

    assert result["weather_time"].shape == (2,)
    assert np.isnan(result["temperature_c"]).all()
    assert np.isnan(result["pressure_hpa"]).all()


def test_missing_surface_weather_fail_policy_stops_processing(monkeypatch) -> None:
    monkeypatch.setattr("milgrau.level0.processing.fetch_surface_weather", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="missing_policy='fail'"):
        fetch_group_weather(_group(), _weather_config("fail"), logging.getLogger("test-weather-fail"))



def test_scc_weather_subset_uses_segment_hour_bracket() -> None:
    weather = {
        "weather_time": np.array(
            [
                "2025-01-01T18:00:00",
                "2025-01-01T19:00:00",
                "2025-01-01T20:00:00",
                "2025-01-01T21:00:00",
            ],
            dtype="datetime64[ns]",
        ),
        "temperature_c": np.array([25.0, 24.0, 22.0, 21.0]),
        "pressure_hpa": np.array([930.0, 931.0, 932.0, 933.0]),
        "relative_humidity_percent": np.array([50.0, 55.0, 60.0, 65.0]),
        "cloud_cover_percent": np.array([10.0, 20.0, 30.0, 40.0]),
        "wind_speed_kmh": np.array([4.0, 5.0, 6.0, 7.0]),
        "source": "synthetic",
        "cadence": "hourly",
    }

    result = _weather_for_interval(
        weather,
        pd.Timestamp("2025-01-01T19:10:00Z"),
        pd.Timestamp("2025-01-01T19:50:00Z"),
    )

    assert result["weather_time"].tolist() == [
        np.datetime64("2025-01-01T19:00:00", "ns"),
        np.datetime64("2025-01-01T20:00:00", "ns"),
    ]
    np.testing.assert_allclose(result["temperature_c"], [24.0, 22.0])
    np.testing.assert_allclose(result["pressure_hpa"], [931.0, 932.0])
