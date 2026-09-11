"""Tests for strict productive Level 1 configuration and calibration resolution."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from milgrau.config.loader import load_config
from milgrau.level1.config import (
    Level1ConfigurationError,
    resolve_channel_calibration,
    resolve_level1_config,
    resolve_station_site,
)
from milgrau.level1.pbl import estimate_pbl_timeseries


def _level1_recipe() -> dict:
    return {
        "level1": {
            "background": {"start_altitude_m": 29_000.0, "stop_altitude_m": 29_999.0},
            "photon_counting": {"deadtime_min_denominator": 0.05},
            "pbl": {
                "reference_channel": "532.AN",
                "min_search_altitude_m": 500.0,
                "max_search_altitude_m": 4000.0,
                "smooth_bins": 15,
            },
            "atmosphere": {
                "source_priority": ["ussa76"],
                "external_profile_outside_coverage": "ussa76",
            },
        }
    }


def test_repository_level1_recipe_is_explicit_and_speed_of_light_is_not_yaml_configurable() -> None:
    config = load_config("config.yaml")
    resolved = resolve_level1_config(config)

    assert resolved.background.start_altitude_m == 29_000.0
    assert resolved.background.stop_altitude_m == 29_999.0
    assert resolved.photon_counting.deadtime_min_denominator == 0.05
    assert resolved.pbl.reference_channel == "532.AN"
    assert resolved.atmosphere.source_priority == ("radiosonde", "era5", "ussa76")
    assert resolved.atmosphere.external_profile_outside_coverage == "ussa76"
    assert resolved.atmosphere.radiosonde is not None
    assert resolved.atmosphere.radiosonde.synoptic_hours_utc == (0, 12)
    assert resolved.atmosphere.radiosonde.selection == "nearest"
    assert resolved.atmosphere.radiosonde.max_time_delta_hours == 6.0
    assert resolved.atmosphere.era5 is not None
    assert len(resolved.atmosphere.era5.pressure_levels_hpa) > 1
    assert "speed_of_light_m_s" not in config["physics"]
    assert "speed_of_light" not in config["physics"]
    assert "background_start_m" not in config["physics"]
    assert "pbl_min_search_m" not in config["physics"]


def test_level1_recipe_rejects_missing_required_sections() -> None:
    config = _level1_recipe()
    del config["level1"]["photon_counting"]

    with pytest.raises(Level1ConfigurationError, match="photon_counting"):
        resolve_level1_config(config)


def test_level1_recipe_rejects_unknown_keys_and_even_pbl_smoothing() -> None:
    unknown = _level1_recipe()
    unknown["level1"]["background"]["fallback_m"] = 30_000.0
    with pytest.raises(Level1ConfigurationError, match="unknown"):
        resolve_level1_config(unknown)

    even = _level1_recipe()
    even["level1"]["pbl"]["smooth_bins"] = 14
    with pytest.raises(Level1ConfigurationError, match="odd"):
        resolve_level1_config(even)


def test_atmosphere_policy_requires_complete_radiosonde_settings_when_selected() -> None:
    config = _level1_recipe()
    config["level1"]["atmosphere"]["source_priority"] = ["radiosonde", "ussa76"]

    with pytest.raises(Level1ConfigurationError, match="radiosonde"):
        resolve_level1_config(config)


def test_atmosphere_policy_requires_complete_era5_settings_when_selected() -> None:
    config = _level1_recipe()
    config["level1"]["atmosphere"].update(
        {
            "source_priority": ["era5", "ussa76"],
            "era5": {
                "cache_dir": "cache",
                "dataset": "reanalysis-era5-pressure-levels",
                "grid_deg": 0.25,
                "area_half_width_deg": 0.25,
            },
        }
    )

    with pytest.raises(Level1ConfigurationError, match="pressure_levels_hpa"):
        resolve_level1_config(config)


def test_atmosphere_policy_rejects_unlisted_dormant_source_configuration() -> None:
    config = _level1_recipe()
    config["level1"]["atmosphere"]["radiosonde"] = {
        "cache_dir": "cache",
        "synoptic_hours_utc": [0, 12],
        "selection": "nearest",
        "max_time_delta_hours": 6.0,
    }

    with pytest.raises(Level1ConfigurationError, match="absent from source_priority"):
        resolve_level1_config(config)


def test_repository_pc_calibration_resolves_not_characterized_without_inventing_rate() -> None:
    config = load_config("config.yaml")
    ds = xr.Dataset(coords={"time": pd.date_range("2025-01-01", periods=1)})
    ds.attrs["Station_Profile"] = "spu-merionc-2024"

    calibration = resolve_channel_calibration(config, ds, "532.PC")

    assert calibration.calibration_id == "spu-channel-corrections-v1"
    assert calibration.detector_mode == "photon_counting"
    assert calibration.saturation_status == "not_characterized"
    assert calibration.saturation_max_rate_mhz is None
    assert calibration.saturation_characterized is False


def test_channel_calibration_resolution_rejects_unknown_channel() -> None:
    config = load_config("config.yaml")
    ds = xr.Dataset(coords={"time": pd.date_range("2025-01-01", periods=1)})
    ds.attrs["Station_Profile"] = "spu-merionc-2024"

    with pytest.raises(Level1ConfigurationError, match="no calibration"):
        resolve_channel_calibration(config, ds, "999.PC")


def test_station_site_resolution_uses_historical_profile_altitude() -> None:
    config = load_config("config.yaml")
    old = xr.Dataset(coords={"time": pd.date_range("2024-09-09", periods=1)})
    new = xr.Dataset(coords={"time": pd.date_range("2024-09-10", periods=1)})

    assert resolve_station_site(config, old)["station_altitude_m"] == 766.0
    assert resolve_station_site(config, new)["station_altitude_m"] == 740.0


def test_pbl_does_not_substitute_another_channel_when_reference_is_missing() -> None:
    config = _level1_recipe()
    time = pd.date_range("2024-01-01", periods=2)
    altitude = np.arange(0.0, 5000.0, 7.5)
    ds = xr.Dataset(
        {
            "range_corrected_signal": (
                ("time", "channel", "altitude"),
                np.ones((2, 1, altitude.size), dtype=np.float64),
            ),
            "channel_correction_success": (("channel",), np.array([1], dtype=np.int8)),
        },
        coords={"time": time, "channel": ["355.AN"], "altitude": altitude},
    )

    result = estimate_pbl_timeseries(ds, altitude, config, logging.getLogger("test-pbl-strict"))

    assert "PBL_Height_km" not in result
