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
        }
    }


def test_repository_level1_recipe_is_explicit_and_speed_of_light_is_not_yaml_configurable() -> None:
    config = load_config("config.yaml")
    resolved = resolve_level1_config(config)

    assert resolved.background.start_altitude_m == 29_000.0
    assert resolved.background.stop_altitude_m == 29_999.0
    assert resolved.photon_counting.deadtime_min_denominator == 0.05
    assert resolved.pbl.reference_channel == "532.AN"
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
