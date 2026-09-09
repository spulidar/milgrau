"""Fail-fast tests for Level 1 instrument calibration access."""

from __future__ import annotations

import logging

import pytest

from milgrau.level1.common import get_channel_constant


LOGGER = logging.getLogger(__name__)


def test_missing_level1_channel_calibration_is_an_error() -> None:
    with pytest.raises(KeyError, match="Missing required instrument calibration"):
        get_channel_constant({}, "532.PC", LOGGER)


def test_level1_rejects_positional_channel_calibration() -> None:
    with pytest.raises(TypeError, match="named fields"):
        get_channel_constant({"532.PC": [0.0035, -3, 0.0]}, "532.PC", LOGGER)  # type: ignore[arg-type]


def test_level1_accepts_complete_named_channel_calibration() -> None:
    calibration = {
        "532.PC": {
            "deadtime_us": 0.0035,
            "bin_shift_bins": -3,
            "background_offset": 0.0,
        }
    }

    assert get_channel_constant(calibration, "532.PC", LOGGER) == (0.0035, -3, 0.0)


def test_level1_rejects_incomplete_channel_calibration() -> None:
    calibration = {
        "532.PC": {
            "deadtime_us": 0.0035,
            "bin_shift_bins": -3,
        }
    }

    with pytest.raises(ValueError, match="must contain exactly"):
        get_channel_constant(calibration, "532.PC", LOGGER)
