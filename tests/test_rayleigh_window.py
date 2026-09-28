"""Regression tests for physical Rayleigh reference-window geometry."""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.level2.rayleigh_window import rayleigh_window_bins


def test_one_km_rayleigh_window_preserves_current_spu_width() -> None:
    altitude_m = np.arange(0.0, 30_000.0, 7.5)

    assert rayleigh_window_bins(altitude_m, 1000.0) == 133


def test_rayleigh_window_is_grid_independent_in_physical_width() -> None:
    altitude_m = np.arange(0.0, 30_000.0, 15.0)

    assert rayleigh_window_bins(altitude_m, 1000.0) == 67


def test_rayleigh_window_rejects_nonuniform_altitude_grid() -> None:
    altitude_m = np.array([0.0, 7.5, 15.0, 30.0])

    with pytest.raises(ValueError, match="uniform"):
        rayleigh_window_bins(altitude_m, 1000.0)


def test_rayleigh_window_does_not_silently_widen_too_narrow_width() -> None:
    altitude_m = np.arange(0.0, 1000.0, 100.0)

    with pytest.raises(ValueError, match="fewer than three bins"):
        rayleigh_window_bins(altitude_m, 150.0)
