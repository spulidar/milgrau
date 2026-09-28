"""Synthetic tests for PBL gradient diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.config.loader import load_config
from milgrau.level1.pbl import calculate_pbl_height_gradient, estimate_pbl_timeseries


class _ListLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.debug_messages: list[str] = []

    @staticmethod
    def _render(message: str, args: tuple[object, ...]) -> str:
        return message % args if args else message

    def info(self, message: str, *args: object) -> None:
        self.info_messages.append(self._render(message, args))

    def warning(self, message: str, *args: object) -> None:
        self.warning_messages.append(self._render(message, args))

    def debug(self, message: str, *args: object) -> None:
        self.debug_messages.append(self._render(message, args))


def test_pbl_gradient_detects_synthetic_aerosol_drop() -> None:
    """A sharp negative RCS gradient should be detected as PBL height."""
    altitude_m = np.arange(0.0, 5000.0, 7.5)
    pbl_true_m = 1500.0

    transition = 1.0 / (1.0 + np.exp((altitude_m - pbl_true_m) / 80.0))
    rcs_signal = 0.2 + 5.0 * transition

    pbl_km = calculate_pbl_height_gradient(
        rcs_signal=rcs_signal,
        alt_m=altitude_m,
        min_search_m=500.0,
        max_search_m=3000.0,
        smooth_bins=15,
    )

    assert np.isfinite(pbl_km)
    assert abs((pbl_km * 1000.0) - pbl_true_m) < 150.0


def test_pbl_gradient_returns_nan_for_increasing_profile() -> None:
    """Profiles without a negative drop should not return a false PBL."""
    altitude_m = np.arange(0.0, 5000.0, 7.5)
    rcs_signal = 1.0 + altitude_m / altitude_m.max()

    pbl_km = calculate_pbl_height_gradient(
        rcs_signal=rcs_signal,
        alt_m=altitude_m,
        min_search_m=500.0,
        max_search_m=3000.0,
        smooth_bins=15,
    )

    assert np.isnan(pbl_km)


def test_pbl_operator_log_reports_mean_result_not_search_window() -> None:
    config = load_config("config.yaml")
    altitude_m = np.arange(0.0, 5000.0, 7.5)
    transition = 1.0 / (1.0 + np.exp((altitude_m - 1500.0) / 80.0))
    profile = 0.2 + 5.0 * transition
    signal = np.stack([profile, profile, profile], axis=0)[:, None, :]
    ds = xr.Dataset(
        {
            "range_corrected_signal": (("time", "channel", "altitude"), signal),
            "channel_correction_success": (("channel",), np.array([1], dtype=np.int8)),
        },
        coords={
            "time": pd.date_range("2025-01-01", periods=3, freq="5min"),
            "channel": ["532.AN"],
            "altitude": altitude_m,
        },
    )
    logger = _ListLogger()

    result = estimate_pbl_timeseries(ds, altitude_m, config, logger)  # type: ignore[arg-type]

    assert "PBL_Height_km" in result
    assert any("km mean" in message and "3/3 valid" in message for message in logger.info_messages)
    assert not any("search=" in message for message in logger.info_messages)
    assert any("search=" in message for message in logger.debug_messages)
