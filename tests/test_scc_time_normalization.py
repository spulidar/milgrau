"""Tests for SCC time-resolution normalization of Licel whole-second timestamps."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from milgrau.level0.netcdf import _scc_time_axis


def _rows(*, shots: tuple[int, ...] = (6000, 6000), stops: tuple[int, ...] = (60, 121)) -> pd.DataFrame:
    reference = pd.Timestamp("2025-09-14T12:00:00Z")
    starts = (0, 60)
    return pd.DataFrame(
        {
            "start_time_utc": [reference + pd.Timedelta(seconds=value) for value in starts],
            "stop_time": [reference + pd.Timedelta(seconds=value) for value in stops],
            "nshots": shots,
            "laser_freq": (100, 100),
        }
    )


def _scc_config() -> dict:
    return {
        "processing": {"laser_shot_tolerance_fraction": 0.002},
        "_resolved_station": {"scc_available": True},
    }


def test_scc_time_axis_normalizes_one_second_header_jitter_when_shots_support_nominal_duration() -> None:
    rows = _rows()
    reference = pd.to_datetime(rows["start_time_utc"], utc=True).iloc[0]

    start, stop, attrs = _scc_time_axis(
        rows,
        reference,
        _scc_config(),
        logging.getLogger("test"),
        label="Measurement",
    )

    np.testing.assert_array_equal(start, np.array([0, 60], dtype=np.int32))
    np.testing.assert_array_equal(stop, np.array([60, 120], dtype=np.int32))
    assert attrs["SCC_Time_Axis_Normalized"] == 1
    assert attrs["SCC_Nominal_Time_Resolution_s"] == 60
    assert attrs["SCC_Time_Adjusted_Profile_Count"] == 1
    assert attrs["SCC_Max_Time_Adjustment_s"] == 1


def test_scc_time_axis_accepts_shot_variation_below_existing_quality_tolerance() -> None:
    rows = _rows(shots=(6000, 6010))
    reference = pd.to_datetime(rows["start_time_utc"], utc=True).iloc[0]

    _, stop, attrs = _scc_time_axis(
        rows,
        reference,
        _scc_config(),
        logging.getLogger("test"),
        label="Measurement",
    )

    np.testing.assert_array_equal(stop, np.array([60, 120], dtype=np.int32))
    assert attrs["SCC_Nominal_Time_Resolution_s"] == 60


def test_scc_time_axis_refuses_real_shot_change_beyond_quality_tolerance() -> None:
    rows = _rows(shots=(6000, 6100))
    reference = pd.to_datetime(rows["start_time_utc"], utc=True).iloc[0]

    with pytest.raises(ValueError, match="laser shots are not consistent"):
        _scc_time_axis(
            rows,
            reference,
            _scc_config(),
            logging.getLogger("test"),
            label="Measurement",
        )


def test_internal_level0_preserves_original_variable_header_durations() -> None:
    rows = _rows()
    reference = pd.to_datetime(rows["start_time_utc"], utc=True).iloc[0]
    config = {
        "processing": {"laser_shot_tolerance_fraction": 0.002},
        "_resolved_station": {"scc_available": False},
    }

    start, stop, attrs = _scc_time_axis(
        rows,
        reference,
        config,
        logging.getLogger("test"),
        label="Measurement",
    )

    np.testing.assert_array_equal(start, np.array([0, 60], dtype=np.int32))
    np.testing.assert_array_equal(stop, np.array([60, 121], dtype=np.int32))
    assert attrs == {}
