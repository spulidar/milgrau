"""Tests for SCC time-axis application of upstream acquisition QA."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from milgrau.level0.netcdf import _scc_time_axis


def _rows(*, stops: tuple[int, ...] = (60, 121), nominal_duration_s: int = 60) -> pd.DataFrame:
    reference = pd.Timestamp("2025-09-14T12:00:00Z")
    starts = (0, 60)
    return pd.DataFrame(
        {
            "start_time_utc": [reference + pd.Timedelta(seconds=value) for value in starts],
            "stop_time": [reference + pd.Timedelta(seconds=value) for value in stops],
            "qa_nominal_duration_s": (nominal_duration_s, nominal_duration_s),
        }
    )


def _scc_config() -> dict:
    return {"_resolved_station": {"scc_available": True}}


def test_scc_time_axis_applies_upstream_nominal_duration_without_revalidating_acquisition() -> None:
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


def test_scc_time_axis_does_not_repeat_laser_shot_or_frequency_validation() -> None:
    rows = _rows()
    rows["nshots"] = (6000, 9999)
    rows["laser_freq"] = (100, 50)
    reference = pd.to_datetime(rows["start_time_utc"], utc=True).iloc[0]

    _, stop, _ = _scc_time_axis(
        rows,
        reference,
        _scc_config(),
        logging.getLogger("test"),
        label="Measurement",
    )

    np.testing.assert_array_equal(stop, np.array([60, 120], dtype=np.int32))


def test_scc_time_axis_without_qa_decision_preserves_header_times() -> None:
    rows = _rows().drop(columns=["qa_nominal_duration_s"])
    reference = pd.to_datetime(rows["start_time_utc"], utc=True).iloc[0]

    start, stop, attrs = _scc_time_axis(
        rows,
        reference,
        _scc_config(),
        logging.getLogger("test"),
        label="Measurement",
    )

    np.testing.assert_array_equal(start, np.array([0, 60], dtype=np.int32))
    np.testing.assert_array_equal(stop, np.array([60, 121], dtype=np.int32))
    assert attrs == {}


def test_internal_level0_preserves_original_header_durations() -> None:
    rows = _rows()
    reference = pd.to_datetime(rows["start_time_utc"], utc=True).iloc[0]
    config = {"_resolved_station": {"scc_available": False}}

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
