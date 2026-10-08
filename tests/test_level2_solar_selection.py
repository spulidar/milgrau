"""Tests for Level 2 solar-regime and segment subsetting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.level2.time_window import subset_level1_context


def _dataset() -> xr.Dataset:
    time = pd.to_datetime(
        [
            "2025-01-01T18:50:00",
            "2025-01-01T19:00:00",
            "2025-01-01T19:10:00",
            "2025-01-02T08:00:00",
            "2025-01-02T08:10:00",
            "2025-01-02T23:00:00",
        ]
    )
    return xr.Dataset(
        data_vars={
            "solar_elevation_deg": (("time",), np.array([2.0, -5.0, -10.0, 1.0, 5.0, -20.0])),
            "solar_regime": (
                ("time",),
                np.array(["day", "night", "night", "day", "day", "night"], dtype=object),
            ),
            "segment_id": (
                ("time",),
                np.array(["seg00", "seg01", "seg01", "seg02", "seg02", "seg03"], dtype=object),
            ),
            "Segment_Label": (
                ("segments",),
                np.array(["seg00", "seg01", "seg02", "seg03"], dtype=object),
            ),
            "Segment_Regime": (
                ("segments",),
                np.array(["day", "night", "day", "night"], dtype=object),
            ),
            "Segment_Start_Time_UTC": (
                ("segments",),
                np.array([0, 1, 2, 3], dtype=np.int64),
            ),
            "Segment_End_Time_UTC": (
                ("segments",),
                np.array([1, 2, 3, 4], dtype=np.int64),
            ),
        },
        coords={"time": time},
    )


def test_regime_selector_can_keep_multiple_disjoint_night_segments() -> None:
    selected, tag = subset_level1_context(_dataset(), regime="night")

    assert tag == "night"
    assert selected.sizes["time"] == 3
    assert selected["segment_id"].values.astype(str).tolist() == ["seg01", "seg01", "seg03"]
    assert selected["Segment_Label"].values.astype(str).tolist() == ["seg01", "seg03"]
    assert selected.attrs["LEBEAR_Solar_Regime"] == "night"


def test_segment_selector_keeps_exactly_one_contiguous_segment() -> None:
    selected, tag = subset_level1_context(_dataset(), segment_id="seg02")

    assert tag == "seg02"
    assert selected.sizes["time"] == 2
    assert selected["solar_regime"].values.astype(str).tolist() == ["day", "day"]
    assert selected["Segment_Label"].values.astype(str).tolist() == ["seg02"]
    assert selected.attrs["LEBEAR_Segment_ID"] == "seg02"


def test_regime_selector_intersects_with_explicit_utc_window() -> None:
    selected, tag = subset_level1_context(
        _dataset(),
        start_utc="2025-01-01T18:55:00",
        stop_utc="2025-01-01T19:15:00",
        regime="night",
    )

    assert selected.sizes["time"] == 2
    assert selected["segment_id"].values.astype(str).tolist() == ["seg01", "seg01"]
    assert tag == "night_1855-1915Z"
