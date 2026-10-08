"""Tests for geometric solar regime and segment construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from milgrau.physics.solar import (
    build_solar_segments,
    solar_elevation_deg,
    solar_regime,
)


def test_solar_elevation_distinguishes_local_day_and_night_at_spu() -> None:
    times = pd.to_datetime(
        ["2025-01-15T15:00:00Z", "2025-01-15T03:00:00Z"],
        utc=True,
    )
    elevation = solar_elevation_deg(times, -23.5615, -46.7383)

    assert elevation[0] > 40.0
    assert elevation[1] < -20.0


def test_solar_regime_uses_configured_minus_three_degree_threshold() -> None:
    values = np.array([-10.0, -3.01, -3.0, 0.0, 25.0])
    observed = solar_regime(values, day_night_threshold_deg=-3.0)

    assert observed.tolist() == ["night", "night", "day", "day", "day"]


def test_build_solar_segments_keeps_repeated_day_periods_distinct() -> None:
    starts = pd.to_datetime(
        [
            "2025-01-01T19:00:00Z",
            "2025-01-01T20:00:00Z",
            "2025-01-01T21:00:00Z",
            "2025-01-02T08:00:00Z",
        ],
        utc=True,
    )
    stops = starts + pd.Timedelta(minutes=30)
    regimes = np.array(["day", "night", "night", "day"], dtype=object)

    labels, segments = build_solar_segments(starts, stops, regimes)

    assert labels.tolist() == ["seg00", "seg01", "seg01", "seg02"]
    assert [segment.regime for segment in segments] == ["day", "night", "day"]
    assert [segment.segment_id for segment in segments] == ["seg00", "seg01", "seg02"]
    assert segments[1].start_time_utc == starts[1]
    assert segments[1].end_time_utc == stops[2]
