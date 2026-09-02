"""Tests for centralized Level-0 acquisition QA."""

from __future__ import annotations

import logging

import pandas as pd

from milgrau.level0.quality import filter_laser_shots


def _row(
    *,
    filepath: str,
    meas_type: str,
    nshots: int,
    duration: float,
    laser_freq: int = 100,
) -> dict:
    return {
        "filepath": filepath,
        "meas_id": "20250914pm",
        "meas_type": meas_type,
        "nshots": nshots,
        "duration": duration,
        "laser_freq": laser_freq,
    }


def test_dark_current_uses_same_shot_tolerance_as_measurements() -> None:
    df = pd.DataFrame(
        [
            _row(filepath="m1", meas_type="measurements", nshots=3000, duration=30),
            _row(filepath="m2", meas_type="measurements", nshots=3001, duration=31),
            _row(filepath="d1", meas_type="dark_current", nshots=3001, duration=30),
            _row(filepath="d2", meas_type="dark_current", nshots=3002, duration=31),
            _row(filepath="d_bad", meas_type="dark_current", nshots=3020, duration=30),
        ]
    )

    good = filter_laser_shots(df, logging.getLogger("test"), tolerance_fraction=0.002)

    assert set(good["filepath"]) == {"m1", "m2", "d1", "d2"}
    dark = good[good["meas_type"] == "dark_current"]
    assert set(dark["qa_nominal_duration_s"]) == {30.0}


def test_one_second_header_jitter_is_accepted_and_marked_for_scc_normalization() -> None:
    df = pd.DataFrame(
        [
            _row(filepath="m1", meas_type="measurements", nshots=3000, duration=30),
            _row(filepath="m2", meas_type="measurements", nshots=3000, duration=31),
        ]
    )

    good = filter_laser_shots(df, logging.getLogger("test"), tolerance_fraction=0.002)

    assert list(good["filepath"]) == ["m1", "m2"]
    assert list(good["qa_nominal_duration_s"]) == [30.0, 30.0]
    assert list(good["qa_header_duration_adjustment_s"]) == [0.0, -1.0]


def test_large_header_duration_anomaly_is_rejected_before_writer() -> None:
    df = pd.DataFrame(
        [
            _row(filepath="m1", meas_type="measurements", nshots=3000, duration=30),
            _row(filepath="m_bad", meas_type="measurements", nshots=3000, duration=91),
        ]
    )

    good = filter_laser_shots(df, logging.getLogger("test"), tolerance_fraction=0.002)

    assert list(good["filepath"]) == ["m1"]


def test_measurements_and_dark_currents_get_independent_nominals() -> None:
    df = pd.DataFrame(
        [
            _row(filepath="m1", meas_type="measurements", nshots=3000, duration=30),
            _row(filepath="m2", meas_type="measurements", nshots=3001, duration=30),
            _row(filepath="d1", meas_type="dark_current", nshots=6000, duration=60),
            _row(filepath="d2", meas_type="dark_current", nshots=6001, duration=61),
        ]
    )

    good = filter_laser_shots(df, logging.getLogger("test"), tolerance_fraction=0.002)

    meas = good[good["meas_type"] == "measurements"]
    dark = good[good["meas_type"] == "dark_current"]
    assert set(meas["qa_nominal_duration_s"]) == {30.0}
    assert set(dark["qa_nominal_duration_s"]) == {60.0}
