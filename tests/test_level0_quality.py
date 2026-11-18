"""Tests for centralized Level-0 acquisition QA."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

import milgrau.level0.quality as quality
from milgrau.level0.quality import filter_laser_shots


QA = {"tolerance_fraction": 0.002, "header_time_jitter_s": 1.0}


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
        "session_id": "spu_20250914-1500Z_20250914-1600Z",
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

    good = filter_laser_shots(df, logging.getLogger("test"), **QA)

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

    good = filter_laser_shots(df, logging.getLogger("test"), **QA)

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

    good = filter_laser_shots(df, logging.getLogger("test"), **QA)

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

    good = filter_laser_shots(df, logging.getLogger("test"), **QA)

    meas = good[good["meas_type"] == "measurements"]
    dark = good[good["meas_type"] == "dark_current"]
    assert set(meas["qa_nominal_duration_s"]) == {30.0}
    assert set(dark["qa_nominal_duration_s"]) == {60.0}


def test_header_jitter_tolerance_is_not_hardcoded() -> None:
    df = pd.DataFrame(
        [
            _row(filepath="m1", meas_type="measurements", nshots=3000, duration=30),
            _row(filepath="m2", meas_type="measurements", nshots=3000, duration=31),
        ]
    )

    good = filter_laser_shots(
        df,
        logging.getLogger("test"),
        tolerance_fraction=0.002,
        header_time_jitter_s=0.5,
    )

    assert list(good["filepath"]) == ["m1"]


def test_unexpected_qa_runtime_failure_is_not_silently_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Programming/runtime defects must fail outward instead of looking like bad data."""
    df = pd.DataFrame(
        [_row(filepath="m1", meas_type="measurements", nshots=3000, duration=30)]
    )

    def fail_unexpectedly(_values) -> float:
        raise RuntimeError("synthetic implementation failure")

    monkeypatch.setattr(quality, "safe_mode", fail_unexpectedly)

    with pytest.raises(RuntimeError, match="synthetic implementation failure"):
        filter_laser_shots(df, logging.getLogger("test"), **QA)


def test_qa_debug_log_identifies_rejected_file_and_reason(caplog) -> None:
    rows = pd.DataFrame([
        _row(filepath="/raw/valid.licel", meas_type="measurements", nshots=3000, duration=30),
        _row(filepath="/raw/bad-duration.licel", meas_type="measurements", nshots=3000, duration=91),
    ])
    logger = logging.getLogger("test.qa.file-reasons")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        good = filter_laser_shots(rows, logger, **QA)

    assert good["filepath"].tolist() == ["/raw/valid.licel"]
    assert any(
        "/raw/bad-duration.licel" in record.message
        and "invalid_or_inconsistent_duration" in record.message
        and "header_duration_s=91" in record.message
        for record in caplog.records
    )


@pytest.mark.parametrize(
    ("nshots", "laser_freq", "header_duration_s", "expected_physical_duration_s"),
    [
        (10, 20, 1, 0.5),       # turbulence: Licel header timestamps are whole seconds
        (21, 10, 2, 2.1),       # short measurement, no integer-rounding prerequisite
        (605, 20, 30, 30.25),  # longer fractional acquisition
        (2001, 10, 200, 200.1),
    ],
)
def test_fractional_acquisitions_are_validated_without_rounding(
    nshots: int,
    laser_freq: int,
    header_duration_s: float,
    expected_physical_duration_s: float,
) -> None:
    rows = pd.DataFrame([
        _row(
            filepath="/raw/fractional.licel",
            meas_type="measurements",
            nshots=nshots,
            laser_freq=laser_freq,
            duration=header_duration_s,
        ),
    ])

    good = filter_laser_shots(rows, logging.getLogger("test"), **QA)

    assert good["filepath"].tolist() == ["/raw/fractional.licel"]
    assert good["qa_nominal_duration_s"].iloc[0] == pytest.approx(expected_physical_duration_s)


def test_large_physical_vs_header_duration_discrepancy_stays_rejected() -> None:
    rows = pd.DataFrame([
        _row(filepath="/raw/valid.licel", meas_type="measurements",
             nshots=601, laser_freq=20, duration=30),
        _row(filepath="/raw/invalid-factor-two.licel", meas_type="measurements",
             nshots=601, laser_freq=20, duration=60),
    ])

    good = filter_laser_shots(rows, logging.getLogger("test"), **QA)

    assert good["filepath"].tolist() == ["/raw/valid.licel"]
