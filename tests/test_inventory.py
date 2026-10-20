"""Tests for continuous-session inventory construction and dark-current association."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from milgrau.level0.inventory import build_session_inventory


def _config(*, max_gap_seconds: float = 1800.0, max_association_hours: float = 12.0) -> dict:
    return {
        "directories": {
            "raw_data": "raw",
            "processed_data": "processed",
            "log_dir": "logs",
        },
        "_station_catalog": {
            "station": {
                "id": "spu",
                "name": "Synthetic SPU",
                "timezone": "America/Sao_Paulo",
                "site": {},
            },
            "calibrations": {
                "cal-a": {"provenance": {"source": "test"}, "channels": {}},
                "cal-b": {"provenance": {"source": "test"}, "channels": {}},
            },
            "profiles": [
                {
                    "id": "profile-a",
                    "calibration_id": "cal-a",
                    "valid_from": "2020-01-01",
                    "valid_to": None,
                }
            ],
        },
        "level0": {
            "acquisition_qa": {
                "laser_shot_tolerance_fraction": 0.002,
                "licel_header_time_jitter_s": 1.0,
            },
            "session": {"max_gap_seconds": max_gap_seconds},
            "solar_regime": {"day_night_threshold_deg": -3.0},
            "dark_current": {"max_association_hours": max_association_hours},
            "surface_weather": {"missing_policy": "nan"},
        },
        "processing": {
            "incremental": False,
            "spurious_extensions": [],
            "raw_scan_ignore_dirs": [],
            "quarantine_dir": "quarantine",
        },
    }


def _patch_inventory(
    monkeypatch,
    paths: list[str],
    kinds: list[str],
    headers: dict[str, tuple[datetime, pd.Timestamp, float, int, float]],
) -> None:
    import milgrau.level0.inventory as inventory_module

    def fake_scan_raw_files(
        raw_dir,
        *,
        spurious_extensions,
        quarantine_dir,
        raw_scan_ignore_dirs,
        logger=None,
    ) -> tuple[list[str], list[str]]:
        return paths, kinds

    def fake_read_licel_header(filepath: str, logger: logging.Logger):
        return headers[filepath]

    monkeypatch.setattr(inventory_module, "scan_raw_files", fake_scan_raw_files)
    monkeypatch.setattr(inventory_module, "read_licel_header", fake_read_licel_header)


def test_continuous_acquisition_crossing_midnight_stays_one_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = [str(tmp_path / "m1"), str(tmp_path / "m2")]
    headers = {
        paths[0]: (
            datetime(2024, 1, 1, 23, 55, 0),
            pd.Timestamp("2024-01-02T00:00:00"),
            300.0,
            1200,
            4.0,
        ),
        paths[1]: (
            datetime(2024, 1, 2, 0, 0, 30),
            pd.Timestamp("2024-01-02T00:05:30"),
            300.0,
            1200,
            4.0,
        ),
    }
    _patch_inventory(monkeypatch, paths, ["measurements", "measurements"], headers)

    df = build_session_inventory(str(tmp_path), _config(), logging.getLogger("test"))

    assert df["session_id"].nunique() == 1
    assert df["session_id"].iloc[0] == "spu_20240101-2355Z_20240102-0005Z"
    assert "period" not in df.columns


def test_session_timestamp_columns_are_timezone_aware_utc(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = str(tmp_path / "m1")
    headers = {
        path: (
            datetime(2024, 6, 20, 21, 3, 48),
            pd.Timestamp("2024-06-20T21:08:48"),
            300.0,
            1200,
            4.0,
        ),
    }
    _patch_inventory(monkeypatch, [path], ["measurements"], headers)

    df = build_session_inventory(str(tmp_path), _config(), logging.getLogger("test-tz-dtype"))

    assert str(df["session_start_utc"].dt.tz) == "UTC"
    assert str(df["session_end_utc"].dt.tz) == "UTC"
    assert df["session_start_utc"].iloc[0] == pd.Timestamp("2024-06-20T21:03:48Z")
    assert df["session_end_utc"].iloc[0] == pd.Timestamp("2024-06-20T21:08:48Z")


def test_acquisition_is_not_split_by_former_six_hour_boundaries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = [str(tmp_path / f"m{index}") for index in range(3)]
    headers = {
        paths[0]: (
            datetime(2024, 1, 1, 5, 55, 0),
            pd.Timestamp("2024-01-01T06:00:00"),
            300.0,
            1200,
            4.0,
        ),
        paths[1]: (
            datetime(2024, 1, 1, 6, 0, 0),
            pd.Timestamp("2024-01-01T06:05:00"),
            300.0,
            1200,
            4.0,
        ),
        paths[2]: (
            datetime(2024, 1, 1, 6, 5, 0),
            pd.Timestamp("2024-01-01T06:10:00"),
            300.0,
            1200,
            4.0,
        ),
    }
    _patch_inventory(monkeypatch, paths, ["measurements"] * 3, headers)

    df = build_session_inventory(str(tmp_path), _config(), logging.getLogger("test"))

    assert df["session_id"].nunique() == 1
    assert df["session_id"].iloc[0] == "spu_20240101-0555Z_20240101-0610Z"


def test_gap_within_30_minutes_keeps_one_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = [str(tmp_path / "m1"), str(tmp_path / "m2")]
    headers = {
        paths[0]: (
            datetime(2024, 1, 1, 0, 0, 0),
            pd.Timestamp("2024-01-01T00:05:00"),
            300.0,
            1200,
            4.0,
        ),
        paths[1]: (
            datetime(2024, 1, 1, 0, 25, 0),
            pd.Timestamp("2024-01-01T00:30:00"),
            300.0,
            1200,
            4.0,
        ),
    }
    _patch_inventory(monkeypatch, paths, ["measurements", "measurements"], headers)

    df = build_session_inventory(str(tmp_path), _config(), logging.getLogger("test"))

    assert df["session_id"].nunique() == 1
    assert df["session_id"].iloc[0] == "spu_20240101-0000Z_20240101-0030Z"


def test_gap_larger_than_30_minutes_starts_new_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = [str(tmp_path / "m1"), str(tmp_path / "m2")]
    headers = {
        paths[0]: (
            datetime(2024, 1, 1, 0, 0, 0),
            pd.Timestamp("2024-01-01T00:05:00"),
            300.0,
            1200,
            4.0,
        ),
        paths[1]: (
            datetime(2024, 1, 1, 0, 36, 0),
            pd.Timestamp("2024-01-01T00:41:00"),
            300.0,
            1200,
            4.0,
        ),
    }
    _patch_inventory(monkeypatch, paths, ["measurements", "measurements"], headers)

    df = build_session_inventory(str(tmp_path), _config(), logging.getLogger("test"))

    assert df["session_id"].nunique() == 2
    assert df["session_id"].tolist() == [
        "spu_20240101-0000Z_20240101-0005Z",
        "spu_20240101-0036Z_20240101-0041Z",
    ]


def test_dark_current_is_associated_to_nearest_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    measurement = str(tmp_path / "measurement")
    dark = str(tmp_path / "dark")
    headers = {
        measurement: (
            datetime(2024, 1, 1, 12, 0, 0),
            pd.Timestamp("2024-01-01T12:05:00"),
            300.0,
            1200,
            4.0,
        ),
        dark: (
            datetime(2024, 1, 1, 11, 30, 0),
            pd.Timestamp("2024-01-01T11:35:00"),
            300.0,
            1200,
            4.0,
        ),
    }
    _patch_inventory(monkeypatch, [measurement, dark], ["measurements", "dark_current"], headers)

    df = build_session_inventory(str(tmp_path), _config(), logging.getLogger("test"))
    session_id = df.loc[df["meas_type"] == "measurements", "session_id"].iloc[0]
    dark_row = df.loc[df["meas_type"] == "dark_current"].iloc[0]

    assert dark_row["session_id"] == session_id
    assert dark_row["association_method"] == "nearest_session"
    assert float(dark_row["dark_current_association_delta_hours"]) == 0.5
    assert str(df["session_start_utc"].dt.tz) == "UTC"
    assert str(df["session_end_utc"].dt.tz) == "UTC"


def test_dark_current_outside_maximum_remains_unassociated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    measurement = str(tmp_path / "measurement")
    dark = str(tmp_path / "dark")
    headers = {
        measurement: (
            datetime(2024, 1, 1, 12, 0, 0),
            pd.Timestamp("2024-01-01T12:05:00"),
            300.0,
            1200,
            4.0,
        ),
        dark: (
            datetime(2024, 1, 1, 18, 0, 0),
            pd.Timestamp("2024-01-01T18:05:00"),
            300.0,
            1200,
            4.0,
        ),
    }
    _patch_inventory(monkeypatch, [measurement, dark], ["measurements", "dark_current"], headers)

    df = build_session_inventory(
        str(tmp_path),
        _config(max_association_hours=5.0),
        logging.getLogger("test"),
    )
    dark_row = df.loc[df["meas_type"] == "dark_current"].iloc[0]

    assert pd.isna(dark_row["session_id"])
    assert dark_row["association_method"] == "unassociated"
    assert pd.isna(dark_row["dark_current_association_delta_hours"])


def test_station_profile_change_splits_otherwise_continuous_acquisition(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = [str(tmp_path / "m1"), str(tmp_path / "m2")]
    headers = {
        paths[0]: (
            datetime(2024, 1, 2, 2, 55, 0),
            pd.Timestamp("2024-01-02T03:00:00"),
            300.0,
            1200,
            4.0,
        ),
        paths[1]: (
            datetime(2024, 1, 2, 3, 0, 0),
            pd.Timestamp("2024-01-02T03:05:00"),
            300.0,
            1200,
            4.0,
        ),
    }
    _patch_inventory(monkeypatch, paths, ["measurements", "measurements"], headers)
    config = _config()
    config["_station_catalog"]["profiles"] = [
        {
            "id": "profile-a",
            "calibration_id": "cal-a",
            "valid_from": "2020-01-01",
            "valid_to": "2024-01-01",
        },
        {
            "id": "profile-b",
            "calibration_id": "cal-b",
            "valid_from": "2024-01-02",
            "valid_to": None,
        },
    ]

    df = build_session_inventory(str(tmp_path), config, logging.getLogger("test-profile-boundary"))

    measurement_rows = df[df["meas_type"] == "measurements"].reset_index(drop=True)
    assert measurement_rows["session_id"].nunique() == 2
    assert measurement_rows["station_profile_id"].tolist() == ["profile-a", "profile-b"]
    assert measurement_rows["instrument_calibration_id"].tolist() == ["cal-a", "cal-b"]
    assert measurement_rows["session_boundary_reason"].tolist() == [
        "acquisition_start",
        "station_context_change",
    ]


def test_collapsed_minute_session_logs_exact_licel_sources(
    tmp_path: Path, monkeypatch, caplog,
) -> None:
    """A short valid header must identify the source, not just a cryptic ID error."""
    path = str(tmp_path / "short_licel_file")
    headers = {
        path: (
            datetime(2024, 7, 16, 12, 35, 10),
            pd.Timestamp("2024-07-16T12:35:16"),
            6.0,
            601,
            100.0,
        )
    }
    _patch_inventory(monkeypatch, [path], ["measurements"], headers)
    logger = logging.getLogger("test.inventory.short-session")
    with caplog.at_level(logging.ERROR, logger=logger.name):
        with pytest.raises(ValueError, match="same minute") as caught:
            build_session_inventory(str(tmp_path), _config(), logger)

    assert path in str(caught.value)
    assert "2024-07-16T12:35:10+00:00" in str(caught.value)
    assert "2024-07-16T12:35:16+00:00" in str(caught.value)
    assert any(
        path in record.message
        and "header_duration_s=6.0" in record.message
        and "nshots=601" in record.message
        for record in caplog.records
    )


def test_optimized_dark_current_matches_exhaustive_nearest_interval() -> None:
    """Nearest-interval lookup preserves overlap, tie, threshold, and UTC semantics."""
    from milgrau.level0.inventory import _associate_dark_currents, _session_table

    base = pd.Timestamp("2024-06-01T00:00:00Z")
    specifications = [
        ("spu-a", 0, 120),
        ("spu-b", 60, 180),
        ("spu-c", 240, 300),
        ("spu-d", 310, 360),
    ]
    measurements = pd.DataFrame(
        [
            {
                "session_id": name,
                "session_start_utc": base + pd.Timedelta(minutes=start),
                "session_end_utc": base + pd.Timedelta(minutes=end),
            }
            for name, start, end in specifications
        ]
    )
    offsets = [-180, 0, 60, 90, 120, 150, 180, 210, 240, 300, 305, 310, 400, 1080, 1081]
    dark_rows = pd.DataFrame(
        {
            "filepath": [f"/raw/dark_{minute:+d}" for minute in offsets],
            "meas_type": ["dark_current"] * len(offsets),
            "start_time_utc": [base + pd.Timedelta(minutes=minute) for minute in offsets],
        }
    )

    output = _associate_dark_currents(
        dark_rows, measurements, _config(), logging.getLogger("test.dark.optimized")
    )
    sessions = _session_table(measurements)

    for index, dark in dark_rows.iterrows():
        timestamp = dark["start_time_utc"]
        distances = [
            max(start - timestamp, timestamp - end, pd.Timedelta(0))
            for start, end in zip(
                sessions["session_start_utc"], sessions["session_end_utc"],
            )
        ]
        winning_position = min(range(len(distances)), key=lambda position: distances[position])
        expected_hours = distances[winning_position].total_seconds() / 3600.0
        if expected_hours > 12.0:
            assert pd.isna(output.loc[index, "session_id"])
            assert output.loc[index, "association_method"] == "unassociated"
        else:
            assert output.loc[index, "session_id"] == sessions.iloc[winning_position]["session_id"]
            assert output.loc[index, "association_method"] == "nearest_session"
            assert float(output.loc[index, "dark_current_association_delta_hours"]) == pytest.approx(
                expected_hours
            )
            assert output.loc[index, "session_start_utc"] == sessions.iloc[winning_position]["session_start_utc"]
            assert output.loc[index, "session_end_utc"] == sessions.iloc[winning_position]["session_end_utc"]

    assert output.loc[3, "session_id"] == "spu-a"  # overlapping intervals
    assert output.loc[7, "session_id"] == "spu-b"  # equal-distance tie


def test_inventory_reports_stages_and_preserves_station_profile_boundary(
    tmp_path: Path, monkeypatch, caplog,
) -> None:
    paths = [str(tmp_path / "first"), str(tmp_path / "second")]
    headers = {
        paths[0]: (
            datetime(2024, 1, 1, 23, 59, 0),
            pd.Timestamp("2024-01-02T00:04:00Z"),
            300.0, 1200, 4.0,
        ),
        paths[1]: (
            datetime(2024, 1, 2, 0, 4, 0),
            pd.Timestamp("2024-01-02T00:09:00Z"),
            300.0, 1200, 4.0,
        ),
    }
    _patch_inventory(monkeypatch, paths, ["measurements", "measurements"], headers)
    logger = logging.getLogger("test.inventory.progress-info")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        df = build_session_inventory(str(tmp_path), _config(), logger)

    assert df["session_id"].nunique() == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("building raw inventory | station=spu" in msg for msg in messages)
    assert any("read 2/2 | 2 valid | 0 invalid" in msg for msg in messages)
    assert any("2 measurement headers | profile/calibration identities" in msg for msg in messages)
    assert any("1 sessions | 2 measurement headers" in msg for msg in messages)
    assert any("ready | 1 sessions | 2 measurement + 0 dark-current files" in msg for msg in messages)
    assert any("header normalization elapsed" in msg for msg in messages)
