"""Tests for continuous-session inventory construction and dark-current association."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from milgrau.level0.inventory import build_session_inventory


def _config(*, max_gap_seconds: float = 60.0, max_association_hours: float = 12.0) -> dict:
    return {
        "directories": {
            "raw_data": "raw",
            "processed_data": "processed",
            "log_dir": "logs",
        },
        "_station_catalog": {
            "station": {
                "id": "spu",
                "timezone": "America/Sao_Paulo",
            }
        },
        "level0": {
            "acquisition_qa": {
                "laser_shot_tolerance_fraction": 0.002,
                "licel_header_time_jitter_s": 1.0,
            },
            "session": {"max_gap_seconds": max_gap_seconds},
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


def test_gap_larger_than_configured_tolerance_starts_new_session(
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
            datetime(2024, 1, 1, 0, 7, 0),
            pd.Timestamp("2024-01-01T00:12:00"),
            300.0,
            1200,
            4.0,
        ),
    }
    _patch_inventory(monkeypatch, paths, ["measurements", "measurements"], headers)

    df = build_session_inventory(
        str(tmp_path),
        _config(max_gap_seconds=60.0),
        logging.getLogger("test"),
    )

    assert df["session_id"].nunique() == 2
    assert df["session_id"].tolist() == [
        "spu_20240101-0000Z_20240101-0005Z",
        "spu_20240101-0007Z_20240101-0012Z",
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
