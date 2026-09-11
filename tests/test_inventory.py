"""Tests for Level 0 inventory construction and dark-current association."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from milgrau.level0.inventory import build_measurement_inventory


def _config(*, incremental: bool = False, max_association_hours: float = 12.0) -> dict:
    return {
        "_station_catalog": {
            "station": {
                "timezone": "America/Sao_Paulo",
            }
        },
        "level0": {
            "acquisition_qa": {
                "laser_shot_tolerance_fraction": 0.002,
                "licel_header_time_jitter_s": 1.0,
            },
            "dark_current": {"max_association_hours": max_association_hours},
            "surface_weather": {"missing_policy": "nan"},
        },
        "processing": {"incremental": incremental},
    }


def test_inventory_reassigns_orphan_dark_current_with_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Orphan dark-current files should be linked to nearby measurement groups."""
    import milgrau.level0.inventory as inventory_module

    measurement_path = str(tmp_path / "measurement_file")
    dark_path = str(tmp_path / "dark_file")

    def fake_scan_raw_files(raw_dir: str, logger: logging.Logger, config: dict) -> tuple[list[str], list[str]]:
        return [measurement_path, dark_path], ["measurements", "dark_current"]

    def fake_read_licel_header(filepath: str, logger: logging.Logger):
        if filepath == measurement_path:
            return datetime(2024, 1, 1, 12, 0, 0), pd.Timestamp("2024-01-01T12:05:00"), 300.0, 1200, 10.0
        return datetime(2024, 1, 1, 18, 0, 0), pd.Timestamp("2024-01-01T18:05:00"), 300.0, 1200, 10.0

    monkeypatch.setattr(inventory_module, "scan_raw_files", fake_scan_raw_files)
    monkeypatch.setattr(inventory_module, "read_licel_header", fake_read_licel_header)

    df = build_measurement_inventory(str(tmp_path), _config(), logging.getLogger("test"))

    assert len(df) == 2
    measurement_id = df.loc[df["meas_type"] == "measurements", "meas_id"].iloc[0]
    dark_row = df.loc[df["meas_type"] == "dark_current"].iloc[0]

    assert dark_row["meas_id"] == measurement_id
    assert dark_row["original_meas_id"] != measurement_id
    assert dark_row["association_method"] == "nearest_measurement"
    assert float(dark_row["dark_current_association_delta_hours"]) == 6.0


def test_inventory_respects_explicit_dark_current_maximum(tmp_path: Path, monkeypatch) -> None:
    import milgrau.level0.inventory as inventory_module

    measurement_path = str(tmp_path / "measurement_file")
    dark_path = str(tmp_path / "dark_file")

    monkeypatch.setattr(
        inventory_module,
        "scan_raw_files",
        lambda raw_dir, logger, config: ([measurement_path, dark_path], ["measurements", "dark_current"]),
    )

    def fake_read_licel_header(filepath: str, logger: logging.Logger):
        hour = 12 if filepath == measurement_path else 18
        return datetime(2024, 1, 1, hour, 0, 0), pd.Timestamp(f"2024-01-01T{hour:02d}:05:00"), 300.0, 1200, 10.0

    monkeypatch.setattr(inventory_module, "read_licel_header", fake_read_licel_header)
    df = build_measurement_inventory(
        str(tmp_path),
        _config(max_association_hours=5.0),
        logging.getLogger("test-dark-limit"),
    )

    dark_row = df.loc[df["meas_type"] == "dark_current"].iloc[0]
    measurement_id = df.loc[df["meas_type"] == "measurements", "meas_id"].iloc[0]
    assert dark_row["meas_id"] != measurement_id
    assert pd.isna(dark_row["dark_current_association_delta_hours"])


def test_inventory_keeps_incremental_decision_out_of_inventory(tmp_path: Path, monkeypatch) -> None:
    """Inventory should not drop groups simply because incremental mode is enabled."""
    import milgrau.level0.inventory as inventory_module

    measurement_path = str(tmp_path / "measurement_file")

    def fake_scan_raw_files(raw_dir: str, logger: logging.Logger, config: dict) -> tuple[list[str], list[str]]:
        return [measurement_path], ["measurements"]

    def fake_read_licel_header(filepath: str, logger: logging.Logger):
        return datetime(2024, 1, 1, 12, 0, 0), pd.Timestamp("2024-01-01T12:05:00"), 300.0, 1200, 10.0

    monkeypatch.setattr(inventory_module, "scan_raw_files", fake_scan_raw_files)
    monkeypatch.setattr(inventory_module, "read_licel_header", fake_read_licel_header)

    config = _config(incremental=True)
    config["directories"] = {"processed_data": str(tmp_path / "processed")}
    df = build_measurement_inventory(str(tmp_path), config, logging.getLogger("test"))

    assert len(df) == 1
    assert df.iloc[0]["meas_type"] == "measurements"


def test_inventory_keeps_post_midnight_measurements_on_same_civil_date(tmp_path: Path, monkeypatch) -> None:
    """Night measurements after midnight should keep their actual civil date."""
    import milgrau.level0.inventory as inventory_module

    measurement_path = str(tmp_path / "night_measurement")

    def fake_scan_raw_files(raw_dir: str, logger: logging.Logger, config: dict) -> tuple[list[str], list[str]]:
        return [measurement_path], ["measurements"]

    def fake_read_licel_header(filepath: str, logger: logging.Logger):
        return datetime(2024, 1, 1, 3, 30, 0), pd.Timestamp("2024-01-01T03:35:00"), 300.0, 1200, 10.0

    monkeypatch.setattr(inventory_module, "scan_raw_files", fake_scan_raw_files)
    monkeypatch.setattr(inventory_module, "read_licel_header", fake_read_licel_header)

    df = build_measurement_inventory(str(tmp_path), _config(), logging.getLogger("test"))

    assert len(df) == 1
    assert df.iloc[0]["meas_id"] == "20240101nt"
