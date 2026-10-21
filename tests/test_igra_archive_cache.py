"""Offline IGRA2 archive precedence, scientific units, and Siphon fallback."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from milgrau.io import igra_archive, radiosonde


# Fixed-width examples from NOAA IGRA2 format for BRM00083779.
ROWS = [
    "21 -9999  92900B-9999   246B-9999    70   290    21",
    "10 -9999  92500   762B  256B-9999    70   290    21",
    "10 -9999  70000  3147B   92B-9999   110   260    72",
    "10 -9999  50000  5860B  -41B-9999   320   285   113",
    "10 -9999  40000  7580B -155B-9999   150   250   180",
]


def _date(day: int, hour: int = 12) -> datetime:
    return datetime(2020, 1, day, hour, tzinfo=timezone.utc)


def _write_archive(root: Path, *, day: int = 7, rows: list[str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    records = ROWS if rows is None else rows
    header = (
        f"#BRM00083779 2020 01 {day:02d} 12 1132 {len(records):4d} "
        "ncdc-gts ncdc-gts -235000  -466333"
    )
    archive = root / "BRM00083779-data.txt"
    archive.write_text(
        header + "\n" + "\n".join(records) + "\n", encoding="ascii",
    )
    return archive


def _fetch(root: Path, measurement: datetime, station: str = "83779"):
    return radiosonde.fetch_wyoming_radiosonde(
        measurement, station, logging.getLogger("test.igra2"),
        cache_dir=root, synoptic_hours_utc=[0, 12],
        selection="nearest", max_time_delta_hours=6.0,
    )


def test_igra2_historical_file_is_used_without_siphon(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "radiosonde"
    _write_archive(cache)

    def no_network(*args, **kwargs):
        raise AssertionError("Siphon must not be used when matching IGRA2 exists.")

    monkeypatch.setattr(radiosonde.WyomingUpperAir, "request_data", no_network)
    monkeypatch.setattr(radiosonde, "_require_current_siphon", no_network)
    profile = _fetch(cache, _date(7).replace(hour=13))
    assert profile is not None
    assert profile["height"].to_list() == [762., 3147., 5860., 7580.]
    assert profile["temperature"].to_list() == [25.6, 9.2, -4.1, -15.5]
    assert profile["pressure"].to_list() == [925., 700., 500., 400.]
    assert profile.attrs["source"] == "NOAA NCEI IGRA2 historical sounding archive"
    assert profile.attrs["historical_station_id"] == "BRM00083779"
    assert profile.attrs["target_datetime_utc"] == _date(7).isoformat()
    assert profile.attrs["time_delta_hours"] == 1.0

    csv_path = cache / "2020" / "01" / "radiosonde_83779_20200107_12Z.csv"
    assert csv_path.is_file()
    assert json.loads(csv_path.with_suffix(".json").read_text())[
        "historical_archive"
    ] == "BRM00083779-data.txt"

    # The newly materialized hourly CSV is the first cache, still no network.
    again = _fetch(cache, _date(7).replace(hour=14))
    assert again is not None
    assert again.attrs["time_delta_hours"] == 2.0
    pd.testing.assert_frame_equal(
        again.reset_index(drop=True), profile.reset_index(drop=True),
    )


def test_missing_igra2_date_falls_back_to_siphon(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "radiosonde"
    _write_archive(cache)
    calls = []

    class FakeWyoming:
        @staticmethod
        def request_data(requested, station):
            calls.append((requested, station))
            return pd.DataFrame({
                "height": [760.0, 3000.0],
                "temperature": [22.0, 9.0],
                "pressure": [930.0, 700.0],
            })

    monkeypatch.setattr(radiosonde, "_require_current_siphon", lambda: "0.11.0")
    monkeypatch.setattr(radiosonde, "WyomingUpperAir", FakeWyoming)
    profile = _fetch(cache, _date(8))
    assert profile is not None
    assert calls == [(_date(8), "83779")]
    assert profile.attrs["source"] == "University of Wyoming Upper Air via Siphon"
    assert "historical_archive" not in profile.attrs


def test_incomplete_igra2_profile_falls_back_not_invent_levels(
    tmp_path: Path, monkeypatch,
) -> None:
    cache = tmp_path / "radiosonde"
    _write_archive(cache, rows=ROWS[:2])
    fallback = []

    class FakeWyoming:
        @staticmethod
        def request_data(date, station):
            fallback.append((date, station))
            return pd.DataFrame({
                "height": [760., 3000.],
                "temperature": [24., 10.],
                "pressure": [930., 700.],
            })

    monkeypatch.setattr(radiosonde, "_require_current_siphon", lambda: "0.11.0")
    monkeypatch.setattr(radiosonde, "WyomingUpperAir", FakeWyoming)
    assert _fetch(cache, _date(7)) is not None
    assert fallback == [(_date(7), "83779")]


def test_wrong_station_never_uses_local_igra_archive(
    tmp_path: Path, monkeypatch,
) -> None:
    cache = tmp_path / "radiosonde"
    _write_archive(cache)
    requested = []

    class FakeWyoming:
        @staticmethod
        def request_data(date, station):
            requested.append(station)
            return pd.DataFrame({
                "height": [760., 3000.], "temperature": [20., 5.],
                "pressure": [930., 700.],
            })

    monkeypatch.setattr(radiosonde, "_require_current_siphon", lambda: "0.11.0")
    monkeypatch.setattr(radiosonde, "WyomingUpperAir", FakeWyoming)
    assert _fetch(cache, _date(7), station="99999") is not None
    assert requested == ["99999"]


def test_igra_index_reuses_offsets_until_archive_changes(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "radiosonde"
    archive = _write_archive(cache)
    igra_archive._archive_index.cache_clear()
    t = archive.stat()
    idx = igra_archive._archive_index(str(archive.resolve()), t.st_size, t.st_mtime_ns)
    assert _date(7) in idx
    t2 = archive.stat()
    assert igra_archive._archive_index(
        str(archive.resolve()), t2.st_size, t2.st_mtime_ns,
    ) is idx
    with archive.open("a", encoding="ascii") as out:
        out.write(
            "#BRM00083779 2020 01 08 12 1131    2 "
            "ncdc-gts ncdc-gts -235000  -466333\n"
        )
        out.write(ROWS[1] + "\n" + ROWS[2] + "\n")
    os.utime(archive, None)
    t3 = archive.stat()
    updated = igra_archive._archive_index(
        str(archive.resolve()), t3.st_size, t3.st_mtime_ns,
    )
    assert _date(8) in updated
    assert updated is not idx


def test_igra_missing_or_qc_removed_levels_are_excluded(tmp_path: Path) -> None:
    cache = tmp_path / "radiosonde"
    invalid = "10 -9999  70000 -8888B-8888B-9999 -9999 -9999 -9999"
    _write_archive(cache, rows=[ROWS[1], invalid, ROWS[2], ROWS[3]])
    profile = igra_archive.load_igra2_cached_sounding(
        cache, "83779", _date(7), logging.getLogger("test.igra2.qc"),
    )
    assert profile is not None
    assert np.isfinite(profile[["height", "temperature", "pressure"]].values).all()
    assert profile["height"].to_list() == [762., 3147., 5860.]


def test_12z_archive_used_at_19z_when_midnight_launch_absent(
    tmp_path: Path, monkeypatch,
) -> None:
    cache = tmp_path / "radiosonde"
    _write_archive(cache, day=7)

    def no_network(*_args, **_kwargs):
        raise AssertionError("Must use valid existing 12Z sounding, not missing 00Z.")

    monkeypatch.setattr(radiosonde.WyomingUpperAir, "request_data", no_network)
    monkeypatch.setattr(radiosonde, "_require_current_siphon", no_network)
    selection = radiosonde.fetch_wyoming_radiosonde(
        _date(7, 19), "83779", logging.getLogger("test.radio.19z"),
        cache_dir=cache, synoptic_hours_utc=[0, 12],
        selection="nearest", max_time_delta_hours=12.0,
    )
    assert selection is not None
    assert selection.attrs["target_datetime_utc"] == _date(7, 12).isoformat()
    assert selection.attrs["time_delta_hours"] == 7.0
    assert selection.attrs["historical_archive"] == "BRM00083779-data.txt"
    # Same physical profile at 20Z, but an independently correct 8h age.
    again = radiosonde.fetch_wyoming_radiosonde(
        _date(7, 20), "83779", logging.getLogger("test.radio.20z"),
        cache_dir=cache, synoptic_hours_utc=[0, 12],
        selection="nearest", max_time_delta_hours=12.0,
    )
    assert again is not None
    assert again.attrs["time_delta_hours"] == 8.0
    assert again.attrs["target_datetime_utc"] == _date(7, 12).isoformat()


def test_12z_archive_used_at_next_day_00z_at_twelve_hour_boundary(
    tmp_path: Path, monkeypatch,
) -> None:
    cache = tmp_path / "radiosonde"
    _write_archive(cache, day=7)
    monkeypatch.setattr(
        radiosonde, "_require_current_siphon",
        lambda: (_ for _ in ()).throw(AssertionError("Unneeded Siphon.")),
    )
    profile = radiosonde.fetch_wyoming_radiosonde(
        _date(8, 0), "83779", logging.getLogger("test.radio.midnight"),
        cache_dir=cache, synoptic_hours_utc=[0, 12],
        selection="nearest", max_time_delta_hours=12.0,
    )
    assert profile is not None
    assert profile.attrs["target_datetime_utc"] == _date(7, 12).isoformat()
    assert profile.attrs["time_delta_hours"] == 12.0


def test_six_hour_policy_not_silently_broadened_for_era5_qa(
    tmp_path: Path, monkeypatch,
) -> None:
    cache = tmp_path / "radiosonde"
    _write_archive(cache, day=7)
    called = []

    class FakeWyoming:
        @staticmethod
        def request_data(requested, station):
            called.append((requested, station))
            return pd.DataFrame({
                "height": [760., 3000.], "temperature": [22., 10.],
                "pressure": [930., 700.],
            })

    monkeypatch.setattr(radiosonde, "_require_current_siphon", lambda: "0.11.0")
    monkeypatch.setattr(radiosonde, "WyomingUpperAir", FakeWyoming)
    # The prior day's 12Z is *seven* hours old and must not be
    # reused under the standard six-hour policy.
    result = radiosonde.fetch_wyoming_radiosonde(
        _date(7, 19), "83779", logging.getLogger("test.radio.strict"),
        cache_dir=cache, synoptic_hours_utc=[0, 12],
        selection="nearest", max_time_delta_hours=6.0,
    )
    assert result is not None
    assert called == [(_date(8, 0), "83779")]
    assert result.attrs["time_delta_hours"] == 5.0


def test_nearest_available_csv_outranks_older_igra2_sounding(
    tmp_path: Path, monkeypatch,
) -> None:
    cache = tmp_path / "radiosonde"
    _write_archive(cache, day=7)
    csv = cache / "2020" / "01" / "radiosonde_83779_20200108_00Z.csv"
    csv.parent.mkdir(parents=True)
    pd.DataFrame({
        "height": [760.0, 3000.0],
        "temperature": [21.0, 10.0],
        "pressure": [930.0, 700.0],
    }).to_csv(csv, index=False)
    monkeypatch.setattr(
        radiosonde, "_require_current_siphon",
        lambda: (_ for _ in ()).throw(AssertionError("Unneeded Siphon.")),
    )
    profile = radiosonde.fetch_wyoming_radiosonde(
        _date(7, 19), "83779", logging.getLogger("test.radio.nearest"),
        cache_dir=cache, synoptic_hours_utc=[0, 12],
        selection="nearest", max_time_delta_hours=12.0,
    )
    assert profile is not None
    assert profile.attrs["target_datetime_utc"] == _date(8, 0).isoformat()
    assert profile.attrs["time_delta_hours"] == 5.0
