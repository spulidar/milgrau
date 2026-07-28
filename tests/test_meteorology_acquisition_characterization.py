"""Frozen characterization of the acquisition path that predates SCI-004B."""

from __future__ import annotations

import inspect
import json
import logging
from datetime import UTC, datetime

import pandas as pd

from milgrau.config.loader import load_config
from milgrau.io import radiosonde as legacy


def _legacy_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pressure": [900.0, 899.5, 800.0],
            "height": [1000.0, 1000.0, 2000.0],
            "temperature": [20.0, 19.8, 10.0],
            "dewpoint": [15.0, 14.8, 5.0],
            "direction": [90.0, 91.0, 100.0],
            "speed": [4.0, 4.1, 5.0],
            "u_wind": [1.0, 1.1, 2.0],
            "v_wind": [2.0, 2.1, 3.0],
            "station": ["SBMT"] * 3,
            "station_number": [83779] * 3,
            "time": [datetime(2026, 7, 5, 12)] * 3,
            "latitude": [-23.5167] * 3,
            "longitude": [-46.6333] * 3,
            "elevation": [722.0] * 3,
            "pw": [25.0] * 3,
        }
    )


def test_installed_siphon_public_result_is_dataframe_but_parser_accesses_raw_text() -> None:
    source = inspect.getsource(legacy.WyomingUpperAir._get_data)
    assert "_get_data_raw" in source
    assert "pd.read_fwf" in source
    assert "df.units" in source
    assert hasattr(legacy.WyomingUpperAir, "_get_data_raw")


def test_legacy_cache_reuses_csv_but_has_no_hash_or_raw_response(tmp_path, monkeypatch) -> None:
    calls = []

    def request_data(_time, _station):
        calls.append(True)
        return _legacy_table()

    monkeypatch.setattr(legacy.WyomingUpperAir, "request_data", request_data)
    measurement = datetime(2026, 7, 5, 12, 30, tzinfo=UTC)
    first = legacy.fetch_wyoming_radiosonde(
        measurement,
        "83779",
        logging.getLogger("legacy-cache-characterization"),
        cache_dir=tmp_path,
    )
    second = legacy.fetch_wyoming_radiosonde(
        measurement,
        "83779",
        logging.getLogger("legacy-cache-characterization"),
        cache_dir=tmp_path,
    )
    assert calls == [True]
    assert first["height"].tolist() == second["height"].tolist() == [1000.0, 2000.0]
    csv_path = next(tmp_path.rglob("*.csv"))
    metadata = json.loads(csv_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert "sha256" not in metadata
    assert "raw_payload_kind" not in metadata
    assert not list(tmp_path.rglob("*.html"))
    assert not list(tmp_path.rglob("*.payload"))


def test_legacy_failure_exhaustion_returns_unstructured_none(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        legacy.WyomingUpperAir,
        "request_data",
        lambda *_args: (_ for _ in ()).throw(OSError("offline")),
    )
    monkeypatch.setattr(legacy.fetch_wyoming_radiosonde.retry, "sleep", lambda _delay: None)
    result = legacy.fetch_wyoming_radiosonde(
        datetime(2026, 7, 5, 12, tzinfo=UTC),
        "83779",
        logging.getLogger("legacy-failure-characterization"),
        cache_dir=tmp_path,
    )
    assert result is None


def test_station_83779_and_legacy_cache_directory_are_configured() -> None:
    config = load_config()
    assert config["radiosonde"]["station_id"] == "83779"
    assert config["radiosonde"]["cache_dir"] == "01-data/wyoming_cache"
