"""Global preflight only reads L0 times and deduplicates ERA5 hours."""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.level1 import era5_prefetch


def test_global_prefetch_deduplicates_overlapping_sessions_and_brackets_utc(
    tmp_path: Path, monkeypatch,
) -> None:
    calls = []

    class Era5:
        def as_io_mapping(self):
            return {"dataset": "test"}

    policy = SimpleNamespace(
        atmosphere=SimpleNamespace(
            source_priority=("era5", "ussa76"), era5=Era5(),
        ),
    )
    monkeypatch.setattr(era5_prefetch, "resolve_level1_config", lambda _: policy)
    monkeypatch.setattr(
        era5_prefetch, "resolve_station_site",
        lambda _config, _dataset: {"latitude": -23.56, "longitude": -46.74},
    )

    def record(times, lat, lon, logger, *, settings):
        calls.append((times, lat, lon, settings))
        return set(times)

    monkeypatch.setattr(
        era5_prefetch, "prefetch_era5_pressure_level_profiles", record,
    )

    def make_file(name, times):
        path = tmp_path / name
        xr.Dataset(coords={"time": np.array(times, dtype="datetime64[ns]")}).to_netcdf(path)
        return path

    files = [
        make_file("one.nc", ["2020-01-07T11:30:00", "2020-01-07T12:20:00"]),
        make_file("two.nc", ["2020-01-07T12:50:00", "2020-01-07T13:15:00"]),
    ]
    assert era5_prefetch.prefetch_selected_level0_atmosphere(
        files, {}, logging.getLogger("test.global.prefetch"),
    )
    assert len(calls) == 1
    times, lat, lon, settings = calls[0]
    assert (lat, lon) == (-23.56, -46.74)
    assert settings == {"dataset": "test"}
    assert [pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M") for t in times] == [
        "2020-01-07T11:00", "2020-01-07T12:00",
        "2020-01-07T13:00", "2020-01-07T14:00",
    ]


def test_global_prefetch_disabled_without_era5(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        era5_prefetch, "resolve_level1_config",
        lambda _: SimpleNamespace(atmosphere=SimpleNamespace(
            source_priority=("ussa76",), era5=None,
        )),
    )
    assert not era5_prefetch.prefetch_selected_level0_atmosphere(
        [tmp_path / "not-needed.nc"], {}, logging.getLogger("test.global.no-era5"),
    )
