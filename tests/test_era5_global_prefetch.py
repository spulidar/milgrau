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
        # Real LIBIDS L0 stores UTC acquisition times as offsets in
        # Raw_Data_Start_Time(time, nb_of_time_scales); it has no time coord.
        times_utc = pd.to_datetime(times, utc=True)
        reference = pd.Timestamp(times_utc[0])
        offsets = np.array(
            [(value - reference).total_seconds() for value in times_utc],
            dtype=np.int32,
        ).reshape(-1, 1)
        native_l0 = xr.Dataset(
            {"Raw_Data_Start_Time": (
                ("time", "nb_of_time_scales"), offsets,
            )},
            attrs={
                "RawData_Start_Date": reference.strftime("%Y%m%d"),
                "RawData_Start_Time_UT": reference.strftime("%H%M%S"),
            },
        )
        assert "time" not in native_l0.coords
        native_l0.to_netcdf(path)
        return path

    files = [
        make_file("one.nc", ["2020-01-07T11:30:00", "2020-01-07T12:20:00"]),
        make_file("two.nc", ["2020-01-07T12:50:00", "2020-01-07T13:15:00"]),
    ]
    assert era5_prefetch.prefetch_selected_level0_atmosphere(
        files, {"level1": {}}, logging.getLogger("test.global.prefetch"),
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
        [tmp_path / "not-needed.nc"], {"level1": {}}, logging.getLogger("test.global.no-era5"),
    )


def test_global_prefetch_uses_decoded_l0_time_for_station_profile_resolution(
    tmp_path: Path, monkeypatch,
) -> None:
    from milgrau.level1.ingestion import _decode_level0_time_axis

    native_path = tmp_path / "native_l0.nc"
    xr.Dataset(
        {
            "Raw_Data_Start_Time": (
                ("time", "nb_of_time_scales"),
                np.array([[0], [300]], dtype=np.int32),
            ),
        },
        attrs={
            "RawData_Start_Date": "20200107",
            "RawData_Start_Time_UT": "113000",
        },
    ).to_netcdf(native_path)

    class Era5:
        def as_io_mapping(self):
            return {"dataset": "test"}

    monkeypatch.setattr(
        era5_prefetch, "resolve_level1_config",
        lambda _: SimpleNamespace(
            atmosphere=SimpleNamespace(
                source_priority=("era5", "ussa76"), era5=Era5(),
            )
        ),
    )

    observed = []

    def check_site(_cfg, dataset):
        # The historical station profile resolver needs the decoded UTC
        # coordinate even when the native file has only a time dimension.
        assert "time" in dataset.coords
        actual = pd.to_datetime(dataset["time"].values)
        assert actual.tolist() == [
            pd.Timestamp("2020-01-07T11:30:00"),
            pd.Timestamp("2020-01-07T11:35:00"),
        ]
        return {"latitude": -23.56, "longitude": -46.74}

    monkeypatch.setattr(era5_prefetch, "resolve_station_site", check_site)
    monkeypatch.setattr(
        era5_prefetch, "prefetch_era5_pressure_level_profiles",
        lambda hours, lat, lon, _logger, *, settings: observed.extend(hours),
    )
    assert era5_prefetch.prefetch_selected_level0_atmosphere(
        [native_path], {"level1": {}}, logging.getLogger("test.l0.native"),
    )
    assert [
        pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M") for t in observed
    ] == ["2020-01-07T11:00", "2020-01-07T12:00"]


def test_global_prefetch_excludes_missing_raw_axis_without_misreading_dimension(
    tmp_path: Path, monkeypatch,
) -> None:
    # A mere time dimension is *not* an acquisition timestamp.
    malformed = tmp_path / "bad_l0.nc"
    xr.Dataset(
        {"dummy": ("time", np.ones(2))},
    ).to_netcdf(malformed)
    monkeypatch.setattr(
        era5_prefetch, "resolve_level1_config",
        lambda _: SimpleNamespace(
            atmosphere=SimpleNamespace(
                source_priority=("era5",),
                era5=SimpleNamespace(as_io_mapping=lambda: {}),
            ),
        ),
    )
    assert not era5_prefetch.prefetch_selected_level0_atmosphere(
        [malformed], {"level1": {}}, logging.getLogger("test.l0.invalid"),
    )
