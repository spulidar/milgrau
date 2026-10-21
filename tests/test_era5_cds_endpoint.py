"""Regression tests for ERA5 Climate Data Store routing and concise failures."""

from __future__ import annotations

import logging
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from milgrau.io.era5 import (
    CDS_API_URL,
    fetch_era5_pressure_level_profile,
    prefetch_era5_pressure_level_profiles,
    plan_era5_requests,
)


def _settings() -> dict:
    return {
        "cache_dir": "cache/era5",
        "dataset": "reanalysis-era5-pressure-levels",
        "pressure_levels_hpa": [1000, 500, 100],
        "grid_deg": 0.25,
        "area_half_width_deg": 0.25,
    }


def _write_fake_era5(path: str | Path) -> None:
    pressure = np.array([1000.0, 500.0, 100.0])
    temperature = np.array([290.0, 255.0, 215.0])
    geopotential_height = np.array([100.0, 5500.0, 16000.0])
    ds = xr.Dataset(
        data_vars={
            "t": (("valid_time", "pressure_level", "latitude", "longitude"), temperature[None, :, None, None]),
            "z": (("valid_time", "pressure_level", "latitude", "longitude"), (geopotential_height * 9.80665)[None, :, None, None]),
        },
        coords={
            "valid_time": np.array([np.datetime64("2025-05-10T02:00:00")]),
            "pressure_level": pressure,
            "latitude": np.array([-23.5]),
            "longitude": np.array([-46.75]),
        },
    )
    ds.to_netcdf(path)


def test_era5_client_is_pinned_to_climate_data_store_even_if_environment_points_to_ads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, *, url: str, quiet: bool) -> None:
            captured["url"] = url
            captured["quiet"] = quiet

        def retrieve(self, dataset: str, request: dict, target: str) -> None:
            captured["dataset"] = dataset
            captured["request"] = request
            _write_fake_era5(target)

    monkeypatch.setenv("CDSAPI_URL", "https://ads.atmosphere.copernicus.eu/api")
    monkeypatch.setitem(sys.modules, "cdsapi", types.SimpleNamespace(Client=FakeClient))

    frame = fetch_era5_pressure_level_profile(
        datetime(2025, 5, 10, 2, tzinfo=timezone.utc),
        -23.5607,
        -46.7398,
        logging.getLogger("test-era5-cds"),
        settings=_settings(),
        root_dir=tmp_path,
    )

    assert frame is not None and not frame.empty
    assert captured["url"] == CDS_API_URL
    assert captured["quiet"] is True
    assert captured["dataset"] == "reanalysis-era5-pressure-levels"


def test_era5_failure_warning_is_one_operator_line(tmp_path: Path, monkeypatch, caplog) -> None:
    class FailingClient:
        def __init__(self, *, url: str, quiet: bool) -> None:
            assert url == CDS_API_URL
            assert quiet is True

        def retrieve(self, dataset: str, request: dict, target: str) -> None:
            raise RuntimeError("404 Client Error\nprocess not found\ndataset not found")

    monkeypatch.setitem(sys.modules, "cdsapi", types.SimpleNamespace(Client=FailingClient))
    logger = logging.getLogger("test-era5-compact")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        frame = fetch_era5_pressure_level_profile(
            datetime(2025, 5, 10, 2, tzinfo=timezone.utc),
            -23.5607,
            -46.7398,
            logger,
            settings=_settings(),
            root_dir=tmp_path,
        )

    assert frame is None
    warning = next(record.getMessage() for record in caplog.records if record.levelno == logging.WARNING)
    assert warning == "ERA5 unavailable | 404 Client Error"
    assert "\n" not in warning



def test_era5_prefetch_monthly_grib_caches_only_required_hours(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict] = []
    decoded: dict[str, xr.Dataset] = {}

    class FakeClient:
        def __init__(self, *, url: str, quiet: bool) -> None:
            assert url == CDS_API_URL
            assert quiet is True

        def retrieve(self, dataset: str, request: dict, target: str) -> None:
            assert dataset == "reanalysis-era5-pressure-levels"
            assert request["data_format"] == "grib"
            assert request["download_format"] == "unarchived"
            calls.append(request)
            # CDS returns all Cartesian day-hour combinations in GRIB.
            hours = [
                np.datetime64(
                    f"{request['year'][0]}-{request['month'][0]}-{day}T{raw[:2]}:00:00"
                )
                for day in request["day"] for raw in request["time"]
            ]
            n_time = len(hours)
            temperature = np.tile(
                np.array([290.0, 255.0, 215.0])[None, :, None, None],
                (n_time, 1, 1, 1),
            )
            geopotential_height = np.array([100.0, 5500.0, 16000.0])
            geopotential = np.tile(
                (geopotential_height * 9.80665)[None, :, None, None],
                (n_time, 1, 1, 1),
            )
            decoded[target] = xr.Dataset(
                data_vars={
                    "t": (
                        ("time", "isobaricInhPa", "latitude", "longitude"),
                        temperature,
                    ),
                    "z": (
                        ("time", "isobaricInhPa", "latitude", "longitude"),
                        geopotential,
                    ),
                },
                coords={
                    "time": np.asarray(hours, dtype="datetime64[ns]"),
                    "valid_time": ("time", np.asarray(hours, dtype="datetime64[ns]")),
                    "isobaricInhPa": np.array([1000., 500., 100.]),
                    "latitude": np.array([-23.5]),
                    "longitude": np.array([-46.75]),
                },
            )
            Path(target).write_bytes(b"GRIBsynthetic")


    monkeypatch.setitem(sys.modules, "cdsapi", types.SimpleNamespace(Client=FakeClient))
    from milgrau.io import era5 as era5_module
    monkeypatch.setattr(era5_module, "_open_era5_grib", lambda path: decoded[str(path)])
    requested = [
        datetime(2024, 6, 20, 21, tzinfo=timezone.utc),
        datetime(2024, 6, 20, 22, tzinfo=timezone.utc),
        datetime(2024, 6, 20, 23, tzinfo=timezone.utc),
        datetime(2024, 6, 21, 0, tzinfo=timezone.utc),
        datetime(2024, 6, 21, 1, tzinfo=timezone.utc),
    ]

    available = prefetch_era5_pressure_level_profiles(
        requested,
        -23.5607,
        -46.7398,
        logging.getLogger("test-era5-batch"),
        settings=_settings(),
        root_dir=tmp_path,
    )

    assert len(calls) == 1
    assert calls[0]["data_format"] == "grib"
    assert calls[0]["day"] == ["20", "21"]
    assert calls[0]["time"] == ["00:00", "01:00", "21:00", "22:00", "23:00"]
    assert len(available) == 5  # Return only the originally requested hours.
    cached = list((tmp_path / "cache" / "era5").rglob("era5_pressure_levels_*Z_*.nc"))
    assert len(cached) == 5  # Unneeded Cartesian hours are NOT kept locally.

    calls.clear()
    available_again = prefetch_era5_pressure_level_profiles(
        requested,
        -23.5607,
        -46.7398,
        logging.getLogger("test-era5-batch-cache"),
        settings=_settings(),
        root_dir=tmp_path,
    )
    assert len(available_again) == 5
    assert calls == []


def test_monthly_grib_planner_counts_cartesian_fields() -> None:
    hours = [
        datetime(2020, 1, 7, hour, tzinfo=timezone.utc)
        for hour in (11, 12)
    ] + [
        datetime(2020, 1, 8, hour, tzinfo=timezone.utc)
        for hour in (11, 12)
    ] + [
        datetime(2020, 1, 9, hour, tzinfo=timezone.utc)
        for hour in (15, 16)
    ] + [
        datetime(2020, 1, 10, hour, tzinfo=timezone.utc)
        for hour in (11, 12)
    ]
    planned = plan_era5_requests(
        hours, -23.5, -46.75, _settings(),
    )
    # Days in same UTC month use a single GRIB request, counting all
    # Cartesian day-hour combinations under the 80k-field ceiling.
    assert len(planned) == 1
    assert sum(fields for _, _, _, fields in planned) == 16 * 2 * 3
    covered = []
    for requested_hours, _, request, fields in planned:
        assert len(requested_hours) * 2 * 3 == fields
        assert len(request["day"]) * len(request["time"]) * 2 * 3 == fields
        assert fields <= 80_000
        assert request["data_format"] == "grib"
        for day in request["day"]:
            for time in request["time"]:
                covered.append(f"2020-01-{day}T{time}")
    expected = {dt.strftime("%Y-%m-%dT%H:00") for dt in hours}
    assert expected.issubset(set(covered))
    assert len(covered) <= 2 * len(expected)


def test_sparse_era5_planner_splits_on_field_ceiling() -> None:
    hours = [
        datetime(2020, 1, day, hour, tzinfo=timezone.utc)
        for day in (7, 8, 9) for hour in (11, 12)
    ]
    planned = plan_era5_requests(
        hours, -23.5, -46.75, _settings(), max_fields=24,
    )
    assert len(planned) == 2
    assert sorted(fields for _, _, _, fields in planned) == [12, 24]
    assert sorted(len(entry[2]["day"]) for entry in planned) == [1, 2]


def test_era5_planner_never_crosses_utc_month_boundary() -> None:
    hours = [
        datetime(2020, 1, 31, 23, tzinfo=timezone.utc),
        datetime(2020, 2, 1, 0, tzinfo=timezone.utc),
    ]
    planned = plan_era5_requests(hours, -23.5, -46.75, _settings())
    assert len(planned) == 2
    assert [entry[2]["month"] for entry in planned] == [["01"], ["02"]]


def test_era5_monthly_packing_merges_sparse_days_with_grib() -> None:
    # Thirty days with just two observation hours each: one cheap request,
    # not 30 small requests. The CDS Cartesian selection is accounted for.
    hours = [
        datetime(2020, 6, day, hour, tzinfo=timezone.utc)
        for day in range(1, 31) for hour in (11, 12)
    ]
    requests = plan_era5_requests(hours, -23.5, -46.75, _settings())
    assert len(requests) == 1
    assert requests[0][3] == 60 * 6
    assert requests[0][3] <= 80_000
    assert len(requests[0][2]["day"]) == 30
    assert requests[0][2]["time"] == ["11:00", "12:00"]


def test_era5_monthly_grib_splits_under_field_ceiling() -> None:
    hours = [
        datetime(2020, 1, day, hour, tzinfo=timezone.utc)
        for day in range(1, 32)
        for hour in ((11, 12) if day % 2 else (18, 19))
    ]
    # Two-hour patterns differ but one month request uses the union.
    request = plan_era5_requests(hours, -23.5, -46.75, _settings())
    assert len(request) == 1
    assert request[0][3] == 31 * 4 * 6
    assert request[0][2]["data_format"] == "grib"

    planned = plan_era5_requests(
        hours, -23.5, -46.75, _settings(), max_fields=120,
    )
    assert all(fields <= 120 for _, _, _, fields in planned)
    assert len(planned) == 7
    for _, _, req, fields in planned:
        assert len(req["day"]) * len(req["time"]) * 6 == fields
        assert req["data_format"] == "grib"


def test_era5_monthly_37_levels_all_days_all_hours_fit_80k() -> None:
    from calendar import monthrange

    cfg = dict(_settings())
    cfg["pressure_levels_hpa"] = list(range(1, 38))
    hours = [
        datetime(2024, 10, day, hour, tzinfo=timezone.utc)
        for day in range(1, monthrange(2024, 10)[1] + 1)
        for hour in range(24)
    ]
    planned = plan_era5_requests(hours, -23.5, -46.75, cfg)
    assert len(planned) == 1
    assert planned[0][3] == 31 * 24 * 74 == 55_056
    assert planned[0][2]["data_format"] == "grib"
    assert len(planned[0][2]["time"]) == 24
    assert len(planned[0][2]["day"]) == 31


def test_era5_prefetch_blocks_new_cds_jobs_when_old_job_unresolved(
    tmp_path, monkeypatch,
) -> None:
    import json

    cache_root = tmp_path / "cache" / "era5"
    ledger = cache_root / "requests" / "previous.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps({
            "status": "submitted", "request_id": "previous",
            "expected_hours_utc": ["2020-01-07T11:00:00+00:00"],
        }),
        encoding="utf-8",
    )
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))

        def retrieve(self, *args):
            calls.append(("retrieve", args))

    monkeypatch.setitem(sys.modules, "cdsapi", types.SimpleNamespace(Client=FakeClient))
    with pytest.raises(RuntimeError, match="uncertain remote state"):
        prefetch_era5_pressure_level_profiles(
            [datetime(2020, 1, 7, 11, tzinfo=timezone.utc)],
            -23.5, -46.75, logging.getLogger("test.era5.pending"),
            settings=_settings(), root_dir=tmp_path,
        )
    assert calls == []
    assert json.loads(ledger.read_text(encoding="utf-8"))["status"] == "submitted"


def test_grib_reader_disables_index_sidecars(monkeypatch, tmp_path: Path) -> None:
    from milgrau.io import era5 as era5_module

    captured = {}

    def fake_open_dataset(path, *, engine, backend_kwargs):
        captured["path"] = path
        captured["engine"] = engine
        captured["backend_kwargs"] = backend_kwargs
        return xr.Dataset()

    monkeypatch.setattr(era5_module.xr, "open_dataset", fake_open_dataset)
    grib = tmp_path / "download.part.grib"
    era5_module._open_era5_grib(grib)
    assert captured == {
        "path": grib,
        "engine": "cfgrib",
        "backend_kwargs": {"indexpath": ""},
    }


def test_grib_single_analysis_time_and_pressure_normalization() -> None:
    from milgrau.io.era5 import _era5_grib_hour_to_netcdf, _era5_grib_hours

    dataset = xr.Dataset(
        {
            "t": ("isobaricInhPa", np.array([290., 255., 215.])),
            "z": ("isobaricInhPa", np.array([981., 9_810., 19_620.])),
        },
        coords={
            "isobaricInhPa": [1000, 500, 100],
            "time": np.datetime64("2020-01-07T11:00:00"),
            "step": np.timedelta64(0, "h"),
            "valid_time": np.datetime64("2020-01-07T11:00:00"),
        },
    )
    hours, axis = _era5_grib_hours(dataset)
    assert axis is None
    assert hours.shape == (1,)
    converted = _era5_grib_hour_to_netcdf(dataset, 0, axis)
    assert set(converted.data_vars) == {"t", "z"}
    assert converted["pressure_level"].values.tolist() == [1000, 500, 100]
    assert "valid_time" not in converted.coords
    np.testing.assert_array_equal(converted["t"].values, dataset["t"].values)


def test_grib_requires_both_temperature_and_geopotential() -> None:
    from milgrau.io.era5 import _era5_grib_hour_to_netcdf

    with pytest.raises(ValueError, match="temperature/geopotential"):
        _era5_grib_hour_to_netcdf(
            xr.Dataset(
                {"t": ("isobaricInhPa", [290., 255.])},
                coords={"isobaricInhPa": [1000, 500]},
            ),
            0, None,
        )
