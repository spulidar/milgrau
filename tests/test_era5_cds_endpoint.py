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



def test_era5_prefetch_groups_adjacent_days_with_bounded_overfetch_and_populates_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict] = []

    class FakeClient:
        def __init__(self, *, url: str, quiet: bool) -> None:
            assert url == CDS_API_URL
            assert quiet is True

        def retrieve(self, dataset: str, request: dict, target: str) -> None:
            calls.append(request)
            pressure = np.array([1000.0, 500.0, 100.0])
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
            ds = xr.Dataset(
                data_vars={
                    "t": (
                        ("valid_time", "pressure_level", "latitude", "longitude"),
                        temperature,
                    ),
                    "z": (
                        ("valid_time", "pressure_level", "latitude", "longitude"),
                        geopotential,
                    ),
                },
                coords={
                    "valid_time": np.asarray(hours, dtype="datetime64[ns]"),
                    "pressure_level": pressure,
                    "latitude": np.array([-23.5]),
                    "longitude": np.array([-46.75]),
                },
            )
            ds.to_netcdf(target)

    monkeypatch.setitem(sys.modules, "cdsapi", types.SimpleNamespace(Client=FakeClient))
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
    assert calls[0]["day"] == ["20", "21"]
    assert calls[0]["time"] == ["00:00", "01:00", "21:00", "22:00", "23:00"]
    assert len(available) == 5  # Return only the originally requested hours.
    cached = list((tmp_path / "cache" / "era5").rglob("era5_pressure_levels_*Z_*.nc"))
    assert len(cached) == 10  # Extra validated hours are also cached for reuse.

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


def test_sparse_monthly_request_planner_avoids_cartesian_overfetch() -> None:
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
    # Four days with two different hour sets can be merged with 2x extra
    # hours, significantly reducing CDS submissions.
    assert len(planned) == 1
    assert sum(fields for _, _, _, fields in planned) == 16 * 2 * 3
    covered = []
    for requested_hours, _, request, fields in planned:
        assert len(requested_hours) * 2 * 3 == fields
        assert len(request["day"]) * len(request["time"]) * 2 * 3 == fields
        assert fields <= 80_000
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


def test_era5_monthly_packing_merges_sparse_days_without_exceeding_overfetch() -> None:
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


def test_era5_monthly_packing_limits_overfetch_and_field_budget() -> None:
    hours = [
        datetime(2020, 1, day, hour, tzinfo=timezone.utc)
        for day in range(1, 32)
        for hour in ((11, 12) if day % 2 else (18, 19))
    ]
    planned = plan_era5_requests(
        hours, -23.5, -46.75, _settings(), max_fields=120,
    )
    requested = sum(fields for _, _, _, fields in planned)
    useful = len(hours) * 6
    assert requested <= useful * 2
    assert all(fields <= 120 for _, _, _, fields in planned)
    for _, _, req, fields in planned:
        assert len(req["day"]) * len(req["time"]) * 6 == fields


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
