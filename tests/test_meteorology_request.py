"""SCI-004B immutable request and minimal ERA5-hour planning."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from milgrau.meteorology.request import (
    AcquisitionMode,
    ERA5_MODEL_LEVELS,
    MeteorologyProvider,
    MeteorologyRequest,
    group_era5_hours_by_month,
    hourly_timestamps_for_interval,
    plan_era5_hours,
    surrounding_grid_points,
)
from tests.meteorology_acquisition_helpers import ANALYSIS_TIME, meteorology_request


def test_exact_hour_requests_only_that_analysis() -> None:
    assert plan_era5_hours((ANALYSIS_TIME,)) == (ANALYSIS_TIME,)


def test_intermediate_time_requests_only_immediate_brackets() -> None:
    measurement = datetime(2026, 7, 5, 12, 30, tzinfo=UTC)
    assert plan_era5_hours((measurement,)) == (
        datetime(2026, 7, 5, 12, tzinfo=UTC),
        datetime(2026, 7, 5, 13, tzinfo=UTC),
    )


def test_planning_deduplicates_sorts_and_handles_month_and_year_boundaries() -> None:
    measurements = (
        datetime(2027, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 12, 31, 23, 30, tzinfo=UTC),
        datetime(2026, 12, 31, 23, 30, tzinfo=UTC),
    )
    planned = plan_era5_hours(measurements)
    assert planned == (
        datetime(2026, 12, 31, 23, tzinfo=UTC),
        datetime(2027, 1, 1, 0, tzinfo=UTC),
    )
    assert group_era5_hours_by_month(planned) == (
        ((2026, 12), (datetime(2026, 12, 31, 23, tzinfo=UTC),)),
        ((2027, 1), (datetime(2027, 1, 1, 0, tzinfo=UTC),)),
    )


def test_large_explicit_interval_contains_no_unrequested_subhourly_values() -> None:
    hours = hourly_timestamps_for_interval(
        datetime(2026, 7, 1, 0, 15, tzinfo=UTC),
        datetime(2026, 7, 2, 3, 45, tzinfo=UTC),
    )
    assert len(hours) == 29
    assert all(value.minute == 0 for value in hours)
    assert hours[0] == datetime(2026, 7, 1, 0, tzinfo=UTC)
    assert hours[-1] == datetime(2026, 7, 2, 4, tzinfo=UTC)


def test_planning_rejects_naive_or_reversed_timestamps() -> None:
    with pytest.raises(TypeError, match="timezone-aware"):
        plan_era5_hours((datetime(2026, 7, 1, 0),))
    with pytest.raises(ValueError, match="precede"):
        hourly_timestamps_for_interval(
            datetime(2026, 7, 2, tzinfo=UTC),
            datetime(2026, 7, 1, tzinfo=UTC),
        )


def test_spu_grid_is_exact_four_point_box() -> None:
    assert surrounding_grid_points(-23.5615, -46.7383) == (
        (-23.75, -46.75),
        (-23.75, -46.5),
        (-23.5, -46.75),
        (-23.5, -46.5),
    )


def test_request_canonicalization_is_order_independent_and_secret_free(tmp_path) -> None:
    timestamps = (
        datetime(2026, 7, 5, 13, tzinfo=UTC),
        datetime(2026, 7, 5, 12, tzinfo=UTC),
    )
    first = meteorology_request(tmp_path, measurement_timestamps=timestamps)
    second = meteorology_request(
        tmp_path,
        measurement_timestamps=tuple(reversed(timestamps)),
    )
    assert first.canonical_payload() == second.canonical_payload()
    serialized = str(first.canonical_payload()).lower()
    assert "token" not in serialized
    assert "password" not in serialized


def test_artifact_identity_ignores_operational_mode(tmp_path) -> None:
    auto = meteorology_request(tmp_path, mode=AcquisitionMode.AUTO)
    cache_only = meteorology_request(tmp_path, mode=AcquisitionMode.CACHE_ONLY)
    assert auto.canonical_payload()["acquisition_mode"] == "auto"
    assert cache_only.canonical_payload()["acquisition_mode"] == "cache_only"
    assert auto.artifact_request_payload(
        provider=MeteorologyProvider.ERA5,
        timestamps=auto.era5_hours,
    ) == cache_only.artifact_request_payload(
        provider=MeteorologyProvider.ERA5,
        timestamps=cache_only.era5_hours,
    )


def test_artifact_identity_ignores_unrelated_provider_inputs(tmp_path) -> None:
    combined = meteorology_request(tmp_path)
    era5_only = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.ERA5,
        radiosonde_nominal_times=(),
    )
    assert combined.artifact_request_payload(
        provider=MeteorologyProvider.ERA5,
        timestamps=combined.era5_hours,
    ) == era5_only.artifact_request_payload(
        provider=MeteorologyProvider.ERA5,
        timestamps=era5_only.era5_hours,
    )

    other_measurement = meteorology_request(
        tmp_path,
        measurement_timestamps=(datetime(2026, 7, 5, 20, 30, tzinfo=UTC),),
    )
    assert combined.artifact_request_payload(
        provider=MeteorologyProvider.RADIOSONDE,
        timestamps=(ANALYSIS_TIME,),
    ) == other_measurement.artifact_request_payload(
        provider=MeteorologyProvider.RADIOSONDE,
        timestamps=(ANALYSIS_TIME,),
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"era5_model_levels": ERA5_MODEL_LEVELS[:-1]}, "1 through 137"),
        ({"era5_variables": ("temperature",)}, "exactly"),
        ({"radiosonde_nominal_times": ()}, "explicitly"),
        (
            {"radiosonde_nominal_times": (datetime(2026, 7, 5, 6, tzinfo=UTC),)},
            "00 or 12",
        ),
        ({"max_retries": 2.5}, "positive integer"),
    ],
)
def test_request_rejects_contract_drift(tmp_path, change, message) -> None:
    kwargs = dict(
        site_id="spu",
        latitude_deg_north=-23.5615,
        longitude_deg_east=-46.7383,
        station_altitude_m=760.0,
        measurement_timestamps=(ANALYSIS_TIME,),
        provider=MeteorologyProvider.BOTH,
        mode=AcquisitionMode.AUTO,
        cache_directory=tmp_path,
        radiosonde_nominal_times=(ANALYSIS_TIME,),
    )
    kwargs.update(change)
    with pytest.raises(ValueError, match=message):
        MeteorologyRequest(**kwargs)
