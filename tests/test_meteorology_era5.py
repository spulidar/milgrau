"""SCI-004A ERA5 L137 reconstruction and horizontal/temporal interpolation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from milgrau.meteorology.contracts import (
    FallbackFlag,
    HumidityFlag,
    InterpolationFlag,
    PrimarySource,
    ProfileQuality,
    QualityFlag,
    create_atmospheric_profile,
)
from milgrau.meteorology.era5_model_levels import (
    full_level_pressures,
    half_level_pressures,
    model_level_geopotential,
)
from milgrau.meteorology.interpolation import (
    bilinear_interpolate_four_points,
    interpolate_profiles_in_time,
)
from milgrau.meteorology.thermodynamics import virtual_temperature


def test_half_and_full_level_pressure_follow_official_hybrid_definition() -> None:
    half = half_level_pressures(
        np.array([0.0, 1000.0, 0.0]),
        np.array([0.0, 0.5, 1.0]),
        np.log(100000.0),
    )

    np.testing.assert_allclose(half, [0.0, 51000.0, 100000.0], rtol=1e-14)
    np.testing.assert_allclose(full_level_pressures(half), [25500.0, 75500.0])


def test_official_l137_coefficients_match_independent_ecmwf_pressure_rows(
    era5_fixture_payload,
) -> None:
    payload, _ = era5_fixture_payload
    half = half_level_pressures(
        np.asarray(payload["hybrid_a_pa"]),
        np.asarray(payload["hybrid_b"]),
        np.log(101325.0),
    )
    full = full_level_pressures(half)

    assert half.shape == (138,)
    assert half[1] / 100.0 == pytest.approx(0.02000365, abs=5e-9)
    assert half[66] / 100.0 == pytest.approx(137.2703, abs=5e-5)
    assert half[137] / 100.0 == pytest.approx(1013.25, abs=1e-10)
    assert full[136] / 100.0 == pytest.approx(1012.0494, abs=1.1e-4)


def test_geopotential_reconstruction_matches_independent_two_layer_calculation() -> None:
    half = np.array([0.0, 40000.0, 100000.0])
    temperature = np.array([230.0, 290.0])
    humidity = np.array([0.001, 0.01])
    surface_geopotential = 7500.0

    result = model_level_geopotential(half, temperature, humidity, surface_geopotential)

    tv = virtual_temperature(temperature, humidity)
    dlog_lower = np.log(100000.0 / 40000.0)
    alpha_lower = 1.0 - 40000.0 / 60000.0 * dlog_lower
    expected_lower = surface_geopotential + 287.05 * tv[1] * alpha_lower
    lower_half_geopotential = surface_geopotential + 287.05 * tv[1] * dlog_lower
    expected_top = lower_half_geopotential + 287.05 * tv[0] * np.log(2.0)

    np.testing.assert_allclose(result, [expected_top, expected_lower], rtol=2e-15)


def test_era5_fixture_reconstructs_137_top_down_levels_and_bottom_up_contract(
    era5_reconstruction,
) -> None:
    reconstruction = era5_reconstruction
    profile = reconstruction.profile

    assert reconstruction.model_levels_top_down.tolist() == list(range(1, 138))
    assert reconstruction.half_level_pressure_pa_top_down.shape == (138,)
    assert reconstruction.full_level_pressure_pa_top_down.shape == (137,)
    assert np.all(np.diff(reconstruction.full_level_pressure_pa_top_down) > 0.0)
    assert np.all(np.diff(reconstruction.geopotential_m2_s2_top_down) < 0.0)
    assert profile.geometric_altitude_m.shape == (137,)
    assert np.all(np.diff(profile.geometric_altitude_m) > 0.0)
    assert np.all(np.diff(profile.pressure_pa) < 0.0)
    assert np.all(profile.primary_source_flag == int(PrimarySource.ERA5))
    assert np.all(profile.interpolation_flag == int(InterpolationFlag.INTERPOLATED))


def test_era5_virtual_temperature_and_densities_are_physically_conformable(
    era5_reconstruction,
) -> None:
    profile = era5_reconstruction.profile

    assert np.all(profile.virtual_temperature_k >= profile.temperature_k)
    assert np.all(profile.air_density_kg_m3 > 0.0)
    assert np.all(profile.molecular_number_density_m3 > 0.0)
    assert profile.air_density_kg_m3[0] > profile.air_density_kg_m3[-1]


def test_bilinear_interpolation_is_order_invariant_at_center_and_edges() -> None:
    coordinates = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0]])
    values = np.array([0.0, 2.0, 4.0, 6.0])
    permutation = np.array([2, 0, 3, 1])

    assert bilinear_interpolate_four_points(coordinates, values, 1.0, 1.0) == pytest.approx(3.0)
    assert bilinear_interpolate_four_points(
        coordinates[permutation], values[permutation], 1.0, 1.0
    ) == pytest.approx(3.0)
    assert bilinear_interpolate_four_points(coordinates, values, 0.0, 2.0) == pytest.approx(2.0)


def test_bilinear_interpolation_rejects_outside_or_nonrectangular_coordinates() -> None:
    coordinates = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="outside"):
        bilinear_interpolate_four_points(coordinates, np.arange(4.0), 3.0, 1.0)
    coordinates[3] = [1.0, 2.0]
    with pytest.raises(ValueError, match="two-by-two|corner"):
        bilinear_interpolate_four_points(coordinates, np.arange(4.0), 1.0, 1.0)


def _era5_time_shift(profile, time: datetime, pressure_scale: float, temperature_offset: float):
    size = profile.geometric_altitude_m.size
    return create_atmospheric_profile(
        geometric_altitude_m=profile.geometric_altitude_m,
        geopotential_m2_s2=profile.geopotential_m2_s2,
        pressure_pa=profile.pressure_pa * pressure_scale,
        temperature_k=profile.temperature_k + temperature_offset,
        specific_humidity_kg_kg=profile.specific_humidity_kg_kg,
        primary_source_flag=np.full(size, int(PrimarySource.ERA5), dtype=np.int8),
        interpolation_flag=np.full(size, int(InterpolationFlag.DIRECT), dtype=np.int8),
        fallback_flag=np.full(size, int(FallbackFlag.NONE), dtype=np.int8),
        humidity_flag=np.full(size, int(HumidityFlag.MEASURED), dtype=np.int8),
        radiosonde_weight=np.zeros(size),
        quality_flag=np.full(size, int(QualityFlag.VALID), dtype=np.int8),
        nominal_time=time,
        observation_time=time,
        latitude_deg_north=profile.latitude_deg_north,
        longitude_deg_east=profile.longitude_deg_east,
        provider=profile.provider,
        station_or_dataset_id=profile.station_or_dataset_id,
        raw_snapshot_sha256=profile.raw_snapshot_sha256,
        normalizer_version=profile.normalizer_version,
        vertical_coverage_m=profile.vertical_coverage_m,
        profile_quality=ProfileQuality.QUANTITATIVE,
        quantitative_retrieval_allowed=True,
    )


def test_temporal_interpolation_handles_exact_midpoint_and_bounds(era5_reconstruction) -> None:
    before = era5_reconstruction.profile
    after_time = before.observation_time + timedelta(hours=1)
    after = _era5_time_shift(before, after_time, 0.98, 2.0)
    midpoint = before.observation_time + timedelta(minutes=30)

    result = interpolate_profiles_in_time(before, after, midpoint)

    np.testing.assert_allclose(result.pressure_pa, 0.99 * before.pressure_pa)
    np.testing.assert_allclose(result.temperature_k, before.temperature_k + 1.0)
    assert interpolate_profiles_in_time(before, after, before.observation_time) is before
    assert interpolate_profiles_in_time(before, after, after.observation_time) is after
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_profiles_in_time(before, after, before.observation_time - timedelta(seconds=1))


def test_era5_reconstruction_does_not_modify_fixture_arrays(era5_fixture_payload) -> None:
    payload, _ = era5_fixture_payload
    a = np.asarray(payload["hybrid_a_pa"], dtype=np.float64)
    before = a.copy()

    half_level_pressures(a, np.asarray(payload["hybrid_b"]), np.log(93000.0))

    np.testing.assert_array_equal(a, before)
