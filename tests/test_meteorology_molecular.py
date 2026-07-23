"""Independent SCI-004A validation of 355/532 nm molecular coefficients."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from milgrau.meteorology.blending import blend_radiosonde_and_era5
from milgrau.meteorology.contracts import (
    FallbackFlag,
    HumidityFlag,
    InterpolationFlag,
    PrimarySource,
    ProfileQuality,
    QualityFlag,
    create_atmospheric_profile,
)
from milgrau.meteorology.molecular import molecular_optical_profile
from milgrau.meteorology.standard_atmosphere import build_standard_atmosphere_profile
from milgrau.meteorology.thermodynamics import (
    hydrostatic_pressure_profile,
    virtual_temperature,
)

_BOLTZMANN_J_K = 1.380649e-23
_BUCHOLTZ_SIGMA_M2 = {355: 2.755150046629428e-30, 532: 5.164829874033986e-31}
_BUCHOLTZ_DEPOLARIZATION = {355: 0.0301, 532: 0.02842}


def _independent_stp_coefficients(wavelength_nm: int) -> tuple[float, float]:
    number_density = 101325.0 / (_BOLTZMANN_J_K * 288.15)
    extinction = number_density * _BUCHOLTZ_SIGMA_M2[wavelength_nm]
    rho = _BUCHOLTZ_DEPOLARIZATION[wavelength_nm]
    gamma = rho / (2.0 - rho)
    phase_pi = 3.0 * ((1.0 + 3.0 * gamma) + (1.0 - gamma)) / (
        4.0 * (1.0 + 2.0 * gamma)
    )
    backscatter = extinction * phase_pi / (4.0 * np.pi)
    return extinction, backscatter


@pytest.mark.parametrize("wavelength_nm", [355, 532])
def test_molecular_coefficients_match_independent_stp_analytic_values(
    wavelength_nm: int,
) -> None:
    profile = build_standard_atmosphere_profile(
        np.array([0.0, 100.0]),
        nominal_time=datetime(2026, 7, 5, 12, tzinfo=UTC),
        latitude_deg_north=0.0,
        longitude_deg_east=0.0,
    )

    result = molecular_optical_profile(profile, wavelength_nm)
    expected_extinction, expected_backscatter = _independent_stp_coefficients(wavelength_nm)

    assert result.extinction_m_inv[0] == pytest.approx(expected_extinction, rel=2e-10)
    assert result.backscatter_m_inv_sr_inv[0] == pytest.approx(
        expected_backscatter, rel=2e-10
    )
    assert result.lidar_ratio_sr[0] == pytest.approx(
        expected_extinction / expected_backscatter, rel=2e-15
    )


def test_molecular_355_532_spectral_relation_and_transmission() -> None:
    profile = build_standard_atmosphere_profile(
        np.arange(0.0, 12001.0, 100.0),
        nominal_time=datetime(2026, 7, 5, 12, tzinfo=UTC),
        latitude_deg_north=-23.5615,
        longitude_deg_east=-46.7383,
    )
    molecular_355 = molecular_optical_profile(profile, 355.0)
    molecular_532 = molecular_optical_profile(profile, 532.0)

    assert molecular_355.extinction_m_inv[0] / molecular_532.extinction_m_inv[0] == pytest.approx(
        _BUCHOLTZ_SIGMA_M2[355] / _BUCHOLTZ_SIGMA_M2[532], rel=2e-15
    )
    assert molecular_355.backscatter_m_inv_sr_inv[0] > molecular_532.backscatter_m_inv_sr_inv[0]
    assert molecular_355.two_way_transmission[-1] < molecular_532.two_way_transmission[-1]
    assert molecular_355.two_way_transmission[0] == 1.0


def _simple_profile(altitude: np.ndarray, pressure: np.ndarray, temperature: np.ndarray, humidity: np.ndarray):
    size = altitude.size
    return create_atmospheric_profile(
        geometric_altitude_m=altitude,
        pressure_pa=pressure,
        temperature_k=temperature,
        specific_humidity_kg_kg=humidity,
        primary_source_flag=np.full(size, int(PrimarySource.ERA5), dtype=np.int8),
        interpolation_flag=np.full(size, int(InterpolationFlag.DIRECT), dtype=np.int8),
        fallback_flag=np.full(size, int(FallbackFlag.NONE), dtype=np.int8),
        humidity_flag=np.full(size, int(HumidityFlag.MEASURED), dtype=np.int8),
        radiosonde_weight=np.zeros(size),
        quality_flag=np.full(size, int(QualityFlag.VALID), dtype=np.int8),
        nominal_time=datetime(2026, 7, 5, 12, tzinfo=UTC),
        observation_time=datetime(2026, 7, 5, 12, tzinfo=UTC),
        latitude_deg_north=-23.5615,
        longitude_deg_east=-46.7383,
        provider="analytic test",
        station_or_dataset_id="analytic",
        raw_snapshot_sha256="1" * 64,
        normalizer_version="analytic-v1",
        vertical_coverage_m=(float(altitude[0]), float(altitude[-1])),
        profile_quality=ProfileQuality.QUANTITATIVE,
        quantitative_retrieval_allowed=True,
    )


def test_moist_hydrostatic_profile_differs_from_dry_low_troposphere() -> None:
    altitude = np.arange(0.0, 3001.0, 100.0)
    temperature = 296.0 - 0.006 * altitude
    dry_q = np.zeros(altitude.shape)
    moist_q = 0.016 * np.exp(-altitude / 1800.0)
    dry_pressure = hydrostatic_pressure_profile(
        altitude, virtual_temperature(temperature, dry_q), 101325.0
    )
    moist_pressure = hydrostatic_pressure_profile(
        altitude, virtual_temperature(temperature, moist_q), 101325.0
    )
    dry = _simple_profile(altitude, dry_pressure, temperature, dry_q)
    moist = _simple_profile(altitude, moist_pressure, temperature, moist_q)

    assert moist.air_density_kg_m3[0] < dry.air_density_kg_m3[0]
    assert moist.pressure_pa[-1] > dry.pressure_pa[-1]
    assert molecular_optical_profile(moist, 532.0).extinction_m_inv[-1] > molecular_optical_profile(
        dry, 532.0
    ).extinction_m_inv[-1]


def test_radiosonde_era5_and_hybrid_all_generate_355_532_profiles(
    complete_radiosonde_normalization, era5_reconstruction
) -> None:
    target = np.arange(800.0, 12001.0, 200.0)
    hybrid = blend_radiosonde_and_era5(
        complete_radiosonde_normalization.profile,
        era5_reconstruction.profile,
        target,
        blend_width_m=1000.0,
    ).profile

    for wavelength in (355.0, 532.0):
        era5_molecular = molecular_optical_profile(era5_reconstruction.profile, wavelength)
        hybrid_molecular = molecular_optical_profile(hybrid, wavelength)
        assert np.isfinite(era5_molecular.extinction_m_inv).all()
        assert np.isfinite(hybrid_molecular.extinction_m_inv).all()
        assert hybrid_molecular.extinction_m_inv.shape == target.shape


def test_molecular_function_does_not_mutate_normalized_profile(era5_reconstruction) -> None:
    profile = era5_reconstruction.profile
    number_density = profile.molecular_number_density_m3.copy()

    molecular_optical_profile(profile, 532.0)

    np.testing.assert_array_equal(profile.molecular_number_density_m3, number_density)
