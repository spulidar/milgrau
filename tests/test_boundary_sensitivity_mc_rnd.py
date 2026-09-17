"""R&D tests for nested boundary-scenario Monte Carlo uncertainty."""

from __future__ import annotations

import numpy as np

from milgrau.level2.boundary_sensitivity import (
    boundary_fraction_monte_carlo_sensitivity,
)
from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def _clean_synthetic_profile() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    altitude_m = np.arange(300.0, 10_000.0 + 30.0, 30.0, dtype=np.float64)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_m)
    beta_mol, _ = calculate_molecular_profile(
        temperature_k,
        pressure_hpa,
        532.0,
    )
    beta_aer = 2.5e-6 * np.exp(-(altitude_m - altitude_m[0]) / 1600.0)
    beta_aer[altitude_m >= 7000.0] = 0.0
    lidar_ratio = np.full_like(altitude_m, 55.0)
    rcs = elastic_lidar_forward_model(
        altitude_m,
        beta_mol,
        beta_aer,
        RAYLEIGH_LIDAR_RATIO_SR,
        lidar_ratio,
    )
    reference_index = int(np.argmin(np.abs(altitude_m - 9000.0)))
    return altitude_m, beta_mol, rcs, reference_index


def test_boundary_fraction_mc_keeps_systematic_scenarios_separate_from_random_spread() -> None:
    altitude, beta_mol, rcs, ref_idx = _clean_synthetic_profile()
    rcs_error = np.abs(rcs) * 0.002

    result = boundary_fraction_monte_carlo_sensitivity(
        rcs=rcs,
        rcs_error=rcs_error,
        altitude_m=altitude,
        beta_mol=beta_mol,
        reference_index=ref_idx,
        aerosol_lidar_ratio_sr=55.0,
        aerosol_lidar_ratio_std_sr=2.0,
        residual_fractions=(0.0, 0.02, 0.05),
        n_iterations=40,
        beta_ref_relative_std=0.01,
        seed=1234,
    )

    assert np.array_equal(
        result.residual_aerosol_fraction_of_molecular,
        np.array([0.0, 0.02, 0.05]),
    )
    assert result.aerosol_backscatter_mean.shape == (3, altitude.size)
    assert result.aerosol_backscatter_random_std.shape == (3, altitude.size)
    assert result.backward_valid_count.shape == (3,)
    assert result.backward_valid_fraction.shape == (3,)
    assert np.all((result.backward_valid_fraction >= 0.0) & (result.backward_valid_fraction <= 1.0))
    assert result.n_iterations == 40

    lower = (altitude >= 600.0) & (altitude <= 6000.0)
    difference_0_to_5pct = np.nanmax(
        np.abs(
            result.aerosol_backscatter_mean[2, lower]
            - result.aerosol_backscatter_mean[0, lower]
        )
    )
    assert difference_0_to_5pct > 0.0

    # Within-scenario random dispersion is reported independently from the
    # difference between the declared f scenarios.
    assert np.nanmax(result.aerosol_backscatter_random_std[:, lower]) > 0.0
    assert "nested_boundary_scenarios" in result.uncertainty_scope


def test_boundary_fraction_mc_reuses_seed_for_paired_reproducible_scenarios() -> None:
    altitude, beta_mol, rcs, ref_idx = _clean_synthetic_profile()
    kwargs = dict(
        rcs=rcs,
        rcs_error=np.abs(rcs) * 0.001,
        altitude_m=altitude,
        beta_mol=beta_mol,
        reference_index=ref_idx,
        aerosol_lidar_ratio_sr=55.0,
        aerosol_lidar_ratio_std_sr=1.0,
        residual_fractions=(0.0, 0.05),
        n_iterations=20,
        beta_ref_relative_std=0.01,
        seed=42,
    )

    first = boundary_fraction_monte_carlo_sensitivity(**kwargs)
    second = boundary_fraction_monte_carlo_sensitivity(**kwargs)

    assert np.array_equal(first.backward_valid_count, second.backward_valid_count)
    assert np.allclose(
        first.aerosol_backscatter_mean,
        second.aerosol_backscatter_mean,
        equal_nan=True,
    )
    assert np.allclose(
        first.aerosol_backscatter_random_std,
        second.aerosol_backscatter_random_std,
        equal_nan=True,
    )
