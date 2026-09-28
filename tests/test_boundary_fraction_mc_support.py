"""Contracts for altitude-resolved Monte-Carlo validity diagnostics."""

from __future__ import annotations

import numpy as np

from milgrau.level2.boundary_sensitivity import boundary_fraction_monte_carlo_sensitivity
from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def test_nested_boundary_mc_reports_profile_and_complete_branch_validity() -> None:
    altitude = np.arange(300.0, 12_000.0 + 30.0, 30.0, dtype=np.float64)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude)
    beta_mol, _ = calculate_molecular_profile(temperature_k, pressure_hpa, 532.0)
    beta_aer = 2.0e-6 * np.exp(-(altitude - altitude[0]) / 1800.0)
    beta_aer[altitude >= 7000.0] = 0.0
    signal = elastic_lidar_forward_model(
        altitude,
        beta_mol,
        beta_aer,
        RAYLEIGH_LIDAR_RATIO_SR,
        55.0,
    )
    reference_index = int(np.argmin(np.abs(altitude - 10_000.0)))
    error = np.sqrt((0.01 * signal) ** 2 + (0.15 * signal[reference_index]) ** 2)

    result = boundary_fraction_monte_carlo_sensitivity(
        rcs=signal,
        rcs_error=error,
        altitude_m=altitude,
        beta_mol=beta_mol,
        reference_index=reference_index,
        aerosol_lidar_ratio_sr=55.0,
        aerosol_lidar_ratio_std_sr=5.0,
        residual_fractions=(0.0, 0.05),
        n_iterations=40,
        beta_ref_relative_std=0.05,
        min_lidar_ratio_sr=10.0,
        allow_negative_aerosol=False,
        seed=17,
    )

    assert result.aerosol_backscatter_valid_count.shape == (2, altitude.size)
    assert result.aerosol_backscatter_valid_fraction.shape == (2, altitude.size)
    assert np.all(result.aerosol_backscatter_valid_count >= 0)
    assert np.all(result.aerosol_backscatter_valid_count <= result.n_iterations)
    assert np.all(result.aerosol_backscatter_valid_fraction >= 0.0)
    assert np.all(result.aerosol_backscatter_valid_fraction <= 1.0)

    # Backward mode has no retrieval above the selected reference.
    assert np.all(result.aerosol_backscatter_valid_count[:, reference_index + 1 :] == 0)
    # Every complete-branch survivor must also be finite at every retrieved cell.
    assert np.all(
        result.aerosol_backscatter_valid_count[:, : reference_index + 1]
        >= result.backward_valid_count[:, np.newaxis]
    )
    np.testing.assert_allclose(
        result.backward_valid_fraction,
        result.backward_valid_count / result.n_iterations,
    )
    np.testing.assert_allclose(
        result.aerosol_backscatter_valid_fraction,
        result.aerosol_backscatter_valid_count / result.n_iterations,
    )
