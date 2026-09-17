"""Lightweight contracts for the selection-aware method-v5 Monte Carlo."""

from __future__ import annotations

import numpy as np

from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.level2.selection_aware_mc_rnd import (
    selection_aware_boundary_monte_carlo_rnd,
)
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def test_selection_aware_mc_reselects_reference_and_reports_support() -> None:
    altitude = np.arange(300.0, 15_000.0 + 30.0, 30.0, dtype=np.float64)
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
    molecular_signal = elastic_lidar_forward_model(
        altitude,
        beta_mol,
        np.zeros_like(beta_aer),
        RAYLEIGH_LIDAR_RATIO_SR,
        55.0,
    )
    error = np.sqrt(
        (0.01 * signal) ** 2
        + (0.03 * signal[np.argmin(np.abs(altitude - 10_000.0))]) ** 2
    )

    result = selection_aware_boundary_monte_carlo_rnd(
        range_corrected_signal=signal,
        range_corrected_signal_error=error,
        molecular_backscatter=beta_mol,
        simulated_molecular_range_corrected_signal=molecular_signal,
        altitude_m=altitude,
        aerosol_lidar_ratio_sr=55.0,
        aerosol_lidar_ratio_std_sr=2.0,
        residual_fractions=(0.0, 0.05),
        n_iterations=12,
        beta_ref_relative_std=0.02,
        min_lidar_ratio_sr=10.0,
        allow_negative_aerosol=False,
        seed=23,
        max_relative_slope=0.25,
        max_relative_variance=0.50,
        min_valid_fraction=0.50,
        progressive_grid_schedule=((0.0, 30.0), (6000.0, 60.0), (10_000.0, 90.0)),
        search_min_altitude_m=9000.0,
        search_max_altitude_m=14_000.0,
        rayleigh_window_m=900.0,
    )

    assert result.n_iterations == 12
    assert result.residual_aerosol_fraction_of_molecular.tolist() == [0.0, 0.05]
    assert result.aerosol_backscatter_mean.shape == (2, result.altitude_m.size)
    assert result.aerosol_backscatter_random_std.shape == (2, result.altitude_m.size)
    assert result.aerosol_backscatter_random_q025.shape == (2, result.altitude_m.size)
    assert result.aerosol_backscatter_random_q975.shape == (2, result.altitude_m.size)
    assert result.aerosol_backscatter_valid_fraction.shape == (2, result.altitude_m.size)
    assert result.selected_reference_index_samples.shape == (12,)
    assert result.selected_reference_altitude_m_samples.shape == (12,)
    assert result.selected_reference_tier_min_altitude_m_samples.shape == (12,)
    assert result.selected_reference_tier_index_samples.shape == (12,)
    assert 0 < result.selection_success_count <= result.n_iterations
    assert np.isclose(
        result.selection_success_fraction,
        result.selection_success_count / result.n_iterations,
    )
    selected = result.selected_reference_altitude_m_samples[
        np.isfinite(result.selected_reference_altitude_m_samples)
    ]
    selected_tiers = result.selected_reference_tier_min_altitude_m_samples[
        np.isfinite(result.selected_reference_tier_min_altitude_m_samples)
    ]
    assert selected.size == result.selection_success_count
    assert selected_tiers.size == result.selection_success_count
    assert np.all((selected >= 9000.0) & (selected <= 14_000.0))
    assert np.all(selected_tiers == 9000.0)
    assert np.all(result.aerosol_backscatter_valid_fraction >= 0.0)
    assert np.all(result.aerosol_backscatter_valid_fraction <= 1.0)

    # Because the same selection/noise/LR random draws are paired across f,
    # the support geometry is identical between boundary scenarios.
    np.testing.assert_array_equal(
        result.aerosol_backscatter_valid_count[0],
        result.aerosol_backscatter_valid_count[1],
    )
