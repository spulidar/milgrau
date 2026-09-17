"""End-to-end contract for the isolated method-v5 R&D retrieval."""

from __future__ import annotations

import numpy as np

from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.method_v5_rnd import retrieve_method_v5_rnd
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def test_method_v5_rnd_runs_grid_selector_and_nested_mc_without_productive_gate() -> None:
    altitude = np.arange(300.0, 20_000.0 + 30.0, 30.0, dtype=np.float64)
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
    error = np.abs(signal) * 0.002

    result = retrieve_method_v5_rnd(
        range_corrected_signal=signal,
        range_corrected_signal_error=error,
        molecular_backscatter=beta_mol,
        simulated_molecular_range_corrected_signal=molecular_signal,
        altitude_m=altitude,
        aerosol_lidar_ratio_sr=55.0,
        aerosol_lidar_ratio_std_sr=2.0,
        residual_fractions=(0.0, 0.05),
        n_iterations=20,
        beta_ref_relative_std=0.02,
        min_lidar_ratio_sr=10.0,
        allow_negative_aerosol=False,
        seed=9,
        max_relative_slope=0.25,
        max_relative_variance=0.50,
        min_valid_fraction=0.50,
        progressive_grid_schedule=((0.0, 30.0), (6000.0, 60.0), (10_000.0, 90.0)),
        search_min_altitude_m=10_000.0,
        search_max_altitude_m=18_000.0,
        rayleigh_window_m=1000.0,
    )

    assert result.selector_name == "minimum_existing_rayleigh_cost_after_qa_and_path"
    assert result.selected_reference.accepted
    assert result.selected_reference.nominal_path_admissible
    assert 10_000.0 <= result.selected_reference.altitude_m <= 18_000.0
    assert result.monte_carlo.residual_aerosol_fraction_of_molecular.tolist() == [0.0, 0.05]
    assert result.monte_carlo.aerosol_backscatter_mean.shape[0] == 2
    assert result.monte_carlo.aerosol_backscatter_valid_fraction.shape == (
        2,
        result.prepared.grid.n_cells,
    )
    assert np.all(result.monte_carlo.backward_valid_fraction >= 0.0)
    assert np.all(result.monte_carlo.backward_valid_fraction <= 1.0)
