"""Executable contracts for the method-v5 high-column R&D layer."""

from __future__ import annotations

import numpy as np

from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.high_column_rnd import (
    catalogue_high_column_reference_cells,
    contiguous_usable_top_index,
    prepare_high_column_profile,
    run_high_column_reference_cell_monte_carlo,
)
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def _synthetic_profile():
    altitude_m = np.arange(300.0, 12_000.0 + 30.0, 30.0, dtype=np.float64)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_m)
    beta_mol, _ = calculate_molecular_profile(
        temperature_k,
        pressure_hpa,
        532.0,
    )
    beta_aer = 2.0e-6 * np.exp(-(altitude_m - altitude_m[0]) / 1700.0)
    beta_aer[altitude_m >= 7000.0] = 0.0
    lidar_ratio = np.full_like(altitude_m, 55.0)
    measured_rcs = elastic_lidar_forward_model(
        altitude_m,
        beta_mol,
        beta_aer,
        RAYLEIGH_LIDAR_RATIO_SR,
        lidar_ratio,
    )
    molecular_rcs = elastic_lidar_forward_model(
        altitude_m,
        beta_mol,
        np.zeros_like(beta_aer),
        RAYLEIGH_LIDAR_RATIO_SR,
        lidar_ratio,
    )
    return altitude_m, beta_mol, measured_rcs, molecular_rcs


def test_high_column_layer_keeps_rayleigh_qa_native_and_kfs_grid_progressive() -> None:
    altitude, beta_mol, measured_rcs, molecular_rcs = _synthetic_profile()
    error = np.abs(measured_rcs) * 0.002
    schedule = (
        (0.0, 30.0),
        (6_000.0, 60.0),
        (9_000.0, 90.0),
    )

    prepared = prepare_high_column_profile(
        range_corrected_signal=measured_rcs,
        range_corrected_signal_error=error,
        molecular_backscatter=beta_mol,
        altitude_m=altitude,
        uncertainty_mode="independent",
        schedule=schedule,
    )
    assert prepared.grid.n_cells < altitude.size
    assert np.all(prepared.grid.source_count[prepared.grid.altitude_m < 6000.0] == 1)
    assert contiguous_usable_top_index(prepared) == prepared.grid.n_cells - 1

    catalogue = catalogue_high_column_reference_cells(
        prepared=prepared,
        native_range_corrected_signal=measured_rcs,
        native_range_corrected_signal_error=error,
        native_simulated_molecular_signal=molecular_rcs,
        native_altitude_m=altitude,
        search_min_altitude_m=8_000.0,
        search_max_altitude_m=11_000.0,
        rayleigh_window_m=1_000.0,
        max_relative_slope=0.25,
        max_relative_variance=0.50,
        min_valid_fraction=0.50,
    )

    accepted = catalogue.accepted_and_admissible
    assert accepted
    assert all(cell.nominal_path_admissible for cell in accepted)
    assert all(cell.native_rayleigh_candidate.accepted for cell in accepted)
    assert all(8_000.0 <= cell.altitude_m <= 11_000.0 for cell in accepted)
    # Above 9 km the numerical boundary representation is coarser than the
    # native 30 m Rayleigh-QA sampling.
    assert any(cell.effective_resolution_m > 30.0 for cell in accepted)


def test_explicit_reference_cell_runs_nested_boundary_mc_without_auto_selector() -> None:
    altitude, beta_mol, measured_rcs, molecular_rcs = _synthetic_profile()
    error = np.abs(measured_rcs) * 0.001
    schedule = (
        (0.0, 30.0),
        (6_000.0, 60.0),
        (9_000.0, 90.0),
    )
    prepared = prepare_high_column_profile(
        range_corrected_signal=measured_rcs,
        range_corrected_signal_error=error,
        molecular_backscatter=beta_mol,
        altitude_m=altitude,
        uncertainty_mode="independent",
        schedule=schedule,
    )
    catalogue = catalogue_high_column_reference_cells(
        prepared=prepared,
        native_range_corrected_signal=measured_rcs,
        native_range_corrected_signal_error=error,
        native_simulated_molecular_signal=molecular_rcs,
        native_altitude_m=altitude,
        search_min_altitude_m=8_000.0,
        search_max_altitude_m=11_000.0,
        rayleigh_window_m=1_000.0,
        max_relative_slope=0.25,
        max_relative_variance=0.50,
        min_valid_fraction=0.50,
    )
    chosen = catalogue.accepted_and_admissible[-1]

    result = run_high_column_reference_cell_monte_carlo(
        prepared=prepared,
        reference_cell_index=chosen.cell_index,
        aerosol_lidar_ratio_sr=55.0,
        aerosol_lidar_ratio_std_sr=1.0,
        residual_fractions=(0.0, 0.05),
        n_iterations=20,
        beta_ref_relative_std=0.01,
        min_lidar_ratio_sr=10.0,
        allow_negative_aerosol=False,
        seed=7,
    )

    assert result.reference_index == chosen.cell_index
    assert result.aerosol_backscatter_mean.shape == (2, prepared.grid.n_cells)
    assert np.all(result.backward_valid_fraction >= 0.0)
    assert np.all(result.backward_valid_fraction <= 1.0)
    assert np.all(result.backward_valid_count > 0)
    assert not np.allclose(
        result.aerosol_backscatter_mean[0],
        result.aerosol_backscatter_mean[1],
        equal_nan=True,
    )
