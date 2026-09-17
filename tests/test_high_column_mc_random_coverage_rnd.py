"""R&D separation of deterministic method bias and random-MC coverage.

The random Monte-Carlo interval should be calibrated around the deterministic
noiseless result of the declared v5 method.  Difference between that noiseless
method result and physical synthetic truth is reported separately as method /
representation bias rather than silently charged to random uncertainty.
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from milgrau.level2.adaptive_grid import aggregate_to_progressive_grid
from milgrau.level2.boundary_sensitivity import boundary_fraction_sensitivity_profiles
from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.high_column_rnd import (
    catalogue_high_column_reference_cells,
    prepare_high_column_profile,
    run_high_column_reference_cell_monte_carlo,
)
from milgrau.level2.high_column_selector import select_minimum_cost_high_column_reference
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def _case(wavelength_nm: int) -> tuple[np.ndarray, ...]:
    altitude = np.arange(300.0, 20_000.0 + 7.5, 7.5, dtype=np.float64)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude)
    beta_mol, _ = calculate_molecular_profile(
        temperature_k,
        pressure_hpa,
        float(wavelength_nm),
    )
    beta_aer = 2.4e-6 * (532.0 / float(wavelength_nm)) * np.exp(
        -(altitude - altitude[0]) / 1800.0
    )
    taper = np.ones_like(altitude)
    transition = (altitude > 5500.0) & (altitude < 7000.0)
    taper[altitude >= 7000.0] = 0.0
    taper[transition] = 0.5 * (
        1.0 + np.cos(np.pi * (altitude[transition] - 5500.0) / 1500.0)
    )
    beta_aer *= taper
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
    return altitude, beta_mol, beta_aer, signal, molecular_signal


def _catalogue(prepared, signal, error, molecular_signal, altitude):
    return catalogue_high_column_reference_cells(
        prepared=prepared,
        native_range_corrected_signal=signal,
        native_range_corrected_signal_error=error,
        native_simulated_molecular_signal=molecular_signal,
        native_altitude_m=altitude,
        search_min_altitude_m=8000.0,
        search_max_altitude_m=19_500.0,
        rayleigh_window_m=1000.0,
        max_relative_slope=0.25,
        max_relative_variance=0.50,
        min_valid_fraction=0.50,
    )


def _lower_mask(altitude: np.ndarray, *arrays: np.ndarray) -> np.ndarray:
    mask = (altitude >= 600.0) & (altitude <= 6000.0)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def _noiseless_target(wavelength_nm: int):
    altitude, beta_mol, beta_aer, signal, molecular_signal = _case(wavelength_nm)
    tiny_error = np.abs(signal) * 1.0e-12
    prepared = prepare_high_column_profile(
        range_corrected_signal=signal,
        range_corrected_signal_error=tiny_error,
        molecular_backscatter=beta_mol,
        altitude_m=altitude,
        uncertainty_mode="independent",
    )
    catalogue = _catalogue(
        prepared, signal, tiny_error, molecular_signal, altitude
    )
    selected = select_minimum_cost_high_column_reference(
        catalogue,
        min_altitude_m=10_000.0,
        max_altitude_m=19_500.0,
    )
    target = boundary_fraction_sensitivity_profiles(
        rcs=prepared.range_corrected_signal,
        altitude_m=prepared.grid.altitude_m,
        beta_mol=prepared.molecular_backscatter,
        reference_index=selected.cell_index,
        aerosol_lidar_ratio_sr=55.0,
        residual_fractions=(0.0,),
        min_lidar_ratio_sr=10.0,
        allow_negative_aerosol=False,
    ).aerosol_backscatter[0]
    truth = aggregate_to_progressive_grid(
        beta_aer,
        prepared.grid,
        require_positive=False,
    ).values
    mask = _lower_mask(prepared.grid.altitude_m, target, truth)
    method_bias_l2 = float(
        np.linalg.norm(target[mask] - truth[mask]) / np.linalg.norm(truth[mask])
    )
    return (
        altitude,
        beta_mol,
        signal,
        molecular_signal,
        prepared.grid.altitude_m,
        target,
        selected.altitude_m,
        method_bias_l2,
    )


def _one(
    wavelength_nm: int,
    noise_fraction: float,
    observation_seed: int,
) -> dict[str, float] | None:
    (
        altitude,
        beta_mol,
        noiseless,
        molecular_signal,
        progressive_altitude,
        method_target,
        noiseless_reference_altitude,
        method_bias_l2,
    ) = _noiseless_target(wavelength_nm)
    idx_10km = int(np.argmin(np.abs(altitude - 10_000.0)))
    floor_sigma = float(noise_fraction) * float(noiseless[idx_10km])
    error = np.sqrt((0.01 * noiseless) ** 2 + floor_sigma**2)
    observed = noiseless + np.random.default_rng(observation_seed).normal(0.0, error)
    prepared = prepare_high_column_profile(
        range_corrected_signal=observed,
        range_corrected_signal_error=error,
        molecular_backscatter=beta_mol,
        altitude_m=altitude,
        uncertainty_mode="independent",
    )
    catalogue = _catalogue(prepared, observed, error, molecular_signal, altitude)
    try:
        selected = select_minimum_cost_high_column_reference(
            catalogue,
            min_altitude_m=10_000.0,
            max_altitude_m=19_500.0,
        )
    except ValueError:
        return None
    mc = run_high_column_reference_cell_monte_carlo(
        prepared=prepared,
        reference_cell_index=selected.cell_index,
        aerosol_lidar_ratio_sr=55.0,
        aerosol_lidar_ratio_std_sr=5.0,
        residual_fractions=(0.0,),
        n_iterations=120,
        beta_ref_relative_std=0.05,
        min_lidar_ratio_sr=10.0,
        allow_negative_aerosol=False,
        seed=30_000 + observation_seed,
    )
    mean = mc.aerosol_backscatter_mean[0]
    std = mc.aerosol_backscatter_random_std[0]
    q025 = mc.aerosol_backscatter_random_q025[0]
    q975 = mc.aerosol_backscatter_random_q975[0]
    assert np.array_equal(prepared.grid.altitude_m, progressive_altitude)
    mask = _lower_mask(progressive_altitude, method_target, mean, std, q025, q975)
    gaussian = (method_target[mask] >= mean[mask] - 1.96 * std[mask]) & (
        method_target[mask] <= mean[mask] + 1.96 * std[mask]
    )
    empirical = (method_target[mask] >= q025[mask]) & (
        method_target[mask] <= q975[mask]
    )
    return {
        "method_bias_l2_vs_truth": method_bias_l2,
        "noiseless_reference_altitude_m": float(noiseless_reference_altitude),
        "selected_reference_altitude_m": float(selected.altitude_m),
        "gaussian_random_coverage": float(np.mean(gaussian)),
        "empirical_random_coverage": float(np.mean(empirical)),
        "branch_valid_fraction": float(mc.backward_valid_fraction[0]),
        "minimum_lower_valid_fraction": float(
            np.min(mc.aerosol_backscatter_valid_fraction[0][mask])
        ),
    }


def test_random_mc_coverage_is_separate_from_method_bias() -> None:
    report: dict[str, dict[str, float]] = {}
    for wavelength_nm in (355, 532):
        for noise_fraction in (0.05, 0.15, 0.30):
            rows = [
                _one(wavelength_nm, noise_fraction, seed)
                for seed in (601, 602, 603, 604)
            ]
            valid = [row for row in rows if row is not None]
            assert valid
            key = f"{wavelength_nm}nm__noise10km_{noise_fraction:.2f}"
            report[key] = {
                "success_fraction": float(len(valid) / len(rows)),
                "method_bias_l2_vs_truth": float(valid[0]["method_bias_l2_vs_truth"]),
                "noiseless_reference_altitude_m": float(
                    valid[0]["noiseless_reference_altitude_m"]
                ),
                "median_selected_reference_altitude_m": float(
                    np.median([row["selected_reference_altitude_m"] for row in valid])
                ),
                "mean_gaussian_random_coverage": float(
                    np.mean([row["gaussian_random_coverage"] for row in valid])
                ),
                "mean_empirical_random_coverage": float(
                    np.mean([row["empirical_random_coverage"] for row in valid])
                ),
                "median_branch_valid_fraction": float(
                    np.median([row["branch_valid_fraction"] for row in valid])
                ),
                "minimum_lower_valid_fraction": float(
                    np.min([row["minimum_lower_valid_fraction"] for row in valid])
                ),
            }

    warnings.warn(
        "METHOD_V5_RANDOM_COVERAGE_RND=" + json.dumps(report, sort_keys=True),
        RuntimeWarning,
        stacklevel=1,
    )
