"""Synthetic uncertainty-coverage study for method-v5 Monte Carlo semantics.

This is R&D evidence rather than a productive validity threshold.  The study
uses a clean high-column atmosphere for which the nominal ``f=0`` boundary is
correct, repeatedly perturbs the observation, selects the v5 reference, and
checks how often the known lower-column truth falls inside mean +/- 1.96 sigma.
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from milgrau.level2.adaptive_grid import aggregate_to_progressive_grid
from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.high_column_rnd import (
    catalogue_high_column_reference_cells,
    prepare_high_column_profile,
    run_high_column_reference_cell_monte_carlo,
)
from milgrau.level2.high_column_selector import (
    select_minimum_cost_high_column_reference,
)
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def _clean_case(wavelength_nm: int) -> tuple[np.ndarray, ...]:
    altitude = np.arange(300.0, 25_500.0 + 7.5, 7.5, dtype=np.float64)
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


def _one_repeated_observation(
    wavelength_nm: int,
    noise_fraction_at_10km: float,
    observation_seed: int,
) -> dict[str, float] | None:
    altitude, beta_mol, beta_aer, noiseless, molecular_signal = _clean_case(
        wavelength_nm
    )
    idx_10km = int(np.argmin(np.abs(altitude - 10_000.0)))
    floor_sigma = float(noise_fraction_at_10km) * float(noiseless[idx_10km])
    signal_sigma = np.sqrt((0.01 * noiseless) ** 2 + floor_sigma**2)
    observed = noiseless + np.random.default_rng(observation_seed).normal(
        0.0, signal_sigma
    )

    prepared = prepare_high_column_profile(
        range_corrected_signal=observed,
        range_corrected_signal_error=signal_sigma,
        molecular_backscatter=beta_mol,
        altitude_m=altitude,
        uncertainty_mode="independent",
    )
    catalogue = catalogue_high_column_reference_cells(
        prepared=prepared,
        native_range_corrected_signal=observed,
        native_range_corrected_signal_error=signal_sigma,
        native_simulated_molecular_signal=molecular_signal,
        native_altitude_m=altitude,
        search_min_altitude_m=8000.0,
        search_max_altitude_m=25_000.0,
        rayleigh_window_m=1000.0,
        max_relative_slope=0.25,
        max_relative_variance=0.50,
        min_valid_fraction=0.50,
    )
    try:
        chosen = select_minimum_cost_high_column_reference(catalogue)
    except ValueError:
        return None

    result = run_high_column_reference_cell_monte_carlo(
        prepared=prepared,
        reference_cell_index=chosen.cell_index,
        aerosol_lidar_ratio_sr=55.0,
        aerosol_lidar_ratio_std_sr=5.0,
        residual_fractions=(0.0,),
        n_iterations=80,
        beta_ref_relative_std=0.05,
        min_lidar_ratio_sr=10.0,
        allow_negative_aerosol=False,
        seed=10_000 + int(observation_seed),
    )
    mean = result.aerosol_backscatter_mean[0]
    std = result.aerosol_backscatter_random_std[0]
    valid_fraction = result.aerosol_backscatter_valid_fraction[0]
    truth = aggregate_to_progressive_grid(
        beta_aer,
        prepared.grid,
        require_positive=False,
    ).values

    lower = (
        (prepared.grid.altitude_m >= 600.0)
        & (prepared.grid.altitude_m <= 6000.0)
        & np.isfinite(mean)
        & np.isfinite(std)
        & np.isfinite(truth)
    )
    if np.count_nonzero(lower) < 10:
        return None
    inside = (
        (truth[lower] >= mean[lower] - 1.96 * std[lower])
        & (truth[lower] <= mean[lower] + 1.96 * std[lower])
    )
    return {
        "reference_altitude_m": float(chosen.altitude_m),
        "lower_profile_coverage_fraction": float(np.mean(inside)),
        "median_lower_mc_valid_fraction": float(np.median(valid_fraction[lower])),
        "minimum_lower_mc_valid_fraction": float(np.min(valid_fraction[lower])),
        "complete_branch_mc_valid_fraction": float(result.backward_valid_fraction[0]),
    }


def test_v5_mc_coverage_vs_valid_fraction_synthetic_matrix() -> None:
    report: dict[str, dict[str, float]] = {}
    for wavelength_nm in (355, 532):
        for noise_fraction in (0.05, 0.15, 0.30):
            rows = [
                _one_repeated_observation(
                    wavelength_nm,
                    noise_fraction,
                    observation_seed,
                )
                for observation_seed in (401, 402, 403, 404)
            ]
            valid = [row for row in rows if row is not None]
            key = f"{wavelength_nm}nm__noise10km_{noise_fraction:.2f}"
            assert valid, f"no successful synthetic v5 retrievals for {key}"
            report[key] = {
                "retrieval_success_fraction": float(len(valid) / len(rows)),
                "median_reference_altitude_m": float(
                    np.median([row["reference_altitude_m"] for row in valid])
                ),
                "mean_lower_profile_coverage_fraction": float(
                    np.mean(
                        [row["lower_profile_coverage_fraction"] for row in valid]
                    )
                ),
                "median_lower_mc_valid_fraction": float(
                    np.median(
                        [row["median_lower_mc_valid_fraction"] for row in valid]
                    )
                ),
                "minimum_lower_mc_valid_fraction": float(
                    np.min(
                        [row["minimum_lower_mc_valid_fraction"] for row in valid]
                    )
                ),
                "median_complete_branch_mc_valid_fraction": float(
                    np.median(
                        [
                            row["complete_branch_mc_valid_fraction"]
                            for row in valid
                        ]
                    )
                ),
            }

    warnings.warn(
        "METHOD_V5_MC_COVERAGE_RND=" + json.dumps(report, sort_keys=True),
        RuntimeWarning,
        stacklevel=1,
    )
