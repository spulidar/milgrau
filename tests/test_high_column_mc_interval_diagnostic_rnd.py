"""R&D diagnostic separating MC interval shape from reference selection effects."""

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
from milgrau.level2.high_column_selector import select_minimum_cost_high_column_reference
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def _base_case() -> tuple[np.ndarray, ...]:
    altitude = np.arange(300.0, 20_000.0 + 7.5, 7.5, dtype=np.float64)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude)
    beta_mol, _ = calculate_molecular_profile(temperature_k, pressure_hpa, 532.0)
    beta_aer = 2.4e-6 * np.exp(-(altitude - altitude[0]) / 1800.0)
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


def _coverage(
    *,
    truth: np.ndarray,
    altitude: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    q025: np.ndarray,
    q975: np.ndarray,
) -> tuple[float, float]:
    mask = (
        (altitude >= 600.0)
        & (altitude <= 6000.0)
        & np.isfinite(truth)
        & np.isfinite(mean)
        & np.isfinite(std)
        & np.isfinite(q025)
        & np.isfinite(q975)
    )
    gaussian = (truth[mask] >= mean[mask] - 1.96 * std[mask]) & (
        truth[mask] <= mean[mask] + 1.96 * std[mask]
    )
    empirical = (truth[mask] >= q025[mask]) & (truth[mask] <= q975[mask])
    return float(np.mean(gaussian)), float(np.mean(empirical))


def _one(noise_fraction: float, seed: int) -> dict[str, float] | None:
    native_altitude, beta_mol, beta_aer, noiseless, molecular_signal = _base_case()
    idx_10km = int(np.argmin(np.abs(native_altitude - 10_000.0)))
    floor_sigma = float(noise_fraction) * float(noiseless[idx_10km])
    signal_error = np.sqrt((0.01 * noiseless) ** 2 + floor_sigma**2)
    observed = noiseless + np.random.default_rng(seed).normal(0.0, signal_error)
    prepared = prepare_high_column_profile(
        range_corrected_signal=observed,
        range_corrected_signal_error=signal_error,
        molecular_backscatter=beta_mol,
        altitude_m=native_altitude,
        uncertainty_mode="independent",
    )
    catalogue = catalogue_high_column_reference_cells(
        prepared=prepared,
        native_range_corrected_signal=observed,
        native_range_corrected_signal_error=signal_error,
        native_simulated_molecular_signal=molecular_signal,
        native_altitude_m=native_altitude,
        search_min_altitude_m=8000.0,
        search_max_altitude_m=19_500.0,
        rayleigh_window_m=1000.0,
        max_relative_slope=0.25,
        max_relative_variance=0.50,
        min_valid_fraction=0.50,
    )
    try:
        selected = select_minimum_cost_high_column_reference(
            catalogue,
            min_altitude_m=10_000.0,
            max_altitude_m=19_500.0,
        )
    except ValueError:
        return None

    fixed_index = int(np.argmin(np.abs(prepared.grid.altitude_m - 10_676.25)))
    if not prepared.kfs_cell_usable[fixed_index]:
        return None

    truth = aggregate_to_progressive_grid(
        beta_aer,
        prepared.grid,
        require_positive=False,
    ).values
    rows: dict[str, float] = {"selected_reference_altitude_m": selected.altitude_m}
    for label, reference_index in (
        ("selected", selected.cell_index),
        ("fixed", fixed_index),
    ):
        try:
            mc = run_high_column_reference_cell_monte_carlo(
                prepared=prepared,
                reference_cell_index=reference_index,
                aerosol_lidar_ratio_sr=55.0,
                aerosol_lidar_ratio_std_sr=5.0,
                residual_fractions=(0.0,),
                n_iterations=120,
                beta_ref_relative_std=0.05,
                min_lidar_ratio_sr=10.0,
                allow_negative_aerosol=False,
                seed=20_000 + seed,
            )
        except ValueError:
            return None
        gaussian, empirical = _coverage(
            truth=truth,
            altitude=prepared.grid.altitude_m,
            mean=mc.aerosol_backscatter_mean[0],
            std=mc.aerosol_backscatter_random_std[0],
            q025=mc.aerosol_backscatter_random_q025[0],
            q975=mc.aerosol_backscatter_random_q975[0],
        )
        rows[f"{label}_gaussian_coverage"] = gaussian
        rows[f"{label}_empirical_coverage"] = empirical
        rows[f"{label}_branch_valid_fraction"] = float(mc.backward_valid_fraction[0])
    return rows


def test_interval_shape_vs_selected_reference_diagnostic() -> None:
    report: dict[str, dict[str, float]] = {}
    for noise_fraction in (0.15, 0.30):
        rows = [_one(noise_fraction, seed) for seed in (501, 502, 503, 504)]
        valid = [row for row in rows if row is not None]
        assert valid
        report[f"noise10km_{noise_fraction:.2f}"] = {
            "success_fraction": float(len(valid) / len(rows)),
            "median_selected_reference_altitude_m": float(
                np.median([row["selected_reference_altitude_m"] for row in valid])
            ),
            "mean_selected_gaussian_coverage": float(
                np.mean([row["selected_gaussian_coverage"] for row in valid])
            ),
            "mean_selected_empirical_coverage": float(
                np.mean([row["selected_empirical_coverage"] for row in valid])
            ),
            "mean_fixed_gaussian_coverage": float(
                np.mean([row["fixed_gaussian_coverage"] for row in valid])
            ),
            "mean_fixed_empirical_coverage": float(
                np.mean([row["fixed_empirical_coverage"] for row in valid])
            ),
            "median_selected_branch_valid_fraction": float(
                np.median([row["selected_branch_valid_fraction"] for row in valid])
            ),
            "median_fixed_branch_valid_fraction": float(
                np.median([row["fixed_branch_valid_fraction"] for row in valid])
            ),
        }

    warnings.warn(
        "METHOD_V5_MC_INTERVAL_DIAGNOSTIC=" + json.dumps(report, sort_keys=True),
        RuntimeWarning,
        stacklevel=1,
    )
