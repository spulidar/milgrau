"""R&D calibration of selection-aware MC using measurement noise only.

This separates the statistical question "does signal-noise propagation through
selection work?" from declared lidar-ratio uncertainty and from the legacy
``beta_ref_relative_std`` parameter.  Both extra sources are set to zero here.
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from milgrau.level2.boundary_sensitivity import boundary_fraction_sensitivity_profiles
from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.high_column_rnd import (
    catalogue_high_column_reference_cells,
    prepare_high_column_profile,
)
from milgrau.level2.high_column_selector import select_minimum_cost_high_column_reference
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.level2.selection_aware_mc_rnd import selection_aware_boundary_monte_carlo_rnd
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model

SCHEDULE = ((0.0, 15.0), (6000.0, 30.0), (10_000.0, 60.0), (15_000.0, 90.0))


def _base():
    altitude = np.arange(300.0, 20_000.0 + 15.0, 15.0, dtype=np.float64)
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
        altitude, beta_mol, beta_aer, RAYLEIGH_LIDAR_RATIO_SR, 55.0
    )
    molecular_signal = elastic_lidar_forward_model(
        altitude, beta_mol, np.zeros_like(beta_aer), RAYLEIGH_LIDAR_RATIO_SR, 55.0
    )
    tiny = np.maximum(np.abs(signal) * 1.0e-12, np.finfo(float).tiny)
    prepared = prepare_high_column_profile(
        range_corrected_signal=signal,
        range_corrected_signal_error=tiny,
        molecular_backscatter=beta_mol,
        altitude_m=altitude,
        uncertainty_mode="independent",
        schedule=SCHEDULE,
    )
    catalogue = catalogue_high_column_reference_cells(
        prepared=prepared,
        native_range_corrected_signal=signal,
        native_range_corrected_signal_error=tiny,
        native_simulated_molecular_signal=molecular_signal,
        native_altitude_m=altitude,
        search_min_altitude_m=10_000.0,
        search_max_altitude_m=19_500.0,
        rayleigh_window_m=900.0,
        max_relative_slope=0.25,
        max_relative_variance=0.50,
        min_valid_fraction=0.50,
    )
    selected = select_minimum_cost_high_column_reference(
        catalogue, min_altitude_m=10_000.0, max_altitude_m=19_500.0
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
    return altitude, beta_mol, signal, molecular_signal, prepared.grid.altitude_m, target


def _one(noise_fraction: float, seed: int) -> dict[str, float]:
    altitude, beta_mol, noiseless, molecular_signal, output_altitude, target = _base()
    idx_10km = int(np.argmin(np.abs(altitude - 10_000.0)))
    floor_sigma = noise_fraction * float(noiseless[idx_10km])
    error = np.sqrt((0.01 * noiseless) ** 2 + floor_sigma**2)
    observed = noiseless + np.random.default_rng(seed).normal(0.0, error)
    mc = selection_aware_boundary_monte_carlo_rnd(
        range_corrected_signal=observed,
        range_corrected_signal_error=error,
        molecular_backscatter=beta_mol,
        simulated_molecular_range_corrected_signal=molecular_signal,
        altitude_m=altitude,
        aerosol_lidar_ratio_sr=55.0,
        aerosol_lidar_ratio_std_sr=0.0,
        residual_fractions=(0.0,),
        n_iterations=80,
        beta_ref_relative_std=0.0,
        min_lidar_ratio_sr=10.0,
        allow_negative_aerosol=False,
        seed=50_000 + seed,
        max_relative_slope=0.25,
        max_relative_variance=0.50,
        min_valid_fraction=0.50,
        uncertainty_mode="independent",
        progressive_grid_schedule=SCHEDULE,
        search_min_altitude_m=10_000.0,
        search_max_altitude_m=19_500.0,
        rayleigh_window_m=900.0,
    )
    mean = mc.aerosol_backscatter_mean[0]
    std = mc.aerosol_backscatter_random_std[0]
    q025 = mc.aerosol_backscatter_random_q025[0]
    q975 = mc.aerosol_backscatter_random_q975[0]
    mask = (
        (output_altitude >= 600.0)
        & (output_altitude <= 6000.0)
        & np.isfinite(target)
        & np.isfinite(mean)
        & np.isfinite(std)
        & np.isfinite(q025)
        & np.isfinite(q975)
    )
    gaussian = (target[mask] >= mean[mask] - 1.96 * std[mask]) & (
        target[mask] <= mean[mask] + 1.96 * std[mask]
    )
    empirical = (target[mask] >= q025[mask]) & (target[mask] <= q975[mask])
    return {
        "gaussian_coverage": float(np.mean(gaussian)),
        "empirical_coverage": float(np.mean(empirical)),
        "selection_success_fraction": float(mc.selection_success_fraction),
        "minimum_lower_valid_fraction": float(
            np.min(mc.aerosol_backscatter_valid_fraction[0][mask])
        ),
    }


def test_measurement_only_selection_aware_coverage() -> None:
    report = {}
    for noise_fraction in (0.15, 0.30):
        rows = [_one(noise_fraction, seed) for seed in (801, 802, 803)]
        report[f"noise10km_{noise_fraction:.2f}"] = {
            "mean_gaussian_coverage": float(np.mean([r["gaussian_coverage"] for r in rows])),
            "mean_empirical_coverage": float(np.mean([r["empirical_coverage"] for r in rows])),
            "mean_selection_success_fraction": float(
                np.mean([r["selection_success_fraction"] for r in rows])
            ),
            "minimum_lower_valid_fraction": float(
                np.min([r["minimum_lower_valid_fraction"] for r in rows])
            ),
        }
    warnings.warn(
        "METHOD_V5_SELECTION_AWARE_MEASUREMENT_ONLY=" + json.dumps(report, sort_keys=True),
        RuntimeWarning,
        stacklevel=1,
    )
