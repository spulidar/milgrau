"""Cross-wavelength synthetic challenge for the method-v5 reference floor."""

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
)
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def _case(wavelength_nm: int, kind: str) -> tuple[np.ndarray, ...]:
    altitude = np.arange(300.0, 30_000.0 + 7.5, 7.5, dtype=np.float64)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude)
    beta_mol, _ = calculate_molecular_profile(
        temperature_k,
        pressure_hpa,
        float(wavelength_nm),
    )
    lower = 2.5e-6 * (532.0 / float(wavelength_nm)) * np.exp(
        -(altitude - altitude[0]) / 1800.0
    )
    taper = np.ones_like(altitude)
    transition = (altitude > 5500.0) & (altitude < 7000.0)
    taper[altitude >= 7000.0] = 0.0
    taper[transition] = 0.5 * (
        1.0 + np.cos(np.pi * (altitude[transition] - 5500.0) / 1500.0)
    )
    beta_aer = lower * taper

    if kind == "clean_high_column":
        pass
    elif kind == "broad_residual_13km":
        beta_aer += 0.10 * beta_mol * np.exp(
            -0.5 * ((altitude - 13_000.0) / 2500.0) ** 2
        )
    elif kind == "stratospheric_aerosol_22km":
        beta_aer += 0.20 * beta_mol * np.exp(
            -0.5 * ((altitude - 22_000.0) / 2500.0) ** 2
        )
    else:
        raise ValueError(kind)

    lidar_ratio = np.full_like(altitude, 55.0)
    signal = elastic_lidar_forward_model(
        altitude,
        beta_mol,
        beta_aer,
        RAYLEIGH_LIDAR_RATIO_SR,
        lidar_ratio,
    )
    molecular_signal = elastic_lidar_forward_model(
        altitude,
        beta_mol,
        np.zeros_like(beta_aer),
        RAYLEIGH_LIDAR_RATIO_SR,
        lidar_ratio,
    )
    return altitude, beta_mol, beta_aer, signal, molecular_signal


def _run(
    wavelength_nm: int,
    kind: str,
    floor_m: float,
    seed: int,
) -> dict[str, float] | None:
    altitude, beta_mol, beta_aer, noiseless, molecular_signal = _case(
        wavelength_nm, kind
    )
    idx_25km = int(np.argmin(np.abs(altitude - 25_000.0)))
    background_sigma = 0.50 * float(noiseless[idx_25km])
    signal_sigma = np.sqrt((0.01 * noiseless) ** 2 + background_sigma**2)
    observed = noiseless + np.random.default_rng(seed).normal(0.0, signal_sigma)

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
    eligible = tuple(
        cell
        for cell in catalogue.accepted_and_admissible
        if cell.altitude_m >= floor_m
    )
    if not eligible:
        return None
    chosen = min(
        eligible,
        key=lambda cell: (
            cell.native_rayleigh_candidate.diagnostic_cost,
            cell.altitude_m,
        ),
    )
    retrieved = boundary_fraction_sensitivity_profiles(
        rcs=prepared.range_corrected_signal,
        altitude_m=prepared.grid.altitude_m,
        beta_mol=prepared.molecular_backscatter,
        reference_index=chosen.cell_index,
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
    mask = (
        (prepared.grid.altitude_m >= 600.0)
        & (prepared.grid.altitude_m <= 6000.0)
        & np.isfinite(retrieved)
        & np.isfinite(truth)
    )
    if np.count_nonzero(mask) < 10:
        return None
    rel_l2 = float(
        np.linalg.norm(retrieved[mask] - truth[mask])
        / np.linalg.norm(truth[mask])
    )
    z = prepared.grid.altitude_m[mask]
    true_column = float(np.trapezoid(truth[mask], z))
    retrieved_column = float(np.trapezoid(retrieved[mask], z))
    return {
        "reference_altitude_m": float(chosen.altitude_m),
        "relative_l2_0p6_6km": rel_l2,
        "absolute_column_fraction_error_0p6_6km": abs(
            retrieved_column / true_column - 1.0
        ),
    }


def test_minimum_cost_floor_cross_wavelength_challenge() -> None:
    report: dict[str, dict[str, dict[str, float]]] = {}
    for wavelength in (355, 532):
        for case in (
            "clean_high_column",
            "broad_residual_13km",
            "stratospheric_aerosol_22km",
        ):
            group = f"{wavelength}nm__{case}"
            report[group] = {}
            for floor in (8000.0, 10_000.0, 12_000.0):
                rows = [
                    _run(wavelength, case, floor, seed)
                    for seed in (211, 212, 213, 214)
                ]
                valid = [row for row in rows if row is not None]
                assert valid, f"no valid retrievals for {group} floor={floor}"
                report[group][f"floor_{int(floor)}"] = {
                    "success_fraction": float(len(valid) / len(rows)),
                    "median_reference_altitude_m": float(
                        np.median([row["reference_altitude_m"] for row in valid])
                    ),
                    "median_relative_l2_0p6_6km": float(
                        np.median([row["relative_l2_0p6_6km"] for row in valid])
                    ),
                    "p90_relative_l2_0p6_6km": float(
                        np.percentile(
                            [row["relative_l2_0p6_6km"] for row in valid], 90
                        )
                    ),
                    "median_absolute_column_fraction_error_0p6_6km": float(
                        np.median(
                            [
                                row["absolute_column_fraction_error_0p6_6km"]
                                for row in valid
                            ]
                        )
                    ),
                }

    warnings.warn(
        "METHOD_V5_FLOOR_CROSS_WAVELENGTH_RND="
        + json.dumps(report, sort_keys=True),
        RuntimeWarning,
        stacklevel=1,
    )
