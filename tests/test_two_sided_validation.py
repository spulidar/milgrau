"""Controlled full-column truth with a clean reference and elevated layer."""

from __future__ import annotations

import numpy as np

from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.level2.two_sided_validation import evaluate_two_sided_truth
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def _elevated_layer_case() -> tuple[np.ndarray, ...]:
    altitude = np.unique(
        np.concatenate(
            [
                np.arange(300.0, 6000.0, 30.0),
                np.arange(6000.0, 10_000.0, 60.0),
                np.arange(10_000.0, 15_000.0, 90.0),
                np.arange(15_000.0, 25_000.0, 120.0),
                np.arange(25_000.0, 30_000.0 + 150.0, 150.0),
            ]
        )
    )
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude)
    beta_mol, _ = calculate_molecular_profile(temperature_k, pressure_hpa, 532.0)

    low = 3.2e-6 * np.exp(-(altitude - altitude[0]) / 1700.0)
    low[altitude >= 8000.0] = 0.0
    elevated = 4.5e-7 * np.exp(-0.5 * ((altitude - 15_000.0) / 850.0) ** 2)
    elevated[(altitude < 12_000.0) | (altitude > 18_000.0)] = 0.0
    beta_aer = low + elevated
    lidar_ratio = np.full_like(altitude, 55.0)
    lidar_ratio[elevated > 0.0] = 40.0
    signal = elastic_lidar_forward_model(
        altitude,
        beta_mol,
        beta_aer,
        RAYLEIGH_LIDAR_RATIO_SR,
        lidar_ratio,
    )
    return altitude, beta_mol, beta_aer, lidar_ratio, signal


def test_two_sided_exact_boundary_recovers_low_and_elevated_layers() -> None:
    altitude, beta_mol, beta_aer, lidar_ratio, signal = _elevated_layer_case()
    ref_idx = int(np.argmin(np.abs(altitude - 10_500.0)))
    exact = evaluate_two_sided_truth(
        range_corrected_signal=signal,
        altitude_m=altitude,
        molecular_backscatter=beta_mol,
        aerosol_backscatter_truth=beta_aer,
        aerosol_lidar_ratio_sr=lidar_ratio,
        beta_total_reference=float(beta_mol[ref_idx] + beta_aer[ref_idx]),
        reference_index=ref_idx,
        backward_domain_m=(600.0, float(altitude[ref_idx])),
        forward_domain_m=(float(altitude[ref_idx]), 25_000.0),
    )
    assert exact.backward.support_fraction == 1.0
    assert exact.forward.support_fraction == 1.0
    # The independent forward and inverse trapezoids agree to well below 0.01%
    # on this deliberately nonuniform high-column grid.
    assert exact.backward.relative_l2_error < 1.0e-4
    assert exact.forward.relative_l2_error < 1.0e-4
    assert abs(exact.backward.integrated_backscatter_relative_error) < 1.0e-4
    assert abs(exact.forward.integrated_backscatter_relative_error) < 1.0e-4
    assert exact.forward_endpoint_altitude_m >= 30_000.0


def test_two_sided_truth_separates_lidar_ratio_and_boundary_bias() -> None:
    altitude, beta_mol, beta_aer, _lidar_ratio, signal = _elevated_layer_case()
    ref_idx = int(np.argmin(np.abs(altitude - 10_500.0)))
    wrong_lidar_ratio = evaluate_two_sided_truth(
        range_corrected_signal=signal,
        altitude_m=altitude,
        molecular_backscatter=beta_mol,
        aerosol_backscatter_truth=beta_aer,
        aerosol_lidar_ratio_sr=55.0,
        beta_total_reference=float(beta_mol[ref_idx]),
        reference_index=ref_idx,
        backward_domain_m=(600.0, float(altitude[ref_idx])),
        forward_domain_m=(float(altitude[ref_idx]), 25_000.0),
    )
    residual_boundary = evaluate_two_sided_truth(
        range_corrected_signal=signal,
        altitude_m=altitude,
        molecular_backscatter=beta_mol,
        aerosol_backscatter_truth=beta_aer,
        aerosol_lidar_ratio_sr=55.0,
        beta_total_reference=float(beta_mol[ref_idx] * 1.05),
        reference_index=ref_idx,
        backward_domain_m=(600.0, float(altitude[ref_idx])),
        forward_domain_m=(float(altitude[ref_idx]), 25_000.0),
    )
    assert wrong_lidar_ratio.forward.relative_l2_error > 1.0e-3
    assert residual_boundary.backward.relative_l2_error > 1.0e-3
    assert residual_boundary.forward.relative_l2_error > wrong_lidar_ratio.forward.relative_l2_error
