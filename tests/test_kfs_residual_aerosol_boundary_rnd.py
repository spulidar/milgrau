"""R&D tests for residual aerosol hidden inside a Rayleigh-like boundary window.

These tests do not change productive method-v4 selection.  They demonstrate a
specific boundary-condition failure mode: a broad, smooth aerosol contribution
can preserve a Rayleigh-like window shape while the productive assumption
``beta_aer(ref) = 0`` materially biases the backward KFS lower column.
"""

from __future__ import annotations

import numpy as np

from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.kfs import fernald_inversion
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.level2.optical_retrieval import (
    evaluate_rayleigh_reference,
    origin_rayleigh_calibration_factor,
)
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def _synthetic_boundary_case(
    residual_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    altitude_m = np.arange(300.0, 15000.0 + 15.0, 30.0, dtype=np.float64)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_m)
    beta_mol, _ = calculate_molecular_profile(
        temperature_k,
        pressure_hpa,
        532.0,
    )

    # A lower-tropospheric aerosol profile is tapered to zero well below the
    # reference so the controlled high-altitude component owns beta_aer(ref).
    lower = 3.0e-6 * np.exp(-(altitude_m - altitude_m[0]) / 1800.0)
    taper = np.ones_like(altitude_m)
    transition = (altitude_m > 5500.0) & (altitude_m < 7000.0)
    taper[altitude_m >= 7000.0] = 0.0
    taper[transition] = 0.5 * (
        1.0
        + np.cos(
            np.pi * (altitude_m[transition] - 5500.0) / 1500.0
        )
    )
    lower *= taper

    reference_index = int(np.argmin(np.abs(altitude_m - 10000.0)))
    # Broad contamination: locally it follows molecular backscatter closely
    # enough that window-shape QA can still look excellent.
    high = (
        float(residual_fraction)
        * beta_mol
        * np.exp(-0.5 * ((altitude_m - altitude_m[reference_index]) / 2000.0) ** 2)
    )
    beta_aer = lower + high
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
    return altitude_m, beta_mol, beta_aer, measured_rcs, reference_index


def _relative_l2(retrieved: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return float(
        np.linalg.norm(retrieved[mask] - truth[mask])
        / np.linalg.norm(truth[mask])
    )


def test_rayleigh_like_window_can_hide_boundary_aerosol_and_bias_lower_column() -> None:
    """Good window shape does not validate the aerosol-free KFS boundary."""
    biases: list[float] = []
    column_biases: list[float] = []

    for residual_fraction in (0.05, 0.20, 0.50):
        altitude, beta_mol, beta_aer, measured_rcs, ref_idx = (
            _synthetic_boundary_case(residual_fraction)
        )
        molecular_rcs = elastic_lidar_forward_model(
            altitude,
            beta_mol,
            np.zeros_like(beta_aer),
            RAYLEIGH_LIDAR_RATIO_SR,
            np.full_like(altitude, 55.0),
        )
        window_bins = int(round(1000.0 / 30.0))
        factor, _start, _stop, _valid_bins = origin_rayleigh_calibration_factor(
            measured_rcs,
            molecular_rcs,
            altitude,
            ref_idx,
            window_bins,
        )
        qa = evaluate_rayleigh_reference(
            measured_signal=measured_rcs,
            simulated_molecular_signal=molecular_rcs,
            altitude_m=altitude,
            reference_center_idx=ref_idx,
            reference_window_bins=window_bins,
            fit_config={
                "max_relative_slope": 0.25,
                "max_relative_variance": 0.50,
                "min_valid_fraction": 0.50,
            },
            calibration_factor=factor,
        )
        assert qa["success_flag"] == 1

        correct = fernald_inversion(
            measured_rcs,
            altitude,
            beta_mol,
            55.0,
            float(beta_mol[ref_idx] + beta_aer[ref_idx]),
            ref_idx,
            lr_mol=RAYLEIGH_LIDAR_RATIO_SR,
            altitude_units="m",
            min_lidar_ratio=10.0,
            allow_negative_aerosol=False,
            mode="backward",
        )
        aerosol_free_assumption = fernald_inversion(
            measured_rcs,
            altitude,
            beta_mol,
            55.0,
            float(beta_mol[ref_idx]),
            ref_idx,
            lr_mol=RAYLEIGH_LIDAR_RATIO_SR,
            altitude_units="m",
            min_lidar_ratio=10.0,
            allow_negative_aerosol=False,
            mode="backward",
        )
        lower = (altitude >= 600.0) & (altitude <= 6000.0)
        assert _relative_l2(correct, beta_aer, lower) < 0.01

        bias = _relative_l2(aerosol_free_assumption, beta_aer, lower)
        truth_column = float(np.trapezoid(beta_aer[lower], altitude[lower]))
        retrieved_column = float(
            np.trapezoid(aerosol_free_assumption[lower], altitude[lower])
        )
        biases.append(bias)
        column_biases.append(retrieved_column / truth_column - 1.0)

    assert biases[0] < biases[1] < biases[2]
    assert biases[1] > 0.08
    assert biases[2] > 0.20
    assert column_biases[1] < -0.10
    assert column_biases[2] < -0.20
