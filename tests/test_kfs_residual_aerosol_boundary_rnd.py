"""R&D tests for residual aerosol hidden inside a Rayleigh-like boundary window.

These tests do not change productive method-v4 selection. They demonstrate a
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
    *,
    contamination_width_m: float = 2000.0,
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
    high = (
        float(residual_fraction)
        * beta_mol
        * np.exp(
            -0.5
            * (
                (altitude_m - altitude_m[reference_index])
                / float(contamination_width_m)
            )
            ** 2
        )
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
    return altitude_m, beta_mol, beta_aer, measured_rcs, reference_index


def _relative_l2(retrieved: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return float(
        np.linalg.norm(retrieved[mask] - truth[mask])
        / np.linalg.norm(truth[mask])
    )


def _aerosol_free_boundary_profile(
    measured_rcs: np.ndarray,
    altitude_m: np.ndarray,
    beta_mol: np.ndarray,
    reference_index: int,
) -> np.ndarray:
    return fernald_inversion(
        measured_rcs,
        altitude_m,
        beta_mol,
        55.0,
        float(beta_mol[reference_index]),
        reference_index,
        lr_mol=RAYLEIGH_LIDAR_RATIO_SR,
        altitude_units="m",
        min_lidar_ratio=10.0,
        allow_negative_aerosol=False,
        mode="backward",
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
        aerosol_free_assumption = _aerosol_free_boundary_profile(
            measured_rcs,
            altitude,
            beta_mol,
            ref_idx,
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


def test_boundary_placement_stability_is_not_a_purity_certificate() -> None:
    """Broad contamination can be placement-stable while remaining biased."""
    lower_mask: np.ndarray | None = None
    scenarios = {
        "localized": {
            "fraction": 0.50,
            "width_m": 200.0,
        },
        "broad": {
            "fraction": 0.20,
            "width_m": 2000.0,
        },
    }
    diagnostics: dict[str, tuple[float, float]] = {}

    for name, scenario in scenarios.items():
        altitude, beta_mol, beta_aer, measured_rcs, center_index = (
            _synthetic_boundary_case(
                scenario["fraction"],
                contamination_width_m=scenario["width_m"],
            )
        )
        lower_mask = (altitude >= 600.0) & (altitude <= 6000.0)
        center_profile = _aerosol_free_boundary_profile(
            measured_rcs,
            altitude,
            beta_mol,
            center_index,
        )
        center_bias = _relative_l2(center_profile, beta_aer, lower_mask)

        placement_changes: list[float] = []
        for reference_altitude_m in (9500.0, 9750.0, 10250.0, 10500.0):
            ref_idx = int(np.argmin(np.abs(altitude - reference_altitude_m)))
            shifted = _aerosol_free_boundary_profile(
                measured_rcs,
                altitude,
                beta_mol,
                ref_idx,
            )
            placement_changes.append(
                _relative_l2(shifted, center_profile, lower_mask)
            )
        diagnostics[name] = (center_bias, max(placement_changes))

    localized_bias, localized_instability = diagnostics["localized"]
    broad_bias, broad_instability = diagnostics["broad"]
    assert localized_bias > 0.20
    assert localized_instability > 0.20
    assert broad_bias > 0.08
    assert broad_instability < 0.03
