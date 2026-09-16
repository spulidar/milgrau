"""R&D tests for a window-fitted signal value at the exact KFS boundary.

These tests do not change the productive method.  They ask whether the local
Rayleigh window can denoise the *signal value* used at one exact boundary
altitude, and retain an explicit counterexample showing why a contaminated
window must never be used blindly.
"""

from __future__ import annotations

import numpy as np

from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.kfs import fernald_inversion
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.level2.rayleigh_uncertainty import origin_calibration_uncertainty
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def _known_layer_case() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    """Return a known aerosol layer with a clean molecular reference window."""
    altitude_m = np.arange(300.0, 12000.0 + 15.0, 30.0, dtype=np.float64)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_m)
    beta_molecular, _ = calculate_molecular_profile(
        temperature_k,
        pressure_hpa,
        532.0,
    )

    # Smooth tropospheric layer that is exactly zero well below the reference.
    beta_aerosol = 2.0e-6 * np.exp(-0.5 * ((altitude_m - 2200.0) / 900.0) ** 2)
    beta_aerosol[altitude_m >= 6000.0] = 0.0
    aerosol_lidar_ratio = np.full_like(altitude_m, 55.0)

    measured_rcs = elastic_lidar_forward_model(
        altitude_m,
        beta_molecular,
        beta_aerosol,
        RAYLEIGH_LIDAR_RATIO_SR,
        aerosol_lidar_ratio,
    )
    molecular_rcs = elastic_lidar_forward_model(
        altitude_m,
        beta_molecular,
        np.zeros_like(beta_aerosol),
        RAYLEIGH_LIDAR_RATIO_SR,
        aerosol_lidar_ratio,
    )
    reference_index = int(np.argmin(np.abs(altitude_m - 9000.0)))
    return (
        altitude_m,
        beta_molecular,
        beta_aerosol,
        aerosol_lidar_ratio,
        measured_rcs,
        molecular_rcs,
        reference_index,
    )


def _relative_l2(
    retrieved: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(retrieved[mask] - truth[mask])
        / np.linalg.norm(truth[mask])
    )


def _backward_inversion(
    rcs: np.ndarray,
    altitude_m: np.ndarray,
    beta_molecular: np.ndarray,
    aerosol_lidar_ratio: np.ndarray,
    reference_index: int,
) -> np.ndarray:
    return fernald_inversion(
        rcs,
        altitude_m,
        beta_molecular,
        aerosol_lidar_ratio,
        float(beta_molecular[reference_index]),
        reference_index,
        lr_mol=RAYLEIGH_LIDAR_RATIO_SR,
        altitude_units="m",
        min_lidar_ratio=10.0,
        allow_negative_aerosol=True,
        mode="backward",
    )


def test_clean_window_fit_reduces_single_bin_boundary_noise_without_moving_boundary() -> None:
    """A clean local window can denoise X_ref while keeping the exact ref altitude."""
    (
        altitude_m,
        beta_molecular,
        beta_aerosol,
        aerosol_lidar_ratio,
        true_rcs,
        molecular_rcs,
        reference_index,
    ) = _known_layer_case()

    noisy_rcs = true_rcs.copy()
    noisy_rcs[reference_index] *= 1.8

    half_window_m = 500.0
    in_window = (
        (altitude_m >= altitude_m[reference_index] - half_window_m)
        & (altitude_m <= altitude_m[reference_index] + half_window_m)
    )
    # Tiny positive uncertainty is sufficient here because this test exercises
    # the fitted scale itself, not an operational error magnitude.
    fit = origin_calibration_uncertainty(
        noisy_rcs[in_window],
        molecular_rcs[in_window],
        np.full(int(in_window.sum()), 1.0),
    )
    fitted_boundary_signal = (
        fit.calibration_factor * molecular_rcs[reference_index]
    )
    fitted_rcs = noisy_rcs.copy()
    fitted_rcs[reference_index] = fitted_boundary_signal

    exact_noisy = _backward_inversion(
        noisy_rcs,
        altitude_m,
        beta_molecular,
        aerosol_lidar_ratio,
        reference_index,
    )
    window_fitted = _backward_inversion(
        fitted_rcs,
        altitude_m,
        beta_molecular,
        aerosol_lidar_ratio,
        reference_index,
    )
    evaluate = (altitude_m >= 600.0) & (altitude_m <= 5500.0)
    exact_error = _relative_l2(exact_noisy, beta_aerosol, evaluate)
    fitted_error = _relative_l2(window_fitted, beta_aerosol, evaluate)

    assert fitted_boundary_signal < noisy_rcs[reference_index]
    assert abs(fitted_boundary_signal / true_rcs[reference_index] - 1.0) < 0.03
    assert fitted_error < exact_error * 0.20


def test_clean_window_fit_reduces_distributed_zero_mean_reference_window_noise() -> None:
    """Window fitting can suppress distributed local noise, not only one outlier."""
    (
        altitude_m,
        beta_molecular,
        beta_aerosol,
        aerosol_lidar_ratio,
        true_rcs,
        molecular_rcs,
        reference_index,
    ) = _known_layer_case()

    in_window = (
        (altitude_m >= altitude_m[reference_index] - 500.0)
        & (altitude_m <= altitude_m[reference_index] + 500.0)
    )
    window_indices = np.flatnonzero(in_window)
    # Deterministic alternating perturbations preserve positivity and represent
    # a noisy molecular window without introducing a net broad-scale slope.
    multipliers = 1.0 + 0.35 * np.where(
        np.arange(window_indices.size) % 2 == 0,
        1.0,
        -1.0,
    )
    noisy_rcs = true_rcs.copy()
    noisy_rcs[window_indices] *= multipliers
    noisy_rcs[reference_index] *= 1.45

    fit = origin_calibration_uncertainty(
        noisy_rcs[in_window],
        molecular_rcs[in_window],
        np.full(int(in_window.sum()), 1.0),
    )
    fitted_rcs = noisy_rcs.copy()
    fitted_rcs[reference_index] = (
        fit.calibration_factor * molecular_rcs[reference_index]
    )

    exact_noisy = _backward_inversion(
        noisy_rcs,
        altitude_m,
        beta_molecular,
        aerosol_lidar_ratio,
        reference_index,
    )
    window_fitted = _backward_inversion(
        fitted_rcs,
        altitude_m,
        beta_molecular,
        aerosol_lidar_ratio,
        reference_index,
    )
    evaluate = (altitude_m >= 600.0) & (altitude_m <= 5500.0)

    assert _relative_l2(window_fitted, beta_aerosol, evaluate) < _relative_l2(
        exact_noisy,
        beta_aerosol,
        evaluate,
    )


def test_contaminated_window_biases_fitted_boundary_even_when_reference_bin_is_clean() -> None:
    """A local aerosol layer is a counterexample to blind window denoising."""
    (
        altitude_m,
        beta_molecular,
        _beta_aerosol,
        aerosol_lidar_ratio,
        _true_rcs,
        molecular_rcs,
        reference_index,
    ) = _known_layer_case()

    contaminated_aerosol = np.zeros_like(altitude_m)
    contaminated_aerosol += 1.5e-6 * np.exp(
        -0.5 * ((altitude_m - 8550.0) / 90.0) ** 2
    )
    # Keep the exact reference itself molecular while contaminating the lower
    # part of the nominal +/-500 m fitting window.
    contaminated_aerosol[altitude_m >= 8850.0] = 0.0
    contaminated_rcs = elastic_lidar_forward_model(
        altitude_m,
        beta_molecular,
        contaminated_aerosol,
        RAYLEIGH_LIDAR_RATIO_SR,
        aerosol_lidar_ratio,
    )
    clean_molecular_rcs = elastic_lidar_forward_model(
        altitude_m,
        beta_molecular,
        np.zeros_like(contaminated_aerosol),
        RAYLEIGH_LIDAR_RATIO_SR,
        aerosol_lidar_ratio,
    )
    in_window = (
        (altitude_m >= altitude_m[reference_index] - 500.0)
        & (altitude_m <= altitude_m[reference_index] + 500.0)
    )
    fit = origin_calibration_uncertainty(
        contaminated_rcs[in_window],
        clean_molecular_rcs[in_window],
        np.full(int(in_window.sum()), 1.0),
    )
    fitted_boundary = fit.calibration_factor * clean_molecular_rcs[reference_index]

    # The exact reference bin is aerosol-free, but a contaminated fitting
    # window drags the fitted boundary away from its true local signal.  This is
    # why cloud/layer/molecular-window QA remains a prerequisite.
    assert contaminated_aerosol[reference_index] == 0.0
    assert abs(fitted_boundary / contaminated_rcs[reference_index] - 1.0) > 0.02
