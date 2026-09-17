"""R&D tests for Monte Carlo propagation of a fitted Rayleigh boundary.

The productive Level 2 method remains unchanged. These tests verify the random
uncertainty bookkeeping of a window-fitted exact-altitude boundary and preserve
an explicit model-bias counterexample for contaminated windows.
"""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.level2.rayleigh_uncertainty import origin_calibration_uncertainty
from milgrau.level2.window_boundary_mc import window_fitted_boundary_monte_carlo
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def _clean_reference_case() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
]:
    altitude_m = np.arange(300.0, 12000.0 + 15.0, 30.0, dtype=np.float64)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_m)
    beta_molecular, _ = calculate_molecular_profile(
        temperature_k,
        pressure_hpa,
        532.0,
    )
    beta_aerosol = 2.0e-6 * np.exp(-0.5 * ((altitude_m - 2200.0) / 900.0) ** 2)
    beta_aerosol[altitude_m >= 6000.0] = 0.0
    lidar_ratio = np.full_like(altitude_m, 55.0)
    measured_rcs = elastic_lidar_forward_model(
        altitude_m,
        beta_molecular,
        beta_aerosol,
        RAYLEIGH_LIDAR_RATIO_SR,
        lidar_ratio,
    )
    molecular_rcs = elastic_lidar_forward_model(
        altitude_m,
        beta_molecular,
        np.zeros_like(beta_aerosol),
        RAYLEIGH_LIDAR_RATIO_SR,
        lidar_ratio,
    )
    ref_idx = int(np.argmin(np.abs(altitude_m - 9000.0)))
    window = np.flatnonzero(
        (altitude_m >= altitude_m[ref_idx] - 500.0)
        & (altitude_m <= altitude_m[ref_idx] + 500.0)
    )
    return (
        altitude_m,
        beta_molecular,
        beta_aerosol,
        lidar_ratio,
        measured_rcs,
        molecular_rcs,
        ref_idx,
        int(window[0]),
        int(window[-1] + 1),
    )


def test_window_boundary_mc_matches_analytic_independent_fit_uncertainty() -> None:
    """MC spread of the fitted scale should reproduce its analytic propagation."""
    (
        altitude,
        beta_mol,
        _beta_aer,
        lidar_ratio,
        rcs,
        molecular_rcs,
        ref_idx,
        start,
        stop,
    ) = _clean_reference_case()
    rcs_error = 0.04 * rcs
    analytic = origin_calibration_uncertainty(
        rcs[start:stop],
        molecular_rcs[start:stop],
        rcs_error[start:stop],
    )
    expected_boundary_sigma = (
        analytic.uncertainty_independent * molecular_rcs[ref_idx]
    )

    result = window_fitted_boundary_monte_carlo(
        rcs,
        rcs_error,
        molecular_rcs,
        altitude,
        beta_mol,
        lidar_ratio,
        float(beta_mol[ref_idx]),
        ref_idx,
        start,
        stop,
        n_simulations=800,
        random_seed=4815,
    )

    # This tolerance is a Monte Carlo sampling guard, not an observational
    # scientific acceptance threshold.
    assert result.successful_simulations == result.requested_simulations
    assert np.isclose(
        result.boundary_signal_std,
        expected_boundary_sigma,
        rtol=0.12,
    )
    assert np.isclose(
        result.calibration_factor_std,
        analytic.uncertainty_independent,
        rtol=0.12,
    )


def test_window_boundary_uses_multi_bin_information_without_double_counting_noise() -> None:
    """A clean multi-bin window should stabilize X_ref relative to one noisy bin."""
    (
        altitude,
        beta_mol,
        _beta_aer,
        lidar_ratio,
        rcs,
        molecular_rcs,
        ref_idx,
        start,
        stop,
    ) = _clean_reference_case()
    rcs_error = 0.05 * rcs

    result = window_fitted_boundary_monte_carlo(
        rcs,
        rcs_error,
        molecular_rcs,
        altitude,
        beta_mol,
        lidar_ratio,
        float(beta_mol[ref_idx]),
        ref_idx,
        start,
        stop,
        n_simulations=500,
        random_seed=2109,
    )

    assert result.boundary_signal_std < 0.40 * rcs_error[ref_idx]
    assert abs(result.boundary_signal_mean / result.nominal_boundary_signal - 1.0) < 0.01


def test_window_boundary_mc_mean_preserves_known_lower_column_for_clean_window() -> None:
    """Random fitted-boundary propagation should not create a lower-column bias."""
    (
        altitude,
        beta_mol,
        beta_aer,
        lidar_ratio,
        rcs,
        molecular_rcs,
        ref_idx,
        start,
        stop,
    ) = _clean_reference_case()
    rcs_error = 0.01 * rcs

    result = window_fitted_boundary_monte_carlo(
        rcs,
        rcs_error,
        molecular_rcs,
        altitude,
        beta_mol,
        lidar_ratio,
        float(beta_mol[ref_idx]),
        ref_idx,
        start,
        stop,
        n_simulations=500,
        random_seed=7701,
    )
    evaluate = (altitude >= 600.0) & (altitude <= 5500.0)
    relative_l2 = float(
        np.linalg.norm(result.beta_aerosol_mean[evaluate] - beta_aer[evaluate])
        / np.linalg.norm(beta_aer[evaluate])
    )

    assert relative_l2 < 0.03
    assert np.all(np.isfinite(result.beta_aerosol_std[evaluate]))
    assert np.all(result.beta_aerosol_std[evaluate] >= 0.0)


def test_contaminated_window_bias_can_exceed_mc_random_spread() -> None:
    """Small MC spread must not hide a systematic contaminated-window bias."""
    (
        altitude,
        beta_mol,
        _beta_aer,
        lidar_ratio,
        _rcs,
        molecular_rcs,
        ref_idx,
        start,
        stop,
    ) = _clean_reference_case()
    contaminated_aerosol = np.zeros_like(altitude)
    contaminated_aerosol += 1.5e-6 * np.exp(
        -0.5 * ((altitude - 8550.0) / 90.0) ** 2
    )
    contaminated_aerosol[altitude >= 8850.0] = 0.0
    contaminated_rcs = elastic_lidar_forward_model(
        altitude,
        beta_mol,
        contaminated_aerosol,
        RAYLEIGH_LIDAR_RATIO_SR,
        lidar_ratio,
    )
    rcs_error = 0.005 * contaminated_rcs

    result = window_fitted_boundary_monte_carlo(
        contaminated_rcs,
        rcs_error,
        molecular_rcs,
        altitude,
        beta_mol,
        lidar_ratio,
        float(beta_mol[ref_idx]),
        ref_idx,
        start,
        stop,
        n_simulations=500,
        random_seed=991,
    )
    exact_clean_reference_signal = float(contaminated_rcs[ref_idx])
    bias = abs(result.boundary_signal_mean - exact_clean_reference_signal)

    assert contaminated_aerosol[ref_idx] == 0.0
    assert bias > 5.0 * result.boundary_signal_std


def test_window_boundary_mc_rejects_missing_window_support() -> None:
    (
        altitude,
        beta_mol,
        _beta_aer,
        lidar_ratio,
        rcs,
        molecular_rcs,
        ref_idx,
        start,
        stop,
    ) = _clean_reference_case()
    rcs_error = 0.02 * rcs
    rcs_error[start + 2] = np.nan

    with pytest.raises(ValueError, match="Fitting-window RCS uncertainty"):
        window_fitted_boundary_monte_carlo(
            rcs,
            rcs_error,
            molecular_rcs,
            altitude,
            beta_mol,
            lidar_ratio,
            float(beta_mol[ref_idx]),
            ref_idx,
            start,
            stop,
            n_simulations=20,
            random_seed=1,
        )
