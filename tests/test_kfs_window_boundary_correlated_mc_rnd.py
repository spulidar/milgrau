"""R&D tests for explicit caller-supplied correlated Rayleigh-window noise."""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.level2.rayleigh_uncertainty import origin_calibration_uncertainty
from milgrau.level2.window_boundary_mc import window_fitted_boundary_monte_carlo
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def _case() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
]:
    altitude = np.arange(300.0, 11000.0, 60.0, dtype=np.float64)
    pressure, temperature = get_standard_atmosphere(altitude)
    beta_mol, _ = calculate_molecular_profile(temperature, pressure, 532.0)
    beta_aer = 1.7e-6 * np.exp(-0.5 * ((altitude - 2100.0) / 800.0) ** 2)
    beta_aer[altitude >= 6000.0] = 0.0
    lidar_ratio = np.full_like(altitude, 55.0)
    rcs = elastic_lidar_forward_model(
        altitude,
        beta_mol,
        beta_aer,
        RAYLEIGH_LIDAR_RATIO_SR,
        lidar_ratio,
    )
    molecular_rcs = elastic_lidar_forward_model(
        altitude,
        beta_mol,
        np.zeros_like(beta_aer),
        RAYLEIGH_LIDAR_RATIO_SR,
        lidar_ratio,
    )
    ref_idx = int(np.argmin(np.abs(altitude - 9000.0)))
    window = np.flatnonzero(
        (altitude >= altitude[ref_idx] - 480.0)
        & (altitude <= altitude[ref_idx] + 480.0)
    )
    return (
        altitude,
        beta_mol,
        lidar_ratio,
        rcs,
        molecular_rcs,
        ref_idx,
        int(window[0]),
        int(window[-1] + 1),
    )


def test_correlated_window_mc_matches_explicit_analytic_covariance_model() -> None:
    """MC must reproduce a fully caller-specified, non-inferred correlation model."""
    altitude, beta_mol, lidar_ratio, rcs, molecular_rcs, ref_idx, start, stop = _case()
    rcs_error = 0.03 * rcs
    window_size = stop - start

    # This equicorrelation matrix is an explicit synthetic fixture, not an
    # inferred operational noise law. Every off-diagonal pair is declared by
    # the test, and the analytic diagnostic receives the equivalent complete
    # lag sequence.
    rho = 0.18
    correlation = np.full((window_size, window_size), rho, dtype=np.float64)
    np.fill_diagonal(correlation, 1.0)
    autocorrelation = np.full(window_size - 1, rho, dtype=np.float64)

    analytic = origin_calibration_uncertainty(
        rcs[start:stop],
        molecular_rcs[start:stop],
        rcs_error[start:stop],
        autocorrelation=autocorrelation,
    )
    expected_boundary_sigma = (
        analytic.uncertainty_autocorrelation_model * molecular_rcs[ref_idx]
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
        n_simulations=1200,
        random_seed=6012,
        window_correlation=correlation,
    )

    assert result.successful_simulations == result.requested_simulations
    assert result.noise_model == (
        "caller_supplied_correlated_window_plus_independent_external_backward_bins"
    )
    assert np.isclose(
        result.calibration_factor_std,
        analytic.uncertainty_autocorrelation_model,
        rtol=0.10,
    )
    assert np.isclose(result.boundary_signal_std, expected_boundary_sigma, rtol=0.10)


def test_positive_window_correlation_reduces_the_naive_independence_gain() -> None:
    altitude, beta_mol, lidar_ratio, rcs, molecular_rcs, ref_idx, start, stop = _case()
    rcs_error = 0.03 * rcs
    n_window = stop - start
    correlation = np.full((n_window, n_window), 0.25, dtype=np.float64)
    np.fill_diagonal(correlation, 1.0)

    independent = window_fitted_boundary_monte_carlo(
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
        n_simulations=700,
        random_seed=77,
    )
    correlated = window_fitted_boundary_monte_carlo(
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
        n_simulations=700,
        random_seed=77,
        window_correlation=correlation,
    )

    assert correlated.boundary_signal_std > 1.5 * independent.boundary_signal_std


@pytest.mark.parametrize(
    "correlation, message",
    [
        (np.array([[1.0, 0.2], [0.1, 1.0]]), "shape"),
        (np.array([[1.0, 1.2], [1.2, 1.0]]), "shape"),
    ],
)
def test_window_correlation_shape_is_strict(
    correlation: np.ndarray,
    message: str,
) -> None:
    altitude, beta_mol, lidar_ratio, rcs, molecular_rcs, ref_idx, start, stop = _case()
    with pytest.raises(ValueError, match=message):
        window_fitted_boundary_monte_carlo(
            rcs,
            0.02 * rcs,
            molecular_rcs,
            altitude,
            beta_mol,
            lidar_ratio,
            float(beta_mol[ref_idx]),
            ref_idx,
            start,
            stop,
            n_simulations=10,
            random_seed=2,
            window_correlation=correlation,
        )


def test_window_correlation_rejects_non_psd_matrix() -> None:
    altitude, beta_mol, lidar_ratio, rcs, molecular_rcs, ref_idx, start, stop = _case()
    n_window = stop - start
    correlation = np.eye(n_window, dtype=np.float64)
    correlation[0, 1] = correlation[1, 0] = 0.9
    correlation[0, 2] = correlation[2, 0] = 0.9
    correlation[1, 2] = correlation[2, 1] = -0.9

    with pytest.raises(ValueError, match="positive semidefinite"):
        window_fitted_boundary_monte_carlo(
            rcs,
            0.02 * rcs,
            molecular_rcs,
            altitude,
            beta_mol,
            lidar_ratio,
            float(beta_mol[ref_idx]),
            ref_idx,
            start,
            stop,
            n_simulations=10,
            random_seed=2,
            window_correlation=correlation,
        )
