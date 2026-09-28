"""Tests for diagnostic weighted Rayleigh-window calibration."""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.level2.weighted_rayleigh_validation import (
    estimate_shrunk_temporal_covariance,
    evaluate_generalized_rayleigh_fit,
    evaluate_weighted_rayleigh_fit,
)


def test_weighted_fit_downweights_noisy_biased_bins() -> None:
    altitude = np.arange(100, dtype=np.float64) * 30.0
    molecular = np.exp(-altitude / 8000.0)
    truth = 4.2
    measured = truth * molecular
    error = np.full(100, 0.01, dtype=np.float64)
    measured[45:55] += 0.8 * molecular[45:55]
    error[45:55] = 1.0
    fit = evaluate_weighted_rayleigh_fit(
        measured, molecular, error, altitude, center_index=50, window_bins=40
    )
    assert abs(fit.weighted_calibration_factor - truth) < abs(
        fit.unweighted_calibration_factor - truth
    )
    assert fit.weighted_reduced_chi_square <= fit.unweighted_reduced_chi_square
    assert fit.effective_weighted_bins < fit.valid_bins
    assert fit.uncertainty_assumption.endswith("diagnostic_only")


def test_weighted_fit_rejects_unsupported_uncertainty_window() -> None:
    altitude = np.arange(30, dtype=np.float64) * 30.0
    molecular = np.exp(-altitude / 8000.0)
    with pytest.raises(ValueError, match="insufficient valid support"):
        evaluate_weighted_rayleigh_fit(
            3.0 * molecular,
            molecular,
            np.full(30, np.nan),
            altitude,
            center_index=15,
            window_bins=20,
        )


def test_gls_fit_uses_correlated_noise_structure() -> None:
    n_bins = 40
    altitude = np.arange(n_bins, dtype=np.float64) * 30.0
    molecular = np.exp(-altitude / 8000.0)
    truth = 4.2
    rho = 0.97
    indices = np.arange(n_bins)
    covariance = 0.02**2 * rho ** np.abs(indices[:, None] - indices[None, :])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    correlated_disturbance = (
        2.0 * np.sqrt(eigenvalues[-1]) * eigenvectors[:, -1]
    )
    measured = truth * molecular + correlated_disturbance

    diagonal = evaluate_weighted_rayleigh_fit(
        measured,
        molecular,
        np.sqrt(np.diag(covariance)),
        altitude,
        center_index=20,
        window_bins=n_bins,
    )
    gls = evaluate_generalized_rayleigh_fit(
        measured,
        molecular,
        covariance,
        altitude,
        center_index=20,
        window_bins=n_bins,
    )
    assert abs(gls.calibration_factor - truth) < abs(
        diagonal.weighted_calibration_factor - truth
    )
    assert 1.0 <= gls.covariance_effective_rank <= n_bins
    assert gls.covariance_condition_number > 1.0
    assert gls.covariance_assumption.endswith("diagnostic_only")


def test_shrunk_temporal_covariance_is_finite_when_bins_exceed_samples() -> None:
    rng = np.random.default_rng(42)
    common = rng.normal(size=(12, 1))
    samples = common + 0.3 * rng.normal(size=(12, 40))
    covariance = estimate_shrunk_temporal_covariance(samples, shrinkage=0.2)
    assert covariance.shape == (40, 40)
    assert np.all(np.isfinite(covariance))
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
