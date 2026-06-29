"""Tests for Rayleigh calibration fit utilities."""

from __future__ import annotations

import numpy as np

from milgrau.physics.rayleigh_fit import fit_rayleigh_scale


def test_weighted_ls_zero_recovers_known_scale_with_heteroscedastic_noise() -> None:
    """WLS through the origin should recover a known multiplicative calibration."""
    altitude_m = np.arange(300, dtype=np.float64) * 7.5
    molecular = np.exp(-altitude_m / 9000.0) + 0.5
    scale_truth = 37.0
    sigma = 0.01 * (1.0 + altitude_m / np.nanmax(altitude_m))
    rng = np.random.default_rng(42)
    measured = scale_truth * molecular + rng.normal(0.0, sigma)

    result = fit_rayleigh_scale(
        measured_signal=measured,
        molecular_signal=molecular,
        altitude_m=altitude_m,
        reference_center_idx=150,
        reference_window_bins=200,
        sigma_measured=sigma,
        method="weighted_ls_zero",
    )

    assert result.success
    assert result.method == "weighted_ls_zero"
    assert result.valid_points == 201
    assert result.valid_fraction == 1.0
    assert abs(result.scale_factor - scale_truth) / scale_truth < 0.005
    assert result.scale_factor_std > 0.0
    assert np.isfinite(result.reduced_chi2)


def test_weighted_huber_zero_reduces_bias_from_positive_outliers() -> None:
    """Huber IRLS should be less biased than WLS when a minority of bins are outliers."""
    altitude_m = np.arange(240, dtype=np.float64) * 7.5
    molecular = 1.0 + 0.2 * np.cos(altitude_m / 400.0)
    scale_truth = 5.0
    sigma = np.full_like(altitude_m, 0.02)
    rng = np.random.default_rng(7)
    measured = scale_truth * molecular + rng.normal(0.0, sigma)
    measured[80:104] += 3.0 * molecular[80:104]

    ls_result = fit_rayleigh_scale(
        measured_signal=measured,
        molecular_signal=molecular,
        altitude_m=altitude_m,
        reference_center_idx=120,
        reference_window_bins=220,
        sigma_measured=sigma,
        method="weighted_ls_zero",
    )
    huber_result = fit_rayleigh_scale(
        measured_signal=measured,
        molecular_signal=molecular,
        altitude_m=altitude_m,
        reference_center_idx=120,
        reference_window_bins=220,
        sigma_measured=sigma,
        method="weighted_huber_zero",
    )

    assert ls_result.success
    assert huber_result.success
    assert abs(huber_result.scale_factor - scale_truth) < abs(ls_result.scale_factor - scale_truth)
    assert abs(huber_result.scale_factor - scale_truth) / scale_truth < 0.02


def test_free_intercept_mode_returns_background_diagnostic() -> None:
    """The free-intercept fit should expose residual background as a diagnostic."""
    altitude_m = np.arange(160, dtype=np.float64) * 7.5
    molecular = np.linspace(0.8, 1.2, altitude_m.size)
    scale_truth = 3.5
    intercept_truth = 0.25
    sigma = np.full_like(altitude_m, 0.01)
    measured = scale_truth * molecular + intercept_truth

    result = fit_rayleigh_scale(
        measured_signal=measured,
        molecular_signal=molecular,
        altitude_m=altitude_m,
        reference_center_idx=80,
        reference_window_bins=120,
        sigma_measured=sigma,
        method="weighted_ls_free_intercept_diagnostic",
    )

    assert result.success
    assert result.method == "weighted_ls_free_intercept_diagnostic"
    assert abs(result.scale_factor - scale_truth) < 1.0e-10
    assert abs(result.intercept_diagnostic - intercept_truth) < 1.0e-10


def test_rayleigh_fit_fails_cleanly_when_window_has_too_few_valid_bins() -> None:
    """Invalid or masked windows should fail without raising and with a reason string."""
    altitude_m = np.arange(50, dtype=np.float64) * 7.5
    molecular = np.ones_like(altitude_m)
    measured = 2.0 * molecular
    valid_mask = np.zeros_like(altitude_m, dtype=bool)
    valid_mask[25] = True

    result = fit_rayleigh_scale(
        measured_signal=measured,
        molecular_signal=molecular,
        altitude_m=altitude_m,
        reference_center_idx=25,
        reference_window_bins=20,
        valid_mask=valid_mask,
        method="weighted_ls_zero",
    )

    assert not result.success
    assert result.valid_points == 1
    assert result.reason
