"""Synthetic P5.4 matrix for Rayleigh-window evidence failure modes.

These tests are R&D only. They do not change productive method v4 or define
high-column thresholds. They exercise whether separate diagnostics respond to
noise, asymmetric contamination, broad contamination and reference placement.
"""

from __future__ import annotations

import numpy as np

from milgrau.level2.rayleigh_uncertainty import origin_calibration_uncertainty


def _window_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    altitude_m = np.arange(8000.0, 10000.0 + 7.5, 7.5, dtype=np.float64)
    molecular_signal = np.exp(-(altitude_m - 8000.0) / 7000.0)
    measured_error = 0.02 * molecular_signal
    reference = (altitude_m >= 8500.0) & (altitude_m <= 9500.0)
    return altitude_m, molecular_signal, measured_error, reference


def _fit_factor(
    measured_signal: np.ndarray,
    molecular_signal: np.ndarray,
    measured_error: np.ndarray,
    mask: np.ndarray,
) -> float:
    return float(
        origin_calibration_uncertainty(
            measured_signal[mask],
            molecular_signal[mask],
            measured_error[mask],
        ).calibration_factor
    )


def _subwindow_disagreement(
    measured_signal: np.ndarray,
    molecular_signal: np.ndarray,
    measured_error: np.ndarray,
    reference: np.ndarray,
) -> float:
    indices = np.flatnonzero(reference)
    midpoint = indices.size // 2
    lower = np.zeros(reference.shape, dtype=bool)
    upper = np.zeros(reference.shape, dtype=bool)
    lower[indices[:midpoint]] = True
    upper[indices[midpoint:]] = True
    full_factor = _fit_factor(measured_signal, molecular_signal, measured_error, reference)
    lower_factor = _fit_factor(measured_signal, molecular_signal, measured_error, lower)
    upper_factor = _fit_factor(measured_signal, molecular_signal, measured_error, upper)
    return abs(lower_factor - upper_factor) / full_factor


def test_wider_noise_sweep_window_fit_remains_less_noisy_than_exact_center_bin() -> None:
    """Averaging gain exists for independent clean noise over a wide amplitude range."""
    altitude_m, molecular_signal, _error, reference = _window_case()
    window_indices = np.flatnonzero(reference)
    center = window_indices[window_indices.size // 2]
    rng = np.random.default_rng(20260917)

    for relative_noise in (0.01, 0.03, 0.10, 0.30):
        exact_errors = []
        fitted_errors = []
        sigma = relative_noise * molecular_signal
        for _ in range(256):
            noisy = molecular_signal + rng.normal(0.0, sigma)
            noisy = np.maximum(noisy, molecular_signal * 1.0e-9)
            factor = _fit_factor(noisy, molecular_signal, sigma, reference)
            exact_errors.append(abs(noisy[center] / molecular_signal[center] - 1.0))
            fitted_errors.append(abs(factor - 1.0))

        assert np.median(fitted_errors) < 0.20 * np.median(exact_errors)


def test_asymmetric_contamination_biases_fit_and_creates_subwindow_disagreement() -> None:
    """One-sided contamination is visible to the half-window disagreement diagnostic."""
    altitude_m, molecular_signal, measured_error, reference = _window_case()
    ratio = 1.0 + 0.25 * np.exp(-0.5 * ((altitude_m - 8600.0) / 220.0) ** 2)
    measured_signal = molecular_signal * ratio

    factor = _fit_factor(measured_signal, molecular_signal, measured_error, reference)
    disagreement = _subwindow_disagreement(
        measured_signal,
        molecular_signal,
        measured_error,
        reference,
    )

    assert factor > 1.08
    assert disagreement > 0.10


def test_broad_symmetric_contamination_can_bias_both_halves_without_disagreement() -> None:
    """Small half-window disagreement cannot certify molecular purity."""
    altitude_m, molecular_signal, measured_error, reference = _window_case()
    ratio = 1.0 + 0.25 * np.exp(-0.5 * ((altitude_m - 9000.0) / 650.0) ** 2)
    measured_signal = molecular_signal * ratio

    factor = _fit_factor(measured_signal, molecular_signal, measured_error, reference)
    disagreement = _subwindow_disagreement(
        measured_signal,
        molecular_signal,
        measured_error,
        reference,
    )

    assert factor > 1.20
    assert disagreement < 0.01


def test_reference_placement_changes_fitted_scale_in_contaminated_profile() -> None:
    """A fitted boundary must expose placement sensitivity rather than hide it."""
    altitude_m, molecular_signal, measured_error, _reference = _window_case()
    ratio = 1.0 + 0.18 * np.exp(-0.5 * ((altitude_m - 8850.0) / 260.0) ** 2)
    measured_signal = molecular_signal * ratio

    factors = []
    for center_m in (8700.0, 8850.0, 9000.0, 9150.0, 9300.0):
        window = (
            (altitude_m >= center_m - 500.0)
            & (altitude_m <= center_m + 500.0)
        )
        factors.append(
            _fit_factor(measured_signal, molecular_signal, measured_error, window)
        )

    assert max(factors) - min(factors) > 0.03
