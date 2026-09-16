"""Tests for explicit Rayleigh-window calibration uncertainty diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.level2.rayleigh_uncertainty import origin_calibration_uncertainty


def test_origin_calibration_uncertainty_separates_dependence_limits() -> None:
    molecular = np.ones(4)
    measured = np.full(4, 2.0)
    error = np.ones(4)

    diagnostic = origin_calibration_uncertainty(measured, molecular, error)

    assert diagnostic.calibration_factor == pytest.approx(2.0)
    assert diagnostic.valid_bins == 4
    assert diagnostic.uncertainty_independent == pytest.approx(0.5)
    assert diagnostic.uncertainty_fully_correlated == pytest.approx(1.0)
    assert diagnostic.snr_independent == pytest.approx(4.0)
    assert diagnostic.snr_fully_correlated == pytest.approx(2.0)
    assert np.isnan(diagnostic.uncertainty_autocorrelation_model)


def test_zero_lag_correlation_model_matches_independent_case() -> None:
    molecular = np.ones(4)
    measured = np.full(4, 2.0)
    error = np.ones(4)

    diagnostic = origin_calibration_uncertainty(
        measured,
        molecular,
        error,
        autocorrelation=np.zeros(3),
    )

    assert diagnostic.uncertainty_autocorrelation_model == pytest.approx(
        diagnostic.uncertainty_independent
    )
    assert diagnostic.snr_autocorrelation_model == pytest.approx(
        diagnostic.snr_independent
    )


def test_unity_lag_correlation_model_matches_fully_correlated_case() -> None:
    molecular = np.ones(4)
    measured = np.full(4, 2.0)
    error = np.ones(4)

    diagnostic = origin_calibration_uncertainty(
        measured,
        molecular,
        error,
        autocorrelation=np.ones(3),
    )

    assert diagnostic.uncertainty_autocorrelation_model == pytest.approx(
        diagnostic.uncertainty_fully_correlated
    )
    assert diagnostic.snr_autocorrelation_model == pytest.approx(
        diagnostic.snr_fully_correlated
    )


def test_short_positive_correlation_reduces_but_preserves_window_information_gain() -> None:
    molecular = np.linspace(1.0, 2.0, 8)
    measured = 3.0 * molecular
    error = np.full(8, 0.5)
    correlation = np.array([0.15, 0.03, 0.01, 0.0, 0.0, 0.0, 0.0])

    diagnostic = origin_calibration_uncertainty(
        measured,
        molecular,
        error,
        autocorrelation=correlation,
    )

    assert diagnostic.snr_fully_correlated < diagnostic.snr_autocorrelation_model
    assert diagnostic.snr_autocorrelation_model < diagnostic.snr_independent


def test_invalid_bins_are_excluded_without_interpolation() -> None:
    molecular = np.ones(5)
    measured = np.array([2.0, 2.0, np.nan, -1.0, 2.0])
    error = np.ones(5)

    diagnostic = origin_calibration_uncertainty(measured, molecular, error)

    assert diagnostic.valid_bins == 3
    assert diagnostic.calibration_factor == pytest.approx(2.0)


def test_autocorrelation_must_cover_actual_valid_bin_separation() -> None:
    molecular = np.ones(5)
    measured = np.array([2.0, np.nan, np.nan, np.nan, 2.0])
    error = np.ones(5)

    with pytest.raises(ValueError, match="cover every lag"):
        origin_calibration_uncertainty(
            measured,
            molecular,
            error,
            autocorrelation=np.zeros(3),
        )
