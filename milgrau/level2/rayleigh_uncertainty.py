"""Uncertainty diagnostics for origin-constrained Rayleigh calibration windows.

The productive method still uses the exact measured RCS bin as the KFS signal
boundary.  These helpers quantify how much information the complete local
Rayleigh window contains about its multiplicative molecular calibration factor;
they do not change that productive boundary semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RayleighCalibrationUncertainty:
    """Calibration-factor uncertainty under explicit dependence assumptions."""

    calibration_factor: float
    valid_bins: int
    uncertainty_independent: float
    uncertainty_fully_correlated: float
    uncertainty_autocorrelation_model: float
    snr_independent: float
    snr_fully_correlated: float
    snr_autocorrelation_model: float


def origin_calibration_uncertainty(
    measured_signal: np.ndarray,
    molecular_signal: np.ndarray,
    measured_error: np.ndarray,
    *,
    autocorrelation: np.ndarray | None = None,
) -> RayleighCalibrationUncertainty:
    """Estimate uncertainty of ``C=sum(x*y)/sum(x^2)`` for one local window.

    ``x`` is the positive molecular model and ``y`` the positive measured RCS.
    Per-bin propagated errors are interpreted as one-sigma values.

    Three dependence views are kept distinct:

    * independent bins;
    * perfect positive correlation of all valid bins;
    * an optional stationary lag-autocorrelation sequence supplied explicitly by
      the caller.  When provided it must cover every lag separating valid bins;
      unreported lags are never silently assumed to be zero.

    This is a diagnostic of the fitted multiplicative scale, not a hard Rayleigh
    QA gate and not a replacement for the exact KFS boundary condition.
    """
    measured = np.asarray(measured_signal, dtype=np.float64)
    molecular = np.asarray(molecular_signal, dtype=np.float64)
    error = np.asarray(measured_error, dtype=np.float64)
    if not (measured.ndim == molecular.ndim == error.ndim == 1):
        raise ValueError("Rayleigh calibration inputs must be one-dimensional.")
    if not (measured.shape == molecular.shape == error.shape):
        raise ValueError("Rayleigh calibration inputs must have identical shapes.")

    valid = (
        np.isfinite(measured)
        & np.isfinite(molecular)
        & np.isfinite(error)
        & (measured > 0.0)
        & (molecular > 0.0)
        & (error > 0.0)
    )
    indices = np.flatnonzero(valid)
    if indices.size < 2:
        return RayleighCalibrationUncertainty(
            calibration_factor=np.nan,
            valid_bins=int(indices.size),
            uncertainty_independent=np.nan,
            uncertainty_fully_correlated=np.nan,
            uncertainty_autocorrelation_model=np.nan,
            snr_independent=np.nan,
            snr_fully_correlated=np.nan,
            snr_autocorrelation_model=np.nan,
        )

    x = molecular[valid]
    y = measured[valid]
    sigma = error[valid]
    denominator = float(np.sum(x**2))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Rayleigh molecular calibration denominator must be positive.")
    factor = float(np.sum(x * y) / denominator)
    gradient = x / denominator

    variance_independent = float(np.sum((gradient * sigma) ** 2))
    sigma_independent = float(np.sqrt(variance_independent))
    sigma_fully_correlated = float(np.sum(np.abs(gradient) * sigma))

    sigma_autocorrelation = np.nan
    if autocorrelation is not None:
        correlation = np.asarray(autocorrelation, dtype=np.float64)
        if correlation.ndim != 1:
            raise ValueError("autocorrelation must be a one-dimensional lag sequence.")
        max_required_lag = int(indices[-1] - indices[0])
        if correlation.size < max_required_lag:
            raise ValueError(
                "autocorrelation must cover every lag separating valid calibration bins."
            )
        used = correlation[:max_required_lag]
        if not np.all(np.isfinite(used)) or np.any(np.abs(used) > 1.0):
            raise ValueError("Required autocorrelation lags must be finite and within [-1, 1].")

        variance = variance_independent
        weighted_sigma = gradient * sigma
        for first in range(indices.size - 1):
            for second in range(first + 1, indices.size):
                lag = int(indices[second] - indices[first])
                variance += (
                    2.0
                    * float(correlation[lag - 1])
                    * float(weighted_sigma[first])
                    * float(weighted_sigma[second])
                )
        if not np.isfinite(variance) or variance <= 0.0:
            raise ValueError("Autocorrelation model implies a non-positive calibration variance.")
        sigma_autocorrelation = float(np.sqrt(variance))

    def _snr(uncertainty: float) -> float:
        if not np.isfinite(factor) or factor <= 0.0 or not np.isfinite(uncertainty) or uncertainty <= 0.0:
            return np.nan
        return float(factor / uncertainty)

    return RayleighCalibrationUncertainty(
        calibration_factor=factor,
        valid_bins=int(indices.size),
        uncertainty_independent=sigma_independent,
        uncertainty_fully_correlated=sigma_fully_correlated,
        uncertainty_autocorrelation_model=sigma_autocorrelation,
        snr_independent=_snr(sigma_independent),
        snr_fully_correlated=_snr(sigma_fully_correlated),
        snr_autocorrelation_model=_snr(sigma_autocorrelation),
    )
