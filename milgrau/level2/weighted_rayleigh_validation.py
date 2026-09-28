"""Weighted Rayleigh-window fits for scientific validation.

This module is intentionally outside productive selection. It compares the
existing unweighted zero-intercept molecular calibration with an inverse-
variance fit on the same native-grid window. The reported standard error and
chi-square assume independent, correctly specified one-sigma errors; glued and
averaged lidar samples can violate that assumption, so these quantities are QA
diagnostics rather than evidence of molecular purity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class WeightedRayleighFit:
    """Side-by-side unweighted and inverse-variance molecular fits."""

    center_index: int
    start_index: int
    stop_index: int
    center_altitude_m: float
    start_altitude_m: float
    stop_altitude_m: float
    valid_bins: int
    total_bins: int
    valid_fraction: float
    effective_weighted_bins: float
    unweighted_calibration_factor: float
    weighted_calibration_factor: float
    weighted_calibration_standard_error: float
    relative_factor_difference: float
    unweighted_reduced_chi_square: float
    weighted_reduced_chi_square: float
    uncertainty_assumption: str


@dataclass(frozen=True, slots=True)
class GeneralizedRayleighFit:
    """Zero-intercept molecular fit using a supplied covariance matrix."""

    center_index: int
    start_index: int
    stop_index: int
    center_altitude_m: float
    start_altitude_m: float
    stop_altitude_m: float
    valid_bins: int
    calibration_factor: float
    calibration_standard_error: float
    reduced_quadratic_residual: float
    covariance_effective_rank: float
    covariance_condition_number: float
    covariance_assumption: str


def estimate_shrunk_temporal_covariance(
    signal_samples: np.ndarray,
    *,
    shrinkage: float = 0.2,
    relative_eigenvalue_floor: float = 1.0e-10,
) -> np.ndarray:
    """Estimate a regularized altitude covariance from repeated profiles.

    ``signal_samples`` has shape ``(sample, altitude)``. Shrinkage toward the
    diagonal is explicit because a 1-km native window generally contains more
    altitude bins than a 20-minute block contains time samples. This estimator
    is diagnostic; it is not a productive noise model.
    """
    samples = np.asarray(signal_samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 3 or samples.shape[1] < 2:
        raise ValueError(
            "signal_samples must have at least three samples and two altitude bins."
        )
    if np.any(~np.isfinite(samples)):
        raise ValueError("signal_samples must be finite for covariance estimation.")
    shrink = float(shrinkage)
    if not 0.0 < shrink <= 1.0:
        raise ValueError("shrinkage must be greater than zero and at most one.")
    floor_fraction = float(relative_eigenvalue_floor)
    if not np.isfinite(floor_fraction) or floor_fraction <= 0.0:
        raise ValueError("relative_eigenvalue_floor must be finite and positive.")

    empirical = np.asarray(np.cov(samples, rowvar=False, ddof=1), dtype=np.float64)
    diagonal = np.diag(np.diag(empirical))
    covariance = (1.0 - shrink) * empirical + shrink * diagonal
    scale = float(np.nanmax(np.diag(covariance)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("temporal samples do not contain positive variance.")
    covariance += np.eye(covariance.shape[0]) * scale * floor_fraction
    return np.asarray(covariance, dtype=np.float64)


def evaluate_generalized_rayleigh_fit(
    measured_signal: np.ndarray,
    simulated_molecular_signal: np.ndarray,
    covariance_window: np.ndarray,
    altitude_m: np.ndarray,
    *,
    center_index: int,
    window_bins: int,
    min_valid_fraction: float = 0.5,
) -> GeneralizedRayleighFit:
    """Fit ``measured = factor * molecular`` using generalized least squares."""
    measured = np.asarray(measured_signal, dtype=np.float64)
    molecular = np.asarray(simulated_molecular_signal, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if not (measured.ndim == molecular.ndim == altitude.ndim == 1):
        raise ValueError("GLS Rayleigh-fit profile inputs must be one-dimensional.")
    if not (measured.shape == molecular.shape == altitude.shape):
        raise ValueError("GLS Rayleigh-fit profile inputs must have identical shapes.")
    if (
        altitude.size < 3
        or np.any(~np.isfinite(altitude))
        or np.any(np.diff(altitude) <= 0.0)
    ):
        raise ValueError("altitude_m must be finite and strictly increasing.")
    window = int(window_bins)
    center = int(center_index)
    half = max(window // 2, 1)
    start = center - half
    stop = start + window
    if window < 3 or start < 0 or stop > altitude.size:
        raise ValueError("GLS Rayleigh-fit window lies outside the altitude grid.")
    covariance = np.asarray(covariance_window, dtype=np.float64)
    if covariance.shape != (window, window):
        raise ValueError("covariance_window must have shape (window_bins, window_bins).")
    if np.any(~np.isfinite(covariance)):
        raise ValueError("covariance_window must be finite.")
    if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-14):
        raise ValueError("covariance_window must be symmetric.")

    x = molecular[start:stop]
    y = measured[start:stop]
    valid = np.isfinite(x) & (x > 0.0) & np.isfinite(y) & (y > 0.0)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 2 or n_valid / window < float(min_valid_fraction):
        raise ValueError("GLS Rayleigh-fit window has insufficient valid support.")
    x = x[valid]
    y = y[valid]
    covariance = covariance[np.ix_(valid, valid)]

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = float(np.max(eigenvalues))
    if not np.isfinite(largest) or largest <= 0.0:
        raise ValueError("covariance_window must contain positive variance.")
    floor = largest * 1.0e-12
    eigenvalues = np.maximum(eigenvalues, floor)
    inverse = (eigenvectors / eigenvalues) @ eigenvectors.T
    information = float(x @ inverse @ x)
    if not np.isfinite(information) or information <= 0.0:
        raise ValueError("GLS molecular fit is singular.")
    factor = float((x @ inverse @ y) / information)
    residual = y - factor * x
    reduced = float((residual @ inverse @ residual) / (n_valid - 1))
    effective_rank = float(
        np.sum(eigenvalues) ** 2 / np.sum(eigenvalues**2)
    )
    return GeneralizedRayleighFit(
        center_index=center,
        start_index=start,
        stop_index=stop,
        center_altitude_m=float(altitude[center]),
        start_altitude_m=float(altitude[start]),
        stop_altitude_m=float(altitude[stop - 1]),
        valid_bins=n_valid,
        calibration_factor=factor,
        calibration_standard_error=float(np.sqrt(1.0 / information)),
        reduced_quadratic_residual=reduced,
        covariance_effective_rank=effective_rank,
        covariance_condition_number=float(np.max(eigenvalues) / np.min(eigenvalues)),
        covariance_assumption=(
            "caller_supplied_regularized_temporal_covariance_diagnostic_only"
        ),
    )


def evaluate_weighted_rayleigh_fit(
    measured_signal: np.ndarray,
    simulated_molecular_signal: np.ndarray,
    measured_signal_error: np.ndarray,
    altitude_m: np.ndarray,
    *,
    center_index: int,
    window_bins: int,
    min_valid_fraction: float = 0.5,
) -> WeightedRayleighFit:
    """Fit ``measured = factor * molecular`` on one native-grid window."""
    measured = np.asarray(measured_signal, dtype=np.float64)
    molecular = np.asarray(simulated_molecular_signal, dtype=np.float64)
    error = np.asarray(measured_signal_error, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if not (
        measured.ndim == molecular.ndim == error.ndim == altitude.ndim == 1
    ):
        raise ValueError("weighted Rayleigh-fit inputs must be one-dimensional.")
    if not (measured.shape == molecular.shape == error.shape == altitude.shape):
        raise ValueError("weighted Rayleigh-fit inputs must have identical shapes.")
    if (
        altitude.size < 3
        or np.any(~np.isfinite(altitude))
        or np.any(np.diff(altitude) <= 0.0)
    ):
        raise ValueError("altitude_m must be finite and strictly increasing.")
    window = int(window_bins)
    if window < 3:
        raise ValueError("window_bins must be at least three.")
    center = int(center_index)
    half = max(window // 2, 1)
    start = center - half
    stop = start + window
    if center < 0 or center >= altitude.size or start < 0 or stop > altitude.size:
        raise ValueError("weighted Rayleigh-fit window lies outside the altitude grid.")
    if not 0.0 <= float(min_valid_fraction) <= 1.0:
        raise ValueError("min_valid_fraction must be between zero and one.")

    y = measured[start:stop]
    x = molecular[start:stop]
    sigma = error[start:stop]
    valid = (
        np.isfinite(y)
        & (y > 0.0)
        & np.isfinite(x)
        & (x > 0.0)
        & np.isfinite(sigma)
        & (sigma > 0.0)
    )
    n_valid = int(np.count_nonzero(valid))
    valid_fraction = float(n_valid / window)
    if n_valid < 2 or valid_fraction < float(min_valid_fraction):
        raise ValueError("weighted Rayleigh-fit window has insufficient valid support.")

    xv = x[valid]
    yv = y[valid]
    sv = sigma[valid]
    sum_xx = float(np.dot(xv, xv))
    if not np.isfinite(sum_xx) or sum_xx <= 0.0:
        raise ValueError("unweighted molecular fit is singular.")
    unweighted_factor = float(np.dot(xv, yv) / sum_xx)

    x_scaled = xv / sv
    y_scaled = yv / sv
    weighted_information = float(np.dot(x_scaled, x_scaled))
    if not np.isfinite(weighted_information) or weighted_information <= 0.0:
        raise ValueError("weighted molecular fit is singular.")
    weighted_factor = float(np.dot(x_scaled, y_scaled) / weighted_information)
    weighted_standard_error = float(np.sqrt(1.0 / weighted_information))

    degrees_of_freedom = n_valid - 1
    unweighted_chi2 = float(
        np.sum(((yv - unweighted_factor * xv) / sv) ** 2) / degrees_of_freedom
    )
    weighted_chi2 = float(
        np.sum(((yv - weighted_factor * xv) / sv) ** 2) / degrees_of_freedom
    )
    relative_difference = float(
        (weighted_factor - unweighted_factor) / unweighted_factor
    )
    relative_weights = (np.min(sv) / sv) ** 2
    effective_bins = float(
        np.sum(relative_weights) ** 2 / np.sum(relative_weights**2)
    )
    return WeightedRayleighFit(
        center_index=center,
        start_index=start,
        stop_index=stop,
        center_altitude_m=float(altitude[center]),
        start_altitude_m=float(altitude[start]),
        stop_altitude_m=float(altitude[stop - 1]),
        valid_bins=n_valid,
        total_bins=window,
        valid_fraction=valid_fraction,
        effective_weighted_bins=effective_bins,
        unweighted_calibration_factor=unweighted_factor,
        weighted_calibration_factor=weighted_factor,
        weighted_calibration_standard_error=weighted_standard_error,
        relative_factor_difference=relative_difference,
        unweighted_reduced_chi_square=unweighted_chi2,
        weighted_reduced_chi_square=weighted_chi2,
        uncertainty_assumption=(
            "independent_correctly_specified_one_sigma_errors_diagnostic_only"
        ),
    )
