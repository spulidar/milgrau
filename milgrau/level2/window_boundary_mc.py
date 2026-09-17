"""R&D Monte Carlo propagation for a window-fitted KFS signal boundary.

This module is deliberately non-productive. Level 2 method v4 still uses the
exact measured range-corrected-signal bin at the selected reference altitude.
Here the local molecular window is re-fitted in every realization to study how
its random signal uncertainty would propagate through a backward KFS retrieval.

The same perturbation drives both the window fit and the retrieval profile, so
window-fit uncertainty is not double-counted as an independent nuisance. The
caller may optionally provide a complete correlation matrix for the fitting
window. No AR(1), Toeplitz or other correlation law is inferred by this module.
Bins on the backward path outside that supplied window remain explicitly
independent from one another and from the window in this R&D model.

Window contamination/model error is outside this Monte Carlo; a small random
spread therefore never certifies molecular purity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.kfs import fernald_inversion


@dataclass(frozen=True, slots=True)
class WindowBoundaryMonteCarloResult:
    """Random-noise propagation for one exact-altitude window-fitted boundary."""

    beta_aerosol_mean: np.ndarray
    beta_aerosol_std: np.ndarray
    nominal_calibration_factor: float
    calibration_factor_mean: float
    calibration_factor_std: float
    nominal_boundary_signal: float
    boundary_signal_mean: float
    boundary_signal_std: float
    successful_simulations: int
    requested_simulations: int
    noise_model: str


def _origin_factor(measured: np.ndarray, molecular: np.ndarray) -> float:
    denominator = float(np.sum(molecular**2))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Molecular window denominator must be finite and positive.")
    factor = float(np.sum(molecular * measured) / denominator)
    return factor if np.isfinite(factor) and factor > 0.0 else np.nan


def _nanmean_std(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(samples)
    count = finite.sum(axis=0)
    safe = np.where(finite, samples, 0.0)
    mean = np.divide(
        safe.sum(axis=0),
        count,
        out=np.full(samples.shape[1], np.nan, dtype=np.float64),
        where=count > 0,
    )
    centered = np.where(finite, samples - mean[np.newaxis, :], 0.0)
    variance = np.divide(
        np.sum(centered**2, axis=0),
        count,
        out=np.full(samples.shape[1], np.nan, dtype=np.float64),
        where=count > 0,
    )
    return mean, np.sqrt(variance)


def _correlation_square_root(
    window_correlation: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Validate a caller-supplied correlation matrix and return a PSD square root."""
    correlation = np.asarray(window_correlation, dtype=np.float64)
    expected_shape = (int(window_size), int(window_size))
    if correlation.shape != expected_shape:
        raise ValueError(
            f"window_correlation must have shape {expected_shape}; got {correlation.shape}."
        )
    if not np.all(np.isfinite(correlation)):
        raise ValueError("window_correlation must be entirely finite.")
    if not np.allclose(correlation, correlation.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("window_correlation must be symmetric.")
    if not np.allclose(np.diag(correlation), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("window_correlation must have a unit diagonal.")
    if np.any(np.abs(correlation) > 1.0 + 1.0e-12):
        raise ValueError("window_correlation entries must lie within [-1, 1].")

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = 1.0e-10 * scale
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("window_correlation must be positive semidefinite.")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(eigenvalues)[np.newaxis, :]) @ eigenvectors.T


def window_fitted_boundary_monte_carlo(
    rcs: np.ndarray,
    rcs_error: np.ndarray,
    molecular_rcs: np.ndarray,
    altitude_m: np.ndarray,
    beta_mol: np.ndarray,
    lidar_ratio_aerosol: float | np.ndarray,
    beta_total_ref: float,
    ref_idx: int,
    window_start_idx: int,
    window_stop_idx: int,
    *,
    n_simulations: int,
    random_seed: int,
    window_correlation: np.ndarray | None = None,
    lr_mol: float = RAYLEIGH_LIDAR_RATIO_SR,
    min_lidar_ratio: float = 10.0,
    allow_negative_aerosol: bool = True,
) -> WindowBoundaryMonteCarloResult:
    """Propagate explicit signal-noise assumptions through a fitted boundary.

    Window start is inclusive and stop is exclusive. The exact reference bin
    must lie inside the window. Complete finite positive support is required in
    the fit window and along the backward retrieval path; nothing is filled,
    clipped or silently removed. A realization that becomes non-positive on
    required support is rejected and counted as unsuccessful.

    With ``window_correlation=None``, all perturbed bins are independent. When a
    matrix is supplied it must describe every pair of fitting-window bins. The
    fit-window perturbation then follows exactly that matrix; backward-path bins
    outside the window remain independent and have zero cross-covariance with
    the supplied window by construction. The function never extrapolates a
    correlation law beyond the matrix provided by the caller.
    """
    signal = np.asarray(rcs, dtype=np.float64)
    error = np.asarray(rcs_error, dtype=np.float64)
    molecular_signal = np.asarray(molecular_rcs, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    beta_molecular = np.asarray(beta_mol, dtype=np.float64)
    arrays = (signal, error, molecular_signal, altitude, beta_molecular)

    if any(array.ndim != 1 for array in arrays):
        raise ValueError("All window-boundary Monte Carlo profiles must be one-dimensional.")
    if not all(array.shape == signal.shape for array in arrays[1:]):
        raise ValueError("All window-boundary Monte Carlo profiles must share one shape.")
    if signal.size < 3:
        raise ValueError("At least three altitude bins are required.")
    if np.any(~np.isfinite(altitude)) or np.any(np.diff(altitude) <= 0.0):
        raise ValueError("altitude_m must be finite and strictly increasing.")
    if not isinstance(n_simulations, (int, np.integer)) or int(n_simulations) < 2:
        raise ValueError("n_simulations must be an integer >= 2.")

    reference = int(ref_idx)
    start = int(window_start_idx)
    stop = int(window_stop_idx)
    if reference < 0 or reference >= signal.size:
        raise ValueError("ref_idx must point inside the profile.")
    if start < 0 or stop > signal.size or stop <= start:
        raise ValueError("Window indices must define a non-empty in-profile slice.")
    if not start <= reference < stop:
        raise ValueError("The exact reference bin must lie inside the fitting window.")
    if stop - start < 2:
        raise ValueError("The fitting window must contain at least two bins.")

    backward_slice = slice(0, reference + 1)
    window_slice = slice(start, stop)
    window_size = stop - start

    if np.any(~np.isfinite(signal[backward_slice])) or np.any(signal[backward_slice] <= 0.0):
        raise ValueError("Backward-path RCS must be finite and positive before perturbation.")
    if np.any(~np.isfinite(beta_molecular[backward_slice])) or np.any(beta_molecular[backward_slice] <= 0.0):
        raise ValueError("Backward-path molecular backscatter must be finite and positive.")
    if np.any(~np.isfinite(error[backward_slice])) or np.any(error[backward_slice] < 0.0):
        raise ValueError("Backward-path RCS uncertainty must be finite and non-negative.")
    if np.any(~np.isfinite(signal[window_slice])) or np.any(signal[window_slice] <= 0.0):
        raise ValueError("Fitting-window measured RCS must be finite and positive.")
    if np.any(~np.isfinite(molecular_signal[window_slice])) or np.any(molecular_signal[window_slice] <= 0.0):
        raise ValueError("Fitting-window molecular RCS must be finite and positive.")
    if np.any(~np.isfinite(error[window_slice])) or np.any(error[window_slice] <= 0.0):
        raise ValueError("Fitting-window RCS uncertainty must be finite and strictly positive.")
    if not np.isfinite(beta_total_ref) or float(beta_total_ref) <= 0.0:
        raise ValueError("beta_total_ref must be finite and positive.")

    correlation_root = None
    noise_model = "independent_gaussian_per_bin"
    if window_correlation is not None:
        correlation_root = _correlation_square_root(window_correlation, window_size)
        noise_model = "caller_supplied_correlated_window_plus_independent_external_backward_bins"

    nominal_factor = _origin_factor(signal[window_slice], molecular_signal[window_slice])
    if not np.isfinite(nominal_factor):
        raise ValueError("Nominal fitting window does not define a positive calibration factor.")
    nominal_boundary = float(nominal_factor * molecular_signal[reference])

    window_indices = np.arange(start, stop, dtype=np.int64)
    backward_indices = np.arange(0, reference + 1, dtype=np.int64)
    external_backward_indices = backward_indices[
        (backward_indices < start) | (backward_indices >= stop)
    ]

    rng = np.random.default_rng(int(random_seed))
    n_mc = int(n_simulations)
    retrieval_samples = np.full((n_mc, signal.size), np.nan, dtype=np.float64)
    factor_samples = np.full(n_mc, np.nan, dtype=np.float64)
    boundary_samples = np.full(n_mc, np.nan, dtype=np.float64)
    successful = 0

    for simulation in range(n_mc):
        perturbed = signal.copy()
        if external_backward_indices.size:
            perturbed[external_backward_indices] += rng.normal(
                0.0,
                error[external_backward_indices],
            )

        if correlation_root is None:
            window_standard_noise = rng.normal(size=window_size)
        else:
            window_standard_noise = correlation_root @ rng.normal(size=window_size)
        perturbed[window_indices] += error[window_indices] * window_standard_noise

        if np.any(~np.isfinite(perturbed[backward_slice])) or np.any(perturbed[backward_slice] <= 0.0):
            continue
        if np.any(~np.isfinite(perturbed[window_slice])) or np.any(perturbed[window_slice] <= 0.0):
            continue

        factor = _origin_factor(perturbed[window_slice], molecular_signal[window_slice])
        if not np.isfinite(factor):
            continue
        fitted_boundary = float(factor * molecular_signal[reference])
        if not np.isfinite(fitted_boundary) or fitted_boundary <= 0.0:
            continue
        perturbed[reference] = fitted_boundary

        retrieved = fernald_inversion(
            perturbed,
            altitude,
            beta_molecular,
            lidar_ratio_aerosol,
            float(beta_total_ref),
            reference,
            lr_mol=float(lr_mol),
            altitude_units="m",
            min_lidar_ratio=float(min_lidar_ratio),
            allow_negative_aerosol=bool(allow_negative_aerosol),
            mode="backward",
        )
        if not np.all(np.isfinite(retrieved[backward_slice])):
            continue

        retrieval_samples[simulation] = retrieved
        factor_samples[simulation] = factor
        boundary_samples[simulation] = fitted_boundary
        successful += 1

    if successful < 2:
        raise ValueError(
            "Fewer than two valid Monte Carlo realizations survived strict support checks."
        )

    beta_mean, beta_std = _nanmean_std(retrieval_samples)
    valid_factors = factor_samples[np.isfinite(factor_samples)]
    valid_boundaries = boundary_samples[np.isfinite(boundary_samples)]
    return WindowBoundaryMonteCarloResult(
        beta_aerosol_mean=beta_mean,
        beta_aerosol_std=beta_std,
        nominal_calibration_factor=float(nominal_factor),
        calibration_factor_mean=float(np.mean(valid_factors)),
        calibration_factor_std=float(np.std(valid_factors, ddof=0)),
        nominal_boundary_signal=nominal_boundary,
        boundary_signal_mean=float(np.mean(valid_boundaries)),
        boundary_signal_std=float(np.std(valid_boundaries, ddof=0)),
        successful_simulations=int(successful),
        requested_simulations=n_mc,
        noise_model=noise_model,
    )
