"""R&D Monte Carlo propagation for a window-fitted KFS signal boundary.

This module is deliberately non-productive.  Level 2 method v4 still uses the
exact measured range-corrected-signal bin at the selected reference altitude.
The helper below asks a narrower research question: if a local molecular window
is used to estimate the signal value at that *same exact altitude*, how does the
window fit uncertainty propagate into a backward Fernald retrieval?

The first implemented noise model is explicit independent Gaussian per-bin
measurement noise.  The same perturbed window is used to re-fit the molecular
scale and to perturb the retrieval profile, so the boundary fit is not sampled
as an independent nuisance on top of the same signal noise.  Correlated-noise
extensions must be added explicitly rather than inferred from this helper.

Window contamination/model error is outside this random-noise Monte Carlo.  A
small Monte Carlo spread therefore does not certify a molecular window.
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
    noise_model: str = "independent_gaussian_per_bin"


def _origin_factor(measured: np.ndarray, molecular: np.ndarray) -> float:
    """Return the origin-constrained least-squares molecular scale."""
    denominator = float(np.sum(molecular**2))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Molecular window denominator must be finite and positive.")
    factor = float(np.sum(molecular * measured) / denominator)
    return factor if np.isfinite(factor) and factor > 0.0 else np.nan


def _nanmean_std(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-bin mean/std without empty-slice warnings."""
    finite = np.isfinite(samples)
    count = finite.sum(axis=0)
    safe = np.where(finite, samples, 0.0)
    total = safe.sum(axis=0)
    mean = np.divide(
        total,
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
    lr_mol: float = RAYLEIGH_LIDAR_RATIO_SR,
    min_lidar_ratio: float = 10.0,
    allow_negative_aerosol: bool = True,
) -> WindowBoundaryMonteCarloResult:
    """Propagate independent signal noise through a local fitted boundary.

    ``window_start_idx`` is inclusive and ``window_stop_idx`` is exclusive.  The
    exact ``ref_idx`` must lie inside that window.  The nominal and every Monte
    Carlo realization estimate one multiplicative molecular scale from the
    complete window and evaluate ``X_ref = C * molecular_rcs[ref_idx]`` without
    moving the reference altitude.

    The helper deliberately requires complete finite positive window support
    and complete finite positive backward-path signal/molecular support.  It
    does not interpolate missing bins or silently shrink the fitting window.
    Simulations whose Gaussian perturbation makes the required backward path or
    fitted scale non-positive are counted as unsuccessful rather than clipped.
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
    if np.any(~np.isfinite(signal[backward_slice])) or np.any(signal[backward_slice] <= 0.0):
        raise ValueError("Backward-path RCS must be finite and positive before Monte Carlo perturbation.")
    if np.any(~np.isfinite(beta_molecular[backward_slice])) or np.any(beta_molecular[backward_slice] <= 0.0):
        raise ValueError("Backward-path molecular backscatter must be finite and positive.")
    if np.any(~np.isfinite(error[backward_slice])) or np.any(error[backward_slice] < 0.0):
        raise ValueError("Backward-path RCS uncertainty must be finite and non-negative.")
    if np.any(~np.isfinite(molecular_signal[window_slice])) or np.any(molecular_signal[window_slice] <= 0.0):
        raise ValueError("Fitting-window molecular RCS must be finite and positive.")
    if np.any(error[window_slice] <= 0.0):
        raise ValueError("Fitting-window RCS uncertainty must be strictly positive.")
    if not np.isfinite(beta_total_ref) or float(beta_total_ref) <= 0.0:
        raise ValueError("beta_total_ref must be finite and positive.")

    nominal_factor = _origin_factor(signal[window_slice], molecular_signal[window_slice])
    if not np.isfinite(nominal_factor):
        raise ValueError("Nominal fitting window does not define a positive calibration factor.")
    nominal_boundary = float(nominal_factor * molecular_signal[reference])

    rng = np.random.default_rng(int(random_seed))
    retrieval_samples = np.full((int(n_simulations), signal.size), np.nan, dtype=np.float64)
    factor_samples = np.full(int(n_simulations), np.nan, dtype=np.float64)
    boundary_samples = np.full(int(n_simulations), np.nan, dtype=np.float64)
    successful = 0

    for simulation in range(int(n_simulations)):
        perturbed = signal.copy()
        perturbed[backward_slice] += rng.normal(
            loc=0.0,
            scale=error[backward_slice],
        )
        if np.any(~np.isfinite(perturbed[backward_slice])) or np.any(perturbed[backward_slice] <= 0.0):
            continue

        factor = _origin_factor(
            perturbed[window_slice],
            molecular_signal[window_slice],
        )
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
        raise ValueError("Fewer than two valid Monte Carlo realizations survived the strict backward-support checks.")

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
        requested_simulations=int(n_simulations),
    )
