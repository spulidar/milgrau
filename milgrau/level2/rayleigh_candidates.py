"""Auditable Rayleigh-reference candidate catalogue and deterministic ranking.

Every complete candidate is diagnosed before productive selection.  Current
productive ranking is intentionally conservative: configured minimum QA is
applied first, then the historical ``relative_slope + relative_variance`` cost
is minimized among accepted candidates only.  No altitude preference or hard
SNR gate is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import Sequence

import numpy as np


class RayleighCandidateRejection(IntFlag):
    """Stable reasons why a Rayleigh-reference candidate fails minimum QA."""

    NONE = 0
    INSUFFICIENT_VALID_FRACTION = 1
    INVALID_CALIBRATION = 2
    EXCESS_RELATIVE_SLOPE = 4
    EXCESS_RELATIVE_VARIANCE = 8


@dataclass(frozen=True, slots=True)
class RayleighReferenceCandidate:
    """One evaluated Rayleigh-reference window on the lidar altitude grid."""

    center_index: int
    start_index: int
    stop_index: int
    center_altitude_m: float
    start_altitude_m: float
    stop_altitude_m: float
    valid_bins: int
    total_bins: int
    valid_fraction: float
    relative_slope: float
    relative_variance: float
    calibration_factor: float
    free_intercept: float
    uncertainty_snr_median: float
    uncertainty_snr_valid_bins: int
    diagnostic_cost: float
    rejection_mask: int

    @property
    def accepted(self) -> bool:
        """Return whether the candidate passes all currently enabled QA gates."""
        return int(self.rejection_mask) == int(RayleighCandidateRejection.NONE)


def _candidate_bounds(center_index: int, window_bins: int, size: int) -> tuple[int, int]:
    half = max(int(window_bins) // 2, 1)
    start = int(center_index) - half
    stop = start + int(window_bins)
    if start < 0 or stop > int(size):
        raise ValueError("Rayleigh candidate window lies outside the altitude grid.")
    return start, stop


def _validate_candidate_inputs(
    measured_signal: np.ndarray,
    simulated_molecular_signal: np.ndarray,
    altitude_m: np.ndarray,
    measured_signal_error: np.ndarray | None,
    *,
    window_bins: int,
    min_valid_fraction: float,
    max_relative_slope: float,
    max_relative_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int]:
    measured = np.asarray(measured_signal, dtype=np.float64)
    simulated = np.asarray(simulated_molecular_signal, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if not (measured.ndim == simulated.ndim == altitude.ndim == 1):
        raise ValueError(
            "measured_signal, simulated_molecular_signal, and altitude_m must be 1D."
        )
    if not (measured.shape == simulated.shape == altitude.shape):
        raise ValueError("Rayleigh candidate inputs must have identical shapes.")
    error = None if measured_signal_error is None else np.asarray(
        measured_signal_error, dtype=np.float64
    )
    if error is not None and (error.ndim != 1 or error.shape != measured.shape):
        raise ValueError(
            "measured_signal_error must be 1D and match measured_signal."
        )
    if (
        altitude.size < 3
        or not np.all(np.isfinite(altitude))
        or not np.all(np.diff(altitude) > 0.0)
    ):
        raise ValueError(
            "altitude_m must be finite, strictly increasing, and contain at least three bins."
        )
    window = int(window_bins)
    if window < 3:
        raise ValueError("window_bins must be at least three.")
    if not 0.0 <= float(min_valid_fraction) <= 1.0:
        raise ValueError("min_valid_fraction must be between zero and one.")
    if float(max_relative_slope) < 0.0 or float(max_relative_variance) < 0.0:
        raise ValueError("Rayleigh QA limits must be non-negative.")
    return measured, simulated, altitude, error, window


def evaluate_rayleigh_candidates(
    measured_signal: np.ndarray,
    simulated_molecular_signal: np.ndarray,
    altitude_m: np.ndarray,
    *,
    center_indices: Sequence[int] | np.ndarray,
    window_bins: int,
    max_relative_slope: float,
    max_relative_variance: float,
    min_valid_fraction: float,
    measured_signal_error: np.ndarray | None = None,
) -> tuple[RayleighReferenceCandidate, ...]:
    """Evaluate multiple equal-width Rayleigh windows in one vectorized pass.

    The scientific definitions are unchanged: zero-intercept calibration,
    free-intercept diagnostic, relative ratio variance/slope and median
    uncertainty SNR are identical quantities to the historical scalar loop.
    Vectorization only removes repeated Python/polyfit overhead in method-v5 MC.
    """
    measured, simulated, altitude, error, window = _validate_candidate_inputs(
        measured_signal,
        simulated_molecular_signal,
        altitude_m,
        measured_signal_error,
        window_bins=window_bins,
        min_valid_fraction=min_valid_fraction,
        max_relative_slope=max_relative_slope,
        max_relative_variance=max_relative_variance,
    )
    centers = np.asarray(center_indices, dtype=np.int64)
    if centers.ndim != 1:
        raise ValueError("center_indices must be one-dimensional.")
    if centers.size == 0:
        return tuple()
    half = max(window // 2, 1)
    starts = centers - half
    stops = starts + window
    if (
        np.any(centers < 0)
        or np.any(centers >= altitude.size)
        or np.any(starts < 0)
        or np.any(stops > altitude.size)
    ):
        raise ValueError("Rayleigh candidate window lies outside the altitude grid.")

    indices = starts[:, None] + np.arange(window, dtype=np.int64)[None, :]
    x = simulated[indices]
    y = measured[indices]
    z = altitude[indices]
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & (x > 0.0)
        & (y > 0.0)
    )
    valid_bins = np.count_nonzero(valid, axis=1).astype(np.int32)
    total_bins = np.full(centers.size, window, dtype=np.int32)
    valid_fraction = valid_bins.astype(np.float64) / float(window)
    n = valid_bins.astype(np.float64)

    x_valid = np.where(valid, x, 0.0)
    y_valid = np.where(valid, y, 0.0)
    z_valid = np.where(valid, z, 0.0)
    sum_x = np.sum(x_valid, axis=1)
    sum_y = np.sum(y_valid, axis=1)
    sum_xx = np.sum(x_valid * x_valid, axis=1)
    sum_xy = np.sum(x_valid * y_valid, axis=1)

    factor = np.full(centers.size, np.nan, dtype=np.float64)
    np.divide(sum_xy, sum_xx, out=factor, where=sum_xx > 0.0)

    free_intercept = np.full(centers.size, np.nan, dtype=np.float64)
    fit_denominator = n * sum_xx - sum_x * sum_x
    free_slope = np.full(centers.size, np.nan, dtype=np.float64)
    fit_ok = (valid_bins >= 2) & np.isfinite(fit_denominator) & (fit_denominator > 0.0)
    np.divide(
        n * sum_xy - sum_x * sum_y,
        fit_denominator,
        out=free_slope,
        where=fit_ok,
    )
    intercept_ok = fit_ok & (n > 0.0)
    np.divide(
        sum_y - free_slope * sum_x,
        n,
        out=free_intercept,
        where=intercept_ok,
    )

    ratio = np.full_like(y, np.nan, dtype=np.float64)
    np.divide(y, x, out=ratio, where=valid)
    sum_ratio = np.nansum(ratio, axis=1)
    mean_ratio = np.full(centers.size, np.nan, dtype=np.float64)
    np.divide(sum_ratio, n, out=mean_ratio, where=n > 0.0)

    centered_ratio = np.where(valid, ratio - mean_ratio[:, None], 0.0)
    ratio_variance = np.full(centers.size, np.inf, dtype=np.float64)
    ratio_ok = (valid_bins >= 3) & np.isfinite(mean_ratio) & (mean_ratio > 0.0)
    np.divide(
        np.sum(centered_ratio * centered_ratio, axis=1),
        n,
        out=ratio_variance,
        where=ratio_ok,
    )
    relative_variance = np.full(centers.size, np.inf, dtype=np.float64)
    np.divide(
        ratio_variance,
        mean_ratio**2,
        out=relative_variance,
        where=ratio_ok,
    )

    sum_z = np.sum(z_valid, axis=1)
    sum_zz = np.sum(z_valid * z_valid, axis=1)
    sum_zr = np.sum(
        np.where(valid, z * np.where(np.isfinite(ratio), ratio, 0.0), 0.0),
        axis=1,
    )
    slope_denominator = n * sum_zz - sum_z * sum_z
    ratio_slope = np.full(centers.size, np.nan, dtype=np.float64)
    slope_ok = ratio_ok & np.isfinite(slope_denominator) & (slope_denominator > 0.0)
    np.divide(
        n * sum_zr - sum_z * sum_ratio,
        slope_denominator,
        out=ratio_slope,
        where=slope_ok,
    )
    min_z = np.min(np.where(valid, z, np.inf), axis=1)
    max_z = np.max(np.where(valid, z, -np.inf), axis=1)
    span = np.maximum(max_z - min_z, 1.0)
    relative_slope = np.full(centers.size, np.inf, dtype=np.float64)
    relative_slope_ok = slope_ok & np.isfinite(ratio_slope)
    np.divide(
        np.abs(ratio_slope) * span,
        mean_ratio,
        out=relative_slope,
        where=relative_slope_ok,
    )

    snr_median = np.full(centers.size, np.nan, dtype=np.float64)
    snr_valid_bins = np.zeros(centers.size, dtype=np.int32)
    if error is not None:
        window_error = error[indices]
        snr_valid = valid & np.isfinite(window_error) & (window_error > 0.0)
        snr_valid_bins = np.count_nonzero(snr_valid, axis=1).astype(np.int32)
        rows = snr_valid_bins > 0
        if np.any(rows):
            snr_values = np.full_like(y, np.nan, dtype=np.float64)
            np.divide(y, window_error, out=snr_values, where=snr_valid)
            snr_median[rows] = np.nanmedian(snr_values[rows], axis=1)

    rejection = np.zeros(centers.size, dtype=np.int32)
    rejection[valid_fraction < float(min_valid_fraction)] |= int(
        RayleighCandidateRejection.INSUFFICIENT_VALID_FRACTION
    )
    rejection[(~np.isfinite(factor)) | (factor <= 0.0)] |= int(
        RayleighCandidateRejection.INVALID_CALIBRATION
    )
    rejection[(~np.isfinite(relative_slope)) | (relative_slope > float(max_relative_slope))] |= int(
        RayleighCandidateRejection.EXCESS_RELATIVE_SLOPE
    )
    rejection[(~np.isfinite(relative_variance)) | (relative_variance > float(max_relative_variance))] |= int(
        RayleighCandidateRejection.EXCESS_RELATIVE_VARIANCE
    )
    diagnostic_cost = np.where(
        np.isfinite(relative_slope) & np.isfinite(relative_variance),
        relative_slope + relative_variance,
        np.inf,
    )

    return tuple(
        RayleighReferenceCandidate(
            center_index=int(centers[i]),
            start_index=int(starts[i]),
            stop_index=int(stops[i]),
            center_altitude_m=float(altitude[centers[i]]),
            start_altitude_m=float(altitude[starts[i]]),
            stop_altitude_m=float(altitude[stops[i] - 1]),
            valid_bins=int(valid_bins[i]),
            total_bins=int(total_bins[i]),
            valid_fraction=float(valid_fraction[i]),
            relative_slope=float(relative_slope[i]),
            relative_variance=float(relative_variance[i]),
            calibration_factor=float(factor[i]),
            free_intercept=float(free_intercept[i]),
            uncertainty_snr_median=float(snr_median[i]),
            uncertainty_snr_valid_bins=int(snr_valid_bins[i]),
            diagnostic_cost=float(diagnostic_cost[i]),
            rejection_mask=int(rejection[i]),
        )
        for i in range(centers.size)
    )


def evaluate_rayleigh_candidate(
    measured_signal: np.ndarray,
    simulated_molecular_signal: np.ndarray,
    altitude_m: np.ndarray,
    *,
    center_index: int,
    window_bins: int,
    max_relative_slope: float,
    max_relative_variance: float,
    min_valid_fraction: float,
    measured_signal_error: np.ndarray | None = None,
) -> RayleighReferenceCandidate:
    """Evaluate one Rayleigh window using the shared vectorized definitions."""
    return evaluate_rayleigh_candidates(
        measured_signal,
        simulated_molecular_signal,
        altitude_m,
        center_indices=(int(center_index),),
        window_bins=window_bins,
        max_relative_slope=max_relative_slope,
        max_relative_variance=max_relative_variance,
        min_valid_fraction=min_valid_fraction,
        measured_signal_error=measured_signal_error,
    )[0]


def catalogue_rayleigh_candidates(
    measured_signal: np.ndarray,
    simulated_molecular_signal: np.ndarray,
    altitude_m: np.ndarray,
    *,
    min_altitude_m: float,
    max_altitude_m: float,
    window_bins: int,
    max_relative_slope: float,
    max_relative_variance: float,
    min_valid_fraction: float,
    measured_signal_error: np.ndarray | None = None,
) -> tuple[RayleighReferenceCandidate, ...]:
    """Return every fully contained candidate in the configured search interval."""
    altitude = np.asarray(altitude_m, dtype=np.float64)
    lower = float(min_altitude_m)
    upper = float(max_altitude_m)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("Rayleigh search bounds must be finite and increasing.")
    window = int(window_bins)
    if window < 3:
        raise ValueError("window_bins must be at least three.")
    half = max(window // 2, 1)
    centers = np.arange(
        half,
        altitude.size - (window - half) + 1,
        dtype=np.int64,
    )
    starts = centers - half
    stops = starts + window
    inside = (altitude[starts] >= lower) & (altitude[stops - 1] <= upper)
    centers = centers[inside]
    if centers.size == 0:
        raise ValueError(
            "No complete Rayleigh candidate window exists inside the configured search interval."
        )
    return evaluate_rayleigh_candidates(
        measured_signal,
        simulated_molecular_signal,
        altitude,
        center_indices=centers,
        window_bins=window,
        max_relative_slope=max_relative_slope,
        max_relative_variance=max_relative_variance,
        min_valid_fraction=min_valid_fraction,
        measured_signal_error=measured_signal_error,
    )

def accepted_rayleigh_candidates(
    candidates: Sequence[RayleighReferenceCandidate],
) -> tuple[RayleighReferenceCandidate, ...]:
    """Filter a catalogue after QA without imposing a ranking policy."""
    return tuple(candidate for candidate in candidates if candidate.accepted)


def minimum_cost_rayleigh_candidate(
    candidates: Sequence[RayleighReferenceCandidate],
) -> RayleighReferenceCandidate:
    """Return deterministic minimum historical diagnostic cost from any candidates.

    This is useful for failure diagnostics only when no candidate passes QA.
    Ties use the lower center index, matching the historical first-window
    behavior rather than introducing an undocumented altitude preference.
    """
    if not candidates:
        raise ValueError("Rayleigh candidate ranking requires at least one candidate.")
    return min(candidates, key=lambda candidate: (candidate.diagnostic_cost, candidate.center_index))


def select_minimum_cost_accepted_candidate(
    candidates: Sequence[RayleighReferenceCandidate],
) -> RayleighReferenceCandidate:
    """Apply QA first, then minimize the historical cost among passers only."""
    accepted = accepted_rayleigh_candidates(candidates)
    if not accepted:
        raise ValueError("No Rayleigh reference candidate passes configured minimum QA.")
    return minimum_cost_rayleigh_candidate(accepted)
