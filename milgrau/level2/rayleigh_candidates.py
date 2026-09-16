"""Auditable Rayleigh-reference candidate catalogue for high-column Level 2 R&D.

The productive selector still lives in :mod:`milgrau.level2.molecular`.  This
module deliberately does not replace it yet.  It enumerates and diagnoses every
candidate window so selection policy can be validated before productive
behavior changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag

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
    """Evaluate one window without selecting or ranking it.

    When propagated signal uncertainty is supplied, a median positive-signal
    SNR diagnostic is recorded.  It is intentionally diagnostic-only here: no
    hard SNR threshold is enabled until SPU evidence justifies one.
    """
    measured = np.asarray(measured_signal, dtype=np.float64)
    simulated = np.asarray(simulated_molecular_signal, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if not (measured.ndim == simulated.ndim == altitude.ndim == 1):
        raise ValueError("measured_signal, simulated_molecular_signal, and altitude_m must be 1D.")
    if not (measured.shape == simulated.shape == altitude.shape):
        raise ValueError("Rayleigh candidate inputs must have identical shapes.")

    error: np.ndarray | None
    if measured_signal_error is None:
        error = None
    else:
        error = np.asarray(measured_signal_error, dtype=np.float64)
        if error.ndim != 1 or error.shape != measured.shape:
            raise ValueError("measured_signal_error must be 1D and match measured_signal.")

    if altitude.size < 3 or not np.all(np.isfinite(altitude)) or not np.all(np.diff(altitude) > 0.0):
        raise ValueError("altitude_m must be finite, strictly increasing, and contain at least three bins.")
    window = int(window_bins)
    if window < 3:
        raise ValueError("window_bins must be at least three.")
    if not 0 <= int(center_index) < altitude.size:
        raise ValueError("center_index is outside the altitude grid.")
    if not 0.0 <= float(min_valid_fraction) <= 1.0:
        raise ValueError("min_valid_fraction must be between zero and one.")
    if float(max_relative_slope) < 0.0 or float(max_relative_variance) < 0.0:
        raise ValueError("Rayleigh QA limits must be non-negative.")

    start, stop = _candidate_bounds(int(center_index), window, altitude.size)
    x = simulated[start:stop]
    y = measured[start:stop]
    z = altitude[start:stop]
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x > 0.0) & (y > 0.0)
    valid_bins = int(valid.sum())
    total_bins = int(stop - start)
    valid_fraction = float(valid_bins / max(total_bins, 1))

    factor = np.nan
    intercept = np.nan
    relative_slope = np.inf
    relative_variance = np.inf
    if valid_bins >= 2:
        denominator = float(np.sum(x[valid] ** 2))
        if np.isfinite(denominator) and denominator > 0.0:
            factor = float(np.sum(x[valid] * y[valid]) / denominator)
        _, intercept = np.polyfit(x[valid], y[valid], 1)

    if valid_bins >= 3:
        ratio = y[valid] / x[valid]
        mean_ratio = float(np.mean(ratio))
        if np.isfinite(mean_ratio) and mean_ratio > 0.0:
            relative_variance = float(np.var(ratio) / (mean_ratio**2))
            slope, _ = np.polyfit(z[valid], ratio, 1)
            span = max(float(np.max(z[valid]) - np.min(z[valid])), 1.0)
            relative_slope = float(abs(slope) * span / mean_ratio)

    snr_median = np.nan
    snr_valid_bins = 0
    if error is not None:
        window_error = error[start:stop]
        snr_valid = valid & np.isfinite(window_error) & (window_error > 0.0)
        snr_valid_bins = int(snr_valid.sum())
        if snr_valid_bins:
            snr_median = float(np.median(y[snr_valid] / window_error[snr_valid]))

    rejection = RayleighCandidateRejection.NONE
    if valid_fraction < float(min_valid_fraction):
        rejection |= RayleighCandidateRejection.INSUFFICIENT_VALID_FRACTION
    if not np.isfinite(factor) or factor <= 0.0:
        rejection |= RayleighCandidateRejection.INVALID_CALIBRATION
    if not np.isfinite(relative_slope) or relative_slope > float(max_relative_slope):
        rejection |= RayleighCandidateRejection.EXCESS_RELATIVE_SLOPE
    if not np.isfinite(relative_variance) or relative_variance > float(max_relative_variance):
        rejection |= RayleighCandidateRejection.EXCESS_RELATIVE_VARIANCE

    cost = (
        float(relative_slope + relative_variance)
        if np.isfinite(relative_slope) and np.isfinite(relative_variance)
        else float("inf")
    )
    return RayleighReferenceCandidate(
        center_index=int(center_index),
        start_index=start,
        stop_index=stop,
        center_altitude_m=float(altitude[int(center_index)]),
        start_altitude_m=float(altitude[start]),
        stop_altitude_m=float(altitude[stop - 1]),
        valid_bins=valid_bins,
        total_bins=total_bins,
        valid_fraction=valid_fraction,
        relative_slope=float(relative_slope),
        relative_variance=float(relative_variance),
        calibration_factor=float(factor),
        free_intercept=float(intercept),
        uncertainty_snr_median=float(snr_median),
        uncertainty_snr_valid_bins=snr_valid_bins,
        diagnostic_cost=cost,
        rejection_mask=int(rejection),
    )


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
    """Return every fully contained candidate in the configured search interval.

    This function performs no final ranking.  Keeping the accepted and rejected
    catalogue intact is intentional: ranking policy will be validated against
    synthetic and SPU evidence before it changes the productive selector.
    """
    altitude = np.asarray(altitude_m, dtype=np.float64)
    measured = np.asarray(measured_signal, dtype=np.float64)
    simulated = np.asarray(simulated_molecular_signal, dtype=np.float64)
    if not (measured.ndim == simulated.ndim == altitude.ndim == 1):
        raise ValueError("Rayleigh catalogue inputs must be one-dimensional.")
    if not (measured.shape == simulated.shape == altitude.shape):
        raise ValueError("Rayleigh catalogue inputs must have identical shapes.")
    error = None if measured_signal_error is None else np.asarray(measured_signal_error, dtype=np.float64)
    if error is not None and (error.ndim != 1 or error.shape != measured.shape):
        raise ValueError("measured_signal_error must be 1D and match measured_signal.")
    if not np.all(np.isfinite(altitude)) or not np.all(np.diff(altitude) > 0.0):
        raise ValueError("altitude_m must be finite and strictly increasing.")
    lower = float(min_altitude_m)
    upper = float(max_altitude_m)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("Rayleigh search bounds must be finite and increasing.")
    window = int(window_bins)
    if window < 3:
        raise ValueError("window_bins must be at least three.")

    half = max(window // 2, 1)
    candidates: list[RayleighReferenceCandidate] = []
    for center in range(half, altitude.size - (window - half) + 1):
        start, stop = _candidate_bounds(center, window, altitude.size)
        if altitude[start] < lower or altitude[stop - 1] > upper:
            continue
        candidates.append(
            evaluate_rayleigh_candidate(
                measured,
                simulated,
                altitude,
                center_index=center,
                window_bins=window,
                max_relative_slope=max_relative_slope,
                max_relative_variance=max_relative_variance,
                min_valid_fraction=min_valid_fraction,
                measured_signal_error=error,
            )
        )
    if not candidates:
        raise ValueError("No complete Rayleigh candidate window exists inside the configured search interval.")
    return tuple(candidates)


def accepted_rayleigh_candidates(
    candidates: tuple[RayleighReferenceCandidate, ...] | list[RayleighReferenceCandidate],
) -> tuple[RayleighReferenceCandidate, ...]:
    """Filter a catalogue after QA without imposing a ranking policy."""
    return tuple(candidate for candidate in candidates if candidate.accepted)
