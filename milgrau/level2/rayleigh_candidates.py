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
    UNIDENTIFIABLE_BACKGROUND = 16


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
    calibration_factor_standard_error: float
    background_offset: float
    background_offset_standard_error: float
    calibration_background_correlation: float
    reduced_chi_square: float
    free_intercept: float
    uncertainty_snr_median: float
    uncertainty_snr_valid_bins: int
    diagnostic_cost: float
    rejection_mask: int

    @property
    def accepted(self) -> bool:
        """Return whether the candidate passes all currently enabled QA gates."""
        return int(self.rejection_mask) == int(RayleighCandidateRejection.NONE)


@dataclass(frozen=True, slots=True)
class RayleighBackgroundFit:
    """One robust broad-span fit of the residual pre-range background."""

    calibration_factor: float
    background_offset: float
    background_offset_standard_error: float
    calibration_background_correlation: float
    success: bool


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


def _robust_profile_background_fit(
    measured: np.ndarray,
    simulated: np.ndarray,
    altitude: np.ndarray,
    error: np.ndarray | None,
    support_indices: np.ndarray,
) -> tuple[float, float, float, float, bool]:
    """Fit one shared raw-signal background over the full Rayleigh search span.

    A 1-km candidate window does not provide enough vertical leverage to
    separate molecular scaling from the ``B*z**2`` RCS term.  This profile fit
    therefore estimates B over the union of all candidate windows using Huber
    iteratively reweighted least squares.  Candidate-local fits subsequently
    estimate only A after removing this shared fitted nuisance.
    """
    index = np.unique(np.asarray(support_indices, dtype=np.int64).ravel())
    x = simulated[index]
    y = measured[index]
    z = altitude[index]
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x > 0.0) & (z > 0.0)
    local_error = None if error is None else error[index]
    if local_error is not None:
        valid &= np.isfinite(local_error) & (local_error > 0.0)
    if np.count_nonzero(valid) < 3:
        return np.nan, np.nan, np.nan, np.nan, False

    x = x[valid]
    y = y[valid]
    z = z[valid]
    if local_error is not None:
        local_error = local_error[valid]
    z_scale = float(np.median(z))
    q = (z / z_scale) ** 2
    base_weights = (
        np.ones_like(y, dtype=np.float64)
        if local_error is None
        else 1.0 / local_error**2
    )

    def solve(weights: np.ndarray) -> tuple[float, float, float, float, float, bool]:
        s_xx = float(np.sum(weights * x * x))
        s_xq = float(np.sum(weights * x * q))
        s_qq = float(np.sum(weights * q * q))
        s_xy = float(np.sum(weights * x * y))
        s_qy = float(np.sum(weights * q * y))
        determinant = s_xx * s_qq - s_xq * s_xq
        ok = bool(
            s_xx > 0.0
            and s_qq > 0.0
            and np.isfinite(determinant)
            and determinant
            > np.finfo(np.float64).eps * (s_xx * s_qq) * 64.0
        )
        if not ok:
            return np.nan, np.nan, s_xx, s_xq, s_qq, False
        factor = (s_xy * s_qq - s_qy * s_xq) / determinant
        background_scaled = (s_qy * s_xx - s_xy * s_xq) / determinant
        return factor, background_scaled, s_xx, s_xq, s_qq, True

    robust_weights = np.ones_like(base_weights)
    factor, background_scaled, s_xx, s_xq, s_qq, ok = solve(base_weights)
    if not ok:
        return np.nan, np.nan, np.nan, np.nan, False
    for _ in range(20):
        residual = y - factor * x - background_scaled * q
        standardized = residual if local_error is None else residual / local_error
        center = float(np.median(standardized))
        scale = 1.482602218505602 * float(np.median(np.abs(standardized - center)))
        scale_floor = np.finfo(np.float64).eps * max(
            1.0, float(np.nanmax(np.abs(standardized)))
        ) * 64.0
        if not np.isfinite(scale) or scale <= scale_floor:
            break
        normalized = np.abs(standardized - center) / (1.345 * scale)
        updated_robust = np.ones_like(normalized)
        outside = normalized > 1.0
        updated_robust[outside] = 1.0 / normalized[outside]
        new_factor, new_background, s_xx, s_xq, s_qq, ok = solve(
            base_weights * updated_robust
        )
        if not ok:
            return np.nan, np.nan, np.nan, np.nan, False
        relative_change = max(
            abs(new_factor - factor) / max(abs(factor), 1.0),
            abs(new_background - background_scaled) / max(abs(background_scaled), 1.0),
        )
        factor, background_scaled = new_factor, new_background
        robust_weights = updated_robust
        if relative_change < 1.0e-10:
            break

    final_weights = base_weights * robust_weights
    factor, background_scaled, s_xx, s_xq, s_qq, ok = solve(final_weights)
    # The broad-span factor is only a nuisance needed to identify B.  Physical
    # positivity is enforced on every candidate-local calibration factor below;
    # rejecting B here would unnecessarily couple background availability to a
    # deliberately broad region that may contain aerosol structure.
    if not ok or not np.isfinite(factor):
        return np.nan, np.nan, np.nan, np.nan, False
    determinant = s_xx * s_qq - s_xq * s_xq
    background = background_scaled / z_scale**2
    background_standard_error = (
        np.sqrt(s_xx / determinant) / z_scale**2 if local_error is not None else np.nan
    )
    correlation = -s_xq / np.sqrt(s_xx * s_qq)
    return (
        float(background),
        float(background_standard_error),
        float(correlation),
        float(factor),
        True,
    )


def fit_rayleigh_background(
    measured_signal: np.ndarray,
    simulated_molecular_signal: np.ndarray,
    altitude_m: np.ndarray,
    *,
    min_altitude_m: float,
    max_altitude_m: float,
    measured_signal_error: np.ndarray | None = None,
) -> RayleighBackgroundFit:
    """Estimate one robust weighted B over a declared vertical search span."""
    measured = np.asarray(measured_signal, dtype=np.float64)
    simulated = np.asarray(simulated_molecular_signal, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    error = (
        None
        if measured_signal_error is None
        else np.asarray(measured_signal_error, dtype=np.float64)
    )
    if not (measured.ndim == simulated.ndim == altitude.ndim == 1):
        raise ValueError("Rayleigh background-fit signal and altitude inputs must be 1D.")
    if not (measured.shape == simulated.shape == altitude.shape):
        raise ValueError("Rayleigh background-fit inputs must have identical shapes.")
    if error is not None and (error.ndim != 1 or error.shape != measured.shape):
        raise ValueError("Rayleigh background-fit error must match the signal shape.")
    if (
        altitude.size < 3
        or not np.all(np.isfinite(altitude))
        or not np.all(np.diff(altitude) > 0.0)
    ):
        raise ValueError(
            "Rayleigh background-fit altitude must be finite, strictly increasing, and contain at least three bins."
        )
    if not np.isfinite(min_altitude_m) or not np.isfinite(max_altitude_m):
        raise ValueError("Rayleigh background-fit altitude limits must be finite.")
    if float(max_altitude_m) <= float(min_altitude_m):
        raise ValueError("Rayleigh background-fit maximum altitude must exceed its minimum.")
    support = np.flatnonzero(
        (altitude >= float(min_altitude_m)) & (altitude <= float(max_altitude_m))
    )
    background, standard_error, correlation, factor, success = (
        _robust_profile_background_fit(
            measured,
            simulated,
            altitude,
            error,
            support,
        )
    )
    return RayleighBackgroundFit(
        calibration_factor=float(factor),
        background_offset=float(background),
        background_offset_standard_error=float(standard_error),
        calibration_background_correlation=float(correlation),
        success=bool(success),
    )


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

    The physical two-parameter model is
    ``RCS(z) = A * molecular_RCS(z) + B * z**2``, equivalent to
    ``signal(z) = A * molecular_signal(z) + B`` before range correction.
    One robust inverse-variance profile fit over the complete candidate-search
    span estimates the shared B; each local window then estimates A and its QA
    after removing B.  Negative measured bins remain available to the fit.
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
    fit_valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & (x > 0.0)
        & (z > 0.0)
    )
    window_error = None
    if error is not None:
        window_error = error[indices]
        fit_valid &= np.isfinite(window_error) & (window_error > 0.0)

    # Estimate one profile/block background over the complete search span.
    # Candidate-local 1-km windows then estimate A conditionally on this shared
    # B, avoiding the near-singular local A/B fit seen in real SPU profiles.
    (
        shared_background,
        shared_background_standard_error,
        shared_parameter_correlation,
        _,
        shared_fit_ok,
    ) = _robust_profile_background_fit(
        measured,
        simulated,
        altitude,
        error,
        indices,
    )
    weights = np.where(fit_valid, 1.0, 0.0)
    if window_error is not None:
        weights = np.divide(
            1.0,
            window_error**2,
            out=np.zeros_like(window_error, dtype=np.float64),
            where=fit_valid,
        )

    x_fit = np.where(fit_valid, x, 0.0)
    background_term = shared_background * z**2
    corrected_for_fit = y - background_term
    y_fit = np.where(fit_valid, corrected_for_fit, 0.0)
    fit_bins = np.count_nonzero(fit_valid, axis=1).astype(np.int32)
    s_xx = np.sum(weights * x_fit * x_fit, axis=1)
    s_xy = np.sum(weights * x_fit * y_fit, axis=1)
    fit_ok = (
        (fit_bins >= 3)
        & np.isfinite(s_xx)
        & (s_xx > 0.0)
        & shared_fit_ok
    )

    factor = np.full(centers.size, np.nan, dtype=np.float64)
    np.divide(
        s_xy,
        s_xx,
        out=factor,
        where=fit_ok,
    )
    background = np.full(centers.size, shared_background, dtype=np.float64)

    fitted = factor[:, None] * x + background_term
    residual = np.where(fit_valid, y - fitted, 0.0)
    degrees_of_freedom = fit_bins.astype(np.float64) - 2.0
    reduced_chi_square = np.full(centers.size, np.nan, dtype=np.float64)
    np.divide(
        np.sum(weights * residual * residual, axis=1),
        degrees_of_freedom,
        out=reduced_chi_square,
        where=fit_ok & (degrees_of_freedom > 0.0),
    )

    factor_standard_error = np.full(centers.size, np.nan, dtype=np.float64)
    if window_error is not None:
        np.sqrt(
            np.divide(1.0, s_xx, out=np.full_like(s_xx, np.nan), where=fit_ok),
            out=factor_standard_error,
        )
    background_standard_error = np.full(
        centers.size, shared_background_standard_error, dtype=np.float64
    )
    parameter_correlation = np.full(
        centers.size, shared_parameter_correlation, dtype=np.float64
    )

    corrected_y = corrected_for_fit
    valid = fit_valid & np.isfinite(corrected_y) & (corrected_y > 0.0)
    valid_bins = np.count_nonzero(valid, axis=1).astype(np.int32)
    total_bins = np.full(centers.size, window, dtype=np.int32)
    valid_fraction = valid_bins.astype(np.float64) / float(window)
    n = valid_bins.astype(np.float64)

    z_valid = np.where(valid, z, 0.0)
    free_intercept = background.copy()

    ratio = np.full_like(y, np.nan, dtype=np.float64)
    np.divide(corrected_y, x, out=ratio, where=valid)
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
        snr_valid = valid & np.isfinite(window_error) & (window_error > 0.0)
        snr_valid_bins = np.count_nonzero(snr_valid, axis=1).astype(np.int32)
        rows = snr_valid_bins > 0
        if np.any(rows):
            snr_values = np.full_like(y, np.nan, dtype=np.float64)
            np.divide(corrected_y, window_error, out=snr_values, where=snr_valid)
            snr_median[rows] = np.nanmedian(snr_values[rows], axis=1)

    rejection = np.zeros(centers.size, dtype=np.int32)
    rejection[valid_fraction < float(min_valid_fraction)] |= int(
        RayleighCandidateRejection.INSUFFICIENT_VALID_FRACTION
    )
    rejection[(~np.isfinite(factor)) | (factor <= 0.0)] |= int(
        RayleighCandidateRejection.INVALID_CALIBRATION
    )
    rejection[~fit_ok] |= int(
        RayleighCandidateRejection.UNIDENTIFIABLE_BACKGROUND
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

    candidates = tuple(
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
            calibration_factor_standard_error=float(factor_standard_error[i]),
            background_offset=float(background[i]),
            background_offset_standard_error=float(background_standard_error[i]),
            calibration_background_correlation=float(parameter_correlation[i]),
            reduced_chi_square=float(reduced_chi_square[i]),
            free_intercept=float(free_intercept[i]),
            uncertainty_snr_median=float(snr_median[i]),
            uncertainty_snr_valid_bins=int(snr_valid_bins[i]),
            diagnostic_cost=float(diagnostic_cost[i]),
            rejection_mask=int(rejection[i]),
        )
        for i in range(centers.size)
    )
    return candidates


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
