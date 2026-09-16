"""Explicit vertical-aggregation diagnostics for high-column Level 2 R&D.

This module does not alter the productive retrieval.  It exists to quantify the
trade between vertical resolution and random-noise reduction before any
high-column backbone uses vertical aggregation.

Two uncertainty cases are carried deliberately:

* ``error_independent`` assumes source-bin errors are mutually independent;
* ``error_fully_correlated`` assumes perfect positive correlation and therefore
  gives the no-noise-cancellation limiting case for an arithmetic mean.

No interpolation is performed and an aggregate is valid only when every source
bin has finite signal and finite non-negative uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VerticalAggregationResult:
    """One non-overlapping aggregation of a uniform vertical grid."""

    altitude_m: np.ndarray
    signal: np.ndarray
    error_independent: np.ndarray
    error_fully_correlated: np.ndarray
    contributing_bin_count: np.ndarray
    source_start_index: np.ndarray
    source_stop_index_exclusive: np.ndarray
    aggregation_width_m: float


@dataclass(frozen=True, slots=True)
class VerticalNoiseCorrelationDiagnostics:
    """Empirical lag correlation of block-demeaned, uncertainty-scaled residuals."""

    lag_bins: np.ndarray
    correlation: np.ndarray
    valid_pair_count: np.ndarray


def _uniform_spacing(altitude_m: np.ndarray) -> float:
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if altitude.ndim != 1 or altitude.size < 2:
        raise ValueError("altitude_m must be a one-dimensional grid with at least two bins.")
    if not np.all(np.isfinite(altitude)) or not np.all(np.diff(altitude) > 0.0):
        raise ValueError("altitude_m must be finite and strictly increasing.")
    spacing = float(altitude[1] - altitude[0])
    if not np.allclose(
        np.diff(altitude),
        spacing,
        rtol=1.0e-7,
        atol=max(abs(spacing) * 1.0e-9, 1.0e-12),
    ):
        raise ValueError("Vertical aggregation currently requires a uniform altitude grid.")
    return spacing


def aggregate_uniform_vertical_bins(
    signal: np.ndarray,
    error: np.ndarray,
    altitude_m: np.ndarray,
    *,
    bins_per_aggregate: int,
) -> VerticalAggregationResult:
    """Average non-overlapping complete groups on a uniform altitude grid.

    The final incomplete source group, if any, is omitted rather than padded.
    Missing/invalid bins invalidate their complete aggregate group; they are
    never interpolated across or silently dropped from the arithmetic mean.

    ``error_independent`` and ``error_fully_correlated`` are deliberately both
    returned because the productive MILGRAU uncertainty model does not yet carry
    altitude-bin covariance.  A future productive aggregation must justify its
    dependence model instead of choosing the more favorable result.
    """
    values = np.asarray(signal, dtype=np.float64)
    uncertainties = np.asarray(error, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if values.ndim != 1 or uncertainties.ndim != 1:
        raise ValueError("signal and error must be one-dimensional.")
    if values.shape != uncertainties.shape or values.shape != altitude.shape:
        raise ValueError("signal, error, and altitude_m must have identical shapes.")
    spacing = _uniform_spacing(altitude)

    group_size = int(bins_per_aggregate)
    if group_size < 1:
        raise ValueError("bins_per_aggregate must be at least one.")
    n_group = values.size // group_size
    if n_group < 1:
        raise ValueError("bins_per_aggregate exceeds the available altitude bins.")

    used = n_group * group_size
    value_group = values[:used].reshape(n_group, group_size)
    error_group = uncertainties[:used].reshape(n_group, group_size)
    altitude_group = altitude[:used].reshape(n_group, group_size)
    valid_group = np.all(
        np.isfinite(value_group)
        & np.isfinite(error_group)
        & (error_group >= 0.0),
        axis=1,
    )

    aggregated_signal = np.full(n_group, np.nan, dtype=np.float64)
    error_independent = np.full(n_group, np.nan, dtype=np.float64)
    error_correlated = np.full(n_group, np.nan, dtype=np.float64)
    count = np.zeros(n_group, dtype=np.int32)
    if np.any(valid_group):
        aggregated_signal[valid_group] = np.mean(value_group[valid_group], axis=1)
        error_independent[valid_group] = (
            np.sqrt(np.sum(error_group[valid_group] ** 2, axis=1)) / group_size
        )
        error_correlated[valid_group] = (
            np.sum(error_group[valid_group], axis=1) / group_size
        )
        count[valid_group] = group_size

    starts = np.arange(n_group, dtype=np.int32) * group_size
    stops = starts + group_size
    return VerticalAggregationResult(
        altitude_m=np.mean(altitude_group, axis=1),
        signal=aggregated_signal,
        error_independent=error_independent,
        error_fully_correlated=error_correlated,
        contributing_bin_count=count,
        source_start_index=starts,
        source_stop_index_exclusive=stops,
        aggregation_width_m=float(group_size * spacing),
    )


def vertical_noise_autocorrelation(
    profile_signal: np.ndarray,
    profile_error: np.ndarray,
    block_labels: np.ndarray,
    *,
    max_lag_bins: int,
) -> VerticalNoiseCorrelationDiagnostics:
    """Estimate vertical correlation after removing each block's mean profile.

    Residuals are divided by the supplied per-profile uncertainty before lag
    correlation is pooled across profiles and altitude.  This is an empirical
    diagnostic, not a proof that the residual field is pure measurement noise:
    unresolved atmospheric variability can also contribute correlation and
    should therefore remain visible rather than be subtracted by assumption.
    """
    signal = np.asarray(profile_signal, dtype=np.float64)
    error = np.asarray(profile_error, dtype=np.float64)
    labels = np.asarray(block_labels)
    if signal.ndim != 2:
        raise ValueError("profile_signal must have dimensions (profile, altitude).")
    if error.shape != signal.shape:
        raise ValueError("profile_error must have the same shape as profile_signal.")
    if labels.ndim != 1 or labels.size != signal.shape[0]:
        raise ValueError("block_labels must contain exactly one label per profile.")
    max_lag = int(max_lag_bins)
    if max_lag < 1 or max_lag >= signal.shape[1]:
        raise ValueError("max_lag_bins must be between one and altitude_bins - 1.")

    normalized = np.full_like(signal, np.nan, dtype=np.float64)
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        if rows.size == 0:
            continue
        block_signal = signal[rows, :]
        block_error = error[rows, :]
        finite_signal_count = np.sum(np.isfinite(block_signal), axis=0)
        signal_sum = np.nansum(block_signal, axis=0)
        block_mean = np.divide(
            signal_sum,
            finite_signal_count,
            out=np.full(signal.shape[1], np.nan, dtype=np.float64),
            where=finite_signal_count > 0,
        )
        residual = block_signal - block_mean[np.newaxis, :]
        valid = (
            np.isfinite(residual)
            & np.isfinite(block_error)
            & (block_error > 0.0)
        )
        normalized[rows, :] = np.divide(
            residual,
            block_error,
            out=np.full_like(residual, np.nan, dtype=np.float64),
            where=valid,
        )

    lags = np.arange(1, max_lag + 1, dtype=np.int32)
    correlation = np.full(max_lag, np.nan, dtype=np.float64)
    valid_pairs = np.zeros(max_lag, dtype=np.int64)
    for output_index, lag in enumerate(lags):
        first = normalized[:, :-lag].ravel()
        second = normalized[:, lag:].ravel()
        valid = np.isfinite(first) & np.isfinite(second)
        valid_pairs[output_index] = int(valid.sum())
        if valid_pairs[output_index] < 2:
            continue
        first_valid = first[valid]
        second_valid = second[valid]
        if np.std(first_valid) == 0.0 or np.std(second_valid) == 0.0:
            continue
        correlation[output_index] = float(np.corrcoef(first_valid, second_valid)[0, 1])

    return VerticalNoiseCorrelationDiagnostics(
        lag_bins=lags,
        correlation=correlation,
        valid_pair_count=valid_pairs,
    )


def autocorrelation_adjusted_snr_gain(
    autocorrelation: np.ndarray,
    *,
    bins_per_aggregate: int,
) -> float:
    """Return mean-SNR gain implied by a stationary lag-correlation sequence.

    For ``N`` equal-variance bins, the variance of their arithmetic mean is
    proportional to ``N + 2*sum((N-k)*rho_k)``.  The caller must provide at
    least lags 1..N-1.  This helper is diagnostic only; it does not claim that
    the real range-bin process is stationary or equal variance.
    """
    correlation = np.asarray(autocorrelation, dtype=np.float64)
    group_size = int(bins_per_aggregate)
    if group_size < 1:
        raise ValueError("bins_per_aggregate must be at least one.")
    if group_size == 1:
        return 1.0
    if correlation.ndim != 1 or correlation.size < group_size - 1:
        raise ValueError("autocorrelation must provide at least lags 1..N-1.")
    used = correlation[: group_size - 1]
    if not np.all(np.isfinite(used)):
        raise ValueError("Required autocorrelation lags must be finite.")

    lags = np.arange(1, group_size, dtype=np.float64)
    variance_numerator = float(
        group_size + 2.0 * np.sum((group_size - lags) * used)
    )
    if not np.isfinite(variance_numerator) or variance_numerator <= 0.0:
        raise ValueError("Autocorrelation sequence implies a non-positive mean variance.")
    variance_ratio = variance_numerator / float(group_size**2)
    return float(1.0 / np.sqrt(variance_ratio))
