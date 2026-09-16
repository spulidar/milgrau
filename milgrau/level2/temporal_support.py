"""Pure temporal-support diagnostics for future high-column backbone R&D.

These helpers do not accept/reject a backbone and do not perform KFS.  They
quantify whether altitude-resolved information is temporally represented across
explicitly weighted retrieval blocks, so a long mean cannot hide that its
far-range signal comes predominantly from a short subperiod.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TemporalSupportDiagnostics:
    """Altitude-resolved support and contribution diagnostics across blocks."""

    valid_flag_block: np.ndarray
    supporting_block_count: np.ndarray
    supporting_weight_fraction: np.ndarray
    contribution_fraction_block: np.ndarray
    dominant_contribution_fraction: np.ndarray
    dominant_block_index: np.ndarray
    weighted_mean_signal: np.ndarray


@dataclass(frozen=True, slots=True)
class ContiguousSubwindowDiagnostics:
    """Weighted signal/support state for every contiguous block subwindow."""

    start_block_index: np.ndarray
    stop_block_index_exclusive: np.ndarray
    supporting_block_count: np.ndarray
    supporting_weight_fraction: np.ndarray
    weighted_mean_signal: np.ndarray


def _validate_inputs(
    block_signal: np.ndarray,
    block_error: np.ndarray,
    block_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = np.asarray(block_signal, dtype=np.float64)
    error = np.asarray(block_error, dtype=np.float64)
    weights = np.asarray(block_weights, dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError("block_signal must have dimensions (block, altitude).")
    if error.shape != signal.shape:
        raise ValueError("block_error must have the same shape as block_signal.")
    if weights.shape != (signal.shape[0],):
        raise ValueError("block_weights must contain exactly one weight per block.")
    if signal.shape[0] < 1 or signal.shape[1] < 1:
        raise ValueError("Temporal support diagnostics require at least one block and altitude bin.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("block_weights must be finite and strictly positive.")
    return signal, error, weights


def temporal_support_diagnostics(
    block_signal: np.ndarray,
    block_error: np.ndarray,
    block_weights: np.ndarray,
) -> TemporalSupportDiagnostics:
    """Quantify temporal support and absolute signal contribution by altitude.

    A block/altitude sample is supported only when signal and one-sigma error are
    finite and the error is non-negative.  ``block_weights`` is explicit so
    unequal profile counts/durations are never silently treated as equal.

    Contribution fractions use ``weight * abs(signal)``.  They are a dominance
    diagnostic, not a physical signed-source decomposition and not an
    uncertainty weight.  No acceptance threshold is applied here.
    """
    signal, error, weights = _validate_inputs(block_signal, block_error, block_weights)
    valid = np.isfinite(signal) & np.isfinite(error) & (error >= 0.0)
    n_block, n_altitude = signal.shape

    supporting_count = valid.sum(axis=0, dtype=np.int32)
    total_weight = float(np.sum(weights))
    supported_weight = np.sum(np.where(valid, weights[:, np.newaxis], 0.0), axis=0)
    weight_fraction = supported_weight / total_weight

    weighted_signal = np.where(valid, weights[:, np.newaxis] * signal, 0.0)
    weighted_sum = np.sum(weighted_signal, axis=0)
    mean_signal = np.full(n_altitude, np.nan, dtype=np.float64)
    has_support = supported_weight > 0.0
    mean_signal[has_support] = weighted_sum[has_support] / supported_weight[has_support]

    absolute_contribution = np.where(
        valid,
        weights[:, np.newaxis] * np.abs(signal),
        0.0,
    )
    absolute_total = np.sum(absolute_contribution, axis=0)
    contribution_fraction = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    nonzero_total = absolute_total > 0.0
    contribution_fraction[:, nonzero_total] = (
        absolute_contribution[:, nonzero_total] / absolute_total[nonzero_total]
    )

    dominant_fraction = np.full(n_altitude, np.nan, dtype=np.float64)
    dominant_index = np.full(n_altitude, -1, dtype=np.int32)
    if np.any(nonzero_total):
        dominant_fraction[nonzero_total] = np.max(
            contribution_fraction[:, nonzero_total], axis=0
        )
        dominant_index[nonzero_total] = np.argmax(
            contribution_fraction[:, nonzero_total], axis=0
        ).astype(np.int32)

    return TemporalSupportDiagnostics(
        valid_flag_block=valid.astype(np.int8),
        supporting_block_count=supporting_count,
        supporting_weight_fraction=weight_fraction,
        contribution_fraction_block=contribution_fraction,
        dominant_contribution_fraction=dominant_fraction,
        dominant_block_index=dominant_index,
        weighted_mean_signal=mean_signal,
    )


def contiguous_subwindow_diagnostics(
    block_signal: np.ndarray,
    block_error: np.ndarray,
    block_weights: np.ndarray,
    *,
    window_blocks: int,
) -> ContiguousSubwindowDiagnostics:
    """Return support and weighted means for every contiguous block window.

    ``window_blocks`` is explicit and therefore belongs to an experiment/design
    choice, not a hidden constant.  The helper reports diagnostics only; it does
    not choose a preferred window or define a productive stability threshold.
    """
    signal, error, weights = _validate_inputs(block_signal, block_error, block_weights)
    window = int(window_blocks)
    n_block, n_altitude = signal.shape
    if window < 1 or window > n_block:
        raise ValueError("window_blocks must be between 1 and the number of blocks.")

    starts = np.arange(0, n_block - window + 1, dtype=np.int32)
    stops = starts + window
    n_window = starts.size
    count = np.zeros((n_window, n_altitude), dtype=np.int32)
    fraction = np.zeros((n_window, n_altitude), dtype=np.float64)
    mean = np.full((n_window, n_altitude), np.nan, dtype=np.float64)

    for window_index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        sub_signal = signal[start:stop]
        sub_error = error[start:stop]
        sub_weights = weights[start:stop]
        valid = np.isfinite(sub_signal) & np.isfinite(sub_error) & (sub_error >= 0.0)
        count[window_index] = valid.sum(axis=0, dtype=np.int32)
        total_weight = float(np.sum(sub_weights))
        supported_weight = np.sum(
            np.where(valid, sub_weights[:, np.newaxis], 0.0), axis=0
        )
        fraction[window_index] = supported_weight / total_weight
        weighted_sum = np.sum(
            np.where(valid, sub_weights[:, np.newaxis] * sub_signal, 0.0), axis=0
        )
        has_support = supported_weight > 0.0
        mean[window_index, has_support] = (
            weighted_sum[has_support] / supported_weight[has_support]
        )

    return ContiguousSubwindowDiagnostics(
        start_block_index=starts,
        stop_block_index_exclusive=stops,
        supporting_block_count=count,
        supporting_weight_fraction=fraction,
        weighted_mean_signal=mean,
    )
