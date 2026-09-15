"""Temporal block utilities for Level 2 retrievals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def nanmean_or_nan(matrix: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return a NaN-safe mean without RuntimeWarning for all-NaN slices."""
    arr = np.asarray(matrix, dtype=np.float64)
    valid = np.isfinite(arr)
    count = valid.sum(axis=axis)
    total = np.where(valid, arr, 0.0).sum(axis=axis)
    return np.divide(
        total,
        count,
        out=np.full_like(total, np.nan, dtype=np.float64),
        where=count > 0,
    )


def error_of_mean(error_matrix: np.ndarray) -> np.ndarray:
    """Combine finite non-negative one-sigma errors into uncertainty of a mean.

    This helper assumes independent sample errors and is retained for
    measurement-noise reductions such as profile-to-block averaging.
    Productive means with reported uncertainty should use
    :func:`mean_and_error_of_mean` so the value and uncertainty share one
    scientific support mask.
    """
    errors = np.asarray(error_matrix, dtype=np.float64)
    valid = np.isfinite(errors) & (errors >= 0.0)
    valid_count = valid.sum(axis=0)
    combined = np.sqrt(np.where(valid, errors**2, 0.0).sum(axis=0))
    return np.divide(
        combined,
        valid_count,
        out=np.full_like(combined, np.nan, dtype=np.float64),
        where=valid_count > 0,
    )


def mean_and_error_of_mean(
    matrix: np.ndarray,
    error_matrix: np.ndarray,
    *,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce values/errors with common support and independent-noise quadrature.

    A sample contributes only when the value is finite and its one-sigma
    uncertainty is finite and non-negative. The returned ``n_effective`` is
    the number of common-support samples entering both the mean and the
    uncertainty denominator. This reduction assumes the retained errors are
    independent; it is appropriate for the current profile-to-block
    measurement-noise model, not for mixed block-level model uncertainty.
    """
    values = np.asarray(matrix, dtype=np.float64)
    errors = np.asarray(error_matrix, dtype=np.float64)
    if values.shape != errors.shape:
        raise ValueError("matrix and error_matrix must have the same shape.")

    valid = np.isfinite(values) & np.isfinite(errors) & (errors >= 0.0)
    n_effective = valid.sum(axis=axis)
    value_total = np.where(valid, values, 0.0).sum(axis=axis)
    mean = np.divide(
        value_total,
        n_effective,
        out=np.full_like(value_total, np.nan, dtype=np.float64),
        where=n_effective > 0,
    )
    combined_error = np.sqrt(np.where(valid, errors**2, 0.0).sum(axis=axis))
    mean_error = np.divide(
        combined_error,
        n_effective,
        out=np.full_like(combined_error, np.nan, dtype=np.float64),
        where=n_effective > 0,
    )
    return mean, mean_error, np.asarray(n_effective, dtype=np.int64)


def mean_and_correlated_error_bound(
    matrix: np.ndarray,
    error_matrix: np.ndarray,
    *,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce values/errors with common support and a full-correlation bound.

    For an equally weighted mean with individual one-sigma uncertainties
    ``sigma_i`` and unknown covariance, the maximum variance permitted by
    pairwise correlations in [-1, 1] is obtained for rho_ij = +1.  The
    resulting conservative bound is ``sigma_mean = sum(sigma_i) / N``.

    MILGRAU uses this bound for block-to-aggregate optical products because the
    current block KFS Monte Carlo mixes independent signal noise with shared or
    unresolved nuisance terms (notably lidar-ratio and reference-boundary
    uncertainty). It therefore must not receive an automatic 1/sqrt(N)
    reduction until those components are explicitly decomposed.
    """
    values = np.asarray(matrix, dtype=np.float64)
    errors = np.asarray(error_matrix, dtype=np.float64)
    if values.shape != errors.shape:
        raise ValueError("matrix and error_matrix must have the same shape.")

    valid = np.isfinite(values) & np.isfinite(errors) & (errors >= 0.0)
    n_effective = valid.sum(axis=axis)
    value_total = np.where(valid, values, 0.0).sum(axis=axis)
    mean = np.divide(
        value_total,
        n_effective,
        out=np.full_like(value_total, np.nan, dtype=np.float64),
        where=n_effective > 0,
    )
    correlated_error_sum = np.where(valid, errors, 0.0).sum(axis=axis)
    mean_error = np.divide(
        correlated_error_sum,
        n_effective,
        out=np.full_like(correlated_error_sum, np.nan, dtype=np.float64),
        where=n_effective > 0,
    )
    return mean, mean_error, np.asarray(n_effective, dtype=np.int64)


def valid_block_mean(block_matrix: np.ndarray, valid_block: np.ndarray) -> np.ndarray:
    """Average a block x altitude product using only accepted retrieval blocks."""
    matrix = np.asarray(block_matrix, dtype=np.float64)
    valid = np.asarray(valid_block, dtype=bool)
    if matrix.ndim != 2 or valid.ndim != 1 or valid.size != matrix.shape[0]:
        raise ValueError("block_matrix must be 2D and valid_block must match its block axis.")
    if valid.any():
        return nanmean_or_nan(matrix[valid, :], axis=0)
    return np.full(matrix.shape[-1], np.nan, dtype=np.float64)


def valid_block_error(block_error_matrix: np.ndarray, valid_block: np.ndarray) -> np.ndarray:
    """Combine block uncertainties using only accepted retrieval blocks.

    This legacy helper retains independent-error quadrature. Productive optical
    aggregation uses :func:`valid_block_mean_and_error`, whose policy is the
    conservative full-correlation bound for the current mixed KFS uncertainty.
    """
    errors = np.asarray(block_error_matrix, dtype=np.float64)
    valid = np.asarray(valid_block, dtype=bool)
    if errors.ndim != 2 or valid.ndim != 1 or valid.size != errors.shape[0]:
        raise ValueError(
            "block_error_matrix must be 2D and valid_block must match its block axis."
        )
    if valid.any():
        return error_of_mean(errors[valid, :])
    return np.full(errors.shape[-1], np.nan, dtype=np.float64)


def valid_block_mean_and_error(
    block_matrix: np.ndarray,
    block_error_matrix: np.ndarray,
    valid_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate accepted optical blocks using the conservative correlation bound."""
    matrix = np.asarray(block_matrix, dtype=np.float64)
    errors = np.asarray(block_error_matrix, dtype=np.float64)
    valid = np.asarray(valid_block, dtype=bool)
    if (
        matrix.ndim != 2
        or errors.shape != matrix.shape
        or valid.ndim != 1
        or valid.size != matrix.shape[0]
    ):
        raise ValueError(
            "block matrices must be matching 2D arrays and valid_block must match their block axis."
        )
    if valid.any():
        return mean_and_correlated_error_bound(
            matrix[valid, :], errors[valid, :], axis=0
        )
    shape = matrix.shape[-1]
    return (
        np.full(shape, np.nan, dtype=np.float64),
        np.full(shape, np.nan, dtype=np.float64),
        np.zeros(shape, dtype=np.int64),
    )


def block_groups(time_values: np.ndarray, minutes: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return block labels and index groups for temporal averaging."""
    times = pd.to_datetime(time_values)
    labels = times.floor(f"{int(minutes)}min")
    unique_labels = pd.Index(labels).unique().sort_values()
    groups = [np.where(labels == label)[0] for label in unique_labels]
    return unique_labels.to_numpy(dtype="datetime64[ns]"), groups


def mean_by_groups(matrix: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Calculate NaN-safe means for a time x altitude matrix over index groups."""
    return np.stack([nanmean_or_nan(matrix[group, :], axis=0) for group in groups], axis=0)


def error_by_groups(error_matrix: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Calculate uncertainty-only grouped means for a time x altitude error matrix."""
    return np.stack([error_of_mean(error_matrix[group, :]) for group in groups], axis=0)


def mean_error_by_groups(
    matrix: np.ndarray,
    error_matrix: np.ndarray,
    groups: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate grouped means/errors/counts with one common support mask."""
    reduced = [
        mean_and_error_of_mean(matrix[group, :], error_matrix[group, :], axis=0)
        for group in groups
    ]
    means, errors, counts = zip(*reduced, strict=True)
    return np.stack(means, axis=0), np.stack(errors, axis=0), np.stack(counts, axis=0)


def mask_by_groups(mask_matrix: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Return the fraction of profiles masked in each temporal block."""
    mask = np.asarray(mask_matrix, dtype=np.float64)
    return np.stack([np.nanmean(mask[group, :], axis=0) for group in groups], axis=0)


def expand_blocks_to_time(block_matrix: np.ndarray, groups: list[np.ndarray], n_time: int) -> np.ndarray:
    """Expand block x altitude products back to time x altitude."""
    out = np.full((n_time, block_matrix.shape[-1]), np.nan, dtype=np.float64)
    for block_idx, group in enumerate(groups):
        out[group, :] = block_matrix[block_idx, :]
    return out


def expand_block_vector_to_time(
    block_values: np.ndarray,
    groups: list[np.ndarray],
    n_time: int,
    dtype: Any = np.float64,
) -> np.ndarray:
    """Expand one value per block back to one value per profile time."""
    fill = 0 if np.issubdtype(np.dtype(dtype), np.integer) else np.nan
    out = np.full(n_time, fill, dtype=dtype)
    for block_idx, group in enumerate(groups):
        out[group] = block_values[block_idx]
    return out
