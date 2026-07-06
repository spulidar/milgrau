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
    total = np.nansum(arr, axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float64), where=count > 0)


def error_of_mean(error_matrix: np.ndarray) -> np.ndarray:
    """Combine profile one-sigma errors into uncertainty of the temporal mean."""
    valid_count = np.sum(np.isfinite(error_matrix), axis=0)
    valid_count = np.maximum(valid_count, 1)
    return np.sqrt(np.nansum(error_matrix**2, axis=0)) / valid_count


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
    """Calculate uncertainty of grouped means for a time x altitude error matrix."""
    return np.stack([error_of_mean(error_matrix[group, :]) for group in groups], axis=0)


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
