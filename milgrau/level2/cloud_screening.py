"""Preliminary cloud and anomalous-layer screening utilities.

These functions are intentionally conservative. They identify sharp, strong
positive anomalies in range-corrected signal profiles, mainly to protect
Rayleigh reference windows from obvious cloud/cirrus contamination. They are not
intended to be a final operational cloud classifier.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from milgrau.level2.config import get_cloud_screening_config


def _moving_percentile(values: np.ndarray, window_bins: int, percentile: float) -> np.ndarray:
    """Return a centered moving percentile with NaN-aware edge handling."""
    arr = np.asarray(values, dtype=np.float64)
    window = max(int(window_bins), 1)
    if window % 2 == 0:
        window += 1
    half = window // 2
    result = np.full(arr.size, np.nan, dtype=np.float64)
    for idx in range(arr.size):
        start = max(idx - half, 0)
        stop = min(idx + half + 1, arr.size)
        local = arr[start:stop]
        if np.isfinite(local).any():
            result[idx] = np.nanpercentile(local, float(percentile))
    return result


def _dilate_mask(mask: np.ndarray, dilation_bins: int) -> np.ndarray:
    """Dilate a boolean 1D mask by a fixed number of bins on both sides."""
    base = np.asarray(mask, dtype=bool)
    dilation = max(int(dilation_bins), 0)
    if dilation == 0 or not base.any():
        return base.copy()
    result = base.copy()
    true_indices = np.flatnonzero(base)
    for idx in true_indices:
        start = max(idx - dilation, 0)
        stop = min(idx + dilation + 1, base.size)
        result[start:stop] = True
    return result


def _remove_short_segments(mask: np.ndarray, min_bins: int) -> np.ndarray:
    """Remove true segments shorter than ``min_bins``."""
    base = np.asarray(mask, dtype=bool)
    minimum = max(int(min_bins), 1)
    result = np.zeros_like(base, dtype=bool)
    idx = 0
    while idx < base.size:
        if not base[idx]:
            idx += 1
            continue
        start = idx
        while idx < base.size and base[idx]:
            idx += 1
        stop = idx
        if stop - start >= minimum:
            result[start:stop] = True
    return result


def detect_anomalous_layer_mask(
    signal: np.ndarray,
    altitude_m: np.ndarray,
    *,
    min_altitude_m: float,
    max_altitude_m: float,
    smooth_bins: int,
    baseline_percentile: float,
    robust_z_threshold: float,
    min_cloud_bins: int,
    vertical_dilation_bins: int,
) -> np.ndarray:
    """Detect strong positive anomalous layers in one RCS-like profile.

    Every detector threshold is supplied explicitly by the caller. The detector
    compares the profile against a configured local low-percentile baseline and
    uses a robust MAD scale. Only positive residuals inside the configured
    altitude interval are flagged.
    """
    profile = np.asarray(signal, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if profile.ndim != 1 or altitude.ndim != 1 or profile.size != altitude.size:
        raise ValueError("signal and altitude_m must be 1D arrays with the same length.")
    if not 0.0 <= float(baseline_percentile) <= 100.0:
        raise ValueError("baseline_percentile must be between 0 and 100.")

    valid_altitude = (altitude >= float(min_altitude_m)) & (altitude <= float(max_altitude_m)) & np.isfinite(altitude)
    finite_profile = np.isfinite(profile)
    candidate = valid_altitude & finite_profile
    if candidate.sum() < max(int(min_cloud_bins), 3):
        return np.zeros(profile.size, dtype=bool)

    baseline = _moving_percentile(profile, smooth_bins, percentile=baseline_percentile)
    residual = profile - baseline
    residual_in_window = residual[candidate]
    residual_in_window = residual_in_window[np.isfinite(residual_in_window)]
    if residual_in_window.size < 3:
        return np.zeros(profile.size, dtype=bool)

    median_residual = float(np.nanmedian(residual_in_window))
    mad = float(np.nanmedian(np.abs(residual_in_window - median_residual)))
    robust_sigma = 1.4826 * mad
    if not np.isfinite(robust_sigma) or robust_sigma <= 0.0:
        finite_values = profile[candidate]
        robust_sigma = float(np.nanstd(finite_values))
    if not np.isfinite(robust_sigma) or robust_sigma <= 0.0:
        return np.zeros(profile.size, dtype=bool)

    robust_z = (residual - median_residual) / robust_sigma
    mask = candidate & np.isfinite(robust_z) & (robust_z >= float(robust_z_threshold)) & (residual > 0.0)
    mask = _remove_short_segments(mask, min_cloud_bins)
    mask = _dilate_mask(mask, vertical_dilation_bins)
    return mask


def detect_reference_contamination(
    cloud_mask: np.ndarray,
    altitude_m: np.ndarray,
    reference_start_m: float,
    reference_stop_m: float,
) -> float:
    """Return the fraction of reference-window bins flagged as anomalous/cloudy."""
    mask = np.asarray(cloud_mask, dtype=bool)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if mask.ndim != 1 or altitude.ndim != 1 or mask.size != altitude.size:
        raise ValueError("cloud_mask and altitude_m must be 1D arrays with the same length.")
    ref_mask = (altitude >= float(reference_start_m)) & (altitude <= float(reference_stop_m)) & np.isfinite(altitude)
    if not ref_mask.any():
        return float("nan")
    return float(mask[ref_mask].sum() / ref_mask.sum())


def cloud_screening_config(config: Mapping) -> dict[str, float | int | bool]:
    """Return the strict explicit cloud-screening policy for Level 2."""
    return get_cloud_screening_config(config)
