"""Analog/photon-counting signal gluing utilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import linregress, shapiro


def _as_1d(values: np.ndarray) -> np.ndarray:
    """Return a contiguous one-dimensional float array."""
    arr = np.ascontiguousarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("gluing signals must be one-dimensional arrays.")
    return arr


def _window_length(value: int) -> int:
    """Return a valid even window length for centered gluing windows."""
    window = max(int(value), 4)
    return window + 1 if window % 2 else window


def _origin_slope(lower: np.ndarray, upper: np.ndarray) -> float:
    """Return the multiplicative factor that maps lower signal to upper signal."""
    valid = np.isfinite(lower) & np.isfinite(upper)
    x = lower[valid]
    y = upper[valid]
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return np.nan
    return float(np.dot(x, y) / denom)


def _window_diagnostics(lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    """Calculate statistical gluing diagnostics for one candidate window."""
    valid = np.isfinite(lower) & np.isfinite(upper)
    if valid.sum() != lower.size or lower.size < 4:
        return {
            "correlation": np.nan,
            "intercept_percent": np.inf,
            "shapiro_p_value": np.nan,
            "minmax_ratio": np.nan,
            "slope": np.nan,
        }

    if float(np.nanstd(lower)) <= 0.0 or float(np.nanstd(upper)) <= 0.0:
        return {
            "correlation": np.nan,
            "intercept_percent": np.inf,
            "shapiro_p_value": np.nan,
            "minmax_ratio": np.nan,
            "slope": np.nan,
        }

    _, intercept_free, correlation, _, _ = linregress(lower, upper)
    upper_mean = float(np.nanmean(upper))
    if not np.isfinite(upper_mean) or abs(upper_mean) <= 1e-30:
        intercept_percent = np.inf
    else:
        intercept_percent = abs(float(intercept_free) / upper_mean * 100.0)

    slope = _origin_slope(lower, upper)
    if np.isfinite(slope):
        residuals = slope * lower - upper
        residuals = residuals[np.isfinite(residuals)]
    else:
        residuals = np.array([], dtype=np.float64)

    if residuals.size < 3:
        shapiro_p_value = np.nan
    elif float(np.nanstd(residuals)) <= 0.0:
        shapiro_p_value = 1.0
    else:
        try:
            _, shapiro_p_value = shapiro(residuals)
            shapiro_p_value = float(shapiro_p_value)
        except Exception:
            shapiro_p_value = np.nan

    lower_max = float(np.nanmax(lower))
    upper_max = float(np.nanmax(upper))
    if lower_max <= 0.0 or upper_max <= 0.0:
        minmax_ratio = np.nan
    else:
        minmax_ratio = float(
            min(
                float(np.nanmin(lower)) / lower_max,
                float(np.nanmin(upper)) / upper_max,
            )
        )

    return {
        "correlation": float(correlation),
        "intercept_percent": float(intercept_percent),
        "shapiro_p_value": float(shapiro_p_value),
        "minmax_ratio": float(minmax_ratio),
        "slope": float(slope),
    }


def _select_window(
    lower: np.ndarray,
    upper: np.ndarray,
    window: int,
    correlation_threshold: float,
    intercept_threshold: float,
    gaussian_threshold: float,
    minmax_threshold: float,
    min_idx: int,
    max_idx: int,
) -> dict[str, float | int | str]:
    """Search for the best gluing window using statistical compatibility tests."""
    n_bins = lower.size
    start = max(int(min_idx), 0)
    stop = min(int(max_idx), n_bins)

    if stop - start < window:
        start = 0
        stop = n_bins

    best: dict[str, float | int | str] = {
        "idx": -1,
        "center": -1,
        "score": np.nan,
        "correlation": np.nan,
        "intercept_percent": np.inf,
        "shapiro_p_value": np.nan,
        "minmax_ratio": np.nan,
        "slope": np.nan,
        "selection_mode": "failed",
    }

    if stop - start < window:
        return best

    best_score = -np.inf
    intercept_scale_value = 40.0

    for idx in range(start, stop - window + 1):
        lower_window = lower[idx : idx + window]
        upper_window = upper[idx : idx + window]
        diag = _window_diagnostics(lower_window, upper_window)

        correlation = float(diag["correlation"])
        intercept_percent = float(diag["intercept_percent"])
        shapiro_p_value = float(diag["shapiro_p_value"])
        minmax_ratio = float(diag["minmax_ratio"])
        slope = float(diag["slope"])

        gluing_possible = (
            np.isfinite(correlation)
            and np.isfinite(intercept_percent)
            and np.isfinite(shapiro_p_value)
            and np.isfinite(minmax_ratio)
            and np.isfinite(slope)
            and correlation > correlation_threshold
            and intercept_percent < intercept_threshold
            and shapiro_p_value >= gaussian_threshold
            and minmax_ratio > minmax_threshold
            and slope > 0.0
        )

        if not gluing_possible:
            continue

        clipped_intercept = min(intercept_percent, intercept_scale_value)
        intercept_score = 1.0 - clipped_intercept / intercept_scale_value
        score = float(correlation * intercept_score * minmax_ratio)

        if score > best_score:
            best_score = score
            best = {
                "idx": int(idx),
                "center": int(idx + window // 2),
                "score": score,
                "correlation": correlation,
                "intercept_percent": intercept_percent,
                "shapiro_p_value": shapiro_p_value,
                "minmax_ratio": minmax_ratio,
                "slope": slope,
                "selection_mode": "statistical",
            }

    return best


def glue_signals_at_bins(
    analog_sig: np.ndarray,
    pc_sig: np.ndarray,
    min_bin: int,
    max_bin: int,
    slope: float,
    intercept: float = 0.0,
) -> np.ndarray:
    """Glue two 1D signals with a linear fade-in/fade-out transition.

    The analog signal is scaled to the photon-counting scale. Below the gluing
    region the scaled analog signal is used. Above the gluing region the photon
    counting signal is used. Inside the region, both detector modes are blended
    smoothly with linear weights.
    """
    analog = _as_1d(analog_sig)
    photon = _as_1d(pc_sig)

    if analog.size != photon.size:
        raise ValueError("analog_sig and pc_sig must have the same length.")

    min_bin = max(int(min_bin), 0)
    max_bin = min(int(max_bin), analog.size)

    if max_bin <= min_bin:
        raise ValueError("max_bin must be greater than min_bin for gluing.")

    scaled_analog = float(slope) * analog + float(intercept)

    glued = photon.copy()
    glued[:min_bin] = scaled_analog[:min_bin]

    gluing_length = max_bin - min_bin
    lower_weights = 1.0 - np.arange(gluing_length, dtype=np.float64) / float(gluing_length)
    upper_weights = 1.0 - lower_weights

    glued[min_bin:max_bin] = (
        lower_weights * scaled_analog[min_bin:max_bin]
        + upper_weights * photon[min_bin:max_bin]
    )

    return glued


def slide_glue_signals(
    analog_sig: np.ndarray,
    pc_sig: np.ndarray,
    altitude: np.ndarray | None = None,
    window_size: int = 150,
    min_corr: float = 0.90,
    search_min_idx: int = 0,
    search_max_idx: int | None = None,
    intercept_threshold: float = 0.5,
    gaussian_threshold: float = 0.1,
    minmax_threshold: float = 0.5,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, int, float, float] | tuple[np.ndarray, int, float, float, dict[str, Any]]:
    """Glue analog and photon-counting signals into one dynamic-range profile."""
    analog = _as_1d(analog_sig)
    photon = _as_1d(pc_sig)

    if analog.size != photon.size:
        raise ValueError("analog_sig and pc_sig must have the same length.")

    window = _window_length(window_size)
    search_stop = analog.size if search_max_idx is None else int(search_max_idx)

    selected = _select_window(
        lower=analog,
        upper=photon,
        window=window,
        correlation_threshold=float(min_corr),
        intercept_threshold=float(intercept_threshold),
        gaussian_threshold=float(gaussian_threshold),
        minmax_threshold=float(minmax_threshold),
        min_idx=int(search_min_idx),
        max_idx=search_stop,
    )

    best_idx = int(selected["idx"])

    if best_idx < 0:
        glued_signal = photon.copy()
        split_point = -1
        min_bin = -1
        max_bin = -1
        slope = 1.0
        intercept = 0.0
    else:
        min_bin = best_idx
        max_bin = min(best_idx + window, analog.size)
        split_point = int(selected["center"])
        slope = float(selected["slope"])
        intercept = 0.0
        glued_signal = glue_signals_at_bins(
            analog_sig=analog,
            pc_sig=photon,
            min_bin=min_bin,
            max_bin=max_bin,
            slope=slope,
            intercept=intercept,
        )

    if return_diagnostics:
        diagnostics: dict[str, Any] = {
            "best_idx": int(best_idx),
            "best_corr": float(selected["correlation"]),
            "split_point": int(split_point),
            "min_bin": int(min_bin),
            "max_bin": int(max_bin),
            "search_min_idx": int(search_min_idx),
            "search_max_idx": int(search_stop),
            "window_size": int(window),
            "gluing_score": float(selected["score"]),
            "selection_mode": str(selected["selection_mode"]),
            "intercept_percent": float(selected["intercept_percent"]),
            "shapiro_p_value": float(selected["shapiro_p_value"]),
            "minmax_ratio": float(selected["minmax_ratio"]),
            "slope_zero_intercept": float(selected["slope"]),
            # Compatibility names used by downstream diagnostics.
            "relative_intercept": (
                float(selected["intercept_percent"]) / 100.0
                if np.isfinite(float(selected["intercept_percent"]))
                else np.nan
            ),
            "gaussian_score": float(selected["shapiro_p_value"]),
            "minmax_score": float(selected["minmax_ratio"]),
        }

        if altitude is not None and split_point >= 0:
            altitude_arr = np.asarray(altitude, dtype=np.float64)
            if altitude_arr.ndim == 1 and split_point < altitude_arr.size:
                diagnostics["split_altitude"] = float(altitude_arr[split_point])
                diagnostics["min_altitude"] = float(altitude_arr[min_bin])
                diagnostics["max_altitude"] = float(altitude_arr[max_bin - 1])

        return glued_signal, split_point, float(slope), float(intercept), diagnostics

    return glued_signal, split_point, float(slope), float(intercept)
