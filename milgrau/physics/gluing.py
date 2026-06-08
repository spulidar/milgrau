"""Analog/photon-counting signal gluing utilities.

The routines in this module are intended to operate on Level 1
``corrected_signal`` profiles, before range correction.  This keeps the analog
background/offset term additive in the native instrumental signal space; after
merging, the caller can multiply the glued profile by range squared.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MERGE_SOURCE_PC = 0
MERGE_SOURCE_BLEND = 1
MERGE_SOURCE_ANALOG = 2
MERGE_SOURCE_INVALID = 3


def _as_1d(values: np.ndarray) -> np.ndarray:
    """Return a contiguous one-dimensional float array."""
    arr = np.ascontiguousarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("gluing signals must be one-dimensional arrays.")
    return arr


def _as_bool_1d(values: np.ndarray | None, size: int) -> np.ndarray:
    """Return a one-dimensional boolean mask, or all false when missing."""
    if values is None:
        return np.zeros(size, dtype=bool)
    arr = np.asarray(values, dtype=bool)
    if arr.ndim != 1 or arr.size != size:
        raise ValueError("pc_saturation_mask must be one-dimensional and match the signal length.")
    return np.ascontiguousarray(arr)


def _window_length(value: int) -> int:
    """Return a valid even window length for centered gluing windows."""
    window = max(int(value), 4)
    return window + 1 if window % 2 else window


def _modified_regression(analog: np.ndarray, photon: np.ndarray) -> tuple[float, float]:
    """Return coefficients mapping analog to virtual photon-counting signal.

    Following the Newsom/ARM MERGE approach, the fit is performed with analog as
    the dependent variable and photon-counting as the independent variable.  This
    avoids the bias introduced when the fit interval is constrained in photon
    count-rate space.  The resulting relation is inverted to return
    ``photon_virtual = slope * analog + intercept``.
    """
    valid = np.isfinite(analog) & np.isfinite(photon)
    x = photon[valid]
    y = analog[valid]
    if x.size < 4 or float(np.nanstd(x)) <= 0.0 or float(np.nanstd(y)) <= 0.0:
        return np.nan, np.nan
    a_prime, b_prime = np.polyfit(x, y, 1)
    if not np.isfinite(a_prime) or abs(float(a_prime)) <= 1e-30:
        return np.nan, np.nan
    slope = 1.0 / float(a_prime)
    intercept = -float(b_prime) / float(a_prime)
    return float(slope), float(intercept)


def _window_diagnostics(
    analog: np.ndarray,
    photon: np.ndarray,
    invalid_mask: np.ndarray,
) -> dict[str, float | int]:
    """Calculate residual-based diagnostics for one candidate gluing window."""
    valid = np.isfinite(analog) & np.isfinite(photon) & ~invalid_mask
    valid_count = int(valid.sum())
    if valid_count < 4:
        return {
            "valid_count": valid_count,
            "correlation": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "relative_rmse": np.inf,
            "relative_bias": np.inf,
            "intercept_percent": np.inf,
            "dynamic_range_ratio": np.nan,
        }

    a = analog[valid]
    p = photon[valid]
    if float(np.nanstd(a)) <= 0.0 or float(np.nanstd(p)) <= 0.0:
        return {
            "valid_count": valid_count,
            "correlation": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "relative_rmse": np.inf,
            "relative_bias": np.inf,
            "intercept_percent": np.inf,
            "dynamic_range_ratio": np.nan,
        }

    slope, intercept = _modified_regression(a, p)
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0.0:
        return {
            "valid_count": valid_count,
            "correlation": np.nan,
            "slope": slope,
            "intercept": intercept,
            "relative_rmse": np.inf,
            "relative_bias": np.inf,
            "intercept_percent": np.inf,
            "dynamic_range_ratio": np.nan,
        }

    virtual_pc = slope * a + intercept
    residual = virtual_pc - p
    scale = float(np.nanmean(np.abs(p)))
    if not np.isfinite(scale) or scale <= 1e-30:
        scale = float(np.nanmax(np.abs(p)))
    if not np.isfinite(scale) or scale <= 1e-30:
        scale = 1.0

    relative_rmse = float(np.sqrt(np.nanmean(residual**2)) / scale)
    relative_bias = float(np.nanmean(residual) / scale)
    intercept_percent = float(abs(intercept) / scale * 100.0)
    correlation = float(np.corrcoef(a, p)[0, 1])
    dynamic_range_ratio = float((np.nanmax(p) - np.nanmin(p)) / scale)

    return {
        "valid_count": valid_count,
        "correlation": correlation,
        "slope": float(slope),
        "intercept": float(intercept),
        "relative_rmse": relative_rmse,
        "relative_bias": relative_bias,
        "intercept_percent": intercept_percent,
        "dynamic_range_ratio": dynamic_range_ratio,
    }


def _select_window(
    analog: np.ndarray,
    photon: np.ndarray,
    invalid_mask: np.ndarray,
    window: int,
    correlation_threshold: float,
    intercept_threshold: float,
    min_dynamic_range_ratio: float,
    min_idx: int,
    max_idx: int,
    max_relative_rmse: float,
    max_relative_bias: float,
    min_valid_fraction: float,
) -> dict[str, float | int | str]:
    """Search for the best gluing window using residual minimization."""
    n_bins = analog.size
    start = max(int(min_idx), 0)
    stop = min(int(max_idx), n_bins)
    if stop - start < window:
        start = 0
        stop = n_bins

    best: dict[str, float | int | str] = {
        "idx": -1,
        "center": -1,
        "score": np.inf,
        "correlation": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "relative_rmse": np.inf,
        "relative_bias": np.inf,
        "intercept_percent": np.inf,
        "dynamic_range_ratio": np.nan,
        "valid_count": 0,
        "selection_mode": "failed",
    }
    if stop - start < window:
        return best

    min_valid_count = max(int(np.ceil(float(min_valid_fraction) * window)), 4)
    best_score = np.inf
    for idx in range(start, stop - window + 1):
        mask_window = invalid_mask[idx : idx + window]
        if np.any(mask_window):
            continue
        diag = _window_diagnostics(
            analog=analog[idx : idx + window],
            photon=photon[idx : idx + window],
            invalid_mask=mask_window,
        )
        valid_count = int(diag["valid_count"])
        correlation = float(diag["correlation"])
        slope = float(diag["slope"])
        intercept = float(diag["intercept"])
        relative_rmse = float(diag["relative_rmse"])
        relative_bias = float(diag["relative_bias"])
        intercept_percent = float(diag["intercept_percent"])
        dynamic_range_ratio = float(diag["dynamic_range_ratio"])

        gluing_possible = (
            valid_count >= min_valid_count
            and np.isfinite(correlation)
            and np.isfinite(slope)
            and np.isfinite(intercept)
            and np.isfinite(relative_rmse)
            and np.isfinite(relative_bias)
            and np.isfinite(intercept_percent)
            and np.isfinite(dynamic_range_ratio)
            and slope > 0.0
            and correlation >= correlation_threshold
            and intercept_percent <= intercept_threshold
            and dynamic_range_ratio >= min_dynamic_range_ratio
            and relative_rmse <= max_relative_rmse
            and abs(relative_bias) <= max_relative_bias
        )
        if not gluing_possible:
            continue

        score = float(relative_rmse + abs(relative_bias) + 0.001 * intercept_percent)
        if score < best_score:
            best_score = score
            best = {
                "idx": int(idx),
                "center": int(idx + window // 2),
                "score": score,
                "correlation": correlation,
                "slope": slope,
                "intercept": intercept,
                "relative_rmse": relative_rmse,
                "relative_bias": relative_bias,
                "intercept_percent": intercept_percent,
                "dynamic_range_ratio": dynamic_range_ratio,
                "valid_count": valid_count,
                "selection_mode": "residual_minimization",
            }

    return best


def merge_source_flags(size: int, min_bin: int, max_bin: int, split_failed: bool = False) -> np.ndarray:
    """Return per-bin merge-source flags for a glued profile."""
    if split_failed:
        return np.zeros(int(size), dtype=np.int8)
    flags = np.zeros(int(size), dtype=np.int8)
    min_bin = max(int(min_bin), 0)
    max_bin = min(int(max_bin), int(size))
    flags[:min_bin] = MERGE_SOURCE_ANALOG
    flags[min_bin:max_bin] = MERGE_SOURCE_BLEND
    flags[max_bin:] = MERGE_SOURCE_PC
    return flags


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
    analog_weights = 1.0 - np.arange(gluing_length, dtype=np.float64) / float(gluing_length)
    photon_weights = 1.0 - analog_weights
    glued[min_bin:max_bin] = (
        analog_weights * scaled_analog[min_bin:max_bin]
        + photon_weights * photon[min_bin:max_bin]
    )
    return glued


def propagate_glued_error(
    analog_error: np.ndarray,
    photon_error: np.ndarray,
    slope: float,
    min_bin: int,
    max_bin: int,
) -> np.ndarray:
    """Propagate one-sigma uncertainties through the gluing weights."""
    analog = _as_1d(analog_error)
    photon = _as_1d(photon_error)
    if analog.size != photon.size:
        raise ValueError("analog_error and photon_error must have the same length.")
    min_bin = max(int(min_bin), 0)
    max_bin = min(int(max_bin), analog.size)
    if max_bin <= min_bin:
        raise ValueError("max_bin must be greater than min_bin for gluing uncertainty propagation.")

    scaled_analog_error = abs(float(slope)) * analog
    glued_error = photon.copy()
    glued_error[:min_bin] = scaled_analog_error[:min_bin]
    gluing_length = max_bin - min_bin
    analog_weights = 1.0 - np.arange(gluing_length, dtype=np.float64) / float(gluing_length)
    photon_weights = 1.0 - analog_weights
    glued_error[min_bin:max_bin] = np.sqrt(
        (analog_weights * scaled_analog_error[min_bin:max_bin]) ** 2
        + (photon_weights * photon[min_bin:max_bin]) ** 2
    )
    return glued_error


def slide_glue_signals(
    analog_sig: np.ndarray,
    pc_sig: np.ndarray,
    altitude: np.ndarray | None = None,
    window_size: int = 150,
    min_corr: float = 0.90,
    search_min_idx: int = 0,
    search_max_idx: int | None = None,
    intercept_threshold: float = 5.0,
    gaussian_threshold: float = 0.1,
    minmax_threshold: float = 0.05,
    return_diagnostics: bool = False,
    pc_saturation_mask: np.ndarray | None = None,
    max_relative_rmse: float = 0.05,
    max_relative_bias: float = 0.03,
    min_valid_fraction: float = 0.80,
) -> tuple[np.ndarray, int, float, float] | tuple[np.ndarray, int, float, float, dict[str, Any]]:
    """Glue analog and photon-counting signals into one dynamic-range profile.

    Parameters use the historical MILGRAU names for compatibility, but the
    selection is now residual-based. ``gaussian_threshold`` is accepted for API
    compatibility and is recorded in diagnostics; it is not used as a criterion.
    """
    analog = _as_1d(analog_sig)
    photon = _as_1d(pc_sig)
    if analog.size != photon.size:
        raise ValueError("analog_sig and pc_sig must have the same length.")

    invalid_mask = _as_bool_1d(pc_saturation_mask, analog.size)
    window = _window_length(window_size)
    search_stop = analog.size if search_max_idx is None else int(search_max_idx)

    selected = _select_window(
        analog=analog,
        photon=photon,
        invalid_mask=invalid_mask,
        window=window,
        correlation_threshold=float(min_corr),
        intercept_threshold=float(intercept_threshold),
        min_dynamic_range_ratio=float(minmax_threshold),
        min_idx=int(search_min_idx),
        max_idx=search_stop,
        max_relative_rmse=float(max_relative_rmse),
        max_relative_bias=float(max_relative_bias),
        min_valid_fraction=float(min_valid_fraction),
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
        intercept = float(selected["intercept"])
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
            "relative_intercept": (
                float(selected["intercept_percent"]) / 100.0
                if np.isfinite(float(selected["intercept_percent"]))
                else np.nan
            ),
            "relative_rmse": float(selected["relative_rmse"]),
            "relative_bias": float(selected["relative_bias"]),
            "dynamic_range_ratio": float(selected["dynamic_range_ratio"]),
            "valid_count": int(selected["valid_count"]),
            "slope": float(selected["slope"]),
            "intercept": float(selected["intercept"]),
            "slope_zero_intercept": np.nan,
            "shapiro_p_value": np.nan,
            "gaussian_score": np.nan,
            "gaussian_threshold_requested": float(gaussian_threshold),
            "minmax_ratio": float(selected["dynamic_range_ratio"]),
            "minmax_score": float(selected["dynamic_range_ratio"]),
            "pc_saturation_fraction_window": (
                float(np.mean(invalid_mask[min_bin:max_bin])) if split_point >= 0 and max_bin > min_bin else np.nan
            ),
        }
        if altitude is not None and split_point >= 0:
            altitude_arr = np.asarray(altitude, dtype=np.float64)
            if altitude_arr.ndim == 1 and split_point < altitude_arr.size:
                diagnostics["split_altitude"] = float(altitude_arr[split_point])
                diagnostics["min_altitude"] = float(altitude_arr[min_bin])
                diagnostics["max_altitude"] = float(altitude_arr[max_bin - 1])
        return glued_signal, split_point, float(slope), float(intercept), diagnostics

    return glued_signal, split_point, float(slope), float(intercept)
