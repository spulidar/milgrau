"""Background-offset bracketing for high-column range-corrected signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BackgroundOffsetScale:
    """Robust raw-equivalent residual scale from a declared altitude band."""

    background_min_altitude_m: float
    background_max_altitude_m: float
    valid_bins: int
    raw_equivalent_median: float
    raw_equivalent_mad_sigma: float


def estimate_background_offset_scale(
    range_corrected_signal: np.ndarray,
    altitude_m: np.ndarray,
    *,
    background_min_altitude_m: float = 29_000.0,
    background_max_altitude_m: float = 30_000.0,
) -> BackgroundOffsetScale:
    """Estimate a robust residual scale after undoing the ``range^2`` factor.

    The returned MAD scale is a sensitivity step, not a standard uncertainty of
    the background estimator. Correlated detector/background structure can make
    the effective offset larger than an independent-bin standard error.
    """
    signal = np.asarray(range_corrected_signal, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if signal.ndim != 1 or altitude.ndim != 1 or signal.shape != altitude.shape:
        raise ValueError("background sensitivity inputs must be matching 1D arrays.")
    if np.any(~np.isfinite(altitude)) or np.any(altitude <= 0.0):
        raise ValueError("altitude_m must be finite and positive.")
    lower = float(background_min_altitude_m)
    upper = float(background_max_altitude_m)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("background altitude bounds must be finite and increasing.")
    inside = (
        (altitude >= lower)
        & (altitude <= upper)
        & np.isfinite(signal)
    )
    n_valid = int(np.count_nonzero(inside))
    if n_valid < 5:
        raise ValueError("background band must contain at least five finite bins.")
    raw_equivalent = signal[inside] / altitude[inside] ** 2
    median = float(np.median(raw_equivalent))
    mad_sigma = float(1.4826 * np.median(np.abs(raw_equivalent - median)))
    if not np.isfinite(mad_sigma) or mad_sigma <= 0.0:
        raise ValueError("background band does not contain a positive residual scale.")
    return BackgroundOffsetScale(
        background_min_altitude_m=lower,
        background_max_altitude_m=upper,
        valid_bins=n_valid,
        raw_equivalent_median=median,
        raw_equivalent_mad_sigma=mad_sigma,
    )


def perturb_rcs_by_raw_background_offset(
    range_corrected_signal: np.ndarray,
    altitude_m: np.ndarray,
    raw_equivalent_offset: float,
) -> np.ndarray:
    """Apply one additive raw-signal offset before the implicit range-square step."""
    signal = np.asarray(range_corrected_signal, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if signal.ndim != 1 or altitude.ndim != 1 or signal.shape != altitude.shape:
        raise ValueError("background perturbation inputs must be matching 1D arrays.")
    offset = float(raw_equivalent_offset)
    if not np.isfinite(offset):
        raise ValueError("raw_equivalent_offset must be finite.")
    return np.asarray(signal + offset * altitude**2, dtype=np.float64)
