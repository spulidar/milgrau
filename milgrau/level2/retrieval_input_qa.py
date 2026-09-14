"""Retrieval-input QA for an automatic Rayleigh-reference search.

The Level 2 molecular-fit altitude bounds define a *search interval*, not a
requirement that every bin across that full interval be positive. Background-
subtracted lidar signals may legitimately become non-positive in the far-range
noise tail. Productive QA therefore asks whether at least one configured
Rayleigh-sized window inside the search interval contains enough scientifically
usable samples. The downstream Rayleigh selector then evaluates molecular shape
(slope/variance) and chooses the best reference window.

No signal values are filled, clipped, extrapolated or interpolated here.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from milgrau.level2.contracts import RetrievalInputInvalidReason


def _invalid_window_reason(
    values: np.ndarray,
    errors: np.ndarray,
    mask: np.ndarray,
    saturation: np.ndarray | None,
) -> RetrievalInputInvalidReason:
    """Return the dominant scientific reason a candidate window is unusable."""
    if saturation is not None and np.any(
        mask & (~np.isfinite(saturation) | (saturation > 0.0))
    ):
        return RetrievalInputInvalidReason.PHOTON_COUNTING_SATURATED
    if np.any(mask & ~np.isfinite(values)):
        return RetrievalInputInvalidReason.NONFINITE_SIGNAL
    if np.any(mask & np.isfinite(values) & (values <= 0.0)):
        return RetrievalInputInvalidReason.NONPOSITIVE_SIGNAL
    if np.any(mask & (~np.isfinite(errors) | (errors < 0.0))):
        return RetrievalInputInvalidReason.INVALID_UNCERTAINTY
    return RetrievalInputInvalidReason.INSUFFICIENT_VERTICAL_COVERAGE


def evaluate_retrieval_input_supported_domain(
    signal: np.ndarray,
    signal_error: np.ndarray,
    altitude_m: np.ndarray,
    fit_config: Mapping[str, Any],
    *,
    correction_valid: bool,
    saturation_fraction: np.ndarray | None = None,
    require_saturation_diagnostic: bool = False,
) -> tuple[bool, RetrievalInputInvalidReason, float]:
    """Require at least one viable Rayleigh-sized window inside the search band.

    ``ref_alt_min_m``/``ref_alt_max_m`` are search bounds. ``ref_window_bins`` is
    the tested window width and ``min_valid_fraction`` is the minimum fraction
    of finite positive samples with finite non-negative uncertainty. Saturation
    diagnostics, when required, participate in the same candidate-window mask.

    This pre-QA intentionally does not impose a hard SNR threshold. SNR is
    exposed as a diagnostic while the instrument-specific threshold remains to
    be characterized. Molecular-shape acceptance is performed downstream.
    """
    values = np.asarray(signal, dtype=np.float64)
    errors = np.asarray(signal_error, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if values.shape != altitude.shape or errors.shape != altitude.shape:
        raise ValueError(
            "Signal, uncertainty, and altitude must have identical one-dimensional shapes."
        )
    if (
        altitude.ndim != 1
        or not np.isfinite(altitude).all()
        or not np.all(np.diff(altitude) > 0.0)
    ):
        raise ValueError(
            "Retrieval altitude must be a finite, strictly increasing one-dimensional grid."
        )
    if not correction_valid:
        return (
            False,
            RetrievalInputInvalidReason.LEVEL1_CORRECTION_FAILED_OR_UNCONFIRMED,
            np.nan,
        )

    ref_alt_min_m = float(fit_config["ref_alt_min_m"])
    ref_alt_max_m = float(fit_config["ref_alt_max_m"])
    window_size = int(fit_config["ref_window_bins"])
    min_valid_fraction = float(fit_config["min_valid_fraction"])
    if window_size < 3:
        raise ValueError("Rayleigh reference window must contain at least three bins.")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("Rayleigh minimum valid fraction must be between 0 and 1.")

    search = (
        (altitude > 0.0)
        & (altitude >= ref_alt_min_m)
        & (altitude <= ref_alt_max_m)
    )
    search_indices = np.flatnonzero(search)
    if search_indices.size < window_size:
        return (
            False,
            RetrievalInputInvalidReason.INSUFFICIENT_VERTICAL_COVERAGE,
            np.nan,
        )

    if require_saturation_diagnostic and saturation_fraction is None:
        return (
            False,
            RetrievalInputInvalidReason.SATURATION_DIAGNOSTIC_MISSING,
            np.nan,
        )
    saturation: np.ndarray | None = None
    if saturation_fraction is not None:
        saturation = np.asarray(saturation_fraction, dtype=np.float64)
        if saturation.shape != altitude.shape:
            raise ValueError(
                "Photon-counting saturation diagnostics must match the altitude grid."
            )

    usable = (
        (altitude > 0.0)
        & np.isfinite(values)
        & (values > 0.0)
        & np.isfinite(errors)
        & (errors >= 0.0)
    )
    if saturation is not None:
        usable &= np.isfinite(saturation) & (saturation <= 0.0)

    best_fraction = -1.0
    best_start = -1
    best_stop = -1
    last_search_index = int(search_indices[-1])
    for offset in range(search_indices.size - window_size + 1):
        start = int(search_indices[offset])
        stop = start + window_size
        if stop - 1 > last_search_index:
            continue
        window = np.zeros_like(search, dtype=bool)
        window[start:stop] = True
        if not np.all(search[window]):
            continue
        valid_fraction = float(np.mean(usable[window]))
        if valid_fraction > best_fraction:
            best_fraction = valid_fraction
            best_start = start
            best_stop = stop

    if best_start < 0:
        return (
            False,
            RetrievalInputInvalidReason.INSUFFICIENT_VERTICAL_COVERAGE,
            np.nan,
        )

    best_window = np.zeros_like(search, dtype=bool)
    best_window[best_start:best_stop] = True
    if best_fraction < min_valid_fraction:
        invalid = best_window & ~usable
        return (
            False,
            _invalid_window_reason(values, errors, invalid, saturation),
            np.nan,
        )

    snr_bins = best_window & usable & (errors > 0.0)
    if not snr_bins.any():
        return False, RetrievalInputInvalidReason.SNR_UNAVAILABLE, np.nan
    snr_median = float(np.nanmedian(np.abs(values[snr_bins]) / errors[snr_bins]))
    if not np.isfinite(snr_median):
        return False, RetrievalInputInvalidReason.SNR_UNAVAILABLE, np.nan
    return True, RetrievalInputInvalidReason.VALID, snr_median
