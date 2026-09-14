"""Retrieval-input support QA shared by the public Level 2 boundary.

The legacy monolithic retrieval implementation historically required every
positive-altitude bin to be finite and positive. Level 1 bin shifts deliberately
create invalid edge bins, while the KFS inversion already treats invalid edge
bins as the end of an oriented branch. This module aligns the pre-inversion QA
with that behavior without allowing internal holes or an invalid Rayleigh
reference interval.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from milgrau.level2.contracts import RetrievalInputInvalidReason


def _invalid_gap_reason(
    values: np.ndarray,
    errors: np.ndarray,
    mask: np.ndarray,
) -> RetrievalInputInvalidReason:
    """Return the first scientific reason for an invalid internal span."""
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
    """Validate the contiguous physical support used by Rayleigh/KFS retrieval.

    The accepted domain is the one contiguous run of finite positive signal and
    finite non-negative uncertainty that contains the complete configured
    Rayleigh-reference interval. Invalid bins before or after that run are edge
    bins and may remain NaN/non-positive, as expected after Level 1 bin shifts
    or when an oriented KFS branch reaches the end of measurable support.

    A second valid run separated by an invalid bin is not treated as another
    edge: it proves an internal hole and the candidate is rejected. No values
    are filled, extrapolated or interpolated by this QA.
    """
    values = np.asarray(signal, dtype=np.float64)
    errors = np.asarray(signal_error, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if values.shape != altitude.shape or errors.shape != altitude.shape:
        raise ValueError("Signal, uncertainty, and altitude must have identical one-dimensional shapes.")
    if altitude.ndim != 1 or not np.isfinite(altitude).all() or not np.all(np.diff(altitude) > 0.0):
        raise ValueError("Retrieval altitude must be a finite, strictly increasing one-dimensional grid.")
    if not correction_valid:
        return (
            False,
            RetrievalInputInvalidReason.LEVEL1_CORRECTION_FAILED_OR_UNCONFIRMED,
            np.nan,
        )

    ref_alt_min_m = float(fit_config["ref_alt_min_m"])
    ref_alt_max_m = float(fit_config["ref_alt_max_m"])
    positive_altitude = altitude > 0.0
    reference = (altitude >= ref_alt_min_m) & (altitude <= ref_alt_max_m)
    if positive_altitude.sum() < 3 or reference.sum() < 3:
        return False, RetrievalInputInvalidReason.INSUFFICIENT_VERTICAL_COVERAGE, np.nan

    finite_signal = np.isfinite(values)
    positive_signal = finite_signal & (values > 0.0)
    valid_error = np.isfinite(errors) & (errors >= 0.0)
    usable = positive_altitude & positive_signal & valid_error

    reference_indices = np.flatnonzero(reference)
    reference_span = np.zeros_like(reference, dtype=bool)
    reference_span[reference_indices[0] : reference_indices[-1] + 1] = True
    if not np.all(usable[reference_span]):
        return False, _invalid_gap_reason(values, errors, reference_span & ~usable), np.nan

    start = int(reference_indices[0])
    stop = int(reference_indices[-1])
    while start > 0 and usable[start - 1]:
        start -= 1
    while stop + 1 < altitude.size and usable[stop + 1]:
        stop += 1

    support = np.zeros_like(positive_altitude, dtype=bool)
    support[start : stop + 1] = True
    support &= positive_altitude

    # Any usable island outside the Rayleigh-anchored run means an invalid bin
    # lies between otherwise usable samples. That is an internal hole, not an
    # edge condition that may be silently cropped.
    if np.any(usable & ~support):
        usable_indices = np.flatnonzero(usable)
        full_span = np.zeros_like(positive_altitude, dtype=bool)
        full_span[usable_indices[0] : usable_indices[-1] + 1] = True
        internal_gap = full_span & positive_altitude & ~usable
        return False, _invalid_gap_reason(values, errors, internal_gap), np.nan

    if (
        support.sum() < 3
        or float(altitude[support][0]) > ref_alt_min_m
        or float(altitude[support][-1]) < ref_alt_max_m
    ):
        return False, RetrievalInputInvalidReason.INSUFFICIENT_VERTICAL_COVERAGE, np.nan

    if require_saturation_diagnostic and saturation_fraction is None:
        return False, RetrievalInputInvalidReason.SATURATION_DIAGNOSTIC_MISSING, np.nan
    if saturation_fraction is not None:
        saturation = np.asarray(saturation_fraction, dtype=np.float64)
        if saturation.shape != altitude.shape:
            raise ValueError("Photon-counting saturation diagnostics must match the altitude grid.")
        if np.any(~np.isfinite(saturation[support])) or np.any(saturation[support] > 0.0):
            return False, RetrievalInputInvalidReason.PHOTON_COUNTING_SATURATED, np.nan

    snr_bins = support & (errors > 0.0)
    if not snr_bins.any():
        return False, RetrievalInputInvalidReason.SNR_UNAVAILABLE, np.nan
    snr_median = float(np.nanmedian(np.abs(values[snr_bins]) / errors[snr_bins]))
    if not np.isfinite(snr_median):
        return False, RetrievalInputInvalidReason.SNR_UNAVAILABLE, np.nan
    return True, RetrievalInputInvalidReason.VALID, snr_median
