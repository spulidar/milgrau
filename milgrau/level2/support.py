"""Altitude-resolved scientific support semantics for Level 2 retrievals.

This module does not publish new NetCDF variables by itself.  It defines the
support contract that must be validated before product-schema exposure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BackwardRetrievalSupport:
    """Contiguous supported domain for one accepted backward retrieval.

    ``flag`` is true only on the contiguous interval that reaches the accepted
    upper inversion/reference boundary.  A missing/invalid bin inside that
    interval breaks the backward integration path; valid-looking values below
    such a gap are therefore not promoted to scientifically supported output.
    """

    flag: np.ndarray
    bottom_altitude_m: float
    top_altitude_m: float


def backward_retrieval_support(
    altitude_m: np.ndarray,
    value: np.ndarray,
    uncertainty: np.ndarray,
    *,
    upper_index: int,
    instrument_valid: np.ndarray | None = None,
) -> BackwardRetrievalSupport:
    """Return scientifically supported bins for a backward retrieval.

    Support requires, at every retained bin:

    - altitude on a finite strictly increasing one-dimensional grid;
    - finite retrieved value;
    - finite, non-negative reported uncertainty;
    - validated instrument support when an ``instrument_valid`` mask is given;
    - membership in the contiguous valid path ending at ``upper_index``.

    Bins above ``upper_index`` are outside the productive backward inversion
    even when numerically finite.  An invalid bin below ``upper_index`` cuts
    support there and also invalidates every lower bin for that branch because
    the backward integral cannot jump across an unsupported gap.

    ``instrument_valid=None`` means that this function has not been asked to
    impose an additional instrument boundary; it must not be interpreted as a
    claim that overlap or any other instrument limitation is characterized.
    """
    altitude = np.asarray(altitude_m, dtype=np.float64)
    retrieved = np.asarray(value, dtype=np.float64)
    sigma = np.asarray(uncertainty, dtype=np.float64)

    if altitude.ndim != 1 or retrieved.ndim != 1 or sigma.ndim != 1:
        raise ValueError("altitude, value, and uncertainty must be one-dimensional.")
    if not (altitude.shape == retrieved.shape == sigma.shape):
        raise ValueError("altitude, value, and uncertainty must have identical shapes.")
    if altitude.size == 0:
        raise ValueError("support cannot be evaluated on an empty altitude grid.")
    if not np.all(np.isfinite(altitude)) or not np.all(np.diff(altitude) > 0.0):
        raise ValueError("altitude must be finite and strictly increasing.")
    if isinstance(upper_index, bool) or not isinstance(upper_index, (int, np.integer)):
        raise ValueError("upper_index must be an integer index.")
    upper = int(upper_index)
    if upper < 0 or upper >= altitude.size:
        raise ValueError("upper_index is outside the altitude grid.")

    if instrument_valid is None:
        instrument = np.ones(altitude.shape, dtype=bool)
    else:
        instrument = np.asarray(instrument_valid)
        if instrument.ndim != 1 or instrument.shape != altitude.shape:
            raise ValueError("instrument_valid must be one-dimensional and match altitude.")
        if instrument.dtype.kind != "b":
            raise ValueError("instrument_valid must be a boolean mask.")
        instrument = instrument.astype(bool, copy=False)

    candidate = np.isfinite(retrieved) & np.isfinite(sigma) & (sigma >= 0.0) & instrument
    flag = np.zeros(altitude.shape, dtype=bool)

    if not candidate[upper]:
        return BackwardRetrievalSupport(
            flag=flag,
            bottom_altitude_m=float("nan"),
            top_altitude_m=float("nan"),
        )

    bottom = upper
    while bottom > 0 and candidate[bottom - 1]:
        bottom -= 1

    flag[bottom : upper + 1] = True
    return BackwardRetrievalSupport(
        flag=flag,
        bottom_altitude_m=float(altitude[bottom]),
        top_altitude_m=float(altitude[upper]),
    )
