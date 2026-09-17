"""Deterministic reference selection for the method-v5 high-column R&D path.

The selector deliberately does not invent a new molecular-purity score.  It
operates only after native-grid Rayleigh minimum QA and progressive-grid path
admissibility have been evaluated by :mod:`milgrau.level2.high_column_rnd`.

The first prototype search domain is 10--25 km.  The 10 km lower bound is a
product-design/R&D search-domain choice, not a claim that aerosol is absent
above that altitude.  Controlled broad aerosol layers can defeat any fixed
altitude floor, so residual boundary aerosol remains an explicit systematic
sensitivity dimension in method v5.
"""

from __future__ import annotations

import math

from milgrau.level2.high_column_rnd import (
    HighColumnReferenceCatalogue,
    HighColumnReferenceCell,
)

V5_REFERENCE_SEARCH_MIN_M = 10_000.0
V5_REFERENCE_SEARCH_MAX_M = 25_000.0


def select_minimum_cost_high_column_reference(
    catalogue: HighColumnReferenceCatalogue,
    *,
    min_altitude_m: float = V5_REFERENCE_SEARCH_MIN_M,
    max_altitude_m: float = V5_REFERENCE_SEARCH_MAX_M,
) -> HighColumnReferenceCell:
    """Select the minimum historical Rayleigh cost from admissible high cells.

    Selection is intentionally staged rather than expressed as one composite
    score:

    1. the candidate must already pass Rayleigh minimum QA;
    2. its progressive-grid cell must lie on the continuous nominal KFS path;
    3. its altitude must lie inside the caller-declared high-column search
       domain;
    4. among those survivors, minimize the existing Rayleigh diagnostic cost
       ``relative_slope + relative_variance``;
    5. exact cost ties choose the lower altitude, preserving the conservative
       historical lower-index tie direction rather than rewarding altitude.

    The returned reference is still conditional on the declared boundary model;
    this function does not infer or validate ``beta_aer(ref) = 0``.
    """
    lower = float(min_altitude_m)
    upper = float(max_altitude_m)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("high-column selector altitude bounds must be finite and increasing.")

    eligible = tuple(
        cell
        for cell in catalogue.accepted_and_admissible
        if lower <= cell.altitude_m <= upper
    )
    if not eligible:
        raise ValueError(
            "No Rayleigh-accepted, path-admissible high-column reference cell exists "
            "inside the requested selector altitude domain."
        )

    return min(
        eligible,
        key=lambda cell: (
            float(cell.native_rayleigh_candidate.diagnostic_cost),
            float(cell.altitude_m),
            int(cell.cell_index),
        ),
    )
