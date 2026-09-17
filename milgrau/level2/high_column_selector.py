"""Deterministic reference selection for the method-v5 high-column R&D path.

The selector deliberately does not invent a new molecular-purity score. It
operates only after native-grid Rayleigh minimum QA and progressive-grid path
admissibility have been evaluated by :mod:`milgrau.level2.high_column_rnd`.

Method v5 prefers genuinely supported high references but may fall back through
explicit lower search tiers when the measurement does not support the preferred
high-column domain. A lower tier is therefore an auditable support fallback,
not a claim that lower altitude is aerosol-free.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from milgrau.level2.high_column_rnd import (
    HighColumnReferenceCatalogue,
    HighColumnReferenceCell,
)

V5_REFERENCE_SEARCH_MIN_M = 10_000.0
V5_REFERENCE_SEARCH_MAX_M = 25_000.0
V5_REFERENCE_SEARCH_TIER_MINIMA_M: tuple[float, ...] = (
    10_000.0,
    9_000.0,
    8_000.0,
    6_000.0,
)


@dataclass(frozen=True, slots=True)
class TieredHighColumnReferenceSelection:
    """One selected reference together with its explicit fallback tier."""

    reference: HighColumnReferenceCell
    tier_min_altitude_m: float
    tier_index: int
    fallback_used: bool


def _validated_tier_minima(
    tier_min_altitudes_m: Sequence[float],
    *,
    max_altitude_m: float,
) -> tuple[float, ...]:
    """Return finite strictly descending tier minima below the upper bound."""
    tiers = tuple(float(value) for value in tier_min_altitudes_m)
    upper = float(max_altitude_m)
    if not math.isfinite(upper):
        raise ValueError("high-column selector maximum altitude must be finite.")
    if not tiers:
        raise ValueError("at least one high-column reference tier is required.")
    if any(not math.isfinite(value) for value in tiers):
        raise ValueError("high-column reference tier minima must be finite.")
    if any(value >= upper for value in tiers):
        raise ValueError("every high-column reference tier minimum must be below max altitude.")
    if any(later >= earlier for earlier, later in zip(tiers, tiers[1:])):
        raise ValueError("high-column reference tier minima must be strictly descending.")
    return tiers


def _minimum_cost_reference(
    catalogue: HighColumnReferenceCatalogue,
    *,
    min_altitude_m: float,
    max_altitude_m: float,
) -> HighColumnReferenceCell:
    eligible = tuple(
        cell
        for cell in catalogue.accepted_and_admissible
        if float(min_altitude_m) <= cell.altitude_m <= float(max_altitude_m)
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


def select_minimum_cost_high_column_reference(
    catalogue: HighColumnReferenceCatalogue,
    *,
    min_altitude_m: float = V5_REFERENCE_SEARCH_MIN_M,
    max_altitude_m: float = V5_REFERENCE_SEARCH_MAX_M,
) -> HighColumnReferenceCell:
    """Select the minimum historical Rayleigh cost from one altitude domain.

    Selection is intentionally staged rather than expressed as one composite
    score: Rayleigh QA and path admissibility are hard prerequisites, then the
    existing diagnostic cost ``relative_slope + relative_variance`` is minimized.
    Exact ties prefer lower altitude / lower grid index. Altitude is not itself
    a ranking reward.
    """
    lower = float(min_altitude_m)
    upper = float(max_altitude_m)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("high-column selector altitude bounds must be finite and increasing.")
    return _minimum_cost_reference(
        catalogue,
        min_altitude_m=lower,
        max_altitude_m=upper,
    )


def select_tiered_high_column_reference(
    catalogue: HighColumnReferenceCatalogue,
    *,
    tier_min_altitudes_m: Sequence[float] = V5_REFERENCE_SEARCH_TIER_MINIMA_M,
    max_altitude_m: float = V5_REFERENCE_SEARCH_MAX_M,
) -> TieredHighColumnReferenceSelection:
    """Select from the highest supported search tier, then minimize Rayleigh cost.

    Tiers are attempted in caller order and must be strictly descending. The
    first tier containing at least one Rayleigh-accepted, path-admissible cell is
    used. Only then is the historical Rayleigh diagnostic cost minimized within
    that tier. This prevents a lower-cost low-altitude candidate from displacing
    an available higher-support solution while still allowing explicit fallback
    when the measurement cannot support the preferred domain.
    """
    tiers = _validated_tier_minima(
        tier_min_altitudes_m,
        max_altitude_m=float(max_altitude_m),
    )
    for tier_index, lower in enumerate(tiers):
        try:
            reference = _minimum_cost_reference(
                catalogue,
                min_altitude_m=lower,
                max_altitude_m=float(max_altitude_m),
            )
        except ValueError:
            continue
        return TieredHighColumnReferenceSelection(
            reference=reference,
            tier_min_altitude_m=float(lower),
            tier_index=int(tier_index),
            fallback_used=bool(tier_index > 0),
        )
    raise ValueError(
        "No Rayleigh-accepted, path-admissible high-column reference cell exists "
        "inside any requested selector tier."
    )
