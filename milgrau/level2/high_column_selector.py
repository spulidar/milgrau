"""Deterministic molecular-reference selection for the high column.

The selector deliberately does not invent a new molecular-purity score. It
operates only after native-grid Rayleigh minimum QA and progressive-grid path
admissibility have been evaluated by :mod:`milgrau.level2.high_column`.

The declared search ranges are tried in order. The second range is an explicit
support fallback, not a claim that lower altitude is aerosol-free.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from milgrau.level2.high_column import (
    HighColumnReferenceCatalogue,
    HighColumnReferenceCell,
)

DEFAULT_REFERENCE_SEARCH_RANGES_M: tuple[tuple[float, float], ...] = (
    (10_000.0, 15_000.0),
    (5_000.0, 20_000.0),
)


@dataclass(frozen=True, slots=True)
class PrioritizedReferenceSelection:
    """One selected reference and the ordered search range that supplied it."""

    reference: HighColumnReferenceCell
    search_min_altitude_m: float
    search_max_altitude_m: float
    search_range_index: int
    fallback_used: bool


def _validated_search_ranges(
    search_ranges_m: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Return finite increasing altitude ranges in caller-declared priority order."""
    ranges = tuple((float(lower), float(upper)) for lower, upper in search_ranges_m)
    if not ranges:
        raise ValueError("at least one reference search range is required.")
    for lower, upper in ranges:
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            raise ValueError(
                "reference search ranges must contain finite increasing altitude bounds."
            )
    return ranges


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
    min_altitude_m: float,
    max_altitude_m: float,
) -> HighColumnReferenceCell:
    """Select the minimum configured Rayleigh cost from one altitude domain.

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


def select_prioritized_high_column_reference(
    catalogue: HighColumnReferenceCatalogue,
    *,
    search_ranges_m: Sequence[tuple[float, float]] = DEFAULT_REFERENCE_SEARCH_RANGES_M,
) -> PrioritizedReferenceSelection:
    """Try explicit altitude ranges in order, then minimize Rayleigh cost.

    The first range containing a Rayleigh-accepted, path-admissible cell wins.
    Range order is policy; altitude is not added to the diagnostic score and is
    not treated as evidence that the selected cell is aerosol-free.
    """
    ranges = _validated_search_ranges(search_ranges_m)
    for range_index, (lower, upper) in enumerate(ranges):
        try:
            reference = _minimum_cost_reference(
                catalogue,
                min_altitude_m=lower,
                max_altitude_m=upper,
            )
        except ValueError:
            continue
        return PrioritizedReferenceSelection(
            reference=reference,
            search_min_altitude_m=lower,
            search_max_altitude_m=upper,
            search_range_index=range_index,
            fallback_used=range_index > 0,
        )
    raise ValueError(
        "No Rayleigh-accepted, path-admissible high-column reference cell exists "
        "inside any requested search range."
    )
