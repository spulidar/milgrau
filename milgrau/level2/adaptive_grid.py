"""Strict progressive vertical grid utilities for method-v5 R&D.

The productive Level-2 retrieval remains method v4.  This module provides the
representation layer needed to test a method-v5 high-column retrieval on a
nonuniform vertical grid without interpolation, padding or gap bridging.

Every output cell owns one contiguous, non-overlapping set of native bins.
Requested physical resolutions are converted to an integer number of native
bins without exceeding the requested width.  The actual/effective cell width
is retained explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

UncertaintyMode = Literal["independent", "fully_correlated"]


@dataclass(frozen=True, slots=True)
class ProgressiveGrid:
    """One strict nonuniform grid backed by contiguous native source bins."""

    altitude_m: np.ndarray
    source_start_index: np.ndarray
    source_stop_index: np.ndarray
    source_count: np.ndarray
    requested_resolution_m: np.ndarray
    effective_resolution_m: np.ndarray
    native_spacing_m: float

    @property
    def n_cells(self) -> int:
        """Return the number of output cells."""
        return int(self.altitude_m.size)


@dataclass(frozen=True, slots=True)
class AggregatedGridValues:
    """Values represented on a :class:`ProgressiveGrid`."""

    values: np.ndarray
    uncertainty: np.ndarray | None
    valid: np.ndarray


def _native_edges(altitude_m: np.ndarray) -> np.ndarray:
    """Return cell edges implied by strictly increasing native bin centers."""
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if altitude.ndim != 1 or altitude.size < 2:
        raise ValueError("altitude_m must be a one-dimensional array with at least two bins.")
    if np.any(~np.isfinite(altitude)) or np.any(np.diff(altitude) <= 0.0):
        raise ValueError("altitude_m must be finite and strictly increasing.")

    edges = np.empty(altitude.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (altitude[:-1] + altitude[1:])
    edges[0] = altitude[0] - 0.5 * (altitude[1] - altitude[0])
    edges[-1] = altitude[-1] + 0.5 * (altitude[-1] - altitude[-2])
    return edges


def _normalize_schedule(
    schedule: Iterable[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    rows = [(float(start), float(width)) for start, width in schedule]
    if not rows:
        raise ValueError("schedule must contain at least one (min_altitude_m, resolution_m) row.")
    starts = np.asarray([row[0] for row in rows], dtype=np.float64)
    widths = np.asarray([row[1] for row in rows], dtype=np.float64)
    if np.any(~np.isfinite(starts)) or np.any(~np.isfinite(widths)):
        raise ValueError("schedule values must be finite.")
    if np.any(widths <= 0.0):
        raise ValueError("requested resolutions must be positive.")
    if np.any(np.diff(starts) <= 0.0):
        raise ValueError("schedule minimum altitudes must be strictly increasing.")
    return starts, widths


def build_progressive_grid(
    altitude_m: np.ndarray,
    schedule: Sequence[tuple[float, float]],
) -> ProgressiveGrid:
    """Build a strict piecewise-progressive grid from native altitude bins.

    ``schedule`` contains ``(min_altitude_m, requested_resolution_m)`` rows.
    Each requested resolution is converted to the largest integer source-bin
    count whose nominal width does not exceed that request.  Output cells never
    cross a schedule transition; a short edge cell is retained rather than
    borrowing bins from the neighboring resolution band.

    The function assumes the native spacing is approximately regular, as is the
    SPU 7.5 m Level-1 grid.  Small numerical spacing jitter is tolerated, while
    materially irregular source grids are rejected because a bin-count policy
    would otherwise have ambiguous physical meaning.
    """
    altitude = np.asarray(altitude_m, dtype=np.float64)
    edges = _native_edges(altitude)
    diffs = np.diff(altitude)
    native_spacing = float(np.median(diffs))
    if not np.isfinite(native_spacing) or native_spacing <= 0.0:
        raise ValueError("native altitude spacing must be finite and positive.")
    spacing_tolerance = max(1.0e-6, native_spacing * 1.0e-4)
    if not np.all(np.abs(diffs - native_spacing) <= spacing_tolerance):
        raise ValueError(
            "build_progressive_grid requires an approximately regular native altitude grid."
        )

    starts, widths = _normalize_schedule(schedule)
    if starts[0] > altitude[0] + spacing_tolerance:
        raise ValueError("the first schedule altitude must cover the first native bin.")

    start_indices: list[int] = []
    stop_indices: list[int] = []
    requested_widths: list[float] = []
    effective_widths: list[float] = []
    output_altitudes: list[float] = []

    i = 0
    n_native = altitude.size
    while i < n_native:
        schedule_index = int(np.searchsorted(starts, altitude[i], side="right") - 1)
        schedule_index = max(schedule_index, 0)
        requested_width = float(widths[schedule_index])
        tolerance = max(1.0e-9, native_spacing * 1.0e-9)
        bins_per_cell = max(
            1,
            int(np.floor((requested_width + tolerance) / native_spacing)),
        )

        stop = min(i + bins_per_cell, n_native)
        if schedule_index + 1 < starts.size:
            transition_index = int(
                np.searchsorted(altitude, starts[schedule_index + 1], side="left")
            )
            if transition_index > i:
                stop = min(stop, transition_index)
        if stop <= i:
            stop = i + 1

        start_indices.append(i)
        stop_indices.append(stop)
        requested_widths.append(requested_width)
        effective_widths.append(float(edges[stop] - edges[i]))
        output_altitudes.append(float(np.mean(altitude[i:stop])))
        i = stop

    source_start = np.asarray(start_indices, dtype=np.int32)
    source_stop = np.asarray(stop_indices, dtype=np.int32)
    source_count = source_stop - source_start
    output_altitude = np.asarray(output_altitudes, dtype=np.float64)
    effective_resolution = np.asarray(effective_widths, dtype=np.float64)
    requested_resolution = np.asarray(requested_widths, dtype=np.float64)

    if np.any(source_count <= 0):
        raise RuntimeError("progressive-grid construction produced an empty cell.")
    if source_start[0] != 0 or source_stop[-1] != n_native:
        raise RuntimeError("progressive grid does not cover the complete native grid.")
    if np.any(source_start[1:] != source_stop[:-1]):
        raise RuntimeError("progressive grid must use every native bin exactly once.")
    if np.any(np.diff(output_altitude) <= 0.0):
        raise RuntimeError("progressive output altitude must be strictly increasing.")

    return ProgressiveGrid(
        altitude_m=output_altitude,
        source_start_index=source_start,
        source_stop_index=source_stop,
        source_count=source_count,
        requested_resolution_m=requested_resolution,
        effective_resolution_m=effective_resolution,
        native_spacing_m=native_spacing,
    )


def aggregate_to_progressive_grid(
    values: np.ndarray,
    grid: ProgressiveGrid,
    *,
    uncertainty: np.ndarray | None = None,
    uncertainty_mode: UncertaintyMode = "independent",
    require_positive: bool = False,
) -> AggregatedGridValues:
    """Strictly aggregate one native profile onto ``grid``.

    A cell is valid only when **all** source samples required by that cell are
    finite.  When ``require_positive=True`` they must also all be positive.
    If uncertainty is supplied, all contributing one-sigma values must be
    finite and nonnegative.  Therefore aggregation cannot hide or bridge a
    missing/masked native sample.

    Background-subtracted elastic RCS is a special case: a finite negative
    native sample can be a legitimate noisy measurement rather than a missing
    sample.  For RCS the intended v5 use is ``require_positive=False`` during
    aggregation, followed by the KFS physical requirement that the **aggregated
    cell estimate** itself be positive.  This is averaging at the declared
    effective resolution, not gap filling.

    ``independent`` propagates the uncertainty of the arithmetic mean as
    ``sqrt(sum(sigma_i**2)) / N``.  ``fully_correlated`` uses
    ``sum(sigma_i) / N``.  These are explicit dependence limits rather than an
    implicit covariance assumption.
    """
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    n_native = int(grid.source_stop_index[-1])
    if source.size != n_native:
        raise ValueError("values length must match the native grid used to construct grid.")
    if uncertainty_mode not in {"independent", "fully_correlated"}:
        raise ValueError("uncertainty_mode must be 'independent' or 'fully_correlated'.")

    if uncertainty is None:
        source_uncertainty = None
    else:
        source_uncertainty = np.asarray(uncertainty, dtype=np.float64)
        if source_uncertainty.shape != source.shape:
            raise ValueError("uncertainty must have the same shape as values.")

    out = np.full(grid.n_cells, np.nan, dtype=np.float64)
    out_uncertainty = (
        np.full(grid.n_cells, np.nan, dtype=np.float64)
        if source_uncertainty is not None
        else None
    )
    valid = np.zeros(grid.n_cells, dtype=bool)

    for cell_index, (start, stop) in enumerate(
        zip(grid.source_start_index, grid.source_stop_index, strict=True)
    ):
        chunk = source[int(start) : int(stop)]
        chunk_valid = np.isfinite(chunk)
        if require_positive:
            chunk_valid &= chunk > 0.0
        if not np.all(chunk_valid):
            continue

        if source_uncertainty is not None:
            sigma = source_uncertainty[int(start) : int(stop)]
            if not np.all(np.isfinite(sigma) & (sigma >= 0.0)):
                continue
        else:
            sigma = None

        out[cell_index] = float(np.mean(chunk))
        valid[cell_index] = True
        if sigma is not None and out_uncertainty is not None:
            count = float(chunk.size)
            if uncertainty_mode == "independent":
                out_uncertainty[cell_index] = float(np.sqrt(np.sum(sigma**2)) / count)
            else:
                out_uncertainty[cell_index] = float(np.sum(sigma) / count)

    return AggregatedGridValues(
        values=out,
        uncertainty=out_uncertainty,
        valid=valid,
    )
