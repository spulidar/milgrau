"""Executable high-column elastic retrieval layer for method-v5 R&D.

Nothing in this module is wired into productive method v4. It keeps three
scientific roles separate:

1. native-grid Rayleigh-window QA diagnoses whether a physical neighborhood is
   compatible with the molecular signal shape;
2. the progressive grid defines the effective vertical representation used by
   the high-column boundary and KFS integration;
3. caller-declared residual-aerosol fractions ``f`` remain outer systematic
   sensitivity scenarios around an inner random Monte Carlo.

The module catalogues admissible reference cells and can execute a retrieval for
an explicitly chosen cell. Automatic ranking/fallback policy lives in
:mod:`milgrau.level2.high_column_selector`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from milgrau.level2.adaptive_grid import (
    AggregatedGridValues,
    ProgressiveGrid,
    UncertaintyMode,
    aggregate_to_progressive_grid,
    build_progressive_grid,
)
from milgrau.level2.boundary_sensitivity import (
    BoundaryFractionMonteCarloSensitivity,
    boundary_fraction_monte_carlo_sensitivity,
)
from milgrau.level2.rayleigh_candidates import (
    RayleighReferenceCandidate,
    evaluate_rayleigh_candidate,
)


V5_PROGRESSIVE_GRID_SCHEDULE: tuple[tuple[float, float], ...] = (
    (0.0, 7.5),
    (6_000.0, 15.0),
    (10_000.0, 30.0),
    (15_000.0, 60.0),
    (20_000.0, 60.0),
    (25_000.0, 100.0),
)


@dataclass(frozen=True, slots=True)
class PreparedHighColumnProfile:
    """Measured and molecular state represented on one progressive grid."""

    grid: ProgressiveGrid
    range_corrected_signal: np.ndarray
    range_corrected_signal_error: np.ndarray
    molecular_backscatter: np.ndarray
    cell_source_supported: np.ndarray
    kfs_cell_usable: np.ndarray
    uncertainty_mode: UncertaintyMode


@dataclass(frozen=True, slots=True)
class HighColumnReferenceCell:
    """One progressive-grid reference cell with native Rayleigh diagnostics."""

    cell_index: int
    altitude_m: float
    source_start_index: int
    source_stop_index: int
    source_count: int
    effective_resolution_m: float
    native_rayleigh_candidate: RayleighReferenceCandidate
    nominal_path_admissible: bool

    @property
    def accepted(self) -> bool:
        """Return current Rayleigh minimum-QA state for this cell neighborhood."""
        return bool(self.native_rayleigh_candidate.accepted)


@dataclass(frozen=True, slots=True)
class HighColumnReferenceCatalogue:
    """Auditable method-v5 R&D reference-cell catalogue without final ranking."""

    cells: tuple[HighColumnReferenceCell, ...]
    path_start_altitude_m: float
    search_min_altitude_m: float
    search_max_altitude_m: float
    rayleigh_window_m: float

    @property
    def accepted_and_admissible(self) -> tuple[HighColumnReferenceCell, ...]:
        """Return cells passing Rayleigh QA and nominal path admissibility."""
        return tuple(
            cell for cell in self.cells if cell.accepted and cell.nominal_path_admissible
        )


def prepare_high_column_profile(
    *,
    range_corrected_signal: np.ndarray,
    range_corrected_signal_error: np.ndarray,
    molecular_backscatter: np.ndarray,
    altitude_m: np.ndarray,
    uncertainty_mode: UncertaintyMode,
    schedule: Sequence[tuple[float, float]] = V5_PROGRESSIVE_GRID_SCHEDULE,
) -> PreparedHighColumnProfile:
    """Represent one native profile on the strict method-v5 progressive grid.

    Finite signed background-subtracted RCS samples are averaged. Missing source
    samples invalidate only their progressive cell. A cell becomes usable by KFS
    only when its aggregated RCS and molecular backscatter are both finite and
    positive.
    """
    altitude = np.asarray(altitude_m, dtype=np.float64)
    signal = np.asarray(range_corrected_signal, dtype=np.float64)
    signal_error = np.asarray(range_corrected_signal_error, dtype=np.float64)
    molecular = np.asarray(molecular_backscatter, dtype=np.float64)
    if not (
        altitude.ndim == signal.ndim == signal_error.ndim == molecular.ndim == 1
    ):
        raise ValueError("all high-column profile inputs must be one-dimensional.")
    if not (altitude.shape == signal.shape == signal_error.shape == molecular.shape):
        raise ValueError("all high-column profile inputs must have identical shapes.")

    grid = build_progressive_grid(altitude, schedule)
    aggregated_signal: AggregatedGridValues = aggregate_to_progressive_grid(
        signal,
        grid,
        uncertainty=signal_error,
        uncertainty_mode=uncertainty_mode,
        require_positive=False,
    )
    aggregated_molecular = aggregate_to_progressive_grid(
        molecular,
        grid,
        require_positive=True,
    )
    if aggregated_signal.uncertainty is None:
        raise RuntimeError("signal uncertainty unexpectedly missing after aggregation.")

    source_supported = aggregated_signal.valid & aggregated_molecular.valid
    kfs_usable = (
        source_supported
        & np.isfinite(aggregated_signal.values)
        & (aggregated_signal.values > 0.0)
        & np.isfinite(aggregated_molecular.values)
        & (aggregated_molecular.values > 0.0)
        & np.isfinite(aggregated_signal.uncertainty)
        & (aggregated_signal.uncertainty >= 0.0)
    )
    return PreparedHighColumnProfile(
        grid=grid,
        range_corrected_signal=np.asarray(aggregated_signal.values, dtype=np.float64),
        range_corrected_signal_error=np.asarray(
            aggregated_signal.uncertainty, dtype=np.float64
        ),
        molecular_backscatter=np.asarray(aggregated_molecular.values, dtype=np.float64),
        cell_source_supported=np.asarray(source_supported, dtype=bool),
        kfs_cell_usable=np.asarray(kfs_usable, dtype=bool),
        uncertainty_mode=uncertainty_mode,
    )


def contiguous_usable_top_index(
    prepared: PreparedHighColumnProfile,
    *,
    path_start_altitude_m: float = 600.0,
) -> int | None:
    """Return the last continuously KFS-usable cell above ``path_start_altitude_m``.

    The first unusable progressive cell terminates the nominal path. No later
    recovery is interpreted as continuous support.
    """
    altitude = prepared.grid.altitude_m
    start = int(np.searchsorted(altitude, float(path_start_altitude_m), side="left"))
    if start >= altitude.size or not prepared.kfs_cell_usable[start]:
        return None
    top = start
    for index in range(start + 1, altitude.size):
        if not prepared.kfs_cell_usable[index]:
            break
        top = index
    return int(top)


def _native_window_bins(altitude_m: np.ndarray, window_m: float) -> int:
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if altitude.ndim != 1 or altitude.size < 3:
        raise ValueError("altitude_m must contain at least three native bins.")
    spacing = float(np.median(np.diff(altitude)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("native altitude spacing must be finite and positive.")
    width = float(window_m)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("rayleigh_window_m must be finite and positive.")
    return max(3, int(round(width / spacing)))


def catalogue_high_column_reference_cells(
    *,
    prepared: PreparedHighColumnProfile,
    native_range_corrected_signal: np.ndarray,
    native_range_corrected_signal_error: np.ndarray | None,
    native_simulated_molecular_signal: np.ndarray,
    native_altitude_m: np.ndarray,
    search_min_altitude_m: float,
    search_max_altitude_m: float,
    rayleigh_window_m: float,
    max_relative_slope: float,
    max_relative_variance: float,
    min_valid_fraction: float,
    path_start_altitude_m: float = 600.0,
) -> HighColumnReferenceCatalogue:
    """Diagnose progressive boundary cells using native-grid Rayleigh windows.

    Rayleigh QA remains on the native grid so a 1 km window retains the same
    physical/sample meaning as method v4. The search bounds apply to the
    **reference-cell center**, not to every sample in the diagnostic window.
    This avoids silently shifting a declared 10 km floor upward by half a
    Rayleigh window. The diagnostic window itself only has to fit inside the
    measured native altitude domain.

    The progressive cell is only the numerical boundary representation. This
    function does not rank accepted cells or assert molecular purity.
    """
    native_signal = np.asarray(native_range_corrected_signal, dtype=np.float64)
    native_molecular = np.asarray(native_simulated_molecular_signal, dtype=np.float64)
    native_altitude = np.asarray(native_altitude_m, dtype=np.float64)
    if not (
        native_signal.ndim == native_molecular.ndim == native_altitude.ndim == 1
    ):
        raise ValueError("native Rayleigh inputs must be one-dimensional.")
    if not (native_signal.shape == native_molecular.shape == native_altitude.shape):
        raise ValueError("native Rayleigh inputs must have identical shapes.")
    if native_range_corrected_signal_error is None:
        native_error = None
    else:
        native_error = np.asarray(native_range_corrected_signal_error, dtype=np.float64)
        if native_error.shape != native_signal.shape:
            raise ValueError("native_range_corrected_signal_error must match native signal.")

    search_min = float(search_min_altitude_m)
    search_max = float(search_max_altitude_m)
    if not np.isfinite(search_min) or not np.isfinite(search_max) or search_max <= search_min:
        raise ValueError("reference search bounds must be finite and increasing.")

    window_bins = _native_window_bins(native_altitude, rayleigh_window_m)
    half = max(window_bins // 2, 1)
    contiguous_top = contiguous_usable_top_index(
        prepared,
        path_start_altitude_m=path_start_altitude_m,
    )
    cells: list[HighColumnReferenceCell] = []

    for cell_index, cell_altitude in enumerate(prepared.grid.altitude_m):
        if cell_altitude < search_min or cell_altitude > search_max:
            continue
        native_center = int(np.argmin(np.abs(native_altitude - cell_altitude)))
        start = native_center - half
        stop = start + window_bins
        if start < 0 or stop > native_altitude.size:
            continue

        candidate = evaluate_rayleigh_candidate(
            native_signal,
            native_molecular,
            native_altitude,
            center_index=native_center,
            window_bins=window_bins,
            max_relative_slope=float(max_relative_slope),
            max_relative_variance=float(max_relative_variance),
            min_valid_fraction=float(min_valid_fraction),
            measured_signal_error=native_error,
        )
        cells.append(
            HighColumnReferenceCell(
                cell_index=int(cell_index),
                altitude_m=float(cell_altitude),
                source_start_index=int(prepared.grid.source_start_index[cell_index]),
                source_stop_index=int(prepared.grid.source_stop_index[cell_index]),
                source_count=int(prepared.grid.source_count[cell_index]),
                effective_resolution_m=float(
                    prepared.grid.effective_resolution_m[cell_index]
                ),
                native_rayleigh_candidate=candidate,
                nominal_path_admissible=bool(
                    contiguous_top is not None and cell_index <= contiguous_top
                ),
            )
        )

    return HighColumnReferenceCatalogue(
        cells=tuple(cells),
        path_start_altitude_m=float(path_start_altitude_m),
        search_min_altitude_m=search_min,
        search_max_altitude_m=search_max,
        rayleigh_window_m=float(rayleigh_window_m),
    )


def run_high_column_reference_cell_monte_carlo(
    *,
    prepared: PreparedHighColumnProfile,
    reference_cell_index: int,
    aerosol_lidar_ratio_sr: float,
    aerosol_lidar_ratio_std_sr: float,
    residual_fractions: np.ndarray | list[float] | tuple[float, ...],
    n_iterations: int,
    beta_ref_relative_std: float,
    min_lidar_ratio_sr: float,
    allow_negative_aerosol: bool,
    seed: int | None,
) -> BoundaryFractionMonteCarloSensitivity:
    """Execute nested boundary-scenario MC for one explicitly chosen v5 cell.

    The caller, not this function, owns the experimental reference-cell
    selection policy. The chosen cell must be source-supported, KFS-usable and
    lie on the continuous nominal path from the lower-column start.
    """
    ref_idx = int(reference_cell_index)
    if ref_idx < 0 or ref_idx >= prepared.grid.n_cells:
        raise ValueError("reference_cell_index must point inside the progressive grid.")
    if not prepared.kfs_cell_usable[ref_idx]:
        raise ValueError("chosen high-column reference cell is not KFS-usable.")
    top = contiguous_usable_top_index(prepared)
    if top is None or ref_idx > top:
        raise ValueError("chosen high-column reference cell is above a nominal path gap.")

    return boundary_fraction_monte_carlo_sensitivity(
        rcs=prepared.range_corrected_signal,
        rcs_error=prepared.range_corrected_signal_error,
        altitude_m=prepared.grid.altitude_m,
        beta_mol=prepared.molecular_backscatter,
        reference_index=ref_idx,
        aerosol_lidar_ratio_sr=float(aerosol_lidar_ratio_sr),
        aerosol_lidar_ratio_std_sr=float(aerosol_lidar_ratio_std_sr),
        residual_fractions=residual_fractions,
        n_iterations=int(n_iterations),
        beta_ref_relative_std=float(beta_ref_relative_std),
        min_lidar_ratio_sr=float(min_lidar_ratio_sr),
        allow_negative_aerosol=bool(allow_negative_aerosol),
        seed=seed,
    )
