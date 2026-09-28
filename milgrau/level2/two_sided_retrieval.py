"""End-to-end two-sided elastic retrieval for one profile.

The retrieval combines progressive vertical representation, native-grid
Rayleigh QA, prioritized molecular-reference ranges and a selection-aware
Monte Carlo ensemble.

No Monte-Carlo valid-fraction cutoff is applied here. The returned diagnostic
fractions are evidence to be interpreted by validation/coverage studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from milgrau.level2.adaptive_grid import UncertaintyMode
from milgrau.level2.high_column import (
    HighColumnReferenceCatalogue,
    HighColumnReferenceCell,
    PreparedHighColumnProfile,
    DEFAULT_PROGRESSIVE_GRID_SCHEDULE,
    catalogue_high_column_reference_cells,
    prepare_high_column_profile,
)
from milgrau.level2.high_column_selector import (
    DEFAULT_REFERENCE_SEARCH_RANGES_M,
    select_prioritized_high_column_reference,
)
from milgrau.level2.kfs import IntegrationMode
from milgrau.level2.rayleigh_candidates import fit_rayleigh_background
from milgrau.level2.uncertainty_mc import (
    SelectionAwareMonteCarloResult,
    run_selection_aware_monte_carlo,
)


@dataclass(frozen=True, slots=True)
class TwoSidedRetrievalResult:
    """Complete auditable state of one two-sided retrieval."""

    prepared: PreparedHighColumnProfile
    reference_catalogue: HighColumnReferenceCatalogue
    selected_reference: HighColumnReferenceCell
    selected_reference_search_min_altitude_m: float
    selected_reference_search_max_altitude_m: float
    selected_reference_search_range_index: int
    selected_reference_fallback_used: bool
    reference_search_ranges_m: tuple[tuple[float, float], ...]
    monte_carlo: SelectionAwareMonteCarloResult
    selector_name: str
    search_min_altitude_m: float
    search_max_altitude_m: float
    rayleigh_window_m: float
    background_fit_min_altitude_m: float
    background_offset: float
    background_offset_standard_error: float
    calibration_background_correlation: float


def retrieve_two_sided_profile(
    *,
    range_corrected_signal: np.ndarray,
    range_corrected_signal_error: np.ndarray,
    molecular_backscatter: np.ndarray,
    simulated_molecular_range_corrected_signal: np.ndarray,
    altitude_m: np.ndarray,
    aerosol_lidar_ratio_sr: float,
    aerosol_lidar_ratio_std_sr: float,
    residual_fractions: np.ndarray | list[float] | tuple[float, ...],
    n_iterations: int,
    beta_ref_relative_std: float,
    min_lidar_ratio_sr: float,
    allow_negative_aerosol: bool,
    seed: int | None,
    max_relative_slope: float,
    max_relative_variance: float,
    min_valid_fraction: float,
    uncertainty_mode: UncertaintyMode = "independent",
    progressive_grid_schedule: Sequence[tuple[float, float]] = (
        DEFAULT_PROGRESSIVE_GRID_SCHEDULE
    ),
    reference_search_ranges_m: Sequence[tuple[float, float]] = (
        DEFAULT_REFERENCE_SEARCH_RANGES_M
    ),
    background_fit_min_altitude_m: float | None = None,
    background_fit_max_altitude_m: float | None = None,
    rayleigh_window_m: float = 1000.0,
    path_start_altitude_m: float = 600.0,
    integration_mode: IntegrationMode = "two_sided",
) -> TwoSidedRetrievalResult:
    """Run the complete two-sided retrieval chain for one profile.

    Scientific semantics:

    * Rayleigh QA is evaluated on the native measurement grid over a physical
      window in meters;
    * KFS operates on the explicit progressive grid;
    * the nominal reference is chosen from the first supported declared search
      range (productive default 10--15 km, fallback 5--20 km), then minimizes
      the existing Rayleigh diagnostic cost within that range;
    * range fallback is explicit metadata, not a molecular-purity claim;
    * random signal perturbations propagate through progressive-grid
      construction, Rayleigh QA, range selection, reference selection and KFS;
    * ``f`` scenarios remain outer systematic conditional experiments;
    * no MC-validity fraction is converted to a pass/fail decision here.

    The callable and productive default is two-sided.
    """
    prioritized_ranges = tuple(
        (float(lower), float(upper)) for lower, upper in reference_search_ranges_m
    )
    if not prioritized_ranges:
        raise ValueError("reference_search_ranges_m must not be empty.")
    catalogue_min_altitude_m = min(lower for lower, _upper in prioritized_ranges)
    catalogue_max_altitude_m = max(upper for _lower, upper in prioritized_ranges)
    background_min_altitude_m = (
        catalogue_min_altitude_m
        if background_fit_min_altitude_m is None
        else float(background_fit_min_altitude_m)
    )
    background_max_altitude_m = (
        catalogue_max_altitude_m
        if background_fit_max_altitude_m is None
        else float(background_fit_max_altitude_m)
    )

    background_fit = fit_rayleigh_background(
        range_corrected_signal,
        simulated_molecular_range_corrected_signal,
        altitude_m,
        min_altitude_m=background_min_altitude_m,
        max_altitude_m=background_max_altitude_m,
        measured_signal_error=range_corrected_signal_error,
    )
    if not background_fit.success:
        raise ValueError("Residual background is not identifiable over the Rayleigh search span.")
    background_corrected_signal = (
        np.asarray(range_corrected_signal, dtype=np.float64)
        - float(background_fit.background_offset)
        * np.asarray(altitude_m, dtype=np.float64) ** 2
    )

    prepared = prepare_high_column_profile(
        range_corrected_signal=background_corrected_signal,
        range_corrected_signal_error=range_corrected_signal_error,
        molecular_backscatter=molecular_backscatter,
        altitude_m=altitude_m,
        uncertainty_mode=uncertainty_mode,
        schedule=progressive_grid_schedule,
    )
    catalogue = catalogue_high_column_reference_cells(
        prepared=prepared,
        native_range_corrected_signal=background_corrected_signal,
        native_range_corrected_signal_error=range_corrected_signal_error,
        native_simulated_molecular_signal=simulated_molecular_range_corrected_signal,
        native_altitude_m=altitude_m,
        search_min_altitude_m=catalogue_min_altitude_m,
        search_max_altitude_m=float(catalogue_max_altitude_m),
        rayleigh_window_m=float(rayleigh_window_m),
        max_relative_slope=float(max_relative_slope),
        max_relative_variance=float(max_relative_variance),
        min_valid_fraction=float(min_valid_fraction),
        path_start_altitude_m=float(path_start_altitude_m),
    )
    range_selection = select_prioritized_high_column_reference(
        catalogue,
        search_ranges_m=prioritized_ranges,
    )
    selected = range_selection.reference
    selection_minimum = range_selection.search_min_altitude_m
    selection_maximum = range_selection.search_max_altitude_m
    selection_index = range_selection.search_range_index
    selection_fallback = range_selection.fallback_used
    mc = run_selection_aware_monte_carlo(
        range_corrected_signal=range_corrected_signal,
        range_corrected_signal_error=range_corrected_signal_error,
        molecular_backscatter=molecular_backscatter,
        simulated_molecular_range_corrected_signal=(
            simulated_molecular_range_corrected_signal
        ),
        altitude_m=altitude_m,
        aerosol_lidar_ratio_sr=float(aerosol_lidar_ratio_sr),
        aerosol_lidar_ratio_std_sr=float(aerosol_lidar_ratio_std_sr),
        residual_fractions=residual_fractions,
        n_iterations=int(n_iterations),
        beta_ref_relative_std=float(beta_ref_relative_std),
        min_lidar_ratio_sr=float(min_lidar_ratio_sr),
        allow_negative_aerosol=bool(allow_negative_aerosol),
        seed=seed,
        max_relative_slope=float(max_relative_slope),
        max_relative_variance=float(max_relative_variance),
        min_valid_fraction=float(min_valid_fraction),
        uncertainty_mode=uncertainty_mode,
        progressive_grid_schedule=progressive_grid_schedule,
        reference_search_ranges_m=prioritized_ranges,
        background_fit_min_altitude_m=background_min_altitude_m,
        background_fit_max_altitude_m=background_max_altitude_m,
        rayleigh_window_m=float(rayleigh_window_m),
        path_start_altitude_m=float(path_start_altitude_m),
        integration_mode=integration_mode,
    )
    return TwoSidedRetrievalResult(
        prepared=prepared,
        reference_catalogue=catalogue,
        selected_reference=selected,
        selected_reference_search_min_altitude_m=float(selection_minimum),
        selected_reference_search_max_altitude_m=float(selection_maximum),
        selected_reference_search_range_index=int(selection_index),
        selected_reference_fallback_used=bool(selection_fallback),
        reference_search_ranges_m=prioritized_ranges,
        monte_carlo=mc,
        selector_name=(
            "first_supported_reference_range_then_minimum_existing_rayleigh_cost_"
            "after_qa_and_path"
        ),
        search_min_altitude_m=float(min(lower for lower, _upper in prioritized_ranges)),
        search_max_altitude_m=float(catalogue_max_altitude_m),
        rayleigh_window_m=float(rayleigh_window_m),
        background_fit_min_altitude_m=background_min_altitude_m,
        background_offset=float(background_fit.background_offset),
        background_offset_standard_error=float(
            background_fit.background_offset_standard_error
        ),
        calibration_background_correlation=float(
            background_fit.calibration_background_correlation
        ),
    )
