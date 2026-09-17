"""End-to-end executable elastic method-v5 R&D retrieval for one profile.

This module is deliberately isolated from productive LEBEAR method v4. It
assembles the explicit v5 R&D components without promoting them: progressive
vertical representation, native-grid Rayleigh QA, tiered deterministic
high-column reference selection, and a selection-aware nested
boundary-sensitivity/Monte-Carlo ensemble.

No Monte-Carlo valid-fraction cutoff is applied here. The returned diagnostic
fractions are evidence to be interpreted by validation/coverage studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from milgrau.level2.adaptive_grid import UncertaintyMode
from milgrau.level2.high_column_rnd import (
    HighColumnReferenceCatalogue,
    HighColumnReferenceCell,
    PreparedHighColumnProfile,
    V5_PROGRESSIVE_GRID_SCHEDULE,
    catalogue_high_column_reference_cells,
    prepare_high_column_profile,
)
from milgrau.level2.high_column_selector import (
    V5_REFERENCE_SEARCH_MAX_M,
    V5_REFERENCE_SEARCH_TIER_MINIMA_M,
    select_tiered_high_column_reference,
)
from milgrau.level2.selection_aware_mc_rnd import (
    SelectionAwareMonteCarloSensitivity,
    selection_aware_boundary_monte_carlo_rnd,
)


@dataclass(frozen=True, slots=True)
class MethodV5RNDResult:
    """Complete auditable state of one experimental v5 retrieval."""

    prepared: PreparedHighColumnProfile
    reference_catalogue: HighColumnReferenceCatalogue
    selected_reference: HighColumnReferenceCell
    selected_reference_tier_min_altitude_m: float
    selected_reference_tier_index: int
    selected_reference_fallback_used: bool
    reference_tier_min_altitudes_m: tuple[float, ...]
    monte_carlo: SelectionAwareMonteCarloSensitivity
    selector_name: str
    search_min_altitude_m: float
    search_max_altitude_m: float
    rayleigh_window_m: float


def retrieve_method_v5_rnd(
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
        V5_PROGRESSIVE_GRID_SCHEDULE
    ),
    reference_tier_min_altitudes_m: Sequence[float] = (
        V5_REFERENCE_SEARCH_TIER_MINIMA_M
    ),
    search_min_altitude_m: float | None = None,
    search_max_altitude_m: float = V5_REFERENCE_SEARCH_MAX_M,
    rayleigh_window_m: float = 1000.0,
    path_start_altitude_m: float = 600.0,
) -> MethodV5RNDResult:
    """Run the complete method-v5 R&D chain for one profile.

    Scientific semantics:

    * Rayleigh QA is evaluated on the native measurement grid over a physical
      window in meters;
    * KFS operates on the explicit progressive grid;
    * the nominal reference is chosen from the highest supported declared tier
      (default 10 -> 9 -> 8 -> 6 km), then minimizes the existing Rayleigh
      diagnostic cost within that tier;
    * tier fallback is explicit metadata, not a molecular-purity claim;
    * random signal perturbations propagate through progressive-grid
      construction, Rayleigh QA, tier selection, reference selection and KFS;
    * ``f`` scenarios remain outer systematic conditional experiments;
    * no MC-validity fraction is converted to a pass/fail decision here.

    ``search_min_altitude_m`` remains available as a backward-compatible
    single-tier override for controlled experiments.
    """
    tier_minima = (
        (float(search_min_altitude_m),)
        if search_min_altitude_m is not None
        else tuple(float(value) for value in reference_tier_min_altitudes_m)
    )
    if not tier_minima:
        raise ValueError("reference_tier_min_altitudes_m must not be empty.")
    catalogue_min_altitude_m = float(min(tier_minima))

    prepared = prepare_high_column_profile(
        range_corrected_signal=range_corrected_signal,
        range_corrected_signal_error=range_corrected_signal_error,
        molecular_backscatter=molecular_backscatter,
        altitude_m=altitude_m,
        uncertainty_mode=uncertainty_mode,
        schedule=progressive_grid_schedule,
    )
    catalogue = catalogue_high_column_reference_cells(
        prepared=prepared,
        native_range_corrected_signal=range_corrected_signal,
        native_range_corrected_signal_error=range_corrected_signal_error,
        native_simulated_molecular_signal=simulated_molecular_range_corrected_signal,
        native_altitude_m=altitude_m,
        search_min_altitude_m=catalogue_min_altitude_m,
        search_max_altitude_m=float(search_max_altitude_m),
        rayleigh_window_m=float(rayleigh_window_m),
        max_relative_slope=float(max_relative_slope),
        max_relative_variance=float(max_relative_variance),
        min_valid_fraction=float(min_valid_fraction),
        path_start_altitude_m=float(path_start_altitude_m),
    )
    selection = select_tiered_high_column_reference(
        catalogue,
        tier_min_altitudes_m=tier_minima,
        max_altitude_m=float(search_max_altitude_m),
    )
    selected = selection.reference
    mc = selection_aware_boundary_monte_carlo_rnd(
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
        reference_tier_min_altitudes_m=tier_minima,
        search_min_altitude_m=None,
        search_max_altitude_m=float(search_max_altitude_m),
        rayleigh_window_m=float(rayleigh_window_m),
        path_start_altitude_m=float(path_start_altitude_m),
    )
    return MethodV5RNDResult(
        prepared=prepared,
        reference_catalogue=catalogue,
        selected_reference=selected,
        selected_reference_tier_min_altitude_m=float(selection.tier_min_altitude_m),
        selected_reference_tier_index=int(selection.tier_index),
        selected_reference_fallback_used=bool(selection.fallback_used),
        reference_tier_min_altitudes_m=tuple(tier_minima),
        monte_carlo=mc,
        selector_name=(
            "highest_supported_reference_tier_then_minimum_existing_rayleigh_cost_"
            "after_qa_and_path"
        ),
        search_min_altitude_m=float(tier_minima[0]),
        search_max_altitude_m=float(search_max_altitude_m),
        rayleigh_window_m=float(rayleigh_window_m),
    )
