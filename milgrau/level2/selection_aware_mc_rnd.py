"""Selection-aware Monte Carlo for the elastic method-v5 R&D path.

The existing KFS Monte Carlo conditions on one already-selected reference.  A
controlled synthetic experiment showed that this misses material uncertainty:
when noisy observations re-select different admissible reference cells, nominal
95% lower-column coverage drops even while every KFS realization is finite.

This R&D implementation therefore perturbs the *native* signal, rebuilds the
progressive representation, reruns Rayleigh QA and reference selection, and
only then evaluates KFS.  Residual aerosol fraction ``f`` remains an outer
systematic scenario rather than a random draw.

The implementation favors transparent semantics over speed.  It is not wired
into productive method v4 and should be optimized only after its statistical
behavior is validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from milgrau.level2.adaptive_grid import UncertaintyMode
from milgrau.level2.high_column_rnd import (
    V5_PROGRESSIVE_GRID_SCHEDULE,
    catalogue_high_column_reference_cells,
    prepare_high_column_profile,
)
from milgrau.level2.high_column_selector import (
    V5_REFERENCE_SEARCH_MAX_M,
    V5_REFERENCE_SEARCH_MIN_M,
    select_minimum_cost_high_column_reference,
)
from milgrau.level2.kfs import fernald_inversion


@dataclass(frozen=True, slots=True)
class SelectionAwareMonteCarloSensitivity:
    """Nested ``f`` scenarios with signal-noise-aware reference re-selection."""

    residual_aerosol_fraction_of_molecular: np.ndarray
    altitude_m: np.ndarray
    aerosol_backscatter_mean: np.ndarray
    aerosol_backscatter_random_std: np.ndarray
    aerosol_backscatter_random_q025: np.ndarray
    aerosol_backscatter_random_q975: np.ndarray
    aerosol_extinction_mean: np.ndarray
    aerosol_extinction_random_std: np.ndarray
    aerosol_extinction_random_q025: np.ndarray
    aerosol_extinction_random_q975: np.ndarray
    aerosol_backscatter_valid_count: np.ndarray
    aerosol_backscatter_valid_fraction: np.ndarray
    selection_success_count: int
    selection_success_fraction: float
    selected_reference_index_samples: np.ndarray
    selected_reference_altitude_m_samples: np.ndarray
    n_iterations: int
    uncertainty_scope: str


def _finite_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return finite-only mean/std/2.5%/97.5% for simulation x altitude values."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("values must have shape (simulation, altitude).")
    n_altitude = array.shape[1]
    mean = np.full(n_altitude, np.nan, dtype=np.float64)
    std = np.full(n_altitude, np.nan, dtype=np.float64)
    q025 = np.full(n_altitude, np.nan, dtype=np.float64)
    q975 = np.full(n_altitude, np.nan, dtype=np.float64)
    for altitude_index in range(n_altitude):
        finite = array[:, altitude_index]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            mean[altitude_index] = float(np.mean(finite))
            std[altitude_index] = float(np.std(finite))
            q025[altitude_index] = float(np.quantile(finite, 0.025))
            q975[altitude_index] = float(np.quantile(finite, 0.975))
    return mean, std, q025, q975


def selection_aware_boundary_monte_carlo_rnd(
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
    search_min_altitude_m: float = V5_REFERENCE_SEARCH_MIN_M,
    search_max_altitude_m: float = V5_REFERENCE_SEARCH_MAX_M,
    rayleigh_window_m: float = 1000.0,
    path_start_altitude_m: float = 600.0,
) -> SelectionAwareMonteCarloSensitivity:
    """Propagate native signal noise through v5 reference selection and KFS.

    Each realization uses one native signal perturbation for both selection and
    inversion, preserving their statistical dependence.  The same selected
    reference and signal realization is then evaluated for every caller-declared
    ``f`` scenario, while the same lidar-ratio and reference-boundary random
    variates are reused across scenarios for paired comparison.

    No valid-fraction threshold is imposed.  A failed selector simply yields an
    invalid realization, which is visible in ``selection_success_fraction`` and
    altitude-resolved finite-realization fractions.
    """
    signal = np.asarray(range_corrected_signal, dtype=np.float64)
    signal_error = np.asarray(range_corrected_signal_error, dtype=np.float64)
    molecular = np.asarray(molecular_backscatter, dtype=np.float64)
    molecular_signal = np.asarray(
        simulated_molecular_range_corrected_signal,
        dtype=np.float64,
    )
    native_altitude = np.asarray(altitude_m, dtype=np.float64)
    fractions = np.asarray(residual_fractions, dtype=np.float64)
    if not (
        signal.ndim
        == signal_error.ndim
        == molecular.ndim
        == molecular_signal.ndim
        == native_altitude.ndim
        == 1
    ):
        raise ValueError("all native method-v5 Monte-Carlo inputs must be one-dimensional.")
    if not (
        signal.shape
        == signal_error.shape
        == molecular.shape
        == molecular_signal.shape
        == native_altitude.shape
    ):
        raise ValueError("all native method-v5 Monte-Carlo inputs must have identical shapes.")
    if np.any(np.isfinite(signal_error) & (signal_error < 0.0)):
        raise ValueError("range_corrected_signal_error must be nonnegative where finite.")
    if fractions.ndim != 1 or fractions.size == 0:
        raise ValueError("residual_fractions must be a non-empty one-dimensional sequence.")
    if np.any(~np.isfinite(fractions)) or np.any(fractions < 0.0):
        raise ValueError("residual_fractions must contain finite non-negative values.")
    if np.unique(fractions).size != fractions.size:
        raise ValueError("residual_fractions must not contain duplicate scenarios.")

    iterations = int(n_iterations)
    if iterations <= 0:
        raise ValueError("n_iterations must be positive.")
    if not np.isfinite(aerosol_lidar_ratio_sr) or aerosol_lidar_ratio_sr <= 0.0:
        raise ValueError("aerosol_lidar_ratio_sr must be finite and positive.")
    if not np.isfinite(aerosol_lidar_ratio_std_sr) or aerosol_lidar_ratio_std_sr < 0.0:
        raise ValueError("aerosol_lidar_ratio_std_sr must be finite and nonnegative.")
    if not np.isfinite(beta_ref_relative_std) or beta_ref_relative_std < 0.0:
        raise ValueError("beta_ref_relative_std must be finite and nonnegative.")

    # Build once to establish the deterministic output grid shape.  Every later
    # realization uses the same altitude/schedule and therefore the same grid.
    baseline_prepared = prepare_high_column_profile(
        range_corrected_signal=signal,
        range_corrected_signal_error=signal_error,
        molecular_backscatter=molecular,
        altitude_m=native_altitude,
        uncertainty_mode=uncertainty_mode,
        schedule=progressive_grid_schedule,
    )
    output_altitude = baseline_prepared.grid.altitude_m
    n_altitude = output_altitude.size
    n_fraction = fractions.size

    beta_sims = np.full(
        (n_fraction, iterations, n_altitude),
        np.nan,
        dtype=np.float64,
    )
    alpha_sims = np.full_like(beta_sims, np.nan)
    selected_indices = np.full(iterations, -1, dtype=np.int32)
    selected_altitudes = np.full(iterations, np.nan, dtype=np.float64)

    rng = np.random.default_rng(seed)
    signal_noise = rng.standard_normal((iterations, signal.size))
    lr_samples = np.maximum(
        rng.normal(
            float(aerosol_lidar_ratio_sr),
            float(aerosol_lidar_ratio_std_sr),
            size=iterations,
        ),
        float(min_lidar_ratio_sr),
    )
    beta_ref_standard_normal = rng.standard_normal(iterations)

    finite_error = np.isfinite(signal_error)
    for iteration in range(iterations):
        perturbed_native = signal.copy()
        perturbed_native[finite_error] = (
            signal[finite_error]
            + signal_error[finite_error] * signal_noise[iteration, finite_error]
        )

        prepared = prepare_high_column_profile(
            range_corrected_signal=perturbed_native,
            range_corrected_signal_error=signal_error,
            molecular_backscatter=molecular,
            altitude_m=native_altitude,
            uncertainty_mode=uncertainty_mode,
            schedule=progressive_grid_schedule,
        )
        if not np.array_equal(prepared.grid.altitude_m, output_altitude):
            raise RuntimeError("progressive-grid geometry changed across MC realizations.")
        catalogue = catalogue_high_column_reference_cells(
            prepared=prepared,
            native_range_corrected_signal=perturbed_native,
            native_range_corrected_signal_error=signal_error,
            native_simulated_molecular_signal=molecular_signal,
            native_altitude_m=native_altitude,
            search_min_altitude_m=float(search_min_altitude_m),
            search_max_altitude_m=float(search_max_altitude_m),
            rayleigh_window_m=float(rayleigh_window_m),
            max_relative_slope=float(max_relative_slope),
            max_relative_variance=float(max_relative_variance),
            min_valid_fraction=float(min_valid_fraction),
            path_start_altitude_m=float(path_start_altitude_m),
        )
        try:
            selected = select_minimum_cost_high_column_reference(
                catalogue,
                min_altitude_m=float(search_min_altitude_m),
                max_altitude_m=float(search_max_altitude_m),
            )
        except ValueError:
            continue

        ref_idx = int(selected.cell_index)
        selected_indices[iteration] = ref_idx
        selected_altitudes[iteration] = float(selected.altitude_m)
        lidar_ratio = float(lr_samples[iteration])
        for fraction_index, fraction in enumerate(fractions):
            beta_ref_mean = float(prepared.molecular_backscatter[ref_idx]) * (
                1.0 + float(fraction)
            )
            beta_ref = beta_ref_mean + (
                abs(beta_ref_mean)
                * float(beta_ref_relative_std)
                * float(beta_ref_standard_normal[iteration])
            )
            try:
                beta = fernald_inversion(
                    prepared.range_corrected_signal,
                    prepared.grid.altitude_m,
                    prepared.molecular_backscatter,
                    lidar_ratio,
                    beta_ref,
                    ref_idx,
                    altitude_units="m",
                    min_lidar_ratio=float(min_lidar_ratio_sr),
                    allow_negative_aerosol=bool(allow_negative_aerosol),
                    mode="backward",
                )
            except ValueError:
                continue
            beta_array = np.asarray(beta, dtype=np.float64)
            beta_sims[fraction_index, iteration, :] = beta_array
            finite_beta = np.isfinite(beta_array)
            alpha_sims[fraction_index, iteration, finite_beta] = (
                beta_array[finite_beta] * lidar_ratio
            )

    beta_means: list[np.ndarray] = []
    beta_stds: list[np.ndarray] = []
    beta_q025: list[np.ndarray] = []
    beta_q975: list[np.ndarray] = []
    alpha_means: list[np.ndarray] = []
    alpha_stds: list[np.ndarray] = []
    alpha_q025: list[np.ndarray] = []
    alpha_q975: list[np.ndarray] = []
    for fraction_index in range(n_fraction):
        b_mean, b_std, b_q025, b_q975 = _finite_stats(
            beta_sims[fraction_index]
        )
        a_mean, a_std, a_q025, a_q975 = _finite_stats(
            alpha_sims[fraction_index]
        )
        beta_means.append(b_mean)
        beta_stds.append(b_std)
        beta_q025.append(b_q025)
        beta_q975.append(b_q975)
        alpha_means.append(a_mean)
        alpha_stds.append(a_std)
        alpha_q025.append(a_q025)
        alpha_q975.append(a_q975)

    valid_count = np.count_nonzero(np.isfinite(beta_sims), axis=1).astype(np.int32)
    selection_success_count = int(np.count_nonzero(selected_indices >= 0))
    return SelectionAwareMonteCarloSensitivity(
        residual_aerosol_fraction_of_molecular=fractions.copy(),
        altitude_m=np.asarray(output_altitude, dtype=np.float64),
        aerosol_backscatter_mean=np.stack(beta_means, axis=0),
        aerosol_backscatter_random_std=np.stack(beta_stds, axis=0),
        aerosol_backscatter_random_q025=np.stack(beta_q025, axis=0),
        aerosol_backscatter_random_q975=np.stack(beta_q975, axis=0),
        aerosol_extinction_mean=np.stack(alpha_means, axis=0),
        aerosol_extinction_random_std=np.stack(alpha_stds, axis=0),
        aerosol_extinction_random_q025=np.stack(alpha_q025, axis=0),
        aerosol_extinction_random_q975=np.stack(alpha_q975, axis=0),
        aerosol_backscatter_valid_count=valid_count,
        aerosol_backscatter_valid_fraction=(
            valid_count.astype(np.float64) / float(iterations)
        ),
        selection_success_count=selection_success_count,
        selection_success_fraction=float(selection_success_count / iterations),
        selected_reference_index_samples=selected_indices,
        selected_reference_altitude_m_samples=selected_altitudes,
        n_iterations=iterations,
        uncertainty_scope=(
            "selection_aware_native_signal_noise_plus_lidar_ratio_and_reference_boundary_"
            "dispersion_with_outer_fixed_f_scenarios"
        ),
    )
