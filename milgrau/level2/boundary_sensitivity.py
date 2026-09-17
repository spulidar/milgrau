"""Explicit KFS boundary-condition sensitivity for high-column R&D.

Productive method v4 assumes zero aerosol backscatter at the exact Rayleigh
reference bin. This module does not estimate or correct that assumption. It
only evaluates caller-declared residual aerosol fractions so the scientific
dependence on the boundary condition remains distinct from random measurement
noise and lidar-ratio uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from milgrau.level2.kfs import fernald_inversion, kfs_inversion_monte_carlo


@dataclass(frozen=True, slots=True)
class BoundaryFractionSensitivity:
    """Deterministic backward-KFS profiles for declared boundary fractions."""

    residual_aerosol_fraction_of_molecular: np.ndarray
    beta_total_reference: np.ndarray
    aerosol_backscatter: np.ndarray
    reference_index: int


@dataclass(frozen=True, slots=True)
class BoundaryFractionMonteCarloSensitivity:
    """Nested random-MC results for caller-declared boundary scenarios.

    The first dimension of every profile quantity is the residual-aerosol
    scenario ``f = beta_aer(ref) / beta_mol(ref)``.  ``f`` is not sampled from a
    probability density: every scenario is an explicit conditional experiment.
    Random signal/LR/reference-estimator perturbations occur *within* each
    scenario.  The same random seed is reused for each scenario to provide a
    paired Monte-Carlo comparison.

    ``backward_valid_fraction`` is the fraction of simulations that survive the
    complete requested backward branch. ``aerosol_backscatter_valid_fraction``
    is altitude resolved and reports how many simulations are finite at each
    retrieval cell.  The latter is the relevant support diagnostic for an
    altitude-resolved uncertainty product and is not converted to a binary gate.
    """

    residual_aerosol_fraction_of_molecular: np.ndarray
    beta_total_reference_nominal: np.ndarray
    aerosol_backscatter_mean: np.ndarray
    aerosol_backscatter_random_std: np.ndarray
    aerosol_extinction_mean: np.ndarray
    aerosol_extinction_random_std: np.ndarray
    aerosol_backscatter_valid_count: np.ndarray
    aerosol_backscatter_valid_fraction: np.ndarray
    backward_valid_count: np.ndarray
    backward_valid_fraction: np.ndarray
    n_iterations: int
    reference_index: int
    uncertainty_scope: str


def _validate_boundary_inputs(
    *,
    rcs: np.ndarray,
    altitude_m: np.ndarray,
    beta_mol: np.ndarray,
    reference_index: int,
    residual_fractions: np.ndarray | list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    signal = np.asarray(rcs, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    molecular = np.asarray(beta_mol, dtype=np.float64)
    fractions = np.asarray(residual_fractions, dtype=np.float64)

    if signal.ndim != 1 or altitude.ndim != 1 or molecular.ndim != 1:
        raise ValueError("rcs, altitude_m and beta_mol must be one-dimensional.")
    if not (signal.shape == altitude.shape == molecular.shape):
        raise ValueError("rcs, altitude_m and beta_mol must have identical shapes.")
    if fractions.ndim != 1 or fractions.size == 0:
        raise ValueError(
            "residual_fractions must be a non-empty one-dimensional sequence."
        )
    if np.any(~np.isfinite(fractions)) or np.any(fractions < 0.0):
        raise ValueError("residual_fractions must contain finite non-negative values.")
    if np.unique(fractions).size != fractions.size:
        raise ValueError("residual_fractions must not contain duplicate scenarios.")

    ref_idx = int(reference_index)
    if ref_idx < 0:
        ref_idx += signal.size
    if ref_idx < 0 or ref_idx >= signal.size:
        raise ValueError("reference_index must point inside the altitude grid.")
    beta_mol_ref = float(molecular[ref_idx])
    if not np.isfinite(beta_mol_ref) or beta_mol_ref <= 0.0:
        raise ValueError("beta_mol must be finite and positive at reference_index.")

    return signal, altitude, molecular, ref_idx, fractions


def boundary_fraction_sensitivity_profiles(
    *,
    rcs: np.ndarray,
    altitude_m: np.ndarray,
    beta_mol: np.ndarray,
    reference_index: int,
    aerosol_lidar_ratio_sr: float | np.ndarray,
    residual_fractions: np.ndarray | list[float] | tuple[float, ...],
    min_lidar_ratio_sr: float = 10.0,
    allow_negative_aerosol: bool = False,
) -> BoundaryFractionSensitivity:
    """Evaluate explicit ``beta_aer(ref)/beta_mol(ref)`` sensitivity values.

    ``residual_fractions`` are scenario inputs, not inferred quantities or
    probabilities. For each value ``f``, the exact KFS boundary is
    ``beta_total(ref) = beta_mol(ref) * (1 + f)``. The signal, grid and aerosol
    lidar ratio remain unchanged. No score, preferred fraction or pass/fail
    state is produced.
    """
    signal, altitude, molecular, ref_idx, fractions = _validate_boundary_inputs(
        rcs=rcs,
        altitude_m=altitude_m,
        beta_mol=beta_mol,
        reference_index=reference_index,
        residual_fractions=residual_fractions,
    )

    beta_mol_ref = float(molecular[ref_idx])
    boundary = beta_mol_ref * (1.0 + fractions)
    profiles = np.stack(
        [
            fernald_inversion(
                signal,
                altitude,
                molecular,
                aerosol_lidar_ratio_sr,
                float(beta_total_ref),
                ref_idx,
                altitude_units="m",
                min_lidar_ratio=float(min_lidar_ratio_sr),
                allow_negative_aerosol=bool(allow_negative_aerosol),
                mode="backward",
            )
            for beta_total_ref in boundary
        ],
        axis=0,
    )
    return BoundaryFractionSensitivity(
        residual_aerosol_fraction_of_molecular=fractions.copy(),
        beta_total_reference=np.asarray(boundary, dtype=np.float64),
        aerosol_backscatter=profiles,
        reference_index=ref_idx,
    )


def boundary_fraction_monte_carlo_sensitivity(
    *,
    rcs: np.ndarray,
    rcs_error: np.ndarray | None,
    altitude_m: np.ndarray,
    beta_mol: np.ndarray,
    reference_index: int,
    aerosol_lidar_ratio_sr: float,
    aerosol_lidar_ratio_std_sr: float,
    residual_fractions: np.ndarray | list[float] | tuple[float, ...],
    n_iterations: int = 300,
    beta_ref_relative_std: float = 0.10,
    min_lidar_ratio_sr: float = 10.0,
    allow_negative_aerosol: bool = False,
    seed: int | None = None,
) -> BoundaryFractionMonteCarloSensitivity:
    """Run paired random Monte Carlo inside explicit boundary ``f`` scenarios.

    This function intentionally does **not** draw ``f`` randomly.  The caller
    supplies a finite family of physically interpretable sensitivity scenarios.
    For every scenario the KFS Monte Carlo propagates the existing random
    ingredients (signal uncertainty, scalar aerosol lidar-ratio uncertainty and
    reference-boundary estimator perturbation).  Reusing the same ``seed`` for
    every scenario gives paired random draws, making the between-scenario
    boundary effect easier to interpret.

    Complete-branch and altitude-resolved valid-realization fractions are
    diagnostic-only. No cutoff is applied and no productive method-v4 validity
    semantics are changed.
    """
    signal, altitude, molecular, ref_idx, fractions = _validate_boundary_inputs(
        rcs=rcs,
        altitude_m=altitude_m,
        beta_mol=beta_mol,
        reference_index=reference_index,
        residual_fractions=residual_fractions,
    )
    iterations = int(n_iterations)
    if iterations <= 0:
        raise ValueError("n_iterations must be positive.")

    beta_means: list[np.ndarray] = []
    beta_stds: list[np.ndarray] = []
    alpha_means: list[np.ndarray] = []
    alpha_stds: list[np.ndarray] = []
    profile_valid_counts: list[np.ndarray] = []
    branch_valid_counts: list[int] = []

    for fraction in fractions:
        beta_mean, beta_std, alpha_mean, alpha_std, diagnostics = (
            kfs_inversion_monte_carlo(
                rcs=signal,
                altitude=altitude,
                beta_mol=molecular,
                lr_base=float(aerosol_lidar_ratio_sr),
                lr_std=float(aerosol_lidar_ratio_std_sr),
                ref_idx=ref_idx,
                n_iterations=iterations,
                rcs_error=rcs_error,
                beta_ref_relative_std=float(beta_ref_relative_std),
                aerosol_ref_fraction=float(fraction),
                altitude_units="m",
                min_lidar_ratio=float(min_lidar_ratio_sr),
                allow_negative_aerosol=bool(allow_negative_aerosol),
                seed=seed,
                return_diagnostics=True,
                mode="backward",
            )
        )
        beta_means.append(np.asarray(beta_mean, dtype=np.float64))
        beta_stds.append(np.asarray(beta_std, dtype=np.float64))
        alpha_means.append(np.asarray(alpha_mean, dtype=np.float64))
        alpha_stds.append(np.asarray(alpha_std, dtype=np.float64))
        beta_sims = np.asarray(diagnostics["beta_aer_sims"], dtype=np.float64)
        profile_valid_counts.append(
            np.count_nonzero(np.isfinite(beta_sims), axis=0).astype(np.int32)
        )
        branch_valid_counts.append(
            int(np.count_nonzero(diagnostics["backward_valid_simulations"]))
        )

    profile_valid_count_arr = np.stack(profile_valid_counts, axis=0)
    branch_valid_count_arr = np.asarray(branch_valid_counts, dtype=np.int32)
    beta_mol_ref = float(molecular[ref_idx])
    return BoundaryFractionMonteCarloSensitivity(
        residual_aerosol_fraction_of_molecular=fractions.copy(),
        beta_total_reference_nominal=beta_mol_ref * (1.0 + fractions),
        aerosol_backscatter_mean=np.stack(beta_means, axis=0),
        aerosol_backscatter_random_std=np.stack(beta_stds, axis=0),
        aerosol_extinction_mean=np.stack(alpha_means, axis=0),
        aerosol_extinction_random_std=np.stack(alpha_stds, axis=0),
        aerosol_backscatter_valid_count=profile_valid_count_arr,
        aerosol_backscatter_valid_fraction=(
            profile_valid_count_arr.astype(np.float64) / float(iterations)
        ),
        backward_valid_count=branch_valid_count_arr,
        backward_valid_fraction=(
            branch_valid_count_arr.astype(np.float64) / float(iterations)
        ),
        n_iterations=iterations,
        reference_index=ref_idx,
        uncertainty_scope=(
            "nested_boundary_scenarios_within_scenario_partial_monte_carlo_dispersion"
        ),
    )
