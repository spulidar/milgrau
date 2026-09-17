"""Explicit KFS boundary-condition sensitivity for high-column R&D.

Productive method v4 assumes zero aerosol backscatter at the exact Rayleigh
reference bin. This module does not estimate or correct that assumption. It
only evaluates a caller-declared family of residual aerosol fractions so the
scientific dependence on the boundary condition can be exposed separately from
signal noise, lidar-ratio uncertainty and vertical aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from milgrau.level2.kfs import fernald_inversion


@dataclass(frozen=True, slots=True)
class BoundaryFractionSensitivity:
    """Deterministic backward-KFS profiles for declared boundary fractions."""

    residual_aerosol_fraction_of_molecular: np.ndarray
    beta_total_reference: np.ndarray
    aerosol_backscatter: np.ndarray
    reference_index: int


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
    signal = np.asarray(rcs, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    molecular = np.asarray(beta_mol, dtype=np.float64)
    fractions = np.asarray(residual_fractions, dtype=np.float64)

    if signal.ndim != 1 or altitude.ndim != 1 or molecular.ndim != 1:
        raise ValueError("rcs, altitude_m and beta_mol must be one-dimensional.")
    if not (signal.shape == altitude.shape == molecular.shape):
        raise ValueError("rcs, altitude_m and beta_mol must have identical shapes.")
    if fractions.ndim != 1 or fractions.size == 0:
        raise ValueError("residual_fractions must be a non-empty one-dimensional sequence.")
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
