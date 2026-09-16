"""Synthetic KFS truth experiments for explicit pre-retrieval vertical aggregation.

The purpose is to separate inversion bias from the unavoidable loss of vertical
resolution.  Productive Level 2 remains on the native grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.kfs import fernald_inversion
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.level2.vertical_aggregation import aggregate_uniform_vertical_bins
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def _synthetic_high_resolution_case() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    altitude_m = np.arange(300.0, 12000.0, 7.5, dtype=np.float64)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_m)
    beta_molecular, _ = calculate_molecular_profile(
        temperature_k,
        pressure_hpa,
        532.0,
    )

    broad_layer = 1.8e-6 * np.exp(-0.5 * ((altitude_m - 2100.0) / 850.0) ** 2)
    narrow_layer = 1.2e-6 * np.exp(-0.5 * ((altitude_m - 4300.0) / 75.0) ** 2)
    beta_aerosol = broad_layer + narrow_layer
    beta_aerosol[altitude_m >= 6500.0] = 0.0
    aerosol_lidar_ratio = np.full_like(altitude_m, 55.0)

    rcs = elastic_lidar_forward_model(
        altitude_m,
        beta_molecular,
        beta_aerosol,
        RAYLEIGH_LIDAR_RATIO_SR,
        aerosol_lidar_ratio,
    )
    return altitude_m, beta_molecular, beta_aerosol, aerosol_lidar_ratio, rcs


def _group_mean(values: np.ndarray, group_size: int) -> np.ndarray:
    used = (values.size // group_size) * group_size
    return values[:used].reshape(-1, group_size).mean(axis=1)


def _integrated_column(values: np.ndarray, altitude_m: np.ndarray, mask: np.ndarray) -> float:
    return float(np.trapz(values[mask], altitude_m[mask]))


@pytest.mark.parametrize("group_size", [2, 4, 8, 16, 32])
def test_pre_retrieval_vertical_aggregation_preserves_coarse_grid_kfs_truth(
    group_size: int,
) -> None:
    """15--240 m rebinning should not itself create a large KFS bias."""
    altitude, beta_mol, beta_aer, lr_aer, rcs = _synthetic_high_resolution_case()
    aggregated = aggregate_uniform_vertical_bins(
        rcs,
        np.zeros_like(rcs),
        altitude,
        bins_per_aggregate=group_size,
    )
    beta_mol_coarse = _group_mean(beta_mol, group_size)
    beta_aer_truth_coarse = _group_mean(beta_aer, group_size)
    lr_aer_coarse = _group_mean(lr_aer, group_size)

    ref_idx = int(np.argmin(np.abs(aggregated.altitude_m - 9000.0)))
    retrieved = fernald_inversion(
        aggregated.signal,
        aggregated.altitude_m,
        beta_mol_coarse,
        lr_aer_coarse,
        float(beta_mol_coarse[ref_idx]),
        ref_idx,
        lr_mol=RAYLEIGH_LIDAR_RATIO_SR,
        altitude_units="m",
        min_lidar_ratio=10.0,
        allow_negative_aerosol=True,
        mode="backward",
    )

    evaluate = (
        (aggregated.altitude_m >= 600.0)
        & (aggregated.altitude_m <= 6000.0)
    )
    assert np.all(np.isfinite(retrieved[evaluate]))
    relative_l2 = float(
        np.linalg.norm(retrieved[evaluate] - beta_aer_truth_coarse[evaluate])
        / np.linalg.norm(beta_aer_truth_coarse[evaluate])
    )
    # This is deliberately a loose R&D guard: the goal is to catch material
    # inversion bias while allowing expected coarse-grid trapezoidal error.
    assert relative_l2 < 0.05

    retrieved_column = _integrated_column(
        retrieved,
        aggregated.altitude_m,
        evaluate,
    )
    truth_column = _integrated_column(
        beta_aer_truth_coarse,
        aggregated.altitude_m,
        evaluate,
    )
    assert abs(retrieved_column / truth_column - 1.0) < 0.05


def test_vertical_aggregation_exposes_narrow_layer_resolution_loss_separately_from_kfs_bias() -> None:
    """Coarsening may lower a narrow peak even when the inversion remains correct."""
    altitude, _beta_mol, beta_aer, _lr_aer, rcs = _synthetic_high_resolution_case()
    native_layer = (altitude >= 4000.0) & (altitude <= 4600.0)
    native_peak = float(np.max(beta_aer[native_layer]))

    peak_ratios: list[float] = []
    centroid_offsets_m: list[float] = []
    true_centroid = float(
        np.sum(altitude[native_layer] * beta_aer[native_layer])
        / np.sum(beta_aer[native_layer])
    )
    for group_size in (2, 4, 8, 16, 32):
        aggregated = aggregate_uniform_vertical_bins(
            rcs,
            np.zeros_like(rcs),
            altitude,
            bins_per_aggregate=group_size,
        )
        beta_truth_coarse = _group_mean(beta_aer, group_size)
        layer = (
            (aggregated.altitude_m >= 4000.0)
            & (aggregated.altitude_m <= 4600.0)
        )
        peak_ratios.append(float(np.max(beta_truth_coarse[layer]) / native_peak))
        centroid = float(
            np.sum(aggregated.altitude_m[layer] * beta_truth_coarse[layer])
            / np.sum(beta_truth_coarse[layer])
        )
        centroid_offsets_m.append(abs(centroid - true_centroid))

    # The 240 m representation must visibly lose peak amplitude relative to the
    # 15 m representation; that is declared resolution loss, not hidden bias.
    assert peak_ratios[-1] < peak_ratios[0]
    assert peak_ratios[-1] < 0.95
    # The layer location should remain approximately centered even as its peak
    # is broadened.  Half the coarsest 240 m cell is a physically transparent
    # location tolerance for this representation-only diagnostic.
    assert centroid_offsets_m[-1] <= 120.0
