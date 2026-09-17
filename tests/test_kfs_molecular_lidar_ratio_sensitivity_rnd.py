"""R&D sensitivity tests for molecular lidar-ratio semantics.

These tests do not change productive method v4. They quantify the internal
consequence of combining Bucholtz total molecular extinction with the current
wavelength-independent 8*pi/3 KFS molecular lidar ratio.
"""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.kfs import fernald_inversion
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def _relative_l2(retrieved: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return float(
        np.linalg.norm(retrieved[mask] - truth[mask])
        / np.linalg.norm(truth[mask])
    )


@pytest.mark.parametrize("wavelength_nm", [355.0, 532.0])
def test_bucholtz_model_implies_wavelength_dependent_alpha_over_beta(
    wavelength_nm: float,
) -> None:
    altitude = np.arange(300.0, 12000.0, 60.0)
    pressure, temperature = get_standard_atmosphere(altitude)
    beta_mol, alpha_mol = calculate_molecular_profile(
        temperature,
        pressure,
        wavelength_nm,
    )
    ratio = alpha_mol / beta_mol

    assert np.all(np.isfinite(ratio))
    assert float(np.nanmax(ratio) - np.nanmin(ratio)) < 1.0e-10
    model_ratio = float(np.nanmedian(ratio))
    # This is a model-internal identity check, not an operational acceptance
    # tolerance: the currently coded Bucholtz extinction/backscatter semantics
    # differ materially from the isotropic 8*pi/3 constant.
    assert abs(model_ratio / float(RAYLEIGH_LIDAR_RATIO_SR) - 1.0) > 0.01


@pytest.mark.parametrize("wavelength_nm", [355.0, 532.0])
def test_model_consistent_molecular_lidar_ratio_reduces_controlled_kfs_bias(
    wavelength_nm: float,
) -> None:
    """A forward model using alpha/beta should invert best with the same semantics."""
    altitude = np.arange(300.0, 15000.0, 30.0)
    pressure, temperature = get_standard_atmosphere(altitude)
    beta_mol, alpha_mol = calculate_molecular_profile(
        temperature,
        pressure,
        wavelength_nm,
    )
    model_lr_mol = float(np.nanmedian(alpha_mol / beta_mol))

    beta_aer = 1.4e-6 * np.exp(-0.5 * ((altitude - 2200.0) / 1300.0) ** 2)
    beta_aer[altitude >= 7000.0] = 0.0
    lidar_ratio_aer = np.full_like(altitude, 55.0)
    signal = elastic_lidar_forward_model(
        altitude,
        beta_mol,
        beta_aer,
        model_lr_mol,
        lidar_ratio_aer,
    )
    ref_idx = int(np.argmin(np.abs(altitude - 11000.0)))
    beta_total_ref = float(beta_mol[ref_idx] + beta_aer[ref_idx])

    consistent = fernald_inversion(
        signal,
        altitude,
        beta_mol,
        lidar_ratio_aer,
        beta_total_ref,
        ref_idx,
        lr_mol=model_lr_mol,
        altitude_units="m",
        min_lidar_ratio=10.0,
        allow_negative_aerosol=True,
        mode="backward",
    )
    current_constant = fernald_inversion(
        signal,
        altitude,
        beta_mol,
        lidar_ratio_aer,
        beta_total_ref,
        ref_idx,
        lr_mol=float(RAYLEIGH_LIDAR_RATIO_SR),
        altitude_units="m",
        min_lidar_ratio=10.0,
        allow_negative_aerosol=True,
        mode="backward",
    )
    evaluate = (altitude >= 600.0) & (altitude <= 6500.0)
    consistent_error = _relative_l2(consistent, beta_aer, evaluate)
    fixed_error = _relative_l2(current_constant, beta_aer, evaluate)

    assert consistent_error < 0.01
    assert fixed_error > consistent_error
    assert not np.allclose(
        current_constant[evaluate],
        consistent[evaluate],
        rtol=1.0e-5,
        atol=0.0,
    )
