"""Contracts for the current two-sided retrieval branches."""

from __future__ import annotations

import numpy as np

from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.kfs import fernald_inversion
from milgrau.level2.two_sided_retrieval import retrieve_two_sided_profile
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def _profile() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    altitude = np.arange(300.0, 26_000.0 + 30.0, 30.0, dtype=np.float64)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude)
    beta_mol, _ = calculate_molecular_profile(temperature_k, pressure_hpa, 532.0)
    beta_aer = 2.0e-6 * np.exp(-(altitude - altitude[0]) / 1800.0)
    beta_aer[altitude >= 7000.0] = 0.0
    signal = elastic_lidar_forward_model(
        altitude, beta_mol, beta_aer, RAYLEIGH_LIDAR_RATIO_SR, 55.0
    )
    molecular_signal = elastic_lidar_forward_model(
        altitude,
        beta_mol,
        np.zeros_like(beta_aer),
        RAYLEIGH_LIDAR_RATIO_SR,
        55.0,
    )
    return altitude, beta_mol, signal, molecular_signal


def test_two_sided_preserves_backward_branch_exactly() -> None:
    altitude, beta_mol, signal, _ = _profile()
    ref_idx = int(np.argmin(np.abs(altitude - 10_500.0)))
    beta_ref = float(beta_mol[ref_idx])
    backward = fernald_inversion(
        signal, altitude, beta_mol, 55.0, beta_ref, ref_idx,
        altitude_units="m", mode="backward",
    )
    two_sided, diagnostics = fernald_inversion(
        signal, altitude, beta_mol, 55.0, beta_ref, ref_idx,
        altitude_units="m", mode="two_sided", return_diagnostics=True,
    )
    np.testing.assert_array_equal(two_sided[: ref_idx + 1], backward[: ref_idx + 1])
    assert diagnostics["backward_endpoint_index"] == 0
    assert diagnostics["backward_termination_reason"] == "grid_edge"
    assert diagnostics["forward_endpoint_altitude_m"] >= 25_000.0
    assert diagnostics["forward_termination_reason"] == "grid_edge"


def test_two_sided_mc_reports_branch_endpoints() -> None:
    altitude, beta_mol, signal, molecular_signal = _profile()
    result = retrieve_two_sided_profile(
        range_corrected_signal=signal,
        range_corrected_signal_error=np.abs(signal) * 0.001,
        molecular_backscatter=beta_mol,
        simulated_molecular_range_corrected_signal=molecular_signal,
        altitude_m=altitude,
        aerosol_lidar_ratio_sr=55.0,
        aerosol_lidar_ratio_std_sr=1.0,
        residual_fractions=(0.0, 0.02),
        n_iterations=8,
        beta_ref_relative_std=0.01,
        min_lidar_ratio_sr=10.0,
        allow_negative_aerosol=False,
        seed=19,
        max_relative_slope=0.25,
        max_relative_variance=0.50,
        min_valid_fraction=0.50,
        progressive_grid_schedule=(
            (0.0, 30.0), (6000.0, 60.0), (10_000.0, 90.0), (15_000.0, 120.0),
        ),
        reference_search_ranges_m=((10_000.0, 12_000.0),),
        rayleigh_window_m=1000.0,
        integration_mode="two_sided",
    )
    mc = result.monte_carlo
    assert mc.integration_mode == "two_sided"
    assert mc.forward_endpoint_altitude_m_samples.shape == (2, 8)
    assert mc.backward_endpoint_altitude_m_samples.shape == (2, 8)
    selected = mc.selected_reference_altitude_m_samples[np.newaxis, :]
    assert np.all(mc.forward_endpoint_altitude_m_samples > selected)
    assert np.all(mc.backward_endpoint_altitude_m_samples < selected)
    assert np.all(mc.forward_termination_reason_samples != "not_evaluated")
