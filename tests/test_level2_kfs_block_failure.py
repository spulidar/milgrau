"""Regression tests for block-local KFS failures after Rayleigh QA."""

from __future__ import annotations

import logging

import numpy as np

from milgrau.level2.optical_retrieval import MolecularModel, retrieve_optical_blocks
from milgrau.level2.rayleigh_candidates import RayleighReferenceCandidate
from milgrau.level2.signal_selection import BlockGluingResult, WavelengthBlockInputs


def _candidate(altitude_m: np.ndarray) -> RayleighReferenceCandidate:
    center = 10
    return RayleighReferenceCandidate(
        center_index=center,
        start_index=8,
        stop_index=13,
        center_altitude_m=float(altitude_m[center]),
        start_altitude_m=float(altitude_m[8]),
        stop_altitude_m=float(altitude_m[12]),
        valid_bins=5,
        total_bins=5,
        valid_fraction=1.0,
        relative_slope=0.0,
        relative_variance=0.0,
        calibration_factor=1.0,
        free_intercept=0.0,
        uncertainty_snr_median=10.0,
        uncertainty_snr_valid_bins=5,
        diagnostic_cost=0.0,
        rejection_mask=0,
    )


def test_kfs_input_value_error_rejects_only_one_block(monkeypatch) -> None:
    """One unusable exact boundary bin must not abort the whole wavelength."""
    n_altitude = 20
    altitude_m = np.arange(n_altitude, dtype=np.float64) * 7.5
    block_signal = np.ones((2, n_altitude), dtype=np.float64)
    block_error = np.full_like(block_signal, 0.05)

    inputs = WavelengthBlockInputs(
        wavelength_nm=532,
        analog_channel="532.AN",
        photon_channel="532.PC",
        n_time=2,
        n_altitude=n_altitude,
        block_time=np.array([0, 1]),
        block_groups=[np.array([0]), np.array([1])],
        gluing_config={},
        molecular_fit_config={},
        analog_block=None,
        analog_error_block=None,
        analog_correction_valid=True,
        photon_block=None,
        photon_error_block=None,
        photon_mask_block=None,
        photon_correction_valid=True,
    )
    glued = BlockGluingResult(
        source="test",
        corrected_signal=block_signal.copy(),
        corrected_signal_error=block_error.copy(),
        range_corrected_signal=block_signal.copy(),
        range_corrected_signal_error=block_error.copy(),
        merge_source_flag=np.zeros((2, n_altitude), dtype=np.int8),
        attempted_flag=np.ones(2, dtype=np.int8),
        success_flag=np.ones(2, dtype=np.int8),
        single_channel_fallback_flag=np.zeros(2, dtype=np.int8),
        signal_source_flag=np.ones(2, dtype=np.int8),
        retrieval_input_valid_flag=np.ones(2, dtype=np.int8),
        retrieval_input_invalid_reason=np.zeros(2, dtype=np.int16),
        retrieval_input_snr_median=np.full(2, 20.0),
        split_altitude_m=np.full(2, np.nan),
        start_altitude_m=np.full(2, np.nan),
        stop_altitude_m=np.full(2, np.nan),
        slope=np.ones(2),
        intercept=np.zeros(2),
        correlation=np.ones(2),
        relative_rmse=np.zeros(2),
        relative_bias=np.zeros(2),
    )
    molecular = MolecularModel(
        source="synthetic",
        backscatter=np.full(n_altitude, 1.0e-6),
        extinction=np.full(n_altitude, 1.0e-5),
        transmission=np.ones(n_altitude),
        simulated_signal=np.ones(n_altitude),
        simulated_range_corrected_signal=np.ones(n_altitude),
        fit_config={
            "ref_alt_min_m": 0.0,
            "ref_alt_max_m": 150.0,
            "ref_window_bins": 5,
            "max_relative_slope": 0.1,
            "max_relative_variance": 0.1,
            "min_valid_fraction": 0.5,
        },
        lidar_ratio_assumed_sr=50.0,
        lidar_ratio_std_sr=5.0,
        kfs_mode="backward",
    )

    monkeypatch.setattr(
        "milgrau.level2.optical_retrieval.catalogue_rayleigh_candidates",
        lambda *args, **kwargs: (_candidate(altitude_m),),
    )
    calls = {"count": 0}

    def fake_kfs(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError(
                "The range-corrected signal must be finite and positive at ref_idx."
            )
        beta = np.full(n_altitude, 2.0e-6)
        beta_error = np.full(n_altitude, 0.2e-6)
        alpha = np.full(n_altitude, 100.0e-6)
        alpha_error = np.full(n_altitude, 10.0e-6)
        return beta, beta_error, alpha, alpha_error, {
            "backward_valid": True,
            "forward_valid": False,
        }

    monkeypatch.setattr(
        "milgrau.level2.optical_retrieval.run_kfs_profile",
        fake_kfs,
    )

    logger = logging.getLogger("test.level2.kfs_block_failure")
    molecular_out, optical, rayleigh, kfs = retrieve_optical_blocks(
        inputs,
        glued,
        molecular,
        altitude_m,
        {},
        logger,
    )

    assert molecular_out.source == "synthetic"
    np.testing.assert_array_equal(
        rayleigh.reference_success_flag_block,
        np.array([1, 1], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        kfs.backward_valid_flag_block,
        np.array([0, 1], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        optical.retrieval_success_flag,
        np.array([0, 1], dtype=np.int8),
    )
    np.testing.assert_allclose(optical.aerosol_backscatter, 2.0e-6)
    np.testing.assert_allclose(optical.aerosol_extinction, 100.0e-6)
