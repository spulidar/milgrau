"""Regression tests for Level 2 Rayleigh-search input support."""

from __future__ import annotations

import numpy as np

import milgrau.level2._retrieval_impl as _impl
from milgrau.level2.contracts import RetrievalInputInvalidReason
from milgrau.level2.kfs import fernald_inversion
from milgrau.level2.retrieval_input_qa import evaluate_retrieval_input_supported_domain


def _fit_config() -> dict[str, float | int]:
    return {
        "ref_alt_min_m": 200.0,
        "ref_alt_max_m": 700.0,
        "ref_window_bins": 3,
        "min_valid_fraction": 2.0 / 3.0,
    }


def _evaluate(
    signal: np.ndarray,
    error: np.ndarray,
) -> tuple[bool, RetrievalInputInvalidReason, float]:
    altitude = np.arange(signal.size, dtype=np.float64) * 100.0
    return evaluate_retrieval_input_supported_domain(
        signal,
        error,
        altitude,
        _fit_config(),
        correction_valid=True,
    )


def test_level2_installs_rayleigh_search_qa_into_legacy_retrieval_impl() -> None:
    assert _impl._evaluate_retrieval_input is evaluate_retrieval_input_supported_domain


def test_bin_shift_style_edge_nans_are_allowed() -> None:
    signal = np.array([np.nan, 5.0, 4.0, 3.0, 2.0, 1.0, 0.8, np.nan, np.nan])
    error = np.array([np.nan, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, np.nan, np.nan])

    valid, reason, snr = _evaluate(signal, error)

    assert valid is True
    assert reason is RetrievalInputInvalidReason.VALID
    assert np.isfinite(snr)


def test_nonpositive_bins_inside_search_are_allowed_when_another_window_passes() -> None:
    signal = np.array([np.nan, 5.0, 4.0, 3.0, 2.0, -0.1, -0.2, -0.3, np.nan])
    error = np.array([np.nan, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, np.nan])

    valid, reason, _ = _evaluate(signal, error)

    assert valid is True
    assert reason is RetrievalInputInvalidReason.VALID


def test_search_is_rejected_when_no_window_reaches_minimum_valid_fraction() -> None:
    signal = np.array([np.nan, 5.0, -1.0, -1.0, 2.0, -1.0, -1.0, 0.8, np.nan])
    error = np.full(signal.shape, 0.1)
    error[[0, -1]] = np.nan

    valid, reason, snr = _evaluate(signal, error)

    assert valid is False
    assert reason is RetrievalInputInvalidReason.NONPOSITIVE_SIGNAL
    assert np.isnan(snr)


def test_saturation_outside_best_reference_window_does_not_invalidate_pc_candidate() -> None:
    altitude = np.arange(9, dtype=np.float64) * 100.0
    signal = np.array([np.nan, 5.0, 4.0, 3.0, 2.0, 1.0, 0.8, 0.6, np.nan])
    error = np.array([np.nan, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, np.nan])
    saturation = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    valid, reason, _ = evaluate_retrieval_input_supported_domain(
        signal,
        error,
        altitude,
        _fit_config(),
        correction_valid=True,
        saturation_fraction=saturation,
        require_saturation_diagnostic=True,
    )

    assert valid is True
    assert reason is RetrievalInputInvalidReason.VALID


def test_saturation_rejects_pc_when_every_reference_window_is_contaminated() -> None:
    altitude = np.arange(9, dtype=np.float64) * 100.0
    signal = np.array([np.nan, 5.0, 4.0, 3.0, 2.0, 1.0, 0.8, 0.6, np.nan])
    error = np.array([np.nan, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, np.nan])
    saturation = np.ones_like(signal)

    valid, reason, _ = evaluate_retrieval_input_supported_domain(
        signal,
        error,
        altitude,
        _fit_config(),
        correction_valid=True,
        saturation_fraction=saturation,
        require_saturation_diagnostic=True,
    )

    assert valid is False
    assert reason is RetrievalInputInvalidReason.PHOTON_COUNTING_SATURATED


def test_backward_kfs_stops_at_reference_and_does_not_require_noisy_upper_tail() -> None:
    altitude = np.arange(8, dtype=np.float64) * 100.0
    rcs = np.array([np.nan, 1.2, 1.1, 1.0, 0.9, 0.8, -0.1, np.nan])
    beta_mol = np.full_like(rcs, 1.0e-6)

    beta_aer = fernald_inversion(
        rcs=rcs,
        altitude=altitude,
        beta_mol=beta_mol,
        lidar_ratio_aerosol=50.0,
        beta_total_ref=1.0e-6,
        ref_idx=5,
        altitude_units="m",
        min_lidar_ratio=10.0,
        allow_negative_aerosol=False,
        mode="backward",
    )

    assert np.isnan(beta_aer[0])
    assert np.isfinite(beta_aer[1:6]).all()
    assert np.isnan(beta_aer[6:]).all()
