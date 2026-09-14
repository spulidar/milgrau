"""Regression tests for Level 2 contiguous retrieval-input support."""

from __future__ import annotations

import numpy as np

import milgrau.level2._retrieval_impl as _impl
from milgrau.level2.contracts import RetrievalInputInvalidReason
from milgrau.level2.kfs import fernald_inversion
from milgrau.level2.retrieval_input_qa import evaluate_retrieval_input_supported_domain


def _fit_config() -> dict[str, float]:
    return {"ref_alt_min_m": 300.0, "ref_alt_max_m": 500.0}


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


def test_level2_installs_supported_domain_qa_into_legacy_retrieval_impl() -> None:
    assert _impl._evaluate_retrieval_input is evaluate_retrieval_input_supported_domain


def test_bin_shift_style_edge_nans_are_allowed() -> None:
    signal = np.array([np.nan, 5.0, 4.0, 3.0, 2.0, 1.0, 0.8, np.nan, np.nan])
    error = np.array([np.nan, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, np.nan, np.nan])

    valid, reason, snr = _evaluate(signal, error)

    assert valid is True
    assert reason is RetrievalInputInvalidReason.VALID
    assert np.isfinite(snr)


def test_nonpositive_high_altitude_edge_is_allowed_without_filling() -> None:
    signal = np.array([np.nan, 5.0, 4.0, 3.0, 2.0, 1.0, -0.1, -0.2, np.nan])
    error = np.array([np.nan, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, np.nan])

    valid, reason, _ = _evaluate(signal, error)

    assert valid is True
    assert reason is RetrievalInputInvalidReason.VALID


def test_internal_nan_between_valid_samples_is_rejected() -> None:
    signal = np.array([np.nan, 5.0, 4.0, 3.0, 2.0, 1.0, np.nan, 0.5, np.nan])
    error = np.array([np.nan, 0.1, 0.1, 0.1, 0.1, 0.1, np.nan, 0.1, np.nan])

    valid, reason, snr = _evaluate(signal, error)

    assert valid is False
    assert reason is RetrievalInputInvalidReason.NONFINITE_SIGNAL
    assert np.isnan(snr)


def test_nan_inside_rayleigh_reference_is_rejected() -> None:
    signal = np.array([np.nan, 5.0, 4.0, 3.0, np.nan, 1.0, 0.8, np.nan, np.nan])
    error = np.array([np.nan, 0.1, 0.1, 0.1, np.nan, 0.1, 0.1, np.nan, np.nan])

    valid, reason, snr = _evaluate(signal, error)

    assert valid is False
    assert reason is RetrievalInputInvalidReason.NONFINITE_SIGNAL
    assert np.isnan(snr)


def test_saturation_outside_supported_edge_does_not_invalidate_pc_candidate() -> None:
    altitude = np.arange(9, dtype=np.float64) * 100.0
    signal = np.array([np.nan, 5.0, 4.0, 3.0, 2.0, 1.0, 0.8, np.nan, np.nan])
    error = np.array([np.nan, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, np.nan, np.nan])
    saturation = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

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


def test_kfs_keeps_edge_bins_nan_outside_contiguous_support() -> None:
    altitude = np.arange(8, dtype=np.float64) * 100.0
    rcs = np.array([np.nan, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, np.nan])
    beta_mol = np.full_like(rcs, 1.0e-6)

    beta_aer = fernald_inversion(
        rcs=rcs,
        altitude=altitude,
        beta_mol=beta_mol,
        lidar_ratio_aerosol=50.0,
        beta_total_ref=1.0e-6,
        ref_idx=4,
        altitude_units="m",
        min_lidar_ratio=10.0,
        allow_negative_aerosol=False,
        mode="two_sided",
    )

    assert np.isnan(beta_aer[0])
    assert np.isnan(beta_aer[-1])
    assert np.isfinite(beta_aer[1:-1]).all()
