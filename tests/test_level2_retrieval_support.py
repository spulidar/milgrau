"""Synthetic contract tests for altitude-resolved backward retrieval support."""

from __future__ import annotations

import numpy as np

from milgrau.level2.support import backward_retrieval_support


def _profiles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    altitude = np.arange(10, dtype=np.float64) * 100.0
    value = np.linspace(2.0e-6, 0.5e-6, altitude.size)
    uncertainty = np.full(altitude.shape, 0.1e-6)
    return altitude, value, uncertainty


def test_support_respects_known_lower_instrument_and_upper_inversion_boundaries() -> None:
    altitude, value, uncertainty = _profiles()
    instrument_valid = altitude >= 300.0

    support = backward_retrieval_support(
        altitude,
        value,
        uncertainty,
        upper_index=8,
        instrument_valid=instrument_valid,
    )

    np.testing.assert_array_equal(
        support.flag,
        np.array([False, False, False, True, True, True, True, True, True, False]),
    )
    assert support.bottom_altitude_m == 300.0
    assert support.top_altitude_m == 800.0


def test_noisy_or_missing_tail_above_reference_does_not_reduce_backward_support() -> None:
    altitude, value, uncertainty = _profiles()
    value[8:] = np.nan
    uncertainty[8:] = np.nan

    support = backward_retrieval_support(
        altitude,
        value,
        uncertainty,
        upper_index=7,
    )

    np.testing.assert_array_equal(
        support.flag,
        np.array([True, True, True, True, True, True, True, True, False, False]),
    )
    assert support.bottom_altitude_m == 0.0
    assert support.top_altitude_m == 700.0


def test_missing_uncertainty_inside_backward_path_cuts_off_lower_bins() -> None:
    altitude, value, uncertainty = _profiles()
    uncertainty[5] = np.nan

    support = backward_retrieval_support(
        altitude,
        value,
        uncertainty,
        upper_index=8,
    )

    np.testing.assert_array_equal(
        support.flag,
        np.array([False, False, False, False, False, False, True, True, True, False]),
    )
    assert support.bottom_altitude_m == 600.0
    assert support.top_altitude_m == 800.0


def test_internal_value_gap_is_not_bridged() -> None:
    altitude, value, uncertainty = _profiles()
    value[4] = np.nan

    support = backward_retrieval_support(
        altitude,
        value,
        uncertainty,
        upper_index=8,
    )

    np.testing.assert_array_equal(
        support.flag,
        np.array([False, False, False, False, False, True, True, True, True, False]),
    )
    assert support.bottom_altitude_m == 500.0
    assert support.top_altitude_m == 800.0


def test_negative_uncertainty_is_unsupported_and_breaks_path() -> None:
    altitude, value, uncertainty = _profiles()
    uncertainty[6] = -1.0

    support = backward_retrieval_support(
        altitude,
        value,
        uncertainty,
        upper_index=8,
    )

    np.testing.assert_array_equal(
        support.flag,
        np.array([False, False, False, False, False, False, False, True, True, False]),
    )
    assert support.bottom_altitude_m == 700.0
    assert support.top_altitude_m == 800.0


def test_unsupported_upper_boundary_produces_no_scientific_support() -> None:
    altitude, value, uncertainty = _profiles()
    uncertainty[8] = np.nan

    support = backward_retrieval_support(
        altitude,
        value,
        uncertainty,
        upper_index=8,
    )

    assert not np.any(support.flag)
    assert np.isnan(support.bottom_altitude_m)
    assert np.isnan(support.top_altitude_m)
