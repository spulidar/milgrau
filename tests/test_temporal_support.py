"""Synthetic tests for future high-column temporal-support diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.level2.temporal_support import (
    contiguous_subwindow_diagnostics,
    temporal_support_diagnostics,
)


def test_stable_blocks_have_full_support_and_balanced_contribution() -> None:
    signal = np.tile(np.array([1.0, 2.0, 3.0]), (4, 1))
    error = np.full_like(signal, 0.1)
    weights = np.ones(4)

    diagnostic = temporal_support_diagnostics(signal, error, weights)

    np.testing.assert_array_equal(diagnostic.supporting_block_count, [4, 4, 4])
    np.testing.assert_allclose(diagnostic.supporting_weight_fraction, 1.0)
    np.testing.assert_allclose(diagnostic.dominant_contribution_fraction, 0.25)
    np.testing.assert_allclose(diagnostic.weighted_mean_signal, [1.0, 2.0, 3.0])


def test_explicit_block_weights_control_temporal_contribution() -> None:
    signal = np.array([[5.0], [5.0]])
    error = np.ones_like(signal)
    weights = np.array([1.0, 2.0])

    diagnostic = temporal_support_diagnostics(signal, error, weights)

    np.testing.assert_allclose(
        diagnostic.contribution_fraction_block[:, 0],
        [1.0 / 3.0, 2.0 / 3.0],
    )
    assert diagnostic.dominant_contribution_fraction[0] == pytest.approx(2.0 / 3.0)
    assert diagnostic.dominant_block_index[0] == 1


def test_transient_far_range_support_is_visible_in_fraction_and_dominance() -> None:
    signal = np.array(
        [
            [10.0, 100.0],
            [10.0, np.nan],
            [10.0, np.nan],
            [10.0, np.nan],
        ]
    )
    error = np.array(
        [
            [1.0, 2.0],
            [1.0, np.nan],
            [1.0, np.nan],
            [1.0, np.nan],
        ]
    )
    weights = np.ones(4)

    diagnostic = temporal_support_diagnostics(signal, error, weights)

    assert diagnostic.supporting_block_count.tolist() == [4, 1]
    np.testing.assert_allclose(diagnostic.supporting_weight_fraction, [1.0, 0.25])
    np.testing.assert_allclose(diagnostic.dominant_contribution_fraction, [0.25, 1.0])
    assert diagnostic.dominant_block_index.tolist() == [0, 0]


def test_full_support_can_still_be_dominated_by_one_temporal_block() -> None:
    signal = np.array([[100.0], [1.0], [1.0], [1.0]])
    error = np.ones_like(signal)

    diagnostic = temporal_support_diagnostics(signal, error, np.ones(4))

    assert diagnostic.supporting_weight_fraction[0] == 1.0
    assert diagnostic.dominant_contribution_fraction[0] == pytest.approx(100.0 / 103.0)
    assert diagnostic.dominant_block_index[0] == 0


def test_missing_uncertainty_removes_temporal_support_without_becoming_zero_error() -> None:
    signal = np.array([[2.0], [4.0], [6.0]])
    error = np.array([[1.0], [np.nan], [1.0]])

    diagnostic = temporal_support_diagnostics(signal, error, np.ones(3))

    assert diagnostic.valid_flag_block[:, 0].tolist() == [1, 0, 1]
    assert diagnostic.supporting_block_count[0] == 2
    assert diagnostic.supporting_weight_fraction[0] == pytest.approx(2.0 / 3.0)
    assert diagnostic.weighted_mean_signal[0] == pytest.approx(4.0)


def test_contiguous_subwindows_expose_early_late_state_change() -> None:
    signal = np.array([[10.0], [10.0], [1.0], [1.0]])
    error = np.ones_like(signal)

    diagnostic = contiguous_subwindow_diagnostics(
        signal,
        error,
        np.ones(4),
        window_blocks=2,
    )

    assert diagnostic.start_block_index.tolist() == [0, 1, 2]
    assert diagnostic.stop_block_index_exclusive.tolist() == [2, 3, 4]
    np.testing.assert_allclose(diagnostic.weighted_mean_signal[:, 0], [10.0, 5.5, 1.0])
    np.testing.assert_allclose(diagnostic.supporting_weight_fraction[:, 0], 1.0)


def test_temporal_diagnostics_reject_hidden_or_invalid_weight_policy() -> None:
    signal = np.ones((2, 3))
    error = np.ones_like(signal)
    with pytest.raises(ValueError, match="strictly positive"):
        temporal_support_diagnostics(signal, error, np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="window_blocks"):
        contiguous_subwindow_diagnostics(
            signal,
            error,
            np.ones(2),
            window_blocks=3,
        )
