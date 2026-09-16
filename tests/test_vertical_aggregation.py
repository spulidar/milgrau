"""Tests for explicit vertical-aggregation and covariance diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.level2.vertical_aggregation import (
    aggregate_uniform_vertical_bins,
    autocorrelation_adjusted_snr_gain,
    vertical_noise_autocorrelation,
)


def test_single_bin_aggregation_is_identity() -> None:
    altitude = np.array([3.75, 11.25, 18.75])
    signal = np.array([2.0, 4.0, 6.0])
    error = np.array([0.2, 0.4, 0.6])

    result = aggregate_uniform_vertical_bins(
        signal,
        error,
        altitude,
        bins_per_aggregate=1,
    )

    np.testing.assert_allclose(result.altitude_m, altitude)
    np.testing.assert_allclose(result.signal, signal)
    np.testing.assert_allclose(result.error_independent, error)
    np.testing.assert_allclose(result.error_fully_correlated, error)
    np.testing.assert_array_equal(result.contributing_bin_count, [1, 1, 1])
    assert result.aggregation_width_m == pytest.approx(7.5)


def test_constant_signal_exposes_independent_vs_correlated_noise_gain() -> None:
    altitude = np.arange(8, dtype=float) * 7.5 + 3.75
    signal = np.full(8, 10.0)
    error = np.full(8, 2.0)

    result = aggregate_uniform_vertical_bins(
        signal,
        error,
        altitude,
        bins_per_aggregate=4,
    )

    np.testing.assert_allclose(result.signal, 10.0)
    np.testing.assert_allclose(result.error_independent, 1.0)
    np.testing.assert_allclose(result.error_fully_correlated, 2.0)
    np.testing.assert_array_equal(result.contributing_bin_count, [4, 4])
    assert result.aggregation_width_m == pytest.approx(30.0)


def test_missing_source_bin_invalidates_whole_aggregate_without_bridging() -> None:
    altitude = np.arange(6, dtype=float) * 7.5 + 3.75
    signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    error = np.ones(6)

    result = aggregate_uniform_vertical_bins(
        signal,
        error,
        altitude,
        bins_per_aggregate=3,
    )

    assert np.isnan(result.signal[0])
    assert np.isnan(result.error_independent[0])
    assert np.isnan(result.error_fully_correlated[0])
    assert result.contributing_bin_count[0] == 0
    assert result.signal[1] == pytest.approx(5.0)
    assert result.contributing_bin_count[1] == 3


def test_linear_profile_aggregation_preserves_group_center_value() -> None:
    altitude = np.arange(8, dtype=float) * 7.5 + 3.75
    signal = 2.0 * altitude + 5.0
    error = np.ones_like(signal)

    result = aggregate_uniform_vertical_bins(
        signal,
        error,
        altitude,
        bins_per_aggregate=4,
    )

    np.testing.assert_allclose(result.signal, 2.0 * result.altitude_m + 5.0)


def test_trailing_incomplete_group_is_omitted_not_padded() -> None:
    altitude = np.arange(10, dtype=float) * 7.5 + 3.75
    signal = np.arange(10, dtype=float)
    error = np.ones(10)

    result = aggregate_uniform_vertical_bins(
        signal,
        error,
        altitude,
        bins_per_aggregate=4,
    )

    assert result.signal.size == 2
    np.testing.assert_array_equal(result.source_start_index, [0, 4])
    np.testing.assert_array_equal(result.source_stop_index_exclusive, [4, 8])


def test_autocorrelation_gain_recovers_independent_and_fully_correlated_limits() -> None:
    zero_correlation = np.zeros(7)
    unity_correlation = np.ones(7)

    assert autocorrelation_adjusted_snr_gain(
        zero_correlation,
        bins_per_aggregate=8,
    ) == pytest.approx(np.sqrt(8.0))
    assert autocorrelation_adjusted_snr_gain(
        unity_correlation,
        bins_per_aggregate=8,
    ) == pytest.approx(1.0)


def test_positive_short_range_correlation_reduces_but_does_not_remove_gain() -> None:
    correlation = np.array([0.12, 0.02, 0.01, 0.01, 0.0, 0.0, 0.0])

    gain = autocorrelation_adjusted_snr_gain(
        correlation,
        bins_per_aggregate=8,
    )

    assert 1.0 < gain < np.sqrt(8.0)


def test_vertical_noise_autocorrelation_detects_common_range_residual() -> None:
    # Within each temporal block, one profile is uniformly below and the other
    # uniformly above the block mean.  The normalized residual is therefore
    # perfectly correlated across altitude lags.
    signal = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0, 3.0],
            [2.0, 2.0, 2.0, 2.0],
            [4.0, 4.0, 4.0, 4.0],
        ]
    )
    error = np.ones_like(signal)
    labels = np.array([0, 0, 1, 1])

    diagnostic = vertical_noise_autocorrelation(
        signal,
        error,
        labels,
        max_lag_bins=3,
    )

    np.testing.assert_array_equal(diagnostic.lag_bins, [1, 2, 3])
    np.testing.assert_allclose(diagnostic.correlation, 1.0)
    assert np.all(diagnostic.valid_pair_count > 0)


def test_vertical_aggregation_rejects_nonuniform_grid() -> None:
    altitude = np.array([3.75, 11.25, 20.0, 27.5])
    with pytest.raises(ValueError, match="uniform altitude grid"):
        aggregate_uniform_vertical_bins(
            np.ones(4),
            np.ones(4),
            altitude,
            bins_per_aggregate=2,
        )
