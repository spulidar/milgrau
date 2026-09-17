"""P5.4 R&D covariance families for vertical-aggregation precision gain.

These tests do not select a productive aggregation width or infer a real SPU
covariance law. They only preserve how the declared SNR gain changes under
explicit positive-correlation families for the 60 m (8-bin) and 120 m (16-bin)
experiments.
"""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.level2.vertical_aggregation import autocorrelation_adjusted_snr_gain


@pytest.mark.parametrize("bins_per_aggregate", [8, 16])
def test_equicorrelation_family_monotonically_removes_aggregation_gain(
    bins_per_aggregate: int,
) -> None:
    """Positive all-lag correlation must continuously erase naive sqrt(N) gain."""
    rhos = (0.0, 0.1, 0.3, 0.6, 0.9, 1.0)
    gains = []
    for rho in rhos:
        autocorrelation = np.full(bins_per_aggregate - 1, rho, dtype=np.float64)
        gains.append(
            autocorrelation_adjusted_snr_gain(
                autocorrelation,
                bins_per_aggregate=bins_per_aggregate,
            )
        )

    assert np.isclose(gains[0], np.sqrt(bins_per_aggregate))
    assert np.isclose(gains[-1], 1.0)
    assert np.all(np.diff(gains) < 0.0)
    assert all(gain >= 1.0 for gain in gains)


@pytest.mark.parametrize("bins_per_aggregate", [8, 16])
def test_exponential_lag_correlation_family_reduces_gain_without_assuming_full_correlation(
    bins_per_aggregate: int,
) -> None:
    """An explicit rho**lag family spans intermediate covariance behavior."""
    lag = np.arange(1, bins_per_aggregate, dtype=np.float64)
    rhos = (0.0, 0.1, 0.3, 0.6, 0.9)
    gains = []
    for rho in rhos:
        autocorrelation = rho**lag
        gains.append(
            autocorrelation_adjusted_snr_gain(
                autocorrelation,
                bins_per_aggregate=bins_per_aggregate,
            )
        )

    assert np.isclose(gains[0], np.sqrt(bins_per_aggregate))
    assert np.all(np.diff(gains) < 0.0)
    assert gains[-1] > 1.0


def test_60m_and_120m_gain_order_can_shrink_under_strong_correlation() -> None:
    """More source bins do not imply a proportionally larger gain when correlated."""
    rho = 0.9
    gain_60m = autocorrelation_adjusted_snr_gain(
        np.full(7, rho, dtype=np.float64),
        bins_per_aggregate=8,
    )
    gain_120m = autocorrelation_adjusted_snr_gain(
        np.full(15, rho, dtype=np.float64),
        bins_per_aggregate=16,
    )

    # Independent-bin gains would be sqrt(8) and 4. Under strong declared
    # correlation both approach one, so doubling width yields little extra
    # precision and cannot by itself justify the coarser resolution.
    assert 1.0 < gain_60m < 1.10
    assert 1.0 < gain_120m < 1.10
    assert gain_120m / gain_60m < 1.02
