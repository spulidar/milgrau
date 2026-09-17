"""Lightweight contracts for the block-resolved method-v5 R&D profile runner."""

from __future__ import annotations

import numpy as np

from milgrau.level2.v5_profile_cli import (
    _finite_mean,
    _period_support,
    _profile_output_path,
)


def test_period_support_counts_only_finite_block_retrievals() -> None:
    profiles = np.array(
        [
            [1.0, 2.0, np.nan, np.nan],
            [3.0, np.nan, 5.0, np.nan],
            [7.0, 8.0, 9.0, np.nan],
        ],
        dtype=np.float64,
    )
    count, fraction = _period_support(profiles)
    np.testing.assert_array_equal(count, np.array([3, 2, 2, 0], dtype=np.int32))
    np.testing.assert_allclose(fraction, np.array([1.0, 2 / 3, 2 / 3, 0.0]))


def test_finite_mean_matches_altitude_resolved_support() -> None:
    profiles = np.array(
        [
            [1.0, 2.0, np.nan],
            [3.0, np.nan, np.nan],
            [5.0, 8.0, np.nan],
        ],
        dtype=np.float64,
    )
    mean = _finite_mean(profiles, axis=0)
    np.testing.assert_allclose(mean[:2], np.array([3.0, 5.0]))
    assert np.isnan(mean[2])


def test_profile_output_path_is_explicitly_rnd(tmp_path) -> None:
    path = _profile_output_path(
        tmp_path,
        tmp_path / "20240902sant_level1_rcs.nc",
        532,
    )
    assert path.name == "20240902sant_level1_rcs_method_v5_rnd_532nm.nc"
