"""Contracts for explicit high-column background-offset bracketing."""

from __future__ import annotations

import numpy as np

from milgrau.level2.background_validation import (
    estimate_background_offset_scale,
    perturb_rcs_by_raw_background_offset,
)


def test_background_scale_is_estimated_in_raw_equivalent_space() -> None:
    altitude = np.arange(1000.0, 30_000.0 + 30.0, 30.0)
    raw = np.zeros_like(altitude)
    band = (altitude >= 29_000.0) & (altitude <= 30_000.0)
    raw[band] = np.tile(np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]), 7)[: np.sum(band)]
    rcs = raw * altitude**2
    scale = estimate_background_offset_scale(rcs, altitude)
    assert scale.valid_bins == np.count_nonzero(band)
    assert abs(scale.raw_equivalent_median) < 1.0e-12
    assert scale.raw_equivalent_mad_sigma > 0.0


def test_background_offset_maps_to_range_square_rcs_change() -> None:
    altitude = np.asarray([1000.0, 2000.0, 3000.0])
    signal = np.asarray([4.0, 5.0, 6.0])
    perturbed = perturb_rcs_by_raw_background_offset(signal, altitude, 2.0)
    np.testing.assert_allclose(perturbed - signal, 2.0 * altitude**2)
