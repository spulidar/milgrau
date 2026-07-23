"""SCI-004A offline radiosonde/ERA5 smooth fusion and diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.meteorology.blending import blend_radiosonde_and_era5
from milgrau.meteorology.contracts import PrimarySource


def test_era5_only_returns_complete_era5_profile(era5_reconstruction) -> None:
    target = np.arange(800.0, 20001.0, 200.0)

    result = blend_radiosonde_and_era5(
        None,
        era5_reconstruction.profile,
        target,
        blend_width_m=1000.0,
    )

    assert np.all(result.profile.primary_source_flag == int(PrimarySource.ERA5))
    assert np.all(result.diagnostics.radiosonde_weight == 0.0)
    assert np.all(np.diff(result.profile.pressure_pa) < 0.0)


def test_complete_radiosonde_coverage_remains_pure_radiosonde(
    complete_radiosonde_normalization, era5_reconstruction
) -> None:
    target = np.arange(800.0, 6401.0, 200.0)

    result = blend_radiosonde_and_era5(
        complete_radiosonde_normalization.profile,
        era5_reconstruction.profile,
        target,
        blend_width_m=1000.0,
    )

    assert np.all(result.diagnostics.radiosonde_weight == 1.0)
    assert np.all(result.profile.primary_source_flag == int(PrimarySource.RADIOSONDE))


def test_partial_radiosonde_uses_continuous_cosine_blend_and_era5_above(
    complete_radiosonde_normalization, era5_reconstruction
) -> None:
    target = np.arange(800.0, 20001.0, 200.0)

    result = blend_radiosonde_and_era5(
        complete_radiosonde_normalization.profile,
        era5_reconstruction.profile,
        target,
        blend_width_m=1200.0,
    )
    weight = result.profile.radiosonde_weight

    assert np.any(weight == 1.0)
    assert np.any((weight > 0.0) & (weight < 1.0))
    assert np.any(weight == 0.0)
    assert np.all(np.abs(np.diff(weight)) < 1.0)
    assert np.all(np.diff(result.profile.pressure_pa) < 0.0)
    assert np.any(result.profile.primary_source_flag == int(PrimarySource.BLENDED))
    assert result.diagnostics.blend_start_m < result.diagnostics.blend_end_m


def test_blend_exposes_overlap_differences_and_transition_jumps(
    complete_radiosonde_normalization, era5_reconstruction
) -> None:
    target = np.arange(800.0, 12001.0, 200.0)

    result = blend_radiosonde_and_era5(
        complete_radiosonde_normalization.profile,
        era5_reconstruction.profile,
        target,
        blend_width_m=1200.0,
    )
    diagnostic = result.diagnostics

    assert diagnostic.overlap_thickness_m > 0.0
    assert diagnostic.mean_temperature_difference_k > 0.0
    assert diagnostic.maximum_absolute_pressure_difference_pa > 0.0
    assert diagnostic.maximum_relative_pressure_difference > 0.0
    assert diagnostic.maximum_virtual_temperature_difference_k > 0.0
    assert diagnostic.maximum_molecular_number_density_difference_m3 > 0.0
    assert diagnostic.maximum_temperature_jump_after_k <= diagnostic.maximum_temperature_jump_before_k


def test_short_overlap_still_has_nonzero_smooth_weight(
    complete_radiosonde_normalization, era5_reconstruction
) -> None:
    target = np.arange(5000.0, 7201.0, 200.0)

    result = blend_radiosonde_and_era5(
        complete_radiosonde_normalization.profile,
        era5_reconstruction.profile,
        target,
        blend_width_m=5000.0,
    )

    assert np.any((result.profile.radiosonde_weight > 0.0) & (result.profile.radiosonde_weight < 1.0))
    assert np.all(np.diff(result.profile.pressure_pa) < 0.0)


def test_internal_radiosonde_gaps_are_filled_by_era5_without_hidden_source(
    radiosonde_normalization, era5_reconstruction
) -> None:
    target = np.arange(800.0, 12001.0, 200.0)

    result = blend_radiosonde_and_era5(
        radiosonde_normalization.profile,
        era5_reconstruction.profile,
        target,
        blend_width_m=600.0,
        maximum_radiosonde_gap_m=1000.0,
    )
    gap = (target > 1700.0) & (target < 3400.0)

    assert np.all(result.profile.radiosonde_weight[gap] == 0.0)
    assert np.all(result.profile.primary_source_flag[gap] == int(PrimarySource.ERA5))


def test_blend_rejects_zero_width_and_nonoverlap(
    complete_radiosonde_normalization, era5_reconstruction
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        blend_radiosonde_and_era5(
            complete_radiosonde_normalization.profile,
            era5_reconstruction.profile,
            np.arange(800.0, 2001.0, 200.0),
            blend_width_m=0.0,
        )
    with pytest.raises(ValueError, match="overlapping"):
        blend_radiosonde_and_era5(
            complete_radiosonde_normalization.profile,
            era5_reconstruction.profile,
            np.arange(20000.0, 22001.0, 200.0),
            blend_width_m=1000.0,
        )


def test_blending_does_not_mutate_either_input(
    complete_radiosonde_normalization, era5_reconstruction
) -> None:
    radio = complete_radiosonde_normalization.profile
    era5 = era5_reconstruction.profile
    radio_pressure = radio.pressure_pa.copy()
    era5_pressure = era5.pressure_pa.copy()

    blend_radiosonde_and_era5(
        radio,
        era5,
        np.arange(800.0, 12001.0, 200.0),
        blend_width_m=1000.0,
    )

    np.testing.assert_array_equal(radio.pressure_pa, radio_pressure)
    np.testing.assert_array_equal(era5.pressure_pa, era5_pressure)
