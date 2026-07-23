"""SCI-004A vertical lidar-grid interpolation behavior."""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.meteorology.contracts import InterpolationFlag, PrimarySource, QualityFlag
from milgrau.meteorology.interpolation import interpolate_profile_to_altitudes


def test_vertical_interpolation_preserves_coincident_grid(complete_radiosonde_normalization) -> None:
    source = complete_radiosonde_normalization.profile

    result = interpolate_profile_to_altitudes(source, source.geometric_altitude_m)

    np.testing.assert_array_equal(result.pressure_pa, source.pressure_pa)
    np.testing.assert_array_equal(result.temperature_k, source.temperature_k)
    assert np.all(result.interpolation_flag == int(InterpolationFlag.DIRECT))


def test_vertical_interpolation_uses_log_pressure_and_linear_temperature(
    complete_radiosonde_normalization,
) -> None:
    source = complete_radiosonde_normalization.profile
    target = np.array([760.0, 880.0, 1000.0, 1300.0, 1600.0, 2550.0, 3500.0, 5000.0, 6500.0])

    result = interpolate_profile_to_altitudes(source, target)

    expected_pressure = np.sqrt(source.pressure_pa[0] * source.pressure_pa[1])
    assert result.pressure_pa[1] == pytest.approx(expected_pressure)
    assert result.temperature_k[1] == pytest.approx(
        0.5 * (source.temperature_k[0] + source.temperature_k[1])
    )
    assert result.interpolation_flag[1] == int(InterpolationFlag.INTERPOLATED)
    assert np.all(np.diff(result.pressure_pa) < 0.0)


def test_vertical_interpolation_marks_outside_bins_missing_without_extrapolation(
    complete_radiosonde_normalization,
) -> None:
    target = np.array([500.0, 760.0, 1000.0, 1600.0, 3500.0, 6500.0, 7000.0])

    result = interpolate_profile_to_altitudes(
        complete_radiosonde_normalization.profile, target
    )

    assert np.isnan(result.pressure_pa[[0, -1]]).all()
    assert np.all(result.primary_source_flag[[0, -1]] == int(PrimarySource.INVALID))
    assert np.all(result.quality_flag[[0, -1]] == int(QualityFlag.INVALID))
    assert result.vertical_coverage_m == (760.0, 6500.0)


def test_vertical_extrapolation_must_be_explicit_and_is_flagged(
    complete_radiosonde_normalization,
) -> None:
    target = np.array([500.0, 760.0, 1000.0, 1600.0, 3500.0, 6500.0, 7000.0])

    result = interpolate_profile_to_altitudes(
        complete_radiosonde_normalization.profile,
        target,
        allow_extrapolation=True,
    )

    assert np.isfinite(result.pressure_pa).all()
    assert result.interpolation_flag[0] == int(InterpolationFlag.EXTRAPOLATED)
    assert result.interpolation_flag[-1] == int(InterpolationFlag.EXTRAPOLATED)


def test_internal_large_gaps_are_preserved_as_absence(
    complete_radiosonde_normalization,
) -> None:
    target = np.array([760.0, 1000.0, 1600.0, 2000.0, 2500.0, 3000.0, 3500.0, 6500.0])

    result = interpolate_profile_to_altitudes(
        complete_radiosonde_normalization.profile,
        target,
        maximum_gap_m=1000.0,
    )

    assert np.isnan(result.pressure_pa[3:6]).all()
    assert np.all(result.interpolation_flag[3:6] == int(InterpolationFlag.INVALID))


def test_vertical_interpolation_rejects_nonincreasing_lidar_grid(
    complete_radiosonde_normalization,
) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        interpolate_profile_to_altitudes(
            complete_radiosonde_normalization.profile,
            np.array([760.0, 1000.0, 900.0]),
        )


def test_vertical_interpolation_does_not_mutate_source(complete_radiosonde_normalization) -> None:
    source = complete_radiosonde_normalization.profile
    pressure_before = source.pressure_pa.copy()

    interpolate_profile_to_altitudes(source, np.linspace(760.0, 6500.0, 20))

    np.testing.assert_array_equal(source.pressure_pa, pressure_before)
