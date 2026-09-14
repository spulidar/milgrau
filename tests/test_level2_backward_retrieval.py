"""Regression tests for productive backward-only optical aggregation."""

from __future__ import annotations

import numpy as np

from milgrau.level2.optical_retrieval import _reaggregate_backward_optical_products
from milgrau.level2.contracts import KfsDiagnostics, OpticalProducts, RayleighDiagnostics


def _rayleigh() -> RayleighDiagnostics:
    return RayleighDiagnostics(
        reference_altitude_m=8000.0,
        reference_start_altitude_m=7500.0,
        reference_stop_altitude_m=8500.0,
        reference_valid_bins=100,
        reference_success_flag=1,
        reference_relative_slope=0.02,
        reference_relative_variance=0.01,
        reference_valid_fraction=0.95,
        calibration_factor=2.0,
        calibration_intercept=0.0,
        reference_altitude_m_block=np.array([8000.0, 8100.0]),
        reference_start_altitude_m_block=np.array([7500.0, 7600.0]),
        reference_stop_altitude_m_block=np.array([8500.0, 8600.0]),
        reference_valid_bins_block=np.array([100, 100], dtype=np.int32),
        reference_success_flag_block=np.array([1, 1], dtype=np.int8),
        reference_relative_slope_block=np.array([0.02, 0.03]),
        reference_relative_variance_block=np.array([0.01, 0.02]),
        reference_valid_fraction_block=np.array([0.95, 0.90]),
        calibration_factor_block=np.array([2.0, 2.1]),
        calibration_intercept_block=np.array([0.0, 0.0]),
    )


def _kfs() -> KfsDiagnostics:
    return KfsDiagnostics(
        lidar_ratio_assumed_sr=50.0,
        lidar_ratio_std_sr=5.0,
        backward_valid_flag=0,
        forward_valid_flag=0,
        backward_valid_flag_block=np.array([1, 0], dtype=np.int8),
        forward_valid_flag_block=np.array([0, 0], dtype=np.int8),
        branch=np.array([1, 1, 2], dtype=np.int8),
        branch_block=np.array([[1, 1, 2], [1, 1, 2]], dtype=np.int8),
    )


def _optical() -> OpticalProducts:
    block = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    error = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])
    return OpticalProducts(
        scattering_ratio_mean=np.full(3, np.nan),
        scattering_ratio_block=block.copy(),
        aerosol_backscatter=np.full(3, np.nan),
        aerosol_backscatter_error=np.full(3, np.nan),
        aerosol_extinction=np.full(3, np.nan),
        aerosol_extinction_error=np.full(3, np.nan),
        aerosol_backscatter_block=block.copy(),
        aerosol_backscatter_error_block=error.copy(),
        aerosol_extinction_block=(block * 50.0),
        aerosol_extinction_error_block=(error * 50.0),
        retrieval_success_flag=np.zeros(2, dtype=np.int8),
    )


def test_backward_success_does_not_require_forward_branch() -> None:
    updated, valid = _reaggregate_backward_optical_products(
        _optical(), _rayleigh(), _kfs()
    )

    np.testing.assert_array_equal(valid, np.array([True, False]))
    np.testing.assert_array_equal(
        updated.retrieval_success_flag, np.array([1, 0], dtype=np.int8)
    )
    np.testing.assert_allclose(updated.aerosol_backscatter, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(
        updated.aerosol_backscatter_error, np.array([0.1, 0.2, 0.3])
    )
