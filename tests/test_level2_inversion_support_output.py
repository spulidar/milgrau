"""Schema-facing tests for altitude-resolved backward inversion support."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from milgrau.level2.support import assemble_level2_inversion_support
from milgrau.scientific import LEVEL2_PRODUCT_SCHEMA_VERSION


def _result(
    altitude: np.ndarray,
    *,
    reference_indices: tuple[int, ...],
    valid_ranges: tuple[tuple[int, int], ...],
) -> SimpleNamespace:
    n_block = len(reference_indices)
    n_altitude = altitude.size
    beta = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    beta_error = np.full_like(beta, np.nan)
    alpha = np.full_like(beta, np.nan)
    alpha_error = np.full_like(beta, np.nan)
    for block_index, (start, stop) in enumerate(valid_ranges):
        beta[block_index, start:stop] = 2.0e-6
        beta_error[block_index, start:stop] = 2.0e-7
        alpha[block_index, start:stop] = 1.0e-4
        alpha_error[block_index, start:stop] = 1.0e-5

    supported_any = np.any(np.isfinite(beta), axis=0)
    aggregate_beta = np.where(supported_any, 2.0e-6, np.nan)
    aggregate_beta_error = np.where(supported_any, 2.0e-7, np.nan)
    aggregate_alpha = np.where(supported_any, 1.0e-4, np.nan)
    aggregate_alpha_error = np.where(supported_any, 1.0e-5, np.nan)

    return SimpleNamespace(
        optical=SimpleNamespace(
            retrieval_success_flag=np.ones(n_block, dtype=np.int8),
            aerosol_backscatter_block=beta,
            aerosol_backscatter_error_block=beta_error,
            aerosol_extinction_block=alpha,
            aerosol_extinction_error_block=alpha_error,
            aerosol_backscatter=aggregate_beta,
            aerosol_backscatter_error=aggregate_beta_error,
            aerosol_extinction=aggregate_alpha,
            aerosol_extinction_error=aggregate_alpha_error,
        ),
        rayleigh=SimpleNamespace(
            reference_altitude_m_block=np.asarray(
                [altitude[index] for index in reference_indices], dtype=np.float64
            )
        ),
    )


def test_support_reports_altitude_resolved_block_count_and_distinct_block_tops() -> None:
    altitude = np.arange(100.0, 700.0, 100.0)
    result = _result(
        altitude,
        reference_indices=(4, 3),
        valid_ranges=((0, 5), (0, 4)),
    )

    support = assemble_level2_inversion_support([result], altitude)

    assert LEVEL2_PRODUCT_SCHEMA_VERSION == "2"
    assert support.flag.shape == (1, 6)
    assert support.flag_block.shape == (2, 1, 6)
    assert support.flag[0].tolist() == [1, 1, 1, 1, 1, 0]
    assert support.effective_block_count[0].tolist() == [2, 2, 2, 2, 1, 0]
    assert support.top_altitude_m.tolist() == [500.0]
    assert support.bottom_altitude_m.tolist() == [100.0]
    assert support.top_altitude_m_block[:, 0].tolist() == [500.0, 400.0]
    assert support.bottom_altitude_m_block[:, 0].tolist() == [100.0, 100.0]


def test_aggregate_support_does_not_bridge_disjoint_vertical_domains() -> None:
    altitude = np.arange(100.0, 700.0, 100.0)
    result = _result(
        altitude,
        reference_indices=(4, 1),
        valid_ranges=((3, 5), (0, 2)),
    )

    support = assemble_level2_inversion_support([result], altitude)

    # Block support exists below and above a true unsupported gap at 300 m.
    assert support.effective_block_count[0].tolist() == [1, 1, 0, 1, 1, 0]
    # Aggregate support is the contiguous path reaching the highest supported top;
    # it never jumps across the 300 m gap to promote the lower segment.
    assert support.flag[0].tolist() == [0, 0, 0, 1, 1, 0]
    assert support.bottom_altitude_m.tolist() == [400.0]
    assert support.top_altitude_m.tolist() == [500.0]


def test_common_optical_support_rejects_missing_extinction_uncertainty() -> None:
    altitude = np.arange(100.0, 600.0, 100.0)
    result = _result(
        altitude,
        reference_indices=(4,),
        valid_ranges=((0, 5),),
    )
    result.optical.aerosol_extinction_error_block[0, 2] = np.nan
    result.optical.aerosol_extinction_error[2] = np.nan

    support = assemble_level2_inversion_support([result], altitude)

    assert support.flag_block[0, 0].tolist() == [0, 0, 0, 1, 1]
    assert support.flag[0].tolist() == [0, 0, 0, 1, 1]
    assert support.bottom_altitude_m.tolist() == [400.0]
