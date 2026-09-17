"""Tests for explicit, non-ranking KFS boundary-condition sensitivity R&D."""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.level2.boundary_sensitivity import (
    boundary_fraction_sensitivity_profiles,
)
from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.kfs import fernald_inversion
from tests.kfs_forward_model import make_elastic_case


def test_boundary_fraction_sensitivity_matches_explicit_fernald_boundaries() -> None:
    """Each declared fraction must mean exactly beta_total=(1+f)*beta_mol."""
    case = make_elastic_case(
        532,
        vertical_step_m=30.0,
        aerosol=False,
        reference_altitude_m=9000.0,
    )
    fractions = np.array([0.0, 0.05, 0.20])
    result = boundary_fraction_sensitivity_profiles(
        rcs=case.range_corrected_signal,
        altitude_m=case.altitude_m,
        beta_mol=case.molecular_backscatter_m_inv_sr_inv,
        reference_index=case.reference_index,
        aerosol_lidar_ratio_sr=55.0,
        residual_fractions=fractions,
    )

    np.testing.assert_allclose(
        result.beta_total_reference,
        case.molecular_backscatter_m_inv_sr_inv[case.reference_index]
        * (1.0 + fractions),
    )
    assert result.aerosol_backscatter.shape == (
        fractions.size,
        case.altitude_m.size,
    )
    for index, fraction in enumerate(fractions):
        expected = fernald_inversion(
            case.range_corrected_signal,
            case.altitude_m,
            case.molecular_backscatter_m_inv_sr_inv,
            55.0,
            float(
                case.molecular_backscatter_m_inv_sr_inv[case.reference_index]
                * (1.0 + fraction)
            ),
            case.reference_index,
            lr_mol=RAYLEIGH_LIDAR_RATIO_SR,
            altitude_units="m",
            min_lidar_ratio=10.0,
            allow_negative_aerosol=False,
            mode="backward",
        )
        np.testing.assert_allclose(
            result.aerosol_backscatter[index],
            expected,
            equal_nan=True,
        )
        assert result.aerosol_backscatter[index, case.reference_index] == pytest.approx(
            fraction
            * case.molecular_backscatter_m_inv_sr_inv[case.reference_index]
        )


def test_boundary_fraction_sensitivity_rejects_invalid_scenario_vectors() -> None:
    case = make_elastic_case(532, aerosol=False, reference_altitude_m=9000.0)
    common = {
        "rcs": case.range_corrected_signal,
        "altitude_m": case.altitude_m,
        "beta_mol": case.molecular_backscatter_m_inv_sr_inv,
        "reference_index": case.reference_index,
        "aerosol_lidar_ratio_sr": 55.0,
    }
    with pytest.raises(ValueError, match="non-negative"):
        boundary_fraction_sensitivity_profiles(
            **common,
            residual_fractions=[0.0, -0.1],
        )
    with pytest.raises(ValueError, match="duplicate"):
        boundary_fraction_sensitivity_profiles(
            **common,
            residual_fractions=[0.0, 0.0],
        )
