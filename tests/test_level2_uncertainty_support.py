"""Regression tests for Level 2 value/uncertainty support semantics."""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.level2.block_average import (
    mean_and_correlated_error_bound,
    mean_and_error_of_mean,
    mean_error_by_groups,
    valid_block_mean_and_error,
)
from milgrau.level2.kfs import kfs_inversion_monte_carlo
from milgrau.scientific import (
    LEVEL2_RETRIEVAL_METHOD_VERSION,
    elastic_inversion_algorithm_metadata,
)
from tests.kfs_forward_model import make_elastic_case


def test_mean_and_error_share_one_common_sample_mask() -> None:
    """A value without finite uncertainty cannot enter the reported mean."""
    values = np.array([[10.0], [100.0], [14.0]])
    errors = np.array([[1.0], [np.nan], [1.0]])

    mean, mean_error, n_effective = mean_and_error_of_mean(values, errors)

    assert mean[0] == pytest.approx(12.0)
    assert mean_error[0] == pytest.approx(np.sqrt(2.0) / 2.0)
    assert n_effective[0] == 2


def test_negative_or_missing_uncertainty_removes_same_value_sample() -> None:
    """Invalid one-sigma values are unsupported rather than zero uncertainty."""
    values = np.array([[10.0], [20.0], [30.0], [40.0]])
    errors = np.array([[2.0], [-1.0], [np.nan], [2.0]])

    mean, mean_error, n_effective = mean_and_error_of_mean(values, errors)

    assert mean[0] == pytest.approx(25.0)
    assert mean_error[0] == pytest.approx(np.sqrt(8.0) / 2.0)
    assert n_effective[0] == 2


def test_grouped_reduction_exposes_effective_sample_count() -> None:
    values = np.array(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [5.0, 30.0],
            [7.0, 40.0],
        ]
    )
    errors = np.array(
        [
            [1.0, 1.0],
            [1.0, np.nan],
            [1.0, 1.0],
            [np.nan, 1.0],
        ]
    )
    groups = [np.array([0, 1]), np.array([2, 3])]

    means, mean_errors, counts = mean_error_by_groups(values, errors, groups)

    np.testing.assert_array_equal(counts, np.array([[2, 1], [1, 2]]))
    np.testing.assert_allclose(means, np.array([[2.0, 10.0], [5.0, 35.0]]))
    assert np.all(np.isfinite(mean_errors))


def test_correlated_error_bound_does_not_gain_inverse_sqrt_n() -> None:
    """Unknown/shared block covariance must not be silently reduced as independent noise."""
    values = np.array([[10.0], [12.0], [14.0]])
    errors = np.array([[2.0], [2.0], [2.0]])

    mean, mean_error, n_effective = mean_and_correlated_error_bound(values, errors)

    assert mean[0] == pytest.approx(12.0)
    assert mean_error[0] == pytest.approx(2.0)
    assert mean_error[0] > 2.0 / np.sqrt(3.0)
    assert n_effective[0] == 3


def test_correlated_error_bound_uses_common_value_error_support() -> None:
    values = np.array([[10.0], [100.0], [14.0]])
    errors = np.array([[1.0], [np.nan], [3.0]])

    mean, mean_error, n_effective = mean_and_correlated_error_bound(values, errors)

    assert mean[0] == pytest.approx(12.0)
    assert mean_error[0] == pytest.approx(2.0)
    assert n_effective[0] == 2


def test_aggregate_value_and_error_share_block_support_and_correlation_policy() -> None:
    values = np.array([[10.0], [100.0], [14.0]])
    errors = np.array([[1.0], [np.nan], [1.0]])
    accepted = np.array([True, True, True])

    mean, mean_error, n_effective = valid_block_mean_and_error(
        values, errors, accepted
    )

    assert mean[0] == pytest.approx(12.0)
    assert mean_error[0] == pytest.approx(1.0)
    assert n_effective[0] == 2


def _run_backward_mc(rcs_error: np.ndarray):
    case = make_elastic_case(532)
    aerosol_reference_fraction = (
        case.aerosol_backscatter_m_inv_sr_inv[case.reference_index]
        / case.molecular_backscatter_m_inv_sr_inv[case.reference_index]
    )
    result = kfs_inversion_monte_carlo(
        case.range_corrected_signal,
        case.altitude_m,
        case.molecular_backscatter_m_inv_sr_inv,
        lr_base=55.0,
        lr_std=0.0,
        ref_idx=case.reference_index,
        n_iterations=12,
        rcs_error=rcs_error,
        beta_ref_relative_std=0.0,
        aerosol_ref_fraction=float(aerosol_reference_fraction),
        altitude_units="m",
        allow_negative_aerosol=True,
        seed=7,
        return_diagnostics=True,
        mode="backward",
    )
    return case, result


def test_missing_rcs_uncertainty_invalidates_backward_support_instead_of_becoming_zero() -> None:
    """Missing uncertainty must stop the MC-supported backward branch."""
    case = make_elastic_case(532)
    rcs_error = np.zeros_like(case.range_corrected_signal)
    missing_index = case.reference_index - 10
    rcs_error[missing_index] = np.nan

    case, result = _run_backward_mc(rcs_error)
    beta_mean, beta_std, _, _, diagnostics = result

    assert diagnostics["used_rcs_noise"] is True
    assert diagnostics["rcs_uncertainty_support"][missing_index] == 0
    assert diagnostics["backward_uncertainty_complete"] is False
    assert diagnostics["backward_valid"] is False
    assert np.all(np.isnan(beta_mean[: missing_index + 1]))
    assert np.all(np.isnan(beta_std[: missing_index + 1]))
    assert np.all(np.isfinite(beta_mean[missing_index + 1 : case.reference_index + 1]))


def test_zero_uncertainty_is_supported_and_distinct_from_missing_uncertainty() -> None:
    """An explicit sigma=0 remains valid; NaN sigma is a different scientific state."""
    case = make_elastic_case(532)
    _, result = _run_backward_mc(np.zeros_like(case.range_corrected_signal))
    beta_mean, beta_std, _, _, diagnostics = result

    assert diagnostics["backward_uncertainty_complete"] is True
    assert diagnostics["backward_valid"] is True
    assert np.all(np.isfinite(beta_mean[: case.reference_index + 1]))
    assert np.all(np.isfinite(beta_std[: case.reference_index + 1]))


def test_uncertainty_dependence_policy_is_machine_readable() -> None:
    metadata = elastic_inversion_algorithm_metadata()

    assert metadata["optical_block_uncertainty_correlation_policy"] == (
        "fully_correlated_upper_bound"
    )
    assert metadata["optical_block_uncertainty_aggregation_formula"] == (
        "sigma_mean=sum(sigma_block)/n_effective on common value/error support"
    )
    assert "lidar-ratio nuisance shared across blocks" in metadata[
        "uncertainty_component_dependence"
    ]
    assert "slope/intercept uncertainty excluded" in metadata[
        "gluing_uncertainty_scope"
    ]


def test_uncertainty_semantics_are_versioned_as_level2_method_v3() -> None:
    assert LEVEL2_RETRIEVAL_METHOD_VERSION == "3"
