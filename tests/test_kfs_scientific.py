"""Scientific validation of the Klett--Fernald--Sasano inversion.

The primary recovery region excludes five bins at each outer edge, where a
finite sampled domain does not represent an infinite atmosphere.  With a 30 m
grid that is 150 m.  Tolerances are based on the trapezoidal discretization
error measured by the explicit 60 m -> 30 m convergence test below, not on the
historical Level 2 golden product.
"""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.level2.kfs import fernald_inversion, kfs_inversion_monte_carlo
from tests.kfs_forward_model import ElasticSyntheticCase, make_elastic_case

_EDGE_BINS = 5


def _invert(
    case: ElasticSyntheticCase,
    *,
    mode: str,
    allow_negative_aerosol: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run the public deterministic API with the case's exact boundary."""
    return fernald_inversion(
        case.range_corrected_signal,
        case.altitude_m,
        case.molecular_backscatter_m_inv_sr_inv,
        case.aerosol_lidar_ratio_sr,
        case.beta_total_reference_m_inv_sr_inv,
        case.reference_index,
        lr_mol=case.molecular_lidar_ratio_sr,
        altitude_units="m",
        min_lidar_ratio=10.0,
        allow_negative_aerosol=allow_negative_aerosol,
        mode=mode,  # type: ignore[arg-type]
        return_diagnostics=True,
    )


def _relative_l2(retrieved: np.ndarray, truth: np.ndarray, region: slice) -> float:
    delta = retrieved[region] - truth[region]
    return float(np.linalg.norm(delta) / np.linalg.norm(truth[region]))


def _legacy_wrong_sign_backward(case: ElasticSyntheticCase) -> np.ndarray:
    """Reproduce the removed backward sign solely to retain failure evidence."""
    signal = case.range_corrected_signal
    altitude = case.altitude_m
    beta_molecular = case.molecular_backscatter_m_inv_sr_inv
    lidar_ratio = float(case.aerosol_lidar_ratio_sr[0])
    reference = case.reference_index
    result = np.full(signal.shape, np.nan, dtype=np.float64)
    result[reference] = case.beta_total_reference_m_inv_sr_inv - beta_molecular[reference]
    molecular_integral = 0.0
    signal_integral = 0.0
    transformed_previous = signal[reference]
    denominator_at_reference = signal[reference] / case.beta_total_reference_m_inv_sr_inv
    for index in range(reference - 1, -1, -1):
        dz_m = altitude[index + 1] - altitude[index]
        molecular_integral += 0.5 * (beta_molecular[index] + beta_molecular[index + 1]) * dz_m
        # This is the confirmed v1 defect: the positive reverse integral was
        # placed in an exponential with a negative sign.
        transformed = signal[index] * np.exp(
            -2.0 * (lidar_ratio - case.molecular_lidar_ratio_sr) * molecular_integral
        )
        signal_integral += 0.5 * (transformed + transformed_previous) * dz_m
        denominator = denominator_at_reference + 2.0 * lidar_ratio * signal_integral
        result[index] = transformed / denominator - beta_molecular[index]
        transformed_previous = transformed
    return result


@pytest.mark.parametrize("wavelength_nm", [355, 532])
def test_removed_backward_sign_explicitly_fails_known_nonzero_profile(wavelength_nm: int) -> None:
    """Keep quantitative evidence that the pre-v2 equation is not scientific truth."""
    case = make_elastic_case(wavelength_nm)
    legacy = _legacy_wrong_sign_backward(case)
    region = slice(_EDGE_BINS, case.reference_index + 1)

    assert _relative_l2(legacy, case.aerosol_backscatter_m_inv_sr_inv, region) > 0.5


@pytest.mark.parametrize("wavelength_nm", [355, 532])
def test_nonzero_aerosol_backward_recovers_known_profile(wavelength_nm: int) -> None:
    """A noiseless nonzero profile must be recovered below the exact boundary."""
    case = make_elastic_case(wavelength_nm)
    retrieved, diagnostics = _invert(case, mode="backward")
    region = slice(_EDGE_BINS, case.reference_index + 1)

    assert np.all(np.isfinite(retrieved[region]))
    assert diagnostics["backward_valid"] is True
    assert diagnostics["forward_requested"] is False
    # 0.35% covers the measured 30 m trapezoidal error with margin for both
    # molecular profiles; the old-sign implementation is wrong by O(1).
    assert _relative_l2(retrieved, case.aerosol_backscatter_m_inv_sr_inv, region) < 3.5e-3


@pytest.mark.parametrize("wavelength_nm", [355, 532])
def test_nonzero_aerosol_forward_recovers_known_profile(wavelength_nm: int) -> None:
    """The independently oriented forward branch remains correct after SCI-001."""
    case = make_elastic_case(wavelength_nm)
    retrieved, diagnostics = _invert(case, mode="forward")
    region = slice(case.reference_index, -_EDGE_BINS)

    assert np.all(np.isnan(retrieved[: case.reference_index]))
    assert diagnostics["backward_requested"] is False
    assert diagnostics["forward_valid"] is True
    assert _relative_l2(retrieved, case.aerosol_backscatter_m_inv_sr_inv, region) < 3.5e-3


@pytest.mark.parametrize("wavelength_nm", [355, 532])
def test_two_sided_recovers_both_branches_and_exact_reference(wavelength_nm: int) -> None:
    """Both oriented integrals must meet at the one exact reference bin."""
    case = make_elastic_case(wavelength_nm)
    retrieved, diagnostics = _invert(case, mode="two_sided")
    backward = slice(_EDGE_BINS, case.reference_index + 1)
    forward = slice(case.reference_index, -_EDGE_BINS)

    assert retrieved[case.reference_index] == pytest.approx(
        case.aerosol_backscatter_m_inv_sr_inv[case.reference_index],
        rel=0.0,
        abs=1e-18,
    )
    assert _relative_l2(retrieved, case.aerosol_backscatter_m_inv_sr_inv, backward) < 3.5e-3
    assert _relative_l2(retrieved, case.aerosol_backscatter_m_inv_sr_inv, forward) < 3.5e-3
    assert diagnostics["backward_valid"] is True
    assert diagnostics["forward_valid"] is True


def test_two_sided_matches_isolated_branches_without_reference_jump() -> None:
    """Joining writes the exact boundary once and otherwise changes neither side."""
    case = make_elastic_case(532)
    backward, _ = _invert(case, mode="backward")
    forward, _ = _invert(case, mode="forward")
    two_sided, _ = _invert(case, mode="two_sided")

    np.testing.assert_array_equal(two_sided[: case.reference_index + 1], backward[: case.reference_index + 1])
    np.testing.assert_array_equal(two_sided[case.reference_index :], forward[case.reference_index :])
    assert two_sided[case.reference_index] == backward[case.reference_index] == forward[case.reference_index]
    adjacent_error = np.diff(two_sided[case.reference_index - 1 : case.reference_index + 2]) - np.diff(
        case.aerosol_backscatter_m_inv_sr_inv[case.reference_index - 1 : case.reference_index + 2]
    )
    assert np.max(np.abs(adjacent_error)) < 1.0e-11


@pytest.mark.parametrize("wavelength_nm", [355, 532])
@pytest.mark.parametrize("allow_negative", [True, False])
@pytest.mark.parametrize("mode", ["backward", "forward", "two_sided"])
def test_molecular_only_is_zero_without_relying_on_clipping(
    wavelength_nm: int,
    allow_negative: bool,
    mode: str,
) -> None:
    """The analytical molecular solution must not need nonnegative clipping."""
    case = make_elastic_case(wavelength_nm, aerosol=False)
    retrieved, _ = _invert(case, mode=mode, allow_negative_aerosol=allow_negative)
    if mode == "backward":
        valid = slice(_EDGE_BINS, case.reference_index + 1)
    elif mode == "forward":
        valid = slice(case.reference_index, -_EDGE_BINS)
    else:
        valid = slice(_EDGE_BINS, -_EDGE_BINS)
    scale = np.nanmax(case.molecular_backscatter_m_inv_sr_inv[valid])

    assert np.all(np.isfinite(retrieved[valid]))
    assert np.nanmax(np.abs(retrieved[valid])) < scale * 1.0e-4


@pytest.mark.parametrize("allow_negative", [True, False])
def test_positive_recovery_regime_is_not_hidden_by_clipping(allow_negative: bool) -> None:
    """The main aerosol validation stays positive with either clipping policy."""
    case = make_elastic_case(532)
    retrieved, _ = _invert(
        case,
        mode="two_sided",
        allow_negative_aerosol=allow_negative,
    )
    region = slice(_EDGE_BINS, -_EDGE_BINS)

    assert np.nanmin(retrieved[region]) > 0.0
    assert _relative_l2(retrieved, case.aerosol_backscatter_m_inv_sr_inv, region) < 3.5e-3


def test_negative_reference_aerosol_is_explicitly_allowed_or_clipped() -> None:
    """Boundary clipping changes the applied physical boundary, not the equation sign."""
    case = make_elastic_case(532, aerosol=False)
    beta_total_below_molecular = 0.95 * case.molecular_backscatter_m_inv_sr_inv[case.reference_index]
    allowed = fernald_inversion(
        case.range_corrected_signal,
        case.altitude_m,
        case.molecular_backscatter_m_inv_sr_inv,
        case.aerosol_lidar_ratio_sr,
        beta_total_below_molecular,
        case.reference_index,
        altitude_units="m",
        allow_negative_aerosol=True,
    )
    clipped = fernald_inversion(
        case.range_corrected_signal,
        case.altitude_m,
        case.molecular_backscatter_m_inv_sr_inv,
        case.aerosol_lidar_ratio_sr,
        beta_total_below_molecular,
        case.reference_index,
        altitude_units="m",
        allow_negative_aerosol=False,
    )

    assert allowed[case.reference_index] < 0.0
    assert clipped[case.reference_index] == 0.0


@pytest.mark.parametrize("wavelength_nm", [355, 532])
def test_variable_lidar_ratio_profile_uses_generalized_integrals(wavelength_nm: int) -> None:
    """A smooth altitude-varying S_a validates integration and shape handling only."""
    case = make_elastic_case(wavelength_nm, variable_lidar_ratio=True)
    retrieved, diagnostics = _invert(case, mode="two_sided")
    region = slice(_EDGE_BINS, -_EDGE_BINS)

    np.testing.assert_array_equal(diagnostics["lidar_ratio_aerosol_sr"], case.aerosol_lidar_ratio_sr)
    assert _relative_l2(retrieved, case.aerosol_backscatter_m_inv_sr_inv, region) < 3.5e-3


def test_finer_vertical_grid_reduces_recovery_error() -> None:
    """Halving dz should reduce the trapezoidal inversion error."""
    coarse = make_elastic_case(532, vertical_step_m=60.0)
    fine = make_elastic_case(532, vertical_step_m=30.0)
    coarse_result, _ = _invert(coarse, mode="two_sided")
    fine_result, _ = _invert(fine, mode="two_sided")
    coarse_error = _relative_l2(
        coarse_result,
        coarse.aerosol_backscatter_m_inv_sr_inv,
        slice(_EDGE_BINS, -_EDGE_BINS),
    )
    fine_error = _relative_l2(
        fine_result,
        fine.aerosol_backscatter_m_inv_sr_inv,
        slice(_EDGE_BINS, -_EDGE_BINS),
    )

    # Trapezoidal error is second order; the observed ratio is approximately
    # 0.25.  A 0.30 bound permits wavelength-level floating-point variation.
    assert fine_error <= 0.30 * coarse_error


def test_wavelength_changes_molecular_physics_but_not_prescribed_aerosol() -> None:
    """355/532 cases use distinct Bucholtz molecular profiles and one aerosol truth."""
    case_355 = make_elastic_case(355)
    case_532 = make_elastic_case(532)
    retrieved_355, _ = _invert(case_355, mode="two_sided")
    retrieved_532, _ = _invert(case_532, mode="two_sided")

    assert np.all(case_355.molecular_backscatter_m_inv_sr_inv > case_532.molecular_backscatter_m_inv_sr_inv)
    np.testing.assert_array_equal(
        case_355.aerosol_backscatter_m_inv_sr_inv,
        case_532.aerosol_backscatter_m_inv_sr_inv,
    )
    assert not np.array_equal(case_355.range_corrected_signal, case_532.range_corrected_signal)
    np.testing.assert_allclose(
        retrieved_355[_EDGE_BINS:-_EDGE_BINS],
        retrieved_532[_EDGE_BINS:-_EDGE_BINS],
        rtol=6.0e-4,
        atol=1.1e-9,
    )


def test_inputs_are_unchanged_and_output_shape_dtype_are_stable() -> None:
    """The deterministic inversion is side-effect free and returns float64 profiles."""
    case = make_elastic_case(532)
    inputs = [
        case.range_corrected_signal.copy(),
        case.altitude_m.copy(),
        case.molecular_backscatter_m_inv_sr_inv.copy(),
        case.aerosol_lidar_ratio_sr.copy(),
    ]
    snapshots = [value.copy() for value in inputs]
    retrieved = fernald_inversion(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        case.beta_total_reference_m_inv_sr_inv,
        case.reference_index,
        altitude_units="m",
    )

    for value, snapshot in zip(inputs, snapshots, strict=True):
        np.testing.assert_array_equal(value, snapshot)
    assert retrieved.shape == case.altitude_m.shape
    assert retrieved.dtype == np.float64
    assert np.all(np.isfinite(retrieved))


@pytest.mark.parametrize("reference_altitude_m", [360.0, 11940.0])
def test_internal_reference_near_either_edge_keeps_requested_sides(reference_altitude_m: float) -> None:
    """An internal boundary near an edge still has one shared finite reference bin."""
    case = make_elastic_case(532, reference_altitude_m=reference_altitude_m)
    retrieved, diagnostics = _invert(case, mode="two_sided")

    assert diagnostics["backward_valid"] is True
    assert diagnostics["forward_valid"] is True
    assert retrieved[case.reference_index] == pytest.approx(
        case.aerosol_backscatter_m_inv_sr_inv[case.reference_index],
        abs=1e-18,
    )


def test_nonphysical_forward_denominator_invalidates_only_that_side() -> None:
    """A singular forward denominator leaves NaNs instead of borrowing backward values."""
    case = make_elastic_case(532)
    nonphysical_signal = case.range_corrected_signal.copy()
    nonphysical_signal[case.reference_index + 1 :] *= 10.0
    retrieved, diagnostics = fernald_inversion(
        nonphysical_signal,
        case.altitude_m,
        case.molecular_backscatter_m_inv_sr_inv,
        case.aerosol_lidar_ratio_sr,
        case.beta_total_reference_m_inv_sr_inv,
        case.reference_index,
        altitude_units="m",
        mode="two_sided",
        return_diagnostics=True,
    )

    assert diagnostics["backward_valid"] is True
    assert diagnostics["forward_valid"] is False
    assert np.all(np.isfinite(retrieved[: case.reference_index + 1]))
    assert np.any(np.isnan(retrieved[case.reference_index + 1 :]))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nonmonotonic", "strictly increasing"),
        ("negative_boundary", "finite positive total-backscatter"),
        ("reference_outside", "inside the altitude grid"),
        ("reference_signal_invalid", "positive at ref_idx"),
        ("reference_at_edge", "reference bin above the grid bottom"),
        ("lidar_ratio_shape", "scalar or have the same shape"),
    ],
)
def test_invalid_grid_boundary_reference_and_lidar_ratio_are_rejected(
    mutation: str,
    message: str,
) -> None:
    """Invalid scientific inputs fail before entering the compiled kernel."""
    case = make_elastic_case(532)
    altitude = case.altitude_m.copy()
    signal = case.range_corrected_signal.copy()
    boundary = case.beta_total_reference_m_inv_sr_inv
    reference = case.reference_index
    lidar_ratio: float | np.ndarray = case.aerosol_lidar_ratio_sr
    if mutation == "nonmonotonic":
        altitude[20] = altitude[19]
    elif mutation == "negative_boundary":
        boundary = -1.0
    elif mutation == "reference_outside":
        reference = signal.size
    elif mutation == "reference_signal_invalid":
        signal[reference] = np.nan
    elif mutation == "reference_at_edge":
        reference = 0
    elif mutation == "lidar_ratio_shape":
        lidar_ratio = case.aerosol_lidar_ratio_sr[:-1]

    with pytest.raises(ValueError, match=message):
        fernald_inversion(
            signal,
            altitude,
            case.molecular_backscatter_m_inv_sr_inv,
            lidar_ratio,
            boundary,
            reference,
            altitude_units="m",
        )


@pytest.mark.parametrize("wavelength_nm", [355, 532])
def test_partial_monte_carlo_uses_correct_two_sided_kernel_with_controlled_noise(
    wavelength_nm: int,
) -> None:
    """Every seeded realization uses v2 two-sided integration; this is not equation validation."""
    case = make_elastic_case(wavelength_nm)
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
        n_iterations=40,
        rcs_error=5.0e-4 * case.range_corrected_signal,
        beta_ref_relative_std=0.0,
        aerosol_ref_fraction=float(aerosol_reference_fraction),
        altitude_units="m",
        allow_negative_aerosol=True,
        seed=12,
        return_diagnostics=True,
        mode="two_sided",
    )
    beta_mean, beta_std, _, _, diagnostics = result
    region = slice(_EDGE_BINS, -_EDGE_BINS)

    assert diagnostics["mode"] == "two_sided"
    assert diagnostics["uncertainty_scope"] == "partial_monte_carlo_dispersion"
    assert np.all(diagnostics["backward_valid_simulations"])
    assert np.all(diagnostics["forward_valid_simulations"])
    assert _relative_l2(beta_mean, case.aerosol_backscatter_m_inv_sr_inv, region) < 3.5e-3
    assert np.nanmax(beta_std[region]) > 0.0
