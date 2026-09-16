"""Scientific contract tests for the auditable Rayleigh candidate catalogue."""

from __future__ import annotations

import numpy as np

from milgrau.level2.rayleigh_candidates import (
    RayleighCandidateRejection,
    accepted_rayleigh_candidates,
    catalogue_rayleigh_candidates,
)


def _clean_profiles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    altitude = np.arange(0.0, 3000.0, 100.0)
    molecular = np.exp(-altitude / 7000.0)
    measured = 2.5 * molecular
    return altitude, measured, molecular


def _catalogue(measured: np.ndarray) -> tuple:
    altitude, _, molecular = _clean_profiles()
    return catalogue_rayleigh_candidates(
        measured,
        molecular,
        altitude,
        min_altitude_m=500.0,
        max_altitude_m=2500.0,
        window_bins=7,
        max_relative_slope=0.08,
        max_relative_variance=0.01,
        min_valid_fraction=0.80,
    )


def test_clean_molecular_scaling_keeps_all_complete_windows_auditable_and_accepted() -> None:
    altitude, measured, _ = _clean_profiles()

    catalogue = _catalogue(measured)

    assert len(catalogue) > 1
    assert all(candidate.accepted for candidate in catalogue)
    assert all(candidate.rejection_mask == int(RayleighCandidateRejection.NONE) for candidate in catalogue)
    assert all(candidate.start_altitude_m >= 500.0 for candidate in catalogue)
    assert all(candidate.stop_altitude_m <= 2500.0 for candidate in catalogue)
    assert all(candidate.valid_fraction == 1.0 for candidate in catalogue)
    assert all(np.isclose(candidate.calibration_factor, 2.5) for candidate in catalogue)
    assert all(np.isclose(candidate.free_intercept, 0.0, atol=1.0e-12) for candidate in catalogue)
    assert catalogue[0].center_altitude_m < catalogue[-1].center_altitude_m
    assert altitude[catalogue[0].center_index] == catalogue[0].center_altitude_m


def test_high_altitude_ratio_gradient_is_rejected_without_hiding_lower_clean_candidates() -> None:
    altitude, measured, molecular = _clean_profiles()
    contaminated = measured.copy()
    high = altitude >= 1800.0
    contaminated[high] = molecular[high] * (2.5 + 0.003 * (altitude[high] - 1800.0))

    catalogue = _catalogue(contaminated)
    accepted = accepted_rayleigh_candidates(catalogue)
    rejected = tuple(candidate for candidate in catalogue if not candidate.accepted)

    assert accepted
    assert rejected
    assert max(candidate.center_altitude_m for candidate in accepted) < max(
        candidate.center_altitude_m for candidate in catalogue
    )
    assert any(
        candidate.rejection_mask & int(RayleighCandidateRejection.EXCESS_RELATIVE_SLOPE)
        for candidate in rejected
    )
    assert any(candidate.center_altitude_m >= 1800.0 for candidate in rejected)


def test_missing_samples_are_recorded_as_candidate_qa_not_silently_interpolated() -> None:
    altitude, measured, _ = _clean_profiles()
    missing = measured.copy()
    missing[(altitude >= 1200.0) & (altitude <= 1600.0)] = np.nan

    catalogue = _catalogue(missing)
    affected = [
        candidate
        for candidate in catalogue
        if candidate.start_altitude_m <= 1400.0 <= candidate.stop_altitude_m
    ]

    assert affected
    assert any(candidate.valid_fraction < 0.80 for candidate in affected)
    assert any(
        candidate.rejection_mask & int(RayleighCandidateRejection.INSUFFICIENT_VALID_FRACTION)
        for candidate in affected
    )
    assert all(candidate.total_bins == 7 for candidate in catalogue)


def test_catalogue_keeps_rejected_candidates_instead_of_returning_only_one_best_window() -> None:
    altitude, measured, molecular = _clean_profiles()
    modified = measured.copy()
    modified[altitude >= 2000.0] = molecular[altitude >= 2000.0] * 8.0

    catalogue = _catalogue(modified)
    accepted = accepted_rayleigh_candidates(catalogue)

    assert len(catalogue) > len(accepted)
    assert tuple(candidate for candidate in catalogue if candidate.accepted) == accepted
    assert [candidate.center_index for candidate in catalogue] == sorted(
        candidate.center_index for candidate in catalogue
    )
