"""Contracts for deterministic method-v5 high-column reference selection."""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.level2.high_column_rnd import (
    catalogue_high_column_reference_cells,
    prepare_high_column_profile,
)
from milgrau.level2.high_column_selector import (
    V5_REFERENCE_SEARCH_MAX_M,
    V5_REFERENCE_SEARCH_MIN_M,
    select_minimum_cost_high_column_reference,
)
from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.physics.atmosphere import get_standard_atmosphere
from tests.kfs_forward_model import elastic_lidar_forward_model


def _catalogue():
    altitude = np.arange(300.0, 30_000.0 + 7.5, 7.5, dtype=np.float64)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude)
    beta_mol, _ = calculate_molecular_profile(temperature_k, pressure_hpa, 532.0)
    beta_aer = 2.0e-6 * np.exp(-(altitude - altitude[0]) / 1800.0)
    beta_aer[altitude >= 7000.0] = 0.0
    signal = elastic_lidar_forward_model(
        altitude,
        beta_mol,
        beta_aer,
        RAYLEIGH_LIDAR_RATIO_SR,
        55.0,
    )
    molecular_signal = elastic_lidar_forward_model(
        altitude,
        beta_mol,
        np.zeros_like(beta_aer),
        RAYLEIGH_LIDAR_RATIO_SR,
        55.0,
    )
    error = np.abs(signal) * 0.002
    prepared = prepare_high_column_profile(
        range_corrected_signal=signal,
        range_corrected_signal_error=error,
        molecular_backscatter=beta_mol,
        altitude_m=altitude,
        uncertainty_mode="independent",
    )
    return catalogue_high_column_reference_cells(
        prepared=prepared,
        native_range_corrected_signal=signal,
        native_range_corrected_signal_error=error,
        native_simulated_molecular_signal=molecular_signal,
        native_altitude_m=altitude,
        search_min_altitude_m=8000.0,
        search_max_altitude_m=25_000.0,
        rayleigh_window_m=1000.0,
        max_relative_slope=0.25,
        max_relative_variance=0.50,
        min_valid_fraction=0.50,
    )


def test_selector_minimizes_existing_rayleigh_cost_after_physical_filters() -> None:
    catalogue = _catalogue()
    eligible = tuple(
        cell
        for cell in catalogue.accepted_and_admissible
        if V5_REFERENCE_SEARCH_MIN_M
        <= cell.altitude_m
        <= V5_REFERENCE_SEARCH_MAX_M
    )
    assert eligible

    selected = select_minimum_cost_high_column_reference(catalogue)
    expected = min(
        eligible,
        key=lambda cell: (
            cell.native_rayleigh_candidate.diagnostic_cost,
            cell.altitude_m,
            cell.cell_index,
        ),
    )
    assert selected == expected
    assert selected.accepted
    assert selected.nominal_path_admissible
    assert V5_REFERENCE_SEARCH_MIN_M <= selected.altitude_m <= V5_REFERENCE_SEARCH_MAX_M


def test_selector_does_not_reward_highest_altitude() -> None:
    catalogue = _catalogue()
    selected = select_minimum_cost_high_column_reference(catalogue)
    highest = max(
        (
            cell
            for cell in catalogue.accepted_and_admissible
            if V5_REFERENCE_SEARCH_MIN_M
            <= cell.altitude_m
            <= V5_REFERENCE_SEARCH_MAX_M
        ),
        key=lambda cell: cell.altitude_m,
    )
    assert selected.native_rayleigh_candidate.diagnostic_cost <= (
        highest.native_rayleigh_candidate.diagnostic_cost
    )
    # This assertion protects selector semantics, not a universal atmospheric law.
    assert selected.altitude_m <= highest.altitude_m


def test_selector_altitude_domain_is_explicit_and_can_be_overridden() -> None:
    catalogue = _catalogue()
    selected_12km = select_minimum_cost_high_column_reference(
        catalogue,
        min_altitude_m=12_000.0,
        max_altitude_m=20_000.0,
    )
    assert 12_000.0 <= selected_12km.altitude_m <= 20_000.0

    with pytest.raises(ValueError, match="No Rayleigh-accepted"):
        select_minimum_cost_high_column_reference(
            catalogue,
            min_altitude_m=29_000.0,
            max_altitude_m=30_000.0,
        )


def test_selector_rejects_invalid_domain() -> None:
    catalogue = _catalogue()
    with pytest.raises(ValueError, match="finite and increasing"):
        select_minimum_cost_high_column_reference(
            catalogue,
            min_altitude_m=12_000.0,
            max_altitude_m=10_000.0,
        )
