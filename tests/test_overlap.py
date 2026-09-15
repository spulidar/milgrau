"""Tests for the diagnostic geometrical overlap model and station ownership."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from milgrau.config.loader import load_config
from milgrau.config.station import resolve_station_context
from milgrau.physics.overlap import (
    coaxial_full_overlap_range_m,
    coaxial_geometric_overlap,
    receiver_field_stop_diameter_m,
)


def _context(config: dict, when: str) -> dict:
    return resolve_station_context(
        config,
        datetime.fromisoformat(when).replace(tzinfo=timezone.utc),
        "pm",
        ["355.AN", "355.PC", "532.AN", "532.PC"],
    )


def test_full_overlap_formula_for_well_conditioned_coaxial_geometry() -> None:
    result = coaxial_full_overlap_range_m(
        telescope_diameter_m=0.30,
        laser_beam_diameter_m=0.04,
        receiver_fov_full_angle_mrad=0.80,
        laser_divergence_full_angle_mrad=0.10,
    )

    assert result == pytest.approx(485.7142857142857)


def test_equal_receiver_fov_and_laser_divergence_has_no_finite_full_overlap() -> None:
    result = coaxial_full_overlap_range_m(
        telescope_diameter_m=0.30,
        laser_beam_diameter_m=0.04,
        receiver_fov_full_angle_mrad=0.10,
        laser_divergence_full_angle_mrad=0.10,
    )

    assert result is None


def test_geometric_overlap_is_bounded_and_increases_for_current_nominal_case() -> None:
    altitude = np.array([0.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0])
    overlap = coaxial_geometric_overlap(
        altitude,
        telescope_diameter_m=0.30,
        laser_beam_diameter_m=0.04,
        receiver_fov_full_angle_mrad=0.10,
        laser_divergence_full_angle_mrad=0.10,
    )

    assert np.all((0.0 <= overlap) & (overlap <= 1.0))
    assert np.all(np.diff(overlap) >= 0.0)
    assert overlap[2] == pytest.approx(1.0 / 36.0, rel=1e-8)
    assert overlap[-1] < 1.0


def test_reported_spu_full_overlap_is_kept_as_unvalidated_conflicting_evidence() -> None:
    config = load_config("config.yaml")
    context = _context(config, "2025-01-01T15:00:00")
    overlap = context["overlap"]

    assert context["profile_id"] == "spu-merionc-2024"
    assert overlap["status"] == "estimated_unvalidated"
    assert overlap["correction_applied"] is False
    assert overlap["parameter_source"] == "profile"
    assert overlap["model_full_overlap_altitude_m"] is None
    assert overlap["reported_full_overlap"]["altitude_m_agl"] == 500.0
    assert overlap["reported_vs_model_relation"] == "model_has_no_finite_full_overlap"
    assert overlap["model_overlap_fraction_at_reported_altitude"] == pytest.approx(1.0 / 36.0, rel=1e-8)


def test_historical_profile_uses_explicit_station_transmitter_fallback() -> None:
    config = load_config("config.yaml")
    context = _context(config, "2019-06-01T15:00:00")

    assert context["profile_id"] == "spu-raman-2018"
    assert context["overlap"]["parameter_source"] == "station_fallback"
    assert context["overlap"]["transmitter"]["beam_diameter_status"] == "estimated_unverified"
    assert context["overlap"]["correction_policy"] == "diagnostic_only_no_correction"


def test_nominal_fov_implies_015_mm_field_stop_under_full_angle_assumption() -> None:
    diameter = receiver_field_stop_diameter_m(focal_length_m=1.5, fov_full_angle_mrad=0.1)

    assert diameter == pytest.approx(0.00015)
