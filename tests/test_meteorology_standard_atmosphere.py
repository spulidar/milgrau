"""SCI-004A layered U.S. Standard Atmosphere 1976 fallback validation."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from milgrau.meteorology.contracts import ProfileQuality
from milgrau.meteorology.standard_atmosphere import (
    build_standard_atmosphere_profile,
    standard_atmosphere_state,
)
from milgrau.meteorology.thermodynamics import geometric_altitude_from_geopotential_height


@pytest.mark.parametrize(
    ("geopotential_height_m", "temperature_k", "pressure_pa"),
    [
        (0.0, 288.15, 101325.0),
        (11000.0, 216.65, 22632.06),
        (20000.0, 216.65, 5474.889),
        (32000.0, 228.65, 868.0187),
        (47000.0, 270.65, 110.9063),
        (51000.0, 270.65, 66.93887),
        (71000.0, 214.65, 3.956420),
    ],
)
def test_standard_atmosphere_matches_published_layer_base_points(
    geopotential_height_m: float, temperature_k: float, pressure_pa: float
) -> None:
    geometric = geometric_altitude_from_geopotential_height(
        np.array([geopotential_height_m, geopotential_height_m + 0.01])
    )

    pressure, temperature = standard_atmosphere_state(geometric)

    assert temperature[0] == pytest.approx(temperature_k, abs=2e-8)
    assert pressure[0] == pytest.approx(pressure_pa, rel=1.2e-4)


def test_standard_atmosphere_is_continuous_across_layer_transitions() -> None:
    for height in (11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0):
        geometric = geometric_altitude_from_geopotential_height(
            np.array([height - 0.01, height, height + 0.01])
        )
        pressure, temperature = standard_atmosphere_state(geometric)
        assert abs(pressure[2] - pressure[0]) / pressure[1] < 4e-6
        assert abs(temperature[2] - temperature[0]) < 1e-3


def test_standard_atmosphere_profile_is_diagnostic_fallback_only() -> None:
    profile = build_standard_atmosphere_profile(
        np.array([0.0, 1000.0, 5000.0]),
        nominal_time=datetime(2026, 7, 5, 12, tzinfo=UTC),
        latitude_deg_north=-23.5615,
        longitude_deg_east=-46.7383,
    )

    assert profile.profile_quality is ProfileQuality.FALLBACK_DIAGNOSTIC
    assert not profile.quantitative_retrieval_allowed


def test_standard_atmosphere_rejects_altitudes_above_layered_domain() -> None:
    with pytest.raises(ValueError, match="84.852"):
        standard_atmosphere_state(np.array([0.0, 100000.0]))
