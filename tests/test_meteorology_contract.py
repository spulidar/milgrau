"""SCI-004A normalized atmospheric contract and thermodynamic invariants."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from milgrau.meteorology.contracts import (
    FallbackFlag,
    HumidityFlag,
    InterpolationFlag,
    PrimarySource,
    ProfileQuality,
    QualityFlag,
)
from milgrau.meteorology.standard_atmosphere import build_standard_atmosphere_profile
from milgrau.meteorology.thermodynamics import (
    geometric_altitude_from_geopotential,
    geopotential_from_geometric_altitude,
    thermodynamic_state,
    virtual_temperature,
)


def _standard_profile():
    return build_standard_atmosphere_profile(
        np.array([0.0, 1000.0, 3000.0, 6000.0]),
        nominal_time=datetime(2026, 7, 5, 12, tzinfo=UTC),
        latitude_deg_north=-23.5615,
        longitude_deg_east=-46.7383,
    )


def test_contract_is_immutable_and_does_not_alias_input_arrays() -> None:
    altitude = np.array([0.0, 1000.0, 3000.0, 6000.0])
    profile = build_standard_atmosphere_profile(
        altitude,
        nominal_time=datetime(2026, 7, 5, 12, tzinfo=UTC),
        latitude_deg_north=-23.5615,
        longitude_deg_east=-46.7383,
    )
    altitude[:] = -1.0

    assert np.all(profile.geometric_altitude_m >= 0.0)
    with pytest.raises(ValueError, match="read-only"):
        profile.pressure_pa[0] = 1.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("geometric_altitude_m", np.array([0.0, 1000.0, 900.0, 6000.0]), "increasing"),
        ("pressure_pa", np.array([101325.0, 0.0, 70000.0, 50000.0]), "positive"),
        ("pressure_pa", np.array([101325.0, 90000.0, 91000.0, 50000.0]), "decreasing"),
        ("temperature_k", np.array([288.0, -1.0, 260.0, 250.0]), "positive"),
        ("specific_humidity_kg_kg", np.array([0.0, 0.01, 0.2, 0.0]), "within"),
    ],
)
def test_contract_rejects_invalid_shapes_domains_and_vertical_order(
    field: str, value: np.ndarray, message: str
) -> None:
    profile = _standard_profile()
    with pytest.raises(ValueError, match=message):
        replace(profile, **{field: value})


def test_contract_rejects_contradictory_source_fallback_and_weight_flags() -> None:
    profile = _standard_profile()
    with pytest.raises(ValueError, match="must agree"):
        replace(profile, fallback_flag=np.zeros(4, dtype=np.int8))
    with pytest.raises(ValueError, match="radiosonde_weight=0"):
        replace(profile, radiosonde_weight=np.ones(4))
    source = np.full(4, int(PrimarySource.RADIOSONDE), dtype=np.int8)
    with pytest.raises(ValueError, match="radiosonde_weight=1"):
        replace(
            profile,
            primary_source_flag=source,
            fallback_flag=np.zeros(4, dtype=np.int8),
            radiosonde_weight=np.zeros(4),
        )


def test_contract_rejects_invalid_operation_quality_and_weight_states() -> None:
    profile = _standard_profile()
    with pytest.raises(ValueError, match="interpolation operation"):
        replace(
            profile,
            interpolation_flag=np.full(4, int(InterpolationFlag.INVALID), dtype=np.int8),
        )
    source = np.full(4, int(PrimarySource.RADIOSONDE), dtype=np.int8)
    no_fallback = np.full(4, int(FallbackFlag.NONE), dtype=np.int8)
    with pytest.raises(ValueError, match="valid quality"):
        replace(
            profile,
            primary_source_flag=source,
            fallback_flag=no_fallback,
            radiosonde_weight=np.ones(4),
            quality_flag=np.full(4, int(QualityFlag.INVALID), dtype=np.int8),
            profile_quality=ProfileQuality.QUANTITATIVE,
            quantitative_retrieval_allowed=True,
        )


def test_contract_rejects_weight_on_missing_bins() -> None:
    profile = _standard_profile()
    missing = np.array([False, False, True, True])
    updates = {
        "pressure_pa": np.where(missing, np.nan, profile.pressure_pa),
        "temperature_k": np.where(missing, np.nan, profile.temperature_k),
        "specific_humidity_kg_kg": np.where(
            missing, np.nan, profile.specific_humidity_kg_kg
        ),
        "primary_source_flag": np.where(
            missing, int(PrimarySource.INVALID), profile.primary_source_flag
        ),
        "interpolation_flag": np.where(
            missing, int(InterpolationFlag.INVALID), profile.interpolation_flag
        ),
        "fallback_flag": np.where(
            missing, int(FallbackFlag.NONE), profile.fallback_flag
        ),
        "humidity_flag": np.where(
            missing, int(HumidityFlag.MISSING), profile.humidity_flag
        ),
        "quality_flag": np.where(missing, int(QualityFlag.INVALID), profile.quality_flag),
        "radiosonde_weight": np.where(missing, 0.5, profile.radiosonde_weight),
        "vertical_coverage_m": (0.0, 1000.0),
        "profile_quality": ProfileQuality.INCOMPLETE,
    }
    for field in (
        "virtual_temperature_k",
        "air_density_kg_m3",
        "molecular_number_density_m3",
        "dry_air_mass_density_kg_m3",
        "water_vapor_mass_density_kg_m3",
        "dry_air_number_density_m3",
        "water_vapor_number_density_m3",
    ):
        updates[field] = np.where(missing, np.nan, getattr(profile, field))
    with pytest.raises(ValueError, match="radiosonde_weight=0"):
        replace(profile, **updates)


def test_contract_requires_timezone_hash_and_minimum_metadata() -> None:
    profile = _standard_profile()
    with pytest.raises(TypeError, match="timezone-aware"):
        replace(profile, nominal_time=datetime(2026, 7, 5, 12))
    with pytest.raises(ValueError, match="SHA-256"):
        replace(profile, raw_snapshot_sha256="not-a-digest")
    with pytest.raises(TypeError, match="provider"):
        replace(profile, provider="")


def test_xarray_adapter_persists_si_units_and_provenance() -> None:
    dataset = _standard_profile().to_xarray()

    assert dataset["pressure_pa"].attrs["units"] == "Pa"
    assert dataset["molecular_number_density_m3"].attrs["units"] == "m-3"
    assert dataset.attrs["vertical_coordinate"] == "geometric altitude above mean sea level"
    assert dataset.attrs["quantitative_retrieval_allowed"] == 0


def test_standard_atmosphere_semantics_block_quantitative_retrieval() -> None:
    profile = _standard_profile()

    assert profile.profile_quality is ProfileQuality.FALLBACK_DIAGNOSTIC
    assert not profile.quantitative_retrieval_allowed
    assert np.all(profile.primary_source_flag == int(PrimarySource.STANDARD_ATMOSPHERE))
    assert np.all(profile.fallback_flag == int(FallbackFlag.STANDARD_ATMOSPHERE))
    assert np.all(profile.humidity_flag == int(HumidityFlag.DRY_AIR_ASSUMED))


def test_moist_thermodynamics_distinguishes_mass_and_number_density() -> None:
    pressure = np.array([95000.0])
    temperature = np.array([295.0])
    dry = thermodynamic_state(pressure, temperature, np.array([0.0]))
    moist = thermodynamic_state(pressure, temperature, np.array([0.015]))

    assert moist.virtual_temperature_k[0] > temperature[0]
    assert moist.air_density_kg_m3[0] < dry.air_density_kg_m3[0]
    assert moist.water_vapor_mass_density_kg_m3[0] > 0.0
    assert moist.dry_air_mass_density_kg_m3[0] < dry.dry_air_mass_density_kg_m3[0]
    assert moist.molecular_number_density_m3[0] == pytest.approx(
        dry.molecular_number_density_m3[0], rel=2e-16
    )


def test_virtual_temperature_uses_exact_specific_humidity_mixture_relation() -> None:
    result = virtual_temperature(np.array([300.0]), np.array([0.02]))

    assert result[0] == pytest.approx(303.6478873239437, rel=2e-5)


def test_geometric_and_geopotential_conversion_round_trip() -> None:
    altitude = np.array([0.0, 760.0, 11019.067832, 50000.0])
    recovered = geometric_altitude_from_geopotential(
        geopotential_from_geometric_altitude(altitude)
    )

    np.testing.assert_allclose(recovered, altitude, rtol=2e-16, atol=1e-9)
    assert _standard_profile().height_above_station(760.0).tolist() == pytest.approx(
        [-760.0, 240.0, 2240.0, 5240.0]
    )
