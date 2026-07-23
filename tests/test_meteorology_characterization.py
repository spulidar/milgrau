"""Characterization of the pre-SCI-004A meteorological and molecular path."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from milgrau.io import radiosonde as radiosonde_io
from milgrau.level1 import thermodynamics as level1_thermodynamics
from milgrau.level2.atmosphere import get_standard_atmosphere
from milgrau.level2.molecular import calculate_molecular_profile
from milgrau.level2.retrieval import build_thermodynamic_profile


@pytest.mark.parametrize(
    ("measurement_hour", "expected_day", "expected_hour"),
    [(8, 5, 0), (9, 5, 12), (20, 5, 12), (21, 6, 0)],
)
def test_existing_wyoming_selection_uses_fixed_nominal_windows(
    tmp_path, monkeypatch, measurement_hour: int, expected_day: int, expected_hour: int
) -> None:
    requested: list[datetime] = []

    def fake_request(target: datetime, station_id: str) -> pd.DataFrame:
        requested.append(target)
        assert station_id == "83779"
        return pd.DataFrame(
            {
                "height": [1000.0, 1000.0, 2000.0],
                "pressure": [900.0, 899.0, 800.0],
                "temperature": [20.0, 19.0, 10.0],
                "dewpoint": [15.0, 14.0, 5.0],
                "u_wind": [1.0, 2.0, 3.0],
            }
        )

    monkeypatch.setattr(radiosonde_io.WyomingUpperAir, "request_data", fake_request)

    result = radiosonde_io.fetch_wyoming_radiosonde(
        datetime(2026, 7, 5, measurement_hour, tzinfo=UTC),
        "83779",
        logging.getLogger("test-characterization"),
        cache_dir=tmp_path,
    )

    assert requested == [datetime(2026, 7, expected_day, expected_hour, tzinfo=UTC)]
    assert result is not None
    assert result["height"].tolist() == [1000.0, 2000.0]
    assert {"dewpoint", "u_wind"}.issubset(result.columns)


def test_existing_level1_contract_discards_humidity_and_extended_siphon_payload(
    monkeypatch,
) -> None:
    sounding = pd.DataFrame(
        {
            "height": [1000.0, 2000.0, 3000.0],
            "pressure": [900.0, 800.0, 700.0],
            "temperature": [20.0, 10.0, 0.0],
            "dewpoint": [15.0, 5.0, -5.0],
            "relative_humidity": [70.0, 65.0, 60.0],
            "u_wind": [1.0, 2.0, 3.0],
        }
    )
    monkeypatch.setattr(
        level1_thermodynamics,
        "fetch_wyoming_radiosonde",
        lambda *_args, **_kwargs: sounding.copy(),
    )
    dataset = xr.Dataset(coords={"time": [np.datetime64("2026-07-05T12:00:00")]})

    normalized = level1_thermodynamics.integrate_thermodynamics(
        dataset,
        {"radiosonde": {"station_id": "83779"}},
        logging.getLogger("test-characterization"),
    )

    assert {
        "radiosonde_altitude",
        "Radiosonde_Temperature_K",
        "Radiosonde_Pressure_hPa",
    }.issubset(normalized.variables)
    assert not any("humid" in name.lower() or "dew" in name.lower() for name in normalized.variables)
    assert "u_wind" not in normalized.variables
    assert normalized.attrs["radiosonde_station_id"] == "83779"


def test_existing_level2_uses_linear_pressure_and_standard_fallback_outside_sounding() -> None:
    station_altitude_m = 760.0
    lidar_altitude_agl_m = np.array([0.0, 240.0, 490.0, 740.0, 1240.0, 1740.0])
    sounding_altitude_asl_m = np.array([1000.0, 1500.0, 2000.0])
    dataset = xr.Dataset(
        data_vars={
            "Radiosonde_Temperature_K": (("radiosonde_altitude",), [290.0, 280.0, 270.0]),
            "Radiosonde_Pressure_hPa": (("radiosonde_altitude",), [900.0, 800.0, 700.0]),
            "Radiosonde_Dewpoint_K": (("radiosonde_altitude",), [285.0, 275.0, 265.0]),
        },
        coords={"radiosonde_altitude": sounding_altitude_asl_m},
    )
    config = {"site": {"station_altitude_m": station_altitude_m}}

    pressure_hpa, temperature_k, source = build_thermodynamic_profile(
        dataset, lidar_altitude_agl_m, config
    )
    standard_pressure, standard_temperature = get_standard_atmosphere(
        lidar_altitude_agl_m + station_altitude_m
    )

    assert source == "radiosonde_with_standard_fallback"
    assert pressure_hpa[2] == pytest.approx(850.0)
    assert temperature_k[2] == pytest.approx(285.0)
    np.testing.assert_allclose(pressure_hpa[[0, 5]], standard_pressure[[0, 5]])
    np.testing.assert_allclose(temperature_k[[0, 5]], standard_temperature[[0, 5]])


def test_existing_standard_atmosphere_clips_temperature_but_keeps_one_layer_pressure_law() -> None:
    altitude_m = np.array([11000.0, 20000.0, 32000.0])

    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_m)

    assert temperature_k.tolist() == pytest.approx([216.65, 216.65, 216.65])
    assert np.all(np.diff(pressure_hpa) < 0.0)
    tropospheric_exponent = (9.80665 * 0.0289644) / (8.3144598 * 0.0065)
    expected = 1013.25 * np.maximum(1.0 - 0.0065 * altitude_m / 288.15, 1e-6) ** tropospheric_exponent
    np.testing.assert_allclose(pressure_hpa, expected, rtol=0.0, atol=0.0)


def test_existing_bucholtz_path_freezes_stp_355_and_532_values() -> None:
    temperature_k = np.array([288.15])
    pressure_hpa = np.array([1013.25])

    beta_355, alpha_355 = calculate_molecular_profile(temperature_k, pressure_hpa, 355.0)
    beta_532, alpha_532 = calculate_molecular_profile(temperature_k, pressure_hpa, 532.0)

    assert alpha_355[0] == pytest.approx(7.018551883285203e-5, rel=1e-13)
    assert beta_355[0] == pytest.approx(8.253563067620594e-6, rel=1e-13)
    assert alpha_532[0] == pytest.approx(1.3157042566010397e-5, rel=1e-13)
    assert beta_532[0] == pytest.approx(1.5485020564165925e-6, rel=1e-13)
    assert alpha_355[0] / alpha_532[0] == pytest.approx(5.334444916532207, rel=1e-13)
