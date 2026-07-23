"""Shared local-only fixtures for SCI-004A meteorological validation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from milgrau.meteorology.era5_model_levels import normalize_era5_model_levels
from milgrau.meteorology.radiosonde import normalize_wyoming_radiosonde

_FIXTURE_DIR = Path(__file__).with_name("fixtures")


@pytest.fixture(scope="session")
def radiosonde_fixture_payload() -> tuple[dict, bytes]:
    raw = (_FIXTURE_DIR / "radiosonde_campo_de_marte.json").read_bytes()
    return json.loads(raw), raw


@pytest.fixture(scope="session")
def era5_fixture_payload() -> tuple[dict, bytes]:
    raw = (_FIXTURE_DIR / "era5_l137_four_points.json").read_bytes()
    return json.loads(raw), raw


def _era5_arrays(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    levels = np.linspace(0.0, 1.0, 137, dtype=np.float64)
    temperature_definition = payload["temperature_profile"]
    base_temperature = float(temperature_definition["top_k"]) + (
        float(temperature_definition["surface_k"]) - float(temperature_definition["top_k"])
    ) * levels ** float(temperature_definition["exponent"])
    temperature = base_temperature[:, None] + np.asarray(
        temperature_definition["corner_offsets_k"], dtype=np.float64
    )[None, :]
    humidity_definition = payload["specific_humidity_profile"]
    base_humidity = float(humidity_definition["top_kg_kg"]) + (
        float(humidity_definition["surface_kg_kg"])
        - float(humidity_definition["top_kg_kg"])
    ) * levels ** float(humidity_definition["exponent"])
    humidity = base_humidity[:, None] * np.asarray(
        humidity_definition["corner_scale"], dtype=np.float64
    )[None, :]
    return temperature, humidity


@pytest.fixture()
def era5_reconstruction(era5_fixture_payload):
    payload, raw = era5_fixture_payload
    temperature, humidity = _era5_arrays(payload)
    return normalize_era5_model_levels(
        hybrid_a_pa=np.asarray(payload["hybrid_a_pa"], dtype=np.float64),
        hybrid_b=np.asarray(payload["hybrid_b"], dtype=np.float64),
        temperature_k_by_level_corner=temperature,
        specific_humidity_by_level_corner=humidity,
        logarithm_surface_pressure_by_corner=np.log(
            np.asarray(payload["surface_pressure_pa_by_corner"], dtype=np.float64)
        ),
        surface_geopotential_m2_s2_by_corner=np.asarray(
            payload["surface_geopotential_m2_s2_by_corner"], dtype=np.float64
        ),
        corner_coordinates_lat_lon=np.asarray(
            payload["corner_coordinates_lat_lon"], dtype=np.float64
        ),
        target_latitude_deg_north=float(payload["target_latitude_deg_north"]),
        target_longitude_deg_east=float(payload["target_longitude_deg_east"]),
        analysis_time=datetime.fromisoformat(payload["analysis_time_utc"]),
        dataset_id=str(payload["dataset_id"]),
        raw_snapshot=raw,
    )


@pytest.fixture()
def radiosonde_normalization(radiosonde_fixture_payload):
    payload, raw = radiosonde_fixture_payload
    return normalize_wyoming_radiosonde(
        pd.DataFrame(payload["records"]),
        nominal_time=datetime.fromisoformat(payload["nominal_time_utc"]),
        observation_time=datetime.fromisoformat(payload["observation_time_utc"]),
        station_id=str(payload["station_id"]),
        latitude_deg_north=float(payload["latitude_deg_north"]),
        longitude_deg_east=float(payload["longitude_deg_east"]),
        raw_snapshot=raw,
    )


@pytest.fixture()
def complete_radiosonde_normalization(radiosonde_fixture_payload):
    payload, raw = radiosonde_fixture_payload
    records = [
        record
        for record in payload["records"]
        if record["height"] <= 6500.0 and record["pressure"] is not None
    ]
    return normalize_wyoming_radiosonde(
        pd.DataFrame(records),
        nominal_time=datetime.fromisoformat(payload["nominal_time_utc"]),
        observation_time=datetime.fromisoformat(payload["observation_time_utc"]),
        station_id=str(payload["station_id"]),
        latitude_deg_north=float(payload["latitude_deg_north"]),
        longitude_deg_east=float(payload["longitude_deg_east"]),
        raw_snapshot=raw,
    )
