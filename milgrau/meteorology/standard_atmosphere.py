"""Layered U.S. Standard Atmosphere 1976 diagnostic fallback (0-84.852 km H)."""

from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np

from milgrau.meteorology.contracts import (
    AtmosphericProfile,
    FallbackFlag,
    HumidityFlag,
    InterpolationFlag,
    PrimarySource,
    ProfileQuality,
    QualityFlag,
    create_atmospheric_profile,
)
from milgrau.meteorology.thermodynamics import (
    DRY_AIR_GAS_CONSTANT_J_KG_K,
    G0_M_S2,
    geopotential_height_from_geometric_altitude,
)

_BASE_HEIGHT_M = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0])
_BASE_TEMPERATURE_K = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65])
_LAPSE_K_M = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020])
_TOP_GEOPOTENTIAL_HEIGHT_M = 84852.0


def _continuous_base_pressures() -> np.ndarray:
    """Derive each layer base from the previous layer without rounded jumps."""
    pressure = np.empty(_BASE_HEIGHT_M.shape, dtype=np.float64)
    pressure[0] = 101325.0
    for index in range(1, _BASE_HEIGHT_M.size):
        delta_height = _BASE_HEIGHT_M[index] - _BASE_HEIGHT_M[index - 1]
        lapse = _LAPSE_K_M[index - 1]
        temperature_base = _BASE_TEMPERATURE_K[index - 1]
        temperature_top = _BASE_TEMPERATURE_K[index]
        if lapse == 0.0:
            pressure[index] = pressure[index - 1] * np.exp(
                -G0_M_S2
                * delta_height
                / (DRY_AIR_GAS_CONSTANT_J_KG_K * temperature_base)
            )
        else:
            pressure[index] = pressure[index - 1] * (
                temperature_base / temperature_top
            ) ** (G0_M_S2 / (DRY_AIR_GAS_CONSTANT_J_KG_K * lapse))
    return pressure


_BASE_PRESSURE_PA = _continuous_base_pressures()


def standard_atmosphere_state(geometric_altitude_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return pressure [Pa] and temperature [K] on geometric MSL altitude."""
    altitude = np.asarray(geometric_altitude_m, dtype=np.float64)
    if altitude.ndim != 1 or altitude.size < 2 or np.any(~np.isfinite(altitude)):
        raise ValueError("geometric_altitude_m must be a finite 1D array with at least two levels.")
    if np.any(altitude < 0.0):
        raise ValueError("The SCI-004A standard-atmosphere fallback starts at mean sea level.")
    height = geopotential_height_from_geometric_altitude(altitude)
    if np.any(height > _TOP_GEOPOTENTIAL_HEIGHT_M):
        raise ValueError("U.S. Standard Atmosphere layered implementation is limited to H=84.852 km.")
    layer = np.searchsorted(_BASE_HEIGHT_M[1:], height, side="right")
    hb = _BASE_HEIGHT_M[layer]
    tb = _BASE_TEMPERATURE_K[layer]
    pb = _BASE_PRESSURE_PA[layer]
    lapse = _LAPSE_K_M[layer]
    temperature = tb + lapse * (height - hb)
    pressure = np.empty(height.shape, dtype=np.float64)
    isothermal = lapse == 0.0
    pressure[isothermal] = pb[isothermal] * np.exp(
        -G0_M_S2
        * (height[isothermal] - hb[isothermal])
        / (DRY_AIR_GAS_CONSTANT_J_KG_K * tb[isothermal])
    )
    gradient = ~isothermal
    pressure[gradient] = pb[gradient] * (
        tb[gradient] / temperature[gradient]
    ) ** (G0_M_S2 / (DRY_AIR_GAS_CONSTANT_J_KG_K * lapse[gradient]))
    return pressure, temperature


def build_standard_atmosphere_profile(
    geometric_altitude_m: np.ndarray,
    *,
    nominal_time: datetime,
    latitude_deg_north: float,
    longitude_deg_east: float,
) -> AtmosphericProfile:
    altitude = np.asarray(geometric_altitude_m, dtype=np.float64)
    pressure, temperature = standard_atmosphere_state(altitude)
    snapshot = b"US Standard Atmosphere 1976; layers 0-84.852 geopotential km"
    return create_atmospheric_profile(
        geometric_altitude_m=altitude,
        pressure_pa=pressure,
        temperature_k=temperature,
        specific_humidity_kg_kg=np.zeros(altitude.shape, dtype=np.float64),
        primary_source_flag=np.full(
            altitude.shape, int(PrimarySource.STANDARD_ATMOSPHERE), dtype=np.int8
        ),
        interpolation_flag=np.full(
            altitude.shape, int(InterpolationFlag.DIRECT), dtype=np.int8
        ),
        fallback_flag=np.full(
            altitude.shape, int(FallbackFlag.STANDARD_ATMOSPHERE), dtype=np.int8
        ),
        humidity_flag=np.full(
            altitude.shape, int(HumidityFlag.DRY_AIR_ASSUMED), dtype=np.int8
        ),
        radiosonde_weight=np.zeros(altitude.shape, dtype=np.float64),
        quality_flag=np.full(
            altitude.shape, int(QualityFlag.FALLBACK_DIAGNOSTIC), dtype=np.int8
        ),
        nominal_time=nominal_time,
        observation_time=nominal_time,
        latitude_deg_north=latitude_deg_north,
        longitude_deg_east=longitude_deg_east,
        provider="U.S. Standard Atmosphere 1976",
        station_or_dataset_id="USSA1976",
        raw_snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),
        normalizer_version="ussa1976-layers-v1",
        vertical_coverage_m=(float(altitude[0]), float(altitude[-1])),
        profile_quality=ProfileQuality.FALLBACK_DIAGNOSTIC,
        quantitative_retrieval_allowed=False,
    )
