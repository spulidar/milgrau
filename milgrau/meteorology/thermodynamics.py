"""Small SI thermodynamic kernel shared by offline profile normalizers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G0_M_S2 = 9.80665
EARTH_EFFECTIVE_RADIUS_M = 6_356_766.0
UNIVERSAL_GAS_CONSTANT_J_MOL_K = 8.31446261815324
DRY_AIR_GAS_CONSTANT_J_KG_K = 287.05
WATER_VAPOR_GAS_CONSTANT_J_KG_K = 461.5
DRY_AIR_MOLAR_MASS_KG_MOL = UNIVERSAL_GAS_CONSTANT_J_MOL_K / DRY_AIR_GAS_CONSTANT_J_KG_K
WATER_MOLAR_MASS_KG_MOL = UNIVERSAL_GAS_CONSTANT_J_MOL_K / WATER_VAPOR_GAS_CONSTANT_J_KG_K
AVOGADRO_MOL_INV = 6.02214076e23
EPSILON = DRY_AIR_GAS_CONSTANT_J_KG_K / WATER_VAPOR_GAS_CONSTANT_J_KG_K
THERMODYNAMIC_FORMULA_VERSION = "moist-ideal-mixture-v1"


@dataclass(frozen=True, slots=True)
class ThermodynamicState:
    virtual_temperature_k: np.ndarray
    air_density_kg_m3: np.ndarray
    molecular_number_density_m3: np.ndarray
    dry_air_mass_density_kg_m3: np.ndarray
    water_vapor_mass_density_kg_m3: np.ndarray
    dry_air_number_density_m3: np.ndarray
    water_vapor_number_density_m3: np.ndarray


def virtual_temperature(temperature_k: np.ndarray, specific_humidity_kg_kg: np.ndarray) -> np.ndarray:
    """Return exact ideal-mixture virtual temperature for specific humidity q."""
    temperature = np.asarray(temperature_k, dtype=np.float64)
    humidity = np.asarray(specific_humidity_kg_kg, dtype=np.float64)
    if temperature.shape != humidity.shape:
        raise ValueError("temperature_k and specific_humidity_kg_kg must have equal shapes.")
    return temperature * (1.0 + humidity * (1.0 / EPSILON - 1.0))


def thermodynamic_state(
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    specific_humidity_kg_kg: np.ndarray,
) -> ThermodynamicState:
    """Compute moist-air mass and molecular number densities without mutation."""
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    humidity = np.asarray(specific_humidity_kg_kg, dtype=np.float64)
    if not (pressure.shape == temperature.shape == humidity.shape):
        raise ValueError("Pressure, temperature and specific humidity must have equal shapes.")
    valid = (
        np.isfinite(pressure)
        & np.isfinite(temperature)
        & np.isfinite(humidity)
        & (pressure > 0.0)
        & (temperature > 0.0)
        & (humidity >= 0.0)
        & (humidity <= 0.1)
    )
    tv = np.full(pressure.shape, np.nan, dtype=np.float64)
    density = np.full(pressure.shape, np.nan, dtype=np.float64)
    tv[valid] = virtual_temperature(temperature[valid], humidity[valid])
    density[valid] = pressure[valid] / (DRY_AIR_GAS_CONSTANT_J_KG_K * tv[valid])
    vapor_mass = density * humidity
    dry_mass = density * (1.0 - humidity)
    dry_number = dry_mass / DRY_AIR_MOLAR_MASS_KG_MOL * AVOGADRO_MOL_INV
    vapor_number = vapor_mass / WATER_MOLAR_MASS_KG_MOL * AVOGADRO_MOL_INV
    total_number = dry_number + vapor_number
    return ThermodynamicState(
        virtual_temperature_k=tv,
        air_density_kg_m3=density,
        molecular_number_density_m3=total_number,
        dry_air_mass_density_kg_m3=dry_mass,
        water_vapor_mass_density_kg_m3=vapor_mass,
        dry_air_number_density_m3=dry_number,
        water_vapor_number_density_m3=vapor_number,
    )


def saturation_vapor_pressure_pa(dewpoint_k: np.ndarray) -> np.ndarray:
    """Bolton liquid-water expression evaluated at dew-point temperature."""
    dewpoint = np.asarray(dewpoint_k, dtype=np.float64)
    dewpoint_c = dewpoint - 273.15
    return 611.2 * np.exp(17.67 * dewpoint_c / (dewpoint_c + 243.5))


def specific_humidity_from_dewpoint(
    pressure_pa: np.ndarray,
    dewpoint_k: np.ndarray,
) -> np.ndarray:
    """Convert dew point and total pressure to specific humidity."""
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    dewpoint = np.asarray(dewpoint_k, dtype=np.float64)
    if pressure.shape != dewpoint.shape:
        raise ValueError("pressure_pa and dewpoint_k must have equal shapes.")
    vapor_pressure = saturation_vapor_pressure_pa(dewpoint)
    if np.any(vapor_pressure >= pressure):
        raise ValueError("Dew-point vapor pressure must remain below total pressure.")
    mixing_ratio = EPSILON * vapor_pressure / (pressure - vapor_pressure)
    return mixing_ratio / (1.0 + mixing_ratio)


def geopotential_height_from_geometric_altitude(geometric_altitude_m: np.ndarray) -> np.ndarray:
    altitude = np.asarray(geometric_altitude_m, dtype=np.float64)
    if np.any(~np.isfinite(altitude)) or np.any(altitude <= -EARTH_EFFECTIVE_RADIUS_M):
        raise ValueError("geometric_altitude_m contains an invalid value.")
    return EARTH_EFFECTIVE_RADIUS_M * altitude / (EARTH_EFFECTIVE_RADIUS_M + altitude)


def geometric_altitude_from_geopotential_height(geopotential_height_m: np.ndarray) -> np.ndarray:
    height = np.asarray(geopotential_height_m, dtype=np.float64)
    if np.any(~np.isfinite(height)) or np.any(height >= EARTH_EFFECTIVE_RADIUS_M):
        raise ValueError("geopotential_height_m must be finite and below the effective Earth radius.")
    return EARTH_EFFECTIVE_RADIUS_M * height / (EARTH_EFFECTIVE_RADIUS_M - height)


def geopotential_from_geometric_altitude(geometric_altitude_m: np.ndarray) -> np.ndarray:
    return G0_M_S2 * geopotential_height_from_geometric_altitude(geometric_altitude_m)


def geometric_altitude_from_geopotential(geopotential_m2_s2: np.ndarray) -> np.ndarray:
    return geometric_altitude_from_geopotential_height(
        np.asarray(geopotential_m2_s2, dtype=np.float64) / G0_M_S2
    )


def hydrostatic_pressure_profile(
    geometric_altitude_m: np.ndarray,
    virtual_temperature_k: np.ndarray,
    base_pressure_pa: float,
) -> np.ndarray:
    """Integrate d(ln p)/dz=-g0/(Rd Tv) upward with trapezoids."""
    altitude = np.asarray(geometric_altitude_m, dtype=np.float64)
    tv = np.asarray(virtual_temperature_k, dtype=np.float64)
    if altitude.ndim != 1 or tv.shape != altitude.shape or altitude.size < 2:
        raise ValueError("Altitude and virtual temperature must be conformable 1D arrays.")
    if not np.all(np.diff(altitude) > 0.0):
        raise ValueError("geometric_altitude_m must be strictly increasing.")
    if not np.isfinite(tv).all() or np.any(tv <= 0.0):
        raise ValueError("virtual_temperature_k must be finite and positive.")
    if not np.isfinite(base_pressure_pa) or base_pressure_pa <= 0.0:
        raise ValueError("base_pressure_pa must be finite and positive.")
    inverse_tv = 1.0 / tv
    increments = (
        -G0_M_S2
        / DRY_AIR_GAS_CONSTANT_J_KG_K
        * 0.5
        * (inverse_tv[:-1] + inverse_tv[1:])
        * np.diff(altitude)
    )
    log_pressure = np.empty(altitude.shape, dtype=np.float64)
    log_pressure[0] = np.log(float(base_pressure_pa))
    log_pressure[1:] = log_pressure[0] + np.cumsum(increments)
    return np.exp(log_pressure)
