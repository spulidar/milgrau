"""Offline ERA5 hybrid model-level pressure and hydrostatic reconstruction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
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
from milgrau.meteorology.interpolation import bilinear_interpolate_four_points
from milgrau.meteorology.thermodynamics import (
    DRY_AIR_GAS_CONSTANT_J_KG_K,
    geometric_altitude_from_geopotential,
    virtual_temperature,
)

ERA5_NORMALIZER_VERSION = "era5-model-levels-hydrostatic-v1"


@dataclass(frozen=True, slots=True)
class Era5Reconstruction:
    profile: AtmosphericProfile
    half_level_pressure_pa_top_down: np.ndarray
    full_level_pressure_pa_top_down: np.ndarray
    geopotential_m2_s2_top_down: np.ndarray
    model_levels_top_down: np.ndarray

    def __post_init__(self) -> None:
        for name, dtype in (
            ("half_level_pressure_pa_top_down", np.float64),
            ("full_level_pressure_pa_top_down", np.float64),
            ("geopotential_m2_s2_top_down", np.float64),
            ("model_levels_top_down", np.int16),
        ):
            values = np.array(getattr(self, name), dtype=dtype, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)


def half_level_pressures(
    hybrid_a_pa: np.ndarray,
    hybrid_b: np.ndarray,
    logarithm_surface_pressure: float,
) -> np.ndarray:
    """Compute p_half(n)=a(n)+b(n)*exp(lnsp), in Pa."""
    a = np.asarray(hybrid_a_pa, dtype=np.float64)
    b = np.asarray(hybrid_b, dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape or a.size < 3:
        raise ValueError("Hybrid a and b must be conformable 1D interface arrays.")
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(a < 0.0):
        raise ValueError("Hybrid coefficients must be finite and a must be non-negative.")
    surface_pressure = float(np.exp(float(logarithm_surface_pressure)))
    if not np.isfinite(surface_pressure) or surface_pressure <= 0.0:
        raise ValueError("lnsp must represent positive finite surface pressure.")
    pressure = a + b * surface_pressure
    if abs(pressure[0]) > 1e-9 or not np.isclose(pressure[-1], surface_pressure, rtol=1e-10):
        raise ValueError("Hybrid interfaces must start at zero and end at surface pressure.")
    if not np.all(np.diff(pressure) > 0.0):
        raise ValueError("Hybrid half-level pressure must increase from model top to surface.")
    return pressure


def full_level_pressures(half_level_pressure_pa: np.ndarray) -> np.ndarray:
    """ECMWF documented arithmetic mean of adjacent half-level pressures."""
    half = np.asarray(half_level_pressure_pa, dtype=np.float64)
    if half.ndim != 1 or half.size < 3 or not np.all(np.diff(half) > 0.0):
        raise ValueError("half_level_pressure_pa must increase top-down.")
    return 0.5 * (half[:-1] + half[1:])


def model_level_geopotential(
    half_level_pressure_pa: np.ndarray,
    temperature_k_top_down: np.ndarray,
    specific_humidity_kg_kg_top_down: np.ndarray,
    surface_geopotential_m2_s2: float,
) -> np.ndarray:
    """Integrate hydrostatically upward using the ECMWF full-level alpha convention."""
    half = np.asarray(half_level_pressure_pa, dtype=np.float64)
    temperature = np.asarray(temperature_k_top_down, dtype=np.float64)
    humidity = np.asarray(specific_humidity_kg_kg_top_down, dtype=np.float64)
    level_count = half.size - 1
    if temperature.shape != (level_count,) or humidity.shape != (level_count,):
        raise ValueError("Temperature/q must contain one top-down value per model layer.")
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("ERA5 model-level temperature must be finite and positive.")
    if np.any(~np.isfinite(humidity)) or np.any((humidity < 0.0) | (humidity > 0.1)):
        raise ValueError("ERA5 specific humidity must be finite and within [0, 0.1].")
    surface_geopotential = float(surface_geopotential_m2_s2)
    if not np.isfinite(surface_geopotential):
        raise ValueError("surface_geopotential_m2_s2 must be finite.")
    moist_temperature = virtual_temperature(temperature, humidity)
    full_geopotential = np.empty(level_count, dtype=np.float64)
    half_geopotential_below = surface_geopotential
    for level_index in range(level_count - 1, -1, -1):
        pressure_above = half[level_index]
        pressure_below = half[level_index + 1]
        if level_index == 0:
            dlog_pressure = np.log(pressure_below / 0.1)
            alpha = np.log(2.0)
        else:
            dlog_pressure = np.log(pressure_below / pressure_above)
            alpha = 1.0 - pressure_above / (pressure_below - pressure_above) * dlog_pressure
        rd_tv = DRY_AIR_GAS_CONSTANT_J_KG_K * moist_temperature[level_index]
        full_geopotential[level_index] = half_geopotential_below + rd_tv * alpha
        half_geopotential_below += rd_tv * dlog_pressure
    return full_geopotential


def _snapshot_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def normalize_era5_model_levels(
    *,
    hybrid_a_pa: np.ndarray,
    hybrid_b: np.ndarray,
    temperature_k_by_level_corner: np.ndarray,
    specific_humidity_by_level_corner: np.ndarray,
    logarithm_surface_pressure_by_corner: np.ndarray,
    surface_geopotential_m2_s2_by_corner: np.ndarray,
    corner_coordinates_lat_lon: np.ndarray,
    target_latitude_deg_north: float,
    target_longitude_deg_east: float,
    analysis_time: datetime,
    dataset_id: str = "ERA5",
    raw_snapshot: bytes | None = None,
    require_137_levels: bool = True,
) -> Era5Reconstruction:
    """Bilinearly sample four local corners then reconstruct one ERA5 column."""
    a = np.asarray(hybrid_a_pa, dtype=np.float64)
    b = np.asarray(hybrid_b, dtype=np.float64)
    temperature_corners = np.asarray(temperature_k_by_level_corner, dtype=np.float64)
    humidity_corners = np.asarray(specific_humidity_by_level_corner, dtype=np.float64)
    coordinates = np.asarray(corner_coordinates_lat_lon, dtype=np.float64)
    level_count = a.size - 1
    if require_137_levels and level_count != 137:
        raise ValueError("ERA5 scientific reconstruction requires all 137 model levels.")
    if temperature_corners.shape != (level_count, 4) or humidity_corners.shape != (
        level_count,
        4,
    ):
        raise ValueError("Temperature and humidity must have shape (model_levels, 4 corners).")
    lnsp_corners = np.asarray(logarithm_surface_pressure_by_corner, dtype=np.float64)
    surface_geopotential_corners = np.asarray(
        surface_geopotential_m2_s2_by_corner, dtype=np.float64
    )
    if lnsp_corners.shape != (4,) or surface_geopotential_corners.shape != (4,):
        raise ValueError("lnsp and surface geopotential must contain four corner values.")

    temperature = bilinear_interpolate_four_points(
        coordinates,
        temperature_corners.T,
        target_latitude_deg_north,
        target_longitude_deg_east,
    )
    humidity = bilinear_interpolate_four_points(
        coordinates,
        humidity_corners.T,
        target_latitude_deg_north,
        target_longitude_deg_east,
    )
    lnsp = float(
        bilinear_interpolate_four_points(
            coordinates,
            lnsp_corners,
            target_latitude_deg_north,
            target_longitude_deg_east,
        )
    )
    surface_geopotential = float(
        bilinear_interpolate_four_points(
            coordinates,
            surface_geopotential_corners,
            target_latitude_deg_north,
            target_longitude_deg_east,
        )
    )
    half_pressure = half_level_pressures(a, b, lnsp)
    full_pressure = full_level_pressures(half_pressure)
    geopotential_top_down = model_level_geopotential(
        half_pressure,
        temperature,
        humidity,
        surface_geopotential,
    )
    geometric_altitude_top_down = geometric_altitude_from_geopotential(geopotential_top_down)
    if not np.all(np.diff(geometric_altitude_top_down) < 0.0):
        raise ValueError("Reconstructed ERA5 geometric altitude must decrease from level 1 to 137.")

    reverse = slice(None, None, -1)
    altitude = geometric_altitude_top_down[reverse]
    pressure = full_pressure[reverse]
    temperature_bottom_up = temperature[reverse]
    humidity_bottom_up = humidity[reverse]
    geopotential_bottom_up = geopotential_top_down[reverse]
    if raw_snapshot is None:
        snapshot_hash = _snapshot_digest(
            a,
            b,
            temperature_corners,
            humidity_corners,
            lnsp_corners,
            surface_geopotential_corners,
            coordinates,
        )
    else:
        snapshot_hash = hashlib.sha256(raw_snapshot).hexdigest()
    profile = create_atmospheric_profile(
        geometric_altitude_m=altitude,
        geopotential_m2_s2=geopotential_bottom_up,
        pressure_pa=pressure,
        temperature_k=temperature_bottom_up,
        specific_humidity_kg_kg=humidity_bottom_up,
        primary_source_flag=np.full(level_count, int(PrimarySource.ERA5), dtype=np.int8),
        interpolation_flag=np.full(
            level_count, int(InterpolationFlag.INTERPOLATED), dtype=np.int8
        ),
        fallback_flag=np.full(level_count, int(FallbackFlag.NONE), dtype=np.int8),
        humidity_flag=np.full(level_count, int(HumidityFlag.MEASURED), dtype=np.int8),
        radiosonde_weight=np.zeros(level_count, dtype=np.float64),
        quality_flag=np.full(level_count, int(QualityFlag.VALID), dtype=np.int8),
        nominal_time=analysis_time,
        observation_time=analysis_time,
        latitude_deg_north=target_latitude_deg_north,
        longitude_deg_east=target_longitude_deg_east,
        provider="ECMWF ERA5 offline model-level snapshot",
        station_or_dataset_id=dataset_id,
        raw_snapshot_sha256=snapshot_hash,
        normalizer_version=ERA5_NORMALIZER_VERSION,
        vertical_coverage_m=(float(altitude[0]), float(altitude[-1])),
        profile_quality=ProfileQuality.QUANTITATIVE,
        quantitative_retrieval_allowed=True,
    )
    return Era5Reconstruction(
        profile=profile,
        half_level_pressure_pa_top_down=np.array(half_pressure, copy=True),
        full_level_pressure_pa_top_down=np.array(full_pressure, copy=True),
        geopotential_m2_s2_top_down=np.array(geopotential_top_down, copy=True),
        model_levels_top_down=np.arange(1, level_count + 1, dtype=np.int16),
    )
