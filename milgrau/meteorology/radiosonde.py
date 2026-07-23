"""Offline normalization of the current Siphon/Wyoming table contract."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

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
    specific_humidity_from_dewpoint,
    virtual_temperature,
)

RADIOSONDE_NORMALIZER_VERSION = "wyoming-table-v1"


@dataclass(frozen=True, slots=True)
class HydrostaticDiagnostic:
    compared_layer_count: int
    mean_absolute_log_pressure_residual: float
    maximum_absolute_log_pressure_residual: float


@dataclass(frozen=True, slots=True)
class RadiosondeNormalizationResult:
    profile: AtmosphericProfile
    duplicate_levels_removed: int
    incomplete_levels_removed: int
    preserved_gap_count: int
    maximum_gap_m: float
    hydrostatic: HydrostaticDiagnostic


def _snapshot_bytes(table: pd.DataFrame, raw_snapshot: bytes | None) -> bytes:
    if raw_snapshot is not None:
        return bytes(raw_snapshot)
    return table.to_json(
        orient="records",
        date_format="iso",
        double_precision=15,
    ).encode("utf-8")


def _best_unique_levels(table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    working = table.copy()
    humidity_columns = [
        name
        for name in ("specific_humidity_kg_kg", "dewpoint", "dewpoint_c")
        if name in working
    ]
    quality_columns = ["height", "pressure", "temperature", *humidity_columns]
    working["__input_order"] = np.arange(len(working), dtype=np.int64)
    working["__quality"] = working[quality_columns].notna().sum(axis=1)
    working = working.sort_values(
        ["height", "__quality", "__input_order"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    duplicate_count = int(working.duplicated(subset=["height"], keep="first").sum())
    unique = working.drop_duplicates(subset=["height"], keep="first")
    return unique.drop(columns=["__input_order", "__quality"]), duplicate_count


def normalize_wyoming_radiosonde(
    payload: pd.DataFrame | Sequence[Mapping[str, object]],
    *,
    nominal_time: datetime,
    observation_time: datetime,
    station_id: str,
    latitude_deg_north: float,
    longitude_deg_east: float,
    raw_snapshot: bytes | None = None,
    assume_dry_when_humidity_missing: bool = False,
    gap_threshold_m: float = 1000.0,
) -> RadiosondeNormalizationResult:
    """Normalize a local Wyoming/Siphon payload without fetching or extrapolating."""
    table = payload.copy() if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
    required = {"height", "pressure", "temperature"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(f"Wyoming payload is missing required columns: {missing}")
    if table.empty:
        raise ValueError("Wyoming payload is empty.")
    snapshot = _snapshot_bytes(table, raw_snapshot)
    table, duplicate_count = _best_unique_levels(table)

    numeric = table[["height", "pressure", "temperature"]].apply(pd.to_numeric, errors="coerce")
    invalid_finite_pressure = numeric["pressure"].notna() & (numeric["pressure"] <= 0.0)
    invalid_finite_temperature = numeric["temperature"].notna() & (
        numeric["temperature"] + 273.15 <= 0.0
    )
    if invalid_finite_pressure.any():
        raise ValueError("Radiosonde pressure must be positive.")
    if invalid_finite_temperature.any():
        raise ValueError("Radiosonde temperature must be above absolute zero.")
    complete = numeric.notna().all(axis=1)
    incomplete_count = int((~complete).sum())
    table = table.loc[complete].copy()
    numeric = numeric.loc[complete]
    if len(table) < 2:
        raise ValueError("At least two complete radiosonde levels are required.")

    altitude = numeric["height"].to_numpy(dtype=np.float64)
    pressure = numeric["pressure"].to_numpy(dtype=np.float64) * 100.0
    temperature = numeric["temperature"].to_numpy(dtype=np.float64) + 273.15
    order = np.argsort(altitude, kind="stable")
    altitude = altitude[order]
    pressure = pressure[order]
    temperature = temperature[order]
    table = table.iloc[order]
    if not np.all(np.diff(altitude) > 0.0):
        raise ValueError("Radiosonde height must be strictly increasing after duplicate resolution.")

    humidity = np.full(altitude.shape, np.nan, dtype=np.float64)
    humidity_flag = np.full(altitude.shape, int(HumidityFlag.MISSING), dtype=np.int8)
    if "specific_humidity_kg_kg" in table:
        supplied_q = pd.to_numeric(table["specific_humidity_kg_kg"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        supplied = np.isfinite(supplied_q)
        humidity[supplied] = supplied_q[supplied]
        humidity_flag[supplied] = int(HumidityFlag.MEASURED)
    dewpoint_column = "dewpoint" if "dewpoint" in table else "dewpoint_c" if "dewpoint_c" in table else None
    if dewpoint_column is not None:
        dewpoint_c = pd.to_numeric(table[dewpoint_column], errors="coerce").to_numpy(dtype=np.float64)
        derived = ~np.isfinite(humidity) & np.isfinite(dewpoint_c)
        humidity[derived] = specific_humidity_from_dewpoint(
            pressure[derived], dewpoint_c[derived] + 273.15
        )
        humidity_flag[derived] = int(HumidityFlag.DERIVED_FROM_DEWPOINT)
    if assume_dry_when_humidity_missing:
        assumed = ~np.isfinite(humidity)
        humidity[assumed] = 0.0
        humidity_flag[assumed] = int(HumidityFlag.DRY_AIR_ASSUMED)
    finite_humidity = np.isfinite(humidity)
    if np.any((humidity[finite_humidity] < 0.0) | (humidity[finite_humidity] > 0.1)):
        raise ValueError("Normalized radiosonde specific humidity is outside [0, 0.1].")

    quality = np.where(
        finite_humidity,
        int(QualityFlag.VALID),
        int(QualityFlag.MISSING_HUMIDITY),
    ).astype(np.int8)
    gaps = np.diff(altitude)
    tv = virtual_temperature(temperature, humidity)
    hydro_valid = np.isfinite(tv[:-1]) & np.isfinite(tv[1:])
    observed = np.log(pressure[1:] / pressure[:-1])
    expected = (
        -G0_M_S2
        * gaps
        / DRY_AIR_GAS_CONSTANT_J_KG_K
        * 0.5
        * (1.0 / tv[:-1] + 1.0 / tv[1:])
    )
    residual = np.abs(observed[hydro_valid] - expected[hydro_valid])
    hydrostatic = HydrostaticDiagnostic(
        compared_layer_count=int(residual.size),
        mean_absolute_log_pressure_residual=float(np.mean(residual)) if residual.size else np.nan,
        maximum_absolute_log_pressure_residual=float(np.max(residual)) if residual.size else np.nan,
    )

    profile_quality = (
        ProfileQuality.QUANTITATIVE if finite_humidity.all() else ProfileQuality.INCOMPLETE
    )
    profile = create_atmospheric_profile(
        geometric_altitude_m=altitude,
        pressure_pa=pressure,
        temperature_k=temperature,
        specific_humidity_kg_kg=humidity,
        primary_source_flag=np.full(altitude.shape, int(PrimarySource.RADIOSONDE), dtype=np.int8),
        interpolation_flag=np.full(altitude.shape, int(InterpolationFlag.DIRECT), dtype=np.int8),
        fallback_flag=np.full(altitude.shape, int(FallbackFlag.NONE), dtype=np.int8),
        humidity_flag=humidity_flag,
        radiosonde_weight=np.ones(altitude.shape, dtype=np.float64),
        quality_flag=quality,
        nominal_time=nominal_time,
        observation_time=observation_time,
        latitude_deg_north=latitude_deg_north,
        longitude_deg_east=longitude_deg_east,
        provider="University of Wyoming Upper Air via Siphon",
        station_or_dataset_id=str(station_id),
        raw_snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),
        normalizer_version=RADIOSONDE_NORMALIZER_VERSION,
        vertical_coverage_m=(float(altitude[0]), float(altitude[-1])),
        profile_quality=profile_quality,
        quantitative_retrieval_allowed=bool(finite_humidity.all()),
    )
    return RadiosondeNormalizationResult(
        profile=profile,
        duplicate_levels_removed=duplicate_count,
        incomplete_levels_removed=incomplete_count,
        preserved_gap_count=int(np.sum(gaps > float(gap_threshold_m))),
        maximum_gap_m=float(np.max(gaps)),
        hydrostatic=hydrostatic,
    )
