"""Thermodynamic source selection and canonical atmosphere materialization for Level 1."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.io.era5 import fetch_era5_pressure_level_profile
from milgrau.io.radiosonde import fetch_wyoming_radiosonde
from milgrau.level1.common import finite_or_fill
from milgrau.level1.config import (
    AtmosphereConfig,
    resolve_level1_config,
    resolve_radiosonde_station,
    resolve_station_site,
)
from milgrau.level1.tropopause import calculate_tropopause_heights
from milgrau.physics.atmosphere import get_standard_atmosphere


def _clean_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Return a finite, altitude-sorted temperature/pressure profile."""
    required_cols = {"height", "temperature", "pressure"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise KeyError(f"Thermodynamic profile is missing required columns: {missing}")
    cleaned = (
        df.dropna(subset=["height", "temperature", "pressure"])
        .drop_duplicates(subset=["height"], keep="first")
        .sort_values("height")
        .copy()
    )
    finite = (
        np.isfinite(cleaned["height"].to_numpy(dtype=np.float64))
        & np.isfinite(cleaned["temperature"].to_numpy(dtype=np.float64))
        & np.isfinite(cleaned["pressure"].to_numpy(dtype=np.float64))
        & (cleaned["pressure"].to_numpy(dtype=np.float64) > 0.0)
    )
    cleaned = cleaned.loc[finite].copy()
    if len(cleaned) < 2:
        raise ValueError("Thermodynamic profile must contain at least two valid vertical levels.")
    return cleaned


def _profile_metadata(df: pd.DataFrame, source_type: str, fallback_source: str) -> dict[str, Any]:
    metadata = dict(getattr(df, "attrs", {}) or {})
    metadata.setdefault("source_type", source_type)
    metadata.setdefault("source", fallback_source)
    return metadata


def _lidar_altitudes(final_ds: xr.Dataset, station_altitude_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the Level 1 lidar grid as AGL and ASL geometric altitudes."""
    if "altitude" not in final_ds.coords:
        raise KeyError("Level 1 dataset lacks the altitude coordinate required for thermodynamics.")
    altitude_agl_m = np.asarray(final_ds["altitude"].values, dtype=np.float64)
    if altitude_agl_m.ndim != 1 or altitude_agl_m.size < 2:
        raise ValueError("Level 1 altitude must be a one-dimensional grid with at least two bins.")
    if not np.all(np.isfinite(altitude_agl_m)) or not np.all(np.diff(altitude_agl_m) > 0.0):
        raise ValueError("Level 1 altitude must be finite and strictly increasing.")
    altitude_asl_m = altitude_agl_m + float(station_altitude_m)
    return altitude_agl_m, altitude_asl_m


def _write_atmospheric_profile(
    final_ds: xr.Dataset,
    temperature_k: np.ndarray,
    pressure_hpa: np.ndarray,
    *,
    source_type: str,
    source_name: str,
    metadata: Mapping[str, Any],
    standard_fallback_fraction: float,
) -> xr.Dataset:
    """Persist the canonical complete thermodynamic profile on the lidar grid."""
    temperature_k = np.asarray(temperature_k, dtype=np.float64)
    pressure_hpa = np.asarray(pressure_hpa, dtype=np.float64)
    altitude_size = final_ds.sizes.get("altitude", 0)
    if temperature_k.shape != (altitude_size,) or pressure_hpa.shape != (altitude_size,):
        raise ValueError("Atmospheric temperature/pressure must match the Level 1 altitude grid exactly.")
    if not np.all(np.isfinite(temperature_k)) or np.any(temperature_k <= 0.0):
        raise ValueError("Atmospheric temperature must be finite and positive on every Level 1 altitude bin.")
    if not np.all(np.isfinite(pressure_hpa)) or np.any(pressure_hpa <= 0.0):
        raise ValueError("Atmospheric pressure must be finite and positive on every Level 1 altitude bin.")

    final_ds["Atmospheric_Temperature_K"] = (("altitude",), temperature_k.astype(np.float64))
    final_ds["Atmospheric_Pressure_hPa"] = (("altitude",), pressure_hpa.astype(np.float64))
    final_ds["Atmospheric_Temperature_K"].attrs.update(
        {
            "units": "K",
            "long_name": "Atmospheric air temperature on the lidar altitude grid",
            "source": source_name,
            "vertical_coordinate": "altitude (AGL); source interpolation uses resolved station altitude to convert to ASL",
        }
    )
    final_ds["Atmospheric_Pressure_hPa"].attrs.update(
        {
            "units": "hPa",
            "long_name": "Atmospheric pressure on the lidar altitude grid",
            "source": source_name,
            "vertical_coordinate": "altitude (AGL); source interpolation uses resolved station altitude to convert to ASL",
        }
    )

    source_datetime = metadata.get("target_datetime_utc", metadata.get("analysis_datetime_utc", ""))
    final_ds.attrs.update(
        {
            "thermodynamic_profile_available": "true",
            "thermodynamic_profile_source_type": str(metadata.get("source_type", source_type)),
            "thermodynamic_profile_source": str(metadata.get("source", source_name)),
            "thermodynamic_profile_datetime_utc": str(source_datetime),
            "thermodynamic_profile_time_delta_hours": float(metadata.get("time_delta_hours", np.nan)),
            "thermodynamic_profile_station_id": str(metadata.get("station_id", "")),
            "thermodynamic_profile_doi": str(metadata.get("doi", "")),
            "thermodynamic_profile_standard_fallback_fraction": float(standard_fallback_fraction),
            "thermodynamic_profile_grid": "Level 1 lidar altitude grid",
            "thermodynamic_profile_altitude_reference": "AGL",
        }
    )
    return final_ds


def _materialize_external_profile(
    final_ds: xr.Dataset,
    df_profile: pd.DataFrame,
    *,
    station_altitude_m: float,
    source_type: str,
    source_name: str,
    outside_coverage_policy: str,
) -> xr.Dataset:
    """Interpolate one external ASL profile under an explicit outside-coverage policy."""
    metadata = _profile_metadata(df_profile, source_type, source_name)
    profile = _clean_profile(df_profile)
    _, altitude_asl_m = _lidar_altitudes(final_ds, station_altitude_m)

    source_altitude_asl_m = profile["height"].to_numpy(dtype=np.float64)
    source_temperature_k = profile["temperature"].to_numpy(dtype=np.float64) + 273.15
    source_pressure_hpa = profile["pressure"].to_numpy(dtype=np.float64)
    temperature_k = np.interp(
        altitude_asl_m,
        source_altitude_asl_m,
        source_temperature_k,
        left=np.nan,
        right=np.nan,
    )
    log_pressure = np.interp(
        altitude_asl_m,
        source_altitude_asl_m,
        np.log(source_pressure_hpa),
        left=np.nan,
        right=np.nan,
    )
    pressure_hpa = np.exp(log_pressure)
    fallback_mask = ~np.isfinite(temperature_k) | ~np.isfinite(pressure_hpa)

    if np.any(fallback_mask):
        if outside_coverage_policy == "fail":
            raise ValueError(
                f"{source_type} profile does not cover the full Level 1 altitude grid and "
                "level1.atmosphere.external_profile_outside_coverage='fail'."
            )
        if outside_coverage_policy != "ussa76":
            raise ValueError(f"Unsupported external-profile coverage policy: {outside_coverage_policy!r}.")
        standard_pressure, standard_temperature = get_standard_atmosphere(altitude_asl_m)
        temperature_k = np.where(fallback_mask, standard_temperature, temperature_k)
        pressure_hpa = np.where(fallback_mask, standard_pressure, pressure_hpa)

    fallback_fraction = float(np.mean(fallback_mask)) if fallback_mask.size else 0.0
    metadata = dict(metadata)
    metadata["source_profile_min_altitude_asl_m"] = float(source_altitude_asl_m[0])
    metadata["source_profile_max_altitude_asl_m"] = float(source_altitude_asl_m[-1])
    final_ds = _write_atmospheric_profile(
        final_ds,
        temperature_k,
        pressure_hpa,
        source_type=source_type,
        source_name=source_name,
        metadata=metadata,
        standard_fallback_fraction=fallback_fraction,
    )
    final_ds.attrs.update(
        {
            "thermodynamic_source_profile_min_altitude_asl_m": float(source_altitude_asl_m[0]),
            "thermodynamic_source_profile_max_altitude_asl_m": float(source_altitude_asl_m[-1]),
            "thermodynamic_external_profile_outside_coverage_policy": outside_coverage_policy,
        }
    )
    return final_ds


def _materialize_ussa76(final_ds: xr.Dataset, *, station_altitude_m: float) -> xr.Dataset:
    """Materialize USSA76 directly on the Level 1 lidar grid."""
    _, altitude_asl_m = _lidar_altitudes(final_ds, station_altitude_m)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_asl_m)
    return _write_atmospheric_profile(
        final_ds,
        temperature_k,
        pressure_hpa,
        source_type="ussa76",
        source_name="US Standard Atmosphere 1976",
        metadata={"source_type": "ussa76", "source": "US Standard Atmosphere 1976"},
        standard_fallback_fraction=1.0,
    )


def _record_policy(final_ds: xr.Dataset, policy: AtmosphereConfig, attempts: list[str]) -> None:
    final_ds.attrs["thermodynamic_source_priority"] = ",".join(policy.source_priority)
    final_ds.attrs["thermodynamic_source_attempts"] = ",".join(attempts)
    final_ds.attrs["thermodynamic_external_profile_outside_coverage_policy"] = (
        policy.external_profile_outside_coverage
    )


def integrate_thermodynamics(final_ds: xr.Dataset, config: Mapping[str, Any], logger: logging.Logger) -> xr.Dataset:
    """Materialize atmosphere by following only the configured Level 1 source policy."""
    level1_cfg = resolve_level1_config(config)
    policy = level1_cfg.atmosphere
    station_site = resolve_station_site(config, final_ds)
    station_altitude_m = station_site["station_altitude_m"]
    station_id, station_name = resolve_radiosonde_station(config)
    dt_utc = pd.to_datetime(final_ds.time.values[len(final_ds.time) // 2])
    attempts: list[str] = []

    final_ds.attrs.update(
        {
            "radiosonde_station_id": station_id,
            "radiosonde_station_name": station_name,
            "radiosonde_available": "false",
            "tropopause_cpt_km": -999.0,
            "tropopause_lrt_km": -999.0,
            "thermodynamic_station_latitude": float(station_site["latitude"]),
            "thermodynamic_station_longitude": float(station_site["longitude"]),
            "thermodynamic_station_altitude_m": float(station_altitude_m),
        }
    )

    for source in policy.source_priority:
        attempts.append(source)
        if source == "radiosonde":
            if policy.radiosonde is None:
                raise RuntimeError("Resolved atmosphere policy lacks radiosonde settings.")
            try:
                df_radio = fetch_wyoming_radiosonde(
                    dt_utc,
                    station_id,
                    logger,
                    **policy.radiosonde.as_io_mapping(),
                )
            except Exception as exc:
                logger.warning(f"  -> Radiosonde retrieval failed under configured policy: {exc}")
                df_radio = None
            if df_radio is None or df_radio.empty:
                continue
            try:
                result = _materialize_external_profile(
                    final_ds,
                    df_radio,
                    station_altitude_m=station_altitude_m,
                    source_type="radiosonde",
                    source_name="University of Wyoming Upper Air via Siphon",
                    outside_coverage_policy=policy.external_profile_outside_coverage,
                )
                cpt, lrt = calculate_tropopause_heights(_clean_profile(df_radio))
                cpt = finite_or_fill(cpt)
                lrt = finite_or_fill(lrt)
                result.attrs.update(
                    {
                        "radiosonde_available": "true",
                        "tropopause_cpt_km": cpt,
                        "tropopause_lrt_km": lrt,
                    }
                )
                _record_policy(result, policy, attempts)
                logger.info(
                    "  -> Radiosonde atmosphere materialized on Level 1 grid; USSA76 extension fraction %.1f%%.",
                    100.0 * float(result.attrs["thermodynamic_profile_standard_fallback_fraction"]),
                )
                return result
            except Exception as exc:
                logger.warning(f"  -> Radiosonde profile unusable under configured policy: {exc}")
                continue

        if source == "era5":
            if policy.era5 is None:
                raise RuntimeError("Resolved atmosphere policy lacks ERA5 settings.")
            try:
                df_era5 = fetch_era5_pressure_level_profile(
                    dt_utc,
                    station_site["latitude"],
                    station_site["longitude"],
                    logger,
                    settings=policy.era5.as_io_mapping(),
                )
            except Exception as exc:
                logger.warning(f"  -> ERA5 retrieval failed under configured policy: {exc}")
                df_era5 = None
            if df_era5 is None or df_era5.empty:
                continue
            try:
                result = _materialize_external_profile(
                    final_ds,
                    df_era5,
                    station_altitude_m=station_altitude_m,
                    source_type="era5",
                    source_name="Copernicus Climate Change Service ERA5 pressure-level reanalysis",
                    outside_coverage_policy=policy.external_profile_outside_coverage,
                )
                _record_policy(result, policy, attempts)
                logger.info(
                    "  -> ERA5 atmosphere materialized on Level 1 grid; USSA76 extension fraction %.1f%%.",
                    100.0 * float(result.attrs["thermodynamic_profile_standard_fallback_fraction"]),
                )
                return result
            except Exception as exc:
                logger.warning(f"  -> ERA5 profile unusable under configured policy: {exc}")
                continue

        if source == "ussa76":
            result = _materialize_ussa76(final_ds, station_altitude_m=station_altitude_m)
            _record_policy(result, policy, attempts)
            logger.warning("  -> Materialized US Standard Atmosphere 1976 according to configured source priority.")
            return result

        raise RuntimeError(f"Unsupported atmosphere source after validation: {source!r}.")

    raise RuntimeError(
        "Configured Level 1 atmosphere source policy was exhausted without a usable profile: "
        + ", ".join(attempts)
    )
