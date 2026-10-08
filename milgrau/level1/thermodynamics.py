"""Time-resolved thermodynamic atmosphere materialization for Level 1."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.io.era5 import ERA5_DOI, fetch_era5_pressure_level_profile
from milgrau.io.logging_utils import bind_log_context
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
    """Return Level 1 lidar altitude as AGL and geometric ASL arrays."""
    if "altitude" not in final_ds.coords:
        raise KeyError("Level 1 dataset lacks the altitude coordinate required for thermodynamics.")
    altitude_agl_m = np.asarray(final_ds["altitude"].values, dtype=np.float64)
    if altitude_agl_m.ndim != 1 or altitude_agl_m.size < 2:
        raise ValueError("Level 1 altitude must be one-dimensional with at least two bins.")
    if not np.all(np.isfinite(altitude_agl_m)) or not np.all(np.diff(altitude_agl_m) > 0.0):
        raise ValueError("Level 1 altitude must be finite and strictly increasing.")
    return altitude_agl_m, altitude_agl_m + float(station_altitude_m)


def _profile_on_lidar_grid(
    final_ds: xr.Dataset,
    df_profile: pd.DataFrame,
    *,
    station_altitude_m: float,
    source_type: str,
    source_name: str,
    outside_coverage_policy: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float]:
    """Interpolate one external ASL profile onto the Level 1 lidar grid."""
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
    pressure_hpa = np.exp(
        np.interp(
            altitude_asl_m,
            source_altitude_asl_m,
            np.log(source_pressure_hpa),
            left=np.nan,
            right=np.nan,
        )
    )
    fallback_mask = ~np.isfinite(temperature_k) | ~np.isfinite(pressure_hpa)
    if np.any(fallback_mask):
        if outside_coverage_policy == "fail":
            raise ValueError(
                f"{source_type} profile does not cover the complete Level 1 altitude grid."
            )
        if outside_coverage_policy != "ussa76":
            raise ValueError(f"Unsupported external-profile coverage policy: {outside_coverage_policy!r}.")
        standard_pressure, standard_temperature = get_standard_atmosphere(altitude_asl_m)
        temperature_k = np.where(fallback_mask, standard_temperature, temperature_k)
        pressure_hpa = np.where(fallback_mask, standard_pressure, pressure_hpa)

    if not np.all(np.isfinite(temperature_k)) or np.any(temperature_k <= 0.0):
        raise ValueError(f"{source_type} temperature is invalid after Level 1 interpolation.")
    if not np.all(np.isfinite(pressure_hpa)) or np.any(pressure_hpa <= 0.0):
        raise ValueError(f"{source_type} pressure is invalid after Level 1 interpolation.")

    metadata = dict(metadata)
    metadata["source_profile_min_altitude_asl_m"] = float(source_altitude_asl_m[0])
    metadata["source_profile_max_altitude_asl_m"] = float(source_altitude_asl_m[-1])
    fallback_fraction = float(np.mean(fallback_mask)) if fallback_mask.size else 0.0
    return temperature_k, pressure_hpa, metadata, fallback_fraction


def _ussa76_on_lidar_grid(
    final_ds: xr.Dataset,
    *,
    station_altitude_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    _, altitude_asl_m = _lidar_altitudes(final_ds, station_altitude_m)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_asl_m)
    return np.asarray(temperature_k, dtype=np.float64), np.asarray(pressure_hpa, dtype=np.float64)


def _atmosphere_times(final_ds: xr.Dataset, cadence_minutes: int) -> pd.DatetimeIndex:
    """Return source times bracketing the full lidar session at the configured cadence."""
    if "time" not in final_ds.coords or final_ds.sizes.get("time", 0) == 0:
        raise ValueError("Time-resolved atmosphere requires a non-empty Level 1 time coordinate.")
    times = pd.to_datetime(final_ds["time"].values)
    start = pd.Timestamp(times.min())
    stop = pd.Timestamp(times.max())
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
        stop = stop.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
        stop = stop.tz_convert("UTC")
    cadence = f"{int(cadence_minutes)}min"
    first = start.floor(cadence)
    last = stop.ceil(cadence)
    return pd.date_range(first, last, freq=cadence, tz="UTC")


def _compact_error(exc: BaseException) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return lines[0] if lines else type(exc).__name__


def _source_tropopause(profile: pd.DataFrame | None) -> tuple[float, float]:
    if profile is None or profile.empty:
        return -999.0, -999.0
    try:
        cpt, lrt = calculate_tropopause_heights(_clean_profile(profile))
        return finite_or_fill(cpt), finite_or_fill(lrt)
    except Exception:
        return -999.0, -999.0


def _write_time_resolved_atmosphere(
    final_ds: xr.Dataset,
    *,
    atmosphere_times: pd.DatetimeIndex,
    temperature_k: np.ndarray,
    pressure_hpa: np.ndarray,
    source_types: list[str],
    source_time_delta_hours: np.ndarray,
    fallback_fraction: np.ndarray,
    source_min_altitude_m: np.ndarray,
    source_max_altitude_m: np.ndarray,
    tropopause_cpt_km: np.ndarray,
    tropopause_lrt_km: np.ndarray,
    policy: AtmosphereConfig,
    station_site: Mapping[str, float],
) -> xr.Dataset:
    """Persist the canonical hourly atmosphere on the Level 1 lidar altitude grid."""
    n_time = len(atmosphere_times)
    n_altitude = final_ds.sizes.get("altitude", 0)
    expected = (n_time, n_altitude)
    if temperature_k.shape != expected or pressure_hpa.shape != expected:
        raise ValueError(
            f"Time-resolved atmosphere must have shape {expected}; got "
            f"{temperature_k.shape} and {pressure_hpa.shape}."
        )
    if not np.all(np.isfinite(temperature_k)) or np.any(temperature_k <= 0.0):
        raise ValueError("Atmospheric temperature must be finite and positive.")
    if not np.all(np.isfinite(pressure_hpa)) or np.any(pressure_hpa <= 0.0):
        raise ValueError("Atmospheric pressure must be finite and positive.")

    time_values = atmosphere_times.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    final_ds = final_ds.assign_coords(atmosphere_time=("atmosphere_time", time_values))
    final_ds["Atmospheric_Temperature_K"] = (
        ("atmosphere_time", "altitude"),
        temperature_k.astype(np.float64),
    )
    final_ds["Atmospheric_Pressure_hPa"] = (
        ("atmosphere_time", "altitude"),
        pressure_hpa.astype(np.float64),
    )
    final_ds["Atmospheric_Source_Type"] = (
        ("atmosphere_time",),
        np.asarray(source_types, dtype=object),
    )
    final_ds["Atmospheric_Source_Time_Delta_hours"] = (
        ("atmosphere_time",),
        np.asarray(source_time_delta_hours, dtype=np.float64),
    )
    final_ds["Atmospheric_USSA76_Fallback_Fraction"] = (
        ("atmosphere_time",),
        np.asarray(fallback_fraction, dtype=np.float64),
    )
    final_ds["Atmospheric_Source_Min_Altitude_ASL_m"] = (
        ("atmosphere_time",),
        np.asarray(source_min_altitude_m, dtype=np.float64),
    )
    final_ds["Atmospheric_Source_Max_Altitude_ASL_m"] = (
        ("atmosphere_time",),
        np.asarray(source_max_altitude_m, dtype=np.float64),
    )
    final_ds["Tropopause_CPT_km"] = (
        ("atmosphere_time",),
        np.asarray(tropopause_cpt_km, dtype=np.float64),
    )
    final_ds["Tropopause_LRT_km"] = (
        ("atmosphere_time",),
        np.asarray(tropopause_lrt_km, dtype=np.float64),
    )

    final_ds["Atmospheric_Temperature_K"].attrs.update(
        {
            "units": "K",
            "long_name": "Hourly atmospheric air temperature on the lidar altitude grid",
            "vertical_coordinate": "altitude (AGL); external sources are interpolated from ASL",
        }
    )
    final_ds["Atmospheric_Pressure_hPa"].attrs.update(
        {
            "units": "hPa",
            "long_name": "Hourly atmospheric pressure on the lidar altitude grid",
            "vertical_coordinate": "altitude (AGL); external sources are interpolated from ASL",
        }
    )

    finite_cpt = np.asarray(tropopause_cpt_km, dtype=np.float64)
    finite_lrt = np.asarray(tropopause_lrt_km, dtype=np.float64)
    cpt_values = finite_cpt[np.isfinite(finite_cpt) & (finite_cpt > 0.0)]
    lrt_values = finite_lrt[np.isfinite(finite_lrt) & (finite_lrt > 0.0)]
    source_set = sorted(set(source_types))
    final_ds.attrs.update(
        {
            "thermodynamic_profile_available": "true",
            "thermodynamic_profile_source_type": "time_resolved",
            "thermodynamic_profile_source": "hourly ERA5 with explicit USSA76 fallback",
            "thermodynamic_profile_doi": ERA5_DOI if "era5" in source_set else "",
            "thermodynamic_profile_standard_fallback_fraction": float(np.mean(fallback_fraction)),
            "thermodynamic_profile_grid": "atmosphere_time x Level 1 lidar altitude",
            "thermodynamic_profile_altitude_reference": "AGL",
            "thermodynamic_time_resolution_minutes": int(policy.time_resolution_minutes),
            "thermodynamic_source_priority": ",".join(policy.source_priority),
            "thermodynamic_sources_present": ",".join(source_set),
            "thermodynamic_external_profile_outside_coverage_policy": policy.external_profile_outside_coverage,
            "thermodynamic_station_latitude": float(station_site["latitude"]),
            "thermodynamic_station_longitude": float(station_site["longitude"]),
            "thermodynamic_station_altitude_m": float(station_site["station_altitude_m"]),
            # Transitional scalar summaries retained for existing plot overlays.
            "tropopause_cpt_km": float(np.median(cpt_values)) if cpt_values.size else -999.0,
            "tropopause_lrt_km": float(np.median(lrt_values)) if lrt_values.size else -999.0,
            "tropopause_source_type": "time_resolved",
        }
    )
    return final_ds


def _attach_radiosonde_qa(
    final_ds: xr.Dataset,
    *,
    policy: AtmosphereConfig,
    station_altitude_m: float,
    station_id: str,
    station_name: str,
    target_time: pd.Timestamp,
    logger: logging.Logger,
) -> xr.Dataset:
    """Attach one mapped radiosonde reference profile for later offline QA figures."""
    final_ds.attrs.update(
        {
            "radiosonde_station_id": station_id,
            "radiosonde_station_name": station_name,
            "radiosonde_available": "false",
        }
    )
    if policy.radiosonde is None:
        return final_ds

    try:
        profile = fetch_wyoming_radiosonde(
            target_time.to_pydatetime(),
            station_id,
            logger,
            **policy.radiosonde.as_io_mapping(),
        )
    except Exception as exc:
        logger.warning("radiosonde QA unavailable | %s", _compact_error(exc))
        logger.debug("radiosonde QA retrieval failure", exc_info=True)
        return final_ds
    if profile is None or profile.empty:
        return final_ds

    try:
        temperature_k, pressure_hpa, metadata, fallback_fraction = _profile_on_lidar_grid(
            final_ds,
            profile,
            station_altitude_m=station_altitude_m,
            source_type="radiosonde",
            source_name="University of Wyoming Upper Air via Siphon",
            outside_coverage_policy=policy.external_profile_outside_coverage,
        )
    except Exception as exc:
        logger.warning("radiosonde QA profile unusable | %s", _compact_error(exc))
        logger.debug("radiosonde QA profile failure", exc_info=True)
        return final_ds

    final_ds["Radiosonde_QA_Temperature_K"] = (("altitude",), temperature_k)
    final_ds["Radiosonde_QA_Pressure_hPa"] = (("altitude",), pressure_hpa)
    final_ds["Radiosonde_QA_Temperature_K"].attrs["units"] = "K"
    final_ds["Radiosonde_QA_Pressure_hPa"].attrs["units"] = "hPa"
    final_ds.attrs.update(
        {
            "radiosonde_available": "true",
            "radiosonde_qa_target_datetime_utc": str(
                metadata.get("target_datetime_utc", metadata.get("analysis_datetime_utc", ""))
            ),
            "radiosonde_qa_time_delta_hours": float(metadata.get("time_delta_hours", np.nan)),
            "radiosonde_qa_source_profile_min_altitude_asl_m": float(
                metadata.get("source_profile_min_altitude_asl_m", np.nan)
            ),
            "radiosonde_qa_source_profile_max_altitude_asl_m": float(
                metadata.get("source_profile_max_altitude_asl_m", np.nan)
            ),
            "radiosonde_qa_ussa76_extension_fraction": float(fallback_fraction),
        }
    )
    return final_ds


def integrate_thermodynamics(
    final_ds: xr.Dataset,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> xr.Dataset:
    """Materialize the hourly canonical atmosphere and an optional radiosonde QA reference."""
    level1_cfg = resolve_level1_config(config)
    policy = level1_cfg.atmosphere
    station_site = resolve_station_site(config, final_ds)
    station_altitude_m = float(station_site["station_altitude_m"])
    atmosphere_times = _atmosphere_times(final_ds, policy.time_resolution_minutes)
    altitude_size = final_ds.sizes.get("altitude", 0)

    temperature_rows: list[np.ndarray] = []
    pressure_rows: list[np.ndarray] = []
    source_types: list[str] = []
    source_time_delta_hours: list[float] = []
    fallback_fractions: list[float] = []
    source_min_altitude: list[float] = []
    source_max_altitude: list[float] = []
    cpt_rows: list[float] = []
    lrt_rows: list[float] = []

    for timestamp in atmosphere_times:
        resolved = False
        for source in policy.source_priority:
            if source == "era5":
                if policy.era5 is None:
                    raise RuntimeError("Resolved atmosphere policy lacks ERA5 settings.")
                try:
                    profile = fetch_era5_pressure_level_profile(
                        timestamp.to_pydatetime(),
                        float(station_site["latitude"]),
                        float(station_site["longitude"]),
                        bind_log_context(logger, stage="era5"),
                        settings=policy.era5.as_io_mapping(),
                    )
                except Exception as exc:
                    logger.warning("ERA5 unavailable at %s | %s", timestamp, _compact_error(exc))
                    logger.debug("ERA5 hourly retrieval failure", exc_info=True)
                    profile = None
                if profile is None or profile.empty:
                    continue
                try:
                    temperature_k, pressure_hpa, metadata, fallback_fraction = _profile_on_lidar_grid(
                        final_ds,
                        profile,
                        station_altitude_m=station_altitude_m,
                        source_type="era5",
                        source_name="Copernicus Climate Change Service ERA5 pressure-level reanalysis",
                        outside_coverage_policy=policy.external_profile_outside_coverage,
                    )
                except Exception as exc:
                    logger.warning("ERA5 profile unusable at %s | %s", timestamp, _compact_error(exc))
                    logger.debug("ERA5 hourly profile failure", exc_info=True)
                    continue
                cpt, lrt = _source_tropopause(profile)
                temperature_rows.append(temperature_k)
                pressure_rows.append(pressure_hpa)
                source_types.append("era5")
                source_time_delta_hours.append(float(metadata.get("time_delta_hours", 0.0)))
                fallback_fractions.append(float(fallback_fraction))
                source_min_altitude.append(float(metadata.get("source_profile_min_altitude_asl_m", np.nan)))
                source_max_altitude.append(float(metadata.get("source_profile_max_altitude_asl_m", np.nan)))
                cpt_rows.append(float(cpt))
                lrt_rows.append(float(lrt))
                resolved = True
                break

            if source == "ussa76":
                temperature_k, pressure_hpa = _ussa76_on_lidar_grid(
                    final_ds,
                    station_altitude_m=station_altitude_m,
                )
                if temperature_k.shape != (altitude_size,) or pressure_hpa.shape != (altitude_size,):
                    raise RuntimeError("USSA76 fallback does not match the Level 1 altitude grid.")
                temperature_rows.append(temperature_k)
                pressure_rows.append(pressure_hpa)
                source_types.append("ussa76")
                source_time_delta_hours.append(np.nan)
                fallback_fractions.append(1.0)
                source_min_altitude.append(np.nan)
                source_max_altitude.append(np.nan)
                cpt_rows.append(-999.0)
                lrt_rows.append(-999.0)
                resolved = True
                logger.warning("USSA76 fallback | %s", timestamp.strftime("%Y-%m-%d %H:%MZ"))
                break

            raise RuntimeError(f"Unsupported productive atmosphere source: {source!r}.")

        if not resolved:
            raise RuntimeError(
                f"Atmosphere source policy exhausted at {timestamp.strftime('%Y-%m-%d %H:%MZ')}."
            )

    result = _write_time_resolved_atmosphere(
        final_ds,
        atmosphere_times=atmosphere_times,
        temperature_k=np.vstack(temperature_rows),
        pressure_hpa=np.vstack(pressure_rows),
        source_types=source_types,
        source_time_delta_hours=np.asarray(source_time_delta_hours, dtype=np.float64),
        fallback_fraction=np.asarray(fallback_fractions, dtype=np.float64),
        source_min_altitude_m=np.asarray(source_min_altitude, dtype=np.float64),
        source_max_altitude_m=np.asarray(source_max_altitude, dtype=np.float64),
        tropopause_cpt_km=np.asarray(cpt_rows, dtype=np.float64),
        tropopause_lrt_km=np.asarray(lrt_rows, dtype=np.float64),
        policy=policy,
        station_site=station_site,
    )

    try:
        station_id, station_name = resolve_radiosonde_station(config)
    except Exception:
        station_id = station_name = ""
    session_midpoint = pd.Timestamp(atmosphere_times[len(atmosphere_times) // 2])
    result = _attach_radiosonde_qa(
        result,
        policy=policy,
        station_altitude_m=station_altitude_m,
        station_id=station_id,
        station_name=station_name,
        target_time=session_midpoint,
        logger=bind_log_context(logger, stage="radiosonde_qa"),
    )
    logger.info(
        "time-resolved atmosphere | %d hourly profiles | sources=%s",
        len(atmosphere_times),
        ",".join(sorted(set(source_types))),
    )
    return result
