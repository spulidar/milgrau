"""Thermodynamic integration for Level 1 products."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.io.era5 import fetch_era5_pressure_level_profile
from milgrau.io.radiosonde import fetch_wyoming_radiosonde
from milgrau.level1.common import finite_or_fill
from milgrau.level1.pbl import calculate_pbl_height_gradient
from milgrau.level1.tropopause import calculate_tropopause_heights
from milgrau.physics.atmosphere import get_standard_atmosphere


def estimate_pbl_timeseries(final_ds: xr.Dataset, z_arr: np.ndarray, config: Mapping[str, Any], logger: logging.Logger) -> xr.Dataset:
    """Estimate Planetary Boundary Layer height for every time profile."""
    try:
        pbl_channel = next((ch for ch in final_ds.channel.values.astype(str) if "an" in ch.lower() and "532" in ch), str(final_ds.channel.values[0]))
        physics_cfg = config.get("physics", {})
        min_search_m = float(physics_cfg.get("pbl_min_search_m", 500.0))
        max_search_m = float(physics_cfg.get("pbl_max_search_m", 4000.0))
        smooth_bins = int(physics_cfg.get("pbl_smooth_bins", 15))
        rcs_matrix = final_ds["range_corrected_signal"].sel(channel=pbl_channel).values
        logger.info(f"  -> Tracking PBL using {pbl_channel} ({min_search_m:.0f}-{max_search_m:.0f} m).")
        pbl_h = [
            calculate_pbl_height_gradient(
                rcs_matrix[t, :],
                z_arr,
                min_search_m=min_search_m,
                max_search_m=max_search_m,
                smooth_bins=smooth_bins,
            )
            for t in range(rcs_matrix.shape[0])
        ]
        final_ds["PBL_Height_km"] = xr.DataArray(pbl_h, dims=["time"], coords={"time": final_ds.time}).astype(np.float32)
        final_ds["PBL_Height_km"].attrs = {
            "units": "km",
            "method": "Gradient method on smoothed RCS",
            "reference_channel": pbl_channel,
            "min_search_m": min_search_m,
            "max_search_m": max_search_m,
            "smooth_bins": smooth_bins,
        }
        return final_ds
    except Exception as exc:
        logger.warning(f"  -> PBL tracking failed: {exc}")
        return final_ds


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
    if cleaned.empty:
        raise ValueError("Thermodynamic profile became empty after cleaning.")
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


def _station_altitude_m(config: Mapping[str, Any]) -> float:
    site = config.get("site", {})
    if not isinstance(site, Mapping):
        site = {}
    physics = config.get("physics", {})
    if not isinstance(physics, Mapping):
        physics = {}
    return float(site.get("station_altitude_m", physics.get("station_altitude_m", 0.0)))


def _lidar_altitudes(final_ds: xr.Dataset, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return the Level 1 lidar grid as AGL and ASL geometric altitudes."""
    if "altitude" not in final_ds.coords:
        raise KeyError("Level 1 dataset lacks the altitude coordinate required for thermodynamics.")
    altitude_agl_m = np.asarray(final_ds["altitude"].values, dtype=np.float64)
    if altitude_agl_m.ndim != 1 or altitude_agl_m.size < 2:
        raise ValueError("Level 1 altitude must be a one-dimensional grid with at least two bins.")
    if not np.all(np.isfinite(altitude_agl_m)) or not np.all(np.diff(altitude_agl_m) > 0.0):
        raise ValueError("Level 1 altitude must be finite and strictly increasing.")
    altitude_asl_m = altitude_agl_m + _station_altitude_m(config)
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
            "vertical_coordinate": "altitude (AGL); source profile interpolation uses station altitude to convert to ASL",
        }
    )
    final_ds["Atmospheric_Pressure_hPa"].attrs.update(
        {
            "units": "hPa",
            "long_name": "Atmospheric pressure on the lidar altitude grid",
            "source": source_name,
            "vertical_coordinate": "altitude (AGL); source profile interpolation uses station altitude to convert to ASL",
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
    config: Mapping[str, Any],
    *,
    source_type: str,
    source_name: str,
) -> xr.Dataset:
    """Interpolate one external ASL profile onto the lidar grid, with USSA76 only outside coverage."""
    metadata = _profile_metadata(df_profile, source_type, source_name)
    profile = _clean_profile(df_profile)
    _, altitude_asl_m = _lidar_altitudes(final_ds, config)
    standard_pressure, standard_temperature = get_standard_atmosphere(altitude_asl_m)

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
    # Pressure is approximately exponential with altitude; interpolate in log(P)
    # rather than linearly in pressure.
    log_pressure = np.interp(
        altitude_asl_m,
        source_altitude_asl_m,
        np.log(source_pressure_hpa),
        left=np.nan,
        right=np.nan,
    )
    pressure_hpa = np.exp(log_pressure)
    fallback_mask = ~np.isfinite(temperature_k) | ~np.isfinite(pressure_hpa)
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
        }
    )
    return final_ds


def _materialize_ussa76(final_ds: xr.Dataset, config: Mapping[str, Any]) -> xr.Dataset:
    """Materialize USSA76 directly on the Level 1 lidar grid."""
    _, altitude_asl_m = _lidar_altitudes(final_ds, config)
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


def integrate_thermodynamics(final_ds: xr.Dataset, config: Mapping[str, Any], logger: logging.Logger) -> xr.Dataset:
    """Resolve and materialize thermodynamics: radiosonde -> ERA5 -> USSA76.

    Every successful Level 1 product receives one complete canonical atmosphere
    on its own lidar altitude grid. Level 2 consumes that stored profile and does
    not perform network IO, source selection, interpolation, or fallback logic.
    """
    dt_utc = pd.to_datetime(final_ds.time.values[len(final_ds.time) // 2])
    radiosonde_cfg = config.get("radiosonde", {})
    if not isinstance(radiosonde_cfg, Mapping):
        radiosonde_cfg = {}
    location_cfg = config.get("location", {})
    if not isinstance(location_cfg, Mapping):
        location_cfg = {}
    station_id = str(radiosonde_cfg.get("station_id", location_cfg.get("station_id", "83779")))

    try:
        df_radio = fetch_wyoming_radiosonde(dt_utc, station_id, logger, config=config)
    except Exception as exc:
        logger.warning(f"  -> Radiosonde retrieval failed: {exc}")
        df_radio = None

    if df_radio is not None and not df_radio.empty:
        try:
            final_ds = _materialize_external_profile(
                final_ds,
                df_radio,
                config,
                source_type="radiosonde",
                source_name="University of Wyoming Upper Air via Siphon",
            )
            cpt, lrt = calculate_tropopause_heights(_clean_profile(df_radio))
            cpt = finite_or_fill(cpt)
            lrt = finite_or_fill(lrt)
            final_ds.attrs.update(
                {
                    "radiosonde_station_id": station_id,
                    "radiosonde_available": "true",
                    "tropopause_cpt_km": cpt,
                    "tropopause_lrt_km": lrt,
                }
            )
            logger.info(
                f"  -> Radiosonde atmosphere materialized on Level 1 grid. "
                f"USSA76 extension fraction: {100.0 * float(final_ds.attrs['thermodynamic_profile_standard_fallback_fraction']):.1f}% | "
                f"CPT: {cpt:.2f} km | LRT: {lrt:.2f} km"
            )
            return final_ds
        except Exception as exc:
            logger.warning(f"  -> Radiosonde profile was unusable after retrieval: {exc}")

    final_ds.attrs.update(
        {
            "radiosonde_station_id": station_id,
            "radiosonde_available": "false",
            "tropopause_cpt_km": -999.0,
            "tropopause_lrt_km": -999.0,
        }
    )

    era5_cfg = config.get("era5", {})
    if isinstance(era5_cfg, Mapping) and bool(era5_cfg.get("enabled", False)):
        site_cfg = config.get("site", {})
        if not isinstance(site_cfg, Mapping):
            site_cfg = {}
        physics_cfg = config.get("physics", {})
        if not isinstance(physics_cfg, Mapping):
            physics_cfg = {}
        latitude = site_cfg.get("latitude", physics_cfg.get("latitude"))
        longitude = site_cfg.get("longitude", physics_cfg.get("longitude"))
        if latitude is None or longitude is None:
            logger.warning("  -> ERA5 fallback enabled but site latitude/longitude are unavailable.")
        else:
            try:
                df_era5 = fetch_era5_pressure_level_profile(
                    dt_utc,
                    float(latitude),
                    float(longitude),
                    logger,
                    config=config,
                )
            except Exception as exc:
                logger.warning(f"  -> ERA5 retrieval failed: {exc}")
                df_era5 = None
            if df_era5 is not None and not df_era5.empty:
                try:
                    final_ds = _materialize_external_profile(
                        final_ds,
                        df_era5,
                        config,
                        source_type="era5",
                        source_name="Copernicus Climate Change Service ERA5 pressure-level reanalysis",
                    )
                    logger.info(
                        "  -> ERA5 atmosphere materialized on Level 1 grid. "
                        f"USSA76 extension fraction: {100.0 * float(final_ds.attrs['thermodynamic_profile_standard_fallback_fraction']):.1f}%"
                    )
                    return final_ds
                except Exception as exc:
                    logger.warning(f"  -> ERA5 profile was unusable after retrieval: {exc}")

    logger.warning("  -> External thermodynamics unavailable. Materializing US Standard Atmosphere 1976 in Level 1.")
    return _materialize_ussa76(final_ds, config)
