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


def _add_profile_to_level1(
    final_ds: xr.Dataset,
    df_profile: pd.DataFrame,
    *,
    source_type: str,
    source_name: str,
) -> xr.Dataset:
    """Persist one external thermodynamic profile plus explicit provenance.

    ``atmospheric_*`` variables are canonical. ``Radiosonde_*`` aliases remain
    temporarily for the existing Level 2 reader; their metadata explicitly says
    when the underlying source is ERA5 rather than a physical sounding.
    """
    metadata = _profile_metadata(df_profile, source_type, source_name)
    profile = _clean_profile(df_profile)
    altitude = profile["height"].to_numpy(dtype=np.float64)
    temperature_k = profile["temperature"].to_numpy(dtype=np.float64) + 273.15
    pressure_hpa = profile["pressure"].to_numpy(dtype=np.float64)

    final_ds = final_ds.assign_coords(atmospheric_altitude=("atmospheric_altitude", altitude))
    final_ds["atmospheric_altitude"].attrs.update(
        {"units": "m", "long_name": "External thermodynamic profile altitude above mean sea level"}
    )
    final_ds["Atmospheric_Temperature_K"] = (("atmospheric_altitude",), temperature_k.astype(np.float32))
    final_ds["Atmospheric_Pressure_hPa"] = (("atmospheric_altitude",), pressure_hpa.astype(np.float32))
    final_ds["Atmospheric_Temperature_K"].attrs.update(
        {"units": "K", "long_name": "External atmospheric air temperature", "source": str(metadata["source"])}
    )
    final_ds["Atmospheric_Pressure_hPa"].attrs.update(
        {"units": "hPa", "long_name": "External atmospheric pressure", "source": str(metadata["source"])}
    )

    # Backward-compatible aliases consumed by the current Level 2 reader. They
    # are data aliases only; provenance below remains authoritative.
    final_ds = final_ds.assign_coords(radiosonde_altitude=("radiosonde_altitude", altitude))
    final_ds["Radiosonde_Temperature_K"] = (("radiosonde_altitude",), temperature_k.astype(np.float32))
    final_ds["Radiosonde_Pressure_hPa"] = (("radiosonde_altitude",), pressure_hpa.astype(np.float32))
    source_label = source_type.upper() if source_type.lower() == "era5" else source_type
    alias_note = (
        "Legacy Level 2 thermodynamic compatibility alias. "
        f"Actual source type is {source_label}; do not infer radiosonde provenance from the variable name."
    )
    final_ds["radiosonde_altitude"].attrs.update({"units": "m", "long_name": alias_note})
    final_ds["Radiosonde_Temperature_K"].attrs.update(
        {"units": "K", "source": str(metadata["source"]), "compatibility_note": alias_note}
    )
    final_ds["Radiosonde_Pressure_hPa"].attrs.update(
        {"units": "hPa", "source": str(metadata["source"]), "compatibility_note": alias_note}
    )

    final_ds.attrs.update(
        {
            "thermodynamic_profile_available": "true",
            "thermodynamic_profile_source_type": str(metadata.get("source_type", source_type)),
            "thermodynamic_profile_source": str(metadata.get("source", source_name)),
            "thermodynamic_profile_datetime_utc": str(
                metadata.get("target_datetime_utc", metadata.get("analysis_datetime_utc", ""))
            ),
            "thermodynamic_profile_time_delta_hours": float(metadata.get("time_delta_hours", np.nan)),
            "thermodynamic_profile_station_id": str(metadata.get("station_id", "")),
            "thermodynamic_profile_doi": str(metadata.get("doi", "")),
        }
    )
    return final_ds


def _mark_standard_fallback(final_ds: xr.Dataset) -> xr.Dataset:
    final_ds.attrs.update(
        {
            "thermodynamic_profile_available": "false",
            "thermodynamic_profile_source_type": "ussa76",
            "thermodynamic_profile_source": "US Standard Atmosphere 1976 deferred to Level 2",
            "thermodynamic_profile_datetime_utc": "",
            "thermodynamic_profile_time_delta_hours": np.nan,
            "thermodynamic_profile_station_id": "",
            "thermodynamic_profile_doi": "",
        }
    )
    return final_ds


def integrate_thermodynamics(final_ds: xr.Dataset, config: Mapping[str, Any], logger: logging.Logger) -> xr.Dataset:
    """Resolve thermodynamics in priority order: radiosonde -> ERA5 -> USSA76.

    Network acquisition is delegated to :mod:`milgrau.io`. Level 1 persists the
    selected external profile and provenance; if both external sources fail, the
    deterministic US Standard Atmosphere 1976 fallback is applied later by
    Level 2 on its exact lidar altitude grid.
    """
    dt_utc = pd.to_datetime(final_ds.time.values[len(final_ds.time) // 2])
    station_id = str(config.get("radiosonde", {}).get("station_id", config.get("location", {}).get("station_id", "83779")))

    try:
        df_radio = fetch_wyoming_radiosonde(dt_utc, station_id, logger, config=config)
    except Exception as exc:
        logger.warning(f"  -> Radiosonde retrieval failed: {exc}")
        df_radio = None

    if df_radio is not None and not df_radio.empty:
        try:
            final_ds = _add_profile_to_level1(
                final_ds,
                df_radio,
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
            logger.info(f"  -> Radiosonde profile integrated. CPT: {cpt:.2f} km | LRT: {lrt:.2f} km")
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
        latitude = site_cfg.get("latitude", config.get("physics", {}).get("latitude"))
        longitude = site_cfg.get("longitude", config.get("physics", {}).get("longitude"))
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
                    final_ds = _add_profile_to_level1(
                        final_ds,
                        df_era5,
                        source_type="era5",
                        source_name="Copernicus Climate Change Service ERA5 pressure-level reanalysis",
                    )
                    logger.info("  -> ERA5 thermodynamic profile integrated into Level 1.")
                    return final_ds
                except Exception as exc:
                    logger.warning(f"  -> ERA5 profile was unusable after retrieval: {exc}")

    logger.warning("  -> External thermodynamics unavailable. Level 2 will use US Standard Atmosphere 1976.")
    return _mark_standard_fallback(final_ds)
