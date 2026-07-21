"""Boundary-layer and radiosonde integration for Level 1 products."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr

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


def integrate_thermodynamics(final_ds: xr.Dataset, config: Mapping[str, Any], logger: logging.Logger) -> xr.Dataset:
    """Add radiosonde thermodynamics and WMO tropopause diagnostics to Level 1."""
    try:
        dt_utc = pd.to_datetime(final_ds.time.values[len(final_ds.time) // 2])
        station_id = str(config.get("radiosonde", {}).get("station_id", config.get("location", {}).get("station_id", "83779")))
        df_radio = fetch_wyoming_radiosonde(dt_utc, station_id, logger, config=config)
        if df_radio is None or df_radio.empty:
            logger.warning("  -> Radiosonde unavailable. Level 1 will keep surface-only thermodynamics.")
            final_ds.attrs.update({"radiosonde_station_id": station_id, "radiosonde_available": "false", "tropopause_cpt_km": -999.0, "tropopause_lrt_km": -999.0})
            return final_ds

        required_cols = {"height", "temperature", "pressure"}
        missing = sorted(required_cols - set(df_radio.columns))
        if missing:
            raise KeyError(f"Radiosonde data is missing required columns: {missing}")

        df_radio = df_radio.dropna(subset=["height", "temperature", "pressure"]).drop_duplicates(subset=["height"], keep="first").sort_values("height")
        if df_radio.empty:
            raise ValueError("Radiosonde data became empty after cleaning.")

        final_ds = final_ds.assign_coords(radiosonde_altitude=("radiosonde_altitude", df_radio["height"].values.astype(np.float64)))
        final_ds["radiosonde_altitude"].attrs.update({"units": "m", "long_name": "Radiosonde altitude above mean sea level"})
        final_ds["Radiosonde_Temperature_K"] = (("radiosonde_altitude",), (df_radio["temperature"].values.astype(np.float64) + 273.15).astype(np.float32))
        final_ds["Radiosonde_Pressure_hPa"] = (("radiosonde_altitude",), df_radio["pressure"].values.astype(np.float32))
        final_ds["Radiosonde_Temperature_K"].attrs.update({"units": "K", "long_name": "Radiosonde air temperature", "source": "Wyoming Upper Air sounding"})
        final_ds["Radiosonde_Pressure_hPa"].attrs.update({"units": "hPa", "long_name": "Radiosonde atmospheric pressure", "source": "Wyoming Upper Air sounding"})

        cpt, lrt = calculate_tropopause_heights(df_radio)
        cpt = finite_or_fill(cpt)
        lrt = finite_or_fill(lrt)
        final_ds.attrs.update({"radiosonde_station_id": station_id, "radiosonde_available": "true", "tropopause_cpt_km": cpt, "tropopause_lrt_km": lrt})
        logger.info(f"  -> Sounding integrated. CPT: {cpt:.2f} km | LRT: {lrt:.2f} km")
        return final_ds
    except Exception as exc:
        logger.warning(f"  -> Sounding integration incomplete: {exc}")
        final_ds.attrs.update({"radiosonde_available": "false", "tropopause_cpt_km": -999.0, "tropopause_lrt_km": -999.0})
        return final_ds
