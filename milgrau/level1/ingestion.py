"""Level 0 ingestion helpers for Level 1 processing."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.io.contracts import validate_level0_contract


def _decode_level0_time_axis(ds: xr.Dataset) -> pd.DatetimeIndex:
    raw_start = ds["Raw_Data_Start_Time"]
    values = np.asarray(raw_start.values)
    if values.ndim == 2:
        values = values[:, 0]
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values, utc=True).tz_localize(None)
    raw_date = str(ds.attrs.get("RawData_Start_Date", ""))
    raw_time = str(ds.attrs.get("RawData_Start_Time_UT", ""))
    if len(raw_date) == 8 and len(raw_time) == 6:
        reference = pd.Timestamp(f"{raw_date}{raw_time}", tz="UTC")
        return pd.to_datetime(reference + pd.to_timedelta(values.astype(float), unit="s")).tz_localize(None)
    return pd.to_datetime(values.astype(float), unit="s", utc=True).tz_localize(None)


def _native_range_resolutions(ds: xr.Dataset) -> np.ndarray:
    resolutions = np.asarray(ds["Raw_Data_Range_Resolution"].values, dtype=np.float64)
    if resolutions.ndim != 1 or resolutions.size != ds.sizes.get("channels", 0):
        raise ValueError("Raw_Data_Range_Resolution must contain one value per channel.")
    if not np.all(np.isfinite(resolutions)) or np.any(resolutions <= 0.0):
        raise ValueError("Raw_Data_Range_Resolution contains non-finite or non-positive values.")
    return resolutions


def _common_level1_altitude_grid(num_points: int, resolutions_m: np.ndarray) -> np.ndarray:
    if num_points <= 0:
        raise ValueError("Level 0 points dimension must be positive.")
    target_dz = float(np.min(resolutions_m))
    return (np.arange(num_points, dtype=np.float64) + 0.5) * target_dz


def load_and_prepare_level0(nc_path: str | Path, logger: logging.Logger) -> tuple[xr.Dataset, np.ndarray]:
    """Load Level 0 and expose a common center-bin grid for Level 1 output.

    Native per-channel range resolution remains available in
    ``Raw_Data_Range_Resolution`` so LIPANCORA can correct each channel on its
    native grid before interpolation.
    """
    try:
        ds = xr.open_dataset(nc_path)
        ds.load()
        validate_level0_contract(ds)
        time_dt = _decode_level0_time_axis(ds)
        ds = ds.assign_coords(time=time_dt)
        dz_values = _native_range_resolutions(ds)
        z_arr = _common_level1_altitude_grid(ds.sizes["points"], dz_values)
        if not np.allclose(dz_values, dz_values[0], rtol=0.0, atol=1e-6):
            logger.warning(
                "  -> Level 0 channels use different native range resolutions "
                f"({', '.join(f'{value:.6f}' for value in dz_values)} m). "
                f"Corrections will run on each native grid and outputs will be interpolated to {np.min(dz_values):.6f} m."
            )
        channel_strings = ds["channel_string"].values.astype(str)
        ds = ds.rename({"points": "altitude", "channels": "channel"})
        ds = ds.assign_coords(altitude=z_arr, channel=channel_strings)
        ds["altitude"].attrs.update({"units": "m", "long_name": "Altitude above station (range-bin centers)"})
        logger.info(
            f"  -> Level 0 ingestion successful: {ds.sizes.get('time', 0)} profiles, "
            f"{ds.sizes.get('channel', 0)} channels, {ds.sizes.get('altitude', 0)} bins."
        )
        return ds, z_arr
    except Exception as exc:
        logger.error(f"  -> Failed to ingest Level 0 file {nc_path}: {exc}")
        raise
