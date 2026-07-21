"""Level 0 ingestion helpers for Level 1 processing."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.io.contracts import validate_level0_contract


def _decode_level0_time_axis(ds: xr.Dataset) -> pd.DatetimeIndex:
    """Decode SCC-style relative start times into absolute UTC timestamps."""
    raw_start = ds["Raw_Data_Start_Time"]
    values = np.asarray(raw_start.values)
    if values.ndim == 2:
        values = values[:, 0]
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values, utc=True)

    raw_date = str(ds.attrs.get("RawData_Start_Date", ""))
    raw_time = str(ds.attrs.get("RawData_Start_Time_UT", ""))
    if len(raw_date) == 8 and len(raw_time) == 6:
        reference = pd.Timestamp(f"{raw_date}{raw_time}", tz="UTC")
        return pd.to_datetime(reference + pd.to_timedelta(values.astype(float), unit="s"))
    return pd.to_datetime(values.astype(float), unit="s", utc=True)


def load_and_prepare_level0(nc_path: str | Path, logger: logging.Logger) -> tuple[xr.Dataset, np.ndarray]:
    """Load one Level 0 NetCDF file and standardize its coordinates."""
    try:
        ds = xr.open_dataset(nc_path)
        ds.load()
        validate_level0_contract(ds)

        time_dt = _decode_level0_time_axis(ds)
        ds = ds.assign_coords(time=time_dt)

        dz_values = np.asarray(ds["Raw_Data_Range_Resolution"].values, dtype=float)
        dz_values = dz_values[np.isfinite(dz_values)]
        if dz_values.size == 0:
            raise ValueError("Raw_Data_Range_Resolution contains no finite values.")
        dz = float(dz_values[0])
        if not np.allclose(dz_values, dz, rtol=0.0, atol=1e-6):
            logger.warning(f"  -> Not all channels have identical range resolution. Using the first value: {dz:.6f} m.")

        z_arr = np.arange(ds.sizes["points"], dtype=np.float64) * dz
        channel_strings = ds["channel_string"].values.astype(str)
        ds = ds.rename({"points": "altitude", "channels": "channel"})
        ds = ds.assign_coords(altitude=z_arr, channel=channel_strings)
        ds["altitude"].attrs.update({"units": "m", "long_name": "Altitude above station"})
        logger.info(
            f"  -> Level 0 ingestion successful: {ds.sizes.get('time', 0)} profiles, "
            f"{ds.sizes.get('channel', 0)} channels, {ds.sizes.get('altitude', 0)} bins."
        )
        return ds, z_arr
    except Exception as exc:
        logger.error(f"  -> Failed to ingest Level 0 file {nc_path}: {exc}")
        raise
