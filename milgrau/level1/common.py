"""Shared helpers for Level 1 processing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from milgrau.io.paths import level1_output_path as canonical_level1_output_path


def incremental_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether incremental processing is enabled."""
    return bool(config.get("processing", {}).get("incremental", False))


def level1_output_path(nc_file: str | Path, config: Mapping[str, Any]) -> Path:
    """Return the Level 1 output path for one Level 0 NetCDF file."""
    return canonical_level1_output_path(nc_file, config)


def finite_or_fill(value: Any, fill_value: float = -999.0) -> float:
    """Convert a numeric value to float, replacing invalid values by a fill value."""
    try:
        value = float(value)
        return value if np.isfinite(value) else float(fill_value)
    except Exception:
        return float(fill_value)


def get_channel_constant(
    channels_config: Mapping[str, Sequence[float] | Mapping[str, float | int]],
    ch_name: str,
    logger: logging.Logger,
) -> tuple[float, int, float]:
    """Return instrumental constants for one channel."""
    if ch_name not in channels_config:
        logger.warning(f"  -> Channel {ch_name} is missing from physics.channels. Using neutral correction constants.")
    constants = channels_config.get(ch_name, {"deadtime_us": 0.0, "bin_shift_bins": 0, "background_offset": 0.0})
    if isinstance(constants, Mapping):
        deadtime = constants["deadtime_us"]
        shift = constants["bin_shift_bins"]
        bg_offset = constants["background_offset"]
    else:
        deadtime, shift, bg_offset = constants
    return float(deadtime), int(shift), float(bg_offset)


def level0_dark_current_available(ds: xr.Dataset, channel_index: int) -> bool:
    """Return whether a Level 0 dark-current profile is available for one channel."""
    if "Background_Profile" not in ds:
        return False
    if "Background_Profile_Available" in ds:
        try:
            return bool(int(ds["Background_Profile_Available"].isel(channel=channel_index).values) == 1)
        except Exception:
            return False
    return True


def diagnostic_vector(diagnostics: dict[str, Any], name: str, time_coord: xr.DataArray) -> xr.DataArray:
    """Return a per-time diagnostic vector aligned with the Level 1 time coordinate."""
    value = diagnostics[name]
    if isinstance(value, xr.DataArray):
        return value.rename({"range": "altitude"}) if "range" in value.dims else value
    return xr.DataArray(np.full(time_coord.size, value), dims=["time"], coords={"time": time_coord})
