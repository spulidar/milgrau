"""Shared helpers for Level 1 processing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

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
    channels_config: Mapping[str, Mapping[str, float | int]],
    ch_name: str,
    logger: logging.Logger,
) -> tuple[float, int, float]:
    """Return required instrumental correction constants for one channel.

    Missing calibration is a configuration error. Level 1 must never silently
    replace an unknown channel with neutral dead-time, shift, or background
    corrections because that changes the scientific product while appearing
    successful.
    """
    del logger  # retained temporarily for call-site compatibility
    if ch_name not in channels_config:
        raise KeyError(
            f"Missing required instrument calibration for channel {ch_name!r}; "
            "the resolved station profile must provide deadtime_us, bin_shift_bins, and background_offset."
        )
    constants = channels_config[ch_name]
    if not isinstance(constants, Mapping):
        raise TypeError(
            f"Instrument calibration for channel {ch_name!r} must use named fields; positional correction lists are not supported."
        )
    required = {"deadtime_us", "bin_shift_bins", "background_offset"}
    missing = sorted(required - set(constants))
    unknown = sorted(set(constants) - required)
    if missing or unknown:
        raise ValueError(
            f"Instrument calibration for channel {ch_name!r} must contain exactly {sorted(required)}; "
            f"missing={missing}, unknown={unknown}."
        )
    deadtime = float(constants["deadtime_us"])
    shift_raw = constants["bin_shift_bins"]
    if isinstance(shift_raw, bool) or not isinstance(shift_raw, (int, np.integer)):
        raise ValueError(f"Instrument calibration bin_shift_bins for channel {ch_name!r} must be an integer.")
    bg_offset = float(constants["background_offset"])
    if not np.isfinite(deadtime) or deadtime < 0.0:
        raise ValueError(f"Instrument calibration deadtime_us for channel {ch_name!r} must be finite and non-negative.")
    if not np.isfinite(bg_offset):
        raise ValueError(f"Instrument calibration background_offset for channel {ch_name!r} must be finite.")
    return deadtime, int(shift_raw), bg_offset


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
