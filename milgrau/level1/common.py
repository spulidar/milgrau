"""Shared helpers for Level 1 processing."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import xarray as xr


def incremental_enabled(config: Mapping[str, Any]) -> bool:
    """Return the explicitly configured incremental-processing policy."""
    processing = config.get("processing")
    if not isinstance(processing, Mapping):
        raise KeyError("Configuration processing section is required.")
    if "incremental" not in processing:
        raise KeyError("Missing required configuration: processing.incremental")
    value = processing["incremental"]
    if not isinstance(value, bool):
        raise ValueError("Configuration processing.incremental must be a boolean.")
    return value


def finite_or_fill(value: Any, fill_value: float = -999.0) -> float:
    """Convert a numeric value to float, replacing invalid values by a fill value."""
    try:
        value = float(value)
        return value if np.isfinite(value) else float(fill_value)
    except (TypeError, ValueError, OverflowError):
        return float(fill_value)


def level0_dark_current_available(ds: xr.Dataset, channel_index: int) -> bool:
    """Return whether a Level 0 dark-current profile is available for one channel."""
    if "Background_Profile" not in ds:
        return False
    if "Background_Profile_Available" in ds:
        try:
            return bool(int(ds["Background_Profile_Available"].isel(channel=channel_index).values) == 1)
        except (IndexError, TypeError, ValueError):
            return False
    return True


def diagnostic_vector(diagnostics: dict[str, Any], name: str, time_coord: xr.DataArray) -> xr.DataArray:
    """Return a per-time diagnostic vector aligned with the Level 1 time coordinate."""
    value = diagnostics[name]
    if isinstance(value, xr.DataArray):
        return value.rename({"range": "altitude"}) if "range" in value.dims else value
    return xr.DataArray(np.full(time_coord.size, value), dims=["time"], coords={"time": time_coord})
