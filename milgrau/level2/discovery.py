"""Discovery and channel-resolution helpers for Level 2 processing."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any

import xarray as xr

from milgrau.io.paths import processed_data_root


def discover_level1_files(config: Mapping[str, Any], root_dir: str | Path | None = None) -> list[Path]:
    """Discover Level 1 RCS NetCDF files available for LEBEAR processing."""
    return sorted(processed_data_root(config, root_dir=root_dir).rglob("*_level1_rcs.nc"))


def infer_channel_pair(ds_l1: xr.Dataset, wavelength_nm: int) -> tuple[str | None, str | None]:
    """Infer Analog and Photon Counting channel names for one wavelength."""
    channels = [str(channel) for channel in ds_l1["channel"].values]
    prefix = f"{int(wavelength_nm)}."
    analog = next((channel for channel in channels if channel.startswith(prefix) and channel.upper().endswith(".AN")), None)
    photon = next(
        (
            channel
            for channel in channels
            if channel.startswith(prefix) and (channel.upper().endswith(".PC") or channel.upper().endswith(".PH"))
        ),
        None,
    )
    return analog, photon
