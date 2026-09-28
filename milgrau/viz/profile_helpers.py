"""Small plotting helpers shared by the canonical Level 2 QA panels."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.signal import savgol_filter


def altitude_to_km(altitude_values: np.ndarray | xr.DataArray | list[float]) -> np.ndarray:
    """Return altitude in kilometres, accepting coordinates stored in metres or km."""
    altitude = np.asarray(altitude_values, dtype=float)
    if altitude.size == 0:
        return altitude
    return altitude / 1000.0 if np.nanmax(altitude) > 100.0 else altitude


def format_wavelength_label(wavelength_nm: int | float | str) -> str:
    """Return a compact wavelength label such as ``532 nm``."""
    return f"{int(float(wavelength_nm))} nm"


def get_wavelength_values(ds_l2: xr.Dataset) -> list[int]:
    """Return valid integer wavelength-coordinate values."""
    if "wavelength" not in ds_l2.coords:
        return []
    values: list[int] = []
    for wavelength in ds_l2["wavelength"].values:
        try:
            values.append(int(wavelength))
        except (TypeError, ValueError):
            continue
    return values


def smooth_for_plot(values: np.ndarray | xr.DataArray, bins: int) -> np.ndarray:
    """Return a smoothed copy for visualization without changing saved products."""
    array = np.asarray(values, dtype=np.float64)
    if bins <= 2 or array.size < 5:
        return array.copy()
    window = int(bins)
    if window % 2 == 0:
        window += 1
    window = min(window, array.size if array.size % 2 == 1 else array.size - 1)
    if window < 5:
        return array.copy()
    finite = np.isfinite(array)
    if finite.sum() < window:
        return array.copy()
    indices = np.arange(array.size)
    filled = array.copy()
    filled[~finite] = np.interp(indices[~finite], indices[finite], array[finite])
    smoothed = savgol_filter(
        filled,
        window_length=window,
        polyorder=min(3, window - 2),
        mode="interp",
    )
    smoothed[~finite] = np.nan
    return smoothed


def display_scale_factor(
    analog: np.ndarray,
    photon: np.ndarray,
    start: int = 1000,
    stop: int = 1500,
) -> tuple[float, tuple[int, int]]:
    """Return a bounded PC/analog scale factor used only for QA display."""
    size = min(np.asarray(analog).size, np.asarray(photon).size)
    start = max(0, min(int(start), max(size - 2, 0)))
    stop = max(start + 1, min(int(stop), size))
    analog_window = np.asarray(analog[start:stop], dtype=np.float64)
    photon_window = np.asarray(photon[start:stop], dtype=np.float64)
    valid = np.isfinite(analog_window) & np.isfinite(photon_window)
    denominator = float(np.nansum(analog_window[valid])) if valid.any() else np.nan
    numerator = float(np.nansum(photon_window[valid])) if valid.any() else np.nan
    if (
        not np.isfinite(denominator)
        or abs(denominator) <= 1.0e-30
        or not np.isfinite(numerator)
    ):
        return 1.0, (start, stop)
    return float(numerator / denominator), (start, stop)


def infer_l1_channels_for_wavelength(
    ds_l1: xr.Dataset | None,
    wavelength_nm: int | float,
) -> tuple[str | None, str | None]:
    """Infer analog and photon-counting channel names for a wavelength."""
    if ds_l1 is None or "channel" not in ds_l1.coords:
        return None, None
    wavelength = str(int(wavelength_nm))
    channels = [str(channel) for channel in ds_l1["channel"].values]
    analog = next(
        (
            channel
            for channel in channels
            if channel.startswith(f"{wavelength}.") and channel.upper().endswith(".AN")
        ),
        None,
    )
    photon = next(
        (
            channel
            for channel in channels
            if channel.startswith(f"{wavelength}.")
            and (channel.upper().endswith(".PC") or channel.upper().endswith(".PH"))
        ),
        None,
    )
    return analog, photon


__all__ = [
    "altitude_to_km",
    "display_scale_factor",
    "format_wavelength_label",
    "get_wavelength_values",
    "infer_l1_channels_for_wavelength",
    "smooth_for_plot",
]
