"""Read-only scientific arrays for interactive L2 displays.

These accessors never perform lidar inversion or invent values in unsupported
regions. The caller is responsible for holding the underlying Dataset open.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

OPTICAL_FIELDS = {
    "Retroespalhamento": {
        "nominal": "aerosol_backscatter_nominal_block",
        "mc": "aerosol_backscatter_mc_mean",
        "lower": "aerosol_backscatter_mc_q025",
        "upper": "aerosol_backscatter_mc_q975",
        "sigma": "aerosol_backscatter_mc_std",
        "unit": "Mm⁻¹ sr⁻¹",
    },
    "Extinção": {
        "nominal": "aerosol_extinction_nominal_block",
        "mc": "aerosol_extinction_mc_mean",
        "lower": "aerosol_extinction_mc_q025",
        "upper": "aerosol_extinction_mc_q975",
        "sigma": "aerosol_extinction_mc_std",
        "unit": "Mm⁻¹",
    },
}
OPTICAL_SCALE = 1.0e6


def wavelength_options(ds: xr.Dataset) -> list[int]:
    """Only scientifically processed wavelengths, not failed requests."""
    if "wavelength" not in ds.coords:
        return []
    return [int(v) for v in np.asarray(ds["wavelength"].values).reshape(-1)]


def scenario_options(ds: xr.Dataset) -> list[float]:
    if "residual_fraction" not in ds.coords:
        return [0.0]
    return [float(v) for v in np.asarray(ds["residual_fraction"].values).reshape(-1)]


def optical_field(
    ds: xr.Dataset, quantity: str, source: str, wavelength: int,
    residual_fraction: float = 0.0,
) -> xr.DataArray:
    """Return one wavelength / scenario without loading the full product."""
    field = OPTICAL_FIELDS[quantity][source]
    da = ds[field].sel(wavelength=wavelength)
    if "residual_fraction" in da.dims:
        da = da.sel(residual_fraction=residual_fraction)
    return da


def support_mask(
    ds: xr.Dataset, quantity: str, wavelength: int,
    residual_fraction: float = 0.0,
) -> xr.DataArray:
    """MC display requires both native retrieval support and MC coverage."""
    nominal = optical_field(ds, quantity, "nominal", wavelength)
    mask = np.isfinite(nominal)
    if "mc_valid_fraction" in ds:
        fraction = ds["mc_valid_fraction"].sel(
            wavelength=wavelength, residual_fraction=residual_fraction
        )
        mask = mask & np.isfinite(fraction) & (fraction > 0)
    return mask


def bounded_view(
    da: xr.DataArray, *, start: int = 0, stop: int | None = None,
    top_km: float = 30.0, max_time: int = 500, max_altitude: int = 900,
) -> xr.DataArray:
    """Select the requested physical span before loading a plot-size array."""
    if "block_time" in da.dims:
        count = da.sizes["block_time"]
        left = max(0, min(int(start), count))
        right = count if stop is None else max(left, min(int(stop), count))
        da = da.isel(block_time=slice(left, right))
    if "altitude" in da.dims:
        da = da.sel(altitude=slice(None, float(top_km) * 1000.0))
    steps = {}
    for dim, maximum in (("block_time", max_time), ("altitude", max_altitude)):
        if dim in da.dims and da.sizes[dim] > maximum:
            steps[dim] = slice(None, None, int(np.ceil(da.sizes[dim] / maximum)))
    return da.isel(steps) if steps else da


def display_block_mask(
    ds: xr.Dataset, shown_times: np.ndarray,
    regimes: list[str] | None = None,
    segments: list[str] | None = None,
) -> np.ndarray:
    """Choose visible blocks without merging or renaming scientific sessions.

    Filtered blocks are represented by NaNs at their original timestamps.
    """
    n = int(ds.sizes["block_time"])
    allowed = np.ones(n, dtype=bool)
    if regimes is not None and "solar_regime" in ds:
        allowed &= np.isin(ds["solar_regime"].values.astype(str), regimes)
    if segments is not None and "segment_id" in ds:
        allowed &= np.isin(ds["segment_id"].values.astype(str), segments)
    return np.isin(np.asarray(shown_times), ds["block_time"].values[allowed])


def time_label(value: object) -> str:
    value = np.datetime64(value, "m")
    return np.datetime_as_string(value, unit="m").replace("T", " ") + " UTC"


__all__ = [
    "OPTICAL_FIELDS", "OPTICAL_SCALE", "wavelength_options",
    "scenario_options", "optical_field", "support_mask",
    "bounded_view", "display_block_mask", "time_label",
]
