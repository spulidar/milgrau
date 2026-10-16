"""Lazy NetCDF access, coordinate selection, bounded display reductions."""
from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
from typing import Any

LEVEL0_VERTICAL_DIMS = ("points", "range_bin", "bins")
ALTITUDE_DIMS = ("altitude", "height", "range")
PREFERRED_VARIABLES = [
    "Raw_Lidar_Data",
    "Background_Profile",
    "range_corrected_signal",
    "range_corrected_signal_error",
    "corrected_signal",
    "glued_range_corrected_signal",
    "glued_corrected_signal",
    "glued_range_corrected_signal_mean",
    "glued_corrected_signal_mean",
    "scaled_molecular_range_corrected_signal",
    "scattering_ratio_mean",
    "scattering_ratio_error_mean",
    "aerosol_backscatter_mean",
    "aerosol_extinction_mean",
    "aerosol_backscatter",
    "aerosol_backscatter_error",
    "aerosol_extinction",
    "aerosol_extinction_error",
    "molecular_backscatter",
    "molecular_extinction",
]

def open_dataset(path: str) -> xr.Dataset:
    """Open a lazy dataset. Callers must close it (normally using 'with')."""
    return xr.open_dataset(path, decode_times=True, mask_and_scale=True, cache=False)


def coord_name(obj: xr.Dataset | xr.DataArray, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in obj.dims or name in getattr(obj, "coords", {}):
            return name
    return None


def channel_name(obj: xr.Dataset | xr.DataArray) -> str | None:
    return coord_name(obj, ("channel", "channels"))


def time_name(obj: xr.Dataset | xr.DataArray) -> str | None:
    return coord_name(obj, ("time", "Time", "Raw_Data_Start_Time", "block_time", "profile"))


def vertical_name(obj: xr.Dataset | xr.DataArray, level: str) -> str | None:
    if level == "Level 0":
        return coord_name(obj, LEVEL0_VERTICAL_DIMS)
    return coord_name(obj, ALTITUDE_DIMS + LEVEL0_VERTICAL_DIMS)


def wavelength_coord_name(obj: xr.Dataset | xr.DataArray) -> str | None:
    return coord_name(obj, ("wavelength", "wavelength_nm"))


def coord_values(ds: xr.Dataset, name: str | None) -> list[Any]:
    if name and name in ds.variables:
        return [value.item() if hasattr(value, "item") else value for value in np.asarray(ds[name].values).ravel()]
    return []


def wavelength_values(ds: xr.Dataset) -> list[Any]:
    name = wavelength_coord_name(ds)
    return coord_values(ds, name) if name else []


def numeric_variables(ds: xr.Dataset) -> list[str]:
    variables = [name for name, da in ds.data_vars.items() if da.ndim <= 5 and np.issubdtype(da.dtype, np.number)]
    return [name for name in PREFERRED_VARIABLES if name in variables] + [name for name in variables if name not in PREFERRED_VARIABLES]


def label(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def format_channel_name(raw_name: Any) -> str:
    try:
        parts = str(raw_name).split(".")
        return f"{int(parts[0])}nm {parts[1]}"
    except Exception:
        return str(raw_name)


def wavelength_label(wavelength: Any) -> str:
    try:
        return f"{int(float(wavelength))} nm"
    except Exception:
        return str(wavelength)


def date_title(ds: xr.Dataset) -> str:
    if "time" not in ds.coords:
        start = ds.attrs.get("RawData_Start_Date", "")
        start_time = ds.attrs.get("RawData_Start_Time_UT", "")
        stop_time = ds.attrs.get("RawData_Stop_Time_UT", "")
        if start and start_time:
            return f"{start} {start_time}-{stop_time} UT"
        return str(ds.attrs.get("Session_ID", "Unknown date"))
    try:
        times = pd.to_datetime(ds["time"].values)
        return f"{times.min().strftime('%d %b %Y - %H:%M')} to {times.max().strftime('%H:%M')} UTC"
    except Exception:
        return "Unknown date"


def select_coord(da: xr.DataArray, dim: str | None, value: Any) -> xr.DataArray:
    if dim is None or value is None or dim not in da.dims:
        return da
    if dim not in da.coords:
        return da.isel({dim: 0})
    coord_labels = [label(coord_value) for coord_value in np.asarray(da[dim].values).ravel()]
    index = coord_labels.index(label(value)) if label(value) in coord_labels else 0
    return da.isel({dim: index})


def select_wavelength(da: xr.DataArray, wavelength: Any | None) -> xr.DataArray:
    return select_coord(da, wavelength_coord_name(da), wavelength)


def vertical_values(obj: xr.Dataset | xr.DataArray, level: str) -> np.ndarray:
    vdim = vertical_name(obj, level)
    if vdim is None:
        size = int(obj.shape[-1] if isinstance(obj, xr.DataArray) and obj.shape else 0)
        return np.arange(size, dtype=float)
    if level == "Level 0":
        return np.arange(int(obj.sizes.get(vdim, 0)), dtype=float)
    if vdim in obj.coords:
        values = np.asarray(obj[vdim].values, dtype=float)
        if values.size and np.nanmax(values) > 100.0:
            return values / 1000.0
        return values
    return np.arange(int(obj.sizes.get(vdim, 0)), dtype=float)


def altitude_km(obj: xr.Dataset | xr.DataArray) -> np.ndarray:
    return vertical_values(obj, "Level 1")


def altitude_m(obj: xr.Dataset | xr.DataArray) -> np.ndarray:
    vdim = vertical_name(obj, "Level 1")
    if vdim and vdim in obj.coords:
        values = np.asarray(obj[vdim].values, dtype=float)
    else:
        values = altitude_km(obj)
    if values.size and np.nanmax(values) <= 100.0:
        return values * 1000.0
    return values


def smooth_profile(values: np.ndarray | xr.DataArray, bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if bins <= 1 or arr.size < 3:
        return arr
    return pd.Series(arr).rolling(window=int(bins), min_periods=1, center=True).mean().to_numpy(dtype=np.float64)


def reduce_to_vertical_profile(da: xr.DataArray, level: str, smooth_bins: int = 1) -> xr.DataArray:
    vdim = vertical_name(da, level) or (da.dims[-1] if da.dims else None)
    if vdim is None:
        return da
    reduce_dims = [dim for dim in da.dims if dim != vdim]
    tdim = time_name(da)
    if reduce_dims and tdim in reduce_dims and da.sizes[tdim] > 128:
        # Accumulate sums/counts by profile chunks, never materializing an
        # entire time-altitude cube for an ordinary mean vertical profile.
        weighted_sum = None
        valid_count = None
        for offset in range(0, da.sizes[tdim], 128):
            piece = da.isel({tdim: slice(offset, offset + 128)})
            partial_sum = np.asarray(piece.sum(dim=reduce_dims, skipna=True).values, dtype=float)
            partial_count = np.asarray(piece.count(dim=reduce_dims).values, dtype=np.int64)
            weighted_sum = partial_sum if weighted_sum is None else weighted_sum + partial_sum
            valid_count = partial_count if valid_count is None else valid_count + partial_count
        vals = np.divide(
            weighted_sum,
            valid_count,
            out=np.full_like(weighted_sum, np.nan, dtype=float),
            where=valid_count > 0,
        )
        coords = {vdim: da[vdim]} if vdim in da.coords else None
        reduced = xr.DataArray(vals, dims=(vdim,), coords=coords, attrs=da.attrs, name=da.name)
    else:
        reduced = da.mean(dim=reduce_dims, skipna=True) if reduce_dims else da
    if smooth_bins > 1:
        reduced = xr.DataArray(
            smooth_profile(reduced.values, smooth_bins),
            dims=reduced.dims,
            coords=reduced.coords,
            attrs=reduced.attrs,
            name=reduced.name,
        )
    return reduced


def reduce_to_altitude_profile(da: xr.DataArray, smooth_bins: int = 1) -> xr.DataArray:
    return reduce_to_vertical_profile(da, "Level 1", smooth_bins)


def error_of_mean(err_da: xr.DataArray) -> xr.DataArray:
    tdim = time_name(err_da)
    if not tdim or tdim not in err_da.dims:
        return err_da
    n_profiles = max(int(err_da.sizes.get(tdim, 1)), 1)
    if n_profiles <= 128:
        return np.sqrt((err_da**2).sum(dim=tdim, skipna=True)) / n_profiles
    accumulated = None
    for offset in range(0, n_profiles, 128):
        chunk = err_da.isel({tdim: slice(offset, offset + 128)})
        part = (chunk**2).sum(dim=tdim, skipna=True)
        accumulated = part if accumulated is None else accumulated + part
    return np.sqrt(accumulated) / n_profiles


def robust_positive_xlim(values: np.ndarray, default_max: float = 6.0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, default_max
    high = float(np.nanpercentile(finite, 99.0))
    high = max(1.5, min(max(default_max, high * 1.15), 20.0))
    return 0.0, high


def robust_centered_xlim(values: np.ndarray, default_abs: float, percentile: float = 98.0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -default_abs, default_abs
    lim = float(np.nanpercentile(np.abs(finite), percentile))
    lim = max(float(default_abs), lim * 1.35)
    return -lim, lim


def parse_color_range(text: str) -> tuple[float | None, float | None]:
    if not text.strip():
        return None, None
    try:
        left, right = text.split(",", 1)
        return float(left), float(right)
    except Exception:
        return None, None


def robust_color_limits(z: np.ndarray, manual_range: str, robust: bool, low: float, high: float) -> tuple[float | None, float | None]:
    zmin, zmax = parse_color_range(manual_range)
    if zmin is not None or zmax is not None:
        return zmin, zmax
    if not robust:
        return None, None
    values = np.asarray(z, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None, None
    lo = float(np.nanpercentile(finite, low))
    hi = float(np.nanpercentile(finite, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None, None
    return lo, hi


def raw_time_strings(ds: xr.Dataset, length: int) -> list[str] | None:
    if "Raw_Data_Start_Time" not in ds:
        return None
    try:
        values = ds["Raw_Data_Start_Time"].values
        if np.asarray(values).ndim == 2:
            values = np.asarray(values)[:, 0]
        if np.issubdtype(values.dtype, np.datetime64):
            times = pd.to_datetime(values)
        else:
            raw_date = str(ds.attrs.get("RawData_Start_Date", ""))
            raw_time = str(ds.attrs.get("RawData_Start_Time_UT", ""))
            if len(raw_date) == 8 and len(raw_time) == 6:
                reference = pd.Timestamp(f"{raw_date}{raw_time}", tz="UTC")
                times = pd.to_datetime(reference + pd.to_timedelta(np.asarray(values, dtype=float), unit="s"))
            else:
                times = pd.to_datetime(np.asarray(values, dtype=float), unit="s", utc=True)
        return [t.strftime("%H:%M:%S UTC") for t in times[:length]]
    except Exception:
        return None


def x_axis_values(da: xr.DataArray, level: str, ds: xr.Dataset | None = None) -> tuple[np.ndarray | pd.DatetimeIndex, str, list[str] | None]:
    if not da.dims:
        return np.arange(1), "profile index", None
    xdim = da.dims[0]
    n = int(da.sizes.get(xdim, 0))
    if level == "Level 0":
        hover_time = raw_time_strings(ds, n) if ds is not None and xdim == "time" else None
        if xdim == "time_bck":
            return np.arange(n), "background profile index", hover_time
        return np.arange(n), "profile index", hover_time
    if xdim in da.coords:
        return da[xdim].values, "Time (UTC)" if xdim in ("time", "block_time") else xdim, None
    return np.arange(n), xdim, None


def reduce_for_heatmap(
    da: xr.DataArray,
    level: str,
    channel: Any | None,
    wavelength: Any | None,
    max_vertical: float,
    max_points: int = 900,
) -> xr.DataArray:
    da = select_coord(da, channel_name(da), channel)
    da = select_wavelength(da, wavelength)
    vdim = vertical_name(da, level)
    if vdim and vdim in da.dims:
        try:
            values = vertical_values(da, level)
            da = da.isel({vdim: values <= max_vertical})
        except Exception:
            pass
    tdim = time_name(da)
    if level == "Level 0":
        tdim = "time_bck" if "time_bck" in da.dims else ("time" if "time" in da.dims else tdim)
    keep = {dim for dim in (tdim, vdim) if dim and dim in da.dims}
    for dim in list(da.dims):
        if dim not in keep:
            da = da.isel({dim: 0})
    if da.ndim == 1:
        da = da.expand_dims({"profile": [0]})
    while da.ndim > 2:
        da = da.isel({da.dims[-1]: 0})
    slices = {}
    for dim in da.dims:
        if da.sizes[dim] > max_points:
            slices[dim] = slice(None, None, int(np.ceil(da.sizes[dim] / max_points)))
    return da.isel(slices) if slices else da
