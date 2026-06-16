"""Runnable Streamlit app for browsing MILGRAU NetCDF products.

The app is read-only: it scans and opens existing files but never writes to the
MILGRAU processed-data directory.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from plotly.subplots import make_subplots

from milgrau.config.loader import load_config
from milgrau.io.paths import LEVEL1_SUFFIX, LEVEL2_SUFFIX, processed_data_root
from milgrau.visualization.style import channel_color

LEVEL_TO_PATH = {
    "Level 0": "level0_path",
    "Level 1": "level1_path",
    "Level 2": "level2_path",
}
PREFERRED_VARIABLES = [
    "glued_range_corrected_signal",
    "range_corrected_signal",
    "corrected_signal",
    "Raw_Lidar_Data",
    "glued_corrected_signal",
    "glued_range_corrected_signal_mean",
    "glued_corrected_signal_mean",
    "scaled_molecular_range_corrected_signal",
    "scattering_ratio_mean",
    "scattering_ratio_error_mean",
    "aerosol_backscatter_mean",
    "aerosol_extinction_mean",
    "aerosol_backscatter",
    "aerosol_extinction",
    "molecular_backscatter",
    "molecular_extinction",
]
DIAGNOSTIC_NAMES = [
    "PBL_Height_km",
    "pc_saturation_fraction",
    "deadtime_clipping_fraction",
    "bin_shift_invalid_fraction",
    "channel_correction_success",
    "dark_current_used",
    "gluing_success_flag",
    "gluing_fallback_flag",
    "gluing_split_altitude_m",
    "gluing_start_altitude_m",
    "gluing_stop_altitude_m",
    "gluing_slope",
    "gluing_intercept",
    "gluing_correlation",
    "gluing_relative_rmse",
    "gluing_relative_bias",
    "rayleigh_reference_success_flag",
    "rayleigh_reference_altitude_m",
    "rayleigh_reference_valid_fraction",
    "rayleigh_calibration_factor",
    "rayleigh_calibration_intercept",
    "valid_retrieval_block_flag",
    "lidar_ratio_assumed_sr",
    "kfs_branch",
]


def safe_key(*parts: Any) -> str:
    """Return a stable Streamlit widget key from arbitrary labels."""

    text = "_".join(str(part) for part in parts if part is not None)
    return re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_") or "widget"


def processed_root_from_config(config_path: str) -> Path:
    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    return processed_data_root(config, root_dir=config_file.parent)


def parse_save_id(save_id: str) -> tuple[date | None, str]:
    try:
        day = datetime.strptime(save_id[:8], "%Y%m%d").date()
    except ValueError:
        day = None
    return day, save_id[-2:] if len(save_id) >= 2 else "--"


def product_paths(product_dir: Path) -> dict[str, str]:
    save_id = product_dir.name
    level0 = product_dir / f"{save_id}.nc"
    if not level0.exists():
        candidates = sorted(
            path
            for path in product_dir.glob("*.nc")
            if not path.name.endswith(LEVEL1_SUFFIX) and not path.name.endswith(LEVEL2_SUFFIX)
        )
        level0 = candidates[0] if candidates else level0
    level1 = product_dir / f"{save_id}{LEVEL1_SUFFIX}"
    level2 = product_dir / f"{save_id}{LEVEL2_SUFFIX}"
    return {
        "level0_path": str(level0) if level0.exists() else "",
        "level1_path": str(level1) if level1.exists() else "",
        "level2_path": str(level2) if level2.exists() else "",
    }


@st.cache_data(show_spinner="Escaneando produtos NetCDF...")
def discover_products(processed_root: str) -> pd.DataFrame:
    root = Path(processed_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame()
    for product_dir in sorted(root.glob("[0-9][0-9][0-9][0-9]/[0-9][0-9]/*")):
        if not product_dir.is_dir():
            continue
        paths = product_paths(product_dir)
        if not any(paths.values()):
            continue
        day, period = parse_save_id(product_dir.name)
        mtimes = [Path(path).stat().st_mtime for path in paths.values() if path]
        rows.append(
            {
                "date": day,
                "year": day.year if day else None,
                "month": day.month if day else None,
                "day": day.day if day else None,
                "save_id": product_dir.name,
                "period": period,
                "product_dir": str(product_dir),
                **paths,
                "available_levels": ", ".join(label for label, key in LEVEL_TO_PATH.items() if paths[key]),
                "modified": datetime.fromtimestamp(max(mtimes)).isoformat(timespec="seconds") if mtimes else "",
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["date", "save_id"], na_position="last").reset_index(drop=True)


@st.cache_resource(show_spinner="Abrindo NetCDF...")
def open_dataset(path: str) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=True, mask_and_scale=True)


def coord_name(obj: xr.Dataset | xr.DataArray, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in obj.dims or name in getattr(obj, "coords", {}):
            return name
    return None


def altitude_name(obj: xr.Dataset | xr.DataArray) -> str | None:
    return coord_name(obj, ("altitude", "height", "range", "points", "range_bin"))


def time_name(obj: xr.Dataset | xr.DataArray) -> str | None:
    return coord_name(obj, ("time", "Time", "Raw_Data_Start_Time", "block_time", "profile"))


def channel_name(obj: xr.Dataset | xr.DataArray) -> str | None:
    return coord_name(obj, ("channel", "channels"))


def coord_values(ds: xr.Dataset, name: str | None) -> list[Any]:
    if name and name in ds.variables:
        return [value.item() if hasattr(value, "item") else value for value in np.asarray(ds[name].values).ravel()]
    return []


def wavelength_coord_name(ds: xr.Dataset | xr.DataArray) -> str | None:
    return coord_name(ds, ("wavelength", "wavelength_nm"))


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
        return str(ds.attrs.get("Measurement_ID", "Unknown date"))
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


def altitude_km(obj: xr.Dataset | xr.DataArray) -> np.ndarray:
    alt = altitude_name(obj)
    if not alt or alt not in obj.coords:
        size = int(obj.sizes.get("altitude", obj.shape[-1] if isinstance(obj, xr.DataArray) and obj.shape else 0))
        return np.arange(size, dtype=float)
    values = np.asarray(obj[alt].values, dtype=float)
    if values.size and np.nanmax(values) > 100.0:
        return values / 1000.0
    return values


def altitude_m(obj: xr.Dataset | xr.DataArray) -> np.ndarray:
    alt = altitude_name(obj)
    values = np.asarray(obj[alt].values, dtype=float) if alt and alt in obj.coords else altitude_km(obj)
    if values.size and np.nanmax(values) <= 100.0:
        return values * 1000.0
    return values


def smooth_profile(values: np.ndarray | xr.DataArray, bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if bins <= 1 or arr.size < 3:
        return arr
    return (
        pd.Series(arr)
        .rolling(window=int(bins), min_periods=1, center=True)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def reduce_to_altitude_profile(da: xr.DataArray, smooth_bins: int = 1) -> xr.DataArray:
    alt = altitude_name(da) or (da.dims[-1] if da.dims else None)
    if alt is None:
        return da
    reduce_dims = [dim for dim in da.dims if dim != alt]
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


def error_of_mean(err_da: xr.DataArray) -> xr.DataArray:
    tdim = time_name(err_da)
    if not tdim or tdim not in err_da.dims:
        return err_da
    n_profiles = max(int(err_da.sizes.get(tdim, 1)), 1)
    return np.sqrt((err_da**2).sum(dim=tdim, skipna=True)) / n_profiles


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
        st.warning("Use range de cor no formato `min,max`, ou deixe vazio para automático.")
        return None, None


def reduce_for_heatmap(
    da: xr.DataArray,
    channel: Any | None,
    wavelength: Any | None,
    max_alt_km: float,
    max_points: int = 900,
) -> xr.DataArray:
    da = select_coord(da, channel_name(da), channel)
    da = select_wavelength(da, wavelength)
    alt = altitude_name(da)
    if alt and alt in da.coords:
        try:
            alt_values = altitude_km(da)
            da = da.isel({alt: alt_values <= max_alt_km})
        except Exception:
            pass
    tdim = time_name(da)
    keep = {dim for dim in (tdim, alt) if dim and dim in da.dims}
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


def heatmap_figure(da: xr.DataArray, title: str, log10: bool, color_range: str) -> go.Figure:
    z = np.asarray(da.values, dtype=float)
    if log10:
        z = np.where(z > 0, np.log10(z), np.nan)
    z = z.T if z.ndim == 2 else np.atleast_2d(z)
    xdim = da.dims[0] if da.dims else "profile"
    ydim = da.dims[-1] if len(da.dims) > 1 else "vertical"
    x = da[xdim].values if xdim in da.coords else np.arange(z.shape[1])
    y = altitude_km(da) if ydim == altitude_name(da) else (da[ydim].values if ydim in da.coords else np.arange(z.shape[0]))
    zmin, zmax = parse_color_range(color_range)
    fig = go.Figure(data=go.Heatmap(x=x, y=y, z=z, zmin=zmin, zmax=zmax, colorscale="Jet", colorbar={"title": "log10" if log10 else da.name}))
    fig.update_layout(title=title, xaxis_title=xdim, yaxis_title="Altitude (km a.g.l.)" if ydim == altitude_name(da) else ydim, margin={"l": 60, "r": 20, "t": 70, "b": 50})
    return fig


def profile_figure(da: xr.DataArray, title: str, smooth_bins: int = 20) -> go.Figure:
    prof = reduce_to_altitude_profile(da, smooth_bins=smooth_bins)
    alt = altitude_km(prof)
    fig = go.Figure(data=go.Scatter(x=np.asarray(prof.values, dtype=float), y=alt, mode="lines", name=prof.name or "mean profile"))
    fig.update_layout(title=title, xaxis_title=prof.name or "value", yaxis_title="Altitude (km a.g.l.)", margin={"l": 60, "r": 20, "t": 70, "b": 50})
    fig.update_yaxes(range=[0, float(np.nanmax(alt)) if alt.size else 1.0])
    return fig


def add_profile_band(fig: go.Figure, y: np.ndarray, mean: np.ndarray, sigma: np.ndarray, name: str, color: str, row: int | None = None, col: int | None = None) -> None:
    lower = np.asarray(mean, dtype=float) - np.asarray(sigma, dtype=float)
    upper = np.asarray(mean, dtype=float) + np.asarray(sigma, dtype=float)
    kwargs = {"row": row, "col": col} if row is not None and col is not None else {}
    fig.add_trace(go.Scatter(x=lower, y=y, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"), **kwargs)
    fig.add_trace(
        go.Scatter(
            x=upper,
            y=y,
            mode="lines",
            fill="tonextx",
            fillcolor="rgba(128,128,128,0.22)",
            line={"width": 0},
            name=name,
            hoverinfo="skip",
        ),
        **kwargs,
    )


def maybe_add_altitude_line(fig: go.Figure, y: float, name: str, color: str, dash: str, row: int, col: int, xspan: tuple[float, float] | None = None) -> None:
    x0, x1 = xspan or (0.0, 1.0)
    fig.add_trace(
        go.Scatter(x=[x0, x1], y=[y, y], mode="lines", line={"color": color, "dash": dash, "width": 1.6}, name=name, hovertemplate=f"{name}: {y:.2f} km<extra></extra>"),
        row=row,
        col=col,
    )


def add_atmospheric_boundaries_plotly(fig: go.Figure, ds: xr.Dataset, max_alt_km: float, row: int, col: int, xspan: tuple[float, float] | None = None) -> None:
    if "PBL_Height_km" in ds:
        try:
            pbl = float(ds["PBL_Height_km"].mean(skipna=True).values)
            if np.isfinite(pbl) and 0 < pbl <= max_alt_km:
                maybe_add_altitude_line(fig, pbl, f"Mean PBL ({pbl:.1f} km)", "crimson", "dash", row, col, xspan)
        except Exception:
            pass
    for attr_name, short, color, dash in (
        ("tropopause_cpt_km", "CPT", "royalblue", "dot"),
        ("tropopause_lrt_km", "LRT", "forestgreen", "dashdot"),
    ):
        try:
            val = float(ds.attrs.get(attr_name, -999.0))
            if np.isfinite(val) and 0 < val <= max_alt_km:
                maybe_add_altitude_line(fig, val, f"{short} ({val:.1f} km)", color, dash, row, col, xspan)
        except Exception:
            pass


def level1_quicklook_figure(
    ds: xr.Dataset,
    channel: Any,
    variable: str,
    max_alt_km: float,
    log10: bool,
    smooth_bins: int,
    color_range: str,
) -> go.Figure:
    data = reduce_for_heatmap(ds[variable], channel, None, max_alt_km)
    error_name = f"{variable}_error"
    if error_name not in ds and variable == "range_corrected_signal":
        error_name = "range_corrected_signal_error"
    data_profile = select_coord(ds[variable], channel_name(ds[variable]), channel)
    alt = altitude_name(data_profile)
    if alt and alt in data_profile.coords:
        data_profile = data_profile.isel({alt: altitude_km(data_profile) <= max_alt_km})
    profile = reduce_to_altitude_profile(data_profile, smooth_bins=smooth_bins)
    alt_prof = altitude_km(profile)
    mean_values = np.asarray(profile.values, dtype=float)
    color = channel_color(str(channel))
    pretty_channel = format_channel_name(channel)
    z = np.asarray(data.values, dtype=float)
    if log10:
        z = np.where(z > 0, np.log10(z), np.nan)
    z = z.T if z.ndim == 2 else np.atleast_2d(z)
    xdim = data.dims[0] if data.dims else "profile"
    ydim = data.dims[-1] if len(data.dims) > 1 else "vertical"
    x = data[xdim].values if xdim in data.coords else np.arange(z.shape[1])
    y = altitude_km(data) if ydim == altitude_name(data) else np.arange(z.shape[0])
    zmin, zmax = parse_color_range(color_range)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.78, 0.22], shared_yaxes=True, horizontal_spacing=0.03, subplot_titles=("", "Mean profile"))
    fig.add_trace(go.Heatmap(x=x, y=y, z=z, zmin=zmin, zmax=zmax, colorscale="Jet", colorbar={"title": "log10 RCS" if log10 else "Intensity [a.u.]"}), row=1, col=1)
    if error_name in ds:
        err = select_coord(ds[error_name], channel_name(ds[error_name]), channel)
        if alt and alt in err.coords:
            err = err.isel({alt: altitude_km(err) <= max_alt_km})
        err_profile = reduce_to_altitude_profile(error_of_mean(err), smooth_bins=smooth_bins)
        add_profile_band(fig, alt_prof, mean_values, np.asarray(err_profile.values, dtype=float), "1σ error", color, row=1, col=2)
    fig.add_trace(go.Scatter(x=mean_values, y=alt_prof, mode="lines", line={"color": color, "width": 2.4}, name="Mean RCS"), row=1, col=2)
    finite = mean_values[np.isfinite(mean_values)]
    xspan = None
    if finite.size:
        xmin = min(0.0, float(np.nanmin(finite)))
        xmax = float(np.nanmax(finite))
        pad = max((xmax - xmin) * 0.15, 1e-12)
        xspan = (xmin - pad, xmax + pad)
        fig.update_xaxes(range=[xspan[0], xspan[1]], row=1, col=2)
    add_atmospheric_boundaries_plotly(fig, ds, max_alt_km, row=1, col=2, xspan=xspan)
    lower_alt = 0.16 if "AN" in pretty_channel else 0.5
    fig.update_yaxes(title_text="Altitude (km a.g.l.)", range=[lower_alt, max_alt_km], row=1, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=1, col=1)
    fig.update_xaxes(title_text="Mean RCS", row=1, col=2)
    fig.update_layout(
        title=f"RCS at {pretty_channel} ({lower_alt:g} - {float(max_alt_km):g} km)<br><sup>{date_title(ds)}</sup>",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.22, "xanchor": "center", "x": 0.5},
        margin={"l": 65, "r": 30, "t": 85, "b": 80},
        height=720,
    )
    return fig


def global_mean_rcs_figure(ds: xr.Dataset, channels: list[Any], max_alt_km: float, smooth_bins: int) -> go.Figure:
    fig = go.Figure()
    for channel in channels:
        if "range_corrected_signal" not in ds:
            continue
        da = select_coord(ds["range_corrected_signal"], channel_name(ds["range_corrected_signal"]), channel)
        alt = altitude_name(da)
        if alt and alt in da.coords:
            da = da.isel({alt: altitude_km(da) <= max_alt_km})
        prof = reduce_to_altitude_profile(da, smooth_bins=smooth_bins)
        fig.add_trace(
            go.Scatter(
                x=np.asarray(prof.values, dtype=float),
                y=altitude_km(prof),
                mode="lines",
                name=format_channel_name(channel),
                line={"color": channel_color(str(channel)), "dash": "solid" if "AN" in str(channel).upper() else "dash", "width": 2.0},
            )
        )
    fig.update_layout(
        title=f"Mean RCS (0 - {max_alt_km:g} km)<br><sup>{date_title(ds)}</sup>",
        xaxis_title="Mean RCS [a.u.]",
        yaxis_title="Altitude (km a.g.l.)",
        height=740,
        margin={"l": 65, "r": 30, "t": 85, "b": 60},
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[0, max_alt_km])
    return fig


def safe_median(ds: xr.Dataset, name: str, wavelength: Any) -> float:
    if name not in ds:
        return np.nan
    try:
        return float(np.nanmedian(select_wavelength(ds[name], wavelength).values))
    except Exception:
        return np.nan


def infer_l1_channels_for_wavelength(ds_l1: xr.Dataset | None, wavelength: Any) -> tuple[str | None, str | None]:
    if ds_l1 is None or "channel" not in ds_l1.coords:
        return None, None
    try:
        prefix = str(int(float(wavelength)))
    except Exception:
        prefix = str(wavelength)
    channels = [str(channel) for channel in ds_l1["channel"].values]
    analog = next((ch for ch in channels if ch.startswith(f"{prefix}.") and ch.upper().endswith(".AN")), None)
    photon = next((ch for ch in channels if ch.startswith(f"{prefix}.") and (ch.upper().endswith(".PC") or ch.upper().endswith(".PH"))), None)
    return analog, photon


def legacy_scale_factor(analog: np.ndarray, photon: np.ndarray, start: int = 1000, stop: int = 1500) -> tuple[float, tuple[int, int]]:
    n = min(np.asarray(analog).size, np.asarray(photon).size)
    start = max(0, min(int(start), max(n - 2, 0)))
    stop = max(start + 1, min(int(stop), n))
    a = np.asarray(analog[start:stop], dtype=np.float64)
    p = np.asarray(photon[start:stop], dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(p)
    denom = float(np.nansum(a[valid])) if valid.any() else np.nan
    numer = float(np.nansum(p[valid])) if valid.any() else np.nan
    if not np.isfinite(denom) or abs(denom) <= 1.0e-30 or not np.isfinite(numer):
        return 1.0, (start, stop)
    return float(numer / denom), (start, stop)


def qa_gluing_figure(ds_l2: xr.Dataset, ds_l1: xr.Dataset | None, wavelength: Any, smooth_bins: int) -> go.Figure | None:
    required = {"glued_range_corrected_signal", "gluing_success_flag", "gluing_split_altitude_m"}
    if not required.issubset(set(ds_l2.data_vars)):
        return None
    alt_m = altitude_m(ds_l2)
    alt_km = alt_m / 1000.0
    max_alt_km = min(20.0, float(np.nanmax(alt_km)))
    valid_alt = alt_km <= max_alt_km
    color = channel_color(wavelength)

    success = select_wavelength(ds_l2["gluing_success_flag"], wavelength)
    success_values = np.asarray(success.values, dtype=float)
    n_blocks = int(success.size)
    success_count = int(np.nansum(success_values))
    success_rate = 100.0 * success_count / max(n_blocks, 1)
    valid_success = np.isfinite(success_values) & (success_values == 1)

    fallback_count = int(np.nansum(select_wavelength(ds_l2["gluing_fallback_flag"], wavelength).values)) if "gluing_fallback_flag" in ds_l2 else 0
    split_alt_km = select_wavelength(ds_l2["gluing_split_altitude_m"], wavelength) / 1000.0
    median_split = float(np.nanmedian(split_alt_km.values)) if np.any(np.isfinite(split_alt_km.values)) else np.nan
    median_start = safe_median(ds_l2, "gluing_start_altitude_m", wavelength) / 1000.0
    median_stop = safe_median(ds_l2, "gluing_stop_altitude_m", wavelength) / 1000.0
    corr_med = safe_median(ds_l2, "gluing_correlation", wavelength)
    rmse_med = safe_median(ds_l2, "gluing_relative_rmse", wavelength)
    bias_med = safe_median(ds_l2, "gluing_relative_bias", wavelength)
    slope_med = safe_median(ds_l2, "gluing_slope", wavelength)
    intercept_med = safe_median(ds_l2, "gluing_intercept", wavelength)

    if "glued_corrected_signal_mean" in ds_l2:
        glued = np.asarray(select_wavelength(ds_l2["glued_corrected_signal_mean"], wavelength).values, dtype=np.float64) * alt_m**2
    else:
        glued_da = reduce_to_altitude_profile(select_wavelength(ds_l2["glued_range_corrected_signal"], wavelength))
        glued = np.asarray(glued_da.values, dtype=np.float64)
    glued = smooth_profile(glued, smooth_bins)

    analog_plot = np.full_like(glued, np.nan, dtype=np.float64)
    photon_plot = np.full_like(glued, np.nan, dtype=np.float64)
    analog_ch = photon_ch = None
    scaling_note = "L1 channels unavailable"
    clip_mean = np.nan
    if ds_l1 is not None:
        analog_ch, photon_ch = infer_l1_channels_for_wavelength(ds_l1, wavelength)
        if analog_ch and photon_ch:
            if "corrected_signal" in ds_l1:
                analog = reduce_to_altitude_profile(select_coord(ds_l1["corrected_signal"], channel_name(ds_l1["corrected_signal"]), analog_ch)).values
                photon = reduce_to_altitude_profile(select_coord(ds_l1["corrected_signal"], channel_name(ds_l1["corrected_signal"]), photon_ch)).values
                if bool(valid_success.any()) and np.isfinite(slope_med) and np.isfinite(intercept_med) and slope_med > 0:
                    analog_scaled = slope_med * analog + intercept_med
                    scaling_note = "AN scaled with operational coefficients"
                else:
                    factor, bins = legacy_scale_factor(analog, photon)
                    analog_scaled = analog * factor
                    scaling_note = f"AN display-scaled {bins[0]}:{bins[1]}"
                analog_plot = smooth_profile(analog_scaled * alt_m**2, smooth_bins)
                photon_plot = smooth_profile(photon * alt_m**2, smooth_bins)
            elif "range_corrected_signal" in ds_l1:
                analog = reduce_to_altitude_profile(select_coord(ds_l1["range_corrected_signal"], channel_name(ds_l1["range_corrected_signal"]), analog_ch)).values
                photon = reduce_to_altitude_profile(select_coord(ds_l1["range_corrected_signal"], channel_name(ds_l1["range_corrected_signal"]), photon_ch)).values
                factor, bins = legacy_scale_factor(analog, photon)
                analog_plot = smooth_profile(analog * factor, smooth_bins)
                photon_plot = smooth_profile(photon, smooth_bins)
                scaling_note = f"AN RCS display-scaled {bins[0]}:{bins[1]}"
            if "deadtime_clipping_fraction" in ds_l1:
                try:
                    clip_mean = float(np.nanmean(select_coord(ds_l1["deadtime_clipping_fraction"], channel_name(ds_l1["deadtime_clipping_fraction"]), photon_ch).values))
                except Exception:
                    clip_mean = np.nan

    main_values = []
    for arr in (analog_plot[valid_alt], photon_plot[valid_alt], glued[valid_alt]):
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size:
            main_values.append(arr)
    if main_values:
        all_vals = np.concatenate(main_values)
        xmin = max(float(np.nanpercentile(all_vals, 1.0)), 1e-3)
        xmax = max(float(np.nanpercentile(all_vals, 99.5)), xmin * 10.0)
    else:
        xmin, xmax = 1e-3, 1.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=analog_plot[valid_alt], y=alt_km[valid_alt], mode="lines", name=f"{analog_ch or 'AN'} scaled", line={"color": "blue", "dash": "dash", "width": 2.0}))
    fig.add_trace(go.Scatter(x=photon_plot[valid_alt], y=alt_km[valid_alt], mode="lines", name=f"{photon_ch or 'PC'} mean", line={"color": "orange", "dash": "dot", "width": 2.2}))
    fig.add_trace(go.Scatter(x=glued[valid_alt], y=alt_km[valid_alt], mode="lines", name="Glued mean", line={"color": color, "width": 2.8}))
    if np.isfinite(median_start) and np.isfinite(median_stop) and 0.0 < median_start < median_stop <= max_alt_km:
        fig.add_hrect(y0=median_start, y1=median_stop, fillcolor="gold", opacity=0.25, line_width=0, annotation_text=f"Median gluing window {median_start:.2f}-{median_stop:.2f} km", annotation_position="top left")
    if np.isfinite(median_split) and 0.0 < median_split <= max_alt_km:
        fig.add_hline(y=median_split, line_color="black", line_dash="dashdot", annotation_text=f"Median split {median_split:.2f} km", annotation_position="bottom right")
    summary = [
        f"success = {success_count}/{n_blocks} ({success_rate:.1f}%)",
        f"fallback = {fallback_count}",
        f"corr med = {corr_med:.4g}",
        f"rmse med = {rmse_med:.4g}",
        f"bias med = {bias_med:.4g}",
    ]
    if np.isfinite(clip_mean):
        summary.append(f"deadtime clipped = {100.0 * clip_mean:.2f}%")
    summary.append(scaling_note)
    fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.98, align="left", showarrow=False, text="<br>".join(summary), bgcolor="rgba(255,255,255,0.86)", bordercolor="gray")
    fig.update_layout(
        title=f"MILGRAU Level 2 QA - Signal Gluing - {wavelength_label(wavelength)}<br><sup>{date_title(ds_l2)}</sup>",
        xaxis_title="RCS on photon-counting scale [a.u.]",
        yaxis_title="Altitude (km a.g.l.)",
        height=820,
        margin={"l": 70, "r": 30, "t": 90, "b": 60},
    )
    fig.update_xaxes(type="log", range=[np.log10(xmin), np.log10(xmax)])
    fig.update_yaxes(range=[0, max_alt_km])
    return fig


def qa_molecular_figure(ds_l2: xr.Dataset, wavelength: Any, smooth_bins: int) -> go.Figure | None:
    required = {"glued_range_corrected_signal_mean", "scaled_molecular_range_corrected_signal", "rayleigh_calibration_factor", "rayleigh_reference_altitude_m"}
    if not required.issubset(set(ds_l2.data_vars)):
        return None
    alt = altitude_km(ds_l2)
    max_alt_km = min(30.0, float(np.nanmax(alt)))
    valid_alt = alt <= max_alt_km
    mean_glued = smooth_profile(select_wavelength(ds_l2["glued_range_corrected_signal_mean"], wavelength).values, smooth_bins)
    rayleigh = smooth_profile(select_wavelength(ds_l2["scaled_molecular_range_corrected_signal"], wavelength).values, smooth_bins)
    ref_alt_km = float(select_wavelength(ds_l2["rayleigh_reference_altitude_m"], wavelength).values) / 1000.0
    calibration_factor = float(select_wavelength(ds_l2["rayleigh_calibration_factor"], wavelength).values)
    calibration_intercept = safe_median(ds_l2, "rayleigh_calibration_intercept", wavelength)
    ref_min_km = safe_median(ds_l2, "rayleigh_reference_start_altitude_m", wavelength) / 1000.0
    ref_max_km = safe_median(ds_l2, "rayleigh_reference_stop_altitude_m", wavelength) / 1000.0
    if not np.isfinite(ref_min_km):
        ref_min_km = np.nan
    if not np.isfinite(ref_max_km):
        ref_max_km = np.nan
    ref_window = np.isfinite(mean_glued) & np.isfinite(rayleigh)
    if np.isfinite(ref_min_km) and np.isfinite(ref_max_km):
        ref_window = ref_window & (alt >= ref_min_km) & (alt <= ref_max_km)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.46, 0.54], horizontal_spacing=0.12, subplot_titles=("Rayleigh fit region", "Molecular calibration profile"))
    if np.any(ref_window):
        fig.add_trace(go.Scatter(x=rayleigh[ref_window], y=mean_glued[ref_window], mode="lines+markers", name="Reference-region samples", line={"color": "royalblue", "width": 1.4}, marker={"size": 4}), row=1, col=1)
        xfit = np.linspace(float(np.nanmin(rayleigh[ref_window])), float(np.nanmax(rayleigh[ref_window])), 100)
        fig.add_trace(go.Scatter(x=xfit, y=xfit + calibration_intercept, mode="lines", name="Linear fit diagnostic", line={"color": "black", "dash": "dash", "width": 1.8}), row=1, col=1)
    fig.add_trace(go.Scatter(x=mean_glued[valid_alt], y=alt[valid_alt], mode="lines", name="Mean glued RCS", line={"color": channel_color(wavelength), "width": 2.2}), row=1, col=2)
    fig.add_trace(go.Scatter(x=rayleigh[valid_alt], y=alt[valid_alt], mode="lines", name="Scaled Rayleigh molecular RCS", line={"color": "black", "dash": "dash", "width": 2.0}), row=1, col=2)
    if np.isfinite(ref_min_km) and np.isfinite(ref_max_km):
        fig.add_hrect(y0=ref_min_km, y1=ref_max_km, fillcolor="gray", opacity=0.12, line_width=0, row=1, col=2)
    if np.isfinite(ref_alt_km) and 0 < ref_alt_km <= max_alt_km:
        fig.add_hline(y=ref_alt_km, line_color="black", line_dash="dot", annotation_text=f"Reference {ref_alt_km:.2f} km", row=1, col=2)
    fig.update_xaxes(title_text="Scaled molecular RCS [a.u.]", row=1, col=1)
    fig.update_yaxes(title_text="Measured glued RCS [a.u.]", row=1, col=1)
    fig.update_xaxes(title_text="RCS [a.u.]", row=1, col=2)
    fig.update_yaxes(title_text="Altitude (km a.g.l.)", range=[0, max_alt_km], row=1, col=2)
    fig.update_layout(
        title=f"MILGRAU Level 2 QA - Molecular Rayleigh Fit - {wavelength_label(wavelength)}<br><sup>slope={calibration_factor:.3g}, intercept={calibration_intercept:.3g} · {date_title(ds_l2)}</sup>",
        height=740,
        margin={"l": 70, "r": 30, "t": 95, "b": 65},
    )
    return fig


def block_standard_error(block_values: np.ndarray, valid_block: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(block_values, dtype=np.float64)
    if arr.ndim != 2:
        return np.full(arr.shape[-1] if arr.ndim else 0, np.nan, dtype=np.float64)
    if valid_block is not None and np.asarray(valid_block).size == arr.shape[0]:
        arr = arr[np.asarray(valid_block, dtype=bool), :]
    finite = np.isfinite(arr)
    count = finite.sum(axis=0)
    std = np.nanstd(arr, axis=0, ddof=0)
    return np.divide(std, np.sqrt(np.maximum(count, 1)), out=np.full(arr.shape[1], np.nan, dtype=np.float64), where=count > 1)


def qa_scattering_ratio_figure(ds_l2: xr.Dataset, wavelength: Any, smooth_bins: int) -> go.Figure | None:
    if "scattering_ratio_mean" not in ds_l2:
        return None
    alt = altitude_km(ds_l2)
    max_alt_km = min(30.0, float(np.nanmax(alt)))
    valid_alt = alt <= max_alt_km
    sr = smooth_profile(select_wavelength(ds_l2["scattering_ratio_mean"], wavelength).values, smooth_bins)
    sr_sigma = np.full_like(sr, np.nan, dtype=np.float64)
    uncertainty_label = "Block SEM"
    if "scattering_ratio_error_mean" in ds_l2:
        sr_sigma = smooth_profile(select_wavelength(ds_l2["scattering_ratio_error_mean"], wavelength).values, smooth_bins)
        uncertainty_label = "SR 1σ"
    elif "scattering_ratio_block" in ds_l2:
        valid_block = None
        if "valid_retrieval_block_flag" in ds_l2:
            try:
                valid_block = np.asarray(select_wavelength(ds_l2["valid_retrieval_block_flag"], wavelength).values, dtype=bool)
            except Exception:
                valid_block = None
        sr_sigma = smooth_profile(block_standard_error(select_wavelength(ds_l2["scattering_ratio_block"], wavelength).values, valid_block), smooth_bins)
    color = channel_color(wavelength)
    fig = go.Figure()
    if np.isfinite(sr_sigma).any():
        add_profile_band(fig, alt[valid_alt], sr[valid_alt], sr_sigma[valid_alt], uncertainty_label, color)
    fig.add_trace(go.Scatter(x=sr[valid_alt], y=alt[valid_alt], mode="lines", name="Scattering ratio", line={"color": color, "width": 2.4}))
    fig.add_vline(x=1.0, line_color="black", line_dash="dash", annotation_text="SR=1")
    xlim = robust_positive_xlim(sr[valid_alt], default_max=6.0)
    notes = [f"plot smoothing = {smooth_bins} bins"]
    upper_mask = (alt >= 10.0) & valid_alt & np.isfinite(sr)
    if np.any(upper_mask):
        notes.insert(0, f"Mean SR above 10 km = {float(np.nanmean(sr[upper_mask])):.2f}")
    fig.add_annotation(xref="paper", yref="paper", x=0.04, y=0.96, align="left", showarrow=False, text="<br>".join(notes), bgcolor="rgba(255,255,255,0.82)", bordercolor="gray")
    fig.update_layout(
        title=f"MILGRAU Level 2 QA - Scattering Ratio - {wavelength_label(wavelength)}<br><sup>{date_title(ds_l2)}</sup>",
        xaxis_title="Scattering ratio",
        yaxis_title="Altitude (km a.g.l.)",
        height=740,
        margin={"l": 70, "r": 30, "t": 95, "b": 65},
    )
    fig.update_xaxes(range=list(xlim))
    fig.update_yaxes(range=[0, max_alt_km])
    return fig


def first_available(ds: xr.Dataset, names: list[str]) -> str | None:
    return next((name for name in names if name in ds), None)


def qa_kfs_figure(ds_l2: xr.Dataset, wavelength: Any, smooth_bins: int, max_altitude_km: float) -> go.Figure | None:
    beta_name = first_available(ds_l2, ["aerosol_backscatter", "aerosol_backscatter_mean"])
    beta_err_name = first_available(ds_l2, ["aerosol_backscatter_error", "aerosol_backscatter_error_mean"])
    alpha_name = first_available(ds_l2, ["aerosol_extinction", "aerosol_extinction_mean"])
    alpha_err_name = first_available(ds_l2, ["aerosol_extinction_error", "aerosol_extinction_error_mean"])
    if not beta_name or not beta_err_name or not alpha_name or not alpha_err_name:
        return None
    alt = altitude_km(ds_l2)
    max_alt_km = min(float(max_altitude_km), float(np.nanmax(alt)))
    valid_alt = alt <= max_alt_km
    beta = reduce_to_altitude_profile(select_wavelength(ds_l2[beta_name], wavelength), smooth_bins=smooth_bins).values * 1e6
    beta_sigma = reduce_to_altitude_profile(error_of_mean(select_wavelength(ds_l2[beta_err_name], wavelength)), smooth_bins=smooth_bins).values * 1e6
    alpha = reduce_to_altitude_profile(select_wavelength(ds_l2[alpha_name], wavelength), smooth_bins=smooth_bins).values * 1e6
    alpha_sigma = reduce_to_altitude_profile(error_of_mean(select_wavelength(ds_l2[alpha_err_name], wavelength)), smooth_bins=smooth_bins).values * 1e6
    beta_xlim = robust_centered_xlim(beta[valid_alt], default_abs=5.0)
    alpha_xlim = robust_centered_xlim(alpha[valid_alt], default_abs=50.0)
    color = channel_color(wavelength)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.12, subplot_titles=("Aerosol backscatter", "Aerosol extinction"))
    add_profile_band(fig, alt[valid_alt], beta[valid_alt], beta_sigma[valid_alt], "MC 1σ", color, row=1, col=1)
    fig.add_trace(go.Scatter(x=beta[valid_alt], y=alt[valid_alt], mode="lines", name="Mean beta aer", line={"color": color, "width": 2.4}), row=1, col=1)
    fig.add_vline(x=0.0, line_color="black", line_width=0.8, row=1, col=1)
    add_profile_band(fig, alt[valid_alt], alpha[valid_alt], alpha_sigma[valid_alt], "MC 1σ", color, row=1, col=2)
    fig.add_trace(go.Scatter(x=alpha[valid_alt], y=alt[valid_alt], mode="lines", name="Mean alpha aer", line={"color": color, "width": 2.4}), row=1, col=2)
    fig.add_vline(x=0.0, line_color="black", line_width=0.8, row=1, col=2)
    fig.update_xaxes(title_text="β aer [Mm⁻¹ sr⁻¹]", range=list(beta_xlim), row=1, col=1)
    fig.update_xaxes(title_text="α aer [Mm⁻¹]", range=list(alpha_xlim), row=1, col=2)
    fig.update_yaxes(title_text="Altitude (km a.g.l.)", range=[0, max_alt_km], row=1, col=1)
    fig.update_layout(
        title=f"MILGRAU Level 2 QA - Klett-Fernald-Sasano - {wavelength_label(wavelength)}<br><sup>{date_title(ds_l2)}</sup>",
        height=740,
        margin={"l": 70, "r": 30, "t": 95, "b": 65},
    )
    return fig


def render_level1(row: dict[str, Any]) -> None:
    path = row.get("level1_path", "")
    if not path or not Path(path).exists():
        st.info("Level 1 não disponível para esta medida.")
        return
    st.caption(f"Arquivo: `{path}`")
    ds = open_dataset(path)
    channels = coord_values(ds, channel_name(ds))
    if not channels:
        st.warning("Não encontrei coordenada de canal no Level 1.")
        return
    prefix = safe_key(row.get("save_id"), "level1")
    mode = st.radio("Plot", ["Quicklook RCS + perfil", "Global mean RCS", "Genérico"], horizontal=True, key=f"{prefix}_plot_mode")
    max_alt_km = st.number_input("Altitude máx. (km)", min_value=0.1, value=15.0, step=0.5, key=f"{prefix}_alt")
    smooth_bins = st.slider("Suavização vertical para perfis (bins)", 1, 80, 20, key=f"{prefix}_smooth")
    if mode == "Quicklook RCS + perfil":
        variables = numeric_variables(ds)
        default_var = "range_corrected_signal" if "range_corrected_signal" in ds else variables[0]
        c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1.4])
        channel = c1.selectbox("Canal", channels, key=f"{prefix}_channel")
        variable = c2.selectbox("Variável", variables, index=variables.index(default_var), key=f"{prefix}_var")
        log10 = c3.checkbox("log10", value=False, key=f"{prefix}_log")
        color_range = c4.text_input("Range de cor `min,max`", value="", key=f"{prefix}_range")
        st.plotly_chart(level1_quicklook_figure(ds, channel, variable, max_alt_km, log10, smooth_bins, color_range), use_container_width=True)
    elif mode == "Global mean RCS":
        default_channels = channels[: min(6, len(channels))]
        chosen = st.multiselect("Canais", channels, default=default_channels, key=f"{prefix}_global_channels")
        st.plotly_chart(global_mean_rcs_figure(ds, chosen, max_alt_km, smooth_bins), use_container_width=True)
    else:
        render_generic_level(row, "Level 1", ds=ds)
    with st.expander("Metadados deste nível"):
        render_metadata(ds, key_prefix=f"{prefix}_embedded_metadata")
    with st.expander("Diagnósticos deste nível"):
        render_diagnostics(ds)


def render_level2(row: dict[str, Any]) -> None:
    path = row.get("level2_path", "")
    if not path or not Path(path).exists():
        st.info("Level 2 não disponível para esta medida.")
        return
    st.caption(f"Arquivo: `{path}`")
    ds_l2 = open_dataset(path)
    ds_l1 = open_dataset(row["level1_path"]) if row.get("level1_path") else None
    prefix = safe_key(row.get("save_id"), "level2")
    wavelengths = wavelength_values(ds_l2)
    wavelength = st.selectbox("Comprimento de onda", wavelengths, key=f"{prefix}_wave") if wavelengths else None
    smooth_bins = st.slider("Suavização vertical dos perfis (bins)", 1, 80, 15, key=f"{prefix}_smooth")
    available_modes = ["Genérico"]
    if wavelength is not None:
        if {"glued_range_corrected_signal", "gluing_success_flag", "gluing_split_altitude_m"}.issubset(set(ds_l2.data_vars)):
            available_modes.insert(0, "QA gluing")
        if {"glued_range_corrected_signal_mean", "scaled_molecular_range_corrected_signal", "rayleigh_calibration_factor", "rayleigh_reference_altitude_m"}.issubset(set(ds_l2.data_vars)):
            available_modes.append("QA Rayleigh molecular")
        if "scattering_ratio_mean" in ds_l2:
            available_modes.append("QA scattering ratio")
        if first_available(ds_l2, ["aerosol_backscatter", "aerosol_backscatter_mean"]) and first_available(ds_l2, ["aerosol_extinction", "aerosol_extinction_mean"]):
            available_modes.append("QA KFS retrieval")
    mode = st.radio("Plot", available_modes, horizontal=True, key=f"{prefix}_plot_mode")
    if mode == "QA gluing" and wavelength is not None:
        fig = qa_gluing_figure(ds_l2, ds_l1, wavelength, smooth_bins)
    elif mode == "QA Rayleigh molecular" and wavelength is not None:
        fig = qa_molecular_figure(ds_l2, wavelength, smooth_bins)
    elif mode == "QA scattering ratio" and wavelength is not None:
        fig = qa_scattering_ratio_figure(ds_l2, wavelength, smooth_bins)
    elif mode == "QA KFS retrieval" and wavelength is not None:
        max_alt_km = st.number_input("Altitude máx. (km)", min_value=0.1, value=30.0, step=0.5, key=f"{prefix}_kfs_alt")
        fig = qa_kfs_figure(ds_l2, wavelength, smooth_bins, max_alt_km)
    else:
        fig = None
        render_generic_level(row, "Level 2", ds=ds_l2, default_wavelength=wavelength)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    elif mode != "Genérico":
        st.warning("Esse painel QA não encontrou as variáveis necessárias neste arquivo.")
    with st.expander("Metadados deste nível"):
        render_metadata(ds_l2, key_prefix=f"{prefix}_embedded_metadata")
    with st.expander("Diagnósticos deste nível"):
        render_diagnostics(ds_l2)


def render_generic_level(row: dict[str, Any], level: str, ds: xr.Dataset | None = None, default_wavelength: Any | None = None) -> None:
    path = row.get(LEVEL_TO_PATH[level], "")
    if ds is None:
        if not path or not Path(path).exists():
            st.info(f"{level} não disponível para esta medida.")
            return
        st.caption(f"Arquivo: `{path}`")
        ds = open_dataset(path)
    variables = numeric_variables(ds)
    if not variables:
        st.info("Não encontrei variáveis numéricas plottáveis neste nível.")
        return
    prefix = safe_key(row.get("save_id"), level, "generic")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    variable = c1.selectbox("Variável", variables, key=f"{prefix}_var")
    channels = coord_values(ds, channel_name(ds))
    channel = c2.selectbox("Canal", channels, key=f"{prefix}_channel") if channels else None
    wavelengths = wavelength_values(ds)
    wave_index = wavelengths.index(default_wavelength) if default_wavelength in wavelengths else 0
    wavelength = c3.selectbox("Comprimento de onda", wavelengths, index=wave_index, key=f"{prefix}_wave") if wavelengths else None
    max_alt_km = c4.number_input("Altitude máx. (km)", min_value=0.1, value=15.0 if level != "Level 2" else 30.0, step=0.5, key=f"{prefix}_alt")
    c5, c6, c7, c8 = st.columns([1, 1, 1, 2])
    log10 = c5.checkbox("log10", value="signal" in variable.lower(), key=f"{prefix}_log")
    mode = c6.radio("Modo", ["quicklook", "perfil médio"], horizontal=True, key=f"{prefix}_mode")
    smooth_bins = c7.number_input("smooth bins", min_value=1, value=20, step=1, key=f"{prefix}_smooth")
    color_range = c8.text_input("Range de cor `min,max`", value="", key=f"{prefix}_range")
    view = reduce_for_heatmap(ds[variable], channel, wavelength, max_alt_km)
    title = f"{level} · {variable}" + (f" · {channel}" if channel is not None else "") + (f" · {wavelength_label(wavelength)}" if wavelength is not None else "")
    fig = profile_figure(view, title, int(smooth_bins)) if mode == "perfil médio" else heatmap_figure(view, title, log10, color_range)
    st.plotly_chart(fig, use_container_width=True)


def render_level(row: dict[str, Any], level: str) -> None:
    if level == "Level 1":
        render_level1(row)
        return
    if level == "Level 2":
        render_level2(row)
        return
    path = row.get(LEVEL_TO_PATH[level], "")
    if not path or not Path(path).exists():
        st.info(f"{level} não disponível para esta medida.")
        return
    ds = open_dataset(path)
    st.caption(f"Arquivo: `{path}`")
    render_generic_level(row, level, ds=ds)
    with st.expander("Metadados deste nível"):
        render_metadata(ds, key_prefix=f"{safe_key(row.get('save_id'), level)}_embedded_metadata")
    with st.expander("Diagnósticos deste nível"):
        render_diagnostics(ds)


def render_metadata(ds: xr.Dataset, key_prefix: str) -> None:
    tabs = st.tabs(["attrs globais", "dimensões", "variáveis", "attrs variável", "estatísticas leves"])
    prefix = safe_key(key_prefix)
    with tabs[0]:
        st.dataframe(pd.DataFrame([{"attribute": k, "value": str(v)} for k, v in ds.attrs.items()]), use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(pd.DataFrame([{"dimension": k, "size": int(v)} for k, v in ds.sizes.items()]), use_container_width=True, hide_index=True)
    with tabs[2]:
        rows = []
        for name, da in ds.variables.items():
            rows.append({"name": name, "kind": "coord" if name in ds.coords else "data_var", "dims": " × ".join(da.dims), "shape": " × ".join(str(s) for s in da.shape), "dtype": str(da.dtype), "units": da.attrs.get("units", ""), "long_name": da.attrs.get("long_name", da.attrs.get("standard_name", ""))})
        table = pd.DataFrame(rows)
        query = st.text_input("Filtrar variáveis/attrs", key=f"{prefix}_filter")
        if query:
            table = table[table.apply(lambda col: col.astype(str).str.contains(query, case=False, na=False)).any(axis=1)]
        st.dataframe(table, use_container_width=True, hide_index=True)
    with tabs[3]:
        var = st.selectbox("Variável", list(ds.variables), key=f"{prefix}_attrs_var")
        st.dataframe(pd.DataFrame([{"attribute": k, "value": str(v)} for k, v in ds[var].attrs.items()]), use_container_width=True, hide_index=True)
    with tabs[4]:
        names = numeric_variables(ds)
        chosen = st.multiselect("Variáveis", names, default=names[: min(8, len(names))], key=f"{prefix}_stats_vars")
        rows = []
        for name in chosen:
            da = ds[name]
            total = int(np.prod(da.shape)) if da.shape else 1
            step = max(1, int(np.ceil(total / 300_000)))
            subset = da.isel({dim: slice(None, None, step) for dim in da.dims}) if total > 300_000 else da
            values = np.asarray(subset.values, dtype=float).ravel()
            finite = np.isfinite(values)
            valid = values[finite]
            rows.append({"variable": name, "sampled_values": int(values.size), "finite_fraction": float(finite.mean()) if values.size else np.nan, "min": float(np.nanmin(valid)) if valid.size else np.nan, "median": float(np.nanmedian(valid)) if valid.size else np.nan, "max": float(np.nanmax(valid)) if valid.size else np.nan})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_diagnostics(ds: xr.Dataset) -> None:
    rows = []
    for name in DIAGNOSTIC_NAMES:
        if name in ds.variables and ds[name].size <= 200:
            value = ds[name].values
            value = value.item() if np.asarray(value).size == 1 else np.asarray(value).tolist()
            rows.append({"diagnostic": name, "value": str(value)})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum diagnóstico escalar/small-array conhecido encontrado.")


def filter_inventory(inv: pd.DataFrame) -> pd.DataFrame:
    filtered = inv.copy()
    year_options = sorted(int(year) for year in filtered["year"].dropna().unique())
    year = st.selectbox("Ano", year_options, index=len(year_options) - 1, key="inventory_year") if year_options else None
    if year is not None:
        filtered = filtered[filtered["year"] == year]
    month_options = sorted(int(month) for month in filtered["month"].dropna().unique())
    month = st.selectbox("Mês", month_options, index=len(month_options) - 1, format_func=lambda value: f"{value:02d}", key="inventory_month") if month_options else None
    if month is not None:
        filtered = filtered[filtered["month"] == month]
    day_options = ["todos"] + sorted(int(day) for day in filtered["day"].dropna().unique())
    day = st.selectbox("Dia", day_options, format_func=lambda value: value if value == "todos" else f"{value:02d}", key="inventory_day")
    if day != "todos":
        filtered = filtered[filtered["day"] == day]
    periods = sorted(str(period) for period in filtered["period"].dropna().unique())
    selected_periods = st.multiselect("Período", periods, default=periods, key="inventory_periods")
    if selected_periods:
        filtered = filtered[filtered["period"].astype(str).isin(selected_periods)]
    level_options = list(LEVEL_TO_PATH)
    selected_levels = st.multiselect("Níveis disponíveis", level_options, default=level_options, key="inventory_levels")
    if selected_levels:
        mask = pd.Series(False, index=filtered.index)
        for level in selected_levels:
            mask = mask | filtered[LEVEL_TO_PATH[level]].astype(bool)
        filtered = filtered[mask]
    search = st.text_input("Buscar save_id", value="", key="inventory_search")
    if search:
        filtered = filtered[filtered["save_id"].astype(str).str.contains(search, case=False, na=False)]
    return filtered


def measurement_label(inv: pd.DataFrame, index: int) -> str:
    row = inv.loc[index]
    day = row["date"].isoformat() if pd.notna(row["date"]) else "sem data"
    return f"{day} · {row['save_id']} · {row['available_levels']}"


def select_measurement(inv: pd.DataFrame) -> dict[str, Any]:
    st.subheader("Medidas disponíveis")
    filtered = filter_inventory(inv)
    st.caption(f"{len(filtered)} de {len(inv)} medidas após filtros")
    if filtered.empty:
        st.warning("Nenhuma medida bate com os filtros.")
        st.stop()
    filtered = filtered.sort_values(["date", "save_id"], na_position="last")
    selected_index = st.selectbox(
        "Abrir medida",
        list(filtered.index),
        index=len(filtered.index) - 1,
        format_func=lambda index: measurement_label(filtered, index),
        key="selected_measurement",
    )
    preview_cols = ["date", "save_id", "period", "available_levels", "modified"]
    st.dataframe(filtered[preview_cols], use_container_width=True, hide_index=True, height=220)
    return filtered.loc[selected_index].to_dict()


def main() -> None:
    st.set_page_config(page_title="MILGRAU NetCDF Explorer", layout="wide")
    st.title("MILGRAU NetCDF Explorer")
    st.caption("Explorador read-only para Level 0 LIBIDS, Level 1 LIPANCORA e Level 2 LEBEAR.")
    with st.sidebar:
        st.header("Dados")
        config_path = st.text_input("config.yaml", value="config.yaml", key="config_path")
        try:
            default_root = processed_root_from_config(config_path)
        except Exception as exc:
            st.error(f"Não consegui carregar o config: {exc}")
            st.stop()
        processed_root = Path(st.text_input("Pasta processed_data", value=str(default_root), key="processed_root")).expanduser()
        if st.button("Atualizar inventário"):
            discover_products.clear()
            open_dataset.clear()
        inv = discover_products(str(processed_root))
        st.metric("medidas encontradas", len(inv))
        if inv.empty:
            st.warning("Nenhum produto .nc encontrado na pasta de dados.")
            st.stop()
        row = select_measurement(inv)
    tabs = st.tabs(["Resumo", "Level 0", "Level 1", "Level 2", "Metadados", "QA"])
    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("save_id", row["save_id"])
        c2.metric("período", row["period"])
        c3.metric("níveis", row["available_levels"])
        c4.metric("modificado", row["modified"])
        st.caption(f"Pasta de dados: `{processed_root}`")
        st.dataframe(pd.DataFrame([{"level": level, "path": row[key]} for level, key in LEVEL_TO_PATH.items()]), use_container_width=True, hide_index=True)
        with st.expander("Inventário completo"):
            st.dataframe(inv, use_container_width=True, hide_index=True)
    for tab, level in zip(tabs[1:4], ["Level 0", "Level 1", "Level 2"]):
        with tab:
            render_level(row, level)
    with tabs[4]:
        available = [level for level, key in LEVEL_TO_PATH.items() if row.get(key)]
        selected_level = st.selectbox("Nível", available, key=safe_key(row.get("save_id"), "metadata_level"))
        render_metadata(open_dataset(row[LEVEL_TO_PATH[selected_level]]), key_prefix=f"{row.get('save_id')}_{selected_level}_main_metadata")
    with tabs[5]:
        for level, key in LEVEL_TO_PATH.items():
            if row.get(key):
                st.markdown(f"### {level}")
                render_diagnostics(open_dataset(row[key]))


if __name__ == "__main__":
    main()
