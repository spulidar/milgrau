"""Interactive L0/L1 figures, using scientific values already in NetCDF."""
from __future__ import annotations
from typing import Any
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots
from milgrau.viz.style import channel_color
from milgrau.explorer.data import (
    channel_name,
    vertical_name,
    format_channel_name,
    date_title,
    select_coord,
    vertical_values,
    altitude_km,
    reduce_to_vertical_profile,
    reduce_to_altitude_profile,
    error_of_mean,
    robust_color_limits,
    x_axis_values,
    reduce_for_heatmap,
)

def heatmap_figure(
    da: xr.DataArray,
    title: str,
    level: str,
    log10: bool,
    color_range: str,
    robust: bool,
    robust_low: float,
    robust_high: float,
    ds: xr.Dataset | None = None,
) -> go.Figure:
    values = np.asarray(da.values, dtype=float)
    if log10:
        values = np.where(values > 0, np.log10(values), np.nan)
    z = values.T if values.ndim == 2 else np.atleast_2d(values)
    x, x_label, hover_time = x_axis_values(da, level, ds)
    y = vertical_values(da, level)
    y_label = "Range bin / point index" if level == "Level 0" else "Altitude (km a.g.l.)"
    zmin, zmax = robust_color_limits(z, color_range, robust, robust_low, robust_high)
    customdata = None
    hovertemplate = "%{x}<br>%{y}<br>value=%{z:.4g}<extra></extra>"
    if level == "Level 0" and hover_time is not None:
        customdata = np.tile(np.asarray(hover_time, dtype=object), (z.shape[0], 1))
        hovertemplate = "profile=%{x}<br>bin=%{y}<br>start=%{customdata}<br>value=%{z:.4g}<extra></extra>"
    fig = go.Figure(
        data=go.Heatmap(
            x=x,
            y=y,
            z=z,
            zmin=zmin,
            zmax=zmax,
            customdata=customdata,
            hovertemplate=hovertemplate,
            colorscale="Jet",
            colorbar={"title": "log10" if log10 else da.name},
        )
    )
    subtitle = "robust color scale" if robust and not color_range.strip() else "manual/auto color scale"
    fig.update_layout(
        title=f"{title}<br><sup>{subtitle}</sup>",
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin={"l": 70, "r": 20, "t": 85, "b": 55},
        height=720,
    )
    return fig


def profile_figure(da: xr.DataArray, title: str, level: str, smooth_bins: int = 20) -> go.Figure:
    prof = reduce_to_vertical_profile(da, level, smooth_bins=smooth_bins)
    y = vertical_values(prof, level)
    y_label = "Range bin / point index" if level == "Level 0" else "Altitude (km a.g.l.)"
    fig = go.Figure(data=go.Scatter(x=np.asarray(prof.values, dtype=float), y=y, mode="lines", name=prof.name or "mean profile"))
    fig.update_layout(title=title, xaxis_title=prof.name or "value", yaxis_title=y_label, margin={"l": 70, "r": 20, "t": 70, "b": 50}, height=720)
    if y.size:
        fig.update_yaxes(range=[float(np.nanmin(y)), float(np.nanmax(y))])
    return fig


def add_profile_band(fig: go.Figure, y: np.ndarray, mean: np.ndarray, sigma: np.ndarray, name: str, row: int | None = None, col: int | None = None) -> None:
    lower = np.asarray(mean, dtype=float) - np.asarray(sigma, dtype=float)
    upper = np.asarray(mean, dtype=float) + np.asarray(sigma, dtype=float)
    kwargs = {"row": row, "col": col} if row is not None and col is not None else {}
    fig.add_trace(go.Scatter(x=lower, y=y, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"), **kwargs)
    fig.add_trace(
        go.Scatter(x=upper, y=y, mode="lines", fill="tonextx", fillcolor="rgba(128,128,128,0.22)", line={"width": 0}, name=name, hoverinfo="skip"),
        **kwargs,
    )


def add_altitude_reference_lines(fig: go.Figure, ds: xr.Dataset, max_alt_km: float, row: int, col: int, xspan: tuple[float, float] | None = None) -> None:
    x0, x1 = xspan or (0.0, 1.0)
    if "PBL_Height_km" in ds:
        try:
            pbl = float(ds["PBL_Height_km"].mean(skipna=True).values)
            if np.isfinite(pbl) and 0 < pbl <= max_alt_km:
                fig.add_trace(go.Scatter(x=[x0, x1], y=[pbl, pbl], mode="lines", line={"color": "crimson", "dash": "dash"}, name=f"Mean PBL ({pbl:.1f} km)"), row=row, col=col)
        except Exception:
            pass
    for attr_name, short, color, dash in (
        ("tropopause_cpt_km", "CPT", "royalblue", "dot"),
        ("tropopause_lrt_km", "LRT", "forestgreen", "dashdot"),
    ):
        try:
            val = float(ds.attrs.get(attr_name, -999.0))
            if np.isfinite(val) and 0 < val <= max_alt_km:
                fig.add_trace(go.Scatter(x=[x0, x1], y=[val, val], mode="lines", line={"color": color, "dash": dash}, name=f"{short} ({val:.1f} km)"), row=row, col=col)
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
    robust: bool,
    robust_low: float,
    robust_high: float,
) -> go.Figure:
    data = reduce_for_heatmap(ds[variable], "Level 1", channel, None, max_alt_km)
    error_name = f"{variable}_error"
    if error_name not in ds and variable == "range_corrected_signal":
        error_name = "range_corrected_signal_error"
    data_profile = select_coord(ds[variable], channel_name(ds[variable]), channel)
    vdim = vertical_name(data_profile, "Level 1")
    if vdim and vdim in data_profile.dims:
        data_profile = data_profile.isel({vdim: vertical_values(data_profile, "Level 1") <= max_alt_km})
    profile = reduce_to_altitude_profile(data_profile, smooth_bins=smooth_bins)
    alt_prof = altitude_km(profile)
    mean_values = np.asarray(profile.values, dtype=float)
    color = channel_color(str(channel))
    pretty_channel = format_channel_name(channel)
    z = np.asarray(data.values, dtype=float)
    if log10:
        z = np.where(z > 0, np.log10(z), np.nan)
    z = z.T if z.ndim == 2 else np.atleast_2d(z)
    x, x_label, _ = x_axis_values(data, "Level 1", ds)
    y = vertical_values(data, "Level 1")
    zmin, zmax = robust_color_limits(z, color_range, robust, robust_low, robust_high)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.78, 0.22], shared_yaxes=True, horizontal_spacing=0.03, subplot_titles=("", "Mean profile"))
    fig.add_trace(go.Heatmap(x=x, y=y, z=z, zmin=zmin, zmax=zmax, colorscale="Jet", colorbar={"title": "log10 RCS" if log10 else "Intensity [a.u.]"}), row=1, col=1)
    if error_name in ds:
        err = select_coord(ds[error_name], channel_name(ds[error_name]), channel)
        vdim_err = vertical_name(err, "Level 1")
        if vdim_err and vdim_err in err.dims:
            err = err.isel({vdim_err: vertical_values(err, "Level 1") <= max_alt_km})
        err_profile = reduce_to_altitude_profile(error_of_mean(err), smooth_bins=smooth_bins)
        add_profile_band(fig, alt_prof, mean_values, np.asarray(err_profile.values, dtype=float), "1σ error", row=1, col=2)
    fig.add_trace(go.Scatter(x=mean_values, y=alt_prof, mode="lines", line={"color": color, "width": 2.4}, name="Mean RCS"), row=1, col=2)
    finite = mean_values[np.isfinite(mean_values)]
    xspan = None
    if finite.size:
        xmin = min(0.0, float(np.nanmin(finite)))
        xmax = float(np.nanmax(finite))
        pad = max((xmax - xmin) * 0.15, 1e-12)
        xspan = (xmin - pad, xmax + pad)
        fig.update_xaxes(range=[xspan[0], xspan[1]], row=1, col=2)
    add_altitude_reference_lines(fig, ds, max_alt_km, row=1, col=2, xspan=xspan)
    lower_alt = 0.16 if "AN" in pretty_channel else 0.5
    fig.update_yaxes(title_text="Altitude (km a.g.l.)", range=[lower_alt, max_alt_km], row=1, col=1)
    fig.update_xaxes(title_text=x_label, row=1, col=1)
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
        vdim = vertical_name(da, "Level 1")
        if vdim and vdim in da.dims:
            da = da.isel({vdim: vertical_values(da, "Level 1") <= max_alt_km})
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
