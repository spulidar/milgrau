"""Interactive L2 QA diagnostics; visualization only, not retrieval."""
from __future__ import annotations
from typing import Any
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots
from milgrau.viz.style import channel_color
from milgrau.explorer.data import (
    channel_name,
    wavelength_label,
    date_title,
    select_coord,
    select_wavelength,
    altitude_km,
    altitude_m,
    smooth_profile,
    reduce_to_altitude_profile,
    error_of_mean,
    robust_positive_xlim,
    robust_centered_xlim,
)
from milgrau.explorer.plots_signals import (
    add_profile_band,
)

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

    fallback_count = int(np.nansum(select_wavelength(ds_l2["single_channel_fallback_flag"], wavelength).values)) if "single_channel_fallback_flag" in ds_l2 else 0
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
    fig.add_trace(go.Scatter(x=glued[valid_alt], y=alt_km[valid_alt], mode="lines", name="Selected signal mean", line={"color": color, "width": 2.8}))
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
    mean_glued_raw = select_wavelength(ds_l2["glued_range_corrected_signal_mean"], wavelength).values
    rayleigh = smooth_profile(select_wavelength(ds_l2["scaled_molecular_range_corrected_signal"], wavelength).values, smooth_bins)
    ref_alt_km = float(select_wavelength(ds_l2["rayleigh_reference_altitude_m"], wavelength).values) / 1000.0
    calibration_factor = float(select_wavelength(ds_l2["rayleigh_calibration_factor"], wavelength).values)
    background_offset = safe_median(ds_l2, "rayleigh_background_offset", wavelength)
    if not np.isfinite(background_offset):
        background_offset = safe_median(
            ds_l2, "rayleigh_background_offset_block", wavelength
        )
    if not np.isfinite(background_offset):
        background_offset = 0.0
    if "rayleigh_background_corrected_range_corrected_signal" in ds_l2:
        corrected_glued_raw = select_wavelength(ds_l2["rayleigh_background_corrected_range_corrected_signal"], wavelength).values
    else:
        corrected_glued_raw = mean_glued_raw - background_offset * np.asarray(ds_l2["altitude"].values, dtype=np.float64) ** 2
    mean_glued = smooth_profile(corrected_glued_raw, smooth_bins)
    ref_min_km = safe_median(ds_l2, "rayleigh_reference_start_altitude_m", wavelength) / 1000.0
    ref_max_km = safe_median(ds_l2, "rayleigh_reference_stop_altitude_m", wavelength) / 1000.0
    ref_window = np.isfinite(mean_glued) & np.isfinite(rayleigh)
    if np.isfinite(ref_min_km) and np.isfinite(ref_max_km):
        ref_window = ref_window & (alt >= ref_min_km) & (alt <= ref_max_km)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.46, 0.54], horizontal_spacing=0.12, subplot_titles=("Rayleigh fit region", "Molecular calibration profile"))
    if np.any(ref_window):
        fig.add_trace(go.Scatter(x=rayleigh[ref_window], y=mean_glued[ref_window], mode="lines+markers", name="Background-corrected samples", line={"color": "royalblue", "width": 1.4}, marker={"size": 4}), row=1, col=1)
        xfit = np.linspace(float(np.nanmin(rayleigh[ref_window])), float(np.nanmax(rayleigh[ref_window])), 100)
        fig.add_trace(go.Scatter(x=xfit, y=xfit, mode="lines", name="Joint-fit model", line={"color": "black", "dash": "dash", "width": 1.8}), row=1, col=1)
    fig.add_trace(go.Scatter(x=mean_glued[valid_alt], y=alt[valid_alt], mode="lines", name="Background-corrected mean RCS", line={"color": channel_color(wavelength), "width": 2.2}), row=1, col=2)
    fig.add_trace(go.Scatter(x=rayleigh[valid_alt], y=alt[valid_alt], mode="lines", name="Scaled Rayleigh molecular RCS", line={"color": "black", "dash": "dash", "width": 2.0}), row=1, col=2)
    if np.isfinite(ref_min_km) and np.isfinite(ref_max_km):
        fig.add_hrect(y0=ref_min_km, y1=ref_max_km, fillcolor="gray", opacity=0.12, line_width=0, row=1, col=2)
    if np.isfinite(ref_alt_km) and 0 < ref_alt_km <= max_alt_km:
        fig.add_hline(y=ref_alt_km, line_color="black", line_dash="dot", annotation_text=f"Reference {ref_alt_km:.2f} km", row=1, col=2)
    fig.update_xaxes(title_text="Scaled molecular RCS [a.u.]", row=1, col=1)
    fig.update_yaxes(title_text="Background-corrected glued RCS [a.u.]", row=1, col=1)
    fig.update_xaxes(title_text="RCS [a.u.]", row=1, col=2)
    fig.update_yaxes(title_text="Altitude (km a.g.l.)", range=[0, max_alt_km], row=1, col=2)
    fig.update_layout(
        title=f"MILGRAU Level 2 QA - Molecular Rayleigh Fit - {wavelength_label(wavelength)}<br><sup>A={calibration_factor:.3g}, B={background_offset:.3g} · {date_title(ds_l2)}</sup>",
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
        if "retrieval_success_flag" in ds_l2:
            try:
                valid_block = np.asarray(select_wavelength(ds_l2["retrieval_success_flag"], wavelength).values, dtype=bool)
            except Exception:
                valid_block = None
        sr_sigma = smooth_profile(block_standard_error(select_wavelength(ds_l2["scattering_ratio_block"], wavelength).values, valid_block), smooth_bins)
    fig = go.Figure()
    if np.isfinite(sr_sigma).any():
        add_profile_band(fig, alt[valid_alt], sr[valid_alt], sr_sigma[valid_alt], uncertainty_label)
    fig.add_trace(go.Scatter(x=sr[valid_alt], y=alt[valid_alt], mode="lines", name="Scattering ratio", line={"color": channel_color(wavelength), "width": 2.4}))
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
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.12, subplot_titles=("Aerosol backscatter", "Aerosol extinction"))
    add_profile_band(fig, alt[valid_alt], beta[valid_alt], beta_sigma[valid_alt], "MC 1σ", row=1, col=1)
    fig.add_trace(go.Scatter(x=beta[valid_alt], y=alt[valid_alt], mode="lines", name="Mean beta aer", line={"color": channel_color(wavelength), "width": 2.4}), row=1, col=1)
    fig.add_vline(x=0.0, line_color="black", line_width=0.8, row=1, col=1)
    add_profile_band(fig, alt[valid_alt], alpha[valid_alt], alpha_sigma[valid_alt], "MC 1σ", row=1, col=2)
    fig.add_trace(go.Scatter(x=alpha[valid_alt], y=alt[valid_alt], mode="lines", name="Mean alpha aer", line={"color": channel_color(wavelength), "width": 2.4}), row=1, col=2)
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
