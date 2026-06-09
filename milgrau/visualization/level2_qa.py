"""Level 2 QA plotting helpers for LEBEAR products."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import savgol_filter

from milgrau.visualization.quicklooks import (
    extract_datetime_strings,
    safe_error_of_mean,
    safe_time_mean,
)
from milgrau.visualization.style import add_footer_and_logos, channel_color, get_output_settings


def altitude_to_km(altitude_values: np.ndarray | xr.DataArray | list[float]) -> np.ndarray:
    """Return altitude in kilometers, accepting coordinates stored in meters or km."""
    alt = np.asarray(altitude_values, dtype=float)
    if alt.size == 0:
        return alt
    if np.nanmax(alt) > 100.0:
        return alt / 1000.0
    return alt


def format_wavelength_label(wavelength_nm: int | float | str) -> str:
    """Return a compact wavelength label such as '532 nm'."""
    return f"{int(float(wavelength_nm))} nm"


def get_wavelength_values(ds_l2: xr.Dataset) -> list[int]:
    """Return wavelength coordinate values from a Level 2 dataset."""
    if "wavelength" not in ds_l2.coords:
        return []
    values: list[int] = []
    for wavelength in ds_l2["wavelength"].values:
        try:
            values.append(int(wavelength))
        except Exception:
            continue
    return values


def _smooth_for_plot(values: np.ndarray | xr.DataArray, bins: int) -> np.ndarray:
    """Return a smoothed copy for visualization without changing saved products."""
    arr = np.asarray(values, dtype=np.float64)
    if bins <= 2 or arr.size < 5:
        return arr.copy()
    window = int(bins)
    if window % 2 == 0:
        window += 1
    window = min(window, arr.size if arr.size % 2 == 1 else arr.size - 1)
    if window < 5:
        return arr.copy()
    finite = np.isfinite(arr)
    if finite.sum() < window:
        return arr.copy()
    fill_x = np.arange(arr.size)
    filled = arr.copy()
    filled[~finite] = np.interp(fill_x[~finite], fill_x[finite], arr[finite])
    smoothed = savgol_filter(filled, window_length=window, polyorder=min(3, window - 2), mode="interp")
    smoothed[~finite] = np.nan
    return smoothed


def _robust_positive_xlim(values: np.ndarray, default_max: float = 6.0) -> tuple[float, float]:
    """Return a positive x-limit that ignores extreme noisy outliers in plots."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, default_max
    high = float(np.nanpercentile(finite, 99.0))
    high = max(1.5, min(max(default_max, high * 1.15), 20.0))
    return 0.0, high


def _robust_symmetric_xlim(values: np.ndarray, default_abs: float = 1.0) -> tuple[float, float]:
    """Return symmetric robust limits for signed diagnostic series."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -default_abs, default_abs
    lim = float(np.nanpercentile(np.abs(finite), 98.0))
    lim = max(default_abs, lim * 1.2)
    return -lim, lim


def _robust_centered_xlim(values: np.ndarray, default_abs: float, percentile: float = 98.0) -> tuple[float, float]:
    """Return signed x-limits from the central product, not from huge uncertainty tails."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -default_abs, default_abs
    lim = float(np.nanpercentile(np.abs(finite), percentile))
    lim = max(float(default_abs), lim * 1.35)
    return -lim, lim


def _uncertainty_exceeds_xlim(mean: np.ndarray, sigma: np.ndarray, xlim: tuple[float, float]) -> bool:
    """Return whether a 1-sigma band extends beyond plotted x-limits."""
    lower = np.asarray(mean, dtype=np.float64) - np.asarray(sigma, dtype=np.float64)
    upper = np.asarray(mean, dtype=np.float64) + np.asarray(sigma, dtype=np.float64)
    finite = np.isfinite(lower) & np.isfinite(upper)
    if not finite.any():
        return False
    return bool(np.nanmin(lower[finite]) < xlim[0] or np.nanmax(upper[finite]) > xlim[1])


def _safe_median_dataarray(ds: xr.Dataset, name: str, wavelength: int) -> float:
    """Return wavelength-selected median for an optional Level 2 variable."""
    if name not in ds:
        return np.nan
    try:
        return float(np.nanmedian(ds[name].sel(wavelength=wavelength).values))
    except Exception:
        return np.nan


def _block_standard_error(block_values: np.ndarray, valid_block: np.ndarray | None = None) -> np.ndarray:
    """Return standard error across block profiles for a block x altitude matrix."""
    arr = np.asarray(block_values, dtype=np.float64)
    if arr.ndim != 2:
        return np.full(arr.shape[-1] if arr.ndim else 0, np.nan, dtype=np.float64)
    if valid_block is not None and np.asarray(valid_block).size == arr.shape[0]:
        arr = arr[np.asarray(valid_block, dtype=bool), :]
    finite = np.isfinite(arr)
    count = finite.sum(axis=0)
    std = np.nanstd(arr, axis=0, ddof=0)
    return np.divide(std, np.sqrt(np.maximum(count, 1)), out=np.full(arr.shape[1], np.nan, dtype=np.float64), where=count > 1)


def _legacy_scale_factor(analog: np.ndarray, photon: np.ndarray, start: int = 1000, stop: int = 1500) -> tuple[float, tuple[int, int]]:
    """Return legacy display scale factor sum(PC)/sum(AN) over a bounded bin interval."""
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


def _legacy_ylim(*profiles: np.ndarray, fallback: tuple[float, float] = (-1e8, 4e8)) -> tuple[float, float]:
    """Return legacy-like RCS y-limits without letting very high-altitude noise dominate."""
    vals = np.concatenate([np.asarray(profile, dtype=np.float64).ravel() for profile in profiles])
    vals = vals[np.isfinite(vals)]
    if vals.size < 5:
        return fallback
    low = float(np.nanpercentile(vals, 1.0))
    high = float(np.nanpercentile(vals, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return fallback
    pad = 0.08 * (high - low)
    return low - pad, high + pad


def _visual_scale_to_reference(
    lower_signal: np.ndarray,
    upper_signal: np.ndarray,
    altitude_km: np.ndarray,
    min_alt_km: float = 1.5,
    max_alt_km: float = 12.0,
) -> tuple[float, float, str]:
    """Scale one signal to another for diagnostic display.

    The fit is used only for visualization when operational gluing coefficients
    are unavailable.  A clean gluing region should make the two detector modes
    nearly linearly related after this transformation.
    """
    lower = np.asarray(lower_signal, dtype=np.float64)
    upper = np.asarray(upper_signal, dtype=np.float64)
    alt = np.asarray(altitude_km, dtype=np.float64)
    valid = (
        np.isfinite(lower)
        & np.isfinite(upper)
        & np.isfinite(alt)
        & (alt >= min_alt_km)
        & (alt <= max_alt_km)
        & (lower > 0.0)
        & (upper > 0.0)
    )
    if valid.sum() < 10:
        return 1.0, 0.0, "unscaled AN"
    slope, intercept = np.polyfit(lower[valid], upper[valid], 1)
    return float(slope), float(intercept), "AN scaled for display"


def add_atmospheric_boundaries(ax: Any, ds: xr.Dataset, max_alt_km: float) -> bool:
    """Add PBL and tropopause reference lines to an axis when available."""
    has_legend = False
    if "PBL_Height_km" in ds:
        try:
            pbl_km = float(ds["PBL_Height_km"].mean(skipna=True).values)
            if np.isfinite(pbl_km) and 0 < pbl_km <= max_alt_km:
                ax.axvline(pbl_km, color="crimson", linestyle="--", linewidth=1.2, label=f"Mean PBL ({pbl_km:.1f} km)")
                has_legend = True
        except Exception:
            pass

    for attr_name, label, color, linestyle in (
        ("tropopause_cpt_km", "CPT", "royalblue", ":"),
        ("tropopause_lrt_km", "LRT", "forestgreen", "-."),
    ):
        try:
            value = float(ds.attrs.get(attr_name, -999.0))
            if np.isfinite(value) and 0 < value <= max_alt_km:
                ax.axvline(value, color=color, linestyle=linestyle, linewidth=1.2, label=f"{label} ({value:.1f} km)")
                has_legend = True
        except Exception:
            pass
    return has_legend


def infer_l1_channels_for_wavelength(ds_l1: xr.Dataset | None, wavelength_nm: int | float) -> tuple[str | None, str | None]:
    """Infer Analog and Photon Counting channel names for a wavelength."""
    if ds_l1 is None or "channel" not in ds_l1.coords:
        return None, None
    wavelength = str(int(wavelength_nm))
    channels = [str(channel) for channel in ds_l1["channel"].values]
    analog = next((ch for ch in channels if ch.startswith(f"{wavelength}.") and ch.upper().endswith(".AN")), None)
    photon = next((ch for ch in channels if ch.startswith(f"{wavelength}.") and (ch.upper().endswith(".PC") or ch.upper().endswith(".PH"))), None)
    return analog, photon

def plot_qa_gluing(
    ds_l1: xr.Dataset | None,
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
) -> Path | None:
    """Plot clean vertical gluing QA with one main profile panel and a zoom inset.
    """
    wavelength = int(wavelength_nm)
    required = {"glued_range_corrected_signal", "gluing_success_flag", "gluing_split_altitude_m"}
    if not required.issubset(set(ds_l2.data_vars)):
        return None

    output_format, dpi = get_output_settings(config)
    date_title, _ = extract_datetime_strings(ds_l2)

    altitude_m = np.asarray(ds_l2["altitude"].values, dtype=np.float64)
    if np.nanmax(altitude_m) <= 100.0:
        altitude_m = altitude_m * 1000.0
    alt_km = altitude_m / 1000.0

    max_alt_km = min(20.0, float(np.nanmax(alt_km)))
    valid_alt = alt_km <= max_alt_km

    color = channel_color(wavelength)
    smooth_bins = int(config.get("visualization", {}).get("level2_qa", {}).get("smooth_bins", 15))

    success = ds_l2["gluing_success_flag"].sel(wavelength=wavelength)
    success_values = np.asarray(success.values, dtype=float)
    n_blocks = int(success.size)
    success_count = int(np.nansum(success_values))
    success_rate = 100.0 * success_count / max(n_blocks, 1)
    valid_success = np.isfinite(success_values) & (success_values == 1)

    fallback_count = 0
    if "gluing_fallback_flag" in ds_l2:
        fallback_values = np.asarray(
            ds_l2["gluing_fallback_flag"].sel(wavelength=wavelength).values,
            dtype=float,
        )
        fallback_count = int(np.nansum(fallback_values))

    split_alt_km = ds_l2["gluing_split_altitude_m"].sel(wavelength=wavelength) / 1000.0
    median_split = float(np.nanmedian(split_alt_km.values)) if np.any(np.isfinite(split_alt_km.values)) else np.nan

    median_start = _safe_median_dataarray(ds_l2, "gluing_start_altitude_m", wavelength) / 1000.0
    median_stop = _safe_median_dataarray(ds_l2, "gluing_stop_altitude_m", wavelength) / 1000.0

    corr_med = _safe_median_dataarray(ds_l2, "gluing_correlation", wavelength)
    rmse_med = _safe_median_dataarray(ds_l2, "gluing_relative_rmse", wavelength)
    bias_med = _safe_median_dataarray(ds_l2, "gluing_relative_bias", wavelength)
    slope_med = _safe_median_dataarray(ds_l2, "gluing_slope", wavelength)
    intercept_med = _safe_median_dataarray(ds_l2, "gluing_intercept", wavelength)

    glued = ds_l2["glued_range_corrected_signal"].sel(wavelength=wavelength)
    glued_profile = _smooth_for_plot(safe_time_mean(glued).values, smooth_bins)

    # Prefer corrected-signal mean, because operational gluing is done before RCS.
    # Convert to RCS only for visual comparison.
    if "glued_corrected_signal_mean" in ds_l2:
        glued_corrected = np.asarray(
            ds_l2["glued_corrected_signal_mean"].sel(wavelength=wavelength).values,
            dtype=np.float64,
        )
        glued_profile = _smooth_for_plot(glued_corrected * altitude_m**2, smooth_bins)

    analog_profile_plot = np.full_like(glued_profile, np.nan, dtype=np.float64)
    photon_profile_plot = np.full_like(glued_profile, np.nan, dtype=np.float64)
    analog_ch = photon_ch = None
    scaling_note = "L1 channels unavailable"
    clip_mean = np.nan

    if ds_l1 is not None:
        analog_ch, photon_ch = infer_l1_channels_for_wavelength(ds_l1, wavelength)

        if analog_ch is not None and photon_ch is not None:
            if "corrected_signal" in ds_l1:
                analog_corrected = safe_time_mean(ds_l1["corrected_signal"].sel(channel=analog_ch)).values
                photon_corrected = safe_time_mean(ds_l1["corrected_signal"].sel(channel=photon_ch)).values

                if (
                    bool(valid_success.any())
                    and np.isfinite(slope_med)
                    and np.isfinite(intercept_med)
                    and slope_med > 0.0
                ):
                    analog_scaled = slope_med * analog_corrected + intercept_med
                    scaling_note = "AN scaled with operational coefficients"
                else:
                    scale_factor, scale_bins = _legacy_scale_factor(analog_corrected, photon_corrected)
                    analog_scaled = analog_corrected * scale_factor
                    scaling_note = f"AN display-scaled {scale_bins[0]}:{scale_bins[1]}"

                analog_profile_plot = _smooth_for_plot(analog_scaled * altitude_m**2, smooth_bins)
                photon_profile_plot = _smooth_for_plot(photon_corrected * altitude_m**2, smooth_bins)

            elif "range_corrected_signal" in ds_l1:
                analog_rcs = safe_time_mean(ds_l1["range_corrected_signal"].sel(channel=analog_ch)).values
                photon_rcs = safe_time_mean(ds_l1["range_corrected_signal"].sel(channel=photon_ch)).values

                scale_factor, scale_bins = _legacy_scale_factor(analog_rcs, photon_rcs)
                analog_profile_plot = _smooth_for_plot(analog_rcs * scale_factor, smooth_bins)
                photon_profile_plot = _smooth_for_plot(photon_rcs, smooth_bins)
                scaling_note = f"AN RCS display-scaled {scale_bins[0]}:{scale_bins[1]}"

            if "deadtime_clipping_fraction" in ds_l1:
                try:
                    clip = ds_l1["deadtime_clipping_fraction"].sel(channel=photon_ch).values
                    clip_mean = float(np.nanmean(clip))
                except Exception:
                    clip_mean = np.nan

    # ------------------------------------------------------------------
    # Robust x-limits for main panel.
    # Use positive RCS values only and ignore extreme tails.
    # ------------------------------------------------------------------
    main_vals = []
    for arr in (analog_profile_plot[valid_alt], photon_profile_plot[valid_alt], glued_profile[valid_alt]):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size:
            main_vals.append(arr)

    if main_vals:
        all_vals = np.concatenate(main_vals)
        xmin = float(np.nanpercentile(all_vals, 1.0))
        xmax = float(np.nanpercentile(all_vals, 99.5))
        xmin = max(xmin, 1e-3)
        xmax = max(xmax, xmin * 10.0)
    else:
        xmin, xmax = 1e-3, 1.0

    # Portrait-ish 3:4 layout.
    fig, ax = plt.subplots(figsize=(9.0, 12.0))

    # ------------------------------------------------------------------
    # Main curves
    # ------------------------------------------------------------------
    ax.plot(
        analog_profile_plot[valid_alt],
        alt_km[valid_alt],
        linestyle="--",
        linewidth=1.8,
        color="tab:blue",
        label=f"{analog_ch or 'AN'} scaled",
    )
    ax.plot(
        photon_profile_plot[valid_alt],
        alt_km[valid_alt],
        linestyle=":",
        linewidth=2.0,
        color="tab:orange",
        label=f"{photon_ch or 'PC'} mean",
    )
    ax.plot(
        glued_profile[valid_alt],
        alt_km[valid_alt],
        color=color,
        linewidth=2.5,
        label="Glued mean",
    )

    # Visible gluing window.
    if np.isfinite(median_start) and np.isfinite(median_stop) and 0.0 < median_start < median_stop <= max_alt_km:
        ax.axhspan(
            median_start,
            median_stop,
            color="gold",
            alpha=0.30,
            zorder=0,
            label=f"Median gluing window {median_start:.2f}-{median_stop:.2f} km",
        )

    # Median split line.
    if np.isfinite(median_split) and 0.0 < median_split <= max_alt_km:
        ax.axhline(
            median_split,
            color="black",
            linestyle="-.",
            linewidth=1.5,
            label=f"Median split {median_split:.2f} km",
        )

    # ax.set_title(
    #     f"Gluing profile - {format_wavelength_label(wavelength)}",
    #     fontsize=16,
    #     fontweight="bold",
    # )
    ax.set_xlabel("RCS on photon-counting scale [a.u.]", fontsize=13, fontweight="bold")
    ax.set_ylabel("Altitude (km a.g.l.)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max_alt_km)
    ax.set_xlim(xmin, xmax)
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.42)

    # Cleaner summary box. No repeated split/window text.
    summary_lines = [
        f"success = {success_count}/{n_blocks} ({success_rate:.1f}%)",
        f"corr med = {corr_med:.4g}",
        f"rmse med = {rmse_med:.4g}",
        f"bias med = {bias_med:.4g}",
    ]
    if np.isfinite(clip_mean):
        summary_lines.append(f"deadtime clipped = {100.0 * clip_mean:.2f}%")
    summary_lines.append(scaling_note)

    ax.text(
        0.02,
        0.985,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=9.0,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "gray"},
    )

    # ------------------------------------------------------------------
    # Zoom inset on the left.
    # ------------------------------------------------------------------
    if np.isfinite(median_start) and np.isfinite(median_stop):
        zoom_ymin = max(0.0, median_start - 0.5)
        zoom_ymax = min(max_alt_km, median_stop + 0.5)
    elif np.isfinite(median_split):
        zoom_ymin = max(0.0, median_split - 0.5)
        zoom_ymax = min(max_alt_km, median_split + 0.5)
    else:
        zoom_ymin = 1.0
        zoom_ymax = min(max_alt_km, 4.0)

    zoom_mask = (
        (alt_km >= zoom_ymin)
        & (alt_km <= zoom_ymax)
        & np.isfinite(analog_profile_plot)
        & np.isfinite(photon_profile_plot)
        & np.isfinite(glued_profile)
    )

    if zoom_mask.sum() > 5:
        zoom_vals = np.concatenate(
            [
                analog_profile_plot[zoom_mask],
                photon_profile_plot[zoom_mask],
                glued_profile[zoom_mask],
            ]
        )
        zoom_vals = zoom_vals[np.isfinite(zoom_vals) & (zoom_vals > 0)]

        if zoom_vals.size > 5:
            zmin = float(np.nanpercentile(zoom_vals, 1.0))
            zmax = float(np.nanpercentile(zoom_vals, 99.0))

            if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
                # left-side inset: [x0, y0, width, height]
                inset = ax.inset_axes([0.08, 0.10, 0.38, 0.30])

                inset.plot(
                    analog_profile_plot[zoom_mask],
                    alt_km[zoom_mask],
                    linestyle="--",
                    linewidth=1.2,
                    color="tab:blue",
                )
                inset.plot(
                    photon_profile_plot[zoom_mask],
                    alt_km[zoom_mask],
                    linestyle=":",
                    linewidth=1.4,
                    color="tab:orange",
                )
                inset.plot(
                    glued_profile[zoom_mask],
                    alt_km[zoom_mask],
                    linewidth=1.6,
                    color=color,
                )

                if np.isfinite(median_start) and np.isfinite(median_stop):
                    inset.axhspan(median_start, median_stop, color="gold", alpha=0.30)
                if np.isfinite(median_split):
                    inset.axhline(median_split, color="black", linestyle="-.", linewidth=1.0)

                pad = 0.08 * (zmax - zmin)
                inset.set_xlim(max(zmin - pad, 1e-6), zmax + pad)
                inset.set_ylim(zoom_ymin, zoom_ymax)
                inset.set_title("Zoom near gluing region", fontsize=8.5)
                inset.tick_params(labelsize=7)
                inset.grid(True, alpha=0.35)

    # Put legend on right to avoid the left inset/summary.
    ax.legend(fontsize=8.8, loc="center right")

    fig.suptitle(
        f"MILGRAU Level 2 QA - Signal Gluing - {format_wavelength_label(wavelength)}\n{date_title}",
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )

    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.13, right=0.96)
    add_footer_and_logos(fig, root_dir)

    out_path = Path(output_folder) / f"QA_Gluing_{file_name_prefix}_{wavelength}nm.{output_format}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_qa_molecular_fit(
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
) -> Path | None:
    """Plot QA for Rayleigh molecular calibration."""
    wavelength = int(wavelength_nm)
    required = {"glued_range_corrected_signal_mean", "scaled_molecular_range_corrected_signal", "rayleigh_calibration_factor", "rayleigh_reference_altitude_m"}
    if not required.issubset(set(ds_l2.data_vars)):
        return None

    output_format, dpi = get_output_settings(config)
    date_title, _ = extract_datetime_strings(ds_l2)
    alt_km = altitude_to_km(ds_l2["altitude"].values)
    max_alt_km = min(30.0, float(np.nanmax(alt_km)))
    valid_alt = alt_km <= max_alt_km
    smooth_bins = int(config.get("visualization", {}).get("level2_qa", {}).get("smooth_bins", 15))
    mean_glued = _smooth_for_plot(ds_l2["glued_range_corrected_signal_mean"].sel(wavelength=wavelength).values, smooth_bins)
    rayleigh_rcs = _smooth_for_plot(ds_l2["scaled_molecular_range_corrected_signal"].sel(wavelength=wavelength).values, smooth_bins)
    ref_alt_km = float(ds_l2["rayleigh_reference_altitude_m"].sel(wavelength=wavelength).values) / 1000.0
    calibration_factor = float(ds_l2["rayleigh_calibration_factor"].sel(wavelength=wavelength).values)
    calibration_intercept = float(ds_l2.get("rayleigh_calibration_intercept", xr.zeros_like(ds_l2["rayleigh_calibration_factor"])).sel(wavelength=wavelength).values)

    fig = plt.figure(figsize=(13.5, 8.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.1], wspace=0.25)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    fit_cfg = config.get("inversion", {}).get("molecular_fit", {}) or {}
    ref_min_km = float(fit_cfg.get("ref_alt_min_m", np.nan)) / 1000.0
    ref_max_km = float(fit_cfg.get("ref_alt_max_m", np.nan)) / 1000.0
    ref_window = (alt_km >= ref_min_km) & (alt_km <= ref_max_km) & np.isfinite(mean_glued) & np.isfinite(rayleigh_rcs)

    if np.any(ref_window):
        ax0.plot(rayleigh_rcs[ref_window], mean_glued[ref_window], color="royalblue", linewidth=1.4, label="Reference-region samples")
        x_fit = np.linspace(np.nanmin(rayleigh_rcs[ref_window]), np.nanmax(rayleigh_rcs[ref_window]), 100)
        ax0.plot(x_fit, x_fit + calibration_intercept, color="black", linestyle="--", linewidth=1.8, label="Linear fit diagnostic")
    ax0.set_title("Rayleigh fit region", fontsize=14, fontweight="bold")
    ax0.set_xlabel("Scaled molecular RCS [a.u.]", fontsize=12, fontweight="bold")
    ax0.set_ylabel("Measured glued RCS [a.u.]", fontsize=12, fontweight="bold")
    ax0.grid(True, alpha=0.45)
    ax0.legend(fontsize=9, loc="best")

    ax1.plot(mean_glued[valid_alt], alt_km[valid_alt], color=channel_color(wavelength), linewidth=2.2, label="Mean glued RCS")
    ax1.plot(rayleigh_rcs[valid_alt], alt_km[valid_alt], color="black", linestyle="--", linewidth=2.0, label="Scaled Rayleigh molecular RCS")
    if np.isfinite(ref_min_km) and np.isfinite(ref_max_km):
        ax1.axhspan(ref_min_km, ref_max_km, alpha=0.12, color="gray", label="Molecular-fit search range")
    if np.isfinite(ref_alt_km) and 0 < ref_alt_km <= max_alt_km:
        ax1.axhline(ref_alt_km, color="black", linestyle=":", linewidth=1.8, label=f"Reference {ref_alt_km:.2f} km")

    ax1.set_title(f"Molecular calibration profile\nslope={calibration_factor:.3g}, intercept={calibration_intercept:.3g}", fontsize=14, fontweight="bold")
    ax1.set_xlabel("RCS [a.u.]", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Altitude (km a.g.l.)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max_alt_km)
    ax1.set_xscale("symlog", linthresh=1e-3)
    ax1.grid(True, which="both", alpha=0.45)
    ax1.legend(fontsize=9, loc="best")

    fig.suptitle(f"MILGRAU Level 2 QA - Molecular Rayleigh Fit - {format_wavelength_label(wavelength)}\n{date_title}", fontsize=15, fontweight="bold", y=0.97)
    fig.subplots_adjust(top=0.84, bottom=0.14)
    add_footer_and_logos(fig, root_dir)
    out_path = Path(output_folder) / f"QA_Molecular_{file_name_prefix}_{wavelength}nm.{output_format}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_qa_scattering_ratio(
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
) -> Path | None:
    """Plot mean scattering ratio with block-variability uncertainty when available."""
    wavelength = int(wavelength_nm)
    if "scattering_ratio_mean" not in ds_l2:
        return None

    output_format, dpi = get_output_settings(config)
    date_title, _ = extract_datetime_strings(ds_l2)
    alt_km = altitude_to_km(ds_l2["altitude"].values)
    max_alt_km = min(30.0, float(np.nanmax(alt_km)))
    valid_alt = alt_km <= max_alt_km
    smooth_bins = int(config.get("visualization", {}).get("level2_qa", {}).get("smooth_bins", 15))
    sr = _smooth_for_plot(ds_l2["scattering_ratio_mean"].sel(wavelength=wavelength).values, smooth_bins)
    color = channel_color(wavelength)

    sr_sigma = np.full_like(sr, np.nan, dtype=np.float64)
    uncertainty_label = "Block SEM"
    if "scattering_ratio_error_mean" in ds_l2:
        sr_sigma = _smooth_for_plot(ds_l2["scattering_ratio_error_mean"].sel(wavelength=wavelength).values, smooth_bins)
        uncertainty_label = "SR 1σ"
    elif "scattering_ratio_block" in ds_l2:
        valid_block = None
        if "valid_retrieval_block_flag" in ds_l2:
            try:
                valid_block = np.asarray(ds_l2["valid_retrieval_block_flag"].sel(wavelength=wavelength).values, dtype=bool)
            except Exception:
                valid_block = None
        sr_sigma = _smooth_for_plot(_block_standard_error(ds_l2["scattering_ratio_block"].sel(wavelength=wavelength).values, valid_block), smooth_bins)

    fig, ax = plt.subplots(figsize=(8.6, 9.4))
    fig.subplots_adjust(top=0.86, bottom=0.14)
    has_uncertainty = np.isfinite(sr_sigma).any()
    if has_uncertainty:
        ax.fill_betweenx(alt_km[valid_alt], sr[valid_alt] - sr_sigma[valid_alt], sr[valid_alt] + sr_sigma[valid_alt], color=color, alpha=0.22, edgecolor="none", label=uncertainty_label)
    ax.plot(sr[valid_alt], alt_km[valid_alt], color=color, linewidth=2.2, label="Scattering ratio")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.4, label="Molecular reference SR=1")
    ax.set_title(f"Scattering Ratio - {format_wavelength_label(wavelength)}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Scattering ratio", fontsize=12, fontweight="bold")
    ax.set_ylabel("Altitude (km a.g.l.)", fontsize=12, fontweight="bold")
    xlim = _robust_positive_xlim(sr[valid_alt], default_max=6.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, max_alt_km)
    ax.grid(True, alpha=0.45)

    upper_mask = (alt_km >= 10.0) & valid_alt & np.isfinite(sr)
    notes = [f"Savgol plot smoothing = {smooth_bins} bins"]
    if np.any(upper_mask):
        mean_sr = float(np.nanmean(sr[upper_mask]))
        notes.insert(0, f"Mean SR above 10 km = {mean_sr:.2f}")
    if has_uncertainty and _uncertainty_exceeds_xlim(sr[valid_alt], sr_sigma[valid_alt], xlim):
        notes.append("Uncertainty band clipped by robust x-axis")
    ax.text(0.04, 0.96, "\n".join(notes), transform=ax.transAxes, fontsize=10, va="top", bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "gray"})
    ax.legend(fontsize=9, loc="best")

    fig.suptitle(f"MILGRAU Level 2 QA - Scattering Ratio - {format_wavelength_label(wavelength)}\n{date_title}", fontsize=15, fontweight="bold", y=0.97)
    add_footer_and_logos(fig, root_dir)
    out_path = Path(output_folder) / f"QA_ScatteringRatio_{file_name_prefix}_{wavelength}nm.{output_format}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_qa_l2_kfs(
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
    max_altitude_km: float = 30.0,
) -> Path | None:
    """Render Level 2 KFS QA panel with uncertainty bands clipped by robust x-limits."""
    wavelength = int(wavelength_nm)
    required = {"aerosol_backscatter", "aerosol_backscatter_error", "aerosol_extinction", "aerosol_extinction_error"}
    if not required.issubset(set(ds_l2.data_vars)):
        return None

    output_format, dpi = get_output_settings(config)
    date_title, _ = extract_datetime_strings(ds_l2)
    alt_km = altitude_to_km(ds_l2["altitude"].values)
    max_alt_km = min(float(max_altitude_km), float(np.nanmax(alt_km)))
    valid_alt = alt_km <= max_alt_km
    smooth_bins = int(config.get("visualization", {}).get("level2_qa", {}).get("smooth_bins", 15))

    beta = ds_l2["aerosol_backscatter"].sel(wavelength=wavelength)
    beta_err = ds_l2["aerosol_backscatter_error"].sel(wavelength=wavelength)
    alpha = ds_l2["aerosol_extinction"].sel(wavelength=wavelength)
    alpha_err = ds_l2["aerosol_extinction_error"].sel(wavelength=wavelength)
    beta_mean = _smooth_for_plot(safe_time_mean(beta).values, smooth_bins)
    beta_sigma = _smooth_for_plot(safe_error_of_mean(beta_err).values, smooth_bins)
    alpha_mean = _smooth_for_plot(safe_time_mean(alpha).values, smooth_bins)
    alpha_sigma = _smooth_for_plot(safe_error_of_mean(alpha_err).values, smooth_bins)

    beta_plot = beta_mean * 1e6
    beta_sigma_plot = beta_sigma * 1e6
    alpha_plot = alpha_mean * 1e6
    alpha_sigma_plot = alpha_sigma * 1e6
    beta_xlim = _robust_centered_xlim(beta_plot[valid_alt], default_abs=5.0)
    alpha_xlim = _robust_centered_xlim(alpha_plot[valid_alt], default_abs=50.0)
    beta_clipped = _uncertainty_exceeds_xlim(beta_plot[valid_alt], beta_sigma_plot[valid_alt], beta_xlim)
    alpha_clipped = _uncertainty_exceeds_xlim(alpha_plot[valid_alt], alpha_sigma_plot[valid_alt], alpha_xlim)

    color = channel_color(wavelength)
    fig = plt.figure(figsize=(13.5, 8.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharey=ax0)
    ax0.plot(beta_plot[valid_alt], alt_km[valid_alt], color=color, linewidth=2.2, label="Mean beta aer")
    ax0.fill_betweenx(alt_km[valid_alt], beta_plot[valid_alt] - beta_sigma_plot[valid_alt], beta_plot[valid_alt] + beta_sigma_plot[valid_alt], color=color, alpha=0.25, edgecolor="none", label="MC 1σ")
    ax0.axvline(0.0, color="black", linewidth=0.8)
    ax0.set_xlim(*beta_xlim)
    ax0.set_title("Aerosol backscatter", fontsize=14, fontweight="bold")
    ax0.set_xlabel(r"$\beta_{aer}$ [Mm$^{-1}$ sr$^{-1}$]", fontsize=12, fontweight="bold")
    ax0.set_ylabel("Altitude (km a.g.l.)", fontsize=12, fontweight="bold")
    ax0.set_ylim(0, max_alt_km)
    ax0.grid(True, alpha=0.45)
    if beta_clipped:
        ax0.text(0.04, 0.96, "MC 1σ clipped by robust x-axis", transform=ax0.transAxes, fontsize=9, va="top", bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "gray"})

    ax1.plot(alpha_plot[valid_alt], alt_km[valid_alt], color=color, linewidth=2.2, label="Mean alpha aer")
    ax1.fill_betweenx(alt_km[valid_alt], alpha_plot[valid_alt] - alpha_sigma_plot[valid_alt], alpha_plot[valid_alt] + alpha_sigma_plot[valid_alt], color=color, alpha=0.25, edgecolor="none", label="MC 1σ")
    ax1.axvline(0.0, color="black", linewidth=0.8)
    ax1.set_xlim(*alpha_xlim)
    ax1.set_title("Aerosol extinction", fontsize=14, fontweight="bold")
    ax1.set_xlabel(r"$\alpha_{aer}$ [Mm$^{-1}$]", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.45)
    if alpha_clipped:
        ax1.text(0.04, 0.96, "MC 1σ clipped by robust x-axis", transform=ax1.transAxes, fontsize=9, va="top", bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "gray"})
    plt.setp(ax1.get_yticklabels(), visible=False)

    for ax in (ax0, ax1):
        ax.legend(fontsize=9, loc="best")
    fig.suptitle(f"MILGRAU Level 2 QA - KFS Optical Retrieval - {format_wavelength_label(wavelength)}\n{date_title}", fontsize=15, fontweight="bold", y=0.97)
    fig.subplots_adjust(top=0.84, bottom=0.14)
    add_footer_and_logos(fig, root_dir)
    out_path = Path(output_folder) / f"QA_L2_KFS_{file_name_prefix}_{wavelength}nm.{output_format}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_all_level2_qa(
    ds_l2: xr.Dataset,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
    ds_l1: xr.Dataset | None = None,
) -> list[Path]:
    """Generate available Level 2 QA plots for each wavelength."""
    generated: list[Path] = []
    qa_cfg = config.get("visualization", {}).get("level2_qa", {}) or {}
    for wavelength_nm in get_wavelength_values(ds_l2):
        if bool(qa_cfg.get("generate_gluing_qa", True)):
            gluing_path = plot_qa_gluing(ds_l1, ds_l2, wavelength_nm, output_folder, file_name_prefix, config, root_dir)
            if gluing_path is not None:
                generated.append(gluing_path)
        if bool(qa_cfg.get("generate_molecular_fit_qa", True)):
            molecular_path = plot_qa_molecular_fit(ds_l2, wavelength_nm, output_folder, file_name_prefix, config, root_dir)
            if molecular_path is not None:
                generated.append(molecular_path)
        if bool(qa_cfg.get("generate_scattering_ratio_qa", True)):
            sr_path = plot_qa_scattering_ratio(ds_l2, wavelength_nm, output_folder, file_name_prefix, config, root_dir)
            if sr_path is not None:
                generated.append(sr_path)
        if bool(qa_cfg.get("generate_kfs_qa", True)):
            kfs_path = plot_qa_l2_kfs(ds_l2, wavelength_nm, output_folder, file_name_prefix, config, root_dir)
            if kfs_path is not None:
                generated.append(kfs_path)
    return generated
