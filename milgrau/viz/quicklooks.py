"""Level 1 quicklook and mean-profile plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from milgrau.viz.config import resolve_visualization_config
from milgrau.viz.style import add_footer_and_logos, channel_color, get_output_settings

RCS_VARIABLE = "range_corrected_signal"
RCS_ERROR_VARIABLE = "range_corrected_signal_error"


def extract_datetime_strings(ds: xr.Dataset) -> tuple[str, str]:
    """Extract human-readable date strings from an xarray Dataset."""
    try:
        dt_in = pd.to_datetime(ds.time.values.min())
        dt_end = pd.to_datetime(ds.time.values.max())
        date_title = f"{dt_in.strftime('%d %b %Y - %H:%M')} to {dt_end.strftime('%H:%M')} UTC"
        date_footer = dt_in.strftime("%d %b %Y")
        return date_title, date_footer
    except Exception:
        return "Unknown date", "Unknown date"


def channel_file_token(raw_name: str) -> str:
    """Return a compact channel token such as 355AN for figure filenames."""
    text = str(raw_name).strip()
    parts = text.split(".")
    if len(parts) == 2:
        try:
            return f"{int(parts[0])}{parts[1].upper()}"
        except ValueError:
            pass
    return "".join(character for character in text if character.isalnum()) or "channel"


def format_channel_name(raw_name: str) -> str:
    """Convert an internal channel name such as '532.AN' into '532nm AN'."""
    try:
        parts = str(raw_name).split(".")
        return f"{int(parts[0])}nm {parts[1]}"
    except Exception:
        return str(raw_name)


def safe_time_mean(da: xr.DataArray) -> xr.DataArray:
    """Return the time mean of a DataArray when a time dimension is present."""
    if "time" in da.dims:
        return da.mean(dim="time", skipna=True)
    return da


def safe_error_of_mean(err_da: xr.DataArray) -> xr.DataArray:
    """Combine profile one-sigma errors over time as uncertainty of the mean."""
    if "time" not in err_da.dims:
        return err_da
    n_profiles = max(int(err_da.sizes.get("time", 1)), 1)
    return np.sqrt((err_da**2).sum(dim="time", skipna=True)) / n_profiles

def _streaming_time_statistics(
    signal: xr.DataArray,
    error: xr.DataArray,
    *,
    chunk_profiles: int,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute mean signal/error using bounded time chunks.

    This avoids materializing a full time-by-altitude channel twice while
    preserving the same mean/error-of-mean semantics used by the figures.
    """
    if "time" not in signal.dims or "altitude" not in signal.dims:
        return safe_time_mean(signal), safe_error_of_mean(error)

    sig = signal.transpose("time", "altitude")
    err = error.transpose("time", "altitude")
    if sig.shape != err.shape:
        raise ValueError("Signal and error arrays must have identical time/altitude shape.")

    n_time = int(sig.sizes["time"])
    n_altitude = int(sig.sizes["altitude"])
    chunk = max(int(chunk_profiles), 1)
    signal_sum = np.zeros(n_altitude, dtype=np.float64)
    signal_count = np.zeros(n_altitude, dtype=np.int64)
    error_square_sum = np.zeros(n_altitude, dtype=np.float64)

    for start in range(0, n_time, chunk):
        stop = min(start + chunk, n_time)
        sig_values = np.asarray(sig.isel(time=slice(start, stop)).values)
        err_values = np.asarray(err.isel(time=slice(start, stop)).values)

        finite_signal = np.isfinite(sig_values)
        signal_sum += np.nansum(sig_values, axis=0, dtype=np.float64)
        signal_count += np.count_nonzero(finite_signal, axis=0)

        finite_error = np.isfinite(err_values)
        error_square_sum += np.sum(
            np.where(finite_error, err_values, 0.0).astype(np.float64) ** 2,
            axis=0,
            dtype=np.float64,
        )

    mean_values = np.divide(
        signal_sum,
        signal_count,
        out=np.full(n_altitude, np.nan, dtype=np.float64),
        where=signal_count > 0,
    )
    error_values = np.sqrt(error_square_sum) / max(n_time, 1)
    altitude = sig["altitude"].values
    mean_da = xr.DataArray(
        mean_values,
        dims=("altitude",),
        coords={"altitude": altitude},
        attrs=signal.attrs,
        name=signal.name,
    )
    error_da = xr.DataArray(
        error_values,
        dims=("altitude",),
        coords={"altitude": altitude},
        attrs=error.attrs,
        name=error.name,
    )
    mean_da["altitude"].attrs.update(sig["altitude"].attrs)
    error_da["altitude"].attrs.update(sig["altitude"].attrs)
    return mean_da, error_da


def _decimate_for_display(
    data_slice: xr.DataArray,
    config: dict[str, Any],
) -> xr.DataArray:
    """Stride-sample only the rendered heatmap to bound Matplotlib memory."""
    if "time" not in data_slice.dims or "altitude" not in data_slice.dims:
        return data_slice
    quicklook = resolve_visualization_config(config).quicklook
    n_time = int(data_slice.sizes.get("time", 0))
    n_altitude = int(data_slice.sizes.get("altitude", 0))
    time_step = max(1, int(np.ceil(n_time / quicklook.max_time_samples)))
    altitude_step = max(1, int(np.ceil(n_altitude / quicklook.max_altitude_bins)))
    return data_slice.isel(
        time=slice(None, None, time_step),
        altitude=slice(None, None, altitude_step),
    )


def rolling_altitude(da: xr.DataArray, bins: int) -> xr.DataArray:
    """Apply centered rolling smoothing along altitude using an explicit bin count."""
    if "altitude" not in da.dims:
        return da
    bins = int(bins)
    if bins <= 0:
        raise ValueError("Smoothing bins must be positive.")
    return da.rolling(altitude=bins, min_periods=1, center=True).mean()


def _save_figure(fig: Any, out_path: str | Path, dpi: int) -> Path:
    """Save and close a Matplotlib figure."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _get_gap_threshold_minutes(config: dict[str, Any], data_slice: xr.DataArray) -> float:
    """Return the explicitly configured temporal-gap threshold."""
    del data_slice
    return resolve_visualization_config(config).quicklook.max_time_gap_minutes


def _insert_time_gap_markers(data_slice: xr.DataArray, config: dict[str, Any]) -> xr.DataArray:
    """Insert NaN profiles into large temporal gaps so quicklooks show missing data."""
    if "time" not in data_slice.dims or "altitude" not in data_slice.dims:
        return data_slice
    if data_slice.sizes.get("time", 0) < 2:
        return data_slice

    da = data_slice.transpose("time", "altitude")
    times = pd.to_datetime(da["time"].values)
    values = np.asarray(da.values)
    threshold = pd.Timedelta(minutes=_get_gap_threshold_minutes(config, da))
    marker_delta = pd.Timedelta(seconds=1)

    new_times: list[pd.Timestamp] = []
    new_profiles: list[np.ndarray] = []
    inserted = False

    for idx in range(len(times)):
        new_times.append(times[idx])
        new_profiles.append(values[idx, :])
        if idx == len(times) - 1:
            continue

        gap = times[idx + 1] - times[idx]
        if gap > threshold:
            left_marker = times[idx] + marker_delta
            right_marker = times[idx + 1] - marker_delta
            if right_marker <= left_marker:
                midpoint = times[idx] + gap / 2
                left_marker = midpoint
                right_marker = midpoint
            dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.float32
            nan_profile = np.full(values.shape[1], np.nan, dtype=dtype)
            new_times.extend([left_marker, right_marker])
            new_profiles.extend([nan_profile, nan_profile.copy()])
            inserted = True

    if not inserted:
        return data_slice

    result = xr.DataArray(
        np.stack(new_profiles, axis=0),
        dims=("time", "altitude"),
        coords={"time": np.asarray(new_times, dtype="datetime64[ns]"), "altitude": da["altitude"].values},
        attrs=da.attrs,
        name=da.name,
    )
    result["altitude"].attrs.update(da["altitude"].attrs)
    return result


def _quicklook_colormap(config: dict[str, Any]):
    """Return the explicitly configured colormap and missing-data color."""
    quicklook = resolve_visualization_config(config).quicklook
    cmap = plt.get_cmap(quicklook.colormap).copy()
    cmap.set_bad(color=quicklook.missing_data_color)
    return cmap


def plot_quicklook(
    data_slice: xr.DataArray,
    error_slice: xr.DataArray,
    max_altitude: float,
    channel_name: str,
    ds: xr.Dataset,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
    session_id: str | None = None,
    timezone_name: str | None = None,
    pbl_da: xr.DataArray | None = None,
    cpt_km: float = np.nan,
    lrt_km: float = np.nan,
    time_range_utc: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> Path:
    """Render one Level 1 RCS quicklook and side mean profile."""
    resolved = resolve_visualization_config(config)
    output_format, dpi = get_output_settings(config)
    date_title, _ = extract_datetime_strings(ds)
    pretty_channel = format_channel_name(channel_name)
    color = channel_color(channel_name)
    display_data = _insert_time_gap_markers(
        _decimate_for_display(data_slice, config),
        config,
    )

    fig = plt.figure(figsize=(15, 7.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.03)

    ax0 = plt.subplot(gs[0])
    plot = display_data.plot(
        x="time",
        y="altitude",
        cmap=_quicklook_colormap(config),
        robust=True,
        vmin=0,
        add_colorbar=False,
        ax=ax0,
        add_labels=False,
        rasterized=True,
    )

    lower_altitude = 0.16 if "AN" in pretty_channel else 0.5
    if time_range_utc is None:
        del session_id, timezone_name
        ax0.set_facecolor(resolve_visualization_config(config).quicklook.missing_data_color)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=9)
        ax0.xaxis.set_major_locator(locator)
        period_label = None
    else:
        start_utc, end_utc = time_range_utc
        if end_utc <= start_utc:
            raise ValueError("time_range_utc end must be later than start.")
        ax0.set_facecolor(resolve_visualization_config(config).quicklook.missing_data_color)
        ax0.set_xlim(start_utc, end_utc)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=9)
        ax0.xaxis.set_major_locator(locator)
        period_label = (
            f"UTC zoom: {start_utc.strftime('%Y-%m-%d %H:%M')}–"
            f"{end_utc.strftime('%H:%M')}"
        )
    ax0.set_title(
        f"RCS at {pretty_channel} (0 - {float(max_altitude):g} km)\n{date_title}",
        fontsize=15,
        fontweight="bold",
        loc="center",
    )
    ax0.set_xlabel("Time (UTC)", fontsize=13, fontweight="bold")
    ax0.set_ylabel("Altitude (km a.g.l.)", fontsize=13, fontweight="bold")
    ax0.set_ylim(lower_altitude, max_altitude)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax1 = plt.subplot(gs[1], sharey=ax0)
    smooth_bins = resolved.quicklook.mean_profile_smooth_bins
    mean_profile, mean_error = _streaming_time_statistics(
        data_slice,
        error_slice,
        chunk_profiles=resolved.quicklook.mean_chunk_profiles,
    )
    smooth_profile = rolling_altitude(mean_profile, bins=smooth_bins)
    smooth_error = rolling_altitude(mean_error, bins=smooth_bins)
    ax1.plot(smooth_profile, smooth_profile.altitude, color=color, linewidth=2)
    ax1.fill_betweenx(
        smooth_profile.altitude,
        smooth_profile - smooth_error,
        smooth_profile + smooth_error,
        color=color,
        alpha=0.3,
        edgecolor="none",
    )
    ax1.set_xlabel("Mean RCS", fontsize=12, fontweight="bold")
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.grid(True, linestyle="--", alpha=0.6, which="both")
    ax1.set_ylim(lower_altitude, max_altitude)

    values = np.asarray(smooth_profile.values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size:
        p_max = float(np.nanmax(finite_values))
        p_min = float(np.nanmin(finite_values))
        margin = max((p_max - p_min) * 0.15, 1e-12)
        ax1.set_xlim(min(0.0, p_min) - margin, p_max + margin)

    has_legend = False
    if resolved.quicklook.show_pbl and pbl_da is not None:
        try:
            mean_pbl = float(pbl_da.mean(skipna=True).values)
            if np.isfinite(mean_pbl) and 0 < mean_pbl <= max_altitude:
                ax1.axhline(mean_pbl, color="crimson", linestyle="--", linewidth=1.8, zorder=5, label=f"Mean PBL ({mean_pbl:.1f} km)")
                has_legend = True
        except Exception:
            pass

    if resolved.quicklook.show_tropopause:
        if np.isfinite(cpt_km) and 0 < cpt_km <= max_altitude:
            ax1.axhline(cpt_km, color="royalblue", linestyle=":", linewidth=1.8, zorder=5, label=f"CPT ({cpt_km:.1f} km)")
            has_legend = True
        if np.isfinite(lrt_km) and 0 < lrt_km <= max_altitude:
            ax1.axhline(lrt_km, color="forestgreen", linestyle="-.", linewidth=1.8, zorder=5, label=f"LRT ({lrt_km:.1f} km)")
            has_legend = True
    if has_legend:
        ax1.legend(loc="upper right", framealpha=0.9, fontsize=9, facecolor="white", edgecolor="black")

    plt.subplots_adjust(left=0.14, bottom=0.15, right=0.95, top=0.88)
    cb_ax = fig.add_axes([0.06, 0.15, 0.015, 0.73])
    cb = fig.colorbar(plot, cax=cb_ax, orientation="vertical")
    cb.set_label("Intensity [a.u.]", fontsize=12, fontweight="bold")
    cb_ax.yaxis.set_ticks_position("left")
    cb_ax.yaxis.set_label_position("left")
    footer_subtitle = period_label
    add_footer_and_logos(fig, root_dir, subtitle=footer_subtitle)

    out_path = Path(output_folder) / (
        f"{file_name_prefix}_L1_RCS_{channel_file_token(channel_name)}_"
        f"{float(max_altitude):g}km.{output_format}"
    )
    return _save_figure(fig, out_path, dpi=dpi)


def plot_global_mean_rcs(
    ds: xr.Dataset,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
) -> Path | None:
    """Render a comparative global mean RCS profile for configured channels."""
    resolved = resolve_visualization_config(config)
    output_format, dpi = get_output_settings(config)
    max_altitude = max(resolved.altitude_ranges_km)
    smooth_bins = resolved.quicklook.mean_profile_smooth_bins
    date_title, _ = extract_datetime_strings(ds)

    if RCS_VARIABLE not in ds or RCS_ERROR_VARIABLE not in ds:
        raise KeyError(f"Dataset must contain {RCS_VARIABLE} and {RCS_ERROR_VARIABLE}.")

    fig, ax = plt.subplots(figsize=(8, 9.6))
    fig.subplots_adjust(top=0.90, bottom=0.15)
    plotted = False
    available_channels = {str(channel) for channel in ds.channel.values}

    for channel_name in resolved.channels_to_plot:
        if channel_name not in available_channels:
            continue

        rc_sig = ds[RCS_VARIABLE].sel(channel=channel_name).where(ds["altitude"] <= max_altitude, drop=True)
        rc_err = ds[RCS_ERROR_VARIABLE].sel(channel=channel_name).where(ds["altitude"] <= max_altitude, drop=True)
        if rc_sig.size == 0:
            continue

        mean_raw, error_raw = _streaming_time_statistics(
            rc_sig,
            rc_err,
            chunk_profiles=resolved.quicklook.mean_chunk_profiles,
        )
        mean_prof = rolling_altitude(mean_raw, bins=smooth_bins)
        mean_err = rolling_altitude(error_raw, bins=smooth_bins)
        ax.plot(
            mean_prof,
            mean_prof.altitude,
            color=channel_color(channel_name),
            linestyle="-" if "an" in channel_name.lower() else "--",
            label=format_channel_name(channel_name),
            linewidth=2,
        )
        ax.fill_betweenx(mean_prof.altitude, mean_prof - mean_err, mean_prof + mean_err, color=channel_color(channel_name), alpha=0.2, edgecolor="none")
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_title(f"Mean RCS (0 - {max_altitude:g} km)\n{date_title}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean RCS [a.u.]", fontsize=14, fontweight="bold")
    ax.set_ylabel("Altitude (km a.g.l.)", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim(0, max_altitude)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, which="both", alpha=0.5)
    add_footer_and_logos(fig, root_dir)

    out_path = Path(output_folder) / f"{file_name_prefix}_L1_MeanRCS.{output_format}"
    return _save_figure(fig, out_path, dpi=dpi)
