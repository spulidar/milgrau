"""Planetary Boundary Layer diagnostics for lidar RCS profiles."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import xarray as xr

from milgrau.level1.config import resolve_level1_config


def calculate_pbl_height_gradient(
    rcs_signal: np.ndarray,
    alt_m: np.ndarray,
    min_search_m: float,
    max_search_m: float,
    smooth_bins: int,
) -> float:
    """Estimate PBL height with the strongest negative RCS gradient method.

    The smoothing step uses edge padding before convolution. This avoids false
    negative gradients at the search-window boundaries, which can otherwise
    appear when ``np.convolve(..., mode="same")`` implicitly pads with zeros.
    Productive callers must provide the scientific search and smoothing settings
    explicitly; this numerical kernel does not invent them.
    """
    rcs_signal = np.asarray(rcs_signal, dtype=np.float64)
    alt_m = np.asarray(alt_m, dtype=np.float64)

    if rcs_signal.ndim != 1 or alt_m.ndim != 1 or rcs_signal.size != alt_m.size:
        return np.nan

    smooth_bins = int(smooth_bins)
    if smooth_bins < 3 or smooth_bins % 2 == 0:
        raise ValueError("smooth_bins must be an odd integer >= 3.")
    if not np.isfinite(float(min_search_m)) or not np.isfinite(float(max_search_m)):
        raise ValueError("PBL search altitudes must be finite.")
    if float(max_search_m) <= float(min_search_m):
        raise ValueError("max_search_m must exceed min_search_m.")

    valid_idx = np.where(
        (alt_m >= float(min_search_m))
        & (alt_m <= float(max_search_m))
        & np.isfinite(alt_m)
        & np.isfinite(rcs_signal)
    )[0]
    if len(valid_idx) < smooth_bins:
        return np.nan

    search_alt = alt_m[valid_idx]
    search_rcs = rcs_signal[valid_idx]
    finite = np.isfinite(search_rcs)
    if finite.sum() < smooth_bins:
        return np.nan

    median_val = np.nanmedian(search_rcs[finite])
    search_rcs = np.where(np.isfinite(search_rcs), search_rcs, median_val)

    edge_trim = smooth_bins // 2
    window = np.ones(smooth_bins, dtype=np.float64) / smooth_bins
    padded_rcs = np.pad(search_rcs, pad_width=edge_trim, mode="edge")
    smoothed_rcs = np.convolve(padded_rcs, window, mode="valid")
    gradient = np.gradient(smoothed_rcs, search_alt)

    if len(gradient) > 2 * edge_trim:
        min_grad_idx = int(np.argmin(gradient[edge_trim:-edge_trim])) + edge_trim
    else:
        min_grad_idx = int(np.argmin(gradient))

    if not np.isfinite(gradient[min_grad_idx]) or gradient[min_grad_idx] >= 0.0:
        return np.nan
    return float(search_alt[min_grad_idx] / 1000.0)


def estimate_pbl_timeseries(
    final_ds: xr.Dataset,
    z_arr: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> xr.Dataset:
    """Estimate PBL using the explicitly configured reference channel and settings.

    PBL is a diagnostic: a missing configured channel does not substitute another
    signal. The diagnostic is omitted with a warning rather than silently changing
    the scientific method.
    """
    pbl = resolve_level1_config(config).pbl
    channels = set(final_ds.channel.values.astype(str))
    if pbl.reference_channel not in channels:
        logger.warning(
            "  -> PBL diagnostic unavailable: configured reference channel %s is absent; no fallback channel will be used.",
            pbl.reference_channel,
        )
        return final_ds
    if "channel_correction_success" in final_ds:
        correction_ok = int(final_ds["channel_correction_success"].sel(channel=pbl.reference_channel).item()) == 1
        if not correction_ok:
            logger.warning(
                "  -> PBL diagnostic unavailable: configured reference channel %s failed Level 1 correction.",
                pbl.reference_channel,
            )
            return final_ds

    rcs_matrix = final_ds["range_corrected_signal"].sel(channel=pbl.reference_channel).values
    logger.info(
        "  -> Tracking PBL using %s (%.0f-%.0f m).",
        pbl.reference_channel,
        pbl.min_search_altitude_m,
        pbl.max_search_altitude_m,
    )
    pbl_h = [
        calculate_pbl_height_gradient(
            rcs_matrix[t, :],
            z_arr,
            min_search_m=pbl.min_search_altitude_m,
            max_search_m=pbl.max_search_altitude_m,
            smooth_bins=pbl.smooth_bins,
        )
        for t in range(rcs_matrix.shape[0])
    ]
    final_ds["PBL_Height_km"] = xr.DataArray(
        pbl_h,
        dims=["time"],
        coords={"time": final_ds.time},
    ).astype(np.float32)
    final_ds["PBL_Height_km"].attrs = {
        "units": "km",
        "method": "Gradient method on smoothed RCS",
        "reference_channel": pbl.reference_channel,
        "min_search_m": pbl.min_search_altitude_m,
        "max_search_m": pbl.max_search_altitude_m,
        "smooth_bins": pbl.smooth_bins,
    }
    return final_ds
