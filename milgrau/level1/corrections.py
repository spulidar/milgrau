"""Level 1 instrumental correction kernels for lidar signals."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


def _safe_nanmax_xarray(data: xr.DataArray, default: float = 0.0) -> float:
    try:
        value = float(data.max(skipna=True).values)
        return value if np.isfinite(value) else float(default)
    except Exception:
        return float(default)


def _safe_nanmin_xarray(data: xr.DataArray, default: float = np.nan) -> float:
    try:
        value = float(data.min(skipna=True).values)
        return value if np.isfinite(value) else float(default)
    except Exception:
        return float(default)


def _shift_with_nan(data: xr.DataArray, shift: int) -> xr.DataArray:
    if shift == 0:
        return data.copy()
    return data.shift(range=int(shift), fill_value=np.nan)


def _shift_mask_with_false(data: xr.DataArray, shift: int) -> xr.DataArray:
    if shift == 0:
        return data.copy().astype(bool)
    return data.astype(bool).shift(range=int(shift), fill_value=False).astype(bool)


def _invalid_shift_mask(template: xr.DataArray, shift: int) -> xr.DataArray:
    return _shift_with_nan(xr.ones_like(template), shift).isnull()


def _fraction_over_range(mask: xr.DataArray) -> xr.DataArray:
    if "range" not in mask.dims:
        raise ValueError("Diagnostic mask must contain a 'range' dimension.")
    return mask.mean(dim="range", skipna=True)


def _shot_scale(shots: float | np.ndarray | xr.DataArray, sig: xr.DataArray) -> float | xr.DataArray:
    """Validate scalar or per-profile laser shots and return a broadcastable scale."""
    if isinstance(shots, xr.DataArray):
        values = np.asarray(shots.values, dtype=np.float64)
        if values.ndim == 0:
            scalar = float(values)
            if not np.isfinite(scalar) or scalar <= 0.0:
                raise ValueError(f"Invalid laser shots value: {scalar}")
            return scalar
        if shots.dims != ("time",):
            raise ValueError(f"Per-profile laser shots must have dimensions ('time',); got {shots.dims}.")
        if shots.sizes.get("time", 0) != sig.sizes.get("time", 0):
            raise ValueError("Per-profile laser shots length does not match the signal time dimension.")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Per-profile laser shots contain non-finite or non-positive values.")
        return xr.DataArray(values, dims=("time",), coords={"time": sig["time"]} if "time" in sig.coords else None)
    values = np.asarray(shots, dtype=np.float64)
    if values.ndim == 0:
        scalar = float(values)
        if not np.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"Invalid laser shots value: {scalar}")
        return scalar
    if values.ndim != 1 or values.size != sig.sizes.get("time", 0):
        raise ValueError("Per-profile laser shots must be a 1D array matching the signal time dimension.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Per-profile laser shots contain non-finite or non-positive values.")
    return xr.DataArray(values, dims=("time",), coords={"time": sig["time"]} if "time" in sig.coords else None)


def apply_instrumental_corrections(
    sig: xr.DataArray,
    z_da: xr.DataArray,
    shots: float | np.ndarray | xr.DataArray,
    bin_time_us: float,
    deadtime: float,
    shift: int,
    bg_offset: float,
    is_photon: bool,
    bg_mask: xr.DataArray,
    dc_prof: xr.DataArray | None = None,
    dc_err: xr.DataArray | None = None,
    deadtime_min_denominator: float = 0.05,
    pc_saturation_max_rate_mhz: float | None = None,
    return_diagnostics: bool = False,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray] | tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, dict[str, Any]]:
    """Apply Level 1 corrections, accepting SCC Laser_Shots per profile."""
    shots_scale = _shot_scale(shots, sig)
    if bin_time_us is None or not np.isfinite(float(bin_time_us)) or float(bin_time_us) <= 0.0:
        raise ValueError(f"Invalid bin_time_us value: {bin_time_us}")
    bin_time_us = float(bin_time_us)
    deadtime = float(deadtime)
    shift = int(shift)
    bg_offset = float(bg_offset)
    deadtime_min_denominator = float(deadtime_min_denominator)
    rate_scale = shots_scale * bin_time_us

    sig_dc = sig.copy()
    err_dc = xr.zeros_like(sig)
    if dc_prof is not None:
        sig_dc = sig_dc - dc_prof
        if dc_err is not None:
            err_dc = dc_err

    deadtime_clipped_mask = xr.zeros_like(sig_dc, dtype=bool)
    pc_saturation_mask = xr.zeros_like(sig_dc, dtype=bool)
    photon_rate_mhz_max = np.nan
    pc_saturation_rate_limit_mhz = np.nan
    deadtime_denominator_min = np.nan

    if not is_photon:
        sig_dt = sig_dc.copy()
        err_bg = sig_dt.where(bg_mask).std(dim="range", skipna=True)
        err_dt = xr.ones_like(sig_dt) * err_bg
        if dc_prof is not None and dc_err is not None:
            err_dt = np.sqrt(err_dt**2 + err_dc**2)
    else:
        sig_mhz = sig_dc / rate_scale
        photon_rate_mhz_max = _safe_nanmax_xarray(sig_mhz)
        dc_err_mhz = err_dc / rate_scale
        n_photons = xr.where(sig_dc > 0.0, sig_dc, 0.0)
        err_raw = np.sqrt(n_photons) / rate_scale
        if dc_prof is not None and dc_err is not None:
            err_raw = np.sqrt(err_raw**2 + dc_err_mhz**2)
        if pc_saturation_max_rate_mhz is not None and np.isfinite(float(pc_saturation_max_rate_mhz)) and float(pc_saturation_max_rate_mhz) > 0.0:
            pc_saturation_rate_limit_mhz = float(pc_saturation_max_rate_mhz)
            pc_saturation_mask = sig_mhz >= pc_saturation_rate_limit_mhz
        if deadtime > 0.0:
            denom = 1.0 - (sig_mhz * deadtime)
            deadtime_clipped_mask = denom < deadtime_min_denominator
            if not np.isfinite(pc_saturation_rate_limit_mhz):
                pc_saturation_mask = deadtime_clipped_mask.copy()
            deadtime_denominator_min = _safe_nanmin_xarray(denom)
            safe_denom = xr.where(deadtime_clipped_mask, deadtime_min_denominator, denom)
            sig_dt = sig_mhz / safe_denom
            err_dt = err_raw / (safe_denom**2)
        else:
            sig_dt, err_dt = sig_mhz, err_raw

    sig_shift = _shift_with_nan(sig_dt, shift)
    err_shift = _shift_with_nan(err_dt, shift)
    deadtime_clipped_mask_shift = _shift_mask_with_false(deadtime_clipped_mask, shift)
    pc_saturation_mask_shift = _shift_mask_with_false(pc_saturation_mask, shift)
    bin_shift_invalid_mask = _invalid_shift_mask(sig_dt, shift)
    bg_mean = sig_shift.where(bg_mask).mean(dim="range", skipna=True) - bg_offset
    n_bg = max(int(bg_mask.sum().values), 1)
    err_bg_mean = sig_shift.where(bg_mask).std(dim="range", skipna=True) / np.sqrt(n_bg)
    sig_c = sig_shift - bg_mean
    err_c = np.sqrt(err_shift**2 + err_bg_mean**2)
    rcs = sig_c * (z_da**2)
    err_rcs = err_c * (z_da**2)
    if not return_diagnostics:
        return sig_c, err_c, rcs, err_rcs
    diagnostics = {
        "deadtime_clipped_mask": deadtime_clipped_mask_shift,
        "pc_saturation_mask": pc_saturation_mask_shift,
        "deadtime_clipping_fraction": _fraction_over_range(deadtime_clipped_mask_shift),
        "pc_saturation_fraction": _fraction_over_range(pc_saturation_mask_shift),
        "deadtime_min_denominator_observed": deadtime_denominator_min,
        "deadtime_min_denominator_allowed": deadtime_min_denominator,
        "deadtime_correction_applied": bool(is_photon and deadtime > 0.0),
        "photon_rate_mhz_max": photon_rate_mhz_max,
        "pc_saturation_rate_limit_mhz": pc_saturation_rate_limit_mhz,
        "bin_shift_bins": shift,
        "bin_shift_invalid_mask": bin_shift_invalid_mask,
        "bin_shift_invalid_fraction": _fraction_over_range(bin_shift_invalid_mask),
    }
    return sig_c, err_c, rcs, err_rcs, diagnostics
