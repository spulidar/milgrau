"""Level 1 instrumental correction kernels for lidar signals."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


def _safe_nanmax_xarray(data: xr.DataArray, default: float = 0.0) -> float:
    try:
        value = float(data.max(skipna=True).values)
        return value if np.isfinite(value) else float(default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _safe_nanmin_xarray(data: xr.DataArray, default: float = np.nan) -> float:
    try:
        value = float(data.min(skipna=True).values)
        return value if np.isfinite(value) else float(default)
    except (TypeError, ValueError, OverflowError):
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


def _max_over_range(data: xr.DataArray) -> xr.DataArray:
    if "range" not in data.dims:
        raise ValueError("Diagnostic data must contain a 'range' dimension.")
    return data.max(dim="range", skipna=True)


def _robust_background_statistics(
    signal: xr.DataArray,
    background_mask: xr.DataArray,
    *,
    minimum_valid_bins: int,
    outlier_sigma: float,
) -> dict[str, xr.DataArray]:
    """Estimate a per-profile background using median and MAD.

    The Gaussian-consistent robust scale is ``1.4826 * MAD``.  The reported
    location uncertainty uses the asymptotic Gaussian standard error of the
    median, ``1.2533 * sigma / sqrt(n)``.  A zero MAD remains zero rather than
    falling back to a non-robust standard deviation that one extreme value can
    dominate; differing samples are still counted as outliers in that case.
    """
    if "range" not in signal.dims or background_mask.dims != ("range",):
        raise ValueError("Robust background estimation requires a signal range dimension and a 1D mask.")
    minimum = int(minimum_valid_bins)
    threshold_sigma = float(outlier_sigma)
    if minimum < 1:
        raise ValueError("minimum_valid_bins must be at least one.")
    if not np.isfinite(threshold_sigma) or threshold_sigma <= 0.0:
        raise ValueError("outlier_sigma must be finite and positive.")

    window = signal.where(background_mask)
    count = window.count(dim="range")
    location = window.median(dim="range", skipna=True)
    absolute_deviation = np.abs(window - location)
    mad = absolute_deviation.median(dim="range", skipna=True)
    mad_scale = 1.482602218505602 * mad
    scale = mad_scale
    supported = count >= minimum
    location = location.where(supported)
    scale = scale.where(supported)
    standard_error = (1.2533141373155 * scale / np.sqrt(count)).where(supported)
    outlier = xr.where(
        scale > 0.0,
        absolute_deviation > (threshold_sigma * scale),
        absolute_deviation > 0.0,
    )
    outlier_fraction = (
        outlier.where(window.notnull()).sum(dim="range", skipna=True) / count
    ).where(supported)
    return {
        "location": location,
        "scale": scale,
        "standard_error": standard_error,
        "valid_bins": count,
        "outlier_fraction": outlier_fraction,
    }


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
    *,
    deadtime_min_denominator: float,
    pc_saturation_max_rate_mhz: float | None,
    background_minimum_valid_bins: int = 1,
    background_outlier_sigma: float = 4.5,
    return_diagnostics: bool = False,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray] | tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, dict[str, Any]]:
    """Apply Level 1 corrections, accepting SCC Laser_Shots per profile.

    For photon-counting channels, the observed rate diagnostic is calculated
    directly from acquired counts and laser shots before dark-current
    subtraction. The current productive correction order remains unchanged:
    dark current is subtracted in native counts before the non-paralyzable
    dead-time correction. This order is under explicit instrument review and
    must not be changed without quantified real-data evidence.

    The Poisson term is calculated from observed accumulated counts before
    dark-current subtraction. The uncertainty of the estimated dark-current
    profile is treated as an independent term and combined in quadrature after
    both are converted to MHz.

    Numerical dead-time clipping and physical detector saturation are distinct
    diagnostics. A channel without a characterized physical saturation rate is
    never silently labelled saturated merely because the numerical dead-time
    denominator was clipped.
    """
    shots_scale = _shot_scale(shots, sig)
    if bin_time_us is None or not np.isfinite(float(bin_time_us)) or float(bin_time_us) <= 0.0:
        raise ValueError(f"Invalid bin_time_us value: {bin_time_us}")
    bin_time_us = float(bin_time_us)
    deadtime = float(deadtime)
    shift = int(shift)
    bg_offset = float(bg_offset)
    deadtime_min_denominator = float(deadtime_min_denominator)
    if not np.isfinite(deadtime_min_denominator) or not 0.0 < deadtime_min_denominator <= 1.0:
        raise ValueError(
            "deadtime_min_denominator must be a finite value in the interval (0, 1]."
        )
    rate_scale = shots_scale * bin_time_us

    sig_dc = sig.copy()
    err_dc = xr.zeros_like(sig)
    if dc_prof is not None:
        sig_dc = sig_dc - dc_prof
        if dc_err is not None:
            err_dc = dc_err

    deadtime_clipped_mask = xr.zeros_like(sig_dc, dtype=bool)
    raw_deadtime_clipped_mask = xr.zeros_like(sig, dtype=bool)
    pc_saturation_mask = xr.zeros_like(sig, dtype=bool)
    pc_observed_rate_mhz_max = xr.full_like(sig.isel(range=0), np.nan, dtype=np.float64)
    pc_saturation_rate_limit_mhz = np.nan
    pc_saturation_characterized = False
    deadtime_denominator_min = np.nan
    raw_deadtime_denominator_min = np.nan

    if not is_photon:
        if pc_saturation_max_rate_mhz is not None:
            raise ValueError("pc_saturation_max_rate_mhz is only valid for photon-counting channels.")
        sig_dt = sig_dc.copy()
        analog_noise = _robust_background_statistics(
            sig_dt,
            bg_mask,
            minimum_valid_bins=background_minimum_valid_bins,
            outlier_sigma=background_outlier_sigma,
        )
        err_bg = analog_noise["scale"]
        err_dt = xr.ones_like(sig_dt) * err_bg
        if dc_prof is not None and dc_err is not None:
            err_dt = np.sqrt(err_dt**2 + err_dc**2)
    else:
        observed_rate_mhz = sig / rate_scale
        pc_observed_rate_mhz_max = _max_over_range(observed_rate_mhz)
        sig_mhz = sig_dc / rate_scale

        # Raw-count shot noise belongs to the observed counts N, not to the
        # dark-subtracted counts N-D. The estimated dark-current uncertainty
        # sigma_D is independent, so in rate units:
        # sigma^2 = N / rate_scale^2 + sigma_D^2 / rate_scale^2.
        raw_counts = xr.where(sig > 0.0, sig, 0.0)
        err_poisson_mhz = np.sqrt(raw_counts) / rate_scale
        err_dark_mhz = err_dc / rate_scale
        err_raw = err_poisson_mhz
        if dc_prof is not None and dc_err is not None:
            err_raw = np.sqrt(err_poisson_mhz**2 + err_dark_mhz**2)

        if pc_saturation_max_rate_mhz is not None:
            saturation_limit = float(pc_saturation_max_rate_mhz)
            if not np.isfinite(saturation_limit) or saturation_limit <= 0.0:
                raise ValueError("pc_saturation_max_rate_mhz must be positive and finite when characterized.")
            pc_saturation_rate_limit_mhz = saturation_limit
            pc_saturation_characterized = True
            pc_saturation_mask = observed_rate_mhz >= saturation_limit

        if deadtime > 0.0:
            raw_denom = 1.0 - (observed_rate_mhz * deadtime)
            raw_deadtime_clipped_mask = raw_denom < deadtime_min_denominator
            raw_deadtime_denominator_min = _safe_nanmin_xarray(raw_denom)

            denom = 1.0 - (sig_mhz * deadtime)
            deadtime_clipped_mask = denom < deadtime_min_denominator
            deadtime_denominator_min = _safe_nanmin_xarray(denom)
            safe_denom = xr.where(deadtime_clipped_mask, deadtime_min_denominator, denom)
            sig_dt = sig_mhz / safe_denom
            err_dt = err_raw / (safe_denom**2)
        else:
            sig_dt, err_dt = sig_mhz, err_raw

    sig_shift = _shift_with_nan(sig_dt, shift)
    err_shift = _shift_with_nan(err_dt, shift)
    deadtime_clipped_mask_shift = _shift_mask_with_false(deadtime_clipped_mask, shift)
    raw_deadtime_clipped_mask_shift = _shift_mask_with_false(raw_deadtime_clipped_mask, shift)
    pc_saturation_mask_shift = _shift_mask_with_false(pc_saturation_mask, shift)
    bin_shift_invalid_mask = _invalid_shift_mask(sig_dt, shift)
    background = _robust_background_statistics(
        sig_shift,
        bg_mask,
        minimum_valid_bins=background_minimum_valid_bins,
        outlier_sigma=background_outlier_sigma,
    )
    bg_location = background["location"] - bg_offset
    err_bg_location = background["standard_error"]
    sig_c = sig_shift - bg_location
    err_c = np.sqrt(err_shift**2 + err_bg_location**2)
    rcs = sig_c * (z_da**2)
    err_rcs = err_c * (z_da**2)
    if not return_diagnostics:
        return sig_c, err_c, rcs, err_rcs
    diagnostics = {
        "deadtime_clipped_mask": deadtime_clipped_mask_shift,
        "deadtime_raw_clipped_mask": raw_deadtime_clipped_mask_shift,
        "pc_saturation_mask": pc_saturation_mask_shift,
        "deadtime_clipping_fraction": _fraction_over_range(deadtime_clipped_mask_shift),
        "deadtime_raw_clipping_fraction": _fraction_over_range(raw_deadtime_clipped_mask_shift),
        "pc_saturation_fraction": _fraction_over_range(pc_saturation_mask_shift),
        "deadtime_min_denominator_observed": deadtime_denominator_min,
        "deadtime_raw_min_denominator_observed": raw_deadtime_denominator_min,
        "deadtime_min_denominator_allowed": deadtime_min_denominator,
        "deadtime_correction_applied": bool(is_photon and deadtime > 0.0),
        "pc_observed_rate_mhz_max": pc_observed_rate_mhz_max,
        "pc_saturation_characterized": bool(pc_saturation_characterized),
        "pc_saturation_rate_limit_mhz": pc_saturation_rate_limit_mhz,
        "bin_shift_bins": shift,
        "bin_shift_invalid_mask": bin_shift_invalid_mask,
        "bin_shift_invalid_fraction": _fraction_over_range(bin_shift_invalid_mask),
        "signal_pre_background": sig_shift,
        "signal_pre_background_error": err_shift,
        "background_estimate": bg_location,
        "background_standard_error": err_bg_location,
        "background_robust_scale": background["scale"],
        "background_valid_bins": background["valid_bins"],
        "background_outlier_fraction": background["outlier_fraction"],
        "background_estimator": "median_mad",
    }
    return sig_c, err_c, rcs, err_rcs, diagnostics
