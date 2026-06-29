"""Rayleigh molecular calibration fit utilities.

This module contains small, testable estimators for the Level-2 Rayleigh
normalization problem

    measured_rcs(z) ~= C * molecular_rcs(z)

inside a selected molecular reference window.  The production default should
remain a through-origin multiplicative fit; a free intercept is exposed only as
a diagnostic for residual background or baseline offsets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RayleighFitResult:
    """Structured diagnostics for one Rayleigh calibration window."""

    success: bool
    scale_factor: float
    intercept_diagnostic: float
    scale_factor_std: float
    reduced_chi2: float
    residual_slope_per_km: float
    valid_fraction: float
    window_start_idx: int
    window_stop_idx: int
    valid_points: int
    method: str
    reason: str


def _window_bounds(center_idx: int, window_bins: int, size: int) -> tuple[int, int]:
    """Return inclusive-start/exclusive-stop bounds around a reference bin."""
    if size <= 0:
        return 0, 0
    center = int(np.clip(int(center_idx), 0, size - 1))
    half_window = max(int(window_bins) // 2, 1)
    start = max(center - half_window, 0)
    stop = min(center + half_window + 1, size)
    return start, stop


def _empty_result(method: str, start: int, stop: int, valid_fraction: float, valid_points: int, reason: str) -> RayleighFitResult:
    """Return a failed fit result with a consistent payload."""
    return RayleighFitResult(
        success=False,
        scale_factor=np.nan,
        intercept_diagnostic=np.nan,
        scale_factor_std=np.nan,
        reduced_chi2=np.nan,
        residual_slope_per_km=np.nan,
        valid_fraction=float(valid_fraction),
        window_start_idx=int(start),
        window_stop_idx=int(stop),
        valid_points=int(valid_points),
        method=method,
        reason=reason,
    )


def _as_float_array(values: np.ndarray | None) -> np.ndarray | None:
    """Return values as float64 ndarray, preserving None."""
    if values is None:
        return None
    return np.asarray(values, dtype=np.float64)


def _window_data(
    measured_signal: np.ndarray,
    molecular_signal: np.ndarray,
    altitude_m: np.ndarray,
    reference_center_idx: int,
    reference_window_bins: int,
    sigma_measured: np.ndarray | None,
    valid_mask: np.ndarray | None,
    method: str,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, float, int]:
    """Extract finite, positive fit vectors from the selected window."""
    y_all = np.asarray(measured_signal, dtype=np.float64)
    x_all = np.asarray(molecular_signal, dtype=np.float64)
    z_all = np.asarray(altitude_m, dtype=np.float64)
    sigma_all = _as_float_array(sigma_measured)
    mask_all = None if valid_mask is None else np.asarray(valid_mask, dtype=bool)

    if y_all.shape != x_all.shape or y_all.shape != z_all.shape:
        raise ValueError("measured_signal, molecular_signal and altitude_m must have the same shape")
    if sigma_all is not None and sigma_all.shape != y_all.shape:
        raise ValueError("sigma_measured must have the same shape as measured_signal")
    if mask_all is not None and mask_all.shape != y_all.shape:
        raise ValueError("valid_mask must have the same shape as measured_signal")

    start, stop = _window_bounds(reference_center_idx, reference_window_bins, y_all.size)
    window_size = max(stop - start, 1)
    y = y_all[start:stop]
    x = x_all[start:stop]
    z = z_all[start:stop]
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x > 0.0) & (y > 0.0)

    sigma = None
    if sigma_all is not None:
        sigma_window = sigma_all[start:stop]
        valid &= np.isfinite(sigma_window) & (sigma_window > 0.0)
        sigma = sigma_window[valid]

    if mask_all is not None:
        valid &= mask_all[start:stop]
        if sigma_all is not None:
            sigma = sigma_all[start:stop][valid]

    valid_count = int(valid.sum())
    valid_fraction = float(valid_count / window_size)
    return start, stop, x[valid], y[valid], z[valid], sigma, valid_fraction, valid_count


def _weights_from_sigma(sigma: np.ndarray | None, size: int) -> tuple[np.ndarray, bool]:
    """Return inverse-variance weights and whether absolute sigmas were provided."""
    if sigma is None:
        return np.ones(size, dtype=np.float64), False
    sigma = np.asarray(sigma, dtype=np.float64)
    weights = np.divide(1.0, sigma**2, out=np.zeros_like(sigma), where=np.isfinite(sigma) & (sigma > 0.0))
    return weights, True


def _diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sigma: np.ndarray | None,
    weights: np.ndarray,
    scale_factor: float,
    n_parameters: int,
) -> tuple[float, float, float]:
    """Return scale std, reduced chi-square and normalized residual slope."""
    residual = y - scale_factor * x
    dof = max(int(y.size) - int(n_parameters), 1)
    den = float(np.nansum(weights * x**2))

    if sigma is not None:
        reduced_chi2 = float(np.nansum((residual / sigma) ** 2) / dof)
        variance_scale = max(reduced_chi2, 0.0)
    else:
        reduced_chi2 = np.nan
        variance_scale = float(np.nansum(residual**2) / dof)

    scale_factor_std = float(np.sqrt(variance_scale / den)) if den > 0.0 and np.isfinite(variance_scale) else np.nan

    normalized_residual = np.divide(y, scale_factor * x, out=np.full_like(y, np.nan), where=np.isfinite(scale_factor * x) & (scale_factor * x > 0.0)) - 1.0
    valid_trend = np.isfinite(normalized_residual) & np.isfinite(z)
    if valid_trend.sum() >= 3:
        residual_slope_per_km = float(np.polyfit(z[valid_trend] / 1000.0, normalized_residual[valid_trend], 1)[0])
    else:
        residual_slope_per_km = np.nan

    return scale_factor_std, reduced_chi2, residual_slope_per_km


def _fit_zero(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sigma: np.ndarray | None,
    start: int,
    stop: int,
    valid_fraction: float,
    method: str,
    weights: np.ndarray | None = None,
) -> RayleighFitResult:
    """Fit measured ~= C * molecular with the intercept constrained to zero."""
    if x.size < 2:
        return _empty_result(method, start, stop, valid_fraction, int(x.size), "fewer than two valid bins")

    base_weights, _ = _weights_from_sigma(sigma, x.size)
    final_weights = base_weights if weights is None else np.asarray(weights, dtype=np.float64)
    denominator = float(np.nansum(final_weights * x**2))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return _empty_result(method, start, stop, valid_fraction, int(x.size), "non-positive weighted molecular energy")

    scale_factor = float(np.nansum(final_weights * x * y) / denominator)
    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        return _empty_result(method, start, stop, valid_fraction, int(x.size), "non-positive scale factor")

    scale_factor_std, reduced_chi2, residual_slope_per_km = _diagnostics(
        x=x,
        y=y,
        z=z,
        sigma=sigma,
        weights=final_weights,
        scale_factor=scale_factor,
        n_parameters=1,
    )
    return RayleighFitResult(
        success=True,
        scale_factor=scale_factor,
        intercept_diagnostic=0.0,
        scale_factor_std=scale_factor_std,
        reduced_chi2=reduced_chi2,
        residual_slope_per_km=residual_slope_per_km,
        valid_fraction=float(valid_fraction),
        window_start_idx=int(start),
        window_stop_idx=int(stop),
        valid_points=int(x.size),
        method=method,
        reason="ok",
    )


def _fit_free_intercept(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sigma: np.ndarray | None,
    start: int,
    stop: int,
    valid_fraction: float,
    method: str,
) -> RayleighFitResult:
    """Fit measured ~= C * molecular + b for background diagnostics."""
    if x.size < 3:
        return _empty_result(method, start, stop, valid_fraction, int(x.size), "fewer than three valid bins")

    weights, _ = _weights_from_sigma(sigma, x.size)
    design = np.column_stack([x, np.ones_like(x)])
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_y = y * np.sqrt(weights)
    try:
        scale_factor, intercept = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return _empty_result(method, start, stop, valid_fraction, int(x.size), "singular weighted design matrix")

    scale_factor = float(scale_factor)
    intercept = float(intercept)
    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        return _empty_result(method, start, stop, valid_fraction, int(x.size), "non-positive scale factor")

    scale_factor_std, reduced_chi2, residual_slope_per_km = _diagnostics(
        x=x,
        y=y - intercept,
        z=z,
        sigma=sigma,
        weights=weights,
        scale_factor=scale_factor,
        n_parameters=2,
    )
    return RayleighFitResult(
        success=True,
        scale_factor=scale_factor,
        intercept_diagnostic=intercept,
        scale_factor_std=scale_factor_std,
        reduced_chi2=reduced_chi2,
        residual_slope_per_km=residual_slope_per_km,
        valid_fraction=float(valid_fraction),
        window_start_idx=int(start),
        window_stop_idx=int(stop),
        valid_points=int(x.size),
        method=method,
        reason="ok",
    )


def _fit_huber_zero(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sigma: np.ndarray | None,
    start: int,
    stop: int,
    valid_fraction: float,
    method: str,
    huber_delta: float,
    max_iterations: int,
    tolerance: float,
) -> RayleighFitResult:
    """Fit through-origin scale with weighted Huber IRLS."""
    initial = _fit_zero(x, y, z, sigma, start, stop, valid_fraction, "weighted_ls_zero")
    if not initial.success:
        return _empty_result(method, start, stop, valid_fraction, int(x.size), initial.reason)

    base_weights, has_sigma = _weights_from_sigma(sigma, x.size)
    scale_factor = float(initial.scale_factor)
    robust_delta = max(float(huber_delta), 1.0e-6)
    max_iterations = max(int(max_iterations), 1)
    tolerance = max(float(tolerance), 0.0)

    final_weights = base_weights.copy()
    for _ in range(max_iterations):
        residual = y - scale_factor * x
        if has_sigma:
            normalized = residual / sigma
        else:
            median = float(np.nanmedian(residual))
            mad = float(1.4826 * np.nanmedian(np.abs(residual - median)))
            robust_scale = mad if np.isfinite(mad) and mad > 0.0 else float(np.nanstd(residual))
            if not np.isfinite(robust_scale) or robust_scale <= 0.0:
                break
            normalized = residual / robust_scale

        abs_normalized = np.abs(normalized)
        huber_weights = np.ones_like(abs_normalized)
        outlier = abs_normalized > robust_delta
        huber_weights[outlier] = robust_delta / abs_normalized[outlier]
        final_weights = base_weights * huber_weights

        denominator = float(np.nansum(final_weights * x**2))
        if not np.isfinite(denominator) or denominator <= 0.0:
            return _empty_result(method, start, stop, valid_fraction, int(x.size), "non-positive Huber weighted molecular energy")
        updated = float(np.nansum(final_weights * x * y) / denominator)
        if not np.isfinite(updated) or updated <= 0.0:
            return _empty_result(method, start, stop, valid_fraction, int(x.size), "non-positive Huber scale factor")
        if abs(updated - scale_factor) <= tolerance * max(abs(scale_factor), 1.0):
            scale_factor = updated
            break
        scale_factor = updated

    result = _fit_zero(x, y, z, sigma, start, stop, valid_fraction, method, weights=final_weights)
    return result


def fit_rayleigh_scale(
    measured_signal: np.ndarray,
    molecular_signal: np.ndarray,
    altitude_m: np.ndarray,
    reference_center_idx: int,
    reference_window_bins: int,
    sigma_measured: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    method: str = "weighted_ls_zero",
    huber_delta: float = 1.345,
    huber_max_iterations: int = 20,
    huber_tolerance: float = 1.0e-6,
) -> RayleighFitResult:
    """Fit a Rayleigh calibration factor in a selected reference window.

    Parameters
    ----------
    measured_signal:
        Background-corrected measured RCS-like signal.
    molecular_signal:
        Shape-only molecular RCS-like signal.  It must not contain the unknown
        instrumental calibration constant.
    altitude_m:
        Altitude coordinate in metres.
    reference_center_idx, reference_window_bins:
        Window definition around the molecular reference region.
    sigma_measured:
        Optional one-sigma uncertainty of ``measured_signal``.  When supplied,
        inverse-variance weights are used and ``reduced_chi2`` becomes a true
        pull-based diagnostic.
    valid_mask:
        Optional boolean mask.  False bins are excluded from the fit.
    method:
        ``weighted_ls_zero`` for the production multiplicative fit,
        ``weighted_huber_zero`` for robust IRLS, or
        ``weighted_ls_free_intercept_diagnostic`` for background diagnostics.
    """
    method = str(method).strip().lower()
    start, stop, x, y, z, sigma, valid_fraction, valid_count = _window_data(
        measured_signal=measured_signal,
        molecular_signal=molecular_signal,
        altitude_m=altitude_m,
        reference_center_idx=reference_center_idx,
        reference_window_bins=reference_window_bins,
        sigma_measured=sigma_measured,
        valid_mask=valid_mask,
        method=method,
    )

    if method in {"weighted_ls_zero", "wls_zero", "ols_zero"}:
        return _fit_zero(x, y, z, sigma, start, stop, valid_fraction, "weighted_ls_zero")
    if method in {"weighted_huber_zero", "huber_zero"}:
        return _fit_huber_zero(
            x=x,
            y=y,
            z=z,
            sigma=sigma,
            start=start,
            stop=stop,
            valid_fraction=valid_fraction,
            method="weighted_huber_zero",
            huber_delta=huber_delta,
            max_iterations=huber_max_iterations,
            tolerance=huber_tolerance,
        )
    if method in {"weighted_ls_free_intercept_diagnostic", "free_intercept_diagnostic", "wls_free_intercept"}:
        return _fit_free_intercept(x, y, z, sigma, start, stop, valid_fraction, "weighted_ls_free_intercept_diagnostic")

    return _empty_result(method, start, stop, valid_fraction, valid_count, f"unknown fit method: {method}")
