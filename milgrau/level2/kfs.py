"""Klett--Fernald--Sasano elastic inversion and partial Monte Carlo spread.

For range-corrected signal ``X(z)``, total backscatter ``beta = beta_m +
beta_a``, molecular lidar ratio ``S_m`` and aerosol lidar ratio ``S_a(z)``,
the elastic lidar equation is

``X(z) = C beta(z) exp[-2 integral(z0, z) (S_m beta_m + S_a beta_a) ds]``.

With the signed integrals

``M(z) = integral(z_ref, z) (S_a(s) - S_m) beta_m(s) ds`` and
``Y(z) = X(z) exp[-2 M(z)]``, the generalized Fernald solution is

``beta(z) = Y(z) / [X_ref / beta_ref - 2 integral(z_ref, z) S_a(s) Y(s) ds]``.

Altitude is strictly increasing in meters.  Thus both integrals are negative
below the reference and positive above it.  Equivalently, the backward
molecular factor is ``exp(+2 integral(z, z_ref) (S_a-S_m) beta_m ds)``;
this positive sign is the SCI-001 correction.  ``beta_ref`` is total
backscatter at exactly one reference bin.  The reference bin is written once
and shared by the backward and forward branches.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numba import njit, prange

from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR

IntegrationMode = Literal["backward", "forward", "two_sided"]
_MODE_BACKWARD = 0
_MODE_FORWARD = 1
_MODE_TWO_SIDED = 2


def _prepare_altitude_m(altitude: np.ndarray, altitude_units: Literal["auto", "m", "km"]) -> np.ndarray:
    """Return altitude in meters."""
    altitude_arr = np.ascontiguousarray(altitude, dtype=np.float64)
    if altitude_units == "auto":
        return altitude_arr * 1000.0 if np.nanmax(altitude_arr) <= 100.0 else altitude_arr.copy()
    if altitude_units == "km":
        return altitude_arr * 1000.0
    if altitude_units == "m":
        return altitude_arr.copy()
    raise ValueError("altitude_units must be 'auto', 'm', or 'km'.")


def _mode_code(mode: IntegrationMode) -> int:
    """Return the compiled-kernel code for one explicit integration mode."""
    if mode == "backward":
        return _MODE_BACKWARD
    if mode == "forward":
        return _MODE_FORWARD
    if mode == "two_sided":
        return _MODE_TWO_SIDED
    raise ValueError("mode must be 'backward', 'forward', or 'two_sided'.")


def _validate_reference_domain(ref_idx: int, n_bins: int, mode: IntegrationMode) -> None:
    """Require at least one altitude interval on every requested side."""
    if mode in {"backward", "two_sided"} and ref_idx == 0:
        raise ValueError("Backward integration requires a reference bin above the grid bottom.")
    if mode in {"forward", "two_sided"} and ref_idx == n_bins - 1:
        raise ValueError("Forward integration requires a reference bin below the grid top.")


def _prepare_lidar_ratio(
    lidar_ratio_aerosol: float | np.ndarray,
    shape: tuple[int, ...],
    min_lidar_ratio: float,
) -> np.ndarray:
    """Broadcast, validate and apply the existing lower lidar-ratio bound."""
    if not np.isfinite(min_lidar_ratio) or min_lidar_ratio <= 0.0:
        raise ValueError("min_lidar_ratio must be finite and positive.")
    raw = np.asarray(lidar_ratio_aerosol, dtype=np.float64)
    if raw.ndim > 1 or (raw.ndim == 1 and raw.shape != shape):
        raise ValueError("lidar_ratio_aerosol must be scalar or have the same shape as rcs.")
    try:
        profile = np.broadcast_to(raw, shape)
    except ValueError as exc:
        raise ValueError("lidar_ratio_aerosol must be scalar or have the same shape as rcs.") from exc
    if np.any(~np.isfinite(profile)) or np.any(profile <= 0.0):
        raise ValueError("lidar_ratio_aerosol must contain finite positive values.")
    return np.ascontiguousarray(np.maximum(profile, float(min_lidar_ratio)), dtype=np.float64)


def _validate_profile_inputs(
    rcs: np.ndarray,
    altitude_m: np.ndarray,
    beta_mol: np.ndarray,
    beta_total_ref: float,
    ref_idx: int,
    lr_mol: float,
) -> int:
    """Validate the grid and the single physical boundary condition."""
    if rcs.ndim != 1 or altitude_m.ndim != 1 or beta_mol.ndim != 1:
        raise ValueError("rcs, altitude and beta_mol must be 1D arrays.")
    if not (rcs.shape == altitude_m.shape == beta_mol.shape):
        raise ValueError("rcs, altitude and beta_mol must have the same shape.")
    if rcs.size < 3:
        raise ValueError("At least three altitude bins are required.")
    if np.any(~np.isfinite(altitude_m)) or np.any(np.diff(altitude_m) <= 0.0):
        raise ValueError("altitude must be finite and strictly increasing.")
    if ref_idx < 0:
        ref_idx += rcs.size
    ref_idx = int(ref_idx)
    if ref_idx < 0 or ref_idx >= rcs.size:
        raise ValueError("ref_idx must point to a bin inside the altitude grid.")
    if not np.isfinite(rcs[ref_idx]) or rcs[ref_idx] <= 0.0:
        raise ValueError("The range-corrected signal must be finite and positive at ref_idx.")
    if not np.isfinite(beta_mol[ref_idx]) or beta_mol[ref_idx] <= 0.0:
        raise ValueError("Molecular backscatter must be finite and positive at ref_idx.")
    if not np.isfinite(beta_total_ref) or beta_total_ref <= 0.0:
        raise ValueError("beta_total_ref must be a finite positive total-backscatter boundary.")
    if not np.isfinite(lr_mol) or lr_mol <= 0.0:
        raise ValueError("lr_mol must be finite and positive.")
    return ref_idx


def _nanmean_nanstd_no_warning(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return NaN-safe mean and standard deviation without empty-slice warnings.

    ``np.nanmean`` and ``np.nanstd`` intentionally warn when a full reduction
    slice is NaN. In KFS this is an expected scientific outcome for altitude
    bins where the inversion is invalid, so we return NaN quietly for those
    bins and keep finite statistics elsewhere.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    count = finite.sum(axis=axis)
    safe_values = np.where(finite, arr, 0.0)
    summed = safe_values.sum(axis=axis)

    mean = np.full(count.shape, np.nan, dtype=np.float64)
    valid = count > 0
    mean[valid] = summed[valid] / count[valid]

    expanded_mean = np.expand_dims(mean, axis=axis)
    squared = np.where(finite, (arr - expanded_mean) ** 2, 0.0)
    variance_sum = squared.sum(axis=axis)
    std = np.full(count.shape, np.nan, dtype=np.float64)
    std[valid] = np.sqrt(variance_sum[valid] / count[valid])
    return mean, std


@njit
def _fernald_single_profile(
    rcs: np.ndarray,
    altitude_m: np.ndarray,
    beta_mol: np.ndarray,
    lr_aer: np.ndarray,
    beta_total_ref: float,
    ref_idx: int,
    lr_mol: float,
    allow_negative_aerosol: bool,
    mode_code: int,
) -> np.ndarray:
    """Compiled generalized Fernald solution on an already validated grid."""
    n_bins = rcs.shape[0]
    beta_aer = np.empty(n_bins, dtype=np.float64)
    for j in range(n_bins):
        beta_aer[j] = np.nan

    beta_mol_ref = beta_mol[ref_idx]
    if (not np.isfinite(beta_total_ref)) or beta_total_ref <= 0.0:
        return beta_aer

    x_ref = rcs[ref_idx]
    if (not np.isfinite(x_ref)) or x_ref <= 0.0:
        return beta_aer

    beta_aer_ref = beta_total_ref - beta_mol_ref
    if (not allow_negative_aerosol) and beta_aer_ref < 0.0:
        beta_aer_ref = 0.0
        beta_total_ref = beta_mol_ref
    beta_aer[ref_idx] = beta_aer_ref

    denom0 = x_ref / beta_total_ref

    if mode_code == _MODE_BACKWARD or mode_code == _MODE_TWO_SIDED:
        # For z < z_ref, reverse_integral = integral(z, z_ref) is positive.
        # Since M(z)=integral(z_ref,z), Y uses exp(+2*reverse_integral).
        molecular_reverse_integral = 0.0
        denominator_reverse_integral = 0.0
        y_prev = x_ref
        for j in range(ref_idx - 1, -1, -1):
            dz = altitude_m[j + 1] - altitude_m[j]
            x_j = rcs[j]
            bm0 = beta_mol[j]
            bm1 = beta_mol[j + 1]
            if (
                (not np.isfinite(dz))
                or dz <= 0.0
                or (not np.isfinite(x_j))
                or x_j <= 0.0
                or (not np.isfinite(bm0))
                or (not np.isfinite(bm1))
                or bm0 <= 0.0
                or bm1 <= 0.0
            ):
                break
            molecular_reverse_integral += 0.5 * (
                (lr_aer[j] - lr_mol) * bm0
                + (lr_aer[j + 1] - lr_mol) * bm1
            ) * dz
            exponent = 2.0 * molecular_reverse_integral
            if exponent > 700.0:
                exponent = 700.0
            elif exponent < -700.0:
                exponent = -700.0
            y_j = x_j * np.exp(exponent)
            denominator_reverse_integral += 0.5 * (
                lr_aer[j] * y_j + lr_aer[j + 1] * y_prev
            ) * dz
            denom = denom0 + 2.0 * denominator_reverse_integral
            if (not np.isfinite(denom)) or denom <= 0.0:
                break
            beta_aer_j = y_j / denom - bm0
            if (not allow_negative_aerosol) and beta_aer_j < 0.0:
                beta_aer_j = 0.0
            beta_aer[j] = beta_aer_j
            y_prev = y_j

    if mode_code == _MODE_FORWARD or mode_code == _MODE_TWO_SIDED:
        # For z > z_ref both signed integrals follow increasing altitude.
        molecular_forward_integral = 0.0
        denominator_forward_integral = 0.0
        y_prev = x_ref
        for j in range(ref_idx + 1, n_bins):
            dz = altitude_m[j] - altitude_m[j - 1]
            x_j = rcs[j]
            bm0 = beta_mol[j]
            bm1 = beta_mol[j - 1]
            if (
                (not np.isfinite(dz))
                or dz <= 0.0
                or (not np.isfinite(x_j))
                or x_j <= 0.0
                or (not np.isfinite(bm0))
                or (not np.isfinite(bm1))
                or bm0 <= 0.0
                or bm1 <= 0.0
            ):
                break
            molecular_forward_integral += 0.5 * (
                (lr_aer[j - 1] - lr_mol) * bm1
                + (lr_aer[j] - lr_mol) * bm0
            ) * dz
            exponent = -2.0 * molecular_forward_integral
            if exponent > 700.0:
                exponent = 700.0
            elif exponent < -700.0:
                exponent = -700.0
            y_j = x_j * np.exp(exponent)
            denominator_forward_integral += 0.5 * (
                lr_aer[j - 1] * y_prev + lr_aer[j] * y_j
            ) * dz
            denom = denom0 - 2.0 * denominator_forward_integral
            if (not np.isfinite(denom)) or denom <= 0.0:
                break
            beta_aer_j = y_j / denom - bm0
            if (not allow_negative_aerosol) and beta_aer_j < 0.0:
                beta_aer_j = 0.0
            beta_aer[j] = beta_aer_j
            y_prev = y_j

    return beta_aer


def _branch_validity(
    beta_aer: np.ndarray,
    rcs: np.ndarray,
    beta_mol: np.ndarray,
    ref_idx: int,
    mode: IntegrationMode,
) -> dict[str, bool]:
    """Report complete physically sampled branches without edge-bin substitution."""
    backward_requested = mode in {"backward", "two_sided"}
    forward_requested = mode in {"forward", "two_sided"}
    sampled = np.isfinite(rcs) & (rcs > 0.0) & np.isfinite(beta_mol) & (beta_mol > 0.0)
    backward_sampled = sampled[: ref_idx + 1]
    forward_sampled = sampled[ref_idx:]
    return {
        "backward_requested": backward_requested,
        "forward_requested": forward_requested,
        "backward_valid": bool(
            backward_requested
            and np.any(backward_sampled)
            and np.all(np.isfinite(beta_aer[: ref_idx + 1][backward_sampled]))
        ),
        "forward_valid": bool(
            forward_requested
            and np.any(forward_sampled)
            and np.all(np.isfinite(beta_aer[ref_idx:][forward_sampled]))
        ),
        "reference_valid": bool(np.isfinite(beta_aer[ref_idx])),
    }


def fernald_inversion(
    rcs: np.ndarray,
    altitude: np.ndarray,
    beta_mol: np.ndarray,
    lidar_ratio_aerosol: float | np.ndarray,
    beta_total_ref: float,
    ref_idx: int,
    *,
    lr_mol: float = RAYLEIGH_LIDAR_RATIO_SR,
    altitude_units: Literal["auto", "m", "km"] = "auto",
    min_lidar_ratio: float = 10.0,
    allow_negative_aerosol: bool = False,
    mode: IntegrationMode = "two_sided",
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Invert one elastic RCS profile with an exact total-backscatter boundary.

    ``beta_total_ref`` is ``beta_mol + beta_aer`` at exactly ``ref_idx``.  If a
    scattering ratio is used, ``beta_total_ref = SR_ref * beta_mol[ref_idx]``;
    equivalently the configured residual aerosol fraction is ``SR_ref - 1``.
    Invalid bins or a nonphysical denominator terminate only the affected
    oriented branch, whose remaining bins stay NaN and whose validity is false.
    """
    rcs_arr = np.ascontiguousarray(rcs, dtype=np.float64)
    beta_mol_arr = np.ascontiguousarray(beta_mol, dtype=np.float64)
    altitude_m = np.ascontiguousarray(_prepare_altitude_m(altitude, altitude_units), dtype=np.float64)
    ref_idx = _validate_profile_inputs(
        rcs_arr,
        altitude_m,
        beta_mol_arr,
        float(beta_total_ref),
        int(ref_idx),
        float(lr_mol),
    )
    lidar_ratio_profile = _prepare_lidar_ratio(lidar_ratio_aerosol, rcs_arr.shape, min_lidar_ratio)
    mode_code = _mode_code(mode)
    _validate_reference_domain(ref_idx, rcs_arr.size, mode)
    beta_aer = _fernald_single_profile(
        rcs_arr,
        altitude_m,
        beta_mol_arr,
        lidar_ratio_profile,
        float(beta_total_ref),
        ref_idx,
        float(lr_mol),
        bool(allow_negative_aerosol),
        mode_code,
    )
    if not return_diagnostics:
        return beta_aer
    beta_total_ref_applied = (
        max(float(beta_total_ref), float(beta_mol_arr[ref_idx]))
        if not allow_negative_aerosol
        else float(beta_total_ref)
    )
    diagnostics: dict[str, Any] = {
        **_branch_validity(beta_aer, rcs_arr, beta_mol_arr, ref_idx, mode),
        "mode": mode,
        "ref_idx": ref_idx,
        "beta_total_ref_input": float(beta_total_ref),
        "beta_total_ref": beta_total_ref_applied,
        "beta_aerosol_ref": float(beta_aer[ref_idx]),
        "altitude_m": altitude_m,
        "lidar_ratio_aerosol_sr": lidar_ratio_profile,
    }
    return beta_aer, diagnostics


@njit(parallel=True)
def _kfs_fernald_mc_core(
    rcs: np.ndarray,
    rcs_error: np.ndarray,
    altitude_m: np.ndarray,
    beta_mol: np.ndarray,
    lr_samples: np.ndarray,
    beta_total_ref_samples: np.ndarray,
    rcs_noise: np.ndarray,
    ref_idx: int,
    lr_mol: float,
    use_rcs_noise: bool,
    allow_negative_aerosol: bool,
    mode_code: int,
):
    """Numba-compiled Monte Carlo Fernald/Klett-Sasano inversion."""
    n_iter = lr_samples.shape[0]
    n_bins = rcs.shape[0]
    beta_aer_sims = np.empty((n_iter, n_bins), dtype=np.float64)
    alpha_aer_sims = np.empty((n_iter, n_bins), dtype=np.float64)

    for i in prange(n_iter):
        rcs_i = np.empty(n_bins, dtype=np.float64)
        for j in range(n_bins):
            beta_aer_sims[i, j] = np.nan
            alpha_aer_sims[i, j] = np.nan
            if use_rcs_noise:
                rcs_i[j] = rcs[j] + rcs_error[j] * rcs_noise[i, j]
            else:
                rcs_i[j] = rcs[j]

        lr_aer = lr_samples[i]
        if not np.isfinite(lr_aer):
            continue
        lr_profile = np.empty(n_bins, dtype=np.float64)
        for j in range(n_bins):
            lr_profile[j] = lr_aer

        beta_aer = _fernald_single_profile(
            rcs_i,
            altitude_m,
            beta_mol,
            lr_profile,
            beta_total_ref_samples[i],
            ref_idx,
            lr_mol,
            allow_negative_aerosol,
            mode_code,
        )
        for j in range(n_bins):
            beta_aer_sims[i, j] = beta_aer[j]
            if np.isfinite(beta_aer[j]):
                alpha_aer_sims[i, j] = beta_aer[j] * lr_profile[j]

    return beta_aer_sims, alpha_aer_sims


def kfs_inversion_monte_carlo(
    rcs: np.ndarray,
    altitude: np.ndarray,
    beta_mol: np.ndarray,
    lr_base: float,
    lr_std: float = 10.0,
    ref_idx: int = -1,
    n_iterations: int = 300,
    rcs_error: np.ndarray | None = None,
    beta_ref_relative_std: float = 0.10,
    aerosol_ref_fraction: float = 0.0,
    altitude_units: Literal["auto", "m", "km"] = "auto",
    min_lidar_ratio: float = 10.0,
    allow_negative_aerosol: bool = False,
    seed: int | None = None,
    return_diagnostics: bool = False,
    mode: IntegrationMode = "two_sided",
):
    """Run KFS/Fernald-Sasano inversion with Monte Carlo uncertainty.

    The returned standard deviations are the spread of the existing partial
    Monte Carlo ensemble (signal noise, scalar lidar ratio and reference
    boundary perturbations); they are not a total uncertainty budget.
    """
    rcs_arr = np.ascontiguousarray(rcs, dtype=np.float64)
    beta_mol_arr = np.ascontiguousarray(beta_mol, dtype=np.float64)
    altitude_m = np.ascontiguousarray(_prepare_altitude_m(altitude, altitude_units), dtype=np.float64)

    if rcs_arr.ndim != 1 or beta_mol_arr.ndim != 1 or altitude_m.ndim != 1:
        raise ValueError("rcs, altitude and beta_mol must be 1D arrays.")
    if not (rcs_arr.shape == beta_mol_arr.shape == altitude_m.shape):
        raise ValueError("rcs, altitude and beta_mol must have the same shape.")
    n_bins = rcs_arr.shape[0]
    if ref_idx < 0:
        ref_idx += n_bins
    if int(ref_idx) < 0 or int(ref_idx) >= n_bins:
        raise ValueError("ref_idx must point to a bin inside the altitude grid.")
    beta_total_ref_mean = beta_mol_arr[int(ref_idx)] * (1.0 + float(aerosol_ref_fraction))
    ref_idx = _validate_profile_inputs(
        rcs_arr,
        altitude_m,
        beta_mol_arr,
        float(beta_total_ref_mean),
        int(ref_idx),
        RAYLEIGH_LIDAR_RATIO_SR,
    )
    mode_code = _mode_code(mode)
    _validate_reference_domain(ref_idx, n_bins, mode)
    if not np.isfinite(lr_base) or lr_base <= 0.0:
        raise ValueError("lr_base must be finite and positive.")
    if not np.isfinite(lr_std) or lr_std < 0.0:
        raise ValueError("lr_std must be finite and nonnegative.")
    if not np.isfinite(beta_ref_relative_std) or beta_ref_relative_std < 0.0:
        raise ValueError("beta_ref_relative_std must be finite and nonnegative.")
    if not np.isfinite(aerosol_ref_fraction) or aerosol_ref_fraction < 0.0:
        raise ValueError("aerosol_ref_fraction = SR_ref - 1 must be finite and nonnegative.")
    if not np.isfinite(min_lidar_ratio) or min_lidar_ratio <= 0.0:
        raise ValueError("min_lidar_ratio must be finite and positive.")

    use_rcs_noise = rcs_error is not None
    if use_rcs_noise:
        rcs_error_arr = np.ascontiguousarray(rcs_error, dtype=np.float64)
        if rcs_error_arr.shape != rcs_arr.shape:
            raise ValueError("rcs_error must have the same shape as rcs.")
        if np.any(np.isfinite(rcs_error_arr) & (rcs_error_arr < 0.0)):
            raise ValueError("rcs_error must contain nonnegative one-sigma values.")
        rcs_error_arr = np.ascontiguousarray(np.where(np.isfinite(rcs_error_arr), rcs_error_arr, 0.0), dtype=np.float64)
    else:
        rcs_error_arr = np.zeros_like(rcs_arr, dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_iterations = int(n_iterations)
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive.")
    lr_samples = np.ascontiguousarray(rng.normal(float(lr_base), float(lr_std), size=n_iterations), dtype=np.float64)
    lr_samples = np.ascontiguousarray(np.maximum(lr_samples, float(min_lidar_ratio)), dtype=np.float64)

    beta_total_ref_samples = np.ascontiguousarray(
        rng.normal(
            beta_total_ref_mean,
            abs(beta_total_ref_mean) * float(beta_ref_relative_std),
            size=n_iterations,
        ),
        dtype=np.float64,
    )

    if use_rcs_noise:
        rcs_noise = np.ascontiguousarray(rng.standard_normal((n_iterations, n_bins)), dtype=np.float64)
    else:
        rcs_noise = np.empty((1, 1), dtype=np.float64)

    beta_sims, alpha_sims = _kfs_fernald_mc_core(
        rcs_arr,
        rcs_error_arr,
        altitude_m,
        beta_mol_arr,
        lr_samples,
        beta_total_ref_samples,
        rcs_noise,
        ref_idx,
        RAYLEIGH_LIDAR_RATIO_SR,
        bool(use_rcs_noise),
        bool(allow_negative_aerosol),
        mode_code,
    )

    beta_mean, beta_std = _nanmean_nanstd_no_warning(beta_sims, axis=0)
    alpha_mean, alpha_std = _nanmean_nanstd_no_warning(alpha_sims, axis=0)

    if return_diagnostics:
        backward_requested = mode in {"backward", "two_sided"}
        forward_requested = mode in {"forward", "two_sided"}
        sampled = (
            np.isfinite(rcs_arr)
            & (rcs_arr > 0.0)
            & np.isfinite(beta_mol_arr)
            & (beta_mol_arr > 0.0)
        )
        backward_sampled = sampled[: ref_idx + 1]
        forward_sampled = sampled[ref_idx:]
        backward_valid_simulations = (
            np.all(np.isfinite(beta_sims[:, : ref_idx + 1][:, backward_sampled]), axis=1)
            if backward_requested
            else np.zeros(n_iterations, dtype=bool)
        )
        forward_valid_simulations = (
            np.all(np.isfinite(beta_sims[:, ref_idx:][:, forward_sampled]), axis=1)
            if forward_requested
            else np.zeros(n_iterations, dtype=bool)
        )
        diagnostics = {
            "beta_aer_sims": beta_sims,
            "alpha_aer_sims": alpha_sims,
            "lr_samples": lr_samples,
            "beta_total_ref_samples": beta_total_ref_samples,
            "ref_idx": int(ref_idx),
            "altitude_m": altitude_m,
            "used_rcs_noise": bool(use_rcs_noise),
            "mode": mode,
            "backward_requested": backward_requested,
            "forward_requested": forward_requested,
            "backward_valid_simulations": backward_valid_simulations,
            "forward_valid_simulations": forward_valid_simulations,
            "backward_valid": bool(backward_requested and np.all(backward_valid_simulations)),
            "forward_valid": bool(forward_requested and np.all(forward_valid_simulations)),
            "uncertainty_scope": "partial_monte_carlo_dispersion",
        }
        return beta_mean, beta_std, alpha_mean, alpha_std, diagnostics
    return beta_mean, beta_std, alpha_mean, alpha_std
