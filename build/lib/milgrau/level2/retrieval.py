"""Core Level 2 retrieval routines for one wavelength."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import xarray as xr

from milgrau.level2.block_average import (
    block_groups,
    error_by_groups,
    error_of_mean,
    expand_block_vector_to_time,
    expand_blocks_to_time,
    mask_by_groups,
    mean_by_groups,
    nanmean_or_nan,
)
from milgrau.level2.config import (
    get_block_average_minutes,
    get_gluing_config,
    get_kfs_mode,
    get_lidar_ratio,
    get_molecular_fit_config,
)
from milgrau.level2.discovery import infer_channel_pair
from milgrau.level2.atmosphere import get_standard_atmosphere
from milgrau.level2.gluing import merge_source_flags, propagate_glued_error, slide_glue_signals
from milgrau.level2.kfs import kfs_inversion_monte_carlo
from milgrau.level2.molecular import (
    calculate_molecular_profile,
    calculate_simulated_molecular_signal,
    find_optimal_reference_altitude,
    linear_rayleigh_calibration_factor,
)

KFS_BRANCH_BACKWARD_BELOW_REFERENCE = 1
KFS_BRANCH_REFERENCE_WINDOW = 2
KFS_BRANCH_FORWARD_ABOVE_REFERENCE_EXPERIMENTAL = 3


def build_kfs_branch(altitude_m: np.ndarray, ref_start_m: float, ref_stop_m: float, mode: str) -> np.ndarray:
    """Build per-altitude validity/branch flags for KFS products."""
    altitude = np.asarray(altitude_m, dtype=np.float64)
    branch = np.zeros(altitude.size, dtype=np.int8)
    finite = np.isfinite(altitude)
    branch[finite & (altitude < ref_start_m)] = KFS_BRANCH_BACKWARD_BELOW_REFERENCE
    branch[finite & (altitude >= ref_start_m) & (altitude <= ref_stop_m)] = KFS_BRANCH_REFERENCE_WINDOW
    if mode == "two_sided":
        branch[finite & (altitude > ref_stop_m)] = KFS_BRANCH_FORWARD_ABOVE_REFERENCE_EXPERIMENTAL
    return branch


def evaluate_rayleigh_reference(
    measured_signal: np.ndarray,
    simulated_molecular_signal: np.ndarray,
    altitude_m: np.ndarray,
    reference_center_idx: int,
    reference_window_bins: int,
    fit_config: Mapping[str, Any],
    calibration_factor: float,
) -> dict[str, float | int]:
    """Evaluate whether the selected Rayleigh reference window is acceptable."""
    measured = np.asarray(measured_signal, dtype=np.float64)
    simulated = np.asarray(simulated_molecular_signal, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    center = int(reference_center_idx)
    half_window = max(int(reference_window_bins) // 2, 1)
    start = max(center - half_window, 0)
    stop = min(center + half_window + 1, measured.size)
    ratio = measured[start:stop] / simulated[start:stop]
    window_altitude = altitude[start:stop]
    valid = np.isfinite(ratio) & np.isfinite(window_altitude) & (ratio > 0.0)
    valid_count = int(valid.sum())
    window_size = max(int(stop - start), 1)
    valid_fraction = float(valid_count / window_size)

    relative_variance = np.inf
    relative_slope = np.inf
    if valid_count >= 3:
        valid_ratio = ratio[valid]
        valid_altitude = window_altitude[valid]
        mean_ratio = float(np.nanmean(valid_ratio))
        if np.isfinite(mean_ratio) and mean_ratio > 0.0:
            relative_variance = float(np.nanvar(valid_ratio) / (mean_ratio**2))
            slope, _ = np.polyfit(valid_altitude, valid_ratio, 1)
            altitude_span = float(np.nanmax(valid_altitude) - np.nanmin(valid_altitude))
            relative_slope = float(abs(slope) * max(altitude_span, 1.0) / mean_ratio)

    max_relative_slope = float(fit_config.get("max_relative_slope", 0.25))
    max_relative_variance = float(fit_config.get("max_relative_variance", 0.50))
    min_valid_fraction = float(fit_config.get("min_valid_fraction", 0.50))
    success = (
        np.isfinite(calibration_factor)
        and calibration_factor > 0.0
        and valid_fraction >= min_valid_fraction
        and np.isfinite(relative_variance)
        and relative_variance <= max_relative_variance
        and np.isfinite(relative_slope)
        and relative_slope <= max_relative_slope
    )
    return {
        "success_flag": int(success),
        "relative_slope": float(relative_slope),
        "relative_variance": float(relative_variance),
        "valid_fraction": float(valid_fraction),
        "max_relative_slope": max_relative_slope,
        "max_relative_variance": max_relative_variance,
        "min_valid_fraction": min_valid_fraction,
    }


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return numerator/denominator where both terms are finite and positive."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0.0),
    )


def build_thermodynamic_profile(
    ds_l1: xr.Dataset,
    altitude_agl_m: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Build pressure and temperature profiles on the lidar altitude grid."""
    site_cfg = config.get("site", {})
    station_altitude_m = float(
        site_cfg.get("station_altitude_m", config.get("physics", {}).get("station_altitude_m", 0.0))
    )
    altitude_asl_m = altitude_agl_m + station_altitude_m

    standard_pressure, standard_temperature = get_standard_atmosphere(altitude_asl_m)
    if {"Radiosonde_Temperature_K", "Radiosonde_Pressure_hPa", "radiosonde_altitude"}.issubset(
        set(ds_l1.variables) | set(ds_l1.coords)
    ):
        radio_alt = np.asarray(ds_l1["radiosonde_altitude"].values, dtype=np.float64)
        radio_temp = np.asarray(ds_l1["Radiosonde_Temperature_K"].values, dtype=np.float64)
        radio_pressure = np.asarray(ds_l1["Radiosonde_Pressure_hPa"].values, dtype=np.float64)
        valid = (
            np.isfinite(radio_alt)
            & np.isfinite(radio_temp)
            & np.isfinite(radio_pressure)
            & (radio_pressure > 0.0)
            & (radio_temp > 0.0)
        )
        if valid.sum() >= 2:
            order = np.argsort(radio_alt[valid])
            alt_sorted = radio_alt[valid][order]
            temp_sorted = radio_temp[valid][order]
            pressure_sorted = radio_pressure[valid][order]
            temperature = np.interp(altitude_asl_m, alt_sorted, temp_sorted, left=np.nan, right=np.nan)
            pressure = np.interp(altitude_asl_m, alt_sorted, pressure_sorted, left=np.nan, right=np.nan)
            temperature = np.where(np.isfinite(temperature), temperature, standard_temperature)
            pressure = np.where(np.isfinite(pressure), pressure, standard_pressure)
            return pressure.astype(np.float64), temperature.astype(np.float64), "radiosonde_with_standard_fallback"

    return standard_pressure.astype(np.float64), standard_temperature.astype(np.float64), "standard_atmosphere"


def run_kfs_profile(
    rcs: np.ndarray,
    rcs_error: np.ndarray,
    altitude_m: np.ndarray,
    beta_mol: np.ndarray,
    ref_idx: int,
    lr_base: float,
    lr_std: float,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run KFS for one RCS profile using the configured inversion options."""
    inv_cfg = config.get("inversion", {})
    return kfs_inversion_monte_carlo(
        rcs=rcs,
        altitude=altitude_m,
        beta_mol=beta_mol,
        lr_base=lr_base,
        lr_std=lr_std,
        ref_idx=ref_idx,
        n_iterations=int(inv_cfg.get("monte_carlo_iterations", 300)),
        rcs_error=rcs_error,
        beta_ref_relative_std=float(inv_cfg.get("beta_ref_relative_std", 0.10)),
        aerosol_ref_fraction=float(inv_cfg.get("aerosol_ref_fraction", 0.0)),
        altitude_units="m",
        min_lidar_ratio=float(inv_cfg.get("min_lidar_ratio_sr", 10.0)),
        allow_negative_aerosol=bool(inv_cfg.get("allow_negative_aerosol", False)),
        seed=inv_cfg.get("random_seed"),
        mode=get_kfs_mode(config),
    )


def origin_rayleigh_calibration_factor(
    measured_signal: np.ndarray,
    simulated_molecular_signal: np.ndarray,
    altitude_m: np.ndarray,
    reference_center_idx: int,
    reference_window_bins: int,
) -> tuple[float, float, float, int]:
    """Return multiplicative Rayleigh calibration constrained through the origin."""
    measured = np.asarray(measured_signal, dtype=np.float64)
    simulated = np.asarray(simulated_molecular_signal, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    center = int(reference_center_idx)
    half_window = max(int(reference_window_bins) // 2, 1)
    start = max(center - half_window, 0)
    stop = min(center + half_window + 1, measured.size)
    x = simulated[start:stop]
    y = measured[start:stop]
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if valid.sum() < 2:
        return np.nan, float(altitude[start]), float(altitude[stop - 1]), int(valid.sum())
    factor = float(np.nansum(x[valid] * y[valid]) / np.nansum(x[valid] ** 2))
    return factor, float(altitude[start]), float(altitude[stop - 1]), int(valid.sum())


def valid_block_mean(block_matrix: np.ndarray, valid_block: np.ndarray) -> np.ndarray:
    """Average only block products that passed gluing and Rayleigh QA."""
    block_matrix = np.asarray(block_matrix, dtype=np.float64)
    valid = np.asarray(valid_block, dtype=bool)
    if valid.any():
        return nanmean_or_nan(block_matrix[valid, :], axis=0)
    return np.full(block_matrix.shape[-1], np.nan, dtype=np.float64)


def valid_block_error(block_error_matrix: np.ndarray, valid_block: np.ndarray) -> np.ndarray:
    """Combine block uncertainties using only valid retrieval blocks."""
    block_error_matrix = np.asarray(block_error_matrix, dtype=np.float64)
    valid = np.asarray(valid_block, dtype=bool)
    if valid.any():
        return error_of_mean(block_error_matrix[valid, :])
    return np.full(block_error_matrix.shape[-1], np.nan, dtype=np.float64)


def range_square_factor(altitude_m: np.ndarray) -> np.ndarray:
    """Return the range-squared factor used to convert corrected signal to RCS."""
    return np.asarray(altitude_m, dtype=np.float64) ** 2


def to_rcs(corrected: np.ndarray, corrected_error: np.ndarray, altitude_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert corrected signal and uncertainty to range-corrected signal."""
    factor = range_square_factor(altitude_m)
    return corrected * factor, corrected_error * factor


def process_wavelength(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Process one wavelength using block retrievals."""
    analog_ch, photon_ch = infer_channel_pair(ds_l1, wavelength_nm)
    if analog_ch is None and photon_ch is None:
        raise ValueError(f"No channel found for wavelength {wavelength_nm} nm.")

    corrected_da = ds_l1["corrected_signal"]
    corrected_error_da = ds_l1["corrected_signal_error"]
    n_time = ds_l1.sizes.get("time", 1)
    n_alt = altitude_m.size
    block_minutes = get_block_average_minutes(config)
    block_time, block_groups_idx = block_groups(ds_l1["time"].values, block_minutes)
    n_block = len(block_groups_idx)
    gluing_cfg = get_gluing_config(config)
    range_factor = range_square_factor(altitude_m)

    if photon_ch is not None:
        photon_signal = corrected_da.sel(channel=photon_ch).values.astype(np.float64)
        photon_error = corrected_error_da.sel(channel=photon_ch).values.astype(np.float64)
        photon_block = mean_by_groups(photon_signal, block_groups_idx)
        photon_error_block = error_by_groups(photon_error, block_groups_idx)
        if "pc_saturation_mask" in ds_l1:
            photon_mask = ds_l1["pc_saturation_mask"].sel(channel=photon_ch).values.astype(bool)
            photon_mask_block = mask_by_groups(photon_mask, block_groups_idx)
        else:
            photon_mask_block = np.zeros((n_block, n_alt), dtype=bool)
    else:
        photon_block = None
        photon_error_block = None
        photon_mask_block = None

    if analog_ch is not None:
        analog_signal = corrected_da.sel(channel=analog_ch).values.astype(np.float64)
        analog_error = corrected_error_da.sel(channel=analog_ch).values.astype(np.float64)
        analog_block = mean_by_groups(analog_signal, block_groups_idx)
        analog_error_block = error_by_groups(analog_error, block_groups_idx)
    else:
        analog_block = None
        analog_error_block = None

    glued_corrected_block = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    glued_corrected_error_block = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    glued_block = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    glued_error_block = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    merge_source_block = np.full((n_block, n_alt), 3, dtype=np.int8)
    gluing_success_block = np.zeros(n_block, dtype=np.int8)
    gluing_fallback_block = np.zeros(n_block, dtype=np.int8)
    gluing_split_block = np.full(n_block, np.nan, dtype=np.float64)
    gluing_start_block = np.full(n_block, np.nan, dtype=np.float64)
    gluing_stop_block = np.full(n_block, np.nan, dtype=np.float64)
    gluing_slope_block = np.full(n_block, np.nan, dtype=np.float64)
    gluing_intercept_block = np.full(n_block, np.nan, dtype=np.float64)
    gluing_correlation_block = np.full(n_block, np.nan, dtype=np.float64)
    gluing_rmse_block = np.full(n_block, np.nan, dtype=np.float64)
    gluing_bias_block = np.full(n_block, np.nan, dtype=np.float64)

    if analog_block is not None and photon_block is not None and analog_error_block is not None and photon_error_block is not None:
        source = "block_mean_corrected_signal_analog_photon_glued"
        for block_idx in range(n_block):
            glued_profile, split_point, slope_i, intercept_i, diagnostics = slide_glue_signals(
                analog_sig=analog_block[block_idx, :],
                pc_sig=photon_block[block_idx, :],
                altitude=altitude_m,
                window_size=gluing_cfg["window_size"],
                min_corr=gluing_cfg["min_corr"],
                search_min_idx=gluing_cfg["search_min_idx"],
                search_max_idx=gluing_cfg["search_max_idx"],
                intercept_threshold=gluing_cfg["intercept_threshold"],
                gaussian_threshold=gluing_cfg["gaussian_threshold"],
                minmax_threshold=gluing_cfg["minmax_threshold"],
                max_relative_rmse=gluing_cfg["max_relative_rmse"],
                max_relative_bias=gluing_cfg["max_relative_bias"],
                min_valid_fraction=gluing_cfg["min_valid_fraction"],
                max_saturation_fraction=gluing_cfg["max_saturation_fraction"],
                invalid_saturation_fraction=gluing_cfg.get("invalid_saturation_fraction", 1.0),
                pc_saturation_mask=photon_mask_block[block_idx, :] if photon_mask_block is not None else None,
                return_diagnostics=True,
            )
            gluing_slope_block[block_idx] = slope_i
            gluing_intercept_block[block_idx] = intercept_i
            gluing_correlation_block[block_idx] = float(diagnostics.get("best_corr", np.nan))
            gluing_rmse_block[block_idx] = float(diagnostics.get("relative_rmse", np.nan))
            gluing_bias_block[block_idx] = float(diagnostics.get("relative_bias", np.nan))
            if split_point >= 0:
                min_bin = int(diagnostics.get("min_bin", max(split_point - gluing_cfg["window_size"] // 2, 0)))
                max_bin = int(diagnostics.get("max_bin", min(split_point + gluing_cfg["window_size"] // 2, n_alt)))
                glued_corrected_block[block_idx, :] = glued_profile
                glued_corrected_error_block[block_idx, :] = propagate_glued_error(
                    analog_error_block[block_idx, :],
                    photon_error_block[block_idx, :],
                    slope_i,
                    min_bin,
                    max_bin,
                )
                glued_block[block_idx, :], glued_error_block[block_idx, :] = to_rcs(
                    glued_corrected_block[block_idx, :],
                    glued_corrected_error_block[block_idx, :],
                    altitude_m,
                )
                merge_source_block[block_idx, :] = merge_source_flags(n_alt, min_bin, max_bin)
                gluing_success_block[block_idx] = 1
                gluing_split_block[block_idx] = float(altitude_m[split_point])
                gluing_start_block[block_idx] = float(altitude_m[min_bin])
                gluing_stop_block[block_idx] = float(altitude_m[max_bin - 1])
            elif gluing_cfg["fallback_to_photon_counting"]:
                glued_corrected_block[block_idx, :] = photon_block[block_idx, :]
                glued_corrected_error_block[block_idx, :] = photon_error_block[block_idx, :]
                glued_block[block_idx, :], glued_error_block[block_idx, :] = to_rcs(
                    glued_corrected_block[block_idx, :],
                    glued_corrected_error_block[block_idx, :],
                    altitude_m,
                )
                merge_source_block[block_idx, :] = merge_source_flags(n_alt, -1, -1, split_failed=True)
                gluing_fallback_block[block_idx] = 1
        if gluing_success_block.sum() == 0 and not gluing_cfg["fallback_to_photon_counting"]:
            raise ValueError(f"{wavelength_nm} nm has no successful block gluing and photon fallback is disabled.")
        logger.info(
            f"  -> {wavelength_nm} nm block gluing success: {100.0 * gluing_success_block.sum() / max(n_block, 1):.1f}% ({analog_ch} + {photon_ch}); fallback blocks: {int(gluing_fallback_block.sum())}."
        )
    else:
        fallback_ch = photon_ch or analog_ch
        fallback_block = photon_block if photon_block is not None else analog_block
        fallback_error_block = photon_error_block if photon_error_block is not None else analog_error_block
        if fallback_ch is None or fallback_block is None or fallback_error_block is None:
            raise ValueError(f"No usable channel found for wavelength {wavelength_nm} nm.")
        if not gluing_cfg["fallback_to_photon_counting"]:
            raise ValueError(
                f"{wavelength_nm} nm cannot perform gluing because only {fallback_ch} is available and photon fallback is disabled."
            )
        glued_corrected_block[:, :] = fallback_block
        glued_corrected_error_block[:, :] = fallback_error_block
        glued_block[:, :] = fallback_block * range_factor
        glued_error_block[:, :] = fallback_error_block * range_factor
        merge_source_block[:, :] = 0 if fallback_ch == photon_ch else 2
        gluing_fallback_block[:] = 1
        source = f"block_mean_corrected_signal_single_channel_{fallback_ch}"
        logger.warning(f"  -> {wavelength_nm} nm using block single-channel fallback: {fallback_ch}.")

    pressure_hpa, temperature_k, molecular_source = build_thermodynamic_profile(ds_l1, altitude_m, config)
    beta_mol, alpha_mol = calculate_molecular_profile(temperature_k, pressure_hpa, wavelength_nm)
    simulated_signal, molecular_transmission = calculate_simulated_molecular_signal(beta_mol, alpha_mol, altitude_m)
    positive_altitudes = altitude_m[altitude_m > 0.0]
    safe_altitude = np.where(altitude_m > 0.0, altitude_m, positive_altitudes[0] if positive_altitudes.size else 1.0)
    simulated_molecular_rcs = simulated_signal * safe_altitude**2

    fit_cfg = get_molecular_fit_config(config)
    lr_base, lr_std = get_lidar_ratio(config, wavelength_nm, ds_l1["time"].values[0])
    kfs_mode = get_kfs_mode(config)

    rayleigh_success_block = np.zeros(n_block, dtype=np.int8)
    ref_altitude_block = np.full(n_block, np.nan, dtype=np.float64)
    ref_start_block = np.full(n_block, np.nan, dtype=np.float64)
    ref_stop_block = np.full(n_block, np.nan, dtype=np.float64)
    ref_valid_bins_block = np.zeros(n_block, dtype=np.int32)
    ref_relative_slope_block = np.full(n_block, np.nan, dtype=np.float64)
    ref_relative_variance_block = np.full(n_block, np.nan, dtype=np.float64)
    ref_valid_fraction_block = np.full(n_block, np.nan, dtype=np.float64)
    calibration_factor_block = np.full(n_block, np.nan, dtype=np.float64)
    calibration_intercept_block = np.full(n_block, np.nan, dtype=np.float64)
    scaled_molecular_rcs_block = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    scattering_ratio_block = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    beta_block = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    beta_block_std = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    alpha_block = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    alpha_block_std = np.full((n_block, n_alt), np.nan, dtype=np.float64)
    kfs_branch_block = np.zeros((n_block, n_alt), dtype=np.int8)

    for block_idx in range(n_block):
        if gluing_success_block[block_idx] != 1:
            continue
        ref_idx = find_optimal_reference_altitude(
            rcs=glued_block[block_idx, :],
            beta_mol=simulated_molecular_rcs,
            altitude=altitude_m,
            min_alt=fit_cfg["ref_alt_min_m"],
            max_alt=fit_cfg["ref_alt_max_m"],
            window_size=fit_cfg["ref_window_bins"],
            altitude_units="m",
        )
        factor, ref_start_m, ref_stop_m, ref_valid_bins = origin_rayleigh_calibration_factor(
            measured_signal=glued_block[block_idx, :],
            simulated_molecular_signal=simulated_molecular_rcs,
            altitude_m=altitude_m,
            reference_center_idx=ref_idx,
            reference_window_bins=fit_cfg["ref_window_bins"],
        )
        _, intercept_diag, _, _, _ = linear_rayleigh_calibration_factor(
            measured_signal=glued_block[block_idx, :],
            simulated_molecular_signal=simulated_molecular_rcs,
            altitude_m=altitude_m,
            reference_center_idx=ref_idx,
            reference_window_bins=fit_cfg["ref_window_bins"],
        )
        qa = evaluate_rayleigh_reference(
            glued_block[block_idx, :],
            simulated_molecular_rcs,
            altitude_m,
            ref_idx,
            fit_cfg["ref_window_bins"],
            fit_cfg,
            factor,
        )
        calibration_factor_block[block_idx] = factor
        calibration_intercept_block[block_idx] = intercept_diag
        ref_altitude_block[block_idx] = float(altitude_m[ref_idx])
        ref_start_block[block_idx] = ref_start_m
        ref_stop_block[block_idx] = ref_stop_m
        ref_valid_bins_block[block_idx] = int(ref_valid_bins)
        ref_relative_slope_block[block_idx] = float(qa["relative_slope"])
        ref_relative_variance_block[block_idx] = float(qa["relative_variance"])
        ref_valid_fraction_block[block_idx] = float(qa["valid_fraction"])
        scaled_molecular_rcs_block[block_idx, :] = simulated_molecular_rcs * factor
        scattering_ratio_block[block_idx, :] = safe_ratio(glued_block[block_idx, :], scaled_molecular_rcs_block[block_idx, :])
        kfs_branch_block[block_idx, :] = build_kfs_branch(altitude_m, ref_start_m, ref_stop_m, kfs_mode)
        if int(qa["success_flag"]) == 1:
            rayleigh_success_block[block_idx] = 1
            beta_mean, beta_std, alpha_mean, alpha_std = run_kfs_profile(
                glued_block[block_idx, :],
                glued_error_block[block_idx, :],
                altitude_m,
                beta_mol,
                ref_idx,
                lr_base,
                lr_std,
                config,
            )
            beta_block[block_idx, :] = beta_mean
            beta_block_std[block_idx, :] = beta_std
            alpha_block[block_idx, :] = alpha_mean
            alpha_block_std[block_idx, :] = alpha_std

    valid_block = (gluing_success_block == 1) & (rayleigh_success_block == 1)
    if not valid_block.any():
        logger.warning(f"  -> {wavelength_nm} nm has no valid retrieval block. Mean optical products set to NaN.")

    glued_corrected_mean = valid_block_mean(glued_corrected_block, valid_block)
    glued_corrected_error_mean = valid_block_error(glued_corrected_error_block, valid_block)
    glued_mean = valid_block_mean(glued_block, valid_block)
    glued_error_mean = valid_block_error(glued_error_block, valid_block)
    scattering_ratio_mean = valid_block_mean(scattering_ratio_block, valid_block)
    beta_mean = valid_block_mean(beta_block, valid_block)
    beta_std_mean = valid_block_error(beta_block_std, valid_block)
    alpha_mean = valid_block_mean(alpha_block, valid_block)
    alpha_std_mean = valid_block_error(alpha_block_std, valid_block)

    if valid_block.any():
        calibration_factor = float(np.nanmedian(calibration_factor_block[valid_block]))
        calibration_intercept = float(np.nanmedian(calibration_intercept_block[valid_block]))
        ref_altitude = float(np.nanmedian(ref_altitude_block[valid_block]))
        ref_start = float(np.nanmedian(ref_start_block[valid_block]))
        ref_stop = float(np.nanmedian(ref_stop_block[valid_block]))
        ref_valid_bins = int(np.nanmedian(ref_valid_bins_block[valid_block]))
        ref_rel_slope = float(np.nanmedian(ref_relative_slope_block[valid_block]))
        ref_rel_var = float(np.nanmedian(ref_relative_variance_block[valid_block]))
        ref_valid_fraction = float(np.nanmedian(ref_valid_fraction_block[valid_block]))
        scaled_molecular_rcs = simulated_molecular_rcs * calibration_factor
        kfs_branch = build_kfs_branch(altitude_m, ref_start, ref_stop, kfs_mode)
        rayleigh_success = 1
    else:
        calibration_factor = np.nan
        calibration_intercept = np.nan
        ref_altitude = np.nan
        ref_start = np.nan
        ref_stop = np.nan
        ref_valid_bins = 0
        ref_rel_slope = np.nan
        ref_rel_var = np.nan
        ref_valid_fraction = np.nan
        scaled_molecular_rcs = np.full(n_alt, np.nan, dtype=np.float64)
        kfs_branch = np.zeros(n_alt, dtype=np.int8)
        rayleigh_success = 0

    time_glued_corrected = expand_blocks_to_time(glued_corrected_block, block_groups_idx, n_time)
    time_glued_corrected_error = expand_blocks_to_time(glued_corrected_error_block, block_groups_idx, n_time)
    time_glued = expand_blocks_to_time(glued_block, block_groups_idx, n_time)
    time_glued_error = expand_blocks_to_time(glued_error_block, block_groups_idx, n_time)
    time_merge_source = np.full((n_time, n_alt), 3, dtype=np.int8)
    for block_idx, group in enumerate(block_groups_idx):
        time_merge_source[group, :] = merge_source_block[block_idx, :]
    time_gluing_success = expand_block_vector_to_time(gluing_success_block, block_groups_idx, n_time, dtype=np.int8)
    time_gluing_fallback = expand_block_vector_to_time(gluing_fallback_block, block_groups_idx, n_time, dtype=np.int8)
    time_gluing_split = expand_block_vector_to_time(gluing_split_block, block_groups_idx, n_time)
    time_gluing_start = expand_block_vector_to_time(gluing_start_block, block_groups_idx, n_time)
    time_gluing_stop = expand_block_vector_to_time(gluing_stop_block, block_groups_idx, n_time)
    time_gluing_slope = expand_block_vector_to_time(gluing_slope_block, block_groups_idx, n_time)
    time_gluing_intercept = expand_block_vector_to_time(gluing_intercept_block, block_groups_idx, n_time)
    time_gluing_correlation = expand_block_vector_to_time(gluing_correlation_block, block_groups_idx, n_time)
    time_gluing_rmse = expand_block_vector_to_time(gluing_rmse_block, block_groups_idx, n_time)
    time_gluing_bias = expand_block_vector_to_time(gluing_bias_block, block_groups_idx, n_time)

    return {
        "wavelength": wavelength_nm,
        "block_time": block_time,
        "molecular_source": molecular_source,
        "molecular_backscatter": beta_mol,
        "molecular_extinction": alpha_mol,
        "molecular_transmission": molecular_transmission,
        "simulated_molecular_signal": simulated_signal,
        "simulated_molecular_range_corrected_signal": simulated_molecular_rcs,
        "scaled_molecular_range_corrected_signal": scaled_molecular_rcs,
        "scaled_molecular_range_corrected_signal_block": scaled_molecular_rcs_block,
        "glued_corrected_signal": time_glued_corrected,
        "glued_corrected_signal_error": time_glued_corrected_error,
        "glued_corrected_signal_block": glued_corrected_block,
        "glued_corrected_signal_error_block": glued_corrected_error_block,
        "glued_corrected_signal_mean": glued_corrected_mean,
        "glued_corrected_signal_error_mean": glued_corrected_error_mean,
        "glued_range_corrected_signal": time_glued,
        "glued_range_corrected_signal_error": time_glued_error,
        "glued_range_corrected_signal_block": glued_block,
        "glued_range_corrected_signal_error_block": glued_error_block,
        "glued_range_corrected_signal_mean": glued_mean,
        "glued_range_corrected_signal_error_mean": glued_error_mean,
        "gluing_merge_source_flag": time_merge_source,
        "gluing_merge_source_flag_block": merge_source_block,
        "scattering_ratio_mean": scattering_ratio_mean,
        "scattering_ratio_block": scattering_ratio_block,
        "aerosol_backscatter": beta_mean,
        "aerosol_backscatter_error": beta_std_mean,
        "aerosol_extinction": alpha_mean,
        "aerosol_extinction_error": alpha_std_mean,
        "aerosol_backscatter_block": beta_block,
        "aerosol_backscatter_error_block": beta_block_std,
        "aerosol_extinction_block": alpha_block,
        "aerosol_extinction_error_block": alpha_block_std,
        "valid_retrieval_block_flag": valid_block.astype(np.int8),
        "rayleigh_reference_altitude_m": ref_altitude,
        "rayleigh_reference_start_altitude_m": ref_start,
        "rayleigh_reference_stop_altitude_m": ref_stop,
        "rayleigh_reference_valid_bins": ref_valid_bins,
        "rayleigh_reference_success_flag": rayleigh_success,
        "rayleigh_reference_relative_slope": ref_rel_slope,
        "rayleigh_reference_relative_variance": ref_rel_var,
        "rayleigh_reference_valid_fraction": ref_valid_fraction,
        "rayleigh_calibration_factor": calibration_factor,
        "rayleigh_calibration_intercept": calibration_intercept,
        "rayleigh_reference_altitude_m_block": ref_altitude_block,
        "rayleigh_reference_start_altitude_m_block": ref_start_block,
        "rayleigh_reference_stop_altitude_m_block": ref_stop_block,
        "rayleigh_reference_valid_bins_block": ref_valid_bins_block,
        "rayleigh_reference_success_flag_block": rayleigh_success_block,
        "rayleigh_reference_relative_slope_block": ref_relative_slope_block,
        "rayleigh_reference_relative_variance_block": ref_relative_variance_block,
        "rayleigh_reference_valid_fraction_block": ref_valid_fraction_block,
        "rayleigh_calibration_factor_block": calibration_factor_block,
        "rayleigh_calibration_intercept_block": calibration_intercept_block,
        "lidar_ratio_assumed_sr": lr_base,
        "lidar_ratio_std_sr": lr_std,
        "kfs_branch": kfs_branch,
        "kfs_branch_block": kfs_branch_block,
        "gluing_success_flag": time_gluing_success,
        "gluing_fallback_flag": time_gluing_fallback,
        "gluing_split_altitude_m": time_gluing_split,
        "gluing_start_altitude_m": time_gluing_start,
        "gluing_stop_altitude_m": time_gluing_stop,
        "gluing_slope": time_gluing_slope,
        "gluing_intercept": time_gluing_intercept,
        "gluing_correlation": time_gluing_correlation,
        "gluing_relative_rmse": time_gluing_rmse,
        "gluing_relative_bias": time_gluing_bias,
        "gluing_success_flag_block": gluing_success_block,
        "gluing_fallback_flag_block": gluing_fallback_block,
        "gluing_split_altitude_m_block": gluing_split_block,
        "gluing_start_altitude_m_block": gluing_start_block,
        "gluing_stop_altitude_m_block": gluing_stop_block,
        "gluing_slope_block": gluing_slope_block,
        "gluing_intercept_block": gluing_intercept_block,
        "gluing_correlation_block": gluing_correlation_block,
        "gluing_relative_rmse_block": gluing_rmse_block,
        "gluing_relative_bias_block": gluing_bias_block,
        "gluing_source": source,
        "analog_channel": analog_ch,
        "photon_channel": photon_ch,
    }
