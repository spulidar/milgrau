"""Canonical productive Rayleigh and backward Klett--Fernald retrieval."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any, Mapping

import numpy as np

from milgrau.level2.block_average import valid_block_error, valid_block_mean
from milgrau.level2.config import get_kfs_config
from milgrau.level2.contracts import (
    KfsDiagnostics,
    MolecularProfiles,
    OpticalProducts,
    RayleighDiagnostics,
)
from milgrau.level2.kfs import kfs_inversion_monte_carlo
from milgrau.level2.molecular import (
    find_optimal_reference_altitude,
    linear_rayleigh_calibration_factor,
)
from milgrau.level2.signal_selection import BlockGluingResult, WavelengthBlockInputs

KFS_BRANCH_BACKWARD_BELOW_REFERENCE = 1
KFS_BRANCH_REFERENCE_BIN = 2
KFS_BRANCH_FORWARD_ABOVE_REFERENCE = 3


@dataclass(frozen=True, slots=True)
class MolecularModel:
    """Molecular atmosphere and explicit retrieval assumptions for one wavelength."""

    source: str
    backscatter: np.ndarray
    extinction: np.ndarray
    transmission: np.ndarray
    simulated_signal: np.ndarray
    simulated_range_corrected_signal: np.ndarray
    fit_config: dict[str, Any]
    lidar_ratio_assumed_sr: float
    lidar_ratio_std_sr: float
    kfs_mode: str


def build_kfs_branch(
    altitude_m: np.ndarray,
    reference_index: int,
    mode: str,
) -> np.ndarray:
    """Build diagnostic branch flags around the inversion boundary bin."""
    altitude = np.asarray(altitude_m, dtype=np.float64)
    reference_index = int(reference_index)
    if altitude.ndim != 1 or reference_index < 0 or reference_index >= altitude.size:
        raise ValueError(
            "reference_index must identify one bin on the 1D altitude grid."
        )
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"backward", "forward", "two_sided"}:
        raise ValueError("mode must be 'backward', 'forward', or 'two_sided'.")

    branch = np.zeros(altitude.size, dtype=np.int8)
    finite = np.isfinite(altitude)
    indices = np.arange(altitude.size)
    if normalized_mode in {"backward", "two_sided"}:
        branch[finite & (indices < reference_index)] = (
            KFS_BRANCH_BACKWARD_BELOW_REFERENCE
        )
    if finite[reference_index]:
        branch[reference_index] = KFS_BRANCH_REFERENCE_BIN
    if normalized_mode in {"forward", "two_sided"}:
        branch[finite & (indices > reference_index)] = (
            KFS_BRANCH_FORWARD_ABOVE_REFERENCE
        )
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
    """Evaluate the selected Rayleigh window against explicit configured limits."""
    measured = np.asarray(measured_signal, dtype=np.float64)
    simulated = np.asarray(simulated_molecular_signal, dtype=np.float64)
    altitude = np.asarray(altitude_m, dtype=np.float64)
    center = int(reference_center_idx)
    half_window = max(int(reference_window_bins) // 2, 1)
    start = max(center - half_window, 0)
    stop = min(center + half_window + 1, measured.size)
    ratio = measured[start:stop] / simulated[start:stop]
    window_altitude = altitude[start:stop]
    valid = (
        np.isfinite(ratio)
        & np.isfinite(window_altitude)
        & (ratio > 0.0)
    )
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
            relative_variance = float(
                np.nanvar(valid_ratio) / (mean_ratio**2)
            )
            slope, _ = np.polyfit(valid_altitude, valid_ratio, 1)
            altitude_span = float(
                np.nanmax(valid_altitude) - np.nanmin(valid_altitude)
            )
            relative_slope = float(
                abs(slope) * max(altitude_span, 1.0) / mean_ratio
            )

    max_relative_slope = float(fit_config["max_relative_slope"])
    max_relative_variance = float(fit_config["max_relative_variance"])
    min_valid_fraction = float(fit_config["min_valid_fraction"])
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


def origin_rayleigh_calibration_factor(
    measured_signal: np.ndarray,
    simulated_molecular_signal: np.ndarray,
    altitude_m: np.ndarray,
    reference_center_idx: int,
    reference_window_bins: int,
) -> tuple[float, float, float, int]:
    """Return a multiplicative Rayleigh calibration constrained through the origin."""
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
        return (
            np.nan,
            float(altitude[start]),
            float(altitude[stop - 1]),
            int(valid.sum()),
        )
    denominator = float(np.nansum(x[valid] ** 2))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return (
            np.nan,
            float(altitude[start]),
            float(altitude[stop - 1]),
            int(valid.sum()),
        )
    factor = float(np.nansum(x[valid] * y[valid]) / denominator)
    return (
        factor,
        float(altitude[start]),
        float(altitude[stop - 1]),
        int(valid.sum()),
    )


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return numerator/denominator only on finite positive denominator support."""
    numerator_arr = np.asarray(numerator, dtype=np.float64)
    denominator_arr = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator_arr,
        denominator_arr,
        out=np.full_like(numerator_arr, np.nan, dtype=np.float64),
        where=(
            np.isfinite(numerator_arr)
            & np.isfinite(denominator_arr)
            & (denominator_arr > 0.0)
        ),
    )


def run_kfs_profile(
    rcs: np.ndarray,
    rcs_error: np.ndarray,
    altitude_m: np.ndarray,
    beta_mol: np.ndarray,
    ref_idx: int,
    lr_base: float,
    lr_std: float,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Run one KFS Monte Carlo profile from the strict resolved Level 2 config."""
    kfs_cfg = get_kfs_config(config)
    return kfs_inversion_monte_carlo(
        rcs=rcs,
        altitude=altitude_m,
        beta_mol=beta_mol,
        lr_base=lr_base,
        lr_std=lr_std,
        ref_idx=ref_idx,
        n_iterations=int(kfs_cfg["monte_carlo_iterations"]),
        rcs_error=rcs_error,
        beta_ref_relative_std=float(kfs_cfg["beta_ref_relative_std"]),
        aerosol_ref_fraction=float(kfs_cfg["aerosol_ref_fraction"]),
        altitude_units="m",
        min_lidar_ratio=float(kfs_cfg["min_lidar_ratio_sr"]),
        allow_negative_aerosol=bool(kfs_cfg["allow_negative_aerosol"]),
        seed=int(kfs_cfg["random_seed"]),
        return_diagnostics=True,
        mode=str(kfs_cfg["kfs_mode"]),
    )


def _reaggregate_backward_optical_products(
    optical: OpticalProducts,
    rayleigh: RayleighDiagnostics,
    kfs: KfsDiagnostics,
) -> tuple[OpticalProducts, np.ndarray]:
    """Aggregate only blocks accepted by Rayleigh QA and backward KFS."""
    valid_block = (
        (np.asarray(rayleigh.reference_success_flag_block) == 1)
        & (np.asarray(kfs.backward_valid_flag_block) == 1)
    )
    updated = replace(
        optical,
        scattering_ratio_mean=valid_block_mean(
            optical.scattering_ratio_block, valid_block
        ),
        aerosol_backscatter=valid_block_mean(
            optical.aerosol_backscatter_block, valid_block
        ),
        aerosol_backscatter_error=valid_block_error(
            optical.aerosol_backscatter_error_block, valid_block
        ),
        aerosol_extinction=valid_block_mean(
            optical.aerosol_extinction_block, valid_block
        ),
        aerosol_extinction_error=valid_block_error(
            optical.aerosol_extinction_error_block, valid_block
        ),
        retrieval_success_flag=valid_block.astype(np.int8),
    )
    return updated, valid_block


def retrieve_optical_blocks(
    inputs: WavelengthBlockInputs,
    glued: BlockGluingResult,
    molecular: MolecularModel,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> tuple[MolecularProfiles, OpticalProducts, RayleighDiagnostics, KfsDiagnostics]:
    """Run block Rayleigh QA/KFS and aggregate the productive backward solution."""
    if str(molecular.kfs_mode).strip().lower() != "backward":
        raise ValueError("Productive optical retrieval requires backward KFS mode.")

    n_block = len(inputs.block_groups)
    n_altitude = inputs.n_altitude
    rayleigh_success = np.zeros(n_block, dtype=np.int8)
    reference_altitude = np.full(n_block, np.nan, dtype=np.float64)
    reference_start = np.full(n_block, np.nan, dtype=np.float64)
    reference_stop = np.full(n_block, np.nan, dtype=np.float64)
    reference_valid_bins = np.zeros(n_block, dtype=np.int32)
    reference_relative_slope = np.full(n_block, np.nan, dtype=np.float64)
    reference_relative_variance = np.full(n_block, np.nan, dtype=np.float64)
    reference_valid_fraction = np.full(n_block, np.nan, dtype=np.float64)
    calibration_factor = np.full(n_block, np.nan, dtype=np.float64)
    calibration_intercept = np.full(n_block, np.nan, dtype=np.float64)
    scaled_molecular_rcs = np.full(
        (n_block, n_altitude), np.nan, dtype=np.float64
    )
    scattering_ratio = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    aerosol_backscatter = np.full(
        (n_block, n_altitude), np.nan, dtype=np.float64
    )
    aerosol_backscatter_error = np.full(
        (n_block, n_altitude), np.nan, dtype=np.float64
    )
    aerosol_extinction = np.full(
        (n_block, n_altitude), np.nan, dtype=np.float64
    )
    aerosol_extinction_error = np.full(
        (n_block, n_altitude), np.nan, dtype=np.float64
    )
    kfs_branch = np.zeros((n_block, n_altitude), dtype=np.int8)
    kfs_backward_valid = np.zeros(n_block, dtype=np.int8)
    kfs_forward_valid = np.zeros(n_block, dtype=np.int8)
    fit_config = molecular.fit_config

    for block_index in range(n_block):
        if glued.retrieval_input_valid_flag[block_index] != 1:
            continue
        reference_index = find_optimal_reference_altitude(
            rcs=glued.range_corrected_signal[block_index, :],
            beta_mol=molecular.simulated_range_corrected_signal,
            altitude=altitude_m,
            min_alt=fit_config["ref_alt_min_m"],
            max_alt=fit_config["ref_alt_max_m"],
            window_size=fit_config["ref_window_bins"],
            altitude_units="m",
        )
        factor, ref_start_m, ref_stop_m, valid_bins = (
            origin_rayleigh_calibration_factor(
                measured_signal=glued.range_corrected_signal[block_index, :],
                simulated_molecular_signal=molecular.simulated_range_corrected_signal,
                altitude_m=altitude_m,
                reference_center_idx=reference_index,
                reference_window_bins=fit_config["ref_window_bins"],
            )
        )
        _, intercept_diagnostic, _, _, _ = linear_rayleigh_calibration_factor(
            measured_signal=glued.range_corrected_signal[block_index, :],
            simulated_molecular_signal=molecular.simulated_range_corrected_signal,
            altitude_m=altitude_m,
            reference_center_idx=reference_index,
            reference_window_bins=fit_config["ref_window_bins"],
        )
        qa = evaluate_rayleigh_reference(
            glued.range_corrected_signal[block_index, :],
            molecular.simulated_range_corrected_signal,
            altitude_m,
            reference_index,
            fit_config["ref_window_bins"],
            fit_config,
            factor,
        )
        calibration_factor[block_index] = factor
        calibration_intercept[block_index] = intercept_diagnostic
        reference_altitude[block_index] = float(altitude_m[reference_index])
        reference_start[block_index] = ref_start_m
        reference_stop[block_index] = ref_stop_m
        reference_valid_bins[block_index] = int(valid_bins)
        reference_relative_slope[block_index] = float(qa["relative_slope"])
        reference_relative_variance[block_index] = float(qa["relative_variance"])
        reference_valid_fraction[block_index] = float(qa["valid_fraction"])
        scaled_molecular_rcs[block_index, :] = (
            molecular.simulated_range_corrected_signal * factor
        )
        scattering_ratio[block_index, :] = safe_ratio(
            glued.range_corrected_signal[block_index, :],
            scaled_molecular_rcs[block_index, :],
        )
        kfs_branch[block_index, :] = build_kfs_branch(
            altitude_m, reference_index, molecular.kfs_mode
        )
        if int(qa["success_flag"]) != 1:
            continue

        rayleigh_success[block_index] = 1
        beta_mean, beta_std, alpha_mean, alpha_std, kfs_diagnostic = (
            run_kfs_profile(
                glued.range_corrected_signal[block_index, :],
                glued.range_corrected_signal_error[block_index, :],
                altitude_m,
                molecular.backscatter,
                reference_index,
                molecular.lidar_ratio_assumed_sr,
                molecular.lidar_ratio_std_sr,
                config,
            )
        )
        aerosol_backscatter[block_index, :] = beta_mean
        aerosol_backscatter_error[block_index, :] = beta_std
        aerosol_extinction[block_index, :] = alpha_mean
        aerosol_extinction_error[block_index, :] = alpha_std
        kfs_backward_valid[block_index] = np.int8(
            bool(kfs_diagnostic["backward_valid"])
        )
        kfs_forward_valid[block_index] = np.int8(
            bool(kfs_diagnostic["forward_valid"])
        )

    rayleigh_valid_block = (
        (np.asarray(glued.retrieval_input_valid_flag) == 1)
        & (rayleigh_success == 1)
    )
    if rayleigh_valid_block.any():
        aggregate_factor = float(
            np.nanmedian(calibration_factor[rayleigh_valid_block])
        )
        aggregate_intercept = float(
            np.nanmedian(calibration_intercept[rayleigh_valid_block])
        )
        aggregate_reference_altitude = float(
            np.nanmedian(reference_altitude[rayleigh_valid_block])
        )
        aggregate_reference_start = float(
            np.nanmedian(reference_start[rayleigh_valid_block])
        )
        aggregate_reference_stop = float(
            np.nanmedian(reference_stop[rayleigh_valid_block])
        )
        aggregate_valid_bins = int(
            np.nanmedian(reference_valid_bins[rayleigh_valid_block])
        )
        aggregate_relative_slope = float(
            np.nanmedian(reference_relative_slope[rayleigh_valid_block])
        )
        aggregate_relative_variance = float(
            np.nanmedian(reference_relative_variance[rayleigh_valid_block])
        )
        aggregate_valid_fraction = float(
            np.nanmedian(reference_valid_fraction[rayleigh_valid_block])
        )
        aggregate_scaled_molecular = (
            molecular.simulated_range_corrected_signal * aggregate_factor
        )
        aggregate_kfs_branch = build_kfs_branch(
            altitude_m,
            int(
                np.nanargmin(
                    np.abs(altitude_m - aggregate_reference_altitude)
                )
            ),
            molecular.kfs_mode,
        )
        aggregate_rayleigh_success = 1
    else:
        aggregate_factor = np.nan
        aggregate_intercept = np.nan
        aggregate_reference_altitude = np.nan
        aggregate_reference_start = np.nan
        aggregate_reference_stop = np.nan
        aggregate_valid_bins = 0
        aggregate_relative_slope = np.nan
        aggregate_relative_variance = np.nan
        aggregate_valid_fraction = np.nan
        aggregate_scaled_molecular = np.full(
            n_altitude, np.nan, dtype=np.float64
        )
        aggregate_kfs_branch = np.zeros(n_altitude, dtype=np.int8)
        aggregate_rayleigh_success = 0

    molecular_profiles = MolecularProfiles(
        source=molecular.source,
        backscatter=molecular.backscatter,
        extinction=molecular.extinction,
        transmission=molecular.transmission,
        simulated_signal=molecular.simulated_signal,
        simulated_range_corrected_signal=molecular.simulated_range_corrected_signal,
        scaled_range_corrected_signal=aggregate_scaled_molecular,
        scaled_range_corrected_signal_block=scaled_molecular_rcs,
    )
    rayleigh_diagnostics = RayleighDiagnostics(
        reference_altitude_m=aggregate_reference_altitude,
        reference_start_altitude_m=aggregate_reference_start,
        reference_stop_altitude_m=aggregate_reference_stop,
        reference_valid_bins=aggregate_valid_bins,
        reference_success_flag=aggregate_rayleigh_success,
        reference_relative_slope=aggregate_relative_slope,
        reference_relative_variance=aggregate_relative_variance,
        reference_valid_fraction=aggregate_valid_fraction,
        calibration_factor=aggregate_factor,
        calibration_intercept=aggregate_intercept,
        reference_altitude_m_block=reference_altitude,
        reference_start_altitude_m_block=reference_start,
        reference_stop_altitude_m_block=reference_stop,
        reference_valid_bins_block=reference_valid_bins,
        reference_success_flag_block=rayleigh_success,
        reference_relative_slope_block=reference_relative_slope,
        reference_relative_variance_block=reference_relative_variance,
        reference_valid_fraction_block=reference_valid_fraction,
        calibration_factor_block=calibration_factor,
        calibration_intercept_block=calibration_intercept,
    )
    preliminary_optical = OpticalProducts(
        scattering_ratio_mean=np.full(n_altitude, np.nan, dtype=np.float64),
        scattering_ratio_block=scattering_ratio,
        aerosol_backscatter=np.full(n_altitude, np.nan, dtype=np.float64),
        aerosol_backscatter_error=np.full(n_altitude, np.nan, dtype=np.float64),
        aerosol_extinction=np.full(n_altitude, np.nan, dtype=np.float64),
        aerosol_extinction_error=np.full(n_altitude, np.nan, dtype=np.float64),
        aerosol_backscatter_block=aerosol_backscatter,
        aerosol_backscatter_error_block=aerosol_backscatter_error,
        aerosol_extinction_block=aerosol_extinction,
        aerosol_extinction_error_block=aerosol_extinction_error,
        retrieval_success_flag=np.zeros(n_block, dtype=np.int8),
    )
    preliminary_kfs = KfsDiagnostics(
        lidar_ratio_assumed_sr=molecular.lidar_ratio_assumed_sr,
        lidar_ratio_std_sr=molecular.lidar_ratio_std_sr,
        backward_valid_flag=0,
        forward_valid_flag=0,
        backward_valid_flag_block=kfs_backward_valid,
        forward_valid_flag_block=kfs_forward_valid,
        branch=aggregate_kfs_branch,
        branch_block=kfs_branch,
    )
    optical_products, valid_block = _reaggregate_backward_optical_products(
        preliminary_optical, rayleigh_diagnostics, preliminary_kfs
    )
    kfs_diagnostics = replace(
        preliminary_kfs,
        backward_valid_flag=int(valid_block.any()),
        forward_valid_flag=0,
    )

    if valid_block.any():
        logger.info(
            "  -> %d nm Rayleigh reference %.0f m [%.0f, %.0f m] | "
            "valid %.1f%% | slope %.3f | variance %.3f | "
            "backward KFS %d/%d blocks",
            int(inputs.wavelength_nm),
            float(rayleigh_diagnostics.reference_altitude_m),
            float(rayleigh_diagnostics.reference_start_altitude_m),
            float(rayleigh_diagnostics.reference_stop_altitude_m),
            100.0 * float(rayleigh_diagnostics.reference_valid_fraction),
            float(rayleigh_diagnostics.reference_relative_slope),
            float(rayleigh_diagnostics.reference_relative_variance),
            int(valid_block.sum()),
            n_block,
        )
    else:
        logger.warning(
            "  -> %d nm has no valid backward retrieval block. "
            "Mean optical products set to NaN.",
            int(inputs.wavelength_nm),
        )

    return (
        molecular_profiles,
        optical_products,
        rayleigh_diagnostics,
        kfs_diagnostics,
    )
