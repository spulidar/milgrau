"""Canonical productive Rayleigh/Klett--Fernald block retrieval.

The productive elastic aerosol contract is backward Klett--Fernald.  Forward
and two-sided numerical kernels remain available for research, but productive
block success depends only on valid retrieval input, Rayleigh QA, and the
backward branch.  This module owns that aggregation directly; no package import
may replace it at runtime.
"""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import Any, Mapping

import numpy as np

from milgrau.level2._retrieval_impl import (
    BlockGluingResult,
    MolecularModel,
    WavelengthBlockInputs,
    build_kfs_branch,
    evaluate_rayleigh_reference,
    origin_rayleigh_calibration_factor,
    run_kfs_profile,
    safe_ratio,
    valid_block_error,
    valid_block_mean,
)
from milgrau.level2.contracts import (
    KfsDiagnostics,
    MolecularProfiles,
    OpticalProducts,
    RayleighDiagnostics,
)
from milgrau.level2.molecular import (
    find_optimal_reference_altitude,
    linear_rayleigh_calibration_factor,
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
    scaled_molecular_rcs = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    scattering_ratio = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    aerosol_backscatter = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    aerosol_backscatter_error = np.full(
        (n_block, n_altitude), np.nan, dtype=np.float64
    )
    aerosol_extinction = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
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
        factor, ref_start_m, ref_stop_m, valid_bins = origin_rayleigh_calibration_factor(
            measured_signal=glued.range_corrected_signal[block_index, :],
            simulated_molecular_signal=molecular.simulated_range_corrected_signal,
            altitude_m=altitude_m,
            reference_center_idx=reference_index,
            reference_window_bins=fit_config["ref_window_bins"],
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
            altitude_m,
            reference_index,
            molecular.kfs_mode,
        )
        if int(qa["success_flag"]) != 1:
            continue

        rayleigh_success[block_index] = 1
        beta_mean, beta_std, alpha_mean, alpha_std, kfs_diagnostic = run_kfs_profile(
            glued.range_corrected_signal[block_index, :],
            glued.range_corrected_signal_error[block_index, :],
            altitude_m,
            molecular.backscatter,
            reference_index,
            molecular.lidar_ratio_assumed_sr,
            molecular.lidar_ratio_std_sr,
            config,
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
        aggregate_factor = float(np.nanmedian(calibration_factor[rayleigh_valid_block]))
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
        aggregate_scaled_molecular = np.full(n_altitude, np.nan, dtype=np.float64)
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
        preliminary_optical,
        rayleigh_diagnostics,
        preliminary_kfs,
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
