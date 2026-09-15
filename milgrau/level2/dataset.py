"""Assembly of Level 2 xarray datasets from retrieval results."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr

from milgrau.level2.completeness import Level2ProductContract
from milgrau.level2.config import (
    get_block_average_minutes,
    get_kfs_mode,
    get_molecular_fit_config,
    kfs_mode_description,
)
from milgrau.level2.contracts import WavelengthRetrievalResult, validate_retrieval_results
from milgrau.level2.metadata import apply_level2_variable_metadata
from milgrau.scientific import LEVEL2_PRODUCT_SCHEMA_VERSION, elastic_inversion_algorithm_metadata


def build_level2_dataset(
    ds_l1: xr.Dataset,
    results: list[WavelengthRetrievalResult],
    altitude_m: np.ndarray,
    source_file: Path,
    config: Mapping[str, Any],
    product_contract: Level2ProductContract,
) -> xr.Dataset:
    """Build the complete versioned Level 2 dataset for one Level 1 input."""
    time_values = ds_l1["time"].values
    if not isinstance(product_contract, Level2ProductContract):
        raise TypeError("product_contract must be Level2ProductContract.")
    validate_retrieval_results(results, n_time=len(time_values), n_altitude=len(altitude_m))
    product_contract.validate_results(results)
    results = sorted(results, key=lambda result: int(result.wavelength_nm))
    wavelengths = np.asarray([result.wavelength_nm for result in results], dtype=np.int32)
    block_time = results[0].block_time
    coords = {
        "time": time_values,
        "block_time": block_time,
        "wavelength": wavelengths,
        "altitude": altitude_m,
    }

    def stack(selector: Callable[[WavelengthRetrievalResult], np.ndarray]) -> np.ndarray:
        return np.stack(
            [np.asarray(selector(result), dtype=np.float64) for result in results], axis=0
        )

    def stack_time(selector: Callable[[WavelengthRetrievalResult], np.ndarray]) -> np.ndarray:
        return np.stack(
            [np.asarray(selector(result), dtype=np.float64) for result in results], axis=1
        )

    def stack_block(selector: Callable[[WavelengthRetrievalResult], np.ndarray]) -> np.ndarray:
        return np.stack(
            [np.asarray(selector(result), dtype=np.float64) for result in results], axis=1
        )

    def vector(selector: Callable[[WavelengthRetrievalResult], float | int]) -> np.ndarray:
        return np.asarray([selector(result) for result in results], dtype=np.float64)

    kfs_mode = get_kfs_mode(config)
    ds_l2 = xr.Dataset(
        data_vars={
            "molecular_backscatter": (
                ("wavelength", "altitude"),
                stack(lambda result: result.molecular.backscatter),
            ),
            "molecular_extinction": (
                ("wavelength", "altitude"),
                stack(lambda result: result.molecular.extinction),
            ),
            "molecular_transmission": (
                ("wavelength", "altitude"),
                stack(lambda result: result.molecular.transmission),
            ),
            "simulated_molecular_signal": (
                ("wavelength", "altitude"),
                stack(lambda result: result.molecular.simulated_signal),
            ),
            "simulated_molecular_range_corrected_signal": (
                ("wavelength", "altitude"),
                stack(lambda result: result.molecular.simulated_range_corrected_signal),
            ),
            "scaled_molecular_range_corrected_signal": (
                ("wavelength", "altitude"),
                stack(lambda result: result.molecular.scaled_range_corrected_signal),
            ),
            "scaled_molecular_range_corrected_signal_block": (
                ("block_time", "wavelength", "altitude"),
                stack_block(lambda result: result.molecular.scaled_range_corrected_signal_block),
            ),
            "glued_corrected_signal": (
                ("time", "wavelength", "altitude"),
                stack_time(lambda result: result.glued.corrected_signal),
            ),
            "glued_corrected_signal_error": (
                ("time", "wavelength", "altitude"),
                stack_time(lambda result: result.glued.corrected_signal_error),
            ),
            "glued_corrected_signal_block": (
                ("block_time", "wavelength", "altitude"),
                stack_block(lambda result: result.glued.corrected_signal_block),
            ),
            "glued_corrected_signal_error_block": (
                ("block_time", "wavelength", "altitude"),
                stack_block(lambda result: result.glued.corrected_signal_error_block),
            ),
            "glued_corrected_signal_mean": (
                ("wavelength", "altitude"),
                stack(lambda result: result.glued.corrected_signal_mean),
            ),
            "glued_corrected_signal_error_mean": (
                ("wavelength", "altitude"),
                stack(lambda result: result.glued.corrected_signal_error_mean),
            ),
            "glued_range_corrected_signal": (
                ("time", "wavelength", "altitude"),
                stack_time(lambda result: result.glued.range_corrected_signal),
            ),
            "glued_range_corrected_signal_error": (
                ("time", "wavelength", "altitude"),
                stack_time(lambda result: result.glued.range_corrected_signal_error),
            ),
            "glued_range_corrected_signal_block": (
                ("block_time", "wavelength", "altitude"),
                stack_block(lambda result: result.glued.range_corrected_signal_block),
            ),
            "glued_range_corrected_signal_error_block": (
                ("block_time", "wavelength", "altitude"),
                stack_block(lambda result: result.glued.range_corrected_signal_error_block),
            ),
            "glued_range_corrected_signal_mean": (
                ("wavelength", "altitude"),
                stack(lambda result: result.glued.range_corrected_signal_mean),
            ),
            "glued_range_corrected_signal_error_mean": (
                ("wavelength", "altitude"),
                stack(lambda result: result.glued.range_corrected_signal_error_mean),
            ),
            "gluing_merge_source_flag": (
                ("time", "wavelength", "altitude"),
                stack_time(lambda result: result.glued.merge_source_flag).astype(np.int8),
            ),
            "gluing_merge_source_flag_block": (
                ("block_time", "wavelength", "altitude"),
                stack_block(lambda result: result.glued.merge_source_flag_block).astype(np.int8),
            ),
            "scattering_ratio_mean": (
                ("wavelength", "altitude"),
                stack(lambda result: result.optical.scattering_ratio_mean),
            ),
            "scattering_ratio_block": (
                ("block_time", "wavelength", "altitude"),
                stack_block(lambda result: result.optical.scattering_ratio_block),
            ),
            "aerosol_backscatter_mean": (
                ("wavelength", "altitude"),
                stack(lambda result: result.optical.aerosol_backscatter),
            ),
            "aerosol_backscatter_mean_error": (
                ("wavelength", "altitude"),
                stack(lambda result: result.optical.aerosol_backscatter_error),
            ),
            "aerosol_extinction_mean": (
                ("wavelength", "altitude"),
                stack(lambda result: result.optical.aerosol_extinction),
            ),
            "aerosol_extinction_mean_error": (
                ("wavelength", "altitude"),
                stack(lambda result: result.optical.aerosol_extinction_error),
            ),
            "aerosol_backscatter_block": (
                ("block_time", "wavelength", "altitude"),
                stack_block(lambda result: result.optical.aerosol_backscatter_block),
            ),
            "aerosol_backscatter_error_block": (
                ("block_time", "wavelength", "altitude"),
                stack_block(lambda result: result.optical.aerosol_backscatter_error_block),
            ),
            "aerosol_extinction_block": (
                ("block_time", "wavelength", "altitude"),
                stack_block(lambda result: result.optical.aerosol_extinction_block),
            ),
            "aerosol_extinction_error_block": (
                ("block_time", "wavelength", "altitude"),
                stack_block(lambda result: result.optical.aerosol_extinction_error_block),
            ),
            "retrieval_success_flag": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.optical.retrieval_success_flag).astype(np.int8),
            ),
            "retrieval_success_fraction": (
                ("wavelength",),
                vector(lambda result: float(np.mean(result.optical.retrieval_success_flag == 1))),
            ),
            "rayleigh_reference_altitude_m": (
                ("wavelength",),
                vector(lambda result: result.rayleigh.reference_altitude_m),
            ),
            "rayleigh_reference_start_altitude_m": (
                ("wavelength",),
                vector(lambda result: result.rayleigh.reference_start_altitude_m),
            ),
            "rayleigh_reference_stop_altitude_m": (
                ("wavelength",),
                vector(lambda result: result.rayleigh.reference_stop_altitude_m),
            ),
            "rayleigh_reference_valid_bins": (
                ("wavelength",),
                vector(lambda result: result.rayleigh.reference_valid_bins),
            ),
            "rayleigh_reference_success_flag": (
                ("wavelength",),
                vector(lambda result: result.rayleigh.reference_success_flag).astype(np.int8),
            ),
            "rayleigh_reference_relative_slope": (
                ("wavelength",),
                vector(lambda result: result.rayleigh.reference_relative_slope),
            ),
            "rayleigh_reference_relative_variance": (
                ("wavelength",),
                vector(lambda result: result.rayleigh.reference_relative_variance),
            ),
            "rayleigh_reference_valid_fraction": (
                ("wavelength",),
                vector(lambda result: result.rayleigh.reference_valid_fraction),
            ),
            "rayleigh_calibration_factor": (
                ("wavelength",),
                vector(lambda result: result.rayleigh.calibration_factor),
            ),
            "rayleigh_calibration_intercept": (
                ("wavelength",),
                vector(lambda result: result.rayleigh.calibration_intercept),
            ),
            "rayleigh_reference_altitude_m_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.rayleigh.reference_altitude_m_block),
            ),
            "rayleigh_reference_start_altitude_m_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.rayleigh.reference_start_altitude_m_block),
            ),
            "rayleigh_reference_stop_altitude_m_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.rayleigh.reference_stop_altitude_m_block),
            ),
            "rayleigh_reference_valid_bins_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.rayleigh.reference_valid_bins_block),
            ),
            "rayleigh_reference_success_flag_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.rayleigh.reference_success_flag_block).astype(np.int8),
            ),
            "rayleigh_reference_relative_slope_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.rayleigh.reference_relative_slope_block),
            ),
            "rayleigh_reference_relative_variance_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.rayleigh.reference_relative_variance_block),
            ),
            "rayleigh_reference_valid_fraction_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.rayleigh.reference_valid_fraction_block),
            ),
            "rayleigh_calibration_factor_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.rayleigh.calibration_factor_block),
            ),
            "rayleigh_calibration_intercept_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.rayleigh.calibration_intercept_block),
            ),
            "lidar_ratio_assumed_sr": (
                ("wavelength",),
                vector(lambda result: result.kfs.lidar_ratio_assumed_sr),
            ),
            "lidar_ratio_std_sr": (
                ("wavelength",),
                vector(lambda result: result.kfs.lidar_ratio_std_sr),
            ),
            "kfs_backward_valid_flag": (
                ("wavelength",),
                vector(lambda result: result.kfs.backward_valid_flag).astype(np.int8),
            ),
            "kfs_forward_valid_flag": (
                ("wavelength",),
                vector(lambda result: result.kfs.forward_valid_flag).astype(np.int8),
            ),
            "kfs_backward_valid_flag_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.kfs.backward_valid_flag_block).astype(np.int8),
            ),
            "kfs_forward_valid_flag_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.kfs.forward_valid_flag_block).astype(np.int8),
            ),
            "kfs_branch": (
                ("wavelength", "altitude"),
                stack(lambda result: result.kfs.branch).astype(np.int8),
            ),
            "kfs_branch_block": (
                ("block_time", "wavelength", "altitude"),
                stack_block(lambda result: result.kfs.branch_block).astype(np.int8),
            ),
            "gluing_attempted_flag": (
                ("time", "wavelength"),
                stack_time(lambda result: result.gluing.attempted_flag).astype(np.int8),
            ),
            "gluing_success_flag": (
                ("time", "wavelength"),
                stack_time(lambda result: result.gluing.success_flag).astype(np.int8),
            ),
            "single_channel_fallback_flag": (
                ("time", "wavelength"),
                stack_time(lambda result: result.gluing.single_channel_fallback_flag).astype(np.int8),
            ),
            "gluing_split_altitude_m": (
                ("time", "wavelength"),
                stack_time(lambda result: result.gluing.split_altitude_m),
            ),
            "gluing_start_altitude_m": (
                ("time", "wavelength"),
                stack_time(lambda result: result.gluing.start_altitude_m),
            ),
            "gluing_stop_altitude_m": (
                ("time", "wavelength"),
                stack_time(lambda result: result.gluing.stop_altitude_m),
            ),
            "gluing_slope": (
                ("time", "wavelength"),
                stack_time(lambda result: result.gluing.slope),
            ),
            "gluing_intercept": (
                ("time", "wavelength"),
                stack_time(lambda result: result.gluing.intercept),
            ),
            "gluing_correlation": (
                ("time", "wavelength"),
                stack_time(lambda result: result.gluing.correlation),
            ),
            "gluing_relative_rmse": (
                ("time", "wavelength"),
                stack_time(lambda result: result.gluing.relative_rmse),
            ),
            "gluing_relative_bias": (
                ("time", "wavelength"),
                stack_time(lambda result: result.gluing.relative_bias),
            ),
            "gluing_attempted_flag_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.gluing.attempted_flag_block).astype(np.int8),
            ),
            "gluing_success_flag_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.gluing.success_flag_block).astype(np.int8),
            ),
            "single_channel_fallback_flag_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.gluing.single_channel_fallback_flag_block).astype(np.int8),
            ),
            "gluing_split_altitude_m_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.gluing.split_altitude_m_block),
            ),
            "gluing_start_altitude_m_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.gluing.start_altitude_m_block),
            ),
            "gluing_stop_altitude_m_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.gluing.stop_altitude_m_block),
            ),
            "gluing_slope_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.gluing.slope_block),
            ),
            "gluing_intercept_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.gluing.intercept_block),
            ),
            "gluing_correlation_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.gluing.correlation_block),
            ),
            "gluing_relative_rmse_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.gluing.relative_rmse_block),
            ),
            "gluing_relative_bias_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.gluing.relative_bias_block),
            ),
            "signal_source_flag": (
                ("time", "wavelength"),
                stack_time(lambda result: result.signal_selection.source_flag).astype(np.int8),
            ),
            "retrieval_input_valid_flag": (
                ("time", "wavelength"),
                stack_time(lambda result: result.signal_selection.retrieval_input_valid_flag).astype(np.int8),
            ),
            "retrieval_input_invalid_reason": (
                ("time", "wavelength"),
                stack_time(lambda result: result.signal_selection.retrieval_input_invalid_reason).astype(np.int8),
            ),
            "retrieval_input_snr_median": (
                ("time", "wavelength"),
                stack_time(lambda result: result.signal_selection.retrieval_input_snr_median),
            ),
            "signal_source_flag_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.signal_selection.source_flag_block).astype(np.int8),
            ),
            "retrieval_input_valid_flag_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.signal_selection.retrieval_input_valid_flag_block).astype(np.int8),
            ),
            "retrieval_input_invalid_reason_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.signal_selection.retrieval_input_invalid_reason_block).astype(np.int8),
            ),
            "retrieval_input_snr_median_block": (
                ("block_time", "wavelength"),
                stack_block(lambda result: result.signal_selection.retrieval_input_snr_median_block),
            ),
            "requested_wavelengths": (
                ("requested_wavelength",),
                np.asarray(product_contract.requested_wavelengths, dtype=np.int32),
            ),
            "processed_wavelengths": (
                ("processed_wavelength",),
                np.asarray(product_contract.processed_wavelengths, dtype=np.int32),
            ),
            "failed_wavelengths": (
                ("failed_wavelength",),
                np.asarray(product_contract.failed_wavelengths, dtype=np.int32),
            ),
            "failed_wavelength_stage": (
                ("failed_wavelength",),
                np.asarray(
                    [int(item.stage) for item in product_contract.failure_diagnostics],
                    dtype=np.int8,
                ),
            ),
            "failed_wavelength_code": (
                ("failed_wavelength",),
                np.asarray(
                    [int(item.code) for item in product_contract.failure_diagnostics],
                    dtype=np.int16,
                ),
            ),
            "failed_wavelength_message": (
                ("failed_wavelength",),
                np.asarray(
                    [item.message for item in product_contract.failure_diagnostics],
                    dtype=str,
                ),
            ),
            "failed_wavelength_cause": (
                ("failed_wavelength",),
                np.asarray(
                    [item.cause_summary for item in product_contract.failure_diagnostics],
                    dtype=str,
                ),
            ),
        },
        coords=coords,
        attrs=dict(ds_l1.attrs),
    )

    fit_cfg = get_molecular_fit_config(config)
    ds_l2.attrs.update(
        {
            "Processing_level": "Level 2: LEBEAR block-based optical inversion",
            "Pipeline": "MILGRAU/LEBEAR",
            "Input_Level1_File": source_file.name,
            "level2_product_schema_version": LEVEL2_PRODUCT_SCHEMA_VERSION,
            "LEBEAR_Mode": "block_mean_signal_selection_rayleigh_kfs",
            "LEBEAR_Block_Average_Minutes": get_block_average_minutes(config),
            "KFS_Mode": kfs_mode,
            "KFS_Mode_Description": kfs_mode_description(kfs_mode),
            "Molecular_Rayleigh_Method": "Bucholtz-style Rayleigh scattering with angular backscatter at 180 degrees.",
            "Rayleigh_Calibration_Method": "Block-wise multiplicative fit constrained through origin; free intercept retained as a background diagnostic.",
            "Gluing_Method": "Analog/PC gluing on Level 1 corrected_signal before range correction; glued RCS is produced afterward by multiplying by range squared.",
            "Gluing_Error_Propagation": "Weighted one-sigma propagation across fade window: sigma² = w_an²(slope sigma_an)² + w_pc² sigma_pc², followed by range-squared scaling.",
            "Signal_Selection_Policy": "Use approved glued signal; otherwise assess PC and AN independently and select one valid channel by configured priority without mixing.",
            "Single_Channel_Priority": str(
                config.get("inversion", {})
                .get("gluing", {})
                .get("single_channel_priority", "photon_counting")
            ),
            "Single_Channel_QA": "Require at least one viable Rayleigh-sized window with finite positive corrected signal, finite non-negative one-sigma uncertainty, calculable SNR, successful Level 1 correction when exposed, and zero PC saturation where diagnosed.",
            "Rayleigh_Reference_Max_Relative_Slope": float(fit_cfg["max_relative_slope"]),
            "Rayleigh_Reference_Max_Relative_Variance": float(
                fit_cfg["max_relative_variance"]
            ),
            "Rayleigh_Reference_Min_Valid_Fraction": float(fit_cfg["min_valid_fraction"]),
            "Molecular_sources": ";".join(result.molecular.source for result in results),
            "Gluing_sources": ";".join(result.glued.source for result in results),
            "Analog_channels": ";".join(
                str(result.glued.analog_channel) for result in results
            ),
            "Photon_channels": ";".join(
                str(result.glued.photon_channel) for result in results
            ),
            "product_completeness": product_contract.completeness.value,
            "product_status": product_contract.product_status.value,
            "Wavelength_Order": "Ascending numeric order; scientific wavelength equals processed_wavelengths exactly.",
            "Partial_Product_Reuse": "Partial products are never incrementally reusable; the next run recalculates every requested wavelength.",
            "uncertainty_scope": "partial Monte Carlo dispersion; not a total uncertainty budget",
            "scientific_reprocessing_required": "Level 2 products without schema version 1, with productive KFS metadata other than backward, or with Fernald implementation versions before 2 must be reprocessed.",
            **elastic_inversion_algorithm_metadata(),
        }
    )
    apply_level2_variable_metadata(ds_l2)
    return ds_l2
