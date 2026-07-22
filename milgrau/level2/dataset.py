"""Assembly of Level 2 xarray datasets from retrieval results."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Mapping

import numpy as np
import xarray as xr

from milgrau.level2.config import (
    get_block_average_minutes,
    get_kfs_mode,
    get_molecular_fit_config,
    kfs_mode_description,
)
from milgrau.level2.contracts import WavelengthRetrievalResult, validate_retrieval_results


def build_level2_dataset(
    ds_l1: xr.Dataset,
    results: list[WavelengthRetrievalResult],
    altitude_m: np.ndarray,
    source_file: Path,
    config: Mapping[str, Any],
) -> xr.Dataset:
    """Build an xarray Level 2 dataset from wavelength processing results."""
    time_values = ds_l1["time"].values
    validate_retrieval_results(results, n_time=len(time_values), n_altitude=len(altitude_m))
    wavelengths = np.asarray([result.wavelength_nm for result in results], dtype=np.int32)
    block_time = results[0].block_time
    coords = {"time": time_values, "block_time": block_time, "wavelength": wavelengths, "altitude": altitude_m}

    def stack(selector: Callable[[WavelengthRetrievalResult], np.ndarray]) -> np.ndarray:
        return np.stack([np.asarray(selector(result), dtype=np.float64) for result in results], axis=0)

    def stack_time(selector: Callable[[WavelengthRetrievalResult], np.ndarray]) -> np.ndarray:
        return np.stack([np.asarray(selector(result), dtype=np.float64) for result in results], axis=1)

    def stack_block(selector: Callable[[WavelengthRetrievalResult], np.ndarray]) -> np.ndarray:
        return np.stack([np.asarray(selector(result), dtype=np.float64) for result in results], axis=1)

    def vector(selector: Callable[[WavelengthRetrievalResult], float | int]) -> np.ndarray:
        return np.asarray([selector(result) for result in results], dtype=np.float64)

    kfs_mode = get_kfs_mode(config)
    ds_l2 = xr.Dataset(
        data_vars={
            "molecular_backscatter": (("wavelength", "altitude"), stack(lambda result: result.molecular.backscatter)),
            "molecular_extinction": (("wavelength", "altitude"), stack(lambda result: result.molecular.extinction)),
            "molecular_transmission": (("wavelength", "altitude"), stack(lambda result: result.molecular.transmission)),
            "simulated_molecular_signal": (("wavelength", "altitude"), stack(lambda result: result.molecular.simulated_signal)),
            "simulated_molecular_range_corrected_signal": (("wavelength", "altitude"), stack(lambda result: result.molecular.simulated_range_corrected_signal)),
            "scaled_molecular_range_corrected_signal": (("wavelength", "altitude"), stack(lambda result: result.molecular.scaled_range_corrected_signal)),
            "scaled_molecular_range_corrected_signal_block": (("block_time", "wavelength", "altitude"), stack_block(lambda result: result.molecular.scaled_range_corrected_signal_block)),
            "glued_corrected_signal": (("time", "wavelength", "altitude"), stack_time(lambda result: result.glued.corrected_signal)),
            "glued_corrected_signal_error": (("time", "wavelength", "altitude"), stack_time(lambda result: result.glued.corrected_signal_error)),
            "glued_corrected_signal_block": (("block_time", "wavelength", "altitude"), stack_block(lambda result: result.glued.corrected_signal_block)),
            "glued_corrected_signal_error_block": (("block_time", "wavelength", "altitude"), stack_block(lambda result: result.glued.corrected_signal_error_block)),
            "glued_corrected_signal_mean": (("wavelength", "altitude"), stack(lambda result: result.glued.corrected_signal_mean)),
            "glued_corrected_signal_error_mean": (("wavelength", "altitude"), stack(lambda result: result.glued.corrected_signal_error_mean)),
            "glued_range_corrected_signal": (("time", "wavelength", "altitude"), stack_time(lambda result: result.glued.range_corrected_signal)),
            "glued_range_corrected_signal_error": (("time", "wavelength", "altitude"), stack_time(lambda result: result.glued.range_corrected_signal_error)),
            "glued_range_corrected_signal_block": (("block_time", "wavelength", "altitude"), stack_block(lambda result: result.glued.range_corrected_signal_block)),
            "glued_range_corrected_signal_error_block": (("block_time", "wavelength", "altitude"), stack_block(lambda result: result.glued.range_corrected_signal_error_block)),
            "glued_range_corrected_signal_mean": (("wavelength", "altitude"), stack(lambda result: result.glued.range_corrected_signal_mean)),
            "glued_range_corrected_signal_error_mean": (("wavelength", "altitude"), stack(lambda result: result.glued.range_corrected_signal_error_mean)),
            "gluing_merge_source_flag": (("time", "wavelength", "altitude"), stack_time(lambda result: result.glued.merge_source_flag).astype(np.int8)),
            "gluing_merge_source_flag_block": (("block_time", "wavelength", "altitude"), stack_block(lambda result: result.glued.merge_source_flag_block).astype(np.int8)),
            "scattering_ratio_mean": (("wavelength", "altitude"), stack(lambda result: result.optical.scattering_ratio_mean)),
            "scattering_ratio_block": (("block_time", "wavelength", "altitude"), stack_block(lambda result: result.optical.scattering_ratio_block)),
            "aerosol_backscatter_mean": (("wavelength", "altitude"), stack(lambda result: result.optical.aerosol_backscatter)),
            "aerosol_backscatter_mean_error": (("wavelength", "altitude"), stack(lambda result: result.optical.aerosol_backscatter_error)),
            "aerosol_extinction_mean": (("wavelength", "altitude"), stack(lambda result: result.optical.aerosol_extinction)),
            "aerosol_extinction_mean_error": (("wavelength", "altitude"), stack(lambda result: result.optical.aerosol_extinction_error)),
            "aerosol_backscatter": (("wavelength", "altitude"), stack(lambda result: result.optical.aerosol_backscatter)),
            "aerosol_backscatter_error": (("wavelength", "altitude"), stack(lambda result: result.optical.aerosol_backscatter_error)),
            "aerosol_extinction": (("wavelength", "altitude"), stack(lambda result: result.optical.aerosol_extinction)),
            "aerosol_extinction_error": (("wavelength", "altitude"), stack(lambda result: result.optical.aerosol_extinction_error)),
            "aerosol_backscatter_block": (("block_time", "wavelength", "altitude"), stack_block(lambda result: result.optical.aerosol_backscatter_block)),
            "aerosol_backscatter_error_block": (("block_time", "wavelength", "altitude"), stack_block(lambda result: result.optical.aerosol_backscatter_error_block)),
            "aerosol_extinction_block": (("block_time", "wavelength", "altitude"), stack_block(lambda result: result.optical.aerosol_extinction_block)),
            "aerosol_extinction_error_block": (("block_time", "wavelength", "altitude"), stack_block(lambda result: result.optical.aerosol_extinction_error_block)),
            "valid_retrieval_block_flag": (("block_time", "wavelength"), stack_block(lambda result: result.optical.valid_retrieval_block_flag).astype(np.int8)),
            "rayleigh_reference_altitude_m": (("wavelength",), vector(lambda result: result.rayleigh.reference_altitude_m)),
            "rayleigh_reference_start_altitude_m": (("wavelength",), vector(lambda result: result.rayleigh.reference_start_altitude_m)),
            "rayleigh_reference_stop_altitude_m": (("wavelength",), vector(lambda result: result.rayleigh.reference_stop_altitude_m)),
            "rayleigh_reference_valid_bins": (("wavelength",), vector(lambda result: result.rayleigh.reference_valid_bins)),
            "rayleigh_reference_success_flag": (("wavelength",), vector(lambda result: result.rayleigh.reference_success_flag).astype(np.int8)),
            "rayleigh_reference_relative_slope": (("wavelength",), vector(lambda result: result.rayleigh.reference_relative_slope)),
            "rayleigh_reference_relative_variance": (("wavelength",), vector(lambda result: result.rayleigh.reference_relative_variance)),
            "rayleigh_reference_valid_fraction": (("wavelength",), vector(lambda result: result.rayleigh.reference_valid_fraction)),
            "rayleigh_calibration_factor": (("wavelength",), vector(lambda result: result.rayleigh.calibration_factor)),
            "rayleigh_calibration_intercept": (("wavelength",), vector(lambda result: result.rayleigh.calibration_intercept)),
            "rayleigh_reference_altitude_m_block": (("block_time", "wavelength"), stack_block(lambda result: result.rayleigh.reference_altitude_m_block)),
            "rayleigh_reference_start_altitude_m_block": (("block_time", "wavelength"), stack_block(lambda result: result.rayleigh.reference_start_altitude_m_block)),
            "rayleigh_reference_stop_altitude_m_block": (("block_time", "wavelength"), stack_block(lambda result: result.rayleigh.reference_stop_altitude_m_block)),
            "rayleigh_reference_valid_bins_block": (("block_time", "wavelength"), stack_block(lambda result: result.rayleigh.reference_valid_bins_block)),
            "rayleigh_reference_success_flag_block": (("block_time", "wavelength"), stack_block(lambda result: result.rayleigh.reference_success_flag_block).astype(np.int8)),
            "rayleigh_reference_relative_slope_block": (("block_time", "wavelength"), stack_block(lambda result: result.rayleigh.reference_relative_slope_block)),
            "rayleigh_reference_relative_variance_block": (("block_time", "wavelength"), stack_block(lambda result: result.rayleigh.reference_relative_variance_block)),
            "rayleigh_reference_valid_fraction_block": (("block_time", "wavelength"), stack_block(lambda result: result.rayleigh.reference_valid_fraction_block)),
            "rayleigh_calibration_factor_block": (("block_time", "wavelength"), stack_block(lambda result: result.rayleigh.calibration_factor_block)),
            "rayleigh_calibration_intercept_block": (("block_time", "wavelength"), stack_block(lambda result: result.rayleigh.calibration_intercept_block)),
            "lidar_ratio_assumed_sr": (("wavelength",), vector(lambda result: result.kfs.lidar_ratio_assumed_sr)),
            "lidar_ratio_std_sr": (("wavelength",), vector(lambda result: result.kfs.lidar_ratio_std_sr)),
            "kfs_branch": (("wavelength", "altitude"), stack(lambda result: result.kfs.branch).astype(np.int8)),
            "kfs_branch_block": (("block_time", "wavelength", "altitude"), stack_block(lambda result: result.kfs.branch_block).astype(np.int8)),
            "gluing_success_flag": (("time", "wavelength"), stack_time(lambda result: result.gluing.success_flag).astype(np.int8)),
            "gluing_fallback_flag": (("time", "wavelength"), stack_time(lambda result: result.gluing.fallback_flag).astype(np.int8)),
            "gluing_split_altitude_m": (("time", "wavelength"), stack_time(lambda result: result.gluing.split_altitude_m)),
            "gluing_start_altitude_m": (("time", "wavelength"), stack_time(lambda result: result.gluing.start_altitude_m)),
            "gluing_stop_altitude_m": (("time", "wavelength"), stack_time(lambda result: result.gluing.stop_altitude_m)),
            "gluing_slope": (("time", "wavelength"), stack_time(lambda result: result.gluing.slope)),
            "gluing_intercept": (("time", "wavelength"), stack_time(lambda result: result.gluing.intercept)),
            "gluing_correlation": (("time", "wavelength"), stack_time(lambda result: result.gluing.correlation)),
            "gluing_relative_rmse": (("time", "wavelength"), stack_time(lambda result: result.gluing.relative_rmse)),
            "gluing_relative_bias": (("time", "wavelength"), stack_time(lambda result: result.gluing.relative_bias)),
            "gluing_success_flag_block": (("block_time", "wavelength"), stack_block(lambda result: result.gluing.success_flag_block).astype(np.int8)),
            "gluing_fallback_flag_block": (("block_time", "wavelength"), stack_block(lambda result: result.gluing.fallback_flag_block).astype(np.int8)),
            "gluing_split_altitude_m_block": (("block_time", "wavelength"), stack_block(lambda result: result.gluing.split_altitude_m_block)),
            "gluing_start_altitude_m_block": (("block_time", "wavelength"), stack_block(lambda result: result.gluing.start_altitude_m_block)),
            "gluing_stop_altitude_m_block": (("block_time", "wavelength"), stack_block(lambda result: result.gluing.stop_altitude_m_block)),
            "gluing_slope_block": (("block_time", "wavelength"), stack_block(lambda result: result.gluing.slope_block)),
            "gluing_intercept_block": (("block_time", "wavelength"), stack_block(lambda result: result.gluing.intercept_block)),
            "gluing_correlation_block": (("block_time", "wavelength"), stack_block(lambda result: result.gluing.correlation_block)),
            "gluing_relative_rmse_block": (("block_time", "wavelength"), stack_block(lambda result: result.gluing.relative_rmse_block)),
            "gluing_relative_bias_block": (("block_time", "wavelength"), stack_block(lambda result: result.gluing.relative_bias_block)),
        },
        coords=coords,
        attrs=dict(ds_l1.attrs),
    )
    ds_l2["altitude"].attrs.update({"units": "m", "long_name": "Altitude above station"})
    ds_l2["wavelength"].attrs.update({"units": "nm"})
    ds_l2["glued_corrected_signal"].attrs.update({"description": "Analog/PC merged Level 1 corrected signal before range correction."})
    ds_l2["glued_range_corrected_signal"].attrs.update({"description": "Range-corrected signal computed after gluing corrected_signal."})
    ds_l2["gluing_merge_source_flag"].attrs.update({"flag_values": "0, 1, 2, 3", "flag_meanings": "photon_counting blend analog invalid", "description": "Per-bin source used by the corrected-signal gluing step."})
    ds_l2["valid_retrieval_block_flag"].attrs.update({"flag_values": "0, 1", "flag_meanings": "invalid valid", "description": "Block passed both gluing and Rayleigh-reference QA and was used in mean optical products."})
    ds_l2["scattering_ratio_mean"].attrs.update({"units": "1", "description": "Mean of valid block scattering ratios."})
    ds_l2["scattering_ratio_block"].attrs.update({"units": "1", "description": "Block scattering ratio from block-mean glued RCS and block-scaled molecular RCS."})
    ds_l2["rayleigh_calibration_intercept"].attrs.update({"description": "Median intercept from free linear Rayleigh diagnostic fit. The main calibration factor is constrained through the origin."})
    ds_l2["kfs_branch"].attrs.update({"flag_values": "0, 1, 2, 3", "flag_meanings": "invalid backward_below_reference reference_window forward_above_reference_experimental", "description": "KFS/Fernald-Sasano retrieval branch by altitude. Above-reference two-sided retrieval is experimental and noise-sensitive."})
    ds_l2["gluing_success_flag"].attrs.update({"flag_values": "0, 1", "flag_meanings": "failed success"})
    ds_l2["gluing_fallback_flag"].attrs.update({"flag_values": "0, 1", "flag_meanings": "not_used photon_counting_fallback_used"})
    ds_l2["gluing_success_flag_block"].attrs.update({"flag_values": "0, 1", "flag_meanings": "failed success"})
    ds_l2["gluing_fallback_flag_block"].attrs.update({"flag_values": "0, 1", "flag_meanings": "not_used photon_counting_fallback_used"})
    ds_l2["gluing_start_altitude_m"].attrs.update({"units": "m", "description": "Start altitude of analog/photon-counting fade-in/fade-out gluing window."})
    ds_l2["gluing_stop_altitude_m"].attrs.update({"units": "m", "description": "Stop altitude of analog/photon-counting fade-in/fade-out gluing window."})
    ds_l2["rayleigh_reference_success_flag"].attrs.update({"flag_values": "0, 1", "flag_meanings": "failed passed", "description": "Whether at least one block Rayleigh reference passed QA."})
    ds_l2["rayleigh_reference_success_flag_block"].attrs.update({"flag_values": "0, 1", "flag_meanings": "failed passed", "description": "Whether the block Rayleigh reference passed slope, variance, valid-fraction and positive-calibration checks."})
    ds_l2["rayleigh_reference_relative_slope"].attrs.update({"units": "1", "description": "Median relative ratio change across valid block Rayleigh reference windows."})
    ds_l2["rayleigh_reference_relative_variance"].attrs.update({"units": "1", "description": "Median variance of measured/molecular ratio normalized by squared mean ratio."})
    ds_l2["rayleigh_reference_valid_fraction"].attrs.update({"units": "1", "description": "Median fraction of finite positive bins in valid Rayleigh reference windows."})
    fit_cfg = get_molecular_fit_config(config)
    ds_l2.attrs.update(
        {
            "Processing_level": "Level 2: LEBEAR block-based optical inversion",
            "Pipeline": "MILGRAU/LEBEAR",
            "Input_Level1_File": source_file.name,
            "LEBEAR_Mode": "block_mean_corrected_signal_gluing_rayleigh_kfs",
            "LEBEAR_Block_Average_Minutes": get_block_average_minutes(config),
            "KFS_Mode": kfs_mode,
            "KFS_Mode_Description": kfs_mode_description(kfs_mode),
            "Molecular_Rayleigh_Method": "Bucholtz-style Rayleigh scattering with angular backscatter at 180 degrees.",
            "Rayleigh_Calibration_Method": "Block-wise multiplicative fit constrained through origin; free intercept retained as a background diagnostic.",
            "Gluing_Method": "Analog/PC gluing on Level 1 corrected_signal before range correction; glued RCS is produced afterward by multiplying by range squared.",
            "Gluing_Error_Propagation": "Weighted one-sigma propagation across fade window: sigma² = w_an²(slope sigma_an)² + w_pc² sigma_pc², followed by range-squared scaling.",
            "Rayleigh_Reference_Max_Relative_Slope": float(fit_cfg["max_relative_slope"]),
            "Rayleigh_Reference_Max_Relative_Variance": float(fit_cfg["max_relative_variance"]),
            "Rayleigh_Reference_Min_Valid_Fraction": float(fit_cfg["min_valid_fraction"]),
            "Molecular_sources": ";".join(result.molecular.source for result in results),
            "Gluing_sources": ";".join(result.glued.source for result in results),
            "Analog_channels": ";".join(str(result.glued.analog_channel) for result in results),
            "Photon_channels": ";".join(str(result.glued.photon_channel) for result in results),
        }
    )
    return ds_l2
