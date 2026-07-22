"""Core Level 2 retrieval routines for one wavelength."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Mapping, TypeVar

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
from milgrau.level2.contracts import (
    GluedSignals,
    GluingDiagnostics,
    KfsDiagnostics,
    MolecularProfiles,
    OpticalProducts,
    RayleighDiagnostics,
    WavelengthRetrievalResult,
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
KFS_BRANCH_REFERENCE_BIN = 2
KFS_BRANCH_FORWARD_ABOVE_REFERENCE = 3
_StageResult = TypeVar("_StageResult")


class RetrievalStageError(RuntimeError):
    """Identify the retrieval stage that raised an underlying exception."""

    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = stage
        super().__init__(f"[{stage}] {cause}")


def _run_retrieval_stage(stage: str, operation: Callable[[], _StageResult]) -> _StageResult:
    """Run one retrieval stage and attach its stable name to any failure."""
    try:
        return operation()
    except RetrievalStageError:
        raise
    except Exception as exc:
        raise RetrievalStageError(stage, exc) from exc


def build_kfs_branch(altitude_m: np.ndarray, reference_index: int, mode: str) -> np.ndarray:
    """Build branch flags around the one boundary bin used by the inversion."""
    altitude = np.asarray(altitude_m, dtype=np.float64)
    branch = np.zeros(altitude.size, dtype=np.int8)
    finite = np.isfinite(altitude)
    reference_index = int(reference_index)
    if altitude.ndim != 1 or reference_index < 0 or reference_index >= altitude.size:
        raise ValueError("reference_index must identify one bin on the 1D altitude grid.")
    indices = np.arange(altitude.size)
    branch[finite & (indices < reference_index)] = KFS_BRANCH_BACKWARD_BELOW_REFERENCE
    if finite[reference_index]:
        branch[reference_index] = KFS_BRANCH_REFERENCE_BIN
    if mode == "two_sided":
        branch[finite & (indices > reference_index)] = KFS_BRANCH_FORWARD_ABOVE_REFERENCE
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
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
        return_diagnostics=True,
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


@dataclass(frozen=True, slots=True)
class WavelengthBlockInputs:
    """Selected channels and their block-averaged Level 1 inputs."""

    wavelength_nm: int
    analog_channel: str | None
    photon_channel: str | None
    n_time: int
    n_altitude: int
    block_time: np.ndarray
    block_groups: list[np.ndarray]
    gluing_config: dict[str, Any]
    range_factor: np.ndarray
    analog_block: np.ndarray | None
    analog_error_block: np.ndarray | None
    photon_block: np.ndarray | None
    photon_error_block: np.ndarray | None
    photon_mask_block: np.ndarray | None


@dataclass(frozen=True, slots=True)
class BlockGluingResult:
    """Per-block glued signals and numerical gluing diagnostics."""

    source: str
    corrected_signal: np.ndarray
    corrected_signal_error: np.ndarray
    range_corrected_signal: np.ndarray
    range_corrected_signal_error: np.ndarray
    merge_source_flag: np.ndarray
    success_flag: np.ndarray
    fallback_flag: np.ndarray
    split_altitude_m: np.ndarray
    start_altitude_m: np.ndarray
    stop_altitude_m: np.ndarray
    slope: np.ndarray
    intercept: np.ndarray
    correlation: np.ndarray
    relative_rmse: np.ndarray
    relative_bias: np.ndarray


def prepare_wavelength_blocks(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
) -> WavelengthBlockInputs:
    """Select source channels and reduce Level 1 profiles into time blocks."""
    analog_channel, photon_channel = infer_channel_pair(ds_l1, wavelength_nm)
    if analog_channel is None and photon_channel is None:
        raise ValueError(f"No channel found for wavelength {wavelength_nm} nm.")

    corrected = ds_l1["corrected_signal"]
    corrected_error = ds_l1["corrected_signal_error"]
    n_time = ds_l1.sizes.get("time", 1)
    n_altitude = altitude_m.size
    block_time, groups = block_groups(ds_l1["time"].values, get_block_average_minutes(config))
    n_block = len(groups)

    if photon_channel is not None:
        photon_signal = corrected.sel(channel=photon_channel).values.astype(np.float64)
        photon_error = corrected_error.sel(channel=photon_channel).values.astype(np.float64)
        photon_block = mean_by_groups(photon_signal, groups)
        photon_error_block = error_by_groups(photon_error, groups)
        if "pc_saturation_mask" in ds_l1:
            photon_mask = ds_l1["pc_saturation_mask"].sel(channel=photon_channel).values.astype(bool)
            photon_mask_block = mask_by_groups(photon_mask, groups)
        else:
            photon_mask_block = np.zeros((n_block, n_altitude), dtype=bool)
    else:
        photon_block = None
        photon_error_block = None
        photon_mask_block = None

    if analog_channel is not None:
        analog_signal = corrected.sel(channel=analog_channel).values.astype(np.float64)
        analog_error = corrected_error.sel(channel=analog_channel).values.astype(np.float64)
        analog_block = mean_by_groups(analog_signal, groups)
        analog_error_block = error_by_groups(analog_error, groups)
    else:
        analog_block = None
        analog_error_block = None

    return WavelengthBlockInputs(
        wavelength_nm=wavelength_nm,
        analog_channel=analog_channel,
        photon_channel=photon_channel,
        n_time=n_time,
        n_altitude=n_altitude,
        block_time=block_time,
        block_groups=groups,
        gluing_config=get_gluing_config(config),
        range_factor=range_square_factor(altitude_m),
        analog_block=analog_block,
        analog_error_block=analog_error_block,
        photon_block=photon_block,
        photon_error_block=photon_error_block,
        photon_mask_block=photon_mask_block,
    )


def glue_signal_blocks(
    inputs: WavelengthBlockInputs,
    altitude_m: np.ndarray,
    logger: logging.Logger,
) -> BlockGluingResult:
    """Glue block-mean analog/PC signals or apply the configured fallback."""
    n_block = len(inputs.block_groups)
    n_altitude = inputs.n_altitude
    corrected = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    corrected_error = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    rcs = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    rcs_error = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    merge_source = np.full((n_block, n_altitude), 3, dtype=np.int8)
    success = np.zeros(n_block, dtype=np.int8)
    fallback = np.zeros(n_block, dtype=np.int8)
    split = np.full(n_block, np.nan, dtype=np.float64)
    start = np.full(n_block, np.nan, dtype=np.float64)
    stop = np.full(n_block, np.nan, dtype=np.float64)
    slope = np.full(n_block, np.nan, dtype=np.float64)
    intercept = np.full(n_block, np.nan, dtype=np.float64)
    correlation = np.full(n_block, np.nan, dtype=np.float64)
    relative_rmse = np.full(n_block, np.nan, dtype=np.float64)
    relative_bias = np.full(n_block, np.nan, dtype=np.float64)
    gluing_config = inputs.gluing_config

    if (
        inputs.analog_block is not None
        and inputs.photon_block is not None
        and inputs.analog_error_block is not None
        and inputs.photon_error_block is not None
    ):
        source = "block_mean_corrected_signal_analog_photon_glued"
        for block_index in range(n_block):
            glued_profile, split_point, slope_i, intercept_i, diagnostics = slide_glue_signals(
                analog_sig=inputs.analog_block[block_index, :],
                pc_sig=inputs.photon_block[block_index, :],
                altitude=altitude_m,
                window_size=gluing_config["window_size"],
                min_corr=gluing_config["min_corr"],
                search_min_idx=gluing_config["search_min_idx"],
                search_max_idx=gluing_config["search_max_idx"],
                intercept_threshold=gluing_config["intercept_threshold"],
                gaussian_threshold=gluing_config["gaussian_threshold"],
                minmax_threshold=gluing_config["minmax_threshold"],
                max_relative_rmse=gluing_config["max_relative_rmse"],
                max_relative_bias=gluing_config["max_relative_bias"],
                min_valid_fraction=gluing_config["min_valid_fraction"],
                max_saturation_fraction=gluing_config["max_saturation_fraction"],
                invalid_saturation_fraction=gluing_config.get("invalid_saturation_fraction", 1.0),
                pc_saturation_mask=(
                    inputs.photon_mask_block[block_index, :]
                    if inputs.photon_mask_block is not None
                    else None
                ),
                return_diagnostics=True,
            )
            slope[block_index] = slope_i
            intercept[block_index] = intercept_i
            correlation[block_index] = float(diagnostics.get("best_corr", np.nan))
            relative_rmse[block_index] = float(diagnostics.get("relative_rmse", np.nan))
            relative_bias[block_index] = float(diagnostics.get("relative_bias", np.nan))
            if split_point >= 0:
                min_bin = int(diagnostics.get("min_bin", max(split_point - gluing_config["window_size"] // 2, 0)))
                max_bin = int(diagnostics.get("max_bin", min(split_point + gluing_config["window_size"] // 2, n_altitude)))
                corrected[block_index, :] = glued_profile
                corrected_error[block_index, :] = propagate_glued_error(
                    inputs.analog_error_block[block_index, :],
                    inputs.photon_error_block[block_index, :],
                    slope_i,
                    min_bin,
                    max_bin,
                )
                rcs[block_index, :], rcs_error[block_index, :] = to_rcs(
                    corrected[block_index, :],
                    corrected_error[block_index, :],
                    altitude_m,
                )
                merge_source[block_index, :] = merge_source_flags(n_altitude, min_bin, max_bin)
                success[block_index] = 1
                split[block_index] = float(altitude_m[split_point])
                start[block_index] = float(altitude_m[min_bin])
                stop[block_index] = float(altitude_m[max_bin - 1])
            elif gluing_config["fallback_to_photon_counting"]:
                corrected[block_index, :] = inputs.photon_block[block_index, :]
                corrected_error[block_index, :] = inputs.photon_error_block[block_index, :]
                rcs[block_index, :], rcs_error[block_index, :] = to_rcs(
                    corrected[block_index, :],
                    corrected_error[block_index, :],
                    altitude_m,
                )
                merge_source[block_index, :] = merge_source_flags(n_altitude, -1, -1, split_failed=True)
                fallback[block_index] = 1
        if success.sum() == 0 and not gluing_config["fallback_to_photon_counting"]:
            raise ValueError(
                f"{inputs.wavelength_nm} nm has no successful block gluing and photon fallback is disabled."
            )
        logger.info(
            f"  -> {inputs.wavelength_nm} nm block gluing success: "
            f"{100.0 * success.sum() / max(n_block, 1):.1f}% "
            f"({inputs.analog_channel} + {inputs.photon_channel}); fallback blocks: {int(fallback.sum())}."
        )
    else:
        fallback_channel = inputs.photon_channel or inputs.analog_channel
        fallback_block = inputs.photon_block if inputs.photon_block is not None else inputs.analog_block
        fallback_error_block = (
            inputs.photon_error_block if inputs.photon_error_block is not None else inputs.analog_error_block
        )
        if fallback_channel is None or fallback_block is None or fallback_error_block is None:
            raise ValueError(f"No usable channel found for wavelength {inputs.wavelength_nm} nm.")
        if not gluing_config["fallback_to_photon_counting"]:
            raise ValueError(
                f"{inputs.wavelength_nm} nm cannot perform gluing because only {fallback_channel} is available "
                "and photon fallback is disabled."
            )
        corrected[:, :] = fallback_block
        corrected_error[:, :] = fallback_error_block
        rcs[:, :] = fallback_block * inputs.range_factor
        rcs_error[:, :] = fallback_error_block * inputs.range_factor
        merge_source[:, :] = 0 if fallback_channel == inputs.photon_channel else 2
        fallback[:] = 1
        source = f"block_mean_corrected_signal_single_channel_{fallback_channel}"
        logger.warning(
            f"  -> {inputs.wavelength_nm} nm using block single-channel fallback: {fallback_channel}."
        )

    return BlockGluingResult(
        source=source,
        corrected_signal=corrected,
        corrected_signal_error=corrected_error,
        range_corrected_signal=rcs,
        range_corrected_signal_error=rcs_error,
        merge_source_flag=merge_source,
        success_flag=success,
        fallback_flag=fallback,
        split_altitude_m=split,
        start_altitude_m=start,
        stop_altitude_m=stop,
        slope=slope,
        intercept=intercept,
        correlation=correlation,
        relative_rmse=relative_rmse,
        relative_bias=relative_bias,
    )


@dataclass(frozen=True, slots=True)
class MolecularModel:
    """Molecular atmosphere and retrieval parameters for one wavelength."""

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


def build_molecular_model(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
) -> MolecularModel:
    """Build the molecular atmosphere and configured retrieval assumptions."""
    pressure_hpa, temperature_k, source = build_thermodynamic_profile(ds_l1, altitude_m, config)
    backscatter, extinction = calculate_molecular_profile(temperature_k, pressure_hpa, wavelength_nm)
    simulated_signal, transmission = calculate_simulated_molecular_signal(backscatter, extinction, altitude_m)
    positive_altitudes = altitude_m[altitude_m > 0.0]
    safe_altitude = np.where(
        altitude_m > 0.0,
        altitude_m,
        positive_altitudes[0] if positive_altitudes.size else 1.0,
    )
    lidar_ratio, lidar_ratio_std = get_lidar_ratio(config, wavelength_nm, ds_l1["time"].values[0])
    return MolecularModel(
        source=source,
        backscatter=backscatter,
        extinction=extinction,
        transmission=transmission,
        simulated_signal=simulated_signal,
        simulated_range_corrected_signal=simulated_signal * safe_altitude**2,
        fit_config=get_molecular_fit_config(config),
        lidar_ratio_assumed_sr=lidar_ratio,
        lidar_ratio_std_sr=lidar_ratio_std,
        kfs_mode=get_kfs_mode(config),
    )


def retrieve_optical_blocks(
    inputs: WavelengthBlockInputs,
    glued: BlockGluingResult,
    molecular: MolecularModel,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> tuple[MolecularProfiles, OpticalProducts, RayleighDiagnostics, KfsDiagnostics]:
    """Run block Rayleigh QA/KFS and aggregate the valid optical products."""
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
    aerosol_backscatter_error = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    aerosol_extinction = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    aerosol_extinction_error = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    kfs_branch = np.zeros((n_block, n_altitude), dtype=np.int8)
    kfs_backward_valid = np.zeros(n_block, dtype=np.int8)
    kfs_forward_valid = np.zeros(n_block, dtype=np.int8)
    fit_config = molecular.fit_config

    for block_index in range(n_block):
        if glued.success_flag[block_index] != 1:
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
        scaled_molecular_rcs[block_index, :] = molecular.simulated_range_corrected_signal * factor
        scattering_ratio[block_index, :] = safe_ratio(
            glued.range_corrected_signal[block_index, :],
            scaled_molecular_rcs[block_index, :],
        )
        kfs_branch[block_index, :] = build_kfs_branch(
            altitude_m,
            reference_index,
            molecular.kfs_mode,
        )
        if int(qa["success_flag"]) == 1:
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
            kfs_backward_valid[block_index] = np.int8(bool(kfs_diagnostic["backward_valid"]))
            kfs_forward_valid[block_index] = np.int8(bool(kfs_diagnostic["forward_valid"]))

    rayleigh_valid_block = (glued.success_flag == 1) & (rayleigh_success == 1)
    valid_block = rayleigh_valid_block & (kfs_backward_valid == 1) & (kfs_forward_valid == 1)
    if not valid_block.any():
        logger.warning(
            f"  -> {inputs.wavelength_nm} nm has no valid retrieval block. Mean optical products set to NaN."
        )

    if rayleigh_valid_block.any():
        aggregate_factor = float(np.nanmedian(calibration_factor[rayleigh_valid_block]))
        aggregate_intercept = float(np.nanmedian(calibration_intercept[rayleigh_valid_block]))
        aggregate_reference_altitude = float(np.nanmedian(reference_altitude[rayleigh_valid_block]))
        aggregate_reference_start = float(np.nanmedian(reference_start[rayleigh_valid_block]))
        aggregate_reference_stop = float(np.nanmedian(reference_stop[rayleigh_valid_block]))
        aggregate_valid_bins = int(np.nanmedian(reference_valid_bins[rayleigh_valid_block]))
        aggregate_relative_slope = float(np.nanmedian(reference_relative_slope[rayleigh_valid_block]))
        aggregate_relative_variance = float(np.nanmedian(reference_relative_variance[rayleigh_valid_block]))
        aggregate_valid_fraction = float(np.nanmedian(reference_valid_fraction[rayleigh_valid_block]))
        aggregate_scaled_molecular = molecular.simulated_range_corrected_signal * aggregate_factor
        aggregate_kfs_branch = build_kfs_branch(
            altitude_m,
            int(np.nanargmin(np.abs(altitude_m - aggregate_reference_altitude))),
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

    aggregate_backward_valid = int(
        rayleigh_valid_block.any() and np.all(kfs_backward_valid[rayleigh_valid_block] == 1)
    )
    aggregate_forward_valid = int(
        rayleigh_valid_block.any() and np.all(kfs_forward_valid[rayleigh_valid_block] == 1)
    )

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
    optical_products = OpticalProducts(
        scattering_ratio_mean=valid_block_mean(scattering_ratio, valid_block),
        scattering_ratio_block=scattering_ratio,
        aerosol_backscatter=valid_block_mean(aerosol_backscatter, valid_block),
        aerosol_backscatter_error=valid_block_error(aerosol_backscatter_error, valid_block),
        aerosol_extinction=valid_block_mean(aerosol_extinction, valid_block),
        aerosol_extinction_error=valid_block_error(aerosol_extinction_error, valid_block),
        aerosol_backscatter_block=aerosol_backscatter,
        aerosol_backscatter_error_block=aerosol_backscatter_error,
        aerosol_extinction_block=aerosol_extinction,
        aerosol_extinction_error_block=aerosol_extinction_error,
        valid_retrieval_block_flag=valid_block.astype(np.int8),
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
    kfs_diagnostics = KfsDiagnostics(
        lidar_ratio_assumed_sr=molecular.lidar_ratio_assumed_sr,
        lidar_ratio_std_sr=molecular.lidar_ratio_std_sr,
        backward_valid_flag=aggregate_backward_valid,
        forward_valid_flag=aggregate_forward_valid,
        backward_valid_flag_block=kfs_backward_valid,
        forward_valid_flag_block=kfs_forward_valid,
        branch=aggregate_kfs_branch,
        branch_block=kfs_branch,
    )
    return molecular_profiles, optical_products, rayleigh_diagnostics, kfs_diagnostics


def assemble_wavelength_result(
    inputs: WavelengthBlockInputs,
    glued: BlockGluingResult,
    molecular: MolecularProfiles,
    optical: OpticalProducts,
    rayleigh: RayleighDiagnostics,
    kfs: KfsDiagnostics,
) -> WavelengthRetrievalResult:
    """Expand block diagnostics to time and assemble the public typed result."""
    valid_block = optical.valid_retrieval_block_flag.astype(bool)
    time_corrected = expand_blocks_to_time(glued.corrected_signal, inputs.block_groups, inputs.n_time)
    time_corrected_error = expand_blocks_to_time(
        glued.corrected_signal_error,
        inputs.block_groups,
        inputs.n_time,
    )
    time_rcs = expand_blocks_to_time(glued.range_corrected_signal, inputs.block_groups, inputs.n_time)
    time_rcs_error = expand_blocks_to_time(
        glued.range_corrected_signal_error,
        inputs.block_groups,
        inputs.n_time,
    )
    time_merge_source = np.full((inputs.n_time, inputs.n_altitude), 3, dtype=np.int8)
    for block_index, group in enumerate(inputs.block_groups):
        time_merge_source[group, :] = glued.merge_source_flag[block_index, :]

    result = WavelengthRetrievalResult(
        wavelength_nm=inputs.wavelength_nm,
        block_time=inputs.block_time,
        molecular=molecular,
        glued=GluedSignals(
            source=glued.source,
            analog_channel=inputs.analog_channel,
            photon_channel=inputs.photon_channel,
            corrected_signal=time_corrected,
            corrected_signal_error=time_corrected_error,
            corrected_signal_block=glued.corrected_signal,
            corrected_signal_error_block=glued.corrected_signal_error,
            corrected_signal_mean=valid_block_mean(glued.corrected_signal, valid_block),
            corrected_signal_error_mean=valid_block_error(glued.corrected_signal_error, valid_block),
            range_corrected_signal=time_rcs,
            range_corrected_signal_error=time_rcs_error,
            range_corrected_signal_block=glued.range_corrected_signal,
            range_corrected_signal_error_block=glued.range_corrected_signal_error,
            range_corrected_signal_mean=valid_block_mean(glued.range_corrected_signal, valid_block),
            range_corrected_signal_error_mean=valid_block_error(
                glued.range_corrected_signal_error,
                valid_block,
            ),
            merge_source_flag=time_merge_source,
            merge_source_flag_block=glued.merge_source_flag,
        ),
        optical=optical,
        rayleigh=rayleigh,
        kfs=kfs,
        gluing=GluingDiagnostics(
            success_flag=expand_block_vector_to_time(
                glued.success_flag,
                inputs.block_groups,
                inputs.n_time,
                dtype=np.int8,
            ),
            fallback_flag=expand_block_vector_to_time(
                glued.fallback_flag,
                inputs.block_groups,
                inputs.n_time,
                dtype=np.int8,
            ),
            split_altitude_m=expand_block_vector_to_time(
                glued.split_altitude_m,
                inputs.block_groups,
                inputs.n_time,
            ),
            start_altitude_m=expand_block_vector_to_time(
                glued.start_altitude_m,
                inputs.block_groups,
                inputs.n_time,
            ),
            stop_altitude_m=expand_block_vector_to_time(
                glued.stop_altitude_m,
                inputs.block_groups,
                inputs.n_time,
            ),
            slope=expand_block_vector_to_time(glued.slope, inputs.block_groups, inputs.n_time),
            intercept=expand_block_vector_to_time(glued.intercept, inputs.block_groups, inputs.n_time),
            correlation=expand_block_vector_to_time(glued.correlation, inputs.block_groups, inputs.n_time),
            relative_rmse=expand_block_vector_to_time(
                glued.relative_rmse,
                inputs.block_groups,
                inputs.n_time,
            ),
            relative_bias=expand_block_vector_to_time(
                glued.relative_bias,
                inputs.block_groups,
                inputs.n_time,
            ),
            success_flag_block=glued.success_flag,
            fallback_flag_block=glued.fallback_flag,
            split_altitude_m_block=glued.split_altitude_m,
            start_altitude_m_block=glued.start_altitude_m,
            stop_altitude_m_block=glued.stop_altitude_m,
            slope_block=glued.slope,
            intercept_block=glued.intercept,
            correlation_block=glued.correlation,
            relative_rmse_block=glued.relative_rmse,
            relative_bias_block=glued.relative_bias,
        ),
    )
    result.validate(n_time=inputs.n_time, n_altitude=inputs.n_altitude)
    return result


def process_wavelength(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> WavelengthRetrievalResult:
    """Process one wavelength using block retrievals."""
    inputs = _run_retrieval_stage(
        "selection_and_blocking",
        lambda: prepare_wavelength_blocks(ds_l1, wavelength_nm, altitude_m, config),
    )
    glued = _run_retrieval_stage(
        "gluing",
        lambda: glue_signal_blocks(inputs, altitude_m, logger),
    )
    molecular_model = _run_retrieval_stage(
        "molecular_model",
        lambda: build_molecular_model(ds_l1, wavelength_nm, altitude_m, config),
    )
    molecular, optical, rayleigh, kfs = _run_retrieval_stage(
        "rayleigh_kfs",
        lambda: retrieve_optical_blocks(
            inputs,
            glued,
            molecular_model,
            altitude_m,
            config,
            logger,
        ),
    )
    return _run_retrieval_stage(
        "result_assembly",
        lambda: assemble_wavelength_result(inputs, glued, molecular, optical, rayleigh, kfs),
    )
