"""Canonical Level 2 retrieval orchestration and dataset assembly.

Level 2 retrieval operates on configured temporal blocks, preserves the native lower
column, progressively aggregates the high column, selects a reference from the
first supported declared altitude range, and propagates native signal noise
through reference re-selection in the Monte Carlo ensemble.

Residual aerosol fraction ``f`` remains an outer systematic sensitivity
scenario. No Monte-Carlo survival fraction is converted into a hard retrieval
cutoff.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import xarray as xr

from milgrau.level2.adaptive_grid import (
    UncertaintyMode,
    aggregate_to_progressive_grid,
    build_progressive_grid,
)
from milgrau.level2.config import (
    get_block_average_minutes,
    get_kfs_config,
    get_molecular_fit_config,
    get_wavelengths_to_process,
)
from milgrau.level2.high_column import (
    contiguous_usable_top_index,
    prepare_high_column_profile,
)
from milgrau.level2.kfs import fernald_inversion
from milgrau.level2.two_sided_retrieval import retrieve_two_sided_profile
from milgrau.level2.retrieval import prepare_wavelength_state
from milgrau.scientific import (
    LEVEL2_PRODUCT_SCHEMA_CHANGE,
    LEVEL2_PRODUCT_SCHEMA_VERSION,
    elastic_inversion_algorithm_metadata,
)


@dataclass(frozen=True, slots=True)
class RetrievalConfiguration:
    """Explicit productive Level 2 scientific policy parsed from configuration."""

    reference_search_ranges_m: tuple[tuple[float, float], ...]
    path_start_altitude_m: float
    residual_aerosol_fractions: tuple[float, ...]
    uncertainty_mode: UncertaintyMode
    progressive_grid_schedule: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class WavelengthProduct:
    """Block-resolved productive result for one elastic wavelength."""

    wavelength_nm: int
    block_time: np.ndarray
    block_start_utc: np.ndarray
    block_end_utc: np.ndarray
    altitude_m: np.ndarray
    effective_vertical_resolution_m: np.ndarray
    source_bin_count: np.ndarray
    molecular_backscatter: np.ndarray
    molecular_extinction: np.ndarray
    lidar_ratio_assumed_sr_block: np.ndarray
    lidar_ratio_std_sr_block: np.ndarray
    integration_mode: str
    range_corrected_signal_block: np.ndarray
    range_corrected_signal_error_block: np.ndarray
    aerosol_backscatter_nominal_block: np.ndarray
    aerosol_extinction_nominal_block: np.ndarray
    aerosol_backscatter_mc_mean: np.ndarray
    aerosol_backscatter_mc_std: np.ndarray
    aerosol_backscatter_mc_q025: np.ndarray
    aerosol_backscatter_mc_q975: np.ndarray
    aerosol_extinction_mc_mean: np.ndarray
    aerosol_extinction_mc_std: np.ndarray
    aerosol_extinction_mc_q025: np.ndarray
    aerosol_extinction_mc_q975: np.ndarray
    mc_valid_fraction: np.ndarray
    period_mean_aerosol_backscatter_nominal: np.ndarray
    period_mean_aerosol_extinction_nominal: np.ndarray
    period_support_count: np.ndarray
    period_support_fraction: np.ndarray
    retrieval_top_altitude_m: float
    reference_altitude_m_block: np.ndarray
    reference_search_min_altitude_m_block: np.ndarray
    reference_search_max_altitude_m_block: np.ndarray
    reference_search_range_index_block: np.ndarray
    reference_fallback_used_block: np.ndarray
    reference_relative_slope_block: np.ndarray
    reference_relative_variance_block: np.ndarray
    reference_valid_fraction_block: np.ndarray
    reference_diagnostic_cost_block: np.ndarray
    reference_snr_median_block: np.ndarray
    reference_effective_resolution_m_block: np.ndarray
    reference_source_bin_count_block: np.ndarray
    contiguous_path_top_altitude_m_block: np.ndarray
    selection_success_fraction_block: np.ndarray
    selected_reference_altitude_m_mc: np.ndarray
    selected_reference_search_min_altitude_m_mc: np.ndarray
    selected_reference_search_max_altitude_m_mc: np.ndarray
    selected_reference_search_range_index_mc: np.ndarray
    background_offset_block: np.ndarray
    background_offset_standard_error_block: np.ndarray
    calibration_background_correlation_block: np.ndarray
    background_offset_mc: np.ndarray
    backward_valid_flag_block: np.ndarray
    forward_valid_flag_block: np.ndarray
    backward_endpoint_altitude_m_block: np.ndarray
    forward_endpoint_altitude_m_block: np.ndarray
    backward_endpoint_altitude_m_mc: np.ndarray
    forward_endpoint_altitude_m_mc: np.ndarray
    retrieval_input_valid_flag_block: np.ndarray
    retrieval_input_invalid_reason_block: np.ndarray
    retrieval_success_flag_block: np.ndarray
    signal_source_flag_block: np.ndarray
    gluing_attempted_flag_block: np.ndarray
    gluing_success_flag_block: np.ndarray
    single_channel_fallback_flag_block: np.ndarray
    gluing_start_altitude_m_block: np.ndarray
    gluing_split_altitude_m_block: np.ndarray
    gluing_stop_altitude_m_block: np.ndarray
    gluing_slope_block: np.ndarray
    gluing_intercept_block: np.ndarray
    gluing_correlation_block: np.ndarray
    gluing_relative_rmse_block: np.ndarray
    gluing_relative_bias_block: np.ndarray


def _required_mapping(parent: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing required configuration mapping: {path}.{key}")
    return value


def _positive_float(value: Any, path: str, *, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Configuration {path} must be numeric.") from exc
    if not np.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"Configuration {path} must be finite and {relation}.")
    return result


def get_retrieval_config(config: Mapping[str, Any]) -> RetrievalConfiguration:
    """Parse and validate the complete productive Level 2 configuration."""
    inversion = _required_mapping(config, "inversion", "config")
    retrieval = _required_mapping(inversion, "retrieval", "inversion")

    raw_ranges = retrieval.get("reference_search_ranges_m")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise ValueError(
            "inversion.retrieval.reference_search_ranges_m must be a non-empty list."
        )
    ranges: list[tuple[float, float]] = []
    for index, row in enumerate(raw_ranges):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(
                "Each inversion.retrieval.reference_search_ranges_m row must contain "
                "[min_altitude_m, max_altitude_m]."
            )
        lower = _positive_float(
            row[0],
            f"inversion.retrieval.reference_search_ranges_m[{index}][0]",
            allow_zero=True,
        )
        upper = _positive_float(
            row[1],
            f"inversion.retrieval.reference_search_ranges_m[{index}][1]",
        )
        if upper <= lower:
            raise ValueError(
                "retrieval reference search range maxima must exceed their minima."
            )
        ranges.append((lower, upper))
    path_start = _positive_float(
        retrieval.get("path_start_altitude_m"),
        "inversion.retrieval.path_start_altitude_m",
        allow_zero=True,
    )

    raw_fractions = retrieval.get("residual_aerosol_fractions")
    if not isinstance(raw_fractions, list) or not raw_fractions:
        raise ValueError(
            "inversion.retrieval.residual_aerosol_fractions must be a non-empty list."
        )
    fractions = tuple(
        _positive_float(
            value,
            f"inversion.retrieval.residual_aerosol_fractions[{index}]",
            allow_zero=True,
        )
        for index, value in enumerate(raw_fractions)
    )
    if 0.0 not in fractions:
        raise ValueError("retrieval residual_aerosol_fractions must include the nominal f=0 scenario.")
    if len(set(fractions)) != len(fractions):
        raise ValueError("retrieval residual_aerosol_fractions must not contain duplicates.")

    uncertainty_mode = str(retrieval.get("uncertainty_mode", "")).strip()
    if uncertainty_mode not in {"independent", "fully_correlated"}:
        raise ValueError(
            "inversion.retrieval.uncertainty_mode must be 'independent' or 'fully_correlated'."
        )

    raw_schedule = retrieval.get("progressive_grid_schedule")
    if not isinstance(raw_schedule, list) or not raw_schedule:
        raise ValueError(
            "inversion.retrieval.progressive_grid_schedule must be a non-empty list."
        )
    schedule_rows: list[tuple[float, float]] = []
    for index, row in enumerate(raw_schedule):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(
                "Each inversion.retrieval.progressive_grid_schedule row must contain "
                "[min_altitude_m, resolution_m]."
            )
        start = _positive_float(
            row[0],
            f"inversion.retrieval.progressive_grid_schedule[{index}][0]",
            allow_zero=True,
        )
        width = _positive_float(
            row[1],
            f"inversion.retrieval.progressive_grid_schedule[{index}][1]",
        )
        schedule_rows.append((start, width))
    if any(
        right[0] <= left[0]
        for left, right in zip(schedule_rows, schedule_rows[1:], strict=False)
    ):
        raise ValueError("retrieval progressive-grid altitude thresholds must increase strictly.")

    return RetrievalConfiguration(
        reference_search_ranges_m=tuple(ranges),
        path_start_altitude_m=path_start,
        residual_aerosol_fractions=fractions,
        uncertainty_mode=uncertainty_mode,  # type: ignore[arg-type]
        progressive_grid_schedule=tuple(schedule_rows),
    )


def _finite_mean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    count = np.count_nonzero(finite, axis=axis)
    total = np.sum(np.where(finite, array, 0.0), axis=axis)
    out = np.full(np.asarray(total).shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _datetime64ns(value: Any) -> np.datetime64:
    return np.datetime64(value, "ns")


def _period_support(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(profiles, dtype=np.float64)
    count = np.count_nonzero(np.isfinite(values), axis=0).astype(np.int32)
    denominator = values.shape[0]
    fraction = (
        count.astype(np.float64) / float(denominator)
        if denominator
        else np.full(count.shape, np.nan, dtype=np.float64)
    )
    return count, fraction


def retrieve_wavelength(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> WavelengthProduct:
    """Run productive Level 2 retrieval for every temporal block of one wavelength."""
    retrieval_cfg = get_retrieval_config(config)
    kfs_cfg = get_kfs_config(config)
    fit_cfg = get_molecular_fit_config(config)
    inputs, glued, _initial_molecular = prepare_wavelength_state(
        ds_l1, int(wavelength_nm), altitude_m, config, logger
    )

    grid = build_progressive_grid(altitude_m, retrieval_cfg.progressive_grid_schedule)
    block_time = np.asarray(inputs.block_time).astype("datetime64[ns]")
    n_block = block_time.size
    n_altitude = grid.n_cells
    fractions = np.asarray(retrieval_cfg.residual_aerosol_fractions, dtype=np.float64)
    n_fraction = fractions.size
    iterations = int(kfs_cfg["monte_carlo_iterations"])

    block_start = np.full(n_block, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    block_end = np.full_like(block_start, np.datetime64("NaT", "ns"))
    times = np.asarray(ds_l1["time"].values)
    for block_index, group in enumerate(inputs.block_groups):
        if group.size:
            block_start[block_index] = _datetime64ns(times[group[0]])
            block_end[block_index] = _datetime64ns(times[group[-1]])

    molecular_beta_block = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    molecular_alpha_block = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    lidar_ratio_assumed_block = np.full(n_block, np.nan, dtype=np.float64)
    lidar_ratio_std_block = np.full(n_block, np.nan, dtype=np.float64)
    rcs_block = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    rcs_error_block = np.full_like(rcs_block, np.nan)
    beta_nominal = np.full_like(rcs_block, np.nan)
    alpha_nominal = np.full_like(rcs_block, np.nan)
    mc_shape = (n_block, n_fraction, n_altitude)
    beta_mc_mean = np.full(mc_shape, np.nan, dtype=np.float64)
    beta_mc_std = np.full(mc_shape, np.nan, dtype=np.float64)
    beta_mc_q025 = np.full(mc_shape, np.nan, dtype=np.float64)
    beta_mc_q975 = np.full(mc_shape, np.nan, dtype=np.float64)
    alpha_mc_mean = np.full(mc_shape, np.nan, dtype=np.float64)
    alpha_mc_std = np.full(mc_shape, np.nan, dtype=np.float64)
    alpha_mc_q025 = np.full(mc_shape, np.nan, dtype=np.float64)
    alpha_mc_q975 = np.full(mc_shape, np.nan, dtype=np.float64)
    mc_valid_fraction = np.full(mc_shape, np.nan, dtype=np.float64)

    reference_altitude = np.full(n_block, np.nan, dtype=np.float64)
    reference_search_min = np.full(n_block, np.nan, dtype=np.float64)
    reference_search_max = np.full(n_block, np.nan, dtype=np.float64)
    reference_search_range_index = np.full(n_block, -1, dtype=np.int16)
    reference_fallback = np.zeros(n_block, dtype=np.int8)
    reference_relative_slope = np.full(n_block, np.nan, dtype=np.float64)
    reference_relative_variance = np.full(n_block, np.nan, dtype=np.float64)
    reference_valid_fraction = np.full(n_block, np.nan, dtype=np.float64)
    reference_diagnostic_cost = np.full(n_block, np.nan, dtype=np.float64)
    reference_snr_median = np.full(n_block, np.nan, dtype=np.float64)
    reference_effective_resolution = np.full(n_block, np.nan, dtype=np.float64)
    reference_source_bin_count = np.zeros(n_block, dtype=np.int32)
    path_top = np.full(n_block, np.nan, dtype=np.float64)
    selection_success_fraction = np.full(n_block, np.nan, dtype=np.float64)
    selected_reference_mc = np.full((n_block, iterations), np.nan, dtype=np.float64)
    selected_search_min_mc = np.full((n_block, iterations), np.nan, dtype=np.float64)
    selected_search_max_mc = np.full((n_block, iterations), np.nan, dtype=np.float64)
    selected_search_range_index_mc = np.full((n_block, iterations), -1, dtype=np.int16)
    background_offset = np.full(n_block, np.nan, dtype=np.float64)
    background_offset_standard_error = np.full(n_block, np.nan, dtype=np.float64)
    calibration_background_correlation = np.full(n_block, np.nan, dtype=np.float64)
    background_offset_mc = np.full((n_block, iterations), np.nan, dtype=np.float64)
    backward_valid = np.zeros(n_block, dtype=np.int8)
    forward_valid = np.zeros(n_block, dtype=np.int8)
    backward_endpoint = np.full(n_block, np.nan, dtype=np.float64)
    forward_endpoint = np.full(n_block, np.nan, dtype=np.float64)
    endpoint_mc_shape = (n_block, n_fraction, iterations)
    backward_endpoint_mc = np.full(endpoint_mc_shape, np.nan, dtype=np.float64)
    forward_endpoint_mc = np.full(endpoint_mc_shape, np.nan, dtype=np.float64)
    retrieval_success = np.zeros(n_block, dtype=np.int8)
    integration_mode = str(kfs_cfg["kfs_mode"])

    for block_index in range(n_block):
        if int(glued.retrieval_input_valid_flag[block_index]) != 1:
            continue
        block_started = time.perf_counter()
        logger.debug(
            "  -> %d nm block %d/%d selection-aware MC start | iterations=%d",
            int(wavelength_nm),
            int(block_index + 1),
            int(n_block),
            int(iterations),
        )
        native_signal = np.asarray(glued.range_corrected_signal[block_index], dtype=np.float64)
        native_error = np.asarray(glued.range_corrected_signal_error[block_index], dtype=np.float64)
        block_molecular = build_molecular_model(
            ds_l1,
            int(wavelength_nm),
            altitude_m,
            config,
            target_time=block_time[block_index],
        )
        molecular_beta = aggregate_to_progressive_grid(
            np.asarray(block_molecular.backscatter, dtype=np.float64),
            grid,
            require_positive=True,
        )
        molecular_alpha = aggregate_to_progressive_grid(
            np.asarray(block_molecular.extinction, dtype=np.float64),
            grid,
            require_positive=True,
        )
        if not np.all(molecular_beta.valid) or not np.all(molecular_alpha.valid):
            raise ValueError(
                f"Molecular state is invalid on the progressive grid for block {block_index}."
            )
        molecular_beta_block[block_index] = np.asarray(molecular_beta.values, dtype=np.float64)
        molecular_alpha_block[block_index] = np.asarray(molecular_alpha.values, dtype=np.float64)
        lidar_ratio_assumed_block[block_index] = float(block_block_molecular.lidar_ratio_assumed_sr)
        lidar_ratio_std_block[block_index] = float(block_block_molecular.lidar_ratio_std_sr)
        prepared = prepare_high_column_profile(
            range_corrected_signal=native_signal,
            range_corrected_signal_error=native_error,
            molecular_backscatter=np.asarray(block_molecular.backscatter, dtype=np.float64),
            altitude_m=altitude_m,
            uncertainty_mode=retrieval_cfg.uncertainty_mode,
            schedule=retrieval_cfg.progressive_grid_schedule,
        )
        if not np.array_equal(prepared.grid.altitude_m, grid.altitude_m):
            raise RuntimeError("Level 2 progressive-grid geometry changed between blocks.")
        seed = (
            int(kfs_cfg["random_seed"])
            + 100_000 * int(wavelength_nm)
            + int(block_index)
        )
        try:
            result = retrieve_two_sided_profile(
                range_corrected_signal=native_signal,
                range_corrected_signal_error=native_error,
                molecular_backscatter=np.asarray(molecular.backscatter, dtype=np.float64),
                simulated_molecular_range_corrected_signal=np.asarray(
                    block_molecular.simulated_range_corrected_signal, dtype=np.float64
                ),
                altitude_m=altitude_m,
                aerosol_lidar_ratio_sr=float(block_molecular.lidar_ratio_assumed_sr),
                aerosol_lidar_ratio_std_sr=float(block_molecular.lidar_ratio_std_sr),
                residual_fractions=fractions,
                n_iterations=iterations,
                beta_ref_relative_std=float(kfs_cfg["beta_ref_relative_std"]),
                min_lidar_ratio_sr=float(kfs_cfg["min_lidar_ratio_sr"]),
                allow_negative_aerosol=bool(kfs_cfg["allow_negative_aerosol"]),
                seed=seed,
                max_relative_slope=float(fit_cfg["max_relative_slope"]),
                max_relative_variance=float(fit_cfg["max_relative_variance"]),
                min_valid_fraction=float(fit_cfg["min_valid_fraction"]),
                uncertainty_mode=retrieval_cfg.uncertainty_mode,
                progressive_grid_schedule=retrieval_cfg.progressive_grid_schedule,
                reference_search_ranges_m=retrieval_cfg.reference_search_ranges_m,
                background_fit_min_altitude_m=float(fit_cfg["ref_alt_min_m"]),
                background_fit_max_altitude_m=float(fit_cfg["ref_alt_max_m"]),
                rayleigh_window_m=float(fit_cfg["ref_window_m"]),
                path_start_altitude_m=retrieval_cfg.path_start_altitude_m,
                integration_mode=integration_mode,
            )
        except ValueError as exc:
            logger.warning(
                "  -> %d nm block %d/%d unsupported after %.1f s: %s",
                int(wavelength_nm),
                int(block_index + 1),
                int(n_block),
                time.perf_counter() - block_started,
                exc,
            )
            continue

        selected = result.selected_reference
        candidate = selected.native_rayleigh_candidate
        rcs_block[block_index] = result.prepared.range_corrected_signal
        rcs_error_block[block_index] = result.prepared.range_corrected_signal_error
        top_index = contiguous_usable_top_index(
            result.prepared,
            path_start_altitude_m=retrieval_cfg.path_start_altitude_m,
        )
        if top_index is not None:
            path_top[block_index] = float(grid.altitude_m[top_index])
        nominal_raw, nominal_diagnostics = fernald_inversion(
                result.prepared.range_corrected_signal,
                result.prepared.grid.altitude_m,
                result.prepared.molecular_backscatter,
                float(block_molecular.lidar_ratio_assumed_sr),
                float(result.prepared.molecular_backscatter[selected.cell_index]),
                int(selected.cell_index),
                altitude_units="m",
                min_lidar_ratio=float(kfs_cfg["min_lidar_ratio_sr"]),
                allow_negative_aerosol=bool(kfs_cfg["allow_negative_aerosol"]),
                mode=integration_mode,
                return_diagnostics=True,
            )
        nominal = np.asarray(
            nominal_raw,
            dtype=np.float64,
        )
        beta_nominal[block_index] = nominal
        alpha_nominal[block_index] = nominal * float(block_molecular.lidar_ratio_assumed_sr)
        backward_valid[block_index] = np.int8(
            bool(nominal_diagnostics["backward_valid"])
        )
        forward_valid[block_index] = np.int8(
            bool(nominal_diagnostics["forward_valid"])
        )
        backward_endpoint[block_index] = float(
            nominal_diagnostics["backward_endpoint_altitude_m"]
        )
        forward_endpoint[block_index] = float(
            nominal_diagnostics["forward_endpoint_altitude_m"]
        )
        retrieval_success[block_index] = backward_valid[block_index]
        reference_altitude[block_index] = float(selected.altitude_m)
        reference_search_min[block_index] = float(
            result.selected_reference_search_min_altitude_m
        )
        reference_search_max[block_index] = float(
            result.selected_reference_search_max_altitude_m
        )
        reference_search_range_index[block_index] = int(
            result.selected_reference_search_range_index
        )
        reference_fallback[block_index] = int(result.selected_reference_fallback_used)
        reference_relative_slope[block_index] = float(candidate.relative_slope)
        reference_relative_variance[block_index] = float(candidate.relative_variance)
        reference_valid_fraction[block_index] = float(candidate.valid_fraction)
        reference_diagnostic_cost[block_index] = float(candidate.diagnostic_cost)
        reference_snr_median[block_index] = float(candidate.uncertainty_snr_median)
        reference_effective_resolution[block_index] = float(selected.effective_resolution_m)
        reference_source_bin_count[block_index] = int(selected.source_count)
        selection_success_fraction[block_index] = float(
            result.monte_carlo.selection_success_fraction
        )
        logger.debug(
            "  -> %d nm block %d/%d MC done | ref=%.1f m | success=%.1f%% | %.1f s",
            int(wavelength_nm),
            int(block_index + 1),
            int(n_block),
            float(selected.altitude_m),
            100.0 * float(result.monte_carlo.selection_success_fraction),
            time.perf_counter() - block_started,
        )
        beta_mc_mean[block_index] = result.monte_carlo.aerosol_backscatter_mean
        beta_mc_std[block_index] = result.monte_carlo.aerosol_backscatter_random_std
        beta_mc_q025[block_index] = result.monte_carlo.aerosol_backscatter_random_q025
        beta_mc_q975[block_index] = result.monte_carlo.aerosol_backscatter_random_q975
        alpha_mc_mean[block_index] = result.monte_carlo.aerosol_extinction_mean
        alpha_mc_std[block_index] = result.monte_carlo.aerosol_extinction_random_std
        alpha_mc_q025[block_index] = result.monte_carlo.aerosol_extinction_random_q025
        alpha_mc_q975[block_index] = result.monte_carlo.aerosol_extinction_random_q975
        mc_valid_fraction[block_index] = result.monte_carlo.aerosol_backscatter_valid_fraction
        selected_reference_mc[block_index] = (
            result.monte_carlo.selected_reference_altitude_m_samples
        )
        selected_search_min_mc[block_index] = (
            result.monte_carlo.selected_reference_search_min_altitude_m_samples
        )
        selected_search_max_mc[block_index] = (
            result.monte_carlo.selected_reference_search_max_altitude_m_samples
        )
        selected_search_range_index_mc[block_index] = (
            result.monte_carlo.selected_reference_search_range_index_samples
        )
        background_offset[block_index] = float(result.background_offset)
        background_offset_standard_error[block_index] = float(
            result.background_offset_standard_error
        )
        calibration_background_correlation[block_index] = float(
            result.calibration_background_correlation
        )
        background_offset_mc[block_index] = result.monte_carlo.background_offset_samples
        backward_endpoint_mc[block_index] = (
            result.monte_carlo.backward_endpoint_altitude_m_samples
        )
        forward_endpoint_mc[block_index] = (
            result.monte_carlo.forward_endpoint_altitude_m_samples
        )

    support_count, support_fraction = _period_support(beta_nominal)
    supported = np.flatnonzero(support_count > 0)
    retrieval_top = float(grid.altitude_m[supported[-1]]) if supported.size else np.nan

    return WavelengthProduct(
        wavelength_nm=int(wavelength_nm),
        block_time=block_time,
        block_start_utc=block_start,
        block_end_utc=block_end,
        altitude_m=np.asarray(grid.altitude_m, dtype=np.float64),
        effective_vertical_resolution_m=np.asarray(
            grid.effective_resolution_m, dtype=np.float64
        ),
        source_bin_count=np.asarray(grid.source_count, dtype=np.int32),
        molecular_backscatter=molecular_beta_block,
        molecular_extinction=molecular_alpha_block,
        lidar_ratio_assumed_sr_block=lidar_ratio_assumed_block,
        lidar_ratio_std_sr_block=lidar_ratio_std_block,
        integration_mode=integration_mode,
        range_corrected_signal_block=rcs_block,
        range_corrected_signal_error_block=rcs_error_block,
        aerosol_backscatter_nominal_block=beta_nominal,
        aerosol_extinction_nominal_block=alpha_nominal,
        aerosol_backscatter_mc_mean=beta_mc_mean,
        aerosol_backscatter_mc_std=beta_mc_std,
        aerosol_backscatter_mc_q025=beta_mc_q025,
        aerosol_backscatter_mc_q975=beta_mc_q975,
        aerosol_extinction_mc_mean=alpha_mc_mean,
        aerosol_extinction_mc_std=alpha_mc_std,
        aerosol_extinction_mc_q025=alpha_mc_q025,
        aerosol_extinction_mc_q975=alpha_mc_q975,
        mc_valid_fraction=mc_valid_fraction,
        period_mean_aerosol_backscatter_nominal=_finite_mean(beta_nominal, axis=0),
        period_mean_aerosol_extinction_nominal=_finite_mean(alpha_nominal, axis=0),
        period_support_count=support_count,
        period_support_fraction=support_fraction,
        retrieval_top_altitude_m=retrieval_top,
        reference_altitude_m_block=reference_altitude,
        reference_search_min_altitude_m_block=reference_search_min,
        reference_search_max_altitude_m_block=reference_search_max,
        reference_search_range_index_block=reference_search_range_index,
        reference_fallback_used_block=reference_fallback,
        reference_relative_slope_block=reference_relative_slope,
        reference_relative_variance_block=reference_relative_variance,
        reference_valid_fraction_block=reference_valid_fraction,
        reference_diagnostic_cost_block=reference_diagnostic_cost,
        reference_snr_median_block=reference_snr_median,
        reference_effective_resolution_m_block=reference_effective_resolution,
        reference_source_bin_count_block=reference_source_bin_count,
        contiguous_path_top_altitude_m_block=path_top,
        selection_success_fraction_block=selection_success_fraction,
        selected_reference_altitude_m_mc=selected_reference_mc,
        selected_reference_search_min_altitude_m_mc=selected_search_min_mc,
        selected_reference_search_max_altitude_m_mc=selected_search_max_mc,
        selected_reference_search_range_index_mc=selected_search_range_index_mc,
        background_offset_block=background_offset,
        background_offset_standard_error_block=background_offset_standard_error,
        calibration_background_correlation_block=calibration_background_correlation,
        background_offset_mc=background_offset_mc,
        backward_valid_flag_block=backward_valid,
        forward_valid_flag_block=forward_valid,
        backward_endpoint_altitude_m_block=backward_endpoint,
        forward_endpoint_altitude_m_block=forward_endpoint,
        backward_endpoint_altitude_m_mc=backward_endpoint_mc,
        forward_endpoint_altitude_m_mc=forward_endpoint_mc,
        retrieval_input_valid_flag_block=np.asarray(
            glued.retrieval_input_valid_flag, dtype=np.int8
        ),
        retrieval_input_invalid_reason_block=np.asarray(
            glued.retrieval_input_invalid_reason, dtype=np.int8
        ),
        retrieval_success_flag_block=retrieval_success,
        signal_source_flag_block=np.asarray(glued.signal_source_flag, dtype=np.int8),
        gluing_attempted_flag_block=np.asarray(glued.attempted_flag, dtype=np.int8),
        gluing_success_flag_block=np.asarray(glued.success_flag, dtype=np.int8),
        single_channel_fallback_flag_block=np.asarray(
            glued.single_channel_fallback_flag, dtype=np.int8
        ),
        gluing_start_altitude_m_block=np.asarray(glued.start_altitude_m, dtype=np.float64),
        gluing_split_altitude_m_block=np.asarray(glued.split_altitude_m, dtype=np.float64),
        gluing_stop_altitude_m_block=np.asarray(glued.stop_altitude_m, dtype=np.float64),
        gluing_slope_block=np.asarray(glued.slope, dtype=np.float64),
        gluing_intercept_block=np.asarray(glued.intercept, dtype=np.float64),
        gluing_correlation_block=np.asarray(glued.correlation, dtype=np.float64),
        gluing_relative_rmse_block=np.asarray(glued.relative_rmse, dtype=np.float64),
        gluing_relative_bias_block=np.asarray(glued.relative_bias, dtype=np.float64),
    )


def _stack(results: Sequence[WavelengthProduct], name: str) -> np.ndarray:
    return np.stack([np.asarray(getattr(result, name)) for result in results], axis=0)


def _stack_block(results: Sequence[WavelengthProduct], name: str) -> np.ndarray:
    return np.stack([np.asarray(getattr(result, name)) for result in results], axis=1)


def build_level2_dataset(
    ds_l1: xr.Dataset,
    altitude_m: np.ndarray,
    source_file: str | Path,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> xr.Dataset:
    """Build the canonical multispectral Level 2 product."""
    wavelengths = tuple(int(value) for value in get_wavelengths_to_process(config))
    retrieval_cfg = get_retrieval_config(config)
    kfs_cfg = get_kfs_config(config)
    integration_mode = str(kfs_cfg["kfs_mode"])
    block_minutes = int(get_block_average_minutes(config))

    results = tuple(
        retrieve_wavelength(ds_l1, wavelength, altitude_m, config, logger)
        for wavelength in wavelengths
    )
    first = results[0]
    for result in results[1:]:
        if not np.array_equal(result.block_time, first.block_time):
            raise RuntimeError("Level 2 block-time geometry differs between wavelengths.")
        if not np.array_equal(result.altitude_m, first.altitude_m):
            raise RuntimeError("Level 2 progressive altitude differs between wavelengths.")

    n_iterations = int(kfs_cfg["monte_carlo_iterations"])
    residual_fractions = np.asarray(retrieval_cfg.residual_aerosol_fractions, dtype=np.float64)
    success_fraction = np.asarray(
        [np.mean(result.retrieval_success_flag_block == 1) for result in results],
        dtype=np.float64,
    )
    processed = np.asarray(
        [
            result.wavelength_nm
            for result in results
            if np.any(result.retrieval_success_flag_block == 1)
        ],
        dtype=np.int32,
    )
    failed = np.asarray(
        [
            result.wavelength_nm
            for result in results
            if not np.any(result.retrieval_success_flag_block == 1)
        ],
        dtype=np.int32,
    )
    if processed.size == 0:
        raise ValueError("Level 2 retrieval produced no valid retrieval block for any requested wavelength.")
    completeness = "complete" if failed.size == 0 else "partial"

    ds = xr.Dataset(
        data_vars={
            "effective_vertical_resolution_m": (
                ("wavelength", "altitude"),
                _stack(results, "effective_vertical_resolution_m"),
            ),
            "source_bin_count": (
                ("wavelength", "altitude"),
                _stack(results, "source_bin_count").astype(np.int32),
            ),
            "molecular_backscatter": (
                ("block_time", "wavelength", "altitude"),
                _stack_block(results, "molecular_backscatter"),
            ),
            "molecular_extinction": (
                ("block_time", "wavelength", "altitude"),
                _stack_block(results, "molecular_extinction"),
            ),
            "lidar_ratio_assumed_sr": (
                ("block_time", "wavelength"),
                _stack_block(results, "lidar_ratio_assumed_sr_block"),
            ),
            "lidar_ratio_std_sr": (
                ("block_time", "wavelength"),
                _stack_block(results, "lidar_ratio_std_sr_block"),
            ),
            "range_corrected_signal_block": (
                ("block_time", "wavelength", "altitude"),
                _stack_block(results, "range_corrected_signal_block"),
            ),
            "range_corrected_signal_error_block": (
                ("block_time", "wavelength", "altitude"),
                _stack_block(results, "range_corrected_signal_error_block"),
            ),
            "aerosol_backscatter_nominal_block": (
                ("block_time", "wavelength", "altitude"),
                _stack_block(results, "aerosol_backscatter_nominal_block"),
            ),
            "aerosol_extinction_nominal_block": (
                ("block_time", "wavelength", "altitude"),
                _stack_block(results, "aerosol_extinction_nominal_block"),
            ),
            "aerosol_backscatter_mc_mean": (
                ("block_time", "wavelength", "residual_fraction", "altitude"),
                _stack_block(results, "aerosol_backscatter_mc_mean"),
            ),
            "aerosol_backscatter_mc_std": (
                ("block_time", "wavelength", "residual_fraction", "altitude"),
                _stack_block(results, "aerosol_backscatter_mc_std"),
            ),
            "aerosol_backscatter_mc_q025": (
                ("block_time", "wavelength", "residual_fraction", "altitude"),
                _stack_block(results, "aerosol_backscatter_mc_q025"),
            ),
            "aerosol_backscatter_mc_q975": (
                ("block_time", "wavelength", "residual_fraction", "altitude"),
                _stack_block(results, "aerosol_backscatter_mc_q975"),
            ),
            "aerosol_extinction_mc_mean": (
                ("block_time", "wavelength", "residual_fraction", "altitude"),
                _stack_block(results, "aerosol_extinction_mc_mean"),
            ),
            "aerosol_extinction_mc_std": (
                ("block_time", "wavelength", "residual_fraction", "altitude"),
                _stack_block(results, "aerosol_extinction_mc_std"),
            ),
            "aerosol_extinction_mc_q025": (
                ("block_time", "wavelength", "residual_fraction", "altitude"),
                _stack_block(results, "aerosol_extinction_mc_q025"),
            ),
            "aerosol_extinction_mc_q975": (
                ("block_time", "wavelength", "residual_fraction", "altitude"),
                _stack_block(results, "aerosol_extinction_mc_q975"),
            ),
            "mc_valid_fraction": (
                ("block_time", "wavelength", "residual_fraction", "altitude"),
                _stack_block(results, "mc_valid_fraction"),
            ),
            "aerosol_backscatter_mean": (
                ("wavelength", "altitude"),
                _stack(results, "period_mean_aerosol_backscatter_nominal"),
            ),
            "aerosol_extinction_mean": (
                ("wavelength", "altitude"),
                _stack(results, "period_mean_aerosol_extinction_nominal"),
            ),
            "period_support_count": (
                ("wavelength", "altitude"),
                _stack(results, "period_support_count").astype(np.int32),
            ),
            "period_support_fraction": (
                ("wavelength", "altitude"),
                _stack(results, "period_support_fraction"),
            ),
            "retrieval_top_altitude_m": (
                ("wavelength",),
                np.asarray(
                    [result.retrieval_top_altitude_m for result in results],
                    dtype=np.float64,
                ),
            ),
            "rayleigh_reference_altitude_m_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "reference_altitude_m_block"),
            ),
            "rayleigh_reference_search_min_altitude_m_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "reference_search_min_altitude_m_block"),
            ),
            "rayleigh_reference_search_max_altitude_m_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "reference_search_max_altitude_m_block"),
            ),
            "rayleigh_reference_search_range_index_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "reference_search_range_index_block").astype(np.int16),
            ),
            "rayleigh_reference_fallback_used_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "reference_fallback_used_block").astype(np.int8),
            ),
            "rayleigh_reference_relative_slope_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "reference_relative_slope_block"),
            ),
            "rayleigh_reference_relative_variance_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "reference_relative_variance_block"),
            ),
            "rayleigh_reference_valid_fraction_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "reference_valid_fraction_block"),
            ),
            "rayleigh_reference_diagnostic_cost_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "reference_diagnostic_cost_block"),
            ),
            "rayleigh_reference_snr_median_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "reference_snr_median_block"),
            ),
            "rayleigh_reference_effective_resolution_m_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "reference_effective_resolution_m_block"),
            ),
            "rayleigh_reference_source_bin_count_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "reference_source_bin_count_block").astype(np.int32),
            ),
            "contiguous_path_top_altitude_m_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "contiguous_path_top_altitude_m_block"),
            ),
            "selection_success_fraction_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "selection_success_fraction_block"),
            ),
            "selected_reference_altitude_m_mc": (
                ("block_time", "wavelength", "mc_iteration"),
                _stack_block(results, "selected_reference_altitude_m_mc"),
            ),
            "selected_reference_search_min_altitude_m_mc": (
                ("block_time", "wavelength", "mc_iteration"),
                _stack_block(results, "selected_reference_search_min_altitude_m_mc"),
            ),
            "selected_reference_search_max_altitude_m_mc": (
                ("block_time", "wavelength", "mc_iteration"),
                _stack_block(results, "selected_reference_search_max_altitude_m_mc"),
            ),
            "selected_reference_search_range_index_mc": (
                ("block_time", "wavelength", "mc_iteration"),
                _stack_block(results, "selected_reference_search_range_index_mc").astype(np.int16),
            ),
            "rayleigh_background_offset_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "background_offset_block"),
            ),
            "rayleigh_background_offset_standard_error_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "background_offset_standard_error_block"),
            ),
            "rayleigh_calibration_background_correlation_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "calibration_background_correlation_block"),
            ),
            "rayleigh_background_offset_mc": (
                ("block_time", "wavelength", "mc_iteration"),
                _stack_block(results, "background_offset_mc"),
            ),
            "kfs_backward_valid_flag": (
                ("block_time", "wavelength"),
                _stack_block(results, "backward_valid_flag_block").astype(np.int8),
            ),
            "kfs_forward_valid_flag": (
                ("block_time", "wavelength"),
                _stack_block(results, "forward_valid_flag_block").astype(np.int8),
            ),
            "kfs_backward_endpoint_altitude_m": (
                ("block_time", "wavelength"),
                _stack_block(results, "backward_endpoint_altitude_m_block"),
            ),
            "kfs_forward_endpoint_altitude_m": (
                ("block_time", "wavelength"),
                _stack_block(results, "forward_endpoint_altitude_m_block"),
            ),
            "kfs_backward_endpoint_altitude_m_mc": (
                ("block_time", "wavelength", "residual_fraction", "mc_iteration"),
                _stack_block(results, "backward_endpoint_altitude_m_mc"),
            ),
            "kfs_forward_endpoint_altitude_m_mc": (
                ("block_time", "wavelength", "residual_fraction", "mc_iteration"),
                _stack_block(results, "forward_endpoint_altitude_m_mc"),
            ),
            "retrieval_input_valid_flag": (
                ("block_time", "wavelength"),
                _stack_block(results, "retrieval_input_valid_flag_block").astype(np.int8),
            ),
            "retrieval_input_invalid_reason": (
                ("block_time", "wavelength"),
                _stack_block(results, "retrieval_input_invalid_reason_block").astype(np.int8),
            ),
            "retrieval_success_flag": (
                ("block_time", "wavelength"),
                _stack_block(results, "retrieval_success_flag_block").astype(np.int8),
            ),
            "retrieval_success_fraction": (("wavelength",), success_fraction),
            "signal_source_flag": (
                ("block_time", "wavelength"),
                _stack_block(results, "signal_source_flag_block").astype(np.int8),
            ),
            "gluing_attempted_flag": (
                ("block_time", "wavelength"),
                _stack_block(results, "gluing_attempted_flag_block").astype(np.int8),
            ),
            "gluing_success_flag": (
                ("block_time", "wavelength"),
                _stack_block(results, "gluing_success_flag_block").astype(np.int8),
            ),
            "single_channel_fallback_flag": (
                ("block_time", "wavelength"),
                _stack_block(results, "single_channel_fallback_flag_block").astype(np.int8),
            ),
            "gluing_start_altitude_m": (
                ("block_time", "wavelength"),
                _stack_block(results, "gluing_start_altitude_m_block"),
            ),
            "gluing_split_altitude_m": (
                ("block_time", "wavelength"),
                _stack_block(results, "gluing_split_altitude_m_block"),
            ),
            "gluing_stop_altitude_m": (
                ("block_time", "wavelength"),
                _stack_block(results, "gluing_stop_altitude_m_block"),
            ),
            "gluing_slope": (
                ("block_time", "wavelength"),
                _stack_block(results, "gluing_slope_block"),
            ),
            "gluing_intercept": (
                ("block_time", "wavelength"),
                _stack_block(results, "gluing_intercept_block"),
            ),
            "gluing_correlation": (
                ("block_time", "wavelength"),
                _stack_block(results, "gluing_correlation_block"),
            ),
            "gluing_relative_rmse": (
                ("block_time", "wavelength"),
                _stack_block(results, "gluing_relative_rmse_block"),
            ),
            "gluing_relative_bias": (
                ("block_time", "wavelength"),
                _stack_block(results, "gluing_relative_bias_block"),
            ),
            "requested_wavelengths": (
                ("requested_wavelength",),
                np.asarray(wavelengths, dtype=np.int32),
            ),
            "processed_wavelengths": (("processed_wavelength",), processed),
            "failed_wavelengths": (("failed_wavelength",), failed),
        },
        coords={
            "block_time": first.block_time,
            "block_start_utc": (("block_time",), first.block_start_utc),
            "block_end_utc": (("block_time",), first.block_end_utc),
            "wavelength": np.asarray(wavelengths, dtype=np.int32),
            "altitude": first.altitude_m,
            "residual_fraction": residual_fractions,
            "mc_iteration": np.arange(n_iterations, dtype=np.int32),
        },
        attrs={
            "title": "MILGRAU Level 2 elastic optical product",
            "source_level1": str(Path(source_file)),
            "level2_product_schema_version": LEVEL2_PRODUCT_SCHEMA_VERSION,
            "level2_product_schema_change": LEVEL2_PRODUCT_SCHEMA_CHANGE,
            "product_status": "success" if completeness == "complete" else "partial",
            "product_completeness": completeness,
            "configured_block_minutes": block_minutes,
            "reference_search_ranges_m": ";".join(
                f"{lower:g}:{upper:g}"
                for lower, upper in retrieval_cfg.reference_search_ranges_m
            ),
            "reference_selection_policy": (
                "first_supported_declared_range_then_minimum_existing_rayleigh_cost"
            ),
            "reference_search_range_interpretation": (
                "search-domain preference only; not an aerosol-free or molecular-purity criterion"
            ),
            "progressive_grid_schedule": ";".join(
                f"{start:g}:{width:g}" for start, width in retrieval_cfg.progressive_grid_schedule
            ),
            "uncertainty_mode": retrieval_cfg.uncertainty_mode,
            "boundary_nominal_residual_fraction": 0.0,
            "boundary_systematic_scenarios": ",".join(
                f"{value:g}" for value in retrieval_cfg.residual_aerosol_fractions
            ),
            "support_fraction_denominator": "all configured temporal blocks",
            "period_mean_semantics": (
                "finite-only altitude-by-altitude mean; inspect period_support_count/fraction jointly"
            ),
            "mc_valid_fraction_semantics": (
                "finite selection-aware Monte-Carlo realization fraction; diagnostic only, no cutoff"
            ),
            "background_model": (
                "RCS(z)=A*molecular_RCS(z)+B*z^2; B is constant in pre-range-correction signal space"
            ),
            "background_fit_method": (
                "inverse_variance_weighted_Huber_IRLS_over_full_Rayleigh_search_span"
            ),
            "background_fit_min_altitude_m": float(
                get_molecular_fit_config(config)["ref_alt_min_m"]
            ),
            "background_fit_max_altitude_m": float(
                get_molecular_fit_config(config)["ref_alt_max_m"]
            ),
            "background_uncertainty_propagation": (
                "B refitted from each native-signal Monte-Carlo realization before reference selection and KFS"
            ),
            "monte_carlo_iterations": n_iterations,
            "monte_carlo_random_seed": int(kfs_cfg["random_seed"]),
            "kfs_beta_ref_relative_std": float(kfs_cfg["beta_ref_relative_std"]),
            "kfs_min_lidar_ratio_sr": float(kfs_cfg["min_lidar_ratio_sr"]),
            "kfs_allow_negative_aerosol": int(bool(kfs_cfg["allow_negative_aerosol"])),
            **elastic_inversion_algorithm_metadata(integration_mode),
        },
    )

    ds["altitude"].attrs.update({"units": "m", "positive": "up"})
    ds["effective_vertical_resolution_m"].attrs.update(
        {
            "units": "m",
            "long_name": "effective vertical cell width on the progressive retrieval grid",
        }
    )
    ds["source_bin_count"].attrs["long_name"] = (
        "number of contiguous native Level-1 altitude bins represented by each progressive cell"
    )
    ds["aerosol_backscatter_mean"].attrs.update({"units": "m-1 sr-1"})
    ds["aerosol_extinction_mean"].attrs.update({"units": "m-1"})
    ds["aerosol_backscatter_nominal_block"].attrs.update({"units": "m-1 sr-1"})
    ds["aerosol_extinction_nominal_block"].attrs.update({"units": "m-1"})
    ds["period_support_fraction"].attrs["long_name"] = (
        "fraction of configured temporal blocks contributing a finite nominal retrieval"
    )
    ds["selection_success_fraction_block"].attrs["long_name"] = (
        "fraction of Monte-Carlo realizations in which prioritized-range reference selection succeeds"
    )
    ds["kfs_backward_valid_flag"].attrs["long_name"] = (
        "nominal KFS backward branch reaches every required contiguous sampled bin"
    )
    ds["kfs_forward_valid_flag"].attrs["long_name"] = (
        "nominal KFS forward branch reaches every required contiguous sampled bin"
    )
    for name in (
        "kfs_backward_endpoint_altitude_m",
        "kfs_forward_endpoint_altitude_m",
        "kfs_backward_endpoint_altitude_m_mc",
        "kfs_forward_endpoint_altitude_m_mc",
    ):
        ds[name].attrs.update(
            {
                "units": "m",
                "long_name": "last finite altitude reached by the oriented KFS branch",
            }
        )
    ds["rayleigh_reference_diagnostic_cost_block"].attrs["long_name"] = (
        "selected native-grid Rayleigh candidate relative_slope plus relative_variance"
    )
    ds["rayleigh_reference_effective_resolution_m_block"].attrs["units"] = "m"
    for name in (
        "rayleigh_background_offset_block",
        "rayleigh_background_offset_standard_error_block",
        "rayleigh_background_offset_mc",
    ):
        ds[name].attrs.update(
            {
                "long_name": "robust fitted residual background in pre-range-correction signal space",
                "unit_status": "source_dependent_pre_range_correction_signal",
            }
        )
    ds["rayleigh_calibration_background_correlation_block"].attrs.update(
        {
            "long_name": "broad-span Rayleigh calibration/background parameter correlation",
            "units": "1",
        }
    )
    ds["range_corrected_signal_block"].attrs.update(
        {
            "description": (
                "Progressive-grid selected/glued RCS after subtracting the robust fitted B*z^2 term; "
                "this is the signal used by the inversion."
            )
        }
    )
    return ds


__all__ = [
    "RetrievalConfiguration",
    "WavelengthProduct",
    "build_level2_dataset",
    "get_retrieval_config",
    "retrieve_wavelength",
]
