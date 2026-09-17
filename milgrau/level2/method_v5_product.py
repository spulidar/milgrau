"""Productive method-v5 Level 2 retrieval and schema-4 dataset assembly.

Method v5 operates on 20-minute temporal blocks, preserves the native lower
column, progressively aggregates the high column, selects a reference from the
highest supported declared altitude tier, and propagates native signal noise
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
from milgrau.level2.high_column_rnd import (
    contiguous_usable_top_index,
    prepare_high_column_profile,
)
from milgrau.level2.kfs import fernald_inversion
from milgrau.level2.method_v5_rnd import retrieve_method_v5_rnd
from milgrau.level2.retrieval import prepare_wavelength_state
from milgrau.scientific import (
    LEVEL2_PRODUCT_SCHEMA_CHANGE,
    LEVEL2_PRODUCT_SCHEMA_VERSION,
    LEVEL2_RETRIEVAL_METHOD_VERSION,
    elastic_inversion_algorithm_metadata,
)


@dataclass(frozen=True, slots=True)
class MethodV5Configuration:
    """Explicit productive method-v5 scientific policy parsed from configuration."""

    reference_tier_min_altitudes_m: tuple[float, ...]
    reference_search_max_m: float
    path_start_altitude_m: float
    residual_aerosol_fractions: tuple[float, ...]
    uncertainty_mode: UncertaintyMode
    progressive_grid_schedule: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class MethodV5WavelengthProduct:
    """Block-resolved productive v5 result for one elastic wavelength."""

    wavelength_nm: int
    block_time: np.ndarray
    block_start_utc: np.ndarray
    block_end_utc: np.ndarray
    altitude_m: np.ndarray
    effective_vertical_resolution_m: np.ndarray
    source_bin_count: np.ndarray
    molecular_backscatter: np.ndarray
    molecular_extinction: np.ndarray
    lidar_ratio_assumed_sr: float
    lidar_ratio_std_sr: float
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
    reference_tier_min_altitude_m_block: np.ndarray
    reference_tier_index_block: np.ndarray
    reference_fallback_used_block: np.ndarray
    contiguous_path_top_altitude_m_block: np.ndarray
    selection_success_fraction_block: np.ndarray
    selected_reference_altitude_m_mc: np.ndarray
    selected_reference_tier_min_altitude_m_mc: np.ndarray
    selected_reference_tier_index_mc: np.ndarray
    retrieval_input_valid_flag_block: np.ndarray
    retrieval_input_invalid_reason_block: np.ndarray
    retrieval_success_flag_block: np.ndarray
    signal_source_flag_block: np.ndarray
    gluing_attempted_flag_block: np.ndarray
    gluing_success_flag_block: np.ndarray
    single_channel_fallback_flag_block: np.ndarray
    gluing_split_altitude_m_block: np.ndarray
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


def get_method_v5_config(config: Mapping[str, Any]) -> MethodV5Configuration:
    """Parse and validate the complete productive method-v5 configuration."""
    inversion = _required_mapping(config, "inversion", "config")
    v5 = _required_mapping(inversion, "method_v5", "inversion")

    raw_tiers = v5.get("reference_tier_min_altitudes_m")
    if not isinstance(raw_tiers, list) or not raw_tiers:
        raise ValueError(
            "inversion.method_v5.reference_tier_min_altitudes_m must be a non-empty list."
        )
    tiers = tuple(
        _positive_float(value, f"inversion.method_v5.reference_tier_min_altitudes_m[{index}]")
        for index, value in enumerate(raw_tiers)
    )
    if any(next_value >= value for value, next_value in zip(tiers, tiers[1:], strict=False)):
        raise ValueError(
            "inversion.method_v5.reference_tier_min_altitudes_m must be strictly descending."
        )

    search_max = _positive_float(
        v5.get("reference_search_max_m"),
        "inversion.method_v5.reference_search_max_m",
    )
    if search_max <= max(tiers):
        raise ValueError("method-v5 reference_search_max_m must exceed the highest tier minimum.")
    path_start = _positive_float(
        v5.get("path_start_altitude_m"),
        "inversion.method_v5.path_start_altitude_m",
        allow_zero=True,
    )

    raw_fractions = v5.get("residual_aerosol_fractions")
    if not isinstance(raw_fractions, list) or not raw_fractions:
        raise ValueError(
            "inversion.method_v5.residual_aerosol_fractions must be a non-empty list."
        )
    fractions = tuple(
        _positive_float(
            value,
            f"inversion.method_v5.residual_aerosol_fractions[{index}]",
            allow_zero=True,
        )
        for index, value in enumerate(raw_fractions)
    )
    if 0.0 not in fractions:
        raise ValueError("method-v5 residual_aerosol_fractions must include the nominal f=0 scenario.")
    if len(set(fractions)) != len(fractions):
        raise ValueError("method-v5 residual_aerosol_fractions must not contain duplicates.")

    uncertainty_mode = str(v5.get("uncertainty_mode", "")).strip()
    if uncertainty_mode not in {"independent", "fully_correlated"}:
        raise ValueError(
            "inversion.method_v5.uncertainty_mode must be 'independent' or 'fully_correlated'."
        )

    raw_schedule = v5.get("progressive_grid_schedule")
    if not isinstance(raw_schedule, list) or not raw_schedule:
        raise ValueError(
            "inversion.method_v5.progressive_grid_schedule must be a non-empty list."
        )
    schedule_rows: list[tuple[float, float]] = []
    for index, row in enumerate(raw_schedule):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(
                "Each inversion.method_v5.progressive_grid_schedule row must contain "
                "[min_altitude_m, resolution_m]."
            )
        start = _positive_float(
            row[0],
            f"inversion.method_v5.progressive_grid_schedule[{index}][0]",
            allow_zero=True,
        )
        width = _positive_float(
            row[1],
            f"inversion.method_v5.progressive_grid_schedule[{index}][1]",
        )
        schedule_rows.append((start, width))
    if any(
        right[0] <= left[0]
        for left, right in zip(schedule_rows, schedule_rows[1:], strict=False)
    ):
        raise ValueError("method-v5 progressive-grid altitude thresholds must increase strictly.")

    return MethodV5Configuration(
        reference_tier_min_altitudes_m=tiers,
        reference_search_max_m=search_max,
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


def retrieve_wavelength_method_v5(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> MethodV5WavelengthProduct:
    """Run productive method v5 for every temporal block of one wavelength."""
    v5_cfg = get_method_v5_config(config)
    kfs_cfg = get_kfs_config(config)
    fit_cfg = get_molecular_fit_config(config)
    inputs, glued, molecular = prepare_wavelength_state(
        ds_l1, int(wavelength_nm), altitude_m, config, logger
    )

    grid = build_progressive_grid(altitude_m, v5_cfg.progressive_grid_schedule)
    molecular_beta = aggregate_to_progressive_grid(
        np.asarray(molecular.backscatter, dtype=np.float64),
        grid,
        require_positive=True,
    )
    molecular_alpha = aggregate_to_progressive_grid(
        np.asarray(molecular.extinction, dtype=np.float64),
        grid,
        require_positive=True,
    )
    if not np.all(molecular_beta.valid) or not np.all(molecular_alpha.valid):
        raise ValueError("Molecular state is not valid on every method-v5 progressive cell.")

    block_time = np.asarray(inputs.block_time).astype("datetime64[ns]")
    n_block = block_time.size
    n_altitude = grid.n_cells
    fractions = np.asarray(v5_cfg.residual_aerosol_fractions, dtype=np.float64)
    n_fraction = fractions.size
    iterations = int(kfs_cfg["monte_carlo_iterations"])

    block_start = np.full(n_block, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    block_end = np.full_like(block_start, np.datetime64("NaT", "ns"))
    times = np.asarray(ds_l1["time"].values)
    for block_index, group in enumerate(inputs.block_groups):
        if group.size:
            block_start[block_index] = _datetime64ns(times[group[0]])
            block_end[block_index] = _datetime64ns(times[group[-1]])

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
    reference_tier_min = np.full(n_block, np.nan, dtype=np.float64)
    reference_tier_index = np.full(n_block, -1, dtype=np.int16)
    reference_fallback = np.zeros(n_block, dtype=np.int8)
    path_top = np.full(n_block, np.nan, dtype=np.float64)
    selection_success_fraction = np.full(n_block, np.nan, dtype=np.float64)
    selected_reference_mc = np.full((n_block, iterations), np.nan, dtype=np.float64)
    selected_tier_min_mc = np.full((n_block, iterations), np.nan, dtype=np.float64)
    selected_tier_index_mc = np.full((n_block, iterations), -1, dtype=np.int16)
    retrieval_success = np.zeros(n_block, dtype=np.int8)

    for block_index in range(n_block):
        if int(glued.retrieval_input_valid_flag[block_index]) != 1:
            continue
        native_signal = np.asarray(glued.range_corrected_signal[block_index], dtype=np.float64)
        native_error = np.asarray(glued.range_corrected_signal_error[block_index], dtype=np.float64)
        prepared = prepare_high_column_profile(
            range_corrected_signal=native_signal,
            range_corrected_signal_error=native_error,
            molecular_backscatter=np.asarray(molecular.backscatter, dtype=np.float64),
            altitude_m=altitude_m,
            uncertainty_mode=v5_cfg.uncertainty_mode,
            schedule=v5_cfg.progressive_grid_schedule,
        )
        if not np.array_equal(prepared.grid.altitude_m, grid.altitude_m):
            raise RuntimeError("Method-v5 progressive-grid geometry changed between blocks.")
        rcs_block[block_index] = prepared.range_corrected_signal
        rcs_error_block[block_index] = prepared.range_corrected_signal_error
        top_index = contiguous_usable_top_index(
            prepared,
            path_start_altitude_m=v5_cfg.path_start_altitude_m,
        )
        if top_index is not None:
            path_top[block_index] = float(grid.altitude_m[top_index])

        seed = (
            int(kfs_cfg["random_seed"])
            + 100_000 * int(wavelength_nm)
            + int(block_index)
        )
        try:
            result = retrieve_method_v5_rnd(
                range_corrected_signal=native_signal,
                range_corrected_signal_error=native_error,
                molecular_backscatter=np.asarray(molecular.backscatter, dtype=np.float64),
                simulated_molecular_range_corrected_signal=np.asarray(
                    molecular.simulated_range_corrected_signal, dtype=np.float64
                ),
                altitude_m=altitude_m,
                aerosol_lidar_ratio_sr=float(molecular.lidar_ratio_assumed_sr),
                aerosol_lidar_ratio_std_sr=float(molecular.lidar_ratio_std_sr),
                residual_fractions=fractions,
                n_iterations=iterations,
                beta_ref_relative_std=float(kfs_cfg["beta_ref_relative_std"]),
                min_lidar_ratio_sr=float(kfs_cfg["min_lidar_ratio_sr"]),
                allow_negative_aerosol=bool(kfs_cfg["allow_negative_aerosol"]),
                seed=seed,
                max_relative_slope=float(fit_cfg["max_relative_slope"]),
                max_relative_variance=float(fit_cfg["max_relative_variance"]),
                min_valid_fraction=float(fit_cfg["min_valid_fraction"]),
                uncertainty_mode=v5_cfg.uncertainty_mode,
                progressive_grid_schedule=v5_cfg.progressive_grid_schedule,
                reference_tier_min_altitudes_m=v5_cfg.reference_tier_min_altitudes_m,
                search_max_altitude_m=v5_cfg.reference_search_max_m,
                rayleigh_window_m=float(fit_cfg["ref_window_m"]),
                path_start_altitude_m=v5_cfg.path_start_altitude_m,
            )
        except ValueError as exc:
            logger.warning(
                "  -> %d nm v5 block %d unsupported: %s",
                int(wavelength_nm),
                int(block_index),
                exc,
            )
            continue

        selected = result.selected_reference
        nominal = np.asarray(
            fernald_inversion(
                result.prepared.range_corrected_signal,
                result.prepared.grid.altitude_m,
                result.prepared.molecular_backscatter,
                float(molecular.lidar_ratio_assumed_sr),
                float(result.prepared.molecular_backscatter[selected.cell_index]),
                int(selected.cell_index),
                altitude_units="m",
                min_lidar_ratio=float(kfs_cfg["min_lidar_ratio_sr"]),
                allow_negative_aerosol=bool(kfs_cfg["allow_negative_aerosol"]),
                mode="backward",
            ),
            dtype=np.float64,
        )
        beta_nominal[block_index] = nominal
        alpha_nominal[block_index] = nominal * float(molecular.lidar_ratio_assumed_sr)
        retrieval_success[block_index] = 1
        reference_altitude[block_index] = float(selected.altitude_m)
        reference_tier_min[block_index] = float(result.selected_reference_tier_min_altitude_m)
        reference_tier_index[block_index] = int(result.selected_reference_tier_index)
        reference_fallback[block_index] = int(result.selected_reference_fallback_used)
        selection_success_fraction[block_index] = float(
            result.monte_carlo.selection_success_fraction
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
        selected_tier_min_mc[block_index] = (
            result.monte_carlo.selected_reference_tier_min_altitude_m_samples
        )
        selected_tier_index_mc[block_index] = (
            result.monte_carlo.selected_reference_tier_index_samples
        )

    support_count, support_fraction = _period_support(beta_nominal)
    supported = np.flatnonzero(support_count > 0)
    retrieval_top = float(grid.altitude_m[supported[-1]]) if supported.size else np.nan

    return MethodV5WavelengthProduct(
        wavelength_nm=int(wavelength_nm),
        block_time=block_time,
        block_start_utc=block_start,
        block_end_utc=block_end,
        altitude_m=np.asarray(grid.altitude_m, dtype=np.float64),
        effective_vertical_resolution_m=np.asarray(
            grid.effective_resolution_m, dtype=np.float64
        ),
        source_bin_count=np.asarray(grid.source_count, dtype=np.int32),
        molecular_backscatter=np.asarray(molecular_beta.values, dtype=np.float64),
        molecular_extinction=np.asarray(molecular_alpha.values, dtype=np.float64),
        lidar_ratio_assumed_sr=float(molecular.lidar_ratio_assumed_sr),
        lidar_ratio_std_sr=float(molecular.lidar_ratio_std_sr),
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
        reference_tier_min_altitude_m_block=reference_tier_min,
        reference_tier_index_block=reference_tier_index,
        reference_fallback_used_block=reference_fallback,
        contiguous_path_top_altitude_m_block=path_top,
        selection_success_fraction_block=selection_success_fraction,
        selected_reference_altitude_m_mc=selected_reference_mc,
        selected_reference_tier_min_altitude_m_mc=selected_tier_min_mc,
        selected_reference_tier_index_mc=selected_tier_index_mc,
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
        gluing_split_altitude_m_block=np.asarray(glued.split_altitude_m, dtype=np.float64),
        gluing_correlation_block=np.asarray(glued.correlation, dtype=np.float64),
        gluing_relative_rmse_block=np.asarray(glued.relative_rmse, dtype=np.float64),
        gluing_relative_bias_block=np.asarray(glued.relative_bias, dtype=np.float64),
    )


def _stack(results: Sequence[MethodV5WavelengthProduct], name: str) -> np.ndarray:
    return np.stack([np.asarray(getattr(result, name)) for result in results], axis=0)


def _stack_block(results: Sequence[MethodV5WavelengthProduct], name: str) -> np.ndarray:
    return np.stack([np.asarray(getattr(result, name)) for result in results], axis=1)


def build_method_v5_level2_dataset(
    ds_l1: xr.Dataset,
    altitude_m: np.ndarray,
    source_file: str | Path,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> xr.Dataset:
    """Build the canonical multispectral schema-4 method-v5 Level 2 product."""
    wavelengths = tuple(int(value) for value in get_wavelengths_to_process(config))
    v5_cfg = get_method_v5_config(config)
    kfs_cfg = get_kfs_config(config)
    block_minutes = int(get_block_average_minutes(config))

    results = tuple(
        retrieve_wavelength_method_v5(ds_l1, wavelength, altitude_m, config, logger)
        for wavelength in wavelengths
    )
    first = results[0]
    for result in results[1:]:
        if not np.array_equal(result.block_time, first.block_time):
            raise RuntimeError("Method-v5 block-time geometry differs between wavelengths.")
        if not np.array_equal(result.altitude_m, first.altitude_m):
            raise RuntimeError("Method-v5 progressive altitude differs between wavelengths.")

    n_iterations = int(kfs_cfg["monte_carlo_iterations"])
    residual_fractions = np.asarray(v5_cfg.residual_aerosol_fractions, dtype=np.float64)
    success_fraction = np.asarray(
        [np.mean(result.retrieval_success_flag_block == 1) for result in results],
        dtype=np.float64,
    )
    processed = np.asarray(
        [result.wavelength_nm for result in results if np.any(result.retrieval_success_flag_block == 1)],
        dtype=np.int32,
    )
    failed = np.asarray(
        [result.wavelength_nm for result in results if not np.any(result.retrieval_success_flag_block == 1)],
        dtype=np.int32,
    )
    if processed.size == 0:
        raise ValueError("Method v5 produced no valid retrieval block for any requested wavelength.")
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
                ("wavelength", "altitude"),
                _stack(results, "molecular_backscatter"),
            ),
            "molecular_extinction": (
                ("wavelength", "altitude"),
                _stack(results, "molecular_extinction"),
            ),
            "lidar_ratio_assumed_sr": (
                ("wavelength",),
                np.asarray([result.lidar_ratio_assumed_sr for result in results], dtype=np.float64),
            ),
            "lidar_ratio_std_sr": (
                ("wavelength",),
                np.asarray([result.lidar_ratio_std_sr for result in results], dtype=np.float64),
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
                np.asarray([result.retrieval_top_altitude_m for result in results], dtype=np.float64),
            ),
            "rayleigh_reference_altitude_m_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "reference_altitude_m_block"),
            ),
            "rayleigh_reference_tier_min_altitude_m_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "reference_tier_min_altitude_m_block"),
            ),
            "rayleigh_reference_tier_index_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "reference_tier_index_block").astype(np.int16),
            ),
            "rayleigh_reference_fallback_used_block": (
                ("block_time", "wavelength"),
                _stack_block(results, "reference_fallback_used_block").astype(np.int8),
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
            "selected_reference_tier_min_altitude_m_mc": (
                ("block_time", "wavelength", "mc_iteration"),
                _stack_block(results, "selected_reference_tier_min_altitude_m_mc"),
            ),
            "selected_reference_tier_index_mc": (
                ("block_time", "wavelength", "mc_iteration"),
                _stack_block(results, "selected_reference_tier_index_mc").astype(np.int16),
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
            "gluing_split_altitude_m": (
                ("block_time", "wavelength"),
                _stack_block(results, "gluing_split_altitude_m_block"),
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
            "title": "MILGRAU Level 2 method-v5 elastic optical product",
            "source_level1": str(Path(source_file)),
            "level2_product_schema_version": LEVEL2_PRODUCT_SCHEMA_VERSION,
            "level2_product_schema_change": LEVEL2_PRODUCT_SCHEMA_CHANGE,
            "level2_retrieval_method_version": LEVEL2_RETRIEVAL_METHOD_VERSION,
            "product_status": "success" if completeness == "complete" else "partial",
            "product_completeness": completeness,
            "configured_block_minutes": block_minutes,
            "reference_tier_min_altitudes_m": ",".join(
                f"{value:g}" for value in v5_cfg.reference_tier_min_altitudes_m
            ),
            "reference_search_max_m": v5_cfg.reference_search_max_m,
            "reference_selection_policy": (
                "highest_supported_declared_tier_then_minimum_existing_rayleigh_cost"
            ),
            "reference_tier_interpretation": (
                "search-domain preference only; not an aerosol-free or molecular-purity criterion"
            ),
            "progressive_grid_schedule": ";".join(
                f"{start:g}:{width:g}" for start, width in v5_cfg.progressive_grid_schedule
            ),
            "uncertainty_mode": v5_cfg.uncertainty_mode,
            "boundary_nominal_residual_fraction": 0.0,
            "boundary_systematic_scenarios": ",".join(
                f"{value:g}" for value in v5_cfg.residual_aerosol_fractions
            ),
            "support_fraction_denominator": "all configured temporal blocks",
            "period_mean_semantics": (
                "finite-only altitude-by-altitude mean; inspect period_support_count/fraction jointly"
            ),
            "mc_valid_fraction_semantics": (
                "finite selection-aware Monte-Carlo realization fraction; diagnostic only, no cutoff"
            ),
            "monte_carlo_iterations": n_iterations,
            "monte_carlo_random_seed": int(kfs_cfg["random_seed"]),
            "kfs_beta_ref_relative_std": float(kfs_cfg["beta_ref_relative_std"]),
            "kfs_min_lidar_ratio_sr": float(kfs_cfg["min_lidar_ratio_sr"]),
            "kfs_allow_negative_aerosol": int(bool(kfs_cfg["allow_negative_aerosol"])),
            **elastic_inversion_algorithm_metadata(),
        },
    )

    ds["altitude"].attrs.update({"units": "m", "positive": "up"})
    ds["effective_vertical_resolution_m"].attrs.update(
        {"units": "m", "long_name": "effective vertical cell width on the progressive retrieval grid"}
    )
    ds["source_bin_count"].attrs["long_name"] = (
        "number of contiguous native Level-1 altitude bins represented by each progressive cell"
    )
    ds["aerosol_backscatter_mean"].attrs.update({"units": "m-1 sr-1"})
    ds["aerosol_extinction_mean"].attrs.update({"units": "m-1"})
    ds["aerosol_backscatter_nominal_block"].attrs.update({"units": "m-1 sr-1"})
    ds["aerosol_extinction_nominal_block"].attrs.update({"units": "m-1"})
    ds["period_support_fraction"].attrs["long_name"] = (
        "fraction of configured temporal blocks contributing a finite nominal method-v5 retrieval"
    )
    ds["selection_success_fraction_block"].attrs["long_name"] = (
        "fraction of Monte-Carlo realizations in which tiered reference selection succeeds"
    )
    return ds


__all__ = [
    "MethodV5Configuration",
    "MethodV5WavelengthProduct",
    "build_method_v5_level2_dataset",
    "get_method_v5_config",
    "retrieve_wavelength_method_v5",
]
