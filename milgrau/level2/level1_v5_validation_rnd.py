"""Real-Level-1 validation harness for elastic method-v5 R&D.

This module is intentionally not part of productive LEBEAR method v4. It reuses
productive signal selection/gluing and the productive molecular model, then
runs the experimental method-v5 chain block by block. The goal is to compare v4
and v5 from the same Level-1 input without treating v4 as physical truth.

Residual-aerosol boundary fraction ``f`` remains an outer systematic scenario.
The legacy extra ``beta_ref_relative_std`` term is disabled by default because
its independent physical meaning is not established once native signal noise,
reference re-selection, tier fallback and explicit ``f`` sensitivity are
propagated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from milgrau.level2.block_average import block_groups
from milgrau.level2.config import (
    get_block_average_minutes,
    get_kfs_config,
    get_molecular_fit_config,
)
from milgrau.level2.high_column_rnd import contiguous_usable_top_index
from milgrau.level2.kfs import fernald_inversion
from milgrau.level2.method_v5_rnd import retrieve_method_v5_rnd
from milgrau.level2.retrieval import process_wavelength


@dataclass(frozen=True, slots=True)
class V5BlockValidationSummary:
    """Audit summary for one productive Level-1 temporal block."""

    block_index: int
    block_time_utc: str
    block_start_utc: str
    block_end_utc: str
    retrieval_input_valid: bool
    signal_source_flag: int
    v4_retrieval_success: bool
    v4_reference_altitude_m: float
    v5_success: bool
    v5_failure: str
    v5_reference_altitude_m: float
    v5_retrieval_top_altitude_m: float
    v5_reference_tier_min_altitude_m: float
    v5_reference_tier_index: int
    v5_reference_fallback_used: bool
    v5_reference_effective_resolution_m: float
    v5_reference_diagnostic_cost: float
    v5_accepted_admissible_candidates: int
    v5_contiguous_top_altitude_m: float
    mc_selection_success_fraction: float
    mc_reference_altitude_median: float
    mc_reference_altitude_std: float
    mc_reference_tier_counts: dict[str, int]
    mc_lower_valid_fraction_min: float
    v4_v5_lower_relative_l2: float
    residual_fraction_lower_relative_l2: dict[str, float]


@dataclass(frozen=True, slots=True)
class Level1V5ValidationSummary:
    """One-wavelength validation summary from a real Level-1 dataset."""

    wavelength_nm: int
    analog_channel: str | None
    photon_channel: str | None
    n_blocks: int
    residual_fractions: tuple[float, ...]
    n_iterations: int
    beta_ref_relative_std: float
    uncertainty_mode: str
    reference_tier_min_altitudes_m: tuple[float, ...]
    blocks: tuple[V5BlockValidationSummary, ...]


def _time_string(value: Any) -> str:
    """Return a stable UTC-like ISO representation for one numpy datetime."""
    try:
        return str(np.datetime_as_string(np.datetime64(value), unit="s"))
    except Exception:
        return str(value)


def _relative_l2(reference: np.ndarray, comparison: np.ndarray) -> float:
    """Return finite-only relative L2 distance, or NaN when undefined."""
    ref = np.asarray(reference, dtype=np.float64)
    cmp = np.asarray(comparison, dtype=np.float64)
    if ref.shape != cmp.shape:
        raise ValueError("relative-L2 inputs must have identical shapes.")
    valid = np.isfinite(ref) & np.isfinite(cmp)
    if not np.any(valid):
        return float("nan")
    denominator = float(np.linalg.norm(ref[valid]))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(np.linalg.norm(cmp[valid] - ref[valid]) / denominator)


def _lower_common_samples(
    native_altitude_m: np.ndarray,
    native_values: np.ndarray,
    progressive_altitude_m: np.ndarray,
    progressive_values: np.ndarray,
    *,
    min_altitude_m: float = 600.0,
    max_altitude_m: float = 6000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair lower-column samples without interpolating either representation."""
    native_altitude = np.asarray(native_altitude_m, dtype=np.float64)
    native = np.asarray(native_values, dtype=np.float64)
    progressive_altitude = np.asarray(progressive_altitude_m, dtype=np.float64)
    progressive = np.asarray(progressive_values, dtype=np.float64)
    if native_altitude.ndim != 1 or progressive_altitude.ndim != 1:
        raise ValueError("altitude arrays must be one-dimensional.")
    if native.shape != native_altitude.shape or progressive.shape != progressive_altitude.shape:
        raise ValueError("profile arrays must match their altitude arrays.")

    if native_altitude.size < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    tolerance = max(float(np.median(np.diff(native_altitude))) * 0.1, 1e-6)
    ref_samples: list[float] = []
    cmp_samples: list[float] = []
    for index, altitude in enumerate(progressive_altitude):
        if altitude < float(min_altitude_m) or altitude > float(max_altitude_m):
            continue
        native_index = int(np.argmin(np.abs(native_altitude - altitude)))
        if abs(float(native_altitude[native_index] - altitude)) > tolerance:
            continue
        ref_samples.append(float(native[native_index]))
        cmp_samples.append(float(progressive[index]))
    return (
        np.asarray(ref_samples, dtype=np.float64),
        np.asarray(cmp_samples, dtype=np.float64),
    )


def _lower_relative_l2_between_grids(
    native_altitude_m: np.ndarray,
    native_values: np.ndarray,
    progressive_altitude_m: np.ndarray,
    progressive_values: np.ndarray,
) -> float:
    native, progressive = _lower_common_samples(
        native_altitude_m,
        native_values,
        progressive_altitude_m,
        progressive_values,
    )
    return _relative_l2(native, progressive) if native.size else float("nan")


def _block_periods(
    ds_l1: xr.Dataset,
    *,
    minutes: int,
    expected_blocks: int,
) -> list[tuple[str, str]]:
    """Return actual first/last Level-1 sample times contributing to each block."""
    if "time" not in ds_l1:
        return [("", "")] * expected_blocks
    _, groups = block_groups(ds_l1["time"].values, int(minutes))
    if len(groups) != expected_blocks:
        return [("", "")] * expected_blocks
    periods: list[tuple[str, str]] = []
    times = np.asarray(ds_l1["time"].values)
    for group in groups:
        if group.size == 0:
            periods.append(("", ""))
            continue
        block_times = times[group]
        periods.append((_time_string(block_times[0]), _time_string(block_times[-1])))
    return periods


def _tier_counts(samples: np.ndarray) -> dict[str, int]:
    """Count finite MC tier selections using altitude minima as audit keys."""
    values = np.asarray(samples, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {}
    unique, counts = np.unique(finite, return_counts=True)
    return {
        f"{float(value):.6g}": int(count)
        for value, count in zip(unique, counts, strict=True)
    }


def _empty_block_summary(
    *,
    block_index: int,
    block_time_utc: str,
    block_start_utc: str,
    block_end_utc: str,
    retrieval_input_valid: bool,
    signal_source_flag: int,
    v4_retrieval_success: bool,
    v4_reference_altitude_m: float,
    failure: str,
) -> V5BlockValidationSummary:
    return V5BlockValidationSummary(
        block_index=block_index,
        block_time_utc=block_time_utc,
        block_start_utc=block_start_utc,
        block_end_utc=block_end_utc,
        retrieval_input_valid=retrieval_input_valid,
        signal_source_flag=signal_source_flag,
        v4_retrieval_success=v4_retrieval_success,
        v4_reference_altitude_m=v4_reference_altitude_m,
        v5_success=False,
        v5_failure=failure,
        v5_reference_altitude_m=float("nan"),
        v5_retrieval_top_altitude_m=float("nan"),
        v5_reference_tier_min_altitude_m=float("nan"),
        v5_reference_tier_index=-1,
        v5_reference_fallback_used=False,
        v5_reference_effective_resolution_m=float("nan"),
        v5_reference_diagnostic_cost=float("nan"),
        v5_accepted_admissible_candidates=0,
        v5_contiguous_top_altitude_m=float("nan"),
        mc_selection_success_fraction=float("nan"),
        mc_reference_altitude_median=float("nan"),
        mc_reference_altitude_std=float("nan"),
        mc_reference_tier_counts={},
        mc_lower_valid_fraction_min=float("nan"),
        v4_v5_lower_relative_l2=float("nan"),
        residual_fraction_lower_relative_l2={},
    )


def validate_level1_wavelength_v5_rnd(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
    *,
    residual_fractions: Sequence[float] = (0.0, 0.02, 0.05),
    n_iterations: int | None = None,
    beta_ref_relative_std: float = 0.0,
    uncertainty_mode: str = "independent",
    reference_tier_min_altitudes_m: Sequence[float] = (10_000.0, 9_000.0, 8_000.0, 6_000.0),
) -> Level1V5ValidationSummary:
    """Run productive preprocessing plus experimental v5 on every valid block."""
    altitude = np.asarray(altitude_m, dtype=np.float64)
    productive = process_wavelength(ds_l1, int(wavelength_nm), altitude, config, logger)
    fit_cfg = get_molecular_fit_config(config)
    kfs_cfg = get_kfs_config(config)
    minutes = int(get_block_average_minutes(config))
    periods = _block_periods(
        ds_l1,
        minutes=minutes,
        expected_blocks=int(productive.block_time.size),
    )
    iterations = (
        int(kfs_cfg["monte_carlo_iterations"])
        if n_iterations is None
        else int(n_iterations)
    )
    if iterations <= 0:
        raise ValueError("n_iterations must be positive.")
    fractions = tuple(float(value) for value in residual_fractions)
    if not fractions or 0.0 not in fractions:
        raise ValueError("residual_fractions must include the nominal f=0 scenario.")
    tiers = tuple(float(value) for value in reference_tier_min_altitudes_m)

    summaries: list[V5BlockValidationSummary] = []
    for block_index in range(productive.block_time.size):
        block_time = _time_string(productive.block_time[block_index])
        block_start, block_end = periods[block_index]
        input_valid = bool(
            int(productive.signal_selection.retrieval_input_valid_flag_block[block_index]) == 1
        )
        source_flag = int(productive.signal_selection.source_flag_block[block_index])
        v4_success = bool(int(productive.optical.retrieval_success_flag[block_index]) == 1)
        v4_reference = float(productive.rayleigh.reference_altitude_m_block[block_index])
        if not input_valid:
            summaries.append(
                _empty_block_summary(
                    block_index=block_index,
                    block_time_utc=block_time,
                    block_start_utc=block_start,
                    block_end_utc=block_end,
                    retrieval_input_valid=False,
                    signal_source_flag=source_flag,
                    v4_retrieval_success=v4_success,
                    v4_reference_altitude_m=v4_reference,
                    failure="productive retrieval input invalid before v5",
                )
            )
            continue

        signal = np.asarray(
            productive.glued.range_corrected_signal_block[block_index], dtype=np.float64
        )
        signal_error = np.asarray(
            productive.glued.range_corrected_signal_error_block[block_index], dtype=np.float64
        )
        try:
            seed = int(kfs_cfg["random_seed"]) + 100_000 * int(wavelength_nm) + block_index
            result = retrieve_method_v5_rnd(
                range_corrected_signal=signal,
                range_corrected_signal_error=signal_error,
                molecular_backscatter=np.asarray(
                    productive.molecular.backscatter, dtype=np.float64
                ),
                simulated_molecular_range_corrected_signal=np.asarray(
                    productive.molecular.simulated_range_corrected_signal,
                    dtype=np.float64,
                ),
                altitude_m=altitude,
                aerosol_lidar_ratio_sr=float(productive.kfs.lidar_ratio_assumed_sr),
                aerosol_lidar_ratio_std_sr=float(productive.kfs.lidar_ratio_std_sr),
                residual_fractions=fractions,
                n_iterations=iterations,
                beta_ref_relative_std=float(beta_ref_relative_std),
                min_lidar_ratio_sr=float(kfs_cfg["min_lidar_ratio_sr"]),
                allow_negative_aerosol=bool(kfs_cfg["allow_negative_aerosol"]),
                seed=seed,
                max_relative_slope=float(fit_cfg["max_relative_slope"]),
                max_relative_variance=float(fit_cfg["max_relative_variance"]),
                min_valid_fraction=float(fit_cfg["min_valid_fraction"]),
                uncertainty_mode=uncertainty_mode,  # type: ignore[arg-type]
                reference_tier_min_altitudes_m=tiers,
                rayleigh_window_m=float(fit_cfg["ref_window_m"]),
            )

            selected = result.selected_reference
            prepared = result.prepared
            nominal_beta = fernald_inversion(
                prepared.range_corrected_signal,
                prepared.grid.altitude_m,
                prepared.molecular_backscatter,
                float(productive.kfs.lidar_ratio_assumed_sr),
                float(prepared.molecular_backscatter[selected.cell_index]),
                int(selected.cell_index),
                altitude_units="m",
                min_lidar_ratio=float(kfs_cfg["min_lidar_ratio_sr"]),
                allow_negative_aerosol=bool(kfs_cfg["allow_negative_aerosol"]),
                mode="backward",
            )
            v4_v5_l2 = _lower_relative_l2_between_grids(
                altitude,
                np.asarray(
                    productive.optical.aerosol_backscatter_block[block_index],
                    dtype=np.float64,
                ),
                prepared.grid.altitude_m,
                np.asarray(nominal_beta, dtype=np.float64),
            )

            top_index = contiguous_usable_top_index(prepared, path_start_altitude_m=600.0)
            top_altitude = (
                float(prepared.grid.altitude_m[top_index])
                if top_index is not None
                else float("nan")
            )
            selected_samples = np.asarray(
                result.monte_carlo.selected_reference_altitude_m_samples,
                dtype=np.float64,
            )
            finite_selected = selected_samples[np.isfinite(selected_samples)]
            mc_reference_median = (
                float(np.median(finite_selected)) if finite_selected.size else float("nan")
            )
            mc_reference_std = (
                float(np.std(finite_selected)) if finite_selected.size else float("nan")
            )

            lower_grid = (
                (result.monte_carlo.altitude_m >= 600.0)
                & (result.monte_carlo.altitude_m <= 6000.0)
            )
            nominal_fraction_index = fractions.index(0.0)
            nominal_valid_fraction = np.asarray(
                result.monte_carlo.aerosol_backscatter_valid_fraction[
                    nominal_fraction_index
                ],
                dtype=np.float64,
            )
            lower_valid = nominal_valid_fraction[lower_grid]
            lower_valid = lower_valid[np.isfinite(lower_valid)]
            lower_valid_min = (
                float(np.min(lower_valid)) if lower_valid.size else float("nan")
            )

            fraction_sensitivity: dict[str, float] = {}
            nominal_mean = np.asarray(
                result.monte_carlo.aerosol_backscatter_mean[nominal_fraction_index],
                dtype=np.float64,
            )
            for fraction_index, fraction in enumerate(fractions):
                if fraction == 0.0:
                    continue
                comparison = np.asarray(
                    result.monte_carlo.aerosol_backscatter_mean[fraction_index],
                    dtype=np.float64,
                )
                fraction_sensitivity[f"{fraction:.6g}"] = _relative_l2(
                    nominal_mean[lower_grid], comparison[lower_grid]
                )

            summaries.append(
                V5BlockValidationSummary(
                    block_index=block_index,
                    block_time_utc=block_time,
                    block_start_utc=block_start,
                    block_end_utc=block_end,
                    retrieval_input_valid=True,
                    signal_source_flag=source_flag,
                    v4_retrieval_success=v4_success,
                    v4_reference_altitude_m=v4_reference,
                    v5_success=True,
                    v5_failure="",
                    v5_reference_altitude_m=float(selected.altitude_m),
                    v5_retrieval_top_altitude_m=float(selected.altitude_m),
                    v5_reference_tier_min_altitude_m=float(
                        result.selected_reference_tier_min_altitude_m
                    ),
                    v5_reference_tier_index=int(result.selected_reference_tier_index),
                    v5_reference_fallback_used=bool(
                        result.selected_reference_fallback_used
                    ),
                    v5_reference_effective_resolution_m=float(
                        selected.effective_resolution_m
                    ),
                    v5_reference_diagnostic_cost=float(
                        selected.native_rayleigh_candidate.diagnostic_cost
                    ),
                    v5_accepted_admissible_candidates=len(
                        result.reference_catalogue.accepted_and_admissible
                    ),
                    v5_contiguous_top_altitude_m=top_altitude,
                    mc_selection_success_fraction=float(
                        result.monte_carlo.selection_success_fraction
                    ),
                    mc_reference_altitude_median=mc_reference_median,
                    mc_reference_altitude_std=mc_reference_std,
                    mc_reference_tier_counts=_tier_counts(
                        result.monte_carlo.selected_reference_tier_min_altitude_m_samples
                    ),
                    mc_lower_valid_fraction_min=lower_valid_min,
                    v4_v5_lower_relative_l2=v4_v5_l2,
                    residual_fraction_lower_relative_l2=fraction_sensitivity,
                )
            )
        except Exception as exc:
            summaries.append(
                _empty_block_summary(
                    block_index=block_index,
                    block_time_utc=block_time,
                    block_start_utc=block_start,
                    block_end_utc=block_end,
                    retrieval_input_valid=True,
                    signal_source_flag=source_flag,
                    v4_retrieval_success=v4_success,
                    v4_reference_altitude_m=v4_reference,
                    failure=f"{type(exc).__name__}: {exc}",
                )
            )

    return Level1V5ValidationSummary(
        wavelength_nm=int(wavelength_nm),
        analog_channel=productive.glued.analog_channel,
        photon_channel=productive.glued.photon_channel,
        n_blocks=int(productive.block_time.size),
        residual_fractions=fractions,
        n_iterations=iterations,
        beta_ref_relative_std=float(beta_ref_relative_std),
        uncertainty_mode=str(uncertainty_mode),
        reference_tier_min_altitudes_m=tiers,
        blocks=tuple(summaries),
    )


def _json_safe(value: Any) -> Any:
    """Convert dataclass output to strict JSON without NaN/Inf literals."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def validation_summary_dict(summary: Level1V5ValidationSummary) -> dict[str, Any]:
    """Return one strict-JSON-safe mapping for FAIR evidence exports."""
    return _json_safe(asdict(summary))


def write_validation_summary_json(
    summary: Level1V5ValidationSummary,
    path: str | Path,
) -> Path:
    """Write one reproducible validation summary as strict JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(validation_summary_dict(summary), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "Level1V5ValidationSummary",
    "V5BlockValidationSummary",
    "validate_level1_wavelength_v5_rnd",
    "validation_summary_dict",
    "write_validation_summary_json",
]
