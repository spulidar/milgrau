"""Low-memory real-Level-1 profile product for method-v5 R&D.

This runner exists so high-column behavior can be inspected as profiles rather
than only scalar validation summaries. It keeps every configured temporal block
separate, records the exact contributing Level-1 sample period, and computes
period means only from blocks that actually support each altitude.

The output is explicitly R&D and does not replace productive Level-2 method v4.
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr

from milgrau.config.loader import load_config
from milgrau.level2.block_average import block_groups
from milgrau.level2.config import (
    get_block_average_minutes,
    get_kfs_config,
    get_molecular_fit_config,
)
from milgrau.level2.high_column_rnd import (
    V5_PROGRESSIVE_GRID_SCHEDULE,
    contiguous_usable_top_index,
)
from milgrau.level2.kfs import fernald_inversion
from milgrau.level2.method_v5_rnd import retrieve_method_v5_rnd
from milgrau.level2.retrieval import process_wavelength
from milgrau.level2.v5_validation_cli import _altitude_subset


def _profile_output_path(output_dir: Path, source: Path, wavelength_nm: int) -> Path:
    return output_dir / f"{source.stem}_method_v5_rnd_{int(wavelength_nm)}nm.nc"


def _finite_mean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return finite-only mean without emitting all-NaN warnings."""
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    count = np.count_nonzero(finite, axis=axis)
    total = np.sum(np.where(finite, array, 0.0), axis=axis)
    out = np.full(np.asarray(total).shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _period_support(nominal_profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return altitude-resolved contributing-block count and total-block fraction."""
    profiles = np.asarray(nominal_profiles, dtype=np.float64)
    if profiles.ndim != 2:
        raise ValueError("nominal_profiles must have shape (block, altitude).")
    count = np.count_nonzero(np.isfinite(profiles), axis=0).astype(np.int32)
    denominator = profiles.shape[0]
    fraction = (
        count.astype(np.float64) / float(denominator)
        if denominator > 0
        else np.full(count.shape, np.nan, dtype=np.float64)
    )
    return count, fraction


def _as_datetime64(value: Any) -> np.datetime64:
    try:
        return np.datetime64(value, "ns")
    except Exception:
        return np.datetime64("NaT", "ns")


def run_low_memory_v5_profiles(
    source: str | Path,
    *,
    wavelength_nm: int = 532,
    output_dir: str | Path = "v5_profile_results",
    config_path: str | Path = "config.yaml",
    station_config_path: str | Path | None = None,
    n_iterations: int = 60,
    residual_fractions: Sequence[float] = (0.0, 0.02, 0.05),
    reference_tier_min_altitudes_m: Sequence[float] = (
        10_000.0,
        9_000.0,
        8_000.0,
        6_000.0,
    ),
    beta_ref_relative_std: float = 0.0,
    max_altitude_m: float = 30_000.0,
    block_start: int = 0,
    block_stop: int | None = None,
) -> Path:
    """Write one block-resolved experimental v5 profile NetCDF."""
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = _profile_output_path(output_root, source_path, int(wavelength_nm))

    config = load_config(config_path, station_config_path)
    minutes = int(get_block_average_minutes(config))
    fit_cfg = get_molecular_fit_config(config)
    kfs_cfg = get_kfs_config(config)
    iterations = int(n_iterations)
    fractions = tuple(float(value) for value in residual_fractions)
    tiers = tuple(float(value) for value in reference_tier_min_altitudes_m)
    if iterations <= 0:
        raise ValueError("n_iterations must be positive.")
    if not fractions or 0.0 not in fractions:
        raise ValueError("residual_fractions must include f=0.")
    if not tiers:
        raise ValueError("reference_tier_min_altitudes_m must not be empty.")

    logger = logging.getLogger("milgrau.v5_profile")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    block_records: list[dict[str, Any]] = []
    output_altitude: np.ndarray | None = None
    effective_resolution: np.ndarray | None = None
    source_count: np.ndarray | None = None

    with xr.open_dataset(source_path, cache=False) as ds_source:
        ds, altitude_m = _altitude_subset(
            ds_source,
            max_altitude_m=float(max_altitude_m),
        )
        if "time" not in ds:
            raise KeyError("Level 1 dataset lacks required time coordinate.")
        _, groups = block_groups(ds["time"].values, minutes)
        first = max(int(block_start), 0)
        final = len(groups) if block_stop is None else min(int(block_stop), len(groups))
        if final <= first:
            raise ValueError(
                f"Empty block interval [{first}, {final}) for {len(groups)} available blocks."
            )

        for global_index in range(first, final):
            group = groups[global_index]
            times = np.asarray(ds["time"].values)[group]
            start_time = _as_datetime64(times[0]) if times.size else np.datetime64("NaT", "ns")
            end_time = _as_datetime64(times[-1]) if times.size else np.datetime64("NaT", "ns")
            logger.info(
                "%dnm | block %d/%d | %s to %s | profiles=%d",
                int(wavelength_nm),
                global_index + 1,
                len(groups),
                str(start_time),
                str(end_time),
                int(group.size),
            )
            ds_block = ds.isel(time=group)
            productive = process_wavelength(
                ds_block,
                int(wavelength_nm),
                altitude_m,
                config,
                logger,
            )
            if productive.block_time.size != 1:
                raise RuntimeError("v5 profile runner expected one productive block.")

            input_valid = bool(
                int(productive.signal_selection.retrieval_input_valid_flag_block[0]) == 1
            )
            record: dict[str, Any] = {
                "global_index": global_index,
                "block_start": start_time,
                "block_end": end_time,
                "retrieval_input_valid": input_valid,
                "signal_source_flag": int(productive.signal_selection.source_flag_block[0]),
                "v5_success": False,
                "reference_altitude_m": np.nan,
                "reference_tier_min_altitude_m": np.nan,
                "reference_tier_index": -1,
                "reference_fallback_used": False,
                "contiguous_path_top_altitude_m": np.nan,
                "selection_success_fraction": np.nan,
                "nominal_beta": None,
                "nominal_alpha": None,
                "mc_beta_mean": None,
                "mc_beta_std": None,
                "mc_beta_q025": None,
                "mc_beta_q975": None,
                "mc_alpha_mean": None,
                "mc_alpha_std": None,
                "mc_alpha_q025": None,
                "mc_alpha_q975": None,
                "mc_valid_fraction": None,
                "mc_reference_altitude": np.full(iterations, np.nan, dtype=np.float64),
                "mc_reference_tier_min": np.full(iterations, np.nan, dtype=np.float64),
            }
            if input_valid:
                signal = np.asarray(
                    productive.glued.range_corrected_signal_block[0], dtype=np.float64
                )
                signal_error = np.asarray(
                    productive.glued.range_corrected_signal_error_block[0], dtype=np.float64
                )
                try:
                    seed = (
                        int(kfs_cfg["random_seed"])
                        + 100_000 * int(wavelength_nm)
                        + global_index
                    )
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
                        altitude_m=altitude_m,
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
                        uncertainty_mode="independent",
                        reference_tier_min_altitudes_m=tiers,
                        rayleigh_window_m=float(fit_cfg["ref_window_m"]),
                    )
                    selected = result.selected_reference
                    prepared = result.prepared
                    if output_altitude is None:
                        output_altitude = np.asarray(
                            prepared.grid.altitude_m, dtype=np.float64
                        )
                        effective_resolution = np.asarray(
                            prepared.grid.effective_resolution_m, dtype=np.float64
                        )
                        source_count = np.asarray(
                            prepared.grid.source_count, dtype=np.int32
                        )
                    elif not np.array_equal(output_altitude, prepared.grid.altitude_m):
                        raise RuntimeError("progressive grid changed between temporal blocks.")

                    nominal_beta = np.asarray(
                        fernald_inversion(
                            prepared.range_corrected_signal,
                            prepared.grid.altitude_m,
                            prepared.molecular_backscatter,
                            float(productive.kfs.lidar_ratio_assumed_sr),
                            float(prepared.molecular_backscatter[selected.cell_index]),
                            int(selected.cell_index),
                            altitude_units="m",
                            min_lidar_ratio=float(kfs_cfg["min_lidar_ratio_sr"]),
                            allow_negative_aerosol=bool(
                                kfs_cfg["allow_negative_aerosol"]
                            ),
                            mode="backward",
                        ),
                        dtype=np.float64,
                    )
                    nominal_alpha = nominal_beta * float(
                        productive.kfs.lidar_ratio_assumed_sr
                    )
                    top_index = contiguous_usable_top_index(
                        prepared, path_start_altitude_m=600.0
                    )
                    record.update(
                        {
                            "v5_success": True,
                            "reference_altitude_m": float(selected.altitude_m),
                            "reference_tier_min_altitude_m": float(
                                result.selected_reference_tier_min_altitude_m
                            ),
                            "reference_tier_index": int(
                                result.selected_reference_tier_index
                            ),
                            "reference_fallback_used": bool(
                                result.selected_reference_fallback_used
                            ),
                            "contiguous_path_top_altitude_m": (
                                float(prepared.grid.altitude_m[top_index])
                                if top_index is not None
                                else np.nan
                            ),
                            "selection_success_fraction": float(
                                result.monte_carlo.selection_success_fraction
                            ),
                            "nominal_beta": nominal_beta,
                            "nominal_alpha": nominal_alpha,
                            "mc_beta_mean": np.asarray(
                                result.monte_carlo.aerosol_backscatter_mean,
                                dtype=np.float64,
                            ),
                            "mc_beta_std": np.asarray(
                                result.monte_carlo.aerosol_backscatter_random_std,
                                dtype=np.float64,
                            ),
                            "mc_beta_q025": np.asarray(
                                result.monte_carlo.aerosol_backscatter_random_q025,
                                dtype=np.float64,
                            ),
                            "mc_beta_q975": np.asarray(
                                result.monte_carlo.aerosol_backscatter_random_q975,
                                dtype=np.float64,
                            ),
                            "mc_alpha_mean": np.asarray(
                                result.monte_carlo.aerosol_extinction_mean,
                                dtype=np.float64,
                            ),
                            "mc_alpha_std": np.asarray(
                                result.monte_carlo.aerosol_extinction_random_std,
                                dtype=np.float64,
                            ),
                            "mc_alpha_q025": np.asarray(
                                result.monte_carlo.aerosol_extinction_random_q025,
                                dtype=np.float64,
                            ),
                            "mc_alpha_q975": np.asarray(
                                result.monte_carlo.aerosol_extinction_random_q975,
                                dtype=np.float64,
                            ),
                            "mc_valid_fraction": np.asarray(
                                result.monte_carlo.aerosol_backscatter_valid_fraction,
                                dtype=np.float64,
                            ),
                            "mc_reference_altitude": np.asarray(
                                result.monte_carlo.selected_reference_altitude_m_samples,
                                dtype=np.float64,
                            ),
                            "mc_reference_tier_min": np.asarray(
                                result.monte_carlo.selected_reference_tier_min_altitude_m_samples,
                                dtype=np.float64,
                            ),
                        }
                    )
                except ValueError as exc:
                    logger.warning(
                        "%dnm | block %d unsupported by v5: %s",
                        int(wavelength_nm),
                        global_index,
                        exc,
                    )

            block_records.append(record)
            del productive, ds_block
            gc.collect()

    if output_altitude is None:
        # Geometry is deterministic from the native altitude even if no block
        # reaches a valid reference. Use a harmless finite profile to establish it.
        from milgrau.level2.adaptive_grid import build_progressive_grid

        grid = build_progressive_grid(altitude_m, V5_PROGRESSIVE_GRID_SCHEDULE)
        output_altitude = np.asarray(grid.altitude_m, dtype=np.float64)
        effective_resolution = np.asarray(grid.effective_resolution_m, dtype=np.float64)
        source_count = np.asarray(grid.source_count, dtype=np.int32)

    assert effective_resolution is not None
    assert source_count is not None
    n_block = len(block_records)
    n_altitude = output_altitude.size
    n_fraction = len(fractions)

    def profile_stack(key: str) -> np.ndarray:
        values = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
        for index, record in enumerate(block_records):
            item = record[key]
            if item is not None:
                values[index] = np.asarray(item, dtype=np.float64)
        return values

    def mc_stack(key: str) -> np.ndarray:
        values = np.full(
            (n_block, n_fraction, n_altitude), np.nan, dtype=np.float64
        )
        for index, record in enumerate(block_records):
            item = record[key]
            if item is not None:
                values[index] = np.asarray(item, dtype=np.float64)
        return values

    nominal_beta = profile_stack("nominal_beta")
    nominal_alpha = profile_stack("nominal_alpha")
    support_count, support_fraction = _period_support(nominal_beta)
    period_mean_beta = _finite_mean(nominal_beta, axis=0)
    period_mean_alpha = _finite_mean(nominal_alpha, axis=0)

    ds_out = xr.Dataset(
        data_vars={
            "aerosol_backscatter_nominal": (("block", "altitude"), nominal_beta),
            "aerosol_extinction_nominal": (("block", "altitude"), nominal_alpha),
            "aerosol_backscatter_mc_mean": (
                ("block", "residual_fraction", "altitude"),
                mc_stack("mc_beta_mean"),
            ),
            "aerosol_backscatter_mc_std": (
                ("block", "residual_fraction", "altitude"),
                mc_stack("mc_beta_std"),
            ),
            "aerosol_backscatter_mc_q025": (
                ("block", "residual_fraction", "altitude"),
                mc_stack("mc_beta_q025"),
            ),
            "aerosol_backscatter_mc_q975": (
                ("block", "residual_fraction", "altitude"),
                mc_stack("mc_beta_q975"),
            ),
            "aerosol_extinction_mc_mean": (
                ("block", "residual_fraction", "altitude"),
                mc_stack("mc_alpha_mean"),
            ),
            "aerosol_extinction_mc_std": (
                ("block", "residual_fraction", "altitude"),
                mc_stack("mc_alpha_std"),
            ),
            "aerosol_extinction_mc_q025": (
                ("block", "residual_fraction", "altitude"),
                mc_stack("mc_alpha_q025"),
            ),
            "aerosol_extinction_mc_q975": (
                ("block", "residual_fraction", "altitude"),
                mc_stack("mc_alpha_q975"),
            ),
            "mc_valid_fraction": (
                ("block", "residual_fraction", "altitude"),
                mc_stack("mc_valid_fraction"),
            ),
            "period_mean_aerosol_backscatter_nominal": (
                ("altitude",),
                period_mean_beta,
            ),
            "period_mean_aerosol_extinction_nominal": (
                ("altitude",),
                period_mean_alpha,
            ),
            "period_support_count": (("altitude",), support_count),
            "period_support_fraction": (("altitude",), support_fraction),
            "reference_altitude_m": (
                ("block",),
                np.asarray(
                    [record["reference_altitude_m"] for record in block_records],
                    dtype=np.float64,
                ),
            ),
            "reference_tier_min_altitude_m": (
                ("block",),
                np.asarray(
                    [
                        record["reference_tier_min_altitude_m"]
                        for record in block_records
                    ],
                    dtype=np.float64,
                ),
            ),
            "reference_tier_index": (
                ("block",),
                np.asarray(
                    [record["reference_tier_index"] for record in block_records],
                    dtype=np.int16,
                ),
            ),
            "reference_fallback_used": (
                ("block",),
                np.asarray(
                    [record["reference_fallback_used"] for record in block_records],
                    dtype=np.int8,
                ),
            ),
            "contiguous_path_top_altitude_m": (
                ("block",),
                np.asarray(
                    [record["contiguous_path_top_altitude_m"] for record in block_records],
                    dtype=np.float64,
                ),
            ),
            "selection_success_fraction": (
                ("block",),
                np.asarray(
                    [record["selection_success_fraction"] for record in block_records],
                    dtype=np.float64,
                ),
            ),
            "retrieval_input_valid": (
                ("block",),
                np.asarray(
                    [record["retrieval_input_valid"] for record in block_records],
                    dtype=np.int8,
                ),
            ),
            "v5_success": (
                ("block",),
                np.asarray(
                    [record["v5_success"] for record in block_records],
                    dtype=np.int8,
                ),
            ),
            "signal_source_flag": (
                ("block",),
                np.asarray(
                    [record["signal_source_flag"] for record in block_records],
                    dtype=np.int16,
                ),
            ),
            "selected_reference_altitude_m_mc": (
                ("block", "mc_iteration"),
                np.stack(
                    [record["mc_reference_altitude"] for record in block_records],
                    axis=0,
                ),
            ),
            "selected_reference_tier_min_altitude_m_mc": (
                ("block", "mc_iteration"),
                np.stack(
                    [record["mc_reference_tier_min"] for record in block_records],
                    axis=0,
                ),
            ),
            "effective_vertical_resolution_m": (
                ("altitude",),
                effective_resolution,
            ),
            "source_bin_count": (("altitude",), source_count),
        },
        coords={
            "block": np.asarray(
                [record["global_index"] for record in block_records], dtype=np.int32
            ),
            "block_start_utc": (
                ("block",),
                np.asarray([record["block_start"] for record in block_records]),
            ),
            "block_end_utc": (
                ("block",),
                np.asarray([record["block_end"] for record in block_records]),
            ),
            "altitude": output_altitude,
            "residual_fraction": np.asarray(fractions, dtype=np.float64),
            "mc_iteration": np.arange(iterations, dtype=np.int32),
        },
        attrs={
            "title": "MILGRAU method-v5 R&D block-resolved elastic retrieval",
            "source_level1": str(source_path),
            "wavelength_nm": int(wavelength_nm),
            "retrieval_method_version": 5,
            "product_status": "research_and_development_not_productive_level2",
            "configured_block_minutes": minutes,
            "selected_block_start": first,
            "selected_block_stop": final,
            "reference_tier_min_altitudes_m": ",".join(f"{value:g}" for value in tiers),
            "reference_selection_policy": (
                "highest_supported_tier_then_minimum_rayleigh_diagnostic_cost"
            ),
            "support_fraction_denominator": "all requested temporal blocks",
            "period_mean_semantics": (
                "finite-only mean at each altitude; inspect period_support_count/fraction"
            ),
            "boundary_nominal_residual_fraction": 0.0,
            "beta_ref_relative_std": float(beta_ref_relative_std),
            "monte_carlo_iterations": iterations,
        },
    )
    ds_out["altitude"].attrs.update({"units": "m", "positive": "up"})
    ds_out["aerosol_backscatter_nominal"].attrs["units"] = "m-1 sr-1"
    ds_out["aerosol_extinction_nominal"].attrs["units"] = "m-1"
    ds_out["period_support_count"].attrs["long_name"] = (
        "number of temporal blocks contributing a finite nominal retrieval"
    )
    ds_out["period_support_fraction"].attrs["long_name"] = (
        "fraction of requested temporal blocks contributing a finite nominal retrieval"
    )

    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name, variable in ds_out.data_vars.items()
        if variable.ndim >= 1 and np.issubdtype(variable.dtype, np.number)
    }
    ds_out.to_netcdf(output_path, encoding=encoding)
    logger.info("wrote %s", output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write block-resolved method-v5 R&D elastic profiles from one Level-1 file."
    )
    parser.add_argument("source", help="Level-1 NetCDF path")
    parser.add_argument("--wavelength", type=int, default=532)
    parser.add_argument("--output-dir", default="v5_profile_results")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--station-config", default=None)
    parser.add_argument("--n-iterations", type=int, default=60)
    parser.add_argument(
        "--residual-fractions",
        nargs="+",
        type=float,
        default=[0.0, 0.02, 0.05],
    )
    parser.add_argument(
        "--reference-tier-min-altitudes-m",
        nargs="+",
        type=float,
        default=[10_000.0, 9_000.0, 8_000.0, 6_000.0],
    )
    parser.add_argument("--beta-ref-relative-std", type=float, default=0.0)
    parser.add_argument("--max-altitude-m", type=float, default=30_000.0)
    parser.add_argument("--block-start", type=int, default=0)
    parser.add_argument("--block-stop", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    path = run_low_memory_v5_profiles(
        args.source,
        wavelength_nm=args.wavelength,
        output_dir=args.output_dir,
        config_path=args.config,
        station_config_path=args.station_config,
        n_iterations=args.n_iterations,
        residual_fractions=args.residual_fractions,
        reference_tier_min_altitudes_m=args.reference_tier_min_altitudes_m,
        beta_ref_relative_std=args.beta_ref_relative_std,
        max_altitude_m=args.max_altitude_m,
        block_start=args.block_start,
        block_stop=args.block_stop,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
