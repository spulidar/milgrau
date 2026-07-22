"""LIBIDS Level 0 orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

from milgrau.io.contracts import netcdf_satisfies_contract, validate_level0_contract
from milgrau.io.paths import level0_output_path, measurement_save_id, raw_data_root
from milgrau.level0.common import incremental_enabled
from milgrau.level0.inventory import build_measurement_inventory
from milgrau.level0.processing import measurement_group_provenance, process_measurement_group
from milgrau.level0.quality import filter_laser_shots
from milgrau.operations import ExecutionResult, ExecutionStatus, ExecutionSummary
from milgrau.provenance import output_is_current


def _level0_is_current(meas_id: str, group_df, config: dict, output_path) -> bool:
    """Return whether one Level 0 output matches its current inputs and contract."""
    if not Path(output_path).exists():
        return False
    try:
        expected = measurement_group_provenance(meas_id, group_df, config)
    except (KeyError, OSError, ValueError):
        return False
    return output_is_current(
        output_path,
        expected,
        integrity_check=lambda path: netcdf_satisfies_contract(path, validate_level0_contract),
    )


def process_level_0(config: dict, logger: logging.Logger) -> ExecutionSummary:
    """Run LIBIDS Level 0 processing from raw Licel files to NetCDF."""
    raw_dir = raw_data_root(config)
    df_raw = build_measurement_inventory(str(raw_dir), config, logger)

    if df_raw.empty:
        logger.info("=== No new data to process. LIBIDS finished successfully! ===")
        return ExecutionSummary.from_results(
            [ExecutionResult.skipped("level0.discovery", "No new data to process", input_path=raw_dir)]
        )

    tolerance_fraction = float(config.get("processing", {}).get("laser_shot_tolerance_fraction", 2e-3))
    df_good = filter_laser_shots(df_raw, logger, tolerance_fraction=tolerance_fraction)
    if df_good.empty:
        logger.warning("=== No data survived quality control. Exiting. ===")
        return ExecutionSummary.from_results(
            [ExecutionResult.skipped("level0.quality", "No data survived quality control", input_path=raw_dir)]
        )

    incremental = incremental_enabled(config)
    total_groups = len(df_good["meas_id"].unique())
    results: list[ExecutionResult] = []

    for meas_id, group_df in df_good.groupby("meas_id"):
        save_id = measurement_save_id(meas_id)
        netcdf_path = level0_output_path(meas_id, config)
        if incremental and _level0_is_current(meas_id, group_df, config, netcdf_path):
            result = ExecutionResult.skipped(
                "level0.incremental",
                f"Level 0 provenance is current for {save_id}",
                output_path=netcdf_path,
                metadata={"pipeline": "LIBIDS", "save_id": save_id},
            )
            result.log(logger)
            results.append(result)
            continue

        logger.info(f"Processing group [{save_id}]...")

        try:
            result = process_measurement_group(meas_id, group_df, config, logger)
            if not isinstance(result, ExecutionResult):
                raise TypeError(f"process_measurement_group returned {type(result).__name__}; expected ExecutionResult.")
        except Exception as exc:
            result = ExecutionResult.failure(
                "level0.group",
                f"Unexpected error converting {save_id}",
                output_path=netcdf_path,
                cause=exc,
                include_traceback=True,
                metadata={"pipeline": "LIBIDS", "save_id": save_id},
            )
        result.log(logger)
        results.append(result)

    summary = ExecutionSummary.from_results(results)
    counts = summary.counts
    logger.info(
        f"=== LIBIDS finished: processed {counts[ExecutionStatus.SUCCESS]}, "
        f"skipped {counts[ExecutionStatus.SKIPPED]}, total groups {total_groups}. ==="
    )
    return summary
