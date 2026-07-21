"""LIBIDS Level 0 orchestration."""

from __future__ import annotations

import logging
import traceback

from milgrau.io.paths import level0_output_path, measurement_save_id, raw_data_root
from milgrau.level0.common import incremental_enabled
from milgrau.level0.inventory import build_measurement_inventory
from milgrau.level0.processing import process_measurement_group
from milgrau.level0.quality import filter_laser_shots


def process_level_0(config: dict, logger: logging.Logger) -> None:
    """Run LIBIDS Level 0 processing from raw Licel files to NetCDF."""
    raw_dir = raw_data_root(config)
    df_raw = build_measurement_inventory(str(raw_dir), config, logger)

    if df_raw.empty:
        logger.info("=== No new data to process. LIBIDS finished successfully! ===")
        return

    tolerance_fraction = float(config.get("processing", {}).get("laser_shot_tolerance_fraction", 2e-3))
    df_good = filter_laser_shots(df_raw, logger, tolerance_fraction=tolerance_fraction)
    if df_good.empty:
        logger.warning("=== No data survived quality control. Exiting. ===")
        return

    incremental = incremental_enabled(config)
    success_count = 0
    skipped_count = 0
    total_groups = len(df_good["meas_id"].unique())

    for meas_id, group_df in df_good.groupby("meas_id"):
        save_id = measurement_save_id(meas_id)
        netcdf_path = level0_output_path(meas_id, config)
        out_dir = netcdf_path.parent

        if incremental and netcdf_path.exists():
            logger.info(f"[SKIPPED] Level 0 already exists for {save_id}: {netcdf_path}")
            skipped_count += 1
            continue

        logger.info(f"Processing group [{save_id}]...")

        try:
            success, message = process_measurement_group(meas_id, group_df, config, logger)
            logger.info(message) if success else logger.warning(message)
            success_count += int(success)
        except Exception:
            logger.error(f"  -> [ERROR] Fatal error converting {save_id}:\n{traceback.format_exc()}")

    logger.info(f"=== LIBIDS finished: processed {success_count}, skipped {skipped_count}, total groups {total_groups}. ===")
