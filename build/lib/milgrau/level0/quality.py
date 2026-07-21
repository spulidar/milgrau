"""Measurement quality screening for Level 0 processing."""

from __future__ import annotations

import logging

import pandas as pd

from milgrau.level0.common import safe_mode


def filter_laser_shots(
    df_raw: pd.DataFrame,
    logger: logging.Logger,
    tolerance_fraction: float = 2e-3,
) -> pd.DataFrame:
    """Evaluate laser-shot consistency for each measurement period."""
    logger.info("Evaluating laser shots quality and consistency per measurement...")
    good_groups = []

    for meas_id, group in df_raw.groupby("meas_id"):
        try:
            df_meas = group[group["meas_type"] == "measurements"].copy()
            df_dc = group[group["meas_type"] == "dark_current"].copy()
            if df_meas.empty:
                logger.warning(f"  -> [{meas_id}] No measurement files found after inventory stage.")
                continue

            expected_shots = safe_mode(df_meas["nshots"].dropna().values)
            shot_deviation = abs(df_meas["nshots"] - expected_shots)
            bad_meas_condition = (
                df_meas["nshots"].isna()
                | (df_meas["nshots"] <= 0)
                | (shot_deviation >= tolerance_fraction * expected_shots)
            )
            good_meas = df_meas.loc[~bad_meas_condition]
            bad_meas = df_meas.loc[bad_meas_condition]

            if not df_dc.empty:
                good_dc = df_dc.loc[df_dc["nshots"].fillna(0) > 0]
                bad_dc = df_dc.loc[df_dc["nshots"].fillna(0) <= 0]
            else:
                good_dc = df_dc
                bad_dc = df_dc

            total_files = len(group)
            bad_files = len(bad_meas) + len(bad_dc)
            loss_percent = (bad_files / total_files) * 100.0 if total_files > 0 else 0.0
            if bad_files > 0:
                log_msg = (
                    f"  -> [{meas_id}] QA Report: {bad_files}/{total_files} files rejected "
                    f"({loss_percent:.1f}% loss). Measurement rejects: {len(bad_meas)}, "
                    f"dark-current rejects: {len(bad_dc)}."
                )
                if loss_percent > 10.0:
                    logger.warning(log_msg)
                else:
                    logger.info(log_msg)
            else:
                logger.info(f"  -> [{meas_id}] QA Report: 100% data retention. No files rejected.")

            good_group = pd.concat([good_meas, good_dc], ignore_index=True)
            if not good_group.empty:
                good_groups.append(good_group)
        except Exception as exc:
            logger.warning(f"  -> [{meas_id}] Error evaluating quality: {exc}")

    if not good_groups:
        return pd.DataFrame()
    return pd.concat(good_groups, ignore_index=True)
