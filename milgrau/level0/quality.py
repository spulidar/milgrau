"""Measurement quality screening for Level 0 processing."""

from __future__ import annotations

import logging

import pandas as pd

from milgrau.level0.common import DEFAULT_LASER_SHOT_TOLERANCE_FRACTION, safe_mode


def _filter_shot_consistency(
    df: pd.DataFrame,
    *,
    tolerance_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    """Split rows by laser-shot consistency around the group's nominal value."""
    if df.empty:
        return df.copy(), df.copy(), None

    numeric_shots = pd.to_numeric(df["nshots"], errors="coerce")
    positive = numeric_shots[numeric_shots > 0]
    if positive.empty:
        return df.iloc[0:0].copy(), df.copy(), None

    expected_shots = safe_mode(positive.values)
    shot_deviation = abs(numeric_shots - expected_shots)
    shot_limit = tolerance_fraction * expected_shots
    if shot_limit > 0:
        inconsistent = shot_deviation >= shot_limit
    else:
        inconsistent = shot_deviation > 0

    bad_condition = numeric_shots.isna() | (numeric_shots <= 0) | inconsistent
    return df.loc[~bad_condition].copy(), df.loc[bad_condition].copy(), float(expected_shots)


def filter_laser_shots(
    df_raw: pd.DataFrame,
    logger: logging.Logger,
    tolerance_fraction: float = DEFAULT_LASER_SHOT_TOLERANCE_FRACTION,
) -> pd.DataFrame:
    """Evaluate laser-shot consistency for measurements and dark currents."""
    logger.info("Evaluating laser shots quality and consistency per measurement...")
    good_groups = []

    for meas_id, group in df_raw.groupby("meas_id"):
        try:
            df_meas = group[group["meas_type"] == "measurements"].copy()
            df_dc = group[group["meas_type"] == "dark_current"].copy()
            if df_meas.empty:
                logger.warning(f"  -> [{meas_id}] No measurement files found after inventory stage.")
                continue

            good_meas, bad_meas, expected_meas_shots = _filter_shot_consistency(
                df_meas,
                tolerance_fraction=tolerance_fraction,
            )
            good_dc, bad_dc, expected_dc_shots = _filter_shot_consistency(
                df_dc,
                tolerance_fraction=tolerance_fraction,
            )

            total_files = len(group)
            bad_files = len(bad_meas) + len(bad_dc)
            loss_percent = (bad_files / total_files) * 100.0 if total_files > 0 else 0.0
            nominal_details = []
            if expected_meas_shots is not None:
                nominal_details.append(f"measurement nominal={expected_meas_shots:g}")
            if expected_dc_shots is not None:
                nominal_details.append(f"dark-current nominal={expected_dc_shots:g}")
            nominal_suffix = f" ({'; '.join(nominal_details)})" if nominal_details else ""

            if bad_files > 0:
                log_msg = (
                    f"  -> [{meas_id}] QA Report: {bad_files}/{total_files} files rejected "
                    f"({loss_percent:.1f}% loss). Measurement rejects: {len(bad_meas)}, "
                    f"dark-current rejects: {len(bad_dc)}{nominal_suffix}."
                )
                if loss_percent > 10.0:
                    logger.warning(log_msg)
                else:
                    logger.info(log_msg)
            else:
                logger.info(
                    f"  -> [{meas_id}] QA Report: 100% data retention. No files rejected{nominal_suffix}."
                )

            good_group = pd.concat([good_meas, good_dc], ignore_index=True)
            if not good_group.empty:
                good_groups.append(good_group)
        except Exception as exc:
            logger.warning(f"  -> [{meas_id}] Error evaluating quality: {exc}")

    if not good_groups:
        return pd.DataFrame()
    return pd.concat(good_groups, ignore_index=True)
