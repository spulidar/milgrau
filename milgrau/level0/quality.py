"""Measurement quality screening for Level 0 processing."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from milgrau.level0.common import (
    DEFAULT_LASER_SHOT_TOLERANCE_FRACTION,
    LICEL_HEADER_TIME_JITTER_S,
    safe_mode,
)


def _screen_acquisition_rows(
    df: pd.DataFrame,
    *,
    tolerance_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None, float | None]:
    """Apply one acquisition QA rule to one homogeneous measurement class.

    Measurements and dark currents are screened independently. Laser shots and
    repetition rate define the nominal acquisition duration. Whole-second Licel
    header durations that differ by at most one second are accepted as timestamp
    quantization and carry ``qa_nominal_duration_s`` for SCC-only time-axis
    normalization. Larger timing discrepancies are rejected here, before any
    NetCDF writer is called.
    """
    if df.empty:
        return df.copy(), df.copy(), None, None

    rows = df.copy()
    shots = pd.to_numeric(rows["nshots"], errors="coerce")
    rates = pd.to_numeric(rows["laser_freq"], errors="coerce")
    durations = pd.to_numeric(rows["duration"], errors="coerce")

    positive_shots = shots[shots > 0]
    positive_rates = rates[rates > 0]
    if positive_shots.empty or positive_rates.empty:
        return rows.iloc[0:0].copy(), rows, None, None

    expected_shots = float(safe_mode(positive_shots.values))
    expected_rate = float(safe_mode(positive_rates.values))

    shot_limit = tolerance_fraction * expected_shots
    shot_deviation = abs(shots - expected_shots)
    if shot_limit > 0.0:
        bad_shots = shot_deviation >= shot_limit
    else:
        bad_shots = shot_deviation > 0.0

    bad_rates = rates.isna() | (rates <= 0) | (abs(rates - expected_rate) > 1e-9)

    physical_duration_s = expected_shots / expected_rate
    nominal_duration_s = float(round(physical_duration_s))
    physical_tolerance_s = max(abs(physical_duration_s) * tolerance_fraction, 1e-9)
    nominal_duration_supported = (
        nominal_duration_s > 0.0
        and abs(physical_duration_s - nominal_duration_s) <= physical_tolerance_s
    )

    bad_duration = durations.isna() | (durations <= 0)
    if nominal_duration_supported:
        bad_duration = bad_duration | (abs(durations - nominal_duration_s) > LICEL_HEADER_TIME_JITTER_S)
    else:
        # If shots/rate do not support a stable integer-second acquisition,
        # there is no defensible SCC time scale for these rows.
        bad_duration = pd.Series(True, index=rows.index)

    bad_condition = (
        shots.isna()
        | (shots <= 0)
        | bad_shots
        | bad_rates
        | bad_duration
    )

    good = rows.loc[~bad_condition].copy()
    bad = rows.loc[bad_condition].copy()
    if not good.empty:
        good["qa_nominal_shots"] = expected_shots
        good["qa_nominal_laser_freq_hz"] = expected_rate
        good["qa_nominal_duration_s"] = nominal_duration_s
        good["qa_header_duration_adjustment_s"] = nominal_duration_s - pd.to_numeric(
            good["duration"], errors="coerce"
        )

    return good, bad, expected_shots, nominal_duration_s if nominal_duration_supported else None


def filter_laser_shots(
    df_raw: pd.DataFrame,
    logger: logging.Logger,
    tolerance_fraction: float = DEFAULT_LASER_SHOT_TOLERANCE_FRACTION,
) -> pd.DataFrame:
    """Apply standardized acquisition QA to measurements and dark currents."""
    logger.info("Evaluating acquisition quality and consistency per measurement...")
    good_groups = []

    for meas_id, group in df_raw.groupby("meas_id"):
        try:
            df_meas = group[group["meas_type"] == "measurements"].copy()
            df_dc = group[group["meas_type"] == "dark_current"].copy()
            if df_meas.empty:
                logger.warning(f"  -> [{meas_id}] No measurement files found after inventory stage.")
                continue

            good_meas, bad_meas, expected_meas_shots, expected_meas_duration = _screen_acquisition_rows(
                df_meas,
                tolerance_fraction=tolerance_fraction,
            )
            good_dc, bad_dc, expected_dc_shots, expected_dc_duration = _screen_acquisition_rows(
                df_dc,
                tolerance_fraction=tolerance_fraction,
            )

            total_files = len(group)
            bad_files = len(bad_meas) + len(bad_dc)
            loss_percent = (bad_files / total_files) * 100.0 if total_files > 0 else 0.0

            nominal_details = []
            if expected_meas_shots is not None:
                detail = f"measurement nominal={expected_meas_shots:g} shots"
                if expected_meas_duration is not None:
                    detail += f"/{expected_meas_duration:g} s"
                nominal_details.append(detail)
            if expected_dc_shots is not None:
                detail = f"dark-current nominal={expected_dc_shots:g} shots"
                if expected_dc_duration is not None:
                    detail += f"/{expected_dc_duration:g} s"
                nominal_details.append(detail)
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
