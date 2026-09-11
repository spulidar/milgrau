"""Measurement quality screening for Level 0 processing."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from milgrau.io.logging_utils import bind_log_context
from milgrau.io.paths import measurement_save_id
from milgrau.level0.common import safe_mode


def _screen_acquisition_rows(
    df: pd.DataFrame,
    *,
    tolerance_fraction: float,
    header_time_jitter_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None, float | None]:
    """Apply one acquisition QA rule to one homogeneous measurement class."""
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
    bad_shots = shot_deviation >= shot_limit if shot_limit > 0.0 else shot_deviation > 0.0
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
        bad_duration = bad_duration | (abs(durations - nominal_duration_s) > header_time_jitter_s)
    else:
        bad_duration = pd.Series(True, index=rows.index)

    bad_condition = shots.isna() | (shots <= 0) | bad_shots | bad_rates | bad_duration
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
    *,
    tolerance_fraction: float,
    header_time_jitter_s: float,
) -> pd.DataFrame:
    """Apply explicitly configured acquisition QA to measurements and dark currents."""
    good_groups = []

    for meas_id, group in df_raw.groupby("meas_id"):
        save_id = measurement_save_id(meas_id)
        qa_logger = bind_log_context(logger, save_id=save_id, stage="qa")
        try:
            df_meas = group[group["meas_type"] == "measurements"].copy()
            df_dc = group[group["meas_type"] == "dark_current"].copy()
            if df_meas.empty:
                qa_logger.warning("no measurement files after inventory")
                continue

            good_meas, bad_meas, expected_meas_shots, expected_meas_duration = _screen_acquisition_rows(
                df_meas,
                tolerance_fraction=tolerance_fraction,
                header_time_jitter_s=header_time_jitter_s,
            )
            good_dc, bad_dc, expected_dc_shots, expected_dc_duration = _screen_acquisition_rows(
                df_dc,
                tolerance_fraction=tolerance_fraction,
                header_time_jitter_s=header_time_jitter_s,
            )

            total_files = len(group)
            bad_files = len(bad_meas) + len(bad_dc)
            accepted_files = total_files - bad_files
            loss_percent = (bad_files / total_files) * 100.0 if total_files > 0 else 0.0
            message = f"{accepted_files}/{total_files} accepted | {bad_files} rejected"
            if loss_percent > 10.0:
                qa_logger.warning(message)
            else:
                qa_logger.info(message)

            qa_logger.debug(
                "measurement_rejects=%d dark_current_rejects=%d loss=%.1f%% | "
                "measurement_nominal_shots=%s measurement_nominal_duration_s=%s | "
                "dark_nominal_shots=%s dark_nominal_duration_s=%s",
                len(bad_meas),
                len(bad_dc),
                loss_percent,
                expected_meas_shots,
                expected_meas_duration,
                expected_dc_shots,
                expected_dc_duration,
            )

            good_group = pd.concat([good_meas, good_dc], ignore_index=True)
            if not good_group.empty:
                good_groups.append(good_group)
        except Exception as exc:
            qa_logger.warning("quality evaluation failed: %s", exc)
            qa_logger.debug("quality failure details", exc_info=True)

    if not good_groups:
        return pd.DataFrame()
    return pd.concat(good_groups, ignore_index=True)
