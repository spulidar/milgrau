"""Continuous-session inventory construction for MILGRAU Level 0 processing."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from milgrau.io.filesystem import scan_raw_files
from milgrau.io.licel import read_licel_header
from milgrau.io.paths import build_session_id, station_id
from milgrau.level0.config import resolve_level0_config, station_timezone


def _normalize_header_times(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize Licel header times to timezone-aware UTC timestamps."""
    rows = df_raw.copy()
    rows["start_time_utc"] = pd.to_datetime(rows["start_time_utc"], utc=True, errors="coerce")
    rows["stop_time"] = pd.to_datetime(rows["stop_time"], utc=True, errors="coerce")
    duration = pd.to_numeric(rows["duration"], errors="coerce")
    missing_stop = rows["stop_time"].isna() & rows["start_time_utc"].notna() & duration.notna()
    rows.loc[missing_stop, "stop_time"] = (
        rows.loc[missing_stop, "start_time_utc"]
        + pd.to_timedelta(duration.loc[missing_stop], unit="s")
    )
    invalid_stop = rows["stop_time"].notna() & (rows["stop_time"] <= rows["start_time_utc"])
    repairable = invalid_stop & duration.notna() & (duration > 0)
    rows.loc[repairable, "stop_time"] = (
        rows.loc[repairable, "start_time_utc"]
        + pd.to_timedelta(duration.loc[repairable], unit="s")
    )
    return rows


def _sessionize_measurements(
    measurements: pd.DataFrame,
    *,
    station: str,
    max_gap_seconds: float,
) -> pd.DataFrame:
    """Assign continuous measurement rows to canonical complete-session IDs."""
    if measurements.empty:
        return measurements.copy()

    rows = measurements.sort_values(["start_time_utc", "stop_time", "filepath"]).copy()
    if rows["start_time_utc"].isna().any() or rows["stop_time"].isna().any():
        raise ValueError("Cannot sessionize measurement rows with missing start/stop times.")

    max_gap = pd.Timedelta(seconds=float(max_gap_seconds))
    sequence: list[int] = []
    current_sequence = -1
    current_end: pd.Timestamp | None = None

    for row in rows.itertuples():
        start = pd.Timestamp(row.start_time_utc)
        stop = pd.Timestamp(row.stop_time)
        if current_end is None or start > current_end + max_gap:
            current_sequence += 1
            current_end = stop
        else:
            current_end = max(current_end, stop)
        sequence.append(current_sequence)

    rows["_session_sequence"] = sequence
    rows["session_id"] = ""
    rows["session_start_utc"] = pd.NaT
    rows["session_end_utc"] = pd.NaT

    for _sequence, group in rows.groupby("_session_sequence", sort=True):
        session_start = pd.to_datetime(group["start_time_utc"], utc=True).min()
        session_end = pd.to_datetime(group["stop_time"], utc=True).max()
        session_id = build_session_id(
            station,
            session_start.to_pydatetime(),
            session_end.to_pydatetime(),
        )
        rows.loc[group.index, "session_id"] = session_id
        rows.loc[group.index, "session_start_utc"] = session_start
        rows.loc[group.index, "session_end_utc"] = session_end

    return rows.drop(columns=["_session_sequence"])


def _session_table(measurements: pd.DataFrame) -> pd.DataFrame:
    """Return one row per measurement session."""
    if measurements.empty:
        return pd.DataFrame(columns=["session_id", "session_start_utc", "session_end_utc"])
    return (
        measurements[
            ["session_id", "session_start_utc", "session_end_utc"]
        ]
        .drop_duplicates()
        .sort_values("session_start_utc")
        .reset_index(drop=True)
    )


def _distance_to_session_hours(
    timestamp: pd.Timestamp,
    session_start: pd.Timestamp,
    session_end: pd.Timestamp,
) -> float:
    """Return temporal distance from one timestamp to a closed session interval."""
    if timestamp < session_start:
        delta = session_start - timestamp
    elif timestamp > session_end:
        delta = timestamp - session_end
    else:
        delta = pd.Timedelta(0)
    return float(delta.total_seconds() / 3600.0)


def _associate_dark_currents(
    dark_rows: pd.DataFrame,
    measurements: pd.DataFrame,
    config: dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Associate dark-current rows to the nearest continuous measurement session."""
    if dark_rows.empty:
        return dark_rows.copy()

    rows = dark_rows.copy()
    rows["session_id"] = None
    rows["session_start_utc"] = pd.NaT
    rows["session_end_utc"] = pd.NaT
    rows["association_method"] = "unassociated"
    rows["dark_current_association_delta_hours"] = np.nan

    sessions = _session_table(measurements)
    if sessions.empty:
        logger.warning("No measurement sessions available; dark currents remain unassociated.")
        return rows

    max_hours = resolve_level0_config(config).dark_current.max_association_hours
    for index, row in rows.iterrows():
        timestamp = pd.Timestamp(row["start_time_utc"])
        candidates: list[tuple[float, pd.Series]] = []
        for _session_index, session in sessions.iterrows():
            distance_h = _distance_to_session_hours(
                timestamp,
                pd.Timestamp(session["session_start_utc"]),
                pd.Timestamp(session["session_end_utc"]),
            )
            candidates.append((distance_h, session))
        distance_h, selected = min(candidates, key=lambda item: item[0])
        if distance_h > max_hours:
            logger.warning(
                "Dark current remains unassociated; nearest session is %.2f h away "
                "(configured maximum %.2f h): %s",
                distance_h,
                max_hours,
                row["filepath"],
            )
            continue

        rows.at[index, "session_id"] = str(selected["session_id"])
        rows.at[index, "session_start_utc"] = selected["session_start_utc"]
        rows.at[index, "session_end_utc"] = selected["session_end_utc"]
        rows.at[index, "association_method"] = "nearest_session"
        rows.at[index, "dark_current_association_delta_hours"] = distance_h

    return rows


def build_session_inventory(
    raw_dir: str,
    config: dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Build the Level-0 inventory and group continuous Licel acquisitions into sessions."""
    logger.info("Building raw data inventory...")
    level0_config = resolve_level0_config(config)
    discovery = level0_config.discovery

    file_paths, file_types = scan_raw_files(
        raw_dir,
        spurious_extensions=discovery.spurious_extensions,
        quarantine_dir=discovery.quarantine_dir,
        raw_scan_ignore_dirs=discovery.raw_scan_ignore_dirs,
        logger=logger,
    )
    if not file_paths:
        return pd.DataFrame()

    records = []
    for filepath, file_type in zip(file_paths, file_types):
        start_time_utc, stop_time, duration, n_shots, laser_freq = read_licel_header(
            filepath,
            logger=logger,
        )
        if start_time_utc is None:
            continue
        records.append(
            {
                "filepath": filepath,
                "meas_type": file_type,
                "start_time_utc": start_time_utc,
                "stop_time": stop_time,
                "nshots": n_shots,
                "duration": duration,
                "laser_freq": laser_freq,
            }
        )

    df_raw = pd.DataFrame.from_records(records)
    if df_raw.empty:
        return df_raw

    df_raw = _normalize_header_times(df_raw)
    df_raw = df_raw[df_raw["start_time_utc"].notna()].copy()
    if df_raw.empty:
        return df_raw

    timezone_name = station_timezone(config)
    df_raw["start_time_local"] = df_raw["start_time_utc"].dt.tz_convert(timezone_name)
    df_raw["stop_time_local"] = df_raw["stop_time"].dt.tz_convert(timezone_name)

    measurement_rows = df_raw[df_raw["meas_type"] == "measurements"].copy()
    dark_rows = df_raw[df_raw["meas_type"] == "dark_current"].copy()

    measurement_rows = _sessionize_measurements(
        measurement_rows,
        station=station_id(config),
        max_gap_seconds=level0_config.session.max_gap_seconds,
    )
    if not measurement_rows.empty:
        measurement_rows["association_method"] = "measurement"
        measurement_rows["dark_current_association_delta_hours"] = np.nan

    dark_rows = _associate_dark_currents(
        dark_rows,
        measurement_rows,
        config,
        logger,
    )

    inventory = pd.concat([measurement_rows, dark_rows], ignore_index=True, sort=False)
    if inventory.empty:
        return inventory
    inventory["original_session_id"] = inventory["session_id"]
    return inventory.sort_values(["start_time_utc", "filepath"]).reset_index(drop=True)
