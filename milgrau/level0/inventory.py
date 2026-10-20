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


def _station_context_identity(config: dict, timestamp: pd.Timestamp) -> tuple[str, str]:
    """Resolve only identity for an inventory header, without materializing SCC context.

    This uses the same station-local, inclusive profile dates as the full station
    resolver. Full calibration/overlap/SCC context is still resolved by Level 0
    processing, once the actual channels and solar regime are known.
    """
    catalog = config["_station_catalog"]
    local_date = pd.Timestamp(timestamp).tz_convert(station_timezone(config)).date()
    matched = []
    for profile in catalog["profiles"]:
        valid_from = pd.Timestamp(profile["valid_from"]).date()
        valid_to = (
            None if profile.get("valid_to") is None
            else pd.Timestamp(profile["valid_to"]).date()
        )
        if local_date >= valid_from and (valid_to is None or local_date <= valid_to):
            matched.append(profile)
    if len(matched) != 1:
        raise ValueError(
            "Expected exactly one station profile for "
            f"{local_date.isoformat()}, found {[profile['id'] for profile in matched]}."
        )
    return str(matched[0]["id"]), str(matched[0]["calibration_id"])


def _annotate_station_context(measurements: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Attach one homogeneous station profile/calibration identity to each raw file."""
    if measurements.empty:
        return measurements.copy()

    rows = measurements.copy()
    profile_ids: list[str] = []
    calibration_ids: list[str] = []
    for row in rows.itertuples():
        start = pd.Timestamp(row.start_time_utc)
        stop = pd.Timestamp(row.stop_time)
        start_identity = _station_context_identity(config, start)
        end_probe = max(start, stop - pd.Timedelta(microseconds=1))
        end_identity = _station_context_identity(config, end_probe)
        if start_identity != end_identity:
            raise ValueError(
                "One raw Licel file crosses a station profile/calibration boundary and "
                f"cannot belong to a homogeneous session: {row.filepath} "
                f"({start_identity} -> {end_identity})."
            )
        profile_ids.append(start_identity[0])
        calibration_ids.append(start_identity[1])

    rows["station_profile_id"] = profile_ids
    rows["instrument_calibration_id"] = calibration_ids
    return rows


def _sessionize_measurements(
    measurements: pd.DataFrame,
    *,
    station: str,
    max_gap_seconds: float,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Assign continuous measurement rows to canonical complete-session IDs."""
    if measurements.empty:
        return measurements.copy()

    rows = measurements.sort_values(["start_time_utc", "stop_time", "filepath"]).copy()
    if rows["start_time_utc"].isna().any() or rows["stop_time"].isna().any():
        raise ValueError("Cannot sessionize measurement rows with missing start/stop times.")

    max_gap = pd.Timedelta(seconds=float(max_gap_seconds))
    sequence: list[int] = []
    boundary_reasons: list[str] = []
    current_sequence = -1
    current_end: pd.Timestamp | None = None
    current_context: tuple[str, str] | None = None

    for row in rows.itertuples():
        start = pd.Timestamp(row.start_time_utc)
        stop = pd.Timestamp(row.stop_time)
        context = (
            str(row.station_profile_id),
            str(row.instrument_calibration_id),
        )
        reasons: list[str] = []
        if current_end is None:
            reasons.append("acquisition_start")
        else:
            if start > current_end + max_gap:
                reasons.append("time_gap")
            if context != current_context:
                reasons.append("station_context_change")

        if reasons:
            current_sequence += 1
            current_end = stop
            current_context = context
            boundary_reasons.append("+".join(reasons))
        else:
            current_end = max(current_end, stop)
            boundary_reasons.append("continuous")
        sequence.append(current_sequence)

    rows["_session_sequence"] = sequence
    rows["session_boundary_reason"] = boundary_reasons
    rows["session_id"] = ""
    rows["session_start_utc"] = pd.Series(
        pd.NaT,
        index=rows.index,
        dtype="datetime64[ns, UTC]",
    )
    rows["session_end_utc"] = pd.Series(
        pd.NaT,
        index=rows.index,
        dtype="datetime64[ns, UTC]",
    )

    for _sequence, group in rows.groupby("_session_sequence", sort=True):
        session_start = pd.to_datetime(group["start_time_utc"], utc=True).min()
        session_end = pd.to_datetime(group["stop_time"], utc=True).max()
        try:
            session_id = build_session_id(
                station,
                session_start.to_pydatetime(),
                session_end.to_pydatetime(),
            )
        except ValueError as exc:
            # Keep exact timestamps and every implicated Licel path in the log.
            # In particular, an otherwise valid short acquisition can collapse
            # to one minute in the human-readable canonical session ID.
            candidates = group.sort_values(["start_time_utc", "filepath"])
            first_file = str(candidates.iloc[0]["filepath"])
            last_file = str(candidates.iloc[-1]["filepath"])
            message = (
                f"Cannot identify session sequence {_sequence}: "
                f"{len(candidates)} measurement file(s), "
                f"start_utc={session_start.isoformat()}, "
                f"end_utc={session_end.isoformat()}, "
                f"first_file={first_file}, last_file={last_file}: {exc}"
            )
            if logger is not None:
                logger.error("%s", message)
                for row in candidates.itertuples(index=False):
                    logger.error(
                        "Session ID failure source: path=%s | start_utc=%s | "
                        "stop_utc=%s | header_duration_s=%s | nshots=%s | "
                        "laser_freq_hz=%s | profile=%s | calibration=%s | boundary=%s",
                        row.filepath,
                        pd.Timestamp(row.start_time_utc).isoformat(),
                        pd.Timestamp(row.stop_time).isoformat(),
                        row.duration,
                        row.nshots,
                        row.laser_freq,
                        row.station_profile_id,
                        row.instrument_calibration_id,
                        row.session_boundary_reason,
                    )
            raise ValueError(message) from exc
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
    rows["session_start_utc"] = pd.Series(
        pd.NaT,
        index=rows.index,
        dtype="datetime64[ns, UTC]",
    )
    rows["session_end_utc"] = pd.Series(
        pd.NaT,
        index=rows.index,
        dtype="datetime64[ns, UTC]",
    )
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

    measurement_rows = _annotate_station_context(measurement_rows, config)
    measurement_rows = _sessionize_measurements(
        measurement_rows,
        station=station_id(config),
        max_gap_seconds=level0_config.session.max_gap_seconds,
        logger=logger,
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
