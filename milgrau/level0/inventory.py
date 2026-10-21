"""Continuous-session inventory construction for MILGRAU Level 0 processing."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from milgrau.io.filesystem import scan_raw_files
from milgrau.io.licel import read_licel_header
from milgrau.io.logging_utils import bind_log_context
from milgrau.io.paths import build_session_id, session_id_parts, station_id
from milgrau.io.selection import InputSelection
from milgrau.level0.config import resolve_level0_config, station_timezone
from milgrau.level0.header_cache import LicelHeaderCache


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


def _annotate_station_context(
    measurements: pd.DataFrame,
    config: dict,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Attach one homogeneous station profile/calibration identity to each raw file."""
    if measurements.empty:
        return measurements.copy()

    rows = measurements.copy()
    profile_ids: list[str] = []
    calibration_ids: list[str] = []
    started = perf_counter()
    last_progress = started
    progress_every = max(1000, len(rows) // 10)
    for position, row in enumerate(rows.itertuples(), start=1):
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
        if logger is not None and position < len(rows):
            now = perf_counter()
            if position % progress_every == 0 or now - last_progress >= 20.0:
                logger.info(
                    "resolved %d/%d profiles (%.0f%%) | %.1fs",
                    position, len(rows),
                    position * 100.0 / len(rows), now - started,
                )
                last_progress = now

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
    """Associate each dark current with its nearest full session in O(D log S).

    Sessions are sorted by start time. Prefix-maximum end times also cover
    overlapping sessions; ties preserve the first chronological session, as
    in the previous exhaustive nearest-interval search. No persistent state.
    """
    if dark_rows.empty:
        return dark_rows.copy()

    rows = dark_rows.copy()
    rows["session_id"] = None
    rows["session_start_utc"] = pd.Series(
        pd.NaT, index=rows.index, dtype="datetime64[ns, UTC]",
    )
    rows["session_end_utc"] = pd.Series(
        pd.NaT, index=rows.index, dtype="datetime64[ns, UTC]",
    )
    rows["association_method"] = "unassociated"
    rows["dark_current_association_delta_hours"] = np.nan

    sessions = _session_table(measurements)
    if sessions.empty:
        logger.warning(
            "dark currents | %d unassociated | no measurement sessions",
            len(rows),
        )
        return rows

    # Pandas 2.x preserves datetime64 resolutions independently per column.
    # A raw header may be datetime64[us, UTC] while session bounds are [ns, UTC].
    # Comparing their raw .asi8 integers would be off by 1,000x, yielding
    # spurious ~470,000 h distances and dropping every dark current.
    session_starts = pd.DatetimeIndex(sessions["session_start_utc"])
    session_ends = pd.DatetimeIndex(sessions["session_end_utc"])
    dark_times = pd.DatetimeIndex(rows["start_time_utc"])
    logger.debug(
        "time resolution | session start=%s | session end=%s | dark current=%s | normalized=ns",
        session_starts.dtype, session_ends.dtype, dark_times.dtype,
    )
    starts = session_starts.as_unit("ns").asi8
    ends = session_ends.as_unit("ns").asi8
    timestamps = dark_times.as_unit("ns").asi8
    n_sessions = len(sessions)

    # Earliest overlapping interval covering a timestamp can be found by
    # binary-searching the monotonic prefix maximum of session end times.
    prefix_max_ends = np.maximum.accumulate(ends)
    best_end_positions = np.zeros(n_sessions, dtype=np.intp)
    best_position = 0
    for pos in range(1, n_sessions):
        if ends[pos] > ends[best_position]:
            best_position = pos
        best_end_positions[pos] = best_position

    prior = np.searchsorted(starts, timestamps, side="right") - 1
    following = prior + 1
    valid_prior = prior >= 0
    valid_following = following < n_sessions
    nearest_prior = best_end_positions[np.clip(prior, 0, n_sessions - 1)]
    nearest_following = np.clip(following, 0, n_sessions - 1)

    # Distances are nanoseconds, cast to float before differences to avoid
    # integer overflow for unusual but valid historical timestamps.
    previous_distance = np.full(len(rows), np.inf)
    next_distance = np.full(len(rows), np.inf)
    previous_distance[valid_prior] = (
        timestamps[valid_prior].astype(float)
        - ends[nearest_prior[valid_prior]].astype(float)
    )
    next_distance[valid_following] = (
        starts[nearest_following[valid_following]].astype(float)
        - timestamps[valid_following].astype(float)
    )

    covered = valid_prior & (
        prefix_max_ends[np.clip(prior, 0, n_sessions - 1)] >= timestamps
    )
    first_covering = np.searchsorted(prefix_max_ends, timestamps, side="left")
    selected_positions = np.where(
        covered,
        np.clip(first_covering, 0, n_sessions - 1),
        np.where(
            previous_distance <= next_distance,
            nearest_prior,
            nearest_following,
        ),
    )
    distance_hours = np.where(
        covered, 0.0, np.minimum(previous_distance, next_distance) / 3.6e12,
    )
    # A missing UTC timestamp cannot be associated. Normal inventory creation
    # filters these out, but this guard keeps the helper safe independently.
    missing_timestamp = timestamps == np.iinfo(np.int64).min
    max_hours = resolve_level0_config(config).dark_current.max_association_hours
    associated = (distance_hours <= max_hours) & ~missing_timestamp

    matching_index = rows.index[associated]
    matched_sessions = sessions.iloc[selected_positions[associated]]
    if len(matching_index):
        rows.loc[matching_index, "session_id"] = matched_sessions["session_id"].to_numpy()
        rows.loc[matching_index, "session_start_utc"] = list(
            matched_sessions["session_start_utc"]
        )
        rows.loc[matching_index, "session_end_utc"] = list(
            matched_sessions["session_end_utc"]
        )
        rows.loc[matching_index, "association_method"] = "nearest_session"
        rows.loc[matching_index, "dark_current_association_delta_hours"] = distance_hours[associated]

    unassociated_count = int((~associated).sum())
    logger.info(
        "dark currents | %d/%d associated | %d unassociated | limit %.1f h",
        int(associated.sum()),
        len(rows),
        unassociated_count,
        max_hours,
    )
    if unassociated_count:
        logger.warning(
            "dark currents | %d could not be associated within %.1f h; "
            "see DEBUG for affected paths",
            unassociated_count,
            max_hours,
        )
    if logger.isEnabledFor(logging.DEBUG):
        for offset, path in enumerate(rows["filepath"]):
            logger.debug(
                "dark current | %s | %s | nearest=%s | distance=%.4f h",
                "associated" if associated[offset] else "unassociated",
                path,
                sessions.iloc[selected_positions[offset]]["session_id"]
                if not missing_timestamp[offset] else "-",
                distance_hours[offset] if not missing_timestamp[offset] else float("nan"),
            )
    return rows

def _select_temporal_candidates(
    df_raw: pd.DataFrame,
    selection: InputSelection | None,
    config: dict,
    level0_config,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Bound an explicit session/date to its complete local scientific context.

    Include competing sessions within twice the maximum dark-current distance,
    plus two acquisition gap lengths on both sides. This prevents selecting a
    dark current belonging to another nearby session. Final session identity,
    profile boundaries, QA and dark-current associations are still calculated
    from original Licel timestamps, never from directory names.
    """
    if selection is None or selection.is_empty or selection.paths:
        return df_raw
    windows = []
    for session_id in selection.session_ids:
        _station, start, end = session_id_parts(session_id)
        windows.append((pd.Timestamp(start), pd.Timestamp(end)))
    station_zone = ZoneInfo(station_timezone(config))
    for value in selection.dates:
        local = datetime.strptime(value, "%Y%m%d").replace(tzinfo=station_zone)
        start = local.astimezone(timezone.utc)
        end = (local + timedelta(days=1)).astimezone(timezone.utc)
        windows.append((pd.Timestamp(start), pd.Timestamp(end)))
    if not windows:
        return df_raw

    margin = pd.Timedelta(
        hours=2.0 * level0_config.dark_current.max_association_hours,
        seconds=2.0 * level0_config.session.max_gap_seconds,
    )
    starts = df_raw["start_time_utc"]
    ends = df_raw["stop_time"].fillna(starts)
    mask = pd.Series(False, index=df_raw.index)
    for start, end in windows:
        mask |= (starts <= end + margin) & (ends >= start - margin)
    selected = df_raw[mask].copy()
    logger.info(
        "early temporal selection | %d/%d headers retained | margin %.1f h",
        len(selected), len(df_raw), margin.total_seconds() / 3600,
    )
    return selected


def build_session_inventory(
    raw_dir: str,
    config: dict,
    logger: logging.Logger,
    *,
    selection: InputSelection | None = None,
) -> pd.DataFrame:
    """Build the Level-0 inventory; optionally restrict to safe time candidates."""
    started = perf_counter()
    inventory_logger = bind_log_context(logger, stage="inventory")
    discovery_logger = bind_log_context(logger, stage="discovery")
    inventory_logger.info("building raw inventory | station=%s", station_id(config))
    level0_config = resolve_level0_config(config)
    discovery = level0_config.discovery

    scan_started = perf_counter()
    file_paths, file_types = scan_raw_files(
        raw_dir,
        spurious_extensions=discovery.spurious_extensions,
        quarantine_dir=discovery.quarantine_dir,
        raw_scan_ignore_dirs=discovery.raw_scan_ignore_dirs,
        logger=discovery_logger,
    )
    discovery_logger.debug("raw scan elapsed=%.2fs", perf_counter() - scan_started)
    if not file_paths:
        inventory_logger.info("inventory empty | %.1fs", perf_counter() - started)
        return pd.DataFrame()

    header_logger = bind_log_context(logger, stage="headers")
    headers_started = perf_counter()
    last_progress = headers_started
    records = []
    progress_every = max(1000, len(file_paths) // 10)
    config_path = config.get("_config_file")
    # A loaded config provides a stable project-local cache path. Pure library
    # test/config callers without that path stay stateless.
    cache_context = (
        LicelHeaderCache(Path(config_path).resolve().parent / ".cache" / "libids", header_logger)
        if config_path else nullcontext(None)
    )
    with cache_context as cache:
        for position, (filepath, file_type) in enumerate(zip(file_paths, file_types), start=1):
            start_time_utc, stop_time, duration, n_shots, laser_freq = (
                cache.header(
                    filepath, file_type,
                    lambda path: read_licel_header(path, logger=header_logger),
                )
                if cache is not None else read_licel_header(filepath, logger=header_logger)
            )
            if start_time_utc is not None:
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
            now = perf_counter()
            if (
                position % progress_every == 0
                or now - last_progress >= 20.0
            ) and position < len(file_paths):
                header_logger.info(
                    "read %d/%d (%.0f%%) | %d accepted | %d invalid | %.1fs",
                    position,
                    len(file_paths),
                    position * 100.0 / len(file_paths),
                    len(records),
                    position - len(records),
                    now - headers_started,
                )
                last_progress = now
    
    header_logger.info(
        "read %d/%d | %d valid | %d invalid | %.1fs",
        len(file_paths),
        len(file_paths),
        len(records),
        len(file_paths) - len(records),
        perf_counter() - headers_started,
    )
    df_raw = pd.DataFrame.from_records(records)
    if df_raw.empty:
        inventory_logger.warning("inventory has no valid headers | %.1fs", perf_counter() - started)
        return df_raw

    normalization_started = perf_counter()
    df_raw = _normalize_header_times(df_raw)
    df_raw = df_raw[df_raw["start_time_utc"].notna()].copy()
    if df_raw.empty:
        inventory_logger.warning("inventory has no dated headers | %.1fs", perf_counter() - started)
        return df_raw
    inventory_logger.debug("header normalization elapsed=%.2fs", perf_counter() - normalization_started)

    df_raw = _select_temporal_candidates(
        df_raw, selection, config, level0_config,
        bind_log_context(logger, stage="selection"),
    )
    if df_raw.empty:
        inventory_logger.info("no raw candidates in selected time window")
        return pd.DataFrame()

    timezone_name = station_timezone(config)
    df_raw["start_time_local"] = df_raw["start_time_utc"].dt.tz_convert(timezone_name)
    df_raw["stop_time_local"] = df_raw["stop_time"].dt.tz_convert(timezone_name)

    measurement_rows = df_raw[df_raw["meas_type"] == "measurements"].copy()
    dark_rows = df_raw[df_raw["meas_type"] == "dark_current"].copy()

    station_started = perf_counter()
    station_logger = bind_log_context(logger, stage="station")
    measurement_rows = _annotate_station_context(measurement_rows, config, station_logger)
    station_logger.info(
        "%d measurement headers | profile/calibration identities | %.1fs",
        len(measurement_rows),
        perf_counter() - station_started,
    )

    session_started = perf_counter()
    measurement_rows = _sessionize_measurements(
        measurement_rows,
        station=station_id(config),
        max_gap_seconds=level0_config.session.max_gap_seconds,
        logger=bind_log_context(logger, stage="sessions"),
    )
    bind_log_context(logger, stage="sessions").info(
        "%d sessions | %d measurement headers | %.1fs",
        int(measurement_rows["session_id"].nunique()) if not measurement_rows.empty else 0,
        len(measurement_rows),
        perf_counter() - session_started,
    )
    if not measurement_rows.empty:
        measurement_rows["association_method"] = "measurement"
        measurement_rows["dark_current_association_delta_hours"] = np.nan

    dark_started = perf_counter()
    dark_rows = _associate_dark_currents(
        dark_rows,
        measurement_rows,
        config,
        bind_log_context(logger, stage="darkcurrent"),
    )
    inventory_logger.debug("dark-current association elapsed=%.2fs", perf_counter() - dark_started)

    inventory = pd.concat([measurement_rows, dark_rows], ignore_index=True, sort=False)
    if inventory.empty:
        return inventory
    inventory["original_session_id"] = inventory["session_id"]
    inventory_logger.info(
        "ready | %d sessions | %d measurement + %d dark-current files | %.1fs total",
        int(measurement_rows["session_id"].nunique()) if not measurement_rows.empty else 0,
        len(measurement_rows),
        len(dark_rows),
        perf_counter() - started,
    )
    return inventory.sort_values(["start_time_utc", "filepath"]).reset_index(drop=True)
