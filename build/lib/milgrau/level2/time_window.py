"""UTC time-window parsing and subsetting for Level 2 retrievals."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def normalize_utc_timestamp(value: Any) -> pd.Timestamp:
    """Return a naive UTC timestamp for a parsed CLI or dataset time."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        return stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def parse_time_bound(raw: str, reference_time: pd.Timestamp) -> pd.Timestamp:
    """Parse one UTC bound, anchoring bare times to the reference date."""
    value = str(raw).strip()
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", value):
        pieces = [int(part) for part in value.split(":")]
        hour, minute = pieces[0], pieces[1]
        second = pieces[2] if len(pieces) > 2 else 0
        return pd.Timestamp(
            year=reference_time.year,
            month=reference_time.month,
            day=reference_time.day,
            hour=hour,
            minute=minute,
            second=second,
        )
    return normalize_utc_timestamp(value)


def parse_utc_time_window(
    ds_l1: xr.Dataset,
    start_utc: str | None,
    stop_utc: str | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str | None]:
    """Resolve optional CLI time bounds against one Level 1 file."""
    if start_utc is None and stop_utc is None:
        return None, None, None
    if start_utc is None or stop_utc is None:
        raise ValueError("Both start_utc and stop_utc must be provided together.")
    if "time" not in ds_l1.coords or ds_l1["time"].size == 0:
        raise ValueError("Level 1 file has no time coordinate to apply a UTC window.")

    start_raw = str(start_utc).strip()
    stop_raw = str(stop_utc).strip()
    reference_time = normalize_utc_timestamp(ds_l1["time"].values[0])

    start_ts = parse_time_bound(start_raw, reference_time)
    stop_ts = parse_time_bound(stop_raw, reference_time)
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", start_raw) and re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", stop_raw):
        if stop_ts <= start_ts:
            stop_ts = stop_ts + pd.Timedelta(days=1)
    elif stop_ts <= start_ts:
        raise ValueError("stop_utc must be later than start_utc.")

    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_convert("UTC").tz_localize(None)
    if stop_ts.tzinfo is not None:
        stop_ts = stop_ts.tz_convert("UTC").tz_localize(None)

    if start_ts == stop_ts:
        raise ValueError("start_utc and stop_utc must not be equal.")

    if start_ts.date() == stop_ts.date():
        tag = f"{start_ts:%H%M}-{stop_ts:%H%M}"
    else:
        tag = f"{start_ts:%Y%m%dT%H%M}-{stop_ts:%Y%m%dT%H%M}"
    return start_ts, stop_ts, tag


def subset_level1_time_window(
    ds_l1: xr.Dataset,
    start_utc: str | None,
    stop_utc: str | None,
) -> tuple[xr.Dataset, str | None]:
    """Subset a Level 1 dataset to an optional UTC time window."""
    start_ts, stop_ts, tag = parse_utc_time_window(ds_l1, start_utc, stop_utc)
    if start_ts is None or stop_ts is None:
        return ds_l1, None

    time_index = pd.to_datetime(ds_l1["time"].values)
    if getattr(time_index, "tz", None) is not None:
        time_index = time_index.tz_convert("UTC").tz_localize(None)
    mask = (time_index >= start_ts) & (time_index < stop_ts)
    if not bool(np.any(mask)):
        raise ValueError(f"No Level 1 profiles fall within UTC window {start_ts} to {stop_ts}.")

    selected = ds_l1.isel(time=np.flatnonzero(mask))
    selected = selected.copy()
    selected.attrs.update(
        {
            "LEBEAR_Time_Window_UTC": f"{start_ts.isoformat()} to {stop_ts.isoformat()}",
            "LEBEAR_Time_Window_Tag": tag,
        }
    )
    return selected, tag
