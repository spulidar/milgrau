"""Time classification helpers for MILGRAU acquisition grouping."""

from __future__ import annotations

from datetime import timezone
from typing import Any

LOCAL_PERIOD_HOURS = 6


def _require_timezone_aware(local_dt: Any) -> None:
    """Reject naive timestamps because grouping is defined in station local time."""
    if getattr(local_dt, "tzinfo", None) is None or local_dt.utcoffset() is None:
        raise ValueError("MILGRAU period classification requires a timezone-aware local timestamp.")


def classify_period(local_dt: Any) -> str:
    """Return the fixed six-hour local-time period containing the timestamp."""
    _require_timezone_aware(local_dt)
    start_hour = (int(local_dt.hour) // LOCAL_PERIOD_HOURS) * LOCAL_PERIOD_HOURS
    stop_hour = start_hour + LOCAL_PERIOD_HOURS
    return f"{start_hour:02d}-{stop_hour:02d}"


def period_start_local(local_dt: Any) -> Any:
    """Return the station-local wall-clock start of the six-hour period."""
    _require_timezone_aware(local_dt)
    start_hour = (int(local_dt.hour) // LOCAL_PERIOD_HOURS) * LOCAL_PERIOD_HOURS
    return local_dt.replace(hour=start_hour, minute=0, second=0, microsecond=0)


def period_utc_label(local_dt: Any) -> str:
    """Return a compact UTC label derived from the local period start."""
    utc_start = period_start_local(local_dt).astimezone(timezone.utc)
    if int(utc_start.minute) == 0:
        return f"{int(utc_start.hour):02d}z"
    return f"{int(utc_start.hour):02d}{int(utc_start.minute):02d}z"


def measurement_id_for_local_time(local_dt: Any) -> str:
    """Return a local civil date plus the UTC label of the local period start."""
    _require_timezone_aware(local_dt)
    return f"{local_dt.strftime('%Y%m%d')}{period_utc_label(local_dt)}"
