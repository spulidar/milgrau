"""Time classification helpers for MILGRAU acquisition grouping."""

from __future__ import annotations

from typing import Any

from milgrau.io.paths import build_measurement_id

LOCAL_PERIOD_HOURS = 6


def _require_timezone_aware(local_dt: Any) -> None:
    """Reject naive timestamps because grouping is defined in station local time."""
    if getattr(local_dt, "tzinfo", None) is None or local_dt.utcoffset() is None:
        raise ValueError("MILGRAU period classification requires a timezone-aware local timestamp.")


def period_start_hour(local_dt: Any) -> int:
    """Return the local wall-clock start hour of the six-hour period."""
    _require_timezone_aware(local_dt)
    return (int(local_dt.hour) // LOCAL_PERIOD_HOURS) * LOCAL_PERIOD_HOURS


def classify_period(local_dt: Any) -> str:
    """Return the fixed six-hour local-time period containing the timestamp."""
    start_hour = period_start_hour(local_dt)
    return f"{start_hour:02d}-{start_hour + LOCAL_PERIOD_HOURS:02d}"


def period_start_local(local_dt: Any) -> Any:
    """Return the station-local wall-clock start of the six-hour period."""
    start_hour = period_start_hour(local_dt)
    return local_dt.replace(hour=start_hour, minute=0, second=0, microsecond=0)


def measurement_id_for_local_time(local_dt: Any, station_id: str) -> str:
    """Return YYYYMMDD_station_HH using local civil date and local period start."""
    _require_timezone_aware(local_dt)
    return build_measurement_id(
        local_dt.strftime("%Y%m%d"),
        station_id,
        period_start_hour(local_dt),
    )
