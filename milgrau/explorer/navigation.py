"""UTC calendar navigation for continuous MILGRAU scientific sessions.

A session is available on every UTC date it overlaps; choosing a date never
splits, renames or merges the underlying scientific session.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from milgrau.io.paths import session_id_parts


def session_utc_dates(session_id: str) -> tuple[date, ...]:
    """Calendar dates intersecting the UTC start/end of one session."""
    _station, start, end = session_id_parts(session_id)
    current = start.date()
    dates: list[date] = []
    while current <= end.date():
        dates.append(current)
        current += timedelta(days=1)
    return tuple(dates)


def available_utc_dates(inventory: pd.DataFrame) -> list[date]:
    """Newest first, with no local-time or six-hour window assumptions."""
    dates: set[date] = set()
    for session_id in inventory.get("session_id", []):
        try:
            dates.update(session_utc_dates(str(session_id)))
        except ValueError:
            continue
    return sorted(dates, reverse=True)


def sessions_on_utc_date(inventory: pd.DataFrame, selected: date) -> pd.DataFrame:
    """Preserve all intersecting sessions; never concatenate their products."""
    valid: list[bool] = []
    for session_id in inventory.get("session_id", []):
        try:
            valid.append(selected in session_utc_dates(str(session_id)))
        except ValueError:
            valid.append(False)
    return inventory.loc[valid].copy()


__all__ = ["session_utc_dates", "available_utc_dates", "sessions_on_utc_date"]
