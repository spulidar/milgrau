"""Shared parsing for flexible MILGRAU -i/--input selectors."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from milgrau.io.paths import (
    is_session_id,
    processed_data_root,
    product_session_id,
    session_id_parts,
    validate_session_id_for_config,
)


@dataclass(frozen=True)
class InputSelection:
    """Normalized explicit selection independent of pipeline level."""

    dates: frozenset[str]
    session_ids: frozenset[str]
    paths: tuple[Path, ...]

    @property
    def is_empty(self) -> bool:
        return not self.dates and not self.session_ids and not self.paths


def _groups(values: Sequence[Any] | None) -> list[list[str]]:
    if not values:
        return []
    groups: list[list[str]] = []
    for raw_group in values:
        raw_tokens = (
            [str(value) for value in raw_group]
            if isinstance(raw_group, (list, tuple))
            else [str(raw_group)]
        )
        tokens: list[str] = []
        for raw in raw_tokens:
            if not raw.strip():
                continue
            if Path(raw).expanduser().exists():
                tokens.append(raw)
            else:
                tokens.extend(shlex.split(raw) or [raw])
        if tokens:
            groups.append(tokens)
    return groups


def _is_date(value: str) -> bool:
    text = str(value).strip()
    if len(text) != 8 or not text.isdigit():
        return False
    try:
        datetime.strptime(text, "%Y%m%d")
    except ValueError:
        return False
    return True


def _station_timezone_name(config: Mapping[str, Any]) -> str:
    catalog = config.get("_station_catalog")
    if not isinstance(catalog, Mapping):
        raise KeyError("No station catalog is loaded; configure station_config in config.yaml.")
    station = catalog.get("station")
    if not isinstance(station, Mapping):
        raise KeyError("Station catalog must contain station metadata.")
    value = str(station.get("timezone", "")).strip()
    if not value:
        raise ValueError("station.timezone must be a non-empty IANA timezone.")
    return value


def parse_input_selection(
    values: Sequence[Any] | None,
    config: Mapping[str, Any],
) -> InputSelection:
    """Parse session IDs, local civil dates, and explicit existing paths."""
    dates: set[str] = set()
    session_ids: set[str] = set()
    paths: list[Path] = []

    for group in _groups(values):
        for token in group:
            path = Path(token).expanduser()
            if path.exists():
                paths.append(path)
                continue
            value = token.strip()
            if is_session_id(value):
                session_ids.add(validate_session_id_for_config(value, config))
                continue
            if _is_date(value):
                dates.add(value)
                continue
            raise ValueError(
                f"Invalid input selector {token!r}. Expected "
                "station_YYYYMMDD-HHMMZ_YYYYMMDD-HHMMZ, YYYYMMDD, "
                "or an existing file/directory path."
            )

    unique_paths = tuple(dict.fromkeys(path.resolve() for path in paths))
    return InputSelection(
        dates=frozenset(dates),
        session_ids=frozenset(session_ids),
        paths=unique_paths,
    )


def _session_intersects_local_date(session_id: str, date_text: str, timezone_name: str) -> bool:
    """Return whether a UTC session overlaps one station-local civil day."""
    _station, start_utc, end_utc = session_id_parts(session_id)
    zone = ZoneInfo(timezone_name)
    day = datetime.strptime(date_text, "%Y%m%d")
    local_start = day.replace(tzinfo=zone)
    local_end = local_start + timedelta(days=1)
    day_start_utc = local_start.astimezone(start_utc.tzinfo)
    day_end_utc = local_end.astimezone(start_utc.tzinfo)
    return end_utc > day_start_utc and start_utc < day_end_utc


def select_available_session_ids(
    selection: InputSelection,
    available_ids: Sequence[str],
    config: Mapping[str, Any],
) -> set[str]:
    """Resolve exact/date selectors against session IDs that actually exist."""
    available = {validate_session_id_for_config(str(value).strip(), config) for value in available_ids}
    selected = set(selection.session_ids)
    missing_exact = sorted(selected - available)
    if missing_exact:
        raise FileNotFoundError(
            "Requested session(s) not found: " + ", ".join(missing_exact)
        )

    timezone_name = _station_timezone_name(config)
    for date_text in selection.dates:
        matches = {
            value
            for value in available
            if _session_intersects_local_date(value, date_text, timezone_name)
        }
        if not matches:
            raise FileNotFoundError(
                f"No sessions intersect station-local date {date_text}."
            )
        selected.update(matches)
    return selected


def resolve_product_selection(
    selection: InputSelection,
    config: Mapping[str, Any],
    *,
    suffix: str,
) -> list[Path]:
    """Resolve a session/date/path selection to existing canonical product files."""
    root = processed_data_root(config)
    discovered: dict[str, list[Path]] = {}
    for path in sorted(root.rglob(f"*{suffix}")):
        if not path.is_file():
            continue
        try:
            session_id = product_session_id(path)
        except ValueError:
            continue
        discovered.setdefault(session_id, []).append(path)

    if selection.is_empty:
        return [path for paths in discovered.values() for path in paths]

    selected_ids = select_available_session_ids(
        selection,
        list(discovered),
        config,
    ) if (selection.session_ids or selection.dates) else set()

    resolved: list[Path] = []
    for session_id in sorted(selected_ids):
        resolved.extend(discovered.get(session_id, []))

    for path in selection.paths:
        if path.is_dir():
            resolved.extend(
                item
                for item in sorted(path.rglob(f"*{suffix}"))
                if item.is_file()
            )
        else:
            resolved.append(path)

    unique = sorted(dict.fromkeys(path.resolve() for path in resolved))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Selected product(s) not found: " + ", ".join(str(path) for path in missing)
        )
    return unique
