"""Shared parsing for flexible MILGRAU -i/--input selectors."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from milgrau.io.paths import (
    LOCAL_PERIOD_STARTS,
    build_measurement_id,
    is_measurement_id,
    normalize_period_start,
    station_id,
    validate_measurement_id_for_config,
)


@dataclass(frozen=True)
class InputSelection:
    """Normalized explicit selection independent of pipeline level."""

    dates: frozenset[str]
    measurement_ids: frozenset[str]
    paths: tuple[Path, ...]

    @property
    def is_empty(self) -> bool:
        return not self.dates and not self.measurement_ids and not self.paths


def _groups(values: Sequence[Any] | None) -> list[list[str]]:
    if not values:
        return []
    groups: list[list[str]] = []
    for raw_group in values:
        if isinstance(raw_group, (list, tuple)):
            tokens = [str(value) for value in raw_group if str(value).strip()]
        else:
            raw = str(raw_group)
            tokens = shlex.split(raw) or [raw]
        if tokens:
            groups.append(tokens)
    return groups


def _is_date(value: str) -> bool:
    text = str(value).strip()
    if len(text) != 8 or not text.isdigit():
        return False
    try:
        build_measurement_id(text, "x", "00")
    except ValueError:
        return False
    return True


def _is_period_token(value: str) -> bool:
    try:
        normalize_period_start(value)
    except ValueError:
        return False
    return True


def parse_input_selection(
    values: Sequence[Any] | None,
    config: Mapping[str, Any],
) -> InputSelection:
    """Parse IDs, dates, date-plus-period groups, and explicit existing paths."""
    dates: set[str] = set()
    measurement_ids: set[str] = set()
    paths: list[Path] = []
    canonical_station: str | None = None

    for group in _groups(values):
        if _is_date(group[0]) and len(group) > 1 and all(_is_period_token(v) for v in group[1:]):
            date_text = group[0]
            if canonical_station is None:
                canonical_station = station_id(config)
            for period in group[1:]:
                measurement_ids.add(
                    build_measurement_id(date_text, canonical_station, normalize_period_start(period))
                )
            continue

        for token in group:
            path = Path(token).expanduser()
            if path.exists():
                paths.append(path)
                continue
            value = token.strip().lower()
            if is_measurement_id(value):
                measurement_ids.add(validate_measurement_id_for_config(value, config))
                continue
            if _is_date(value):
                if canonical_station is None:
                    canonical_station = station_id(config)
                dates.add(value)
                continue
            raise ValueError(
                f"Invalid input selector {token!r}. Expected YYYYMMDD_station_HH, "
                f"YYYYMMDD, YYYYMMDD followed by period(s) {', '.join(LOCAL_PERIOD_STARTS)}, "
                "or an existing file/directory path."
            )

    unique_paths = tuple(dict.fromkeys(path.resolve() for path in paths))
    return InputSelection(
        dates=frozenset(dates),
        measurement_ids=frozenset(measurement_ids),
        paths=unique_paths,
    )


def select_available_measurement_ids(
    selection: InputSelection,
    available_ids: Sequence[str],
) -> set[str]:
    """Resolve exact/date selectors against IDs that actually exist."""
    available = {str(value).strip().lower() for value in available_ids}
    selected = set(selection.measurement_ids)
    missing_exact = sorted(selected - available)
    if missing_exact:
        raise FileNotFoundError(
            "Requested measurement group(s) not found: " + ", ".join(missing_exact)
        )

    for date_text in selection.dates:
        matches = {value for value in available if value.startswith(f"{date_text}_")}
        if not matches:
            raise FileNotFoundError(f"No measurement groups found for date {date_text}.")
        selected.update(matches)
    return selected
