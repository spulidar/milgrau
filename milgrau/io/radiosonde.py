"""Radiosonde retrieval, deterministic temporal selection, and caching utilities."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
from siphon.simplewebservice.wyoming import WyomingUpperAir
from tenacity import retry, stop_after_attempt, wait_exponential

from milgrau.io.paths import resolve_project_path
from milgrau.io.weather import return_none_on_failure


def _metadata_file_for(cache_file: Path) -> Path:
    return cache_file.with_suffix(".json")


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _attach_metadata(df: pd.DataFrame, metadata: Mapping[str, Any]) -> pd.DataFrame:
    result = df.copy()
    result.attrs.update(dict(metadata))
    return result


def _as_utc_datetime(value: datetime | pd.Timestamp) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def select_radiosonde_target_datetime(
    measurement_dt_utc: datetime | pd.Timestamp,
    *,
    synoptic_hours_utc: Sequence[int],
    selection: str,
    max_time_delta_hours: float,
) -> datetime | None:
    """Resolve the sounding datetime from an explicit temporal-selection policy."""
    measurement_dt = _as_utc_datetime(measurement_dt_utc)
    if str(selection).strip().lower() != "nearest":
        raise ValueError("Radiosonde selection currently supports only 'nearest'.")

    hours: list[int] = []
    for raw_hour in synoptic_hours_utc:
        if isinstance(raw_hour, bool) or not isinstance(raw_hour, int):
            raise ValueError("Radiosonde synoptic hours must be integers between 0 and 23.")
        hour = int(raw_hour)
        if hour < 0 or hour > 23 or hour in hours:
            raise ValueError("Radiosonde synoptic hours must be unique integers between 0 and 23.")
        hours.append(hour)
    if not hours:
        raise ValueError("At least one radiosonde synoptic hour must be configured.")

    max_delta = float(max_time_delta_hours)
    if not math.isfinite(max_delta) or max_delta <= 0.0:
        raise ValueError("Radiosonde max_time_delta_hours must be positive and finite.")

    base_day = measurement_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    candidates = [
        base_day + timedelta(days=day_offset, hours=hour)
        for day_offset in (-1, 0, 1)
        for hour in sorted(hours)
    ]
    target_dt = min(
        candidates,
        key=lambda candidate: (abs((candidate - measurement_dt).total_seconds()), candidate),
    )
    delta_hours = abs((target_dt - measurement_dt).total_seconds()) / 3600.0
    return target_dt if delta_hours <= max_delta else None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=return_none_on_failure,
)
def fetch_wyoming_radiosonde(
    measurement_dt_utc: datetime | pd.Timestamp,
    station_id: str,
    logger: logging.Logger,
    *,
    cache_dir: str | Path,
    synoptic_hours_utc: Sequence[int],
    selection: str,
    max_time_delta_hours: float,
) -> Optional[pd.DataFrame]:
    """Fetch one Wyoming sounding using only the explicitly configured policy."""
    measurement_dt = _as_utc_datetime(measurement_dt_utc)
    station_id = str(station_id).strip()
    if not station_id:
        raise ValueError("A non-empty radiosonde station_id is required.")

    target_dt = select_radiosonde_target_datetime(
        measurement_dt,
        synoptic_hours_utc=synoptic_hours_utc,
        selection=selection,
        max_time_delta_hours=max_time_delta_hours,
    )
    if target_dt is None:
        logger.warning(
            "no configured synoptic sounding within %.2f h",
            float(max_time_delta_hours),
        )
        return None

    cache_root = resolve_project_path(cache_dir)
    cache_path = cache_root / target_dt.strftime("%Y") / target_dt.strftime("%m")
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_filename = f"radiosonde_{station_id}_{target_dt.strftime('%Y%m%d_%H')}Z.csv"
    cache_file = cache_path / cache_filename
    metadata_file = _metadata_file_for(cache_file)
    default_metadata = {
        "station_id": station_id,
        "measurement_datetime_utc": measurement_dt.isoformat(),
        "target_datetime_utc": target_dt.isoformat(),
        "time_delta_hours": abs((target_dt - measurement_dt).total_seconds()) / 3600.0,
        "temporal_selection": str(selection).strip().lower(),
        "synoptic_hours_utc": ",".join(str(int(hour)) for hour in synoptic_hours_utc),
        "max_time_delta_hours": float(max_time_delta_hours),
        "source": "University of Wyoming Upper Air via Siphon",
        "source_type": "radiosonde",
        "csv_file": cache_file.name,
    }

    if cache_file.exists():
        logger.debug("radiosonde cache hit: %s", cache_filename)
        metadata = {**default_metadata, **_read_metadata(metadata_file)}
        return _attach_metadata(pd.read_csv(cache_file), metadata)

    logger.debug(
        "radiosonde fetch: %sZ | station=%s",
        target_dt.strftime("%Y-%m-%d %H:%M"),
        station_id,
    )
    df_raw = WyomingUpperAir.request_data(target_dt, station_id)
    df = df_raw.drop_duplicates(subset=["height"], keep="first").sort_values("height")
    df.to_csv(cache_file, index=False)
    metadata = {
        **default_metadata,
        "download_datetime_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.debug("radiosonde cached: %s", cache_filename)
    return _attach_metadata(df, metadata)
