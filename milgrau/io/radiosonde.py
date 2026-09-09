"""Radiosonde retrieval and caching utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
from siphon.simplewebservice.wyoming import WyomingUpperAir
from tenacity import retry, stop_after_attempt, wait_exponential

from milgrau.io.paths import radiosonde_cache_dir
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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=return_none_on_failure,
)
def fetch_wyoming_radiosonde(
    measurement_dt_utc: datetime,
    station_id: str,
    logger: logging.Logger,
    cache_dir: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    root_dir: str | Path | None = None,
) -> Optional[pd.DataFrame]:
    """Fetch Wyoming radiosonde data and cache the cleaned table locally.

    Returned dataframes carry provenance in ``DataFrame.attrs`` so downstream
    products can preserve the actual sounding time and measurement-time offset.
    """
    measurement_dt = pd.Timestamp(measurement_dt_utc)
    if measurement_dt.tzinfo is None:
        measurement_dt = measurement_dt.tz_localize("UTC")
    else:
        measurement_dt = measurement_dt.tz_convert("UTC")
    measurement_dt_utc = measurement_dt.to_pydatetime()

    hour_utc = measurement_dt_utc.hour
    if 0 <= hour_utc <= 8:
        target_dt = measurement_dt_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    elif 9 <= hour_utc <= 20:
        target_dt = measurement_dt_utc.replace(hour=12, minute=0, second=0, microsecond=0)
    else:
        target_dt = (measurement_dt_utc + timedelta(days=1)).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

    year = target_dt.strftime("%Y")
    month = target_dt.strftime("%m")
    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else radiosonde_cache_dir(config, root_dir=root_dir)
    cache_path = Path(resolved_cache_dir) / year / month
    cache_path.mkdir(parents=True, exist_ok=True)

    cache_filename = f"radiosonde_{station_id}_{target_dt.strftime('%Y%m%d_%H')}Z.csv"
    cache_file = cache_path / cache_filename
    metadata_file = _metadata_file_for(cache_file)
    default_metadata = {
        "station_id": str(station_id),
        "measurement_datetime_utc": measurement_dt_utc.isoformat(),
        "target_datetime_utc": target_dt.isoformat(),
        "time_delta_hours": abs((target_dt - measurement_dt_utc).total_seconds()) / 3600.0,
        "source": "University of Wyoming Upper Air via Siphon",
        "source_type": "radiosonde",
        "csv_file": cache_file.name,
    }

    if cache_file.exists():
        logger.info(f"  -> [RADIOSONDE] Cached sounding found: {cache_filename}. Skipping download.")
        metadata = {**default_metadata, **_read_metadata(metadata_file)}
        return _attach_metadata(pd.read_csv(cache_file), metadata)

    logger.info(
        f"  -> [RADIOSONDE] Fetching {target_dt.strftime('%Y-%m-%d %H:%M')}Z "
        f"for station {station_id} via Siphon..."
    )

    df_raw = WyomingUpperAir.request_data(target_dt, station_id)
    df = df_raw.drop_duplicates(subset=["height"], keep="first").sort_values("height")
    df.to_csv(cache_file, index=False)

    metadata = {
        **default_metadata,
        "download_datetime_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("  -> [OK] Radiosonde data successfully fetched and cached!")
    return _attach_metadata(df, metadata)
