"""Radiosonde retrieval, deterministic temporal selection, and caching utilities."""

from __future__ import annotations

import json
import logging
import math
from importlib.metadata import PackageNotFoundError, version as package_version
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
from siphon.simplewebservice.wyoming import WyomingUpperAir
from tenacity import retry, stop_after_attempt, wait_exponential

from milgrau.io.igra_archive import load_igra2_cached_sounding
from milgrau.io.paths import resolve_project_path


def _require_current_siphon() -> str:
    """Require the Wyoming client generation compatible with the current server."""
    try:
        resolved = package_version("siphon")
    except PackageNotFoundError as exc:
        raise RuntimeError("Siphon is not installed; install MILGRAU runtime dependencies.") from exc

    try:
        major, minor = (int(part) for part in resolved.split(".")[:2])
    except (TypeError, ValueError):
        return resolved
    if (major, minor) < (0, 11):
        raise RuntimeError(
            f"Siphon {resolved} is incompatible with the current University of Wyoming "
            "upper-air server; upgrade to siphon>=0.11.0."
        )
    return resolved


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


def _synoptic_candidates_within_window(
    measurement_dt: datetime,
    *,
    synoptic_hours_utc: Sequence[int],
    selection: str,
    max_time_delta_hours: float,
) -> list[datetime]:
    """Closest nominal UTC slots to inspect for *actual* cached observations."""
    if str(selection).strip().lower() != "nearest":
        raise ValueError("Radiosonde selection currently supports only 'nearest'.")
    hours = list(synoptic_hours_utc)
    if not hours or any(
        isinstance(h, bool) or not isinstance(h, int) or h < 0 or h > 23
        for h in hours
    ) or len(set(hours)) != len(hours):
        raise ValueError("Radiosonde synoptic hours must be unique integers between 0 and 23.")
    max_delta = float(max_time_delta_hours)
    if not math.isfinite(max_delta) or max_delta <= 0.0:
        raise ValueError("Radiosonde max_time_delta_hours must be positive and finite.")
    day = measurement_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    day_span = math.ceil(max_delta / 24.0) + 1
    candidates = [
        day + timedelta(days=offset, hours=hour)
        for offset in range(-day_span, day_span + 1)
        for hour in hours
    ]
    candidates = [
        stamp for stamp in candidates
        if abs((stamp - measurement_dt).total_seconds()) <= max_delta * 3600
    ]
    return sorted(
        candidates,
        key=lambda stamp: (abs((stamp - measurement_dt).total_seconds()), stamp),
    )


def _cached_sounding(
    cache_root: Path,
    target_dt: datetime,
    measurement_dt: datetime,
    station_id: str,
    logger: logging.Logger,
    base_metadata: Mapping[str, Any],
) -> pd.DataFrame | None:
    """Try validated local CSV, then IGRA2 for one nominal UTC launch."""
    cache_path = cache_root / target_dt.strftime("%Y") / target_dt.strftime("%m")
    cache_filename = f"radiosonde_{station_id}_{target_dt.strftime('%Y%m%d_%H')}Z.csv"
    cache_file = cache_path / cache_filename
    metadata_file = _metadata_file_for(cache_file)
    metadata = {
        **base_metadata,
        "measurement_datetime_utc": measurement_dt.isoformat(),
        "target_datetime_utc": target_dt.isoformat(),
        "time_delta_hours": abs((target_dt - measurement_dt).total_seconds()) / 3600.0,
        "csv_file": cache_file.name,
    }

    if cache_file.exists():
        try:
            cached = pd.read_csv(cache_file)
            if not {"height", "temperature", "pressure"}.issubset(cached.columns):
                raise ValueError("Required radiosonde columns missing.")
            values = cached.dropna(subset=["height", "temperature", "pressure"])
            if len(values) < 2:
                raise ValueError("Insufficient valid radiosonde levels.")
            logger.debug("radiosonde hourly CSV cache hit: %s", cache_filename)
            sidecar = _read_metadata(metadata_file)
            source_keys = {
                key: sidecar[key] for key in (
                    "source", "historical_archive", "historical_station_id",
                    "source_type", "download_datetime_utc",
                ) if key in sidecar
            }
            return _attach_metadata(cached, {**metadata, **source_keys})
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            logger.warning("radiosonde CSV cache unusable | %s | %s", cache_filename, exc)

    historical = load_igra2_cached_sounding(cache_root, station_id, target_dt, logger)
    if historical is None:
        return None
    metadata = {**metadata, **dict(historical.attrs)}
    metadata["measurement_datetime_utc"] = measurement_dt.isoformat()
    metadata["time_delta_hours"] = abs((target_dt - measurement_dt).total_seconds()) / 3600.0
    cache_path.mkdir(parents=True, exist_ok=True)
    historical.to_csv(cache_file, index=False)
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.debug("IGRA2 historical radiosonde saved as hourly CSV: %s", cache_filename)
    return _attach_metadata(historical, metadata)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
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
    """Get a sounding from CSV/IGRA2 local caches, then Wyoming Siphon.

    Historical IGRA2 data remains an external operator-supplied cache file;
    it is never downloaded or bundled automatically by MILGRAU.
    """
    measurement_dt = _as_utc_datetime(measurement_dt_utc)
    station_id = str(station_id).strip()
    if not station_id:
        raise ValueError("A non-empty radiosonde station_id is required.")

    candidates = _synoptic_candidates_within_window(
        measurement_dt,
        synoptic_hours_utc=synoptic_hours_utc,
        selection=selection,
        max_time_delta_hours=max_time_delta_hours,
    )
    if not candidates:
        logger.warning("no configured synoptic sounding within %.2f h", float(max_time_delta_hours))
        return None
    cache_root = resolve_project_path(cache_dir)
    default_metadata = {
        "station_id": station_id,
        "measurement_datetime_utc": measurement_dt.isoformat(),
        "temporal_selection": str(selection).strip().lower(),
        "synoptic_hours_utc": ",".join(str(int(hour)) for hour in synoptic_hours_utc),
        "max_time_delta_hours": float(max_time_delta_hours),
        "source": "University of Wyoming Upper Air via Siphon",
        "source_type": "radiosonde",
    }

    # SPU IGRA2 often has 12Z-only records. At 19Z, midnight tomorrow is
    # nominally closer but may not exist; reuse the actually observed 12Z
    # profile if it is inside the explicit temporal tolerance.
    for candidate in candidates:
        local = _cached_sounding(
            cache_root, candidate, measurement_dt, station_id, logger,
            default_metadata,
        )
        if local is not None:
            return local

    target_dt = candidates[0]
    cache_path = cache_root / target_dt.strftime("%Y") / target_dt.strftime("%m")
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_filename = f"radiosonde_{station_id}_{target_dt.strftime('%Y%m%d_%H')}Z.csv"
    cache_file = cache_path / cache_filename
    metadata_file = _metadata_file_for(cache_file)
    default_metadata.update({
        "target_datetime_utc": target_dt.isoformat(),
        "time_delta_hours": abs((target_dt - measurement_dt).total_seconds()) / 3600.0,
        "csv_file": cache_file.name,
    })
    siphon_version = _require_current_siphon()
    logger.info(
        "radiosonde target | %sZ | station=%s | Δt=%.2f h | siphon=%s",
        target_dt.strftime("%Y-%m-%d %H:%M"),
        station_id,
        abs((target_dt - measurement_dt).total_seconds()) / 3600.0,
        siphon_version,
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
