"""ERA5 upper-air retrieval and caching for MILGRAU.

All scientific/processing settings are supplied explicitly by the Level 1
atmosphere policy. This module contains only IO behavior, physical constants and
file-format handling; it does not invent semantic configuration defaults.
"""

from __future__ import annotations

import json
import logging
from hashlib import sha256
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Optional

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.io.paths import resolve_project_path

ERA5_DOI = "10.24381/cds.bd0915c6"
CDS_API_URL: Final[str] = "https://cds.climate.copernicus.eu/api"
_STANDARD_GRAVITY_M_S2 = 9.80665
_EARTH_RADIUS_M = 6_356_766.0
_REQUIRED_ERA5_KEYS = {
    "cache_dir", "dataset", "pressure_levels_hpa", "grid_deg", "area_half_width_deg"
}


def _as_utc_datetime(value: datetime | pd.Timestamp) -> datetime:
    """Return a timezone-aware UTC datetime without changing the instant."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def nearest_era5_analysis_hour(measurement_dt_utc: datetime | pd.Timestamp) -> datetime:
    """Round one measurement time to the nearest ERA5 hourly analysis."""
    value = _as_utc_datetime(measurement_dt_utc)
    rounded = value.replace(minute=0, second=0, microsecond=0)
    if value.minute >= 30:
        rounded += timedelta(hours=1)
    return rounded


def era5_config(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize complete ERA5 IO settings without defaults."""
    if not isinstance(settings, Mapping):
        raise TypeError("ERA5 settings must be a mapping.")
    missing = sorted(_REQUIRED_ERA5_KEYS - set(settings))
    unknown = sorted(set(settings) - _REQUIRED_ERA5_KEYS)
    if missing or unknown:
        raise KeyError(f"ERA5 settings must contain exactly {sorted(_REQUIRED_ERA5_KEYS)}; missing={missing}, unknown={unknown}.")

    cache_dir = settings["cache_dir"]
    dataset = settings["dataset"]
    if not isinstance(cache_dir, str) or not cache_dir.strip():
        raise ValueError("ERA5 cache_dir must be a non-empty string.")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("ERA5 dataset must be a non-empty string.")

    raw_levels = settings["pressure_levels_hpa"]
    if not isinstance(raw_levels, (list, tuple)) or not raw_levels:
        raise ValueError("ERA5 pressure_levels_hpa must be a non-empty list.")
    levels: list[int] = []
    for index, raw in enumerate(raw_levels):
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise ValueError(f"ERA5 pressure_levels_hpa[{index}] must be an integer.")
        value = int(raw)
        if value <= 0:
            raise ValueError(f"ERA5 pressure_levels_hpa[{index}] must be positive.")
        if value in levels:
            raise ValueError(f"ERA5 pressure_levels_hpa contains duplicate level {value}.")
        levels.append(value)

    grid = float(settings["grid_deg"])
    half_width = float(settings["area_half_width_deg"])
    if not np.isfinite(grid) or grid <= 0.0:
        raise ValueError("ERA5 grid_deg must be positive and finite.")
    if not np.isfinite(half_width) or half_width <= 0.0:
        raise ValueError("ERA5 area_half_width_deg must be positive and finite.")
    if grid > 5.0:
        raise ValueError("ERA5 grid_deg must be <= 5 degrees.")
    if half_width > 10.0:
        raise ValueError("ERA5 area_half_width_deg must be <= 10 degrees.")

    return {
        "cache_dir": cache_dir.strip(),
        "dataset": dataset.strip(),
        "pressure_levels_hpa": levels,
        "grid_deg": grid,
        "area_half_width_deg": half_width,
    }


def build_era5_request_for_hours(
    analysis_hours_utc: list[datetime],
    latitude: float,
    longitude: float,
    settings: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Build one CDS request for multiple ERA5 hours from one UTC day."""
    cfg = era5_config(settings)
    if not analysis_hours_utc:
        raise ValueError("ERA5 batch request requires at least one analysis hour.")
    hours = sorted({_as_utc_datetime(value).replace(minute=0, second=0, microsecond=0) for value in analysis_hours_utc})
    dates = {value.date() for value in hours}
    if len(dates) != 1:
        raise ValueError("One ERA5 batch request may contain hours from only one UTC day.")

    reference = hours[0]
    half_width = float(cfg["area_half_width_deg"])
    grid = float(cfg["grid_deg"])
    north = min(float(latitude) + half_width, 90.0)
    south = max(float(latitude) - half_width, -90.0)
    west = max(float(longitude) - half_width, -180.0)
    east = min(float(longitude) + half_width, 180.0)
    request = {
        "product_type": ["reanalysis"],
        "variable": ["temperature", "geopotential"],
        "year": [reference.strftime("%Y")],
        "month": [reference.strftime("%m")],
        "day": [reference.strftime("%d")],
        "time": [value.strftime("%H:00") for value in hours],
        "pressure_level": [str(level) for level in cfg["pressure_levels_hpa"]],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [north, west, south, east],
        "grid": [grid, grid],
    }
    return str(cfg["dataset"]), request


def build_era5_request(
    analysis_dt_utc: datetime,
    latitude: float,
    longitude: float,
    settings: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Build the CDS API request for one ERA5 pressure-level analysis hour."""
    return build_era5_request_for_hours(
        [_as_utc_datetime(analysis_dt_utc)],
        latitude,
        longitude,
        settings,
    )

def _cache_paths(
    measurement_dt_utc: datetime,
    latitude: float,
    longitude: float,
    settings: Mapping[str, Any],
    root_dir: str | Path | None,
) -> tuple[Path, Path]:
    cfg = era5_config(settings)
    cache_root = resolve_project_path(cfg["cache_dir"], root_dir=root_dir)
    analysis_dt = nearest_era5_analysis_hour(measurement_dt_utc)
    cache_dir = cache_root / analysis_dt.strftime("%Y") / analysis_dt.strftime("%m")
    cache_dir.mkdir(parents=True, exist_ok=True)
    lat_tag = f"{float(latitude):+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")
    lon_tag = f"{float(longitude):+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")
    stem = f"era5_pressure_levels_{analysis_dt.strftime('%Y%m%d_%H')}Z_{lat_tag}_{lon_tag}"
    netcdf = cache_dir / f"{stem}.nc"
    return netcdf, netcdf.with_suffix(".json")


def _coordinate_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.variables:
            return name
    raise KeyError(f"ERA5 dataset lacks required coordinate; tried {candidates}.")


def _variable_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"ERA5 dataset lacks required variable; tried {candidates}.")


def _select_site_column(data: xr.DataArray, latitude: float, longitude: float) -> xr.DataArray:
    result = data
    if "latitude" in result.coords:
        result = result.sel(latitude=float(latitude), method="nearest")
    if "longitude" in result.coords:
        lon_coord = np.asarray(result["longitude"].values, dtype=np.float64)
        target_lon = float(longitude)
        if lon_coord.size and np.nanmin(lon_coord) >= 0.0 and target_lon < 0.0:
            target_lon %= 360.0
        result = result.sel(longitude=target_lon, method="nearest")
    return result.squeeze(drop=True)


def _reduce_to_pressure_axis(data: xr.DataArray, pressure_coord: str) -> xr.DataArray:
    result = data
    for dim in tuple(result.dims):
        if dim != pressure_coord:
            result = result.isel({dim: 0})
    if result.dims != (pressure_coord,):
        result = result.transpose(pressure_coord)
    return result


def _geopotential_height_to_geometric_altitude(height_m: np.ndarray) -> np.ndarray:
    height = np.asarray(height_m, dtype=np.float64)
    denominator = _EARTH_RADIUS_M - height
    return np.divide(
        _EARTH_RADIUS_M * height,
        denominator,
        out=np.full_like(height, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )


def era5_profile_from_dataset(ds: xr.Dataset, latitude: float, longitude: float) -> pd.DataFrame:
    """Standardize one ERA5 NetCDF payload to height/temperature/pressure columns."""
    pressure_coord = _coordinate_name(ds, ("pressure_level", "level"))
    temperature_name = _variable_name(ds, ("t", "temperature"))
    geopotential_name = _variable_name(ds, ("z", "geopotential"))

    temperature = _reduce_to_pressure_axis(
        _select_site_column(ds[temperature_name], latitude, longitude), pressure_coord
    )
    geopotential = _reduce_to_pressure_axis(
        _select_site_column(ds[geopotential_name], latitude, longitude), pressure_coord
    )
    pressure = np.asarray(ds[pressure_coord].values, dtype=np.float64).reshape(-1)
    temperature_k = np.asarray(temperature.values, dtype=np.float64).reshape(-1)
    geopotential_m2_s2 = np.asarray(geopotential.values, dtype=np.float64).reshape(-1)
    if not (pressure.size == temperature_k.size == geopotential_m2_s2.size):
        raise ValueError("ERA5 pressure, temperature and geopotential axes are not conformable.")

    geopotential_height_m = geopotential_m2_s2 / _STANDARD_GRAVITY_M_S2
    geometric_altitude_m = _geopotential_height_to_geometric_altitude(geopotential_height_m)
    valid = (
        np.isfinite(pressure)
        & (pressure > 0.0)
        & np.isfinite(temperature_k)
        & (temperature_k > 0.0)
        & np.isfinite(geometric_altitude_m)
    )
    if valid.sum() < 2:
        raise ValueError("ERA5 profile contains fewer than two valid pressure levels.")

    frame = pd.DataFrame(
        {
            "height": geometric_altitude_m[valid],
            "temperature": temperature_k[valid] - 273.15,
            "temperature_k": temperature_k[valid],
            "pressure": pressure[valid],
        }
    )
    return frame.drop_duplicates(subset=["height"], keep="first").sort_values("height").reset_index(drop=True)


def _metadata_for_profile(
    measurement_dt_utc: datetime,
    analysis_dt_utc: datetime,
    latitude: float,
    longitude: float,
    dataset: str,
    cache_file: Path,
) -> dict[str, Any]:
    return {
        "source": "Copernicus Climate Change Service ERA5 pressure-level reanalysis",
        "source_type": "era5",
        "dataset": dataset,
        "doi": ERA5_DOI,
        "measurement_datetime_utc": _as_utc_datetime(measurement_dt_utc).isoformat(),
        "analysis_datetime_utc": _as_utc_datetime(analysis_dt_utc).isoformat(),
        "time_delta_hours": abs(
            (_as_utc_datetime(analysis_dt_utc) - _as_utc_datetime(measurement_dt_utc)).total_seconds()
        ) / 3600.0,
        "requested_latitude": float(latitude),
        "requested_longitude": float(longitude),
        "download_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "cache_file": cache_file.name,
    }


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _compact_exception_message(exc: BaseException) -> str:
    """Return one operator-readable line while detailed traceback stays at DEBUG."""
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    if not lines:
        return type(exc).__name__
    first = lines[0]
    if len(first) > 180:
        first = first[:177] + "..."
    return first


def _era5_time_coordinate(ds: xr.Dataset) -> str:
    return _coordinate_name(ds, ("valid_time", "time"))


ERA5_MAX_REQUEST_FIELDS = 80_000


def plan_era5_requests(
    missing_hours: list[datetime],
    latitude: float,
    longitude: float,
    settings: Mapping[str, Any],
    *,
    max_fields: int = ERA5_MAX_REQUEST_FIELDS,
) -> list[tuple[list[datetime], str, dict[str, Any], int]]:
    """Plan one GRIB CDS request per UTC month of *active days*, when possible.

    A month uses the union of required hours across its active days. The CDS
    returns a day x hour Cartesian product; count every combination under the
    80k-field ceiling, and retain only required hours in the local NetCDF cache.

    When a request would exceed max_fields, split its active days first, and
    split hours within a single day if necessary. No request spans two months.
    """
    cfg = era5_config(settings)
    if isinstance(max_fields, bool) or not isinstance(max_fields, int) or max_fields <= 0:
        raise ValueError("ERA5 max_fields must be a positive integer.")

    month_days: dict[tuple[int, int], dict[date, set[int]]] = {}
    for stamp in missing_hours:
        hour = nearest_era5_analysis_hour(stamp)
        day = hour.date()
        month_days.setdefault((day.year, day.month), {}).setdefault(day, set()).add(
            hour.hour
        )

    fields_per_hour = 2 * len(cfg["pressure_levels_hpa"])
    if fields_per_hour > max_fields:
        raise ValueError("ERA5 one hour exceeds the configured field limit.")
    max_combinations = max_fields // fields_per_hour
    planned: list[tuple[list[datetime], str, dict[str, Any], int]] = []

    for _, active_days in sorted(month_days.items()):
        days = sorted(active_days)
        hours = sorted({hour for day_hours in active_days.values() for hour in day_hours})
        # For the production 37 levels and 2 variables, one 31-day month at
        # 24h/day is 55,056 fields: a single request even in the worst case.
        for hour_start in range(0, len(hours), max_combinations):
            hours_chunk = hours[hour_start:hour_start + max_combinations]
            days_per_request = max(1, max_combinations // len(hours_chunk))
            for day_start in range(0, len(days), days_per_request):
                chosen_days = days[day_start:day_start + days_per_request]
                request_hours = [
                    datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
                    for day in chosen_days for hour in hours_chunk
                ]
                dataset, request = build_era5_request_for_hours(
                    request_hours[:len(hours_chunk)], latitude, longitude, cfg,
                )
                request["day"] = [day.strftime("%d") for day in chosen_days]
                request["data_format"] = "grib"
                request["download_format"] = "unarchived"
                fields = len(request_hours) * fields_per_hour
                if fields > max_fields:
                    raise AssertionError("ERA5 monthly GRIB planner exceeded field limit.")
                planned.append((request_hours, dataset, request, fields))

    return sorted(planned, key=lambda item: item[0][0])


def _open_era5_grib(path: Path) -> xr.Dataset:
    """Read native ECMWF GRIB without writing cfgrib index sidecars."""
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def _era5_grib_hours(ds: xr.Dataset) -> tuple[np.ndarray, str | None]:
    """Return UTC analysis hours and the dimension used to select a record."""
    coordinate = _era5_time_coordinate(ds)
    raw = ds[coordinate]
    if raw.ndim > 1:
        raise ValueError("ERA5 GRIB has an unsupported multidimensional time coordinate.")
    times = np.asarray(raw.values).astype("datetime64[ns]").reshape(-1)
    if times.size == 0 or np.isnat(times).any():
        raise ValueError("ERA5 GRIB has missing or invalid analysis times.")
    if raw.ndim == 0:
        return times, None
    return times, raw.dims[0]


def _era5_grib_hour_to_netcdf(ds: xr.Dataset, index: int, axis: str | None) -> xr.Dataset:
    """Normalize one GRIB analysis time into the established hourly NetCDF schema."""
    selected = ds.isel({axis: index}) if axis is not None else ds
    if "isobaricInhPa" in selected.dims:
        selected = selected.rename({"isobaricInhPa": "pressure_level"})
    elif "level" in selected.dims:
        selected = selected.rename({"level": "pressure_level"})
    if not {"t", "z"}.issubset(selected.data_vars):
        raise ValueError("GRIB ERA5 payload is missing temperature/geopotential fields.")
    if "pressure_level" not in selected.dims:
        raise ValueError("GRIB ERA5 payload is missing isobaric pressure levels.")
    # Drop scalar forecast reference times, valid_time and step once a single
    # valid analysis time has been selected; preserve spatial grid and levels.
    result = selected[["t", "z"]].reset_coords(drop=True)
    return result


def _pending_era5_requests(cache_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Conservatively identify unresolved CDS submissions across old planners."""
    directory = cache_root / "requests"
    if not directory.is_dir():
        return []
    pending = []
    for ledger in sorted(directory.glob("*.json")):
        record = _read_metadata(ledger)
        if record.get("status") in {"submitted", "uncertain"}:
            pending.append((ledger, record))
    return pending


def _valid_hourly_cache(
    cache_file: Path,
    metadata_file: Path,
    latitude: float,
    longitude: float,
    cfg: Mapping[str, Any],
) -> bool:
    """Reject incomplete or incompatible hourly cache files before reuse."""
    if not cache_file.is_file() or cache_file.stat().st_size == 0:
        return False
    metadata = _read_metadata(metadata_file)
    if metadata and (
        metadata.get("dataset", cfg["dataset"]) != cfg["dataset"]
        or float(metadata.get("requested_latitude", latitude)) != latitude
        or float(metadata.get("requested_longitude", longitude)) != longitude
        or (
            "request_grid_deg" in metadata
            and float(metadata["request_grid_deg"]) != cfg["grid_deg"]
        )
        or (
            "request_area_half_width_deg" in metadata
            and float(metadata["request_area_half_width_deg"])
            != cfg["area_half_width_deg"]
        )
    ):
        return False
    try:
        with xr.open_dataset(cache_file) as dataset:
            level_axis = _coordinate_name(dataset, ("pressure_level", "level", "isobaricInhPa"))
            levels = np.asarray(dataset[level_axis].values, dtype=np.float64)
            expected = np.asarray(cfg["pressure_levels_hpa"], dtype=np.float64)
            if levels.shape != expected.shape or not np.array_equal(
                np.sort(levels), np.sort(expected)
            ):
                return False
            profile = era5_profile_from_dataset(dataset, latitude, longitude)
            return not profile.empty
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def _write_request_metadata(path: Path, payload: dict[str, Any]) -> None:
    """Atomic, reconstructible request metadata; never contains CDS credentials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".part.json")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def prefetch_era5_pressure_level_profiles(
    measurement_times_utc: list[datetime | pd.Timestamp],
    latitude: float,
    longitude: float,
    logger: logging.Logger,
    *,
    settings: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> set[datetime]:
    """Prefetch via month-scoped GRIB downloads; cache only needed NetCDF hours.

    Only validated hourly cache files count as available. The request ledger
    under .cache/era5/requests is reconstructible; it does not replace science
    provenance or the authoritative NetCDF files.
    """
    cfg = era5_config(settings)
    analysis_hours = sorted({
        nearest_era5_analysis_hour(value) for value in measurement_times_utc
    })
    if not analysis_hours:
        return set()

    available: set[datetime] = set()
    missing = []
    for stamp in analysis_hours:
        cache_file, metadata_file = _cache_paths(
            stamp, latitude, longitude, cfg, root_dir,
        )
        if _valid_hourly_cache(cache_file, metadata_file, latitude, longitude, cfg):
            available.add(stamp)
        else:
            missing.append(stamp)

    if not missing:
        logger.debug("ERA5 prefetch | all %d hourly caches valid", len(analysis_hours))
        return available
    try:
        import cdsapi  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("ERA5 unavailable | cdsapi is not installed")
        return available

    cache_root = resolve_project_path(cfg["cache_dir"], root_dir=root_dir)
    unresolved = _pending_era5_requests(cache_root)
    if unresolved:
        first_ledger, first_record = unresolved[0]
        request_id = first_record.get("request_id", first_ledger.stem)
        raise RuntimeError(
            f"ERA5 prefetch blocked: {len(unresolved)} previously submitted CDS "
            f"request(s) have uncertain remote state; first={request_id} "
            f"ledger={first_ledger}. Inspect/reconcile the remote CDS job before "
            "resubmitting or starting another ERA5 batch. No new request sent."
        )

    requests = plan_era5_requests(missing, latitude, longitude, cfg)
    useful_fields = len(missing) * 2 * len(cfg["pressure_levels_hpa"])
    requested_fields = sum(item[3] for item in requests)
    logger.info(
        "ERA5 monthly GRIB prefetch | %d missing hours | %d requests | "
        "%d requested fields (%d useful, %d extra)",
        len(missing), len(requests), requested_fields,
        useful_fields, requested_fields - useful_fields,
    )
    client = cdsapi.Client(url=CDS_API_URL, quiet=True)
    for hours, dataset, request, fields in requests:
        digest = sha256(
            json.dumps([dataset, request], sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        ledger = cache_root / "requests" / f"{digest}.json"
        previous = _read_metadata(ledger)
        # Monthly day x hour selection may contain intentional extra hours,
        # but only hours required by the currently selected L0 are retained.
        needed = [hour for hour in hours if hour in missing and hour not in available]
        if not needed:
            continue
        # If a process died while waiting for CDS, its server-side job may
        # still exist. An unknown result cannot authorize blind resubmission.
        if previous.get("status") in {"submitted", "uncertain"}:
            logger.warning(
                "ERA5 request %s may still be queued; not duplicating submission",
                digest,
            )
            break
        request_info = {
            "schema_version": 1,
            "dataset": dataset,
            "request": request,
            "request_id": digest,
            "expected_hours_utc": [hour.isoformat() for hour in needed],
            "requested_hours_utc": [hour.isoformat() for hour in hours],
            "fields": fields,
            "download_format": "grib",
            "status": "submitted",
        }
        first_cache, _ = _cache_paths(hours[0], latitude, longitude, cfg, root_dir)
        temp_batch = first_cache.parent / f"era5_{digest}.part.grib"
        _write_request_metadata(ledger, request_info)
        try:
            logger.info(
                "ERA5 CDS monthly GRIB request %s | %d days | %d hours | %d fields",
                digest, len(request["day"]), len(hours), fields,
            )
            client.retrieve(dataset, request, str(temp_batch))
            with temp_batch.open("rb") as stream:
                if stream.read(4) != b"GRIB":
                    raise ValueError("CDS response is not an unarchived GRIB file.")
            with _open_era5_grib(temp_batch) as batch:
                source_times, time_axis = _era5_grib_hours(batch)
                for stamp in needed:
                    target = np.datetime64(stamp.replace(tzinfo=None), "ns")
                    matches = np.flatnonzero(source_times == target)
                    if len(matches) != 1:
                        logger.warning("ERA5 request %s missing hour %s", digest, stamp)
                        continue
                    selected = _era5_grib_hour_to_netcdf(
                        batch, int(matches[0]), time_axis,
                    )
                    cache_file, metadata_file = _cache_paths(
                        stamp, latitude, longitude, cfg, root_dir,
                    )
                    pending = cache_file.with_suffix(".part.nc")
                    try:
                        selected.to_netcdf(pending)
                        # Refuse malformed, incomplete and wrong-level GRIB.
                        if not _valid_hourly_cache(
                            pending, pending.with_suffix(".json"),
                            latitude, longitude, cfg,
                        ):
                            logger.warning("ERA5 request %s invalid hour %s", digest, stamp)
                            continue
                        pending.replace(cache_file)
                        metadata = _metadata_for_profile(
                            stamp, stamp, latitude, longitude, dataset, cache_file,
                        )
                        metadata["request_id"] = digest
                        metadata["download_format"] = "grib"
                        metadata["request_grid_deg"] = cfg["grid_deg"]
                        metadata["request_area_half_width_deg"] = cfg["area_half_width_deg"]
                        metadata["request_pressure_levels_hpa"] = cfg["pressure_levels_hpa"]
                        _write_request_metadata(metadata_file, metadata)
                        available.add(stamp)
                    finally:
                        pending.unlink(missing_ok=True)
            request_info["status"] = (
                "complete" if all(hour in available for hour in needed)
                else "partial"
            )
            request_info["cached_hours_utc"] = [
                stamp.isoformat() for stamp in needed if stamp in available
            ]
            _write_request_metadata(ledger, request_info)
            logger.info(
                "ERA5 GRIB request %s cached | %d/%d required hours",
                digest, len(request_info["cached_hours_utc"]), len(needed),
            )
        except Exception as exc:
            request_info["error"] = _compact_exception_message(exc)
            # Explicitly rejected capacity/slot requests can be attempted
            # during a later run. Other failures after submission have an
            # unknown remote outcome and must never be blindly resubmitted.
            overload = request_info["error"].lower()
            request_info["status"] = (
                "backpressure" if any(label in overload for label in (
                    "available slots", "too many requests", "429 ", "rate limit",
                )) else "uncertain"
            )
            _write_request_metadata(ledger, request_info)
            logger.warning(
                "ERA5 prefetch stopped at request %s | %s",
                digest, request_info["error"],
            )
            logger.debug("ERA5 prefetch exception", exc_info=True)
            break  # Preserve already validated hours; never saturate CDS queue.
        finally:
            temp_batch.unlink(missing_ok=True)
    # The planner may have downloaded extra valid hours to reduce CDS jobs;
    # callers ask only for the originally requested set.
    return available.intersection(analysis_hours)


def fetch_era5_pressure_level_profile(
    measurement_dt_utc: datetime | pd.Timestamp,
    latitude: float,
    longitude: float,
    logger: logging.Logger,
    *,
    settings: Mapping[str, Any],
    root_dir: str | Path | None = None,
    allow_network: bool = True,
) -> Optional[pd.DataFrame]:
    """Fetch/cache one ERA5 pressure-level profile using explicit settings."""
    cfg = era5_config(settings)
    measurement_dt = _as_utc_datetime(measurement_dt_utc)
    analysis_dt = nearest_era5_analysis_hour(measurement_dt)
    cache_file, metadata_file = _cache_paths(
        measurement_dt, latitude, longitude, cfg, root_dir
    )

    if _valid_hourly_cache(
        cache_file, metadata_file, latitude, longitude, cfg,
    ):
        try:
            with xr.open_dataset(cache_file) as ds:
                ds.load()
                frame = era5_profile_from_dataset(ds, latitude, longitude)
            frame.attrs.update(_read_metadata(metadata_file))
            if not frame.attrs:
                frame.attrs.update(
                    _metadata_for_profile(
                        measurement_dt, analysis_dt, latitude, longitude,
                        str(cfg["dataset"]), cache_file,
                    )
                )
            logger.debug("ERA5 cache hit | %s", cache_file.name)
            return frame
        except Exception as exc:
            logger.warning("ERA5 cache unreadable | %s", _compact_exception_message(exc))
            logger.debug("ERA5 cache failure", exc_info=True)

    if not allow_network:
        return None

    try:
        import cdsapi  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("ERA5 unavailable | cdsapi is not installed")
        logger.debug("cdsapi is a core MILGRAU dependency; reinstall the package and configure CDS credentials.")
        return None

    dataset, request = build_era5_request(analysis_dt, latitude, longitude, cfg)
    temporary_file = cache_file.with_suffix(".part.nc")
    try:
        logger.debug(
            "ERA5 request | %sZ | lat=%.4f | lon=%.4f | endpoint=%s",
            analysis_dt.strftime("%Y-%m-%d %H:%M"),
            latitude,
            longitude,
            CDS_API_URL,
        )
        # ERA5 belongs to the Climate Data Store. Pinning the service URL prevents
        # a machine-wide ADS endpoint from silently redirecting this dataset to
        # the wrong Copernicus service while still reading the user's secret token
        # from CDSAPI_KEY or ~/.cdsapirc.
        client = cdsapi.Client(url=CDS_API_URL, quiet=True)
        client.retrieve(dataset, request, str(temporary_file))
        with xr.open_dataset(temporary_file) as ds:
            ds.load()
            frame = era5_profile_from_dataset(ds, latitude, longitude)
        temporary_file.replace(cache_file)
        metadata = _metadata_for_profile(
            measurement_dt, analysis_dt, latitude, longitude, dataset, cache_file
        )
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        frame.attrs.update(metadata)
        logger.debug("ERA5 cached | %s", cache_file.name)
        return frame
    except Exception as exc:
        logger.warning("ERA5 unavailable | %s", _compact_exception_message(exc))
        logger.debug("ERA5 retrieval failure", exc_info=True)
        try:
            if temporary_file.exists():
                temporary_file.unlink()
        except OSError:
            pass
        return None
