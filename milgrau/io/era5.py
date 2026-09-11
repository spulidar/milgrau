"""ERA5 upper-air retrieval and caching for MILGRAU.

All scientific/processing settings are supplied explicitly by the Level 1
atmosphere policy. This module contains only IO behavior, physical constants and
file-format handling; it does not invent semantic configuration defaults.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.io.paths import resolve_project_path

ERA5_DOI = "10.24381/cds.bd0915c6"
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


def build_era5_request(
    analysis_dt_utc: datetime,
    latitude: float,
    longitude: float,
    settings: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Build the CDS API request for one small ERA5 pressure-level column."""
    cfg = era5_config(settings)
    half_width = float(cfg["area_half_width_deg"])
    grid = float(cfg["grid_deg"])
    north = min(float(latitude) + half_width, 90.0)
    south = max(float(latitude) - half_width, -90.0)
    west = max(float(longitude) - half_width, -180.0)
    east = min(float(longitude) + half_width, 180.0)
    request = {
        "product_type": ["reanalysis"],
        "variable": ["temperature", "geopotential"],
        "year": [analysis_dt_utc.strftime("%Y")],
        "month": [analysis_dt_utc.strftime("%m")],
        "day": [analysis_dt_utc.strftime("%d")],
        "time": [analysis_dt_utc.strftime("%H:00")],
        "pressure_level": [str(level) for level in cfg["pressure_levels_hpa"]],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [north, west, south, east],
        "grid": [grid, grid],
    }
    return str(cfg["dataset"]), request


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


def fetch_era5_pressure_level_profile(
    measurement_dt_utc: datetime | pd.Timestamp,
    latitude: float,
    longitude: float,
    logger: logging.Logger,
    *,
    settings: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> Optional[pd.DataFrame]:
    """Fetch/cache one ERA5 pressure-level profile using explicit settings."""
    cfg = era5_config(settings)
    measurement_dt = _as_utc_datetime(measurement_dt_utc)
    analysis_dt = nearest_era5_analysis_hour(measurement_dt)
    cache_file, metadata_file = _cache_paths(
        measurement_dt, latitude, longitude, cfg, root_dir
    )

    if cache_file.exists():
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
            logger.info(f"  -> [ERA5] Cached pressure-level profile found: {cache_file.name}")
            return frame
        except Exception as exc:
            logger.warning(f"  -> [ERA5] Could not read cached profile {cache_file}: {exc}")

    try:
        import cdsapi  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "  -> [ERA5] ERA5 is present in the configured atmosphere source policy but cdsapi is not installed. "
            "Install MILGRAU with the 'era5' extra and configure ~/.cdsapirc."
        )
        return None

    dataset, request = build_era5_request(analysis_dt, latitude, longitude, cfg)
    temporary_file = cache_file.with_suffix(".part.nc")
    try:
        logger.info(
            f"  -> [ERA5] Fetching {analysis_dt.strftime('%Y-%m-%d %H:%M')}Z "
            f"pressure-level profile near ({latitude:.4f}, {longitude:.4f})..."
        )
        client = cdsapi.Client()
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
        logger.info("  -> [OK] ERA5 pressure-level profile successfully fetched and cached.")
        return frame
    except Exception as exc:
        logger.warning(f"  -> [ERA5] Retrieval unavailable under configured source policy: {exc}")
        try:
            if temporary_file.exists():
                temporary_file.unlink()
        except OSError:
            pass
        return None
