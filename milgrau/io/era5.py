"""ERA5 upper-air retrieval and caching for MILGRAU.

Network access belongs in the IO layer. This module retrieves the minimum ERA5
pressure-level fields needed by the molecular atmosphere (temperature and
geopotential), standardizes them to a simple vertical profile, and records
provenance alongside the cached NetCDF file.
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

ERA5_PRESSURE_LEVELS_HPA = (
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550,
    500, 450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100, 70, 50, 30, 20,
    10, 7, 5, 3, 2, 1,
)
ERA5_DATASET = "reanalysis-era5-pressure-levels"
ERA5_DOI = "10.24381/cds.bd0915c6"
_STANDARD_GRAVITY_M_S2 = 9.80665
_EARTH_RADIUS_M = 6_356_766.0


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


def era5_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return normalized ERA5 IO settings with conservative defaults."""
    section = config.get("era5", {}) if isinstance(config, Mapping) else {}
    if not isinstance(section, Mapping):
        section = {}
    levels = section.get("pressure_levels_hpa", ERA5_PRESSURE_LEVELS_HPA)
    normalized_levels: list[int] = []
    for level in levels:
        try:
            value = int(level)
        except (TypeError, ValueError):
            continue
        if value > 0 and value not in normalized_levels:
            normalized_levels.append(value)
    if not normalized_levels:
        normalized_levels = list(ERA5_PRESSURE_LEVELS_HPA)
    return {
        "enabled": bool(section.get("enabled", False)),
        "cache_dir": str(section.get("cache_dir", ".cache/milgrau/era5")),
        "dataset": str(section.get("dataset", ERA5_DATASET)),
        "pressure_levels_hpa": normalized_levels,
        "grid_deg": float(section.get("grid_deg", 0.25)),
        "area_half_width_deg": float(section.get("area_half_width_deg", 0.25)),
    }


def build_era5_request(
    analysis_dt_utc: datetime,
    latitude: float,
    longitude: float,
    config: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build the CDS API request for one small ERA5 pressure-level column."""
    cfg = era5_config(config)
    half_width = max(float(cfg["area_half_width_deg"]), 0.0)
    grid = max(float(cfg["grid_deg"]), 0.01)
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
    config: Mapping[str, Any] | None,
    root_dir: str | Path | None,
) -> tuple[Path, Path]:
    cfg = era5_config(config)
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
    """Select the nearest horizontal ERA5 grid point and collapse singleton axes."""
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
    """Select the first sample along any remaining non-pressure dimensions."""
    result = data
    for dim in tuple(result.dims):
        if dim != pressure_coord:
            result = result.isel({dim: 0})
    if result.dims != (pressure_coord,):
        result = result.transpose(pressure_coord)
    return result


def _geopotential_height_to_geometric_altitude(height_m: np.ndarray) -> np.ndarray:
    """Convert geopotential height to geometric altitude above mean sea level."""
    height = np.asarray(height_m, dtype=np.float64)
    denominator = _EARTH_RADIUS_M - height
    return np.divide(
        _EARTH_RADIUS_M * height,
        denominator,
        out=np.full_like(height, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )


def era5_profile_from_dataset(
    ds: xr.Dataset,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
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
    return (
        frame.drop_duplicates(subset=["height"], keep="first")
        .sort_values("height")
        .reset_index(drop=True)
    )


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
    config: Mapping[str, Any] | None = None,
    root_dir: str | Path | None = None,
) -> Optional[pd.DataFrame]:
    """Fetch/cached ERA5 pressure-level temperature and geopotential profile.

    The function returns ``None`` whenever ERA5 is disabled, credentials/client
    are unavailable, or retrieval/parsing fails. That behavior is intentional:
    the caller can then continue to the deterministic USSA76 fallback.
    """
    cfg = era5_config(config)
    if not cfg["enabled"]:
        return None

    measurement_dt = _as_utc_datetime(measurement_dt_utc)
    analysis_dt = nearest_era5_analysis_hour(measurement_dt)
    cache_file, metadata_file = _cache_paths(
        measurement_dt, latitude, longitude, config, root_dir
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
            "  -> [ERA5] ERA5 fallback is enabled but cdsapi is not installed. "
            "Install MILGRAU with the 'era5' extra and configure ~/.cdsapirc."
        )
        return None

    dataset, request = build_era5_request(analysis_dt, latitude, longitude, config)
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
        logger.warning(f"  -> [ERA5] Retrieval unavailable; continuing to standard atmosphere: {exc}")
        try:
            if temporary_file.exists():
                temporary_file.unlink()
        except OSError:
            pass
        return None
