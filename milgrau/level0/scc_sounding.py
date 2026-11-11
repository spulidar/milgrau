"""Optional SCC sounding ancillary from observed Wyoming or NOAA IGRA2 caches.

No network IO in LIBIDS. No ERA5 or USSA76 substitution for observed sounding.
The radiosonde launch site differs from SPU; heights are above the launch site.
SCC NetCDF Sounding Data specification v3.6, table 2.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import netCDF4 as nc
import numpy as np
import pandas as pd

from milgrau.io.paths import resolve_project_path
from milgrau.io.radiosonde import select_radiosonde_target_datetime
from milgrau.provenance import file_sha256


def scc_sounding_name(identity: str) -> str:
    from milgrau.level0.scc_staging import scc_raw_filename

    scc_raw_filename(identity)  # validate exact station measurement identity
    # SCC Quick Upload requires rs_YYYYMMDDcccNN.nc (two-digit suffix),
    # not the full four-digit HHMM suffix of Measurement_ID.
    return f"rs_{identity[:13]}.nc"


def validate_sounding(path: Path, *, identity: str) -> dict[str, Any]:
    """Verify mandatory SCC sounding variables, units, metadata, and shape."""
    if path.name != scc_sounding_name(identity) or path.is_symlink():
        raise ValueError("SCC sounding file has invalid identity or symlink.")
    with nc.Dataset(path, "r") as ds:
        if "points" not in ds.dimensions or len(ds.dimensions["points"]) < 4:
            raise ValueError("SCC sounding requires at least 4 vertical points.")
        arrays = {}
        for name in ("Altitude", "Temperature", "Pressure"):
            variable = ds.variables[name]
            if variable.dimensions != ("points",) or variable.dtype != np.dtype("float64"):
                raise ValueError(f"SCC sounding {name} requires double(points).")
            arrays[name] = np.asarray(variable[:], dtype=np.float64)
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"SCC sounding {name} has missing values.")
        if (
            arrays["Altitude"][0] < 0
            or not (np.diff(arrays["Altitude"]) > 0).all()
            or not (np.diff(arrays["Pressure"]) < 0).all()
            or (arrays["Pressure"] <= 0).any()
            or (arrays["Temperature"] < -130).any()
            or (arrays["Temperature"] > 60).any()
        ):
            raise ValueError("SCC sounding altitude/pressure/temperature are unphysical.")
        for name in (
            "Latitude_degrees_north", "Longitude_degrees_east",
            "Altitude_meter_asl", "Sounding_Start_Date", "Sounding_Start_Time_UT",
        ):
            if name not in ds.ncattrs():
                raise ValueError(f"Missing SCC sounding global attribute {name}.")
        if not str(ds.getncattr("Sounding_Start_Date")).isdigit():
            raise ValueError("SCC sounding UTC date is invalid.")
        if len(str(ds.getncattr("Sounding_Start_Time_UT"))) != 6:
            raise ValueError("SCC sounding UTC start time must be HHMMSS.")
        if "RelativeHumidity" in ds.variables:
            rh = ds.variables["RelativeHumidity"]
            if rh.dimensions != ("points",) or rh.dtype != np.dtype("float64"):
                raise ValueError("SCC relative humidity dimensions/type invalid.")
            arr = np.asarray(rh[:], dtype=np.float64)
            if not np.isfinite(arr).all() or (arr < 0).any() or (arr > 100).any():
                raise ValueError("SCC sounding relative humidity out of 0–100%.")
    return {"points": len(arrays["Altitude"]), "sha256": file_sha256(path)}


def stage_cached_sounding(
    staging: Path, identity: str, measurement_start_utc: Any,
    config: Mapping[str, Any],
    *, observed_target_utc: str | None = None,
    observed_origin: str | None = None,
    observed_cache_csv_name: str | None = None,
    validated_max_time_delta_hours: float | None = None,
    measurement_stop_utc: Any | None = None,
) -> Path | None:
    """Stage an SCC rs_YYYYMMDDcccNN.nc from existing observation cache, if opted in.

    Missing/invalid observations are never replaced with modelled ERA5/USSA76 profiles.
    Called only while a new SCC package is being built in an atomic staging
    directory. Never mutate existing journalled/uncertain pending packages.
    """
    policy = (
        config.get("_scc_catalog", {}).get("scc_policy", {})
        .get("sounding", {})
    )
    if not policy.get("enabled", False):
        return None
    if policy.get("source", "cached_observation") not in {"cached_observation", "cached_wyoming"}:
        raise ValueError("SCC sounding requires an explicitly configured observation cache.")
    missing = policy.get("missing_policy", "skip")
    if missing not in {"skip", "fail"}:
        raise ValueError("Invalid SCC sounding missing_policy.")
    station = config["_station_catalog"]["station"]["radiosonde"]
    required_meta = ("station_id", "station_name", "latitude", "longitude", "station_altitude_m")
    if any(key not in station for key in required_meta):
        raise ValueError(
            "SCC sounding requires observed radiosonde launch coordinates/altitude in station.yaml."
        )
    settings = config["level1"]["atmosphere"]["radiosonde"]
    start = pd.Timestamp(measurement_start_utc)
    # LIPANCORA selects its radiosonde QA for the scientific session's
    # midpoint, rather than its first laser profile. Use that same reference
    # for SCC when the full measurement interval is available.
    reference_time = start
    if measurement_stop_utc is not None:
        stop = pd.Timestamp(measurement_stop_utc)
        if stop < start:
            raise ValueError("SCC sounding measurement interval is inverted.")
        reference_time = start + (stop - start) / 2
    tolerance = (
        float(validated_max_time_delta_hours)
        if validated_max_time_delta_hours is not None
        else float(settings["max_time_delta_hours"])
    )
    if not math.isfinite(tolerance) or not 0 < tolerance <= 24:
        raise ValueError("SCC observed sounding tolerance must be finite and in (0, 24] hours.")
    if observed_origin is not None and observed_origin not in {"wyoming_siphon", "igra2"}:
        raise ValueError("SCC sounding observation origin is not a verified provider.")
    if observed_target_utc is not None:
        # Trust the actual LIPANCORA-validated observed launch, *not* a
        # recalculated nominal nearest slot. At 19Z a valid 12Z launch may
        # be the closest real observation while 00Z has no measurement.
        target = pd.Timestamp(observed_target_utc)
        if target.tzinfo is None:
            raise ValueError("LIPANCORA sounding target must be timezone-aware UTC.")
        target = target.tz_convert("UTC")
        if (
            target.minute != 0 or target.second != 0
            or target.hour not in settings["synoptic_hours_utc"]
        ):
            raise ValueError("LIPANCORA observed target is not a configured nominal UTC sounding.")
        if abs((target - reference_time.tz_convert("UTC")).total_seconds()) > tolerance * 3600:
            target = None
    else:
        target = select_radiosonde_target_datetime(
            reference_time,
            synoptic_hours_utc=settings["synoptic_hours_utc"],
            selection=settings["selection"],
            max_time_delta_hours=tolerance,
        )
    if target is None:
        if missing == "fail":
            raise ValueError("No matching LIPANCORA observed sounding within configured SCC time tolerance.")
        return None
    root = resolve_project_path(settings["cache_dir"])
    basename = f"radiosonde_{station['station_id']}_{target.strftime('%Y%m%d_%H')}Z.csv"
    folder = root / target.strftime("%Y") / target.strftime("%m")
    ordinary = folder / basename
    wyoming = ordinary.with_name(ordinary.stem + "_wyoming.csv")
    if observed_cache_csv_name:
        if (
            Path(observed_cache_csv_name).name != observed_cache_csv_name
            or observed_cache_csv_name not in {ordinary.name, wyoming.name}
        ):
            raise ValueError("SCC sounding selected cache filename does not match the verified launch.")
        candidates = [folder / observed_cache_csv_name]
    elif observed_origin == "wyoming_siphon":
        candidates = [wyoming, ordinary]
    else:
        # Historical/unspecified L1 staging keeps the legacy hourly CSV
        # unless an explicitly selected Wyoming suffix was recorded.
        candidates = [ordinary]
    cache = next((item for item in candidates if item.is_file()), candidates[0])
    meta_file = cache.with_suffix(".json")
    if (
        not cache.is_file() or not meta_file.is_file()
        or cache.is_symlink() or meta_file.is_symlink()
    ):
        if missing == "fail":
            raise ValueError(f"Observed sounding/cache metadata missing: {cache}.")
        return None
    provenance = json.loads(meta_file.read_text(encoding="utf-8"))
    nominal = pd.Timestamp(provenance["target_datetime_utc"])
    if nominal.tzinfo is None or nominal.tz_convert("UTC") != target:
        raise ValueError("Cached sounding provenance does not match observed UTC launch.")
    if (
        str(provenance.get("station_id")) != str(station["station_id"])
        or provenance.get("source_type") != "radiosonde"
    ):
        raise ValueError("Cached sounding station/source provenance does not match selection.")
    if provenance.get("historical_archive") or "igra" in str(provenance.get("source", "")).lower():
        actual_origin = "igra2"
        actual_source = "NOAA NCEI IGRA2 historical sounding archive"
    elif "wyoming" in str(provenance.get("source", "")).lower():
        actual_origin = "wyoming_siphon"
        actual_source = "University of Wyoming Upper Air via Siphon"
    else:
        # Legacy valid CSVs did not always record their provider. Retain the
        # uncertainty, rather than claiming Wyoming without evidence.
        actual_origin = "unattributed_legacy"
        actual_source = "Observed sounding from legacy cache, original provider unknown"
    if observed_origin is not None and actual_origin != observed_origin:
        raise ValueError("SCC sounding provenance contradicts LIPANCORA observation origin.")
    df = pd.read_csv(cache)
    columns = {"height", "temperature", "pressure"}
    if not columns.issubset(df):
        raise ValueError(f"Wyoming cache lacks columns {sorted(columns - set(df))}.")
    data = df.sort_values("height").drop_duplicates("height")
    data = data.dropna(subset=sorted(columns))
    data = data[data["height"] >= float(station["station_altitude_m"])]
    height = data["height"].to_numpy(dtype=np.float64) - float(station["station_altitude_m"])
    temperature = data["temperature"].to_numpy(dtype=np.float64)
    pressure = data["pressure"].to_numpy(dtype=np.float64)
    if (
        len(height) < 4 or not np.isfinite(height).all()
        or not np.isfinite(temperature).all()
        or not np.isfinite(pressure).all()
        or (np.diff(height) <= 0).any()
        or (np.diff(pressure) >= 0).any()
        or (pressure <= 0).any()
        or not all(math.isfinite(float(station[key])) for key in (
            "latitude", "longitude", "station_altitude_m"
        ))
    ):
        raise ValueError("Cached sounding fails physical/vertical data QA.")
    output = staging / scc_sounding_name(identity)
    with nc.Dataset(output, "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("points", len(height))
        for name, values, units in (
            ("Altitude", height, "m"),
            ("Temperature", temperature, "degree_Celsius"),
            ("Pressure", pressure, "hPa"),
        ):
            variable = ds.createVariable(name, "f8", ("points",))
            variable[:] = values
            variable.units = units
        if "relative_humidity" in data:
            rh = data["relative_humidity"].to_numpy(dtype=np.float64)
            if np.isfinite(rh).all() and (rh >= 0).all() and (rh <= 100).all():
                ds.createVariable("RelativeHumidity", "f8", ("points",))[:] = rh
        ds.setncatts({
            "Latitude_degrees_north": float(station["latitude"]),
            "Longitude_degrees_east": float(station["longitude"]),
            "Altitude_meter_asl": float(station["station_altitude_m"]),
            "Sounding_Station_Name": str(station["station_name"]),
            "WMO_Station_Number": str(station["station_id"]),
            "Sounding_Start_Date": target.strftime("%Y%m%d"),
            "Sounding_Start_Time_UT": target.strftime("%H%M%S"),
            "Source_Observation": actual_source,
            "Source_Observation_Provider": actual_origin,
            "Source_CSV_SHA256": file_sha256(cache),
            "Selection_Time_Delta_hours": (
                abs((target - reference_time.tz_convert("UTC")).total_seconds()) / 3600
            ),
        })
    validate_sounding(output, identity=identity)
    return output
