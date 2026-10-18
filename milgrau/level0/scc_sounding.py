"""Optional SCC sounding ancillary from observed, locally cached Wyoming profiles.

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
    measurement_stop_utc: Any | None = None,
) -> Path | None:
    """Stage an SCC rs_YYYYMMDDcccNN.nc from existing observation cache, if opted in.

    Missing/invalid Wyoming observations are not replaced with model profiles.
    Called only while a new SCC package is being built in an atomic staging
    directory. Never mutate existing journalled/uncertain pending packages.
    """
    policy = (
        config.get("_scc_catalog", {}).get("scc_policy", {})
        .get("sounding", {})
    )
    if not policy.get("enabled", False):
        return None
    if policy.get("source", "cached_wyoming") != "cached_wyoming":
        raise ValueError("SCC sounding only supports cached_wyoming.")
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
    target = select_radiosonde_target_datetime(
        reference_time,
        synoptic_hours_utc=settings["synoptic_hours_utc"],
        selection=settings["selection"],
        max_time_delta_hours=float(settings["max_time_delta_hours"]),
    )
    if target is not None and observed_target_utc is not None:
        # Only accept the same observed launch already selected and verified
        # by LIPANCORA. No silently different nearby sounding may be staged.
        observed_time = pd.Timestamp(observed_target_utc)
        if observed_time.tzinfo is None:
            raise ValueError("LIPANCORA sounding target must be timezone-aware UTC.")
        if observed_time.tz_convert("UTC") != pd.Timestamp(target):
            target = None
    if target is None:
        if missing == "fail":
            raise ValueError(
                "No matching LIPANCORA-verified synoptic sounding within "
                "configured SCC time tolerance."
            )
        return None
    root = resolve_project_path(settings["cache_dir"])
    cache = (
        root / target.strftime("%Y") / target.strftime("%m")
        / f"radiosonde_{station['station_id']}_{target.strftime('%Y%m%d_%H')}Z.csv"
    )
    meta_file = cache.with_suffix(".json")
    if not cache.is_file() or not meta_file.is_file() or cache.is_symlink():
        if missing == "fail":
            raise ValueError(f"Observed Wyoming sounding/cache metadata missing: {cache}.")
        return None
    provenance = json.loads(meta_file.read_text(encoding="utf-8"))
    if (
        str(provenance.get("station_id")) != str(station["station_id"])
        or pd.Timestamp(provenance["target_datetime_utc"]).tz_convert("UTC") != pd.Timestamp(target)
        or provenance.get("source_type") != "radiosonde"
    ):
        raise ValueError("Cached sounding provenance does not match selected observation.")
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
            "Source_Observation": "University of Wyoming Upper Air",
            "Source_CSV_SHA256": file_sha256(cache),
            "Selection_Time_Delta_hours": (
                abs((pd.Timestamp(target) - reference_time.tz_convert("UTC")).total_seconds()) / 3600
            ),
        })
    validate_sounding(output, identity=identity)
    return output
