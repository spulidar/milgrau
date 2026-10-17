"""Read-only, bounded views of existing MILGRAU NetCDF scientific products.

No retrieval, smoothing, interpolation or inferred QA thresholds. The Explorer
renders only saved observations and contract-defined diagnostic variables.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import xarray as xr

from milgrau.cli.inspect import _session_summary
from milgrau.io.paths import session_id_parts
from milgrau.web.catalog import Catalog

ATMOSPHERE = {
    "Atmospheric_Temperature_K": "Atmospheric temperature",
    "Atmospheric_Pressure_hPa": "Atmospheric pressure",
}
OPTICAL = {
    "aerosol_backscatter_mean": "Mean aerosol backscatter",
    "aerosol_extinction_mean": "Mean aerosol extinction",
    "temporal_support_fraction": "Temporal support fraction",
}
QA_L1 = {
    "background_estimate": "Background estimate",
    "background_outlier_fraction": "Background outlier fraction",
    "pc_saturation_fraction": "PC saturation fraction",
    "deadtime_clipping_fraction": "Dead-time clipping fraction",
    "bin_shift_invalid_fraction": "Invalid bin-shift fraction",
}
QA_L2 = {
    "retrieval_success_flag": "Retrieval success flag",
    "retrieval_input_valid_flag": "Valid retrieval input",
    "gluing_success_flag": "Gluing success flag",
    "single_channel_fallback_flag": "Single-channel fallback flag",
    "rayleigh_reference_fallback_used_block": "Rayleigh reference fallback",
}


def _short(value: Any, *, width: int = 320) -> str:
    """Safely serialize metadata without expanding large coordinate arrays."""
    if isinstance(value, np.ndarray):
        return f"array(shape={value.shape}, dtype={value.dtype})"
    return str(value).replace("\n", " ")[:width]


def _values(values: Any) -> list[float | None]:
    out: list[float | None] = []
    for value in np.asarray(values, dtype=np.float64).reshape(-1):
        out.append(float(value) if np.isfinite(value) else None)
    return out


def _timestamps(values: Any) -> list[str]:
    array = np.asarray(values).astype("datetime64[ns]")
    return [
        np.datetime_as_string(value, unit="s") + "Z"
        if not np.isnat(value) else ""
        for value in array
    ]


def _altitudes(ds: xr.Dataset, indices: slice) -> list[float]:
    z = np.asarray(ds["altitude"].isel(altitude=indices).values, dtype=np.float64)
    units = str(ds["altitude"].attrs.get("units", "m")).lower()
    if units in {"km", "kilometer", "kilometers"}:
        return _values(z)
    return _values(z / 1000.0)


def _canonical(catalog: Catalog, session_id: str, level: int) -> Path | None:
    try:
        return catalog.resolve(session_id, f"{session_id}_L{level}.nc", level=level)
    except KeyError:
        return None


def _choices(ds: xr.Dataset, name: str) -> list[str]:
    if name not in ds.coords:
        return []
    return [
        value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
        for value in ds[name].values
    ]


def _summary_unavailable(message: str) -> dict[str, Any]:
    return {"available": False, "reason": message}


def session_overview(catalog: Catalog, session_id: str) -> dict[str, Any]:
    """Use the same summary contract as milgrau-inspect, without reading signals."""
    products = catalog.products(session_id)
    _station, start, end = session_id_parts(session_id)
    zone_name = str(catalog.site.get("timezone") or "UTC")
    try:
        zone = ZoneInfo(zone_name)
    except ZoneInfoNotFoundError:
        zone_name, zone = "UTC", ZoneInfo("UTC")
    levels = [
        {"level": f"L{n}", "available": any(
            p["id"] == f"{session_id}_L{n}.nc" for p in products
        )}
        for n in (0, 1, 2)
    ]
    answer: dict[str, Any] = {
        "available": True,
        "session_id": session_id,
        "station": catalog.site.get("name", catalog.station_id),
        "start_utc": start.strftime("%Y-%m-%d %H:%M UTC"),
        "end_utc": end.strftime("%Y-%m-%d %H:%M UTC"),
        "local_start": start.astimezone(zone).strftime("%Y-%m-%d %H:%M"),
        "local_end": end.astimezone(zone).strftime("%Y-%m-%d %H:%M"),
        "timezone": zone_name,
        "duration_minutes": round((end - start).total_seconds() / 60, 2),
        "levels": levels,
        "channels": [],
        "regime_sequence": None,
        "segments": None,
        "figures_count": None,
    }
    for n in (1, 0, 2):
        path = _canonical(catalog, session_id, n)
        if path is None:
            continue
        try:
            with xr.open_dataset(path, cache=False) as ds:
                result = _session_summary(path, ds, timezone_name=zone_name)
                if result:
                    answer["regime_sequence"] = (
                        result["regime_sequence"] if result["regime_sequence"] != "--" else None
                    )
                    answer["segments"] = (
                        result["segments"] if result["segments"] != "--" else None
                    )
                    answer["figures_count"] = len(result["figures"])
                if "channel" in ds.coords:
                    answer["channels"] = _choices(ds, "channel")
                elif "channel_string" in ds:
                    answer["channels"] = [str(x) for x in ds["channel_string"].values]
            break
        except (OSError, ValueError, KeyError):
            continue
    return answer


def metadata_view(catalog: Catalog, session_id: str, level: int) -> dict[str, Any]:
    path = _canonical(catalog, session_id, level)
    if path is None:
        return _summary_unavailable(f"Canonical L{level} NetCDF not found for this session.")
    with xr.open_dataset(path, cache=False) as ds:
        coordinates: list[dict[str, Any]] = []
        for name, data in ds.coords.items():
            entry: dict[str, Any] = {
                "name": name, "dims": list(data.dims), "shape": list(data.shape),
                "dtype": str(data.dtype), "units": str(data.attrs.get("units", "")),
            }
            if data.ndim == 1 and data.size > 0:
                sampled = data.isel({data.dims[0]: [0, int(data.size) - 1]}).values
                if np.issubdtype(np.asarray(sampled).dtype, np.datetime64):
                    entry["first_last"] = _timestamps(sampled)
                else:
                    entry["first_last"] = [_short(value, width=80) for value in sampled]
            coordinates.append(entry)
        variables = [
            {"name": name, "dims": list(data.dims), "shape": list(data.shape),
             "dtype": str(data.dtype), "units": str(data.attrs.get("units", "")),
             "long_name": _short(data.attrs.get("long_name", ""), width=160)}
            for name, data in ds.data_vars.items()
        ]
        return {
            "available": True, "level": f"L{level}", "file": path.name,
            "dimensions": [{"name": k, "size": int(v)} for k, v in ds.sizes.items()],
            "coordinates": coordinates, "variables": variables,
            "attributes": [{"name": k, "value": _short(v)} for k, v in ds.attrs.items()],
        }


def profile_options(catalog: Catalog, session_id: str, kind: str) -> dict[str, Any]:
    if kind not in {"atmosphere", "optical"}:
        raise ValueError("Unknown profile kind")
    level = 1 if kind == "atmosphere" else 2
    path = _canonical(catalog, session_id, level)
    if path is None:
        return _summary_unavailable(f"Canonical L{level} NetCDF not available.")
    contract = ATMOSPHERE if level == 1 else OPTICAL
    expected = ("atmosphere_time", "altitude") if level == 1 else ("wavelength", "altitude")
    with xr.open_dataset(path, cache=False) as ds:
        variables = [
            {"id": name, "label": label}
            for name, label in contract.items()
            if name in ds and ds[name].dims == expected
        ]
        dimension = expected[0]
        return {
            "available": bool(variables),
            "reason": None if variables else "No supported profile variables in this NetCDF.",
            "variables": variables,
            "choices": _choices(ds, dimension) if dimension in ds.coords else [],
            "choice_axis": dimension,
        }


def profile_view(
    catalog: Catalog, session_id: str, kind: str, variable: str, index: int,
) -> dict[str, Any]:
    if kind not in {"atmosphere", "optical"}:
        raise ValueError("Unknown profile kind")
    level = 1 if kind == "atmosphere" else 2
    contract = ATMOSPHERE if level == 1 else OPTICAL
    if variable not in contract:
        raise ValueError("Unknown profile variable")
    path = _canonical(catalog, session_id, level)
    if path is None:
        return _summary_unavailable(f"Canonical L{level} NetCDF not available.")
    axis = "atmosphere_time" if level == 1 else "wavelength"
    with xr.open_dataset(path, cache=False) as ds:
        expected = (axis, "altitude")
        if variable not in ds or ds[variable].dims != expected:
            return _summary_unavailable(f"{variable} not found with expected dimensions.")
        n = int(ds.sizes[axis])
        if index < 0 or index >= n:
            raise ValueError("Profile selection out of range")
        n_alt = ds.sizes["altitude"]
        stride = max(1, math.ceil(n_alt / 650))
        selection = slice(None, None, stride)
        signal = ds[variable].isel({axis: index, "altitude": selection})
        chosen = ds[axis].isel({axis: index}).values
        label = (_timestamps([chosen])[0] if axis == "atmosphere_time" else str(chosen))
        answer = {
            "available": True, "variable": variable, "title": contract[variable],
            "level": f"L{level}", "choice": label, "choice_axis": axis,
            "units": str(signal.attrs.get("units", "") or ""),
            "altitude_km": _altitudes(ds, selection),
            "values": _values(signal.values),
            "display_stride": stride,
        }
        if axis == "atmosphere_time" and "Atmospheric_Source_Type" in ds:
            answer["source"] = _short(
                ds["Atmospheric_Source_Type"].isel(atmosphere_time=index).values
            )
        return answer


def qa_options(catalog: Catalog, session_id: str) -> dict[str, Any]:
    available: list[dict[str, Any]] = []
    for level, variables, axis, channel_dim in (
        (1, QA_L1, "time", "channel"), (2, QA_L2, "block_time", "wavelength"),
    ):
        path = _canonical(catalog, session_id, level)
        if path is None:
            continue
        with xr.open_dataset(path, cache=False) as ds:
            for name, title in variables.items():
                if name in ds and ds[name].dims == (axis, channel_dim):
                    available.append({
                        "variable": name, "label": f"L{level} · {title}",
                        "level": level, "choices": _choices(ds, channel_dim),
                        "choice_axis": channel_dim,
                    })
    return {
        "available": bool(available),
        "reason": None if available else "No supported L1/L2 QA time-series variables were saved.",
        "metrics": available,
    }


def qa_series(catalog: Catalog, session_id: str, level: int, variable: str, index: int) -> dict[str, Any]:
    variables = QA_L1 if level == 1 else QA_L2 if level == 2 else {}
    if variable not in variables:
        raise ValueError("Unsupported QA variable")
    path = _canonical(catalog, session_id, level)
    if path is None:
        return _summary_unavailable(f"Canonical L{level} NetCDF not found.")
    axis, choice_axis = ("time", "channel") if level == 1 else ("block_time", "wavelength")
    with xr.open_dataset(path, cache=False) as ds:
        if variable not in ds or ds[variable].dims != (axis, choice_axis):
            return _summary_unavailable(f"{variable} was not saved with supported dimensions.")
        count = ds.sizes[choice_axis]
        if index < 0 or index >= count:
            raise ValueError("QA channel/wavelength index out of range")
        length = ds.sizes[axis]
        step = max(1, math.ceil(length / 1200))
        series = ds[variable].isel({choice_axis: index, axis: slice(None, None, step)})
        return {
            "available": True, "variable": variable, "title": variables[variable],
            "level": f"L{level}", "channel": _choices(ds, choice_axis)[index],
            "time_utc": _timestamps(ds[axis].isel({axis: slice(None, None, step)}).values),
            "values": _values(series.values),
            "units": str(ds[variable].attrs.get("units", "")),
            "sample_step": step,
        }
