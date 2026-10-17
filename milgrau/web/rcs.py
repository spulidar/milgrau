"""Display-only L1 data access; no correction, inversion or reprocessing."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

RCS_VARIABLES = ("range_corrected_signal", "corrected_signal", "signal_pre_background")


def _channels(ds: xr.Dataset) -> list[str]:
    if "channel" not in ds.coords:
        return []
    result = []
    for value in ds["channel"].values:
        result.append(
            value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
        )
    return result


def options(path: Path) -> dict[str, Any]:
    """Inspect only small coordinates and the saved variable contracts."""
    with xr.open_dataset(path, cache=False) as ds:
        variables = [
            name for name in RCS_VARIABLES
            if name in ds and set(ds[name].dims) == {"time", "channel", "altitude"}
        ]
        return {"channels": _channels(ds), "variables": variables}


def _as_utc(value: np.datetime64) -> str:
    return np.datetime_as_string(value, unit="s") + "Z"


def _reduced_column(
    sample: np.ndarray, starts: np.ndarray, *, statistic: str,
) -> np.ndarray:
    finite = np.isfinite(sample)
    sums = np.add.reduceat(np.where(finite, sample, 0.0).sum(axis=0), starts)
    counts = np.add.reduceat(finite.sum(axis=0), starts)
    if statistic == "mean":
        return np.divide(
            sums, counts, out=np.full(len(starts), np.nan), where=counts > 0
        )
    # Optional explicitly labeled peak-preserving display; never a retrieval.
    maxima = np.maximum.reduceat(
        np.where(finite, sample, -np.inf).max(axis=0), starts
    )
    maxima[counts == 0] = np.nan
    return maxima


def rcs_map(
    path: Path, channel: str, variable: str = "range_corrected_signal",
    statistic: str = "mean", max_time: int = 640, max_altitude: int = 720,
    gap_minutes: float = 10.0,
) -> dict[str, Any]:
    """Return bounded visual cells, retaining real gaps and missing observations.

    Each output cell is explicitly a display aggregate, not a new scientific
    product. Reads at most one small time block of one channel at a time.
    """
    if statistic not in {"mean", "max"}:
        raise ValueError("Estatística de visualização inválida")
    with xr.open_dataset(path, cache=False) as ds:
        allowed = options_from_dataset(ds)
        if variable not in allowed["variables"]:
            raise ValueError("Variável incompatível com produto L1")
        if channel not in allowed["channels"]:
            raise ValueError("Canal ausente deste produto")
        da = ds[variable].sel(channel=channel).transpose("time", "altitude")
        n_time, n_alt = da.shape
        if n_time == 0 or n_alt == 0:
            raise ValueError("Produto não contém amostras de tempo e altitude")
        times = np.asarray(ds["time"].values).astype("datetime64[ns]")
        altitude = np.asarray(ds["altitude"].values, dtype=float)
        if np.isnat(times).any() or not np.isfinite(altitude).all():
            raise ValueError("Coordenadas inválidas no NetCDF")
        seconds = np.diff(times).astype("timedelta64[s]").astype(np.int64)
        if np.any(seconds <= 0):
            raise ValueError("Amostras temporais precisam estar em ordem crescente")
        break_at = np.flatnonzero(seconds > float(gap_minutes) * 60) + 1
        boundaries = [0, *break_at.tolist(), n_time]
        time_step = max(1, math.ceil(n_time / max_time))
        altitude_step = max(1, math.ceil(n_alt / max_altitude))
        altitude_starts = np.arange(0, n_alt, altitude_step, dtype=int)
        altitude_ends = np.r_[altitude_starts[1:], n_alt]
        altitude_plot = [
            float(np.mean(altitude[start:end]) / 1000.0)
            for start, end in zip(altitude_starts, altitude_ends, strict=True)
        ]
        profile_sum = np.zeros(n_alt, dtype=np.float64)
        profile_count = np.zeros(n_alt, dtype=np.int64)
        columns: list[np.ndarray] = []
        labels: list[str] = []
        for begin, end in zip(boundaries, boundaries[1:], strict=True):
            if begin > 0:
                # Two empty columns make long instrument gaps visible in Plotly.
                labels.extend([
                    _as_utc(times[begin - 1] + np.timedelta64(1, "s")),
                    _as_utc(times[begin] - np.timedelta64(1, "s")),
                ])
                columns.extend([
                    np.full(len(altitude_starts), np.nan),
                    np.full(len(altitude_starts), np.nan),
                ])
            for first in range(begin, end, time_step):
                last = min(first + time_step, end)
                sample = np.asarray(
                    da.isel(time=slice(first, last)).values, dtype=np.float64
                )
                valid = np.isfinite(sample)
                profile_sum += np.where(valid, sample, 0.0).sum(axis=0)
                profile_count += valid.sum(axis=0)
                columns.append(
                    _reduced_column(sample, altitude_starts, statistic=statistic)
                )
                labels.append(_as_utc(times[(first + last - 1) // 2]))
        binned_profile = np.divide(
            np.add.reduceat(profile_sum, altitude_starts),
            np.add.reduceat(profile_count, altitude_starts),
            out=np.full(len(altitude_starts), np.nan),
            where=np.add.reduceat(profile_count, altitude_starts) > 0,
        )
        # Convert non-finite values to JSON null, never to fabricated zeros.
        z = np.asarray(columns, dtype=float).T
        z_json = np.where(np.isfinite(z), z, None).tolist()
        profile_json = [
            float(value) if np.isfinite(value) else None for value in binned_profile
        ]
        units = str(da.attrs.get("units") or "a.u.")
        return {
            "session": str(ds.attrs.get("Session_ID", "")),
            "channel": channel, "variable": variable, "units": units,
            "statistic": statistic, "time_utc": labels,
            "altitude_km_agl": altitude_plot, "rcs": z_json,
            "mean_profile": profile_json,
            "samples": {"time": n_time, "altitude": n_alt, "display_time": len(labels),
                        "display_altitude": len(altitude_plot)},
        }


def options_from_dataset(ds: xr.Dataset) -> dict[str, Any]:
    variables = [
        name for name in RCS_VARIABLES
        if name in ds and set(ds[name].dims) == {"time", "channel", "altitude"}
    ]
    return {"channels": _channels(ds), "variables": variables}
