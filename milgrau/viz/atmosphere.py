"""Level 1 atmospheric-profile comparison figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from milgrau.viz.style import add_footer_and_logos, get_output_settings

_BOLTZMANN_J_K = 1.380649e-23


def _settings(config: Mapping[str, Any]) -> tuple[float, tuple[tuple[float, float], ...]]:
    viz = config.get("visualization")
    if not isinstance(viz, Mapping):
        raise ValueError("Missing visualization configuration.")
    raw = viz.get("atmospheric_profile")
    if not isinstance(raw, Mapping):
        raise ValueError("Missing visualization.atmospheric_profile configuration.")

    max_altitude_km = float(raw.get("max_altitude_km", np.nan))
    if not np.isfinite(max_altitude_km) or max_altitude_km <= 0.0:
        raise ValueError("visualization.atmospheric_profile.max_altitude_km must be positive.")

    raw_bands = raw.get("comparison_altitude_bands_km")
    if not isinstance(raw_bands, list) or not raw_bands:
        raise ValueError(
            "visualization.atmospheric_profile.comparison_altitude_bands_km "
            "must be a non-empty list."
        )
    bands: list[tuple[float, float]] = []
    for index, item in enumerate(raw_bands):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(value, (int, float)) for value in item)
        ):
            raise ValueError(
                "Each atmospheric comparison altitude band must contain [min_km, max_km]."
            )
        lower, upper = float(item[0]), float(item[1])
        if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 <= lower < upper):
            raise ValueError(
                f"Invalid atmospheric comparison altitude band at index {index}: {item!r}."
            )
        bands.append((lower, upper))
    return max_altitude_km, tuple(bands)


def _nearest_atmosphere_index(ds: xr.Dataset) -> tuple[int, pd.Timestamp, pd.Timestamp | None]:
    atmosphere_times = pd.to_datetime(ds["atmosphere_time"].values)
    if atmosphere_times.size == 0:
        raise ValueError("Level 1 atmosphere_time is empty.")

    radio_target: pd.Timestamp | None = None
    raw_target = str(ds.attrs.get("radiosonde_qa_target_datetime_utc", "")).strip()
    if raw_target:
        parsed = pd.Timestamp(raw_target)
        if parsed.tzinfo is not None:
            parsed = parsed.tz_convert("UTC").tz_localize(None)
        radio_target = parsed

    if radio_target is None:
        index = int(atmosphere_times.size // 2)
    else:
        deltas = np.abs(
            atmosphere_times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            - radio_target.to_datetime64().astype("datetime64[ns]").astype(np.int64)
        )
        index = int(np.argmin(deltas))
    return index, pd.Timestamp(atmosphere_times[index]), radio_target


def _source_mask(
    altitude_agl_m: np.ndarray,
    *,
    station_altitude_m: float,
    min_asl_m: float,
    max_asl_m: float,
) -> np.ndarray:
    if not (np.isfinite(min_asl_m) and np.isfinite(max_asl_m) and max_asl_m > min_asl_m):
        return np.zeros(altitude_agl_m.shape, dtype=bool)
    min_agl = min_asl_m - station_altitude_m
    max_agl = max_asl_m - station_altitude_m
    return (altitude_agl_m >= min_agl) & (altitude_agl_m <= max_agl)


def _number_density_m3(pressure_hpa: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    pressure_pa = np.asarray(pressure_hpa, dtype=np.float64) * 100.0
    temperature = np.asarray(temperature_k, dtype=np.float64)
    return np.divide(
        pressure_pa,
        _BOLTZMANN_J_K * temperature,
        out=np.full_like(pressure_pa, np.nan, dtype=np.float64),
        where=np.isfinite(pressure_pa) & np.isfinite(temperature) & (temperature > 0.0),
    )


def _band_metrics(
    altitude_km: np.ndarray,
    used_t: np.ndarray,
    used_p: np.ndarray,
    radio_t: np.ndarray,
    radio_p: np.ndarray,
    radio_valid: np.ndarray,
    bands: tuple[tuple[float, float], ...],
) -> list[str]:
    lines: list[str] = []
    for lower, upper in bands:
        mask = (
            radio_valid
            & (altitude_km >= lower)
            & (altitude_km < upper)
            & np.isfinite(used_t)
            & np.isfinite(used_p)
            & np.isfinite(radio_t)
            & np.isfinite(radio_p)
            & (radio_p > 0.0)
        )
        count = int(np.count_nonzero(mask))
        if count == 0:
            lines.append(f"{lower:g}-{upper:g} km: no overlap")
            continue
        delta_t = used_t[mask] - radio_t[mask]
        pressure_rel = 100.0 * (used_p[mask] - radio_p[mask]) / radio_p[mask]
        lines.append(
            f"{lower:g}-{upper:g} km: "
            f"dT bias {np.mean(delta_t):+.2f} K, RMSE {np.sqrt(np.mean(delta_t**2)):.2f} K; "
            f"dP {np.mean(pressure_rel):+.2f}%; n={count}"
        )
    return lines


def plot_atmospheric_profile(
    ds: xr.Dataset,
    *,
    output_folder: str | Path,
    file_name_prefix: str,
    config: Mapping[str, Any],
    root_dir: str | Path,
) -> Path:
    """Compare canonical hourly atmosphere with ERA5 coverage and radiosonde QA."""
    required = {
        "Atmospheric_Temperature_K",
        "Atmospheric_Pressure_hPa",
        "Atmospheric_Source_Type",
        "Atmospheric_Source_Min_Altitude_ASL_m",
        "Atmospheric_Source_Max_Altitude_ASL_m",
    }
    missing = sorted(required - set(ds.variables))
    if missing:
        raise KeyError(f"Level 1 atmospheric figure lacks variable(s): {missing}")
    if "atmosphere_time" not in ds.coords or "altitude" not in ds.coords:
        raise KeyError("Level 1 atmospheric figure requires atmosphere_time and altitude.")

    output_format, dpi = get_output_settings(dict(config))
    max_altitude_km, bands = _settings(config)
    index, comparison_time, radio_target = _nearest_atmosphere_index(ds)

    altitude_m = np.asarray(ds["altitude"].values, dtype=np.float64)
    altitude_km = altitude_m / 1000.0
    plot_mask = altitude_km <= max_altitude_km
    station_altitude_m = float(ds.attrs.get("thermodynamic_station_altitude_m", 0.0))

    used_t = np.asarray(ds["Atmospheric_Temperature_K"].isel(atmosphere_time=index).values, dtype=np.float64)
    used_p = np.asarray(ds["Atmospheric_Pressure_hPa"].isel(atmosphere_time=index).values, dtype=np.float64)
    source_type = str(ds["Atmospheric_Source_Type"].isel(atmosphere_time=index).values)

    source_min = float(ds["Atmospheric_Source_Min_Altitude_ASL_m"].isel(atmosphere_time=index).values)
    source_max = float(ds["Atmospheric_Source_Max_Altitude_ASL_m"].isel(atmosphere_time=index).values)
    era5_valid = (
        _source_mask(
            altitude_m,
            station_altitude_m=station_altitude_m,
            min_asl_m=source_min,
            max_asl_m=source_max,
        )
        if source_type == "era5"
        else np.zeros(altitude_m.shape, dtype=bool)
    )
    era5_t = np.where(era5_valid, used_t, np.nan)
    era5_p = np.where(era5_valid, used_p, np.nan)

    radio_available = (
        str(ds.attrs.get("radiosonde_available", "false")).lower() == "true"
        and "Radiosonde_QA_Temperature_K" in ds
        and "Radiosonde_QA_Pressure_hPa" in ds
    )
    if radio_available:
        radio_t_full = np.asarray(ds["Radiosonde_QA_Temperature_K"].values, dtype=np.float64)
        radio_p_full = np.asarray(ds["Radiosonde_QA_Pressure_hPa"].values, dtype=np.float64)
        radio_valid = _source_mask(
            altitude_m,
            station_altitude_m=station_altitude_m,
            min_asl_m=float(ds.attrs.get("radiosonde_qa_source_profile_min_altitude_asl_m", np.nan)),
            max_asl_m=float(ds.attrs.get("radiosonde_qa_source_profile_max_altitude_asl_m", np.nan)),
        )
        radio_t = np.where(radio_valid, radio_t_full, np.nan)
        radio_p = np.where(radio_valid, radio_p_full, np.nan)
    else:
        radio_valid = np.zeros(altitude_m.shape, dtype=bool)
        radio_t = np.full(altitude_m.shape, np.nan, dtype=np.float64)
        radio_p = np.full(altitude_m.shape, np.nan, dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.0), sharey=True)
    ax_t, ax_p, ax_dt, ax_dp = axes.ravel()

    ax_t.plot(used_t[plot_mask], altitude_km[plot_mask], linewidth=2.3, label="Canonical used")
    if np.any(era5_valid & plot_mask):
        ax_t.plot(era5_t[plot_mask], altitude_km[plot_mask], linestyle="--", linewidth=1.7, label="ERA5 source coverage")
    if radio_available and np.any(radio_valid & plot_mask):
        ax_t.plot(radio_t[plot_mask], altitude_km[plot_mask], linestyle=":", linewidth=2.0, label="Radiosonde observed coverage")
    ax_t.set_xlabel("Temperature (K)")
    ax_t.set_ylabel("Altitude AGL (km)")
    ax_t.set_title("Temperature")
    ax_t.legend(fontsize=8)

    ax_p.semilogx(used_p[plot_mask], altitude_km[plot_mask], linewidth=2.3, label="Canonical used")
    if np.any(era5_valid & plot_mask):
        ax_p.semilogx(era5_p[plot_mask], altitude_km[plot_mask], linestyle="--", linewidth=1.7, label="ERA5 source coverage")
    if radio_available and np.any(radio_valid & plot_mask):
        ax_p.semilogx(radio_p[plot_mask], altitude_km[plot_mask], linestyle=":", linewidth=2.0, label="Radiosonde observed coverage")
    ax_p.set_xlabel("Pressure (hPa)")
    ax_p.set_title("Pressure")
    ax_p.legend(fontsize=8)

    if radio_available:
        delta_t = used_t - radio_t
        pressure_rel = np.divide(
            100.0 * (used_p - radio_p),
            radio_p,
            out=np.full_like(used_p, np.nan),
            where=np.isfinite(radio_p) & (radio_p > 0.0),
        )
        used_n = _number_density_m3(used_p, used_t)
        radio_n = _number_density_m3(radio_p, radio_t)
        density_rel = np.divide(
            100.0 * (used_n - radio_n),
            radio_n,
            out=np.full_like(used_n, np.nan),
            where=np.isfinite(radio_n) & (radio_n > 0.0),
        )
        ax_dt.plot(delta_t[plot_mask], altitude_km[plot_mask], linewidth=1.8)
        ax_dt.axvline(0.0, linestyle="--", linewidth=1.0)
        ax_dt.set_xlabel("Canonical - radiosonde (K)")
        ax_dt.set_title("Temperature difference")

        ax_dp.plot(pressure_rel[plot_mask], altitude_km[plot_mask], linewidth=1.8, label="Pressure")
        ax_dp.plot(density_rel[plot_mask], altitude_km[plot_mask], linestyle="--", linewidth=1.6, label="Molecular number density")
        ax_dp.axvline(0.0, linestyle=":", linewidth=1.0)
        ax_dp.set_xlabel("Canonical - radiosonde (%)")
        ax_dp.set_title("Pressure / molecular impact")
        ax_dp.legend(fontsize=8)
        metric_lines = _band_metrics(
            altitude_km,
            used_t,
            used_p,
            radio_t,
            radio_p,
            radio_valid,
            bands,
        )
    else:
        ax_dt.text(0.5, 0.5, "Radiosonde unavailable", ha="center", va="center", transform=ax_dt.transAxes)
        ax_dp.text(0.5, 0.5, "Radiosonde unavailable", ha="center", va="center", transform=ax_dp.transAxes)
        metric_lines = ["Radiosonde unavailable: comparison metrics not computed."]

    for axis in axes.ravel():
        axis.set_ylim(0.0, max_altitude_km)
        axis.grid(True, alpha=0.3)

    time_text = comparison_time.strftime("%Y-%m-%d %H:%M UTC")
    radio_text = "unavailable"
    if radio_target is not None:
        delta_hours = abs((comparison_time - radio_target).total_seconds()) / 3600.0
        radio_text = f"{radio_target.strftime('%Y-%m-%d %H:%M UTC')} (offset {delta_hours:.1f} h)"

    fig.suptitle(
        f"MILGRAU Level 1 — Atmospheric Profile Comparison\n"
        f"canonical time {time_text} | source {source_type} | radiosonde {radio_text}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.055,
        "\n".join(metric_lines),
        ha="center",
        va="bottom",
        fontsize=8.1,
    )
    fig.subplots_adjust(top=0.88, bottom=0.19, left=0.08, right=0.97, hspace=0.20, wspace=0.18)
    add_footer_and_logos(fig, root_dir)

    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{file_name_prefix}_L1_AtmosphericProfile.{output_format}"
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


__all__ = ["plot_atmospheric_profile"]
