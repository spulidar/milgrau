"""Level 1 atmospheric-profile and atmospheric-evolution figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from milgrau.viz.style import add_footer_and_logos, get_output_settings

_BOLTZMANN_J_K = 1.380649e-23


def _settings(
    config: Mapping[str, Any],
) -> tuple[float, tuple[tuple[float, float], ...], int]:
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

    evolution_bins = int(raw.get("evolution_max_altitude_bins", 600))
    if evolution_bins <= 0:
        raise ValueError(
            "visualization.atmospheric_profile.evolution_max_altitude_bins must be positive."
        )
    return max_altitude_km, tuple(bands), evolution_bins


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


def _source_used_text(ds: xr.Dataset, index: int | None = None) -> str:
    sources = np.asarray(ds["Atmospheric_Source_Type"].values).astype(str).reshape(-1)
    fallback = (
        np.asarray(ds["Atmospheric_USSA76_Fallback_Fraction"].values, dtype=np.float64).reshape(-1)
        if "Atmospheric_USSA76_Fallback_Fraction" in ds
        else np.zeros(sources.size, dtype=np.float64)
    )
    if index is not None:
        sources = sources[index : index + 1]
        fallback = fallback[index : index + 1]

    unique = list(dict.fromkeys(value.lower() for value in sources if value))
    if unique == ["era5"]:
        fraction = float(np.nanmean(fallback)) if fallback.size else 0.0
        if np.isfinite(fraction) and fraction > 0.0:
            return f"ERA5 + USSA76 vertical extension ({100.0 * fraction:.1f}% of altitude bins)"
        return "ERA5"
    if unique == ["ussa76"]:
        return "USSA76 fallback"
    if unique:
        return " + ".join(value.upper() if value == "ussa76" else value.upper() for value in unique)
    return "unknown"


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
            f"{lower:g}-{upper:g} km  "
            f"dT {np.mean(delta_t):+.2f} K  "
            f"RMSE {np.sqrt(np.mean(delta_t**2)):.2f} K  "
            f"dP {np.mean(pressure_rel):+.2f}%  n={count}"
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
    """Compare the Level 1 atmosphere used with the available radiosonde reference."""
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
    max_altitude_km, bands, _evolution_bins = _settings(config)
    index, comparison_time, radio_target = _nearest_atmosphere_index(ds)

    altitude_m = np.asarray(ds["altitude"].values, dtype=np.float64)
    altitude_km = altitude_m / 1000.0
    plot_mask = altitude_km <= max_altitude_km
    station_altitude_m = float(ds.attrs.get("thermodynamic_station_altitude_m", 0.0))

    used_t = np.asarray(
        ds["Atmospheric_Temperature_K"].isel(atmosphere_time=index).values,
        dtype=np.float64,
    )
    used_p = np.asarray(
        ds["Atmospheric_Pressure_hPa"].isel(atmosphere_time=index).values,
        dtype=np.float64,
    )
    source_type = str(ds["Atmospheric_Source_Type"].isel(atmosphere_time=index).values).lower()
    fallback_fraction = (
        float(ds["Atmospheric_USSA76_Fallback_Fraction"].isel(atmosphere_time=index).values)
        if "Atmospheric_USSA76_Fallback_Fraction" in ds
        else 0.0
    )

    source_min = float(
        ds["Atmospheric_Source_Min_Altitude_ASL_m"].isel(atmosphere_time=index).values
    )
    source_max = float(
        ds["Atmospheric_Source_Max_Altitude_ASL_m"].isel(atmosphere_time=index).values
    )
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
            min_asl_m=float(
                ds.attrs.get("radiosonde_qa_source_profile_min_altitude_asl_m", np.nan)
            ),
            max_asl_m=float(
                ds.attrs.get("radiosonde_qa_source_profile_max_altitude_asl_m", np.nan)
            ),
        )
        radio_t = np.where(radio_valid, radio_t_full, np.nan)
        radio_p = np.where(radio_valid, radio_p_full, np.nan)
    else:
        radio_valid = np.zeros(altitude_m.shape, dtype=bool)
        radio_t = np.full(altitude_m.shape, np.nan, dtype=np.float64)
        radio_p = np.full(altitude_m.shape, np.nan, dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.8), sharey=True)
    ax_t, ax_dt, ax_mol = axes

    if source_type == "era5":
        ax_t.plot(
            np.where(era5_valid, used_t, np.nan)[plot_mask],
            altitude_km[plot_mask],
            linewidth=2.2,
            label="ERA5",
        )
        extension = (~era5_valid) & plot_mask
        if fallback_fraction > 0.0 and np.any(extension):
            ax_t.plot(
                np.where(~era5_valid, used_t, np.nan)[plot_mask],
                altitude_km[plot_mask],
                linewidth=2.2,
                linestyle="--",
                label="USSA76 extension used",
            )
    elif source_type == "ussa76":
        ax_t.plot(
            used_t[plot_mask],
            altitude_km[plot_mask],
            linewidth=2.2,
            label="USSA76 used",
        )
    else:
        ax_t.plot(
            used_t[plot_mask],
            altitude_km[plot_mask],
            linewidth=2.2,
            label="L1 profile used",
        )

    if radio_available and np.any(radio_valid & plot_mask):
        ax_t.plot(
            radio_t[plot_mask],
            altitude_km[plot_mask],
            linestyle=":",
            linewidth=2.1,
            label="Radiosonde reference",
        )
    ax_t.set_xlabel("Temperature (K)")
    ax_t.set_ylabel("Altitude AGL (km)")
    ax_t.set_title("Temperature profile")
    ax_t.legend(fontsize=9)

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

        ax_dt.plot(delta_t[plot_mask], altitude_km[plot_mask], linewidth=1.9)
        ax_dt.axvline(0.0, linestyle="--", linewidth=1.0)
        ax_dt.set_xlabel("L1 used - radiosonde (K)")
        ax_dt.set_title("Temperature difference")

        ax_mol.plot(
            pressure_rel[plot_mask],
            altitude_km[plot_mask],
            linewidth=1.9,
            label="Pressure",
        )
        ax_mol.plot(
            density_rel[plot_mask],
            altitude_km[plot_mask],
            linestyle="--",
            linewidth=1.7,
            label="Molecular number density",
        )
        ax_mol.axvline(0.0, linestyle=":", linewidth=1.0)
        ax_mol.set_xlabel("L1 used - radiosonde (%)")
        ax_mol.set_title("Pressure / molecular impact")
        ax_mol.legend(fontsize=9)

        metric_lines = _band_metrics(
            altitude_km,
            used_t,
            used_p,
            radio_t,
            radio_p,
            radio_valid,
            bands,
        )
        ax_dt.text(
            0.03,
            0.03,
            "\n".join(metric_lines),
            transform=ax_dt.transAxes,
            fontsize=8.2,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.82},
        )
    else:
        for axis in (ax_dt, ax_mol):
            axis.text(
                0.5,
                0.5,
                "Radiosonde unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    for axis in axes:
        axis.set_ylim(0.0, max_altitude_km)
        axis.grid(True, alpha=0.3)

    radio_text = "unavailable"
    if radio_target is not None:
        delta_hours = abs((comparison_time - radio_target).total_seconds()) / 3600.0
        radio_text = (
            f"{radio_target.strftime('%Y-%m-%d %H:%M UTC')} "
            f"(offset {delta_hours:.1f} h)"
        )
    source_text = _source_used_text(ds, index)

    fig.suptitle(
        "MILGRAU Level 1 — Atmospheric Profile Comparison",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        f"L1 source used: {source_text}",
        ha="center",
        va="top",
        fontsize=11.2,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.918,
        f"Reference time: {comparison_time.strftime('%Y-%m-%d %H:%M UTC')}  |  "
        f"Radiosonde: {radio_text}",
        ha="center",
        va="top",
        fontsize=10.2,
    )
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.07, right=0.98, wspace=0.20)
    add_footer_and_logos(fig, root_dir)

    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{file_name_prefix}_L1_AtmosphericProfile.{output_format}"
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def plot_atmospheric_evolution(
    ds: xr.Dataset,
    *,
    output_folder: str | Path,
    file_name_prefix: str,
    config: Mapping[str, Any],
    root_dir: str | Path,
) -> Path:
    """Visualize how the Level 1 atmospheric state evolves across the session."""
    required = {
        "Atmospheric_Temperature_K",
        "Atmospheric_Pressure_hPa",
        "Atmospheric_Source_Type",
    }
    missing = sorted(required - set(ds.variables))
    if missing:
        raise KeyError(f"Level 1 atmospheric evolution lacks variable(s): {missing}")
    if "atmosphere_time" not in ds.coords or "altitude" not in ds.coords:
        raise KeyError("Atmospheric evolution requires atmosphere_time and altitude.")

    output_format, dpi = get_output_settings(dict(config))
    max_altitude_km, _bands, max_bins = _settings(config)

    altitude_m = np.asarray(ds["altitude"].values, dtype=np.float64)
    altitude_km = altitude_m / 1000.0
    valid_altitude = np.where(altitude_km <= max_altitude_km)[0]
    if valid_altitude.size == 0:
        raise ValueError("No Level 1 altitude bins fall within the atmospheric plot range.")
    step = max(1, int(np.ceil(valid_altitude.size / max_bins)))
    altitude_index = valid_altitude[::step]
    altitude_plot = altitude_km[altitude_index]

    times = pd.to_datetime(ds["atmosphere_time"].values)
    temperature = np.asarray(
        ds["Atmospheric_Temperature_K"].isel(altitude=altitude_index).values,
        dtype=np.float64,
    )
    pressure = np.asarray(
        ds["Atmospheric_Pressure_hPa"].isel(altitude=altitude_index).values,
        dtype=np.float64,
    )
    if temperature.ndim != 2 or pressure.shape != temperature.shape:
        raise ValueError("Atmospheric evolution expects atmosphere_time x altitude fields.")

    temperature_anomaly = temperature - np.nanmean(temperature, axis=0, keepdims=True)
    number_density = _number_density_m3(pressure, temperature)
    reference_density = number_density[0:1, :]
    density_change = np.divide(
        100.0 * (number_density - reference_density),
        reference_density,
        out=np.full_like(number_density, np.nan),
        where=np.isfinite(reference_density) & (reference_density > 0.0),
    )

    fig, axes = plt.subplots(2, 1, figsize=(13.8, 9.0), sharex=True, sharey=True)
    ax_t, ax_n = axes

    mesh_t = ax_t.pcolormesh(
        times,
        altitude_plot,
        temperature_anomaly.T,
        shading="auto",
    )
    cbar_t = fig.colorbar(mesh_t, ax=ax_t, pad=0.015)
    cbar_t.set_label("Temperature anomaly (K)")
    ax_t.set_title("Temperature anomaly relative to the session mean")
    ax_t.set_ylabel("Altitude AGL (km)")

    mesh_n = ax_n.pcolormesh(
        times,
        altitude_plot,
        density_change.T,
        shading="auto",
    )
    cbar_n = fig.colorbar(mesh_n, ax=ax_n, pad=0.015)
    cbar_n.set_label("Molecular number-density change (%)")
    ax_n.set_title("Molecular number density relative to the first hourly profile")
    ax_n.set_ylabel("Altitude AGL (km)")
    ax_n.set_xlabel("Atmospheric analysis time (UTC)")

    for axis in axes:
        axis.set_ylim(0.0, max_altitude_km)
        axis.grid(False)
    ax_n.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))

    source_text = _source_used_text(ds)
    fallback = (
        np.asarray(ds["Atmospheric_USSA76_Fallback_Fraction"].values, dtype=np.float64)
        if "Atmospheric_USSA76_Fallback_Fraction" in ds
        else np.zeros(times.size, dtype=np.float64)
    )
    full_fallback_hours = int(np.count_nonzero(fallback >= 0.999))
    source_suffix = (
        f"  |  full USSA76 fallback hours: {full_fallback_hours}/{times.size}"
        if full_fallback_hours
        else ""
    )
    fig.suptitle(
        "MILGRAU Level 1 — Atmospheric Evolution\n"
        f"Hourly L1 source used: {source_text}{source_suffix}",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.08, right=0.92, hspace=0.20)
    add_footer_and_logos(fig, root_dir)

    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{file_name_prefix}_L1_AtmosphericEvolution.{output_format}"
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


__all__ = ["plot_atmospheric_evolution", "plot_atmospheric_profile"]
