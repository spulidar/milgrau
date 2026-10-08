"""Canonical Level 2 scientific and diagnostic figures.

These figures consume the current Level 2 variables directly. They do not
create compatibility aliases for older schemas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from milgrau.viz.profile_helpers import (
    altitude_to_km,
    display_scale_factor,
    format_wavelength_label,
    get_wavelength_values,
    infer_l1_channels_for_wavelength,
    smooth_for_plot,
)
from milgrau.viz.quicklooks import extract_datetime_strings, safe_time_mean
from milgrau.viz.style import add_footer_and_logos, channel_color, get_output_settings


def is_level2_dataset(ds_l2: xr.Dataset) -> bool:
    """Return whether the dataset exposes the current productive contract."""
    required = {
        "range_corrected_signal_block",
        "aerosol_backscatter_nominal_block",
        "rayleigh_reference_altitude_m_block",
        "selected_reference_altitude_m_mc",
        "period_support_fraction",
    }
    return required.issubset(set(ds_l2.data_vars))


def _date_title(ds_l2: xr.Dataset) -> str:
    """Return a readable observation interval for block products."""
    if "time" in ds_l2.coords:
        return extract_datetime_strings(ds_l2)[0]
    if "block_time" not in ds_l2.coords or ds_l2.sizes.get("block_time", 0) == 0:
        return "Unknown date"
    values = np.asarray(ds_l2["block_time"].values)
    try:
        start = np.datetime_as_string(values.min(), unit="m").replace("T", " ")
        stop = np.datetime_as_string(values.max(), unit="m").replace("T", " ")
        return f"{start} to {stop} UTC"
    except Exception:
        return "Unknown date"

def _block_x(ds_l2: xr.Dataset) -> tuple[np.ndarray, list[str]]:
    n = int(ds_l2.sizes.get("block_time", 0))
    x = np.arange(n, dtype=np.int32)
    if "block_time" not in ds_l2.coords:
        return x, [str(i + 1) for i in x]
    values = np.asarray(ds_l2["block_time"].values)
    labels: list[str] = []
    for value in values:
        try:
            labels.append(np.datetime_as_string(value, unit="m")[11:16])
        except Exception:
            labels.append(str(len(labels) + 1))
    return x, labels


def _save(fig: Any, output_folder: str | Path, name: str, dpi: int) -> Path:
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _finite_mean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return a finite-only mean without empty-slice warnings."""
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    count = np.count_nonzero(finite, axis=axis)
    total = np.sum(np.where(finite, array, 0.0), axis=axis)
    result = np.full(np.asarray(total).shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def _finite_rms(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return finite-only RMS, used as typical block-level MC dispersion."""
    return np.sqrt(_finite_mean(np.asarray(values, dtype=np.float64) ** 2, axis=axis))


def _finite_median(values: np.ndarray) -> float:
    """Return a finite median without all-NaN warnings."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else np.nan


def _profile_xlim(
    profile: np.ndarray,
    uncertainty: np.ndarray | None,
    *,
    default_abs: float,
) -> tuple[float, float]:
    """Return robust symmetric limits driven by the central optical profile."""
    central = np.asarray(profile, dtype=np.float64)
    finite = central[np.isfinite(central)]
    limit = (
        max(float(default_abs), 1.35 * float(np.percentile(np.abs(finite), 98.0)))
        if finite.size
        else float(default_abs)
    )
    if uncertainty is not None:
        band = np.abs(central) + np.asarray(uncertainty, dtype=np.float64)
        finite_band = band[np.isfinite(band)]
        if finite_band.size:
            limit = max(
                limit,
                min(2.0 * limit, float(np.percentile(finite_band, 95.0))),
            )
    return -limit, limit


def _median_variable(ds_l2: xr.Dataset, name: str, wavelength: int) -> float:
    """Return a wavelength-selected finite median for an optional variable."""
    if name not in ds_l2:
        return np.nan
    values = np.asarray(ds_l2[name].sel(wavelength=wavelength).values, dtype=np.float64)
    return _finite_median(values)


def _molecular_rcs_shape(
    altitude_m: np.ndarray,
    molecular_backscatter: np.ndarray,
    molecular_extinction: np.ndarray,
) -> np.ndarray:
    """Return the uncalibrated molecular RCS shape beta_mol * T_mol^2."""
    altitude = np.asarray(altitude_m, dtype=np.float64)
    beta = np.asarray(molecular_backscatter, dtype=np.float64)
    alpha = np.asarray(molecular_extinction, dtype=np.float64)
    if not (altitude.shape == beta.shape == alpha.shape):
        return np.full_like(altitude, np.nan, dtype=np.float64)
    tau = np.zeros_like(altitude, dtype=np.float64)
    if altitude.size > 1:
        layers = 0.5 * (alpha[:-1] + alpha[1:]) * np.diff(altitude)
        invalid = np.flatnonzero(~np.isfinite(layers))
        if invalid.size:
            layers[int(invalid[0]) :] = np.nan
        tau[1:] = np.cumsum(layers)
    return beta * np.exp(-2.0 * tau)


def _display_scale_factor(
    observed: np.ndarray,
    molecular: np.ndarray,
    altitude_km: np.ndarray,
    reference_km: float,
) -> float:
    """Scale a molecular shape to observed RCS near the median reference."""
    valid = (
        np.isfinite(observed)
        & np.isfinite(molecular)
        & (molecular > 0.0)
        & (altitude_km >= reference_km - 0.5)
        & (altitude_km <= reference_km + 0.5)
    )
    if not np.any(valid) and np.isfinite(reference_km):
        candidates = np.flatnonzero(
            np.isfinite(observed) & np.isfinite(molecular) & (molecular > 0.0)
        )
        if candidates.size:
            order = np.argsort(np.abs(altitude_km[candidates] - reference_km))
            valid[candidates[order[: min(5, candidates.size)]]] = True
    ratios = np.asarray(observed[valid] / molecular[valid], dtype=np.float64)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
    return float(np.median(ratios)) if ratios.size else np.nan


def plot_molecular_reference(
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
) -> Path | None:
    """Plot a molecular-profile comparison with current diagnostics."""
    wavelength = int(wavelength_nm)
    required = {
        "rayleigh_reference_altitude_m_block",
        "range_corrected_signal_block",
        "molecular_backscatter",
        "molecular_extinction",
        "rayleigh_reference_relative_slope_block",
        "rayleigh_reference_relative_variance_block",
        "rayleigh_reference_snr_median_block",
        "selection_success_fraction_block",
    }
    if not required.issubset(set(ds_l2.data_vars)):
        return None

    output_format, dpi = get_output_settings(config)
    date_title = _date_title(ds_l2)
    sel = dict(wavelength=wavelength)
    altitude_m = np.asarray(ds_l2["altitude"].values, dtype=np.float64)
    altitude_km = altitude_to_km(altitude_m)
    figure_cfg = config.get("visualization", {}).get("level2_figures", {}) or {}
    max_altitude_km = min(
        float(figure_cfg.get("max_altitude_km", 30.0)),
        float(np.nanmax(altitude_km)),
    )
    valid_alt = altitude_km <= max_altitude_km
    smooth_bins = int(figure_cfg.get("smooth_bins", 15))
    observed = smooth_for_plot(
        _finite_mean(
            np.asarray(ds_l2["range_corrected_signal_block"].sel(**sel).values)
        ),
        smooth_bins,
    )
    beta_mol = _finite_mean(
        np.asarray(
            ds_l2["molecular_backscatter"].sel(**sel).values,
            dtype=np.float64,
        ),
        axis=0,
    )
    alpha_mol = _finite_mean(
        np.asarray(
            ds_l2["molecular_extinction"].sel(**sel).values,
            dtype=np.float64,
        ),
        axis=0,
    )
    molecular_shape = _molecular_rcs_shape(altitude_m, beta_mol, alpha_mol)

    references = (
        np.asarray(
            ds_l2["rayleigh_reference_altitude_m_block"].sel(**sel).values,
            dtype=np.float64,
        )
        / 1000.0
    )
    finite_references = references[np.isfinite(references)]
    reference = _finite_median(finite_references)

    fit_cfg = config.get("inversion", {}).get("molecular_fit", {}) or {}
    ref_min_km = float(fit_cfg.get("ref_alt_min_m", np.nan)) / 1000.0
    ref_max_km = float(fit_cfg.get("ref_alt_max_m", np.nan)) / 1000.0
    if not np.isfinite(reference) and np.isfinite(ref_min_km + ref_max_km):
        reference = 0.5 * (ref_min_km + ref_max_km)
    scale = _display_scale_factor(observed, molecular_shape, altitude_km, reference)
    molecular_scaled = molecular_shape * scale
    ratio = np.divide(
        observed,
        molecular_scaled,
        out=np.full_like(observed, np.nan),
        where=np.isfinite(molecular_scaled) & (molecular_scaled > 0.0),
    )

    slope = np.asarray(
        ds_l2["rayleigh_reference_relative_slope_block"].sel(**sel).values,
        dtype=float,
    )
    variance = np.asarray(
        ds_l2["rayleigh_reference_relative_variance_block"].sel(**sel).values,
        dtype=float,
    )
    snr = np.asarray(
        ds_l2["rayleigh_reference_snr_median_block"].sel(**sel).values,
        dtype=float,
    )
    selection_fraction = np.asarray(
        ds_l2["selection_success_fraction_block"].sel(**sel).values,
        dtype=float,
    )
    color = channel_color(wavelength)

    fig = plt.figure(figsize=(13.8, 8.9))
    grid = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.08], wspace=0.28)
    ax_fit = fig.add_subplot(grid[0])
    ax_profile = fig.add_subplot(grid[1])

    fit_window = (
        valid_alt
        & np.isfinite(observed)
        & np.isfinite(molecular_scaled)
        & (altitude_km >= ref_min_km)
        & (altitude_km <= ref_max_km)
    )
    if np.any(fit_window):
        points = ax_fit.scatter(
            molecular_scaled[fit_window],
            observed[fit_window],
            c=altitude_km[fit_window],
            s=13,
            alpha=0.75,
            cmap="viridis",
        )
        paired = np.concatenate(
            [molecular_scaled[fit_window], observed[fit_window]]
        )
        low, high = float(np.nanmin(paired)), float(np.nanmax(paired))
        ax_fit.plot([low, high], [low, high], "--", color="black", linewidth=1.4)
        fig.colorbar(points, ax=ax_fit, pad=0.02, label="Altitude (km a.g.l.)")
    ax_fit.set_title("Molecular agreement in search column", fontweight="bold")
    ax_fit.set_xlabel("Display-scaled molecular RCS [a.u.]", fontweight="bold")
    ax_fit.set_ylabel("Background-corrected selected RCS [a.u.]", fontweight="bold")
    ax_fit.grid(True, alpha=0.38)

    ax_profile.plot(
        observed[valid_alt],
        altitude_km[valid_alt],
        color=color,
        linewidth=2.2,
        label="Selected RCS selected mean",
    )
    ax_profile.plot(
        molecular_scaled[valid_alt],
        altitude_km[valid_alt],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Molecular RCS shape (display-scaled)",
    )
    if np.isfinite(ref_min_km) and np.isfinite(ref_max_km):
        ax_profile.axhspan(
            ref_min_km,
            min(ref_max_km, max_altitude_km),
            color="0.7",
            alpha=0.12,
            label="Background / reference search column",
        )
    if finite_references.size:
        q25, q75 = np.quantile(finite_references, [0.25, 0.75])
        ax_profile.axhspan(q25, q75, color=color, alpha=0.13, label="Reference IQR")
    if np.isfinite(reference):
        ax_profile.axhline(
            reference,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"Median reference {reference:.2f} km",
        )
    profile_values = np.concatenate(
        [observed[valid_alt], molecular_scaled[valid_alt]]
    )
    profile_abs = np.abs(
        profile_values[np.isfinite(profile_values) & (profile_values != 0.0)]
    )
    profile_linthresh = (
        max(float(np.percentile(profile_abs, 10.0)), 1.0e-12)
        if profile_abs.size
        else 1.0e-3
    )
    ax_profile.set_xscale("symlog", linthresh=profile_linthresh)
    ax_profile.set_ylim(0.0, max_altitude_km)
    ax_profile.set_title("Measured and molecular vertical profiles", fontweight="bold")
    ax_profile.set_xlabel("RCS [a.u.]", fontweight="bold")
    ax_profile.set_ylabel("Altitude (km a.g.l.)", fontweight="bold")
    ax_profile.grid(True, which="both", alpha=0.38)
    ax_profile.legend(fontsize=8.1, loc="upper left")

    inset = ax_profile.inset_axes([0.06, 0.08, 0.37, 0.28])
    inset.plot(ratio[valid_alt], altitude_km[valid_alt], color=color, linewidth=1.2)
    inset.axvline(1.0, color="black", linestyle="--", linewidth=0.9)
    inset.set_xlim(0.0, 2.5)
    inset.set_ylim(max(0.0, ref_min_km), min(max_altitude_km, ref_max_km))
    inset.set_title("Measured / molecular", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(True, alpha=0.3)

    background = _median_variable(ds_l2, "rayleigh_background_offset_block", wavelength)
    background_corr = _median_variable(
        ds_l2, "rayleigh_calibration_background_correlation_block", wavelength
    )
    summary = [
        f"relative slope med = {_finite_median(slope):.3g}",
        f"relative variance med = {_finite_median(variance):.3g}",
        f"reference SNR med = {_finite_median(snr):.3g}",
        f"MC selection success med = {_finite_median(selection_fraction):.1%}",
    ]
    if np.isfinite(background):
        summary.append(f"background B med = {background:.3g}")
    if np.isfinite(background_corr):
        summary.append(f"corr(A,B) med = {background_corr:.3f}")
    ax_fit.text(
        0.03,
        0.97,
        "\n".join(summary),
        transform=ax_fit.transAxes,
        va="top",
        fontsize=8.8,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.65"},
    )
    fig.suptitle(
        f"MILGRAU Level 2 — Molecular Reference — {format_wavelength_label(wavelength)}\n{date_title}",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.095,
        "The molecular curve is reconstructed from stored molecular backscatter/extinction and scaled only for display near the median reference; "
        "operational fitting and reference selection remain block-resolved.",
        ha="center",
        fontsize=8.2,
    )
    fig.subplots_adjust(top=0.86, bottom=0.17, left=0.08, right=0.96)
    add_footer_and_logos(fig, root_dir)
    return _save(
        fig,
        output_folder,
        f"{file_name_prefix}_L2_MolecularReference_{wavelength}nm.{output_format}",
        dpi,
    )


def plot_optical_profiles(
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
) -> Path | None:
    """Plot publication-style optical profiles and the supported retrieval top."""
    wavelength = int(wavelength_nm)
    required = {
        "aerosol_backscatter_mean",
        "aerosol_extinction_mean",
    }
    if not required.issubset(set(ds_l2.data_vars)):
        return None

    output_format, dpi = get_output_settings(config)
    date_title = _date_title(ds_l2)
    altitude_km = altitude_to_km(ds_l2["altitude"].values)
    figure_cfg = config.get("visualization", {}).get("level2_figures", {}) or {}
    max_altitude_km = min(
        float(figure_cfg.get("max_altitude_km", 30.0)),
        float(np.nanmax(altitude_km)),
    )
    valid_alt = altitude_km <= max_altitude_km
    smooth_bins = int(figure_cfg.get("smooth_bins", 15))
    sel = dict(wavelength=wavelength)
    beta_mean = smooth_for_plot(
        np.asarray(ds_l2["aerosol_backscatter_mean"].sel(**sel).values, dtype=float)
        * 1e6,
        smooth_bins,
    )
    alpha_mean = smooth_for_plot(
        np.asarray(ds_l2["aerosol_extinction_mean"].sel(**sel).values, dtype=float)
        * 1e6,
        smooth_bins,
    )
    color = channel_color(wavelength)

    beta_sigma = np.full_like(beta_mean, np.nan)
    alpha_sigma = np.full_like(alpha_mean, np.nan)
    if {
        "aerosol_backscatter_mc_std",
        "aerosol_extinction_mc_std",
        "residual_fraction",
    }.issubset(set(ds_l2.variables)):
        fractions = np.asarray(ds_l2["residual_fraction"].values, dtype=float)
        nominal_fraction = float(fractions[np.argmin(np.abs(fractions))])
        mc_sel = {"wavelength": wavelength, "residual_fraction": nominal_fraction}
        beta_sigma = smooth_for_plot(
            _finite_rms(
                np.asarray(ds_l2["aerosol_backscatter_mc_std"].sel(**mc_sel).values)
            )
            * 1e6,
            smooth_bins,
        )
        alpha_sigma = smooth_for_plot(
            _finite_rms(
                np.asarray(ds_l2["aerosol_extinction_mc_std"].sel(**mc_sel).values)
            )
            * 1e6,
            smooth_bins,
        )

    fig = plt.figure(figsize=(12.8, 9.2))
    grid = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.18)
    ax_beta = fig.add_subplot(grid[0])
    ax_alpha = fig.add_subplot(grid[1], sharey=ax_beta)

    if np.any(np.isfinite(beta_sigma)):
        ax_beta.fill_betweenx(
            altitude_km[valid_alt],
            (beta_mean - beta_sigma)[valid_alt],
            (beta_mean + beta_sigma)[valid_alt],
            color=color,
            alpha=0.22,
            edgecolor="none",
            label="Typical block MC 1σ (f=0)",
        )
    if np.any(np.isfinite(alpha_sigma)):
        ax_alpha.fill_betweenx(
            altitude_km[valid_alt],
            (alpha_mean - alpha_sigma)[valid_alt],
            (alpha_mean + alpha_sigma)[valid_alt],
            color=color,
            alpha=0.22,
            edgecolor="none",
            label="Typical block MC 1σ (f=0)",
        )
    ax_beta.plot(
        beta_mean[valid_alt], altitude_km[valid_alt], color=color, linewidth=2.35,
        label="Selected nominal mean",
    )
    ax_alpha.plot(
        alpha_mean[valid_alt], altitude_km[valid_alt], color=color, linewidth=2.35,
        label="Selected nominal mean",
    )
    for axis in (ax_beta, ax_alpha):
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.grid(True, alpha=0.38)
        axis.set_ylim(0.0, max_altitude_km)
        for target in (20.0, 25.0):
            if target <= max_altitude_km:
                axis.axhline(target, color="0.72", linestyle="--", linewidth=0.8)

    ax_beta.set_xlim(
        *_profile_xlim(beta_mean[valid_alt], beta_sigma[valid_alt], default_abs=5.0)
    )
    ax_alpha.set_xlim(
        *_profile_xlim(alpha_mean[valid_alt], alpha_sigma[valid_alt], default_abs=50.0)
    )
    ax_beta.set_xlabel(r"$\beta_{aer}$ [Mm$^{-1}$ sr$^{-1}$]", fontweight="bold")
    ax_beta.set_ylabel("Altitude (km a.g.l.)", fontweight="bold")
    ax_beta.set_title("Aerosol backscatter", fontweight="bold")
    ax_alpha.set_xlabel(r"$\alpha_{aer}$ [Mm$^{-1}$]", fontweight="bold")
    ax_alpha.set_title(
        "Aerosol extinction\n(conditional on lidar ratio)", fontweight="bold"
    )
    plt.setp(ax_alpha.get_yticklabels(), visible=False)

    top = (
        float(ds_l2["retrieval_top_altitude_m"].sel(**sel).values) / 1000.0
        if "retrieval_top_altitude_m" in ds_l2
        else np.nan
    )
    reference = _median_variable(
        ds_l2, "rayleigh_reference_altitude_m_block", wavelength
    ) / 1000.0
    forward_endpoint = _median_variable(
        ds_l2, "kfs_forward_endpoint_altitude_m", wavelength
    ) / 1000.0
    for axis in (ax_beta, ax_alpha):
        if np.isfinite(top):
            axis.axhline(top, linestyle="--", color="black", linewidth=1.25)
        if np.isfinite(reference):
            axis.axhline(reference, color="black", linestyle=":", linewidth=1.25)
        if np.isfinite(forward_endpoint):
            axis.axhline(
                forward_endpoint, color="purple", linestyle="-.", linewidth=1.1
            )

    lidar_ratio = _median_variable(ds_l2, "lidar_ratio_assumed_sr", wavelength)
    summary = [
        f"LR = {lidar_ratio:.1f} sr" if np.isfinite(lidar_ratio) else "LR = n/a"
    ]
    if np.isfinite(reference):
        summary.append(f"ref med = {reference:.2f} km")
    if np.isfinite(forward_endpoint):
        summary.append(f"forward med = {forward_endpoint:.2f} km")
    if np.isfinite(top):
        summary.append(f"retrieval top = {top:.2f} km")
    ax_beta.text(
        0.03, 0.97, "\n".join(summary), transform=ax_beta.transAxes, va="top",
        fontsize=8.8,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.65"},
    )
    ax_beta.legend(fontsize=8.2, loc="lower right")
    ax_alpha.legend(fontsize=8.2, loc="lower right")
    fig.suptitle(
        f"MILGRAU Level 2 — Elastic Optical Profiles — {format_wavelength_label(wavelength)}\n{date_title}",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.095,
        "Band = RMS block-level random MC dispersion, not uncertainty of the selected mean. "
        "Black dashed = retrieval top; dotted = median reference; purple dash-dot = median forward endpoint.",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.86, bottom=0.17, left=0.065, right=0.97)
    add_footer_and_logos(fig, root_dir)
    return _save(
        fig,
        output_folder,
        f"{file_name_prefix}_L2_OpticalProfiles_{wavelength}nm.{output_format}",
        dpi,
    )


def plot_mc_reference(
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
) -> Path | None:
    """Plot selection-aware MC reference-altitude distributions per time block."""
    wavelength = int(wavelength_nm)
    if "selected_reference_altitude_m_mc" not in ds_l2:
        return None
    output_format, dpi = get_output_settings(config)
    date_title = _date_title(ds_l2)
    samples = np.asarray(
        ds_l2["selected_reference_altitude_m_mc"].sel(wavelength=wavelength).values,
        dtype=float,
    ) / 1000.0
    x, labels = _block_x(ds_l2)

    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    finite_sets = [row[np.isfinite(row)] for row in samples]
    positions = [int(i) for i, row in enumerate(finite_sets) if row.size]
    values = [finite_sets[i] for i in positions]
    if values:
        ax.boxplot(values, positions=positions, widths=0.55, showfliers=False)
    nominal = np.asarray(
        ds_l2["rayleigh_reference_altitude_m_block"].sel(wavelength=wavelength).values,
        dtype=float,
    ) / 1000.0
    ax.plot(x, nominal, "o-", color=channel_color(wavelength), label="Nominal selected reference")
    step = max(1, int(np.ceil(max(len(x), 1) / 14)))
    ax.set_xticks(x[::step], labels[::step], rotation=45, ha="right")
    ax.set_ylabel("Reference altitude (km a.g.l.)")
    ax.set_xlabel("Block UTC")
    ax.set_title("Selection-aware Monte Carlo reference distribution")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(
        f"MILGRAU MILGRAU Level 2 — MC Reference - {format_wavelength_label(wavelength)}\n{date_title}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    add_footer_and_logos(fig, root_dir)
    return _save(
        fig,
        output_folder,
        f"{file_name_prefix}_L2_MCReference_{wavelength}nm.{output_format}",
        dpi,
    )


def plot_gluing(
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
    ds_l1: xr.Dataset | None = None,
) -> Path | None:
    """Plot a vertical signal profile with gluing diagnostics."""
    wavelength = int(wavelength_nm)
    required = {
        "gluing_attempted_flag",
        "gluing_success_flag",
        "gluing_start_altitude_m",
        "gluing_split_altitude_m",
        "gluing_stop_altitude_m",
        "gluing_correlation",
        "gluing_relative_rmse",
        "gluing_relative_bias",
    }
    if not required.issubset(set(ds_l2.data_vars)):
        return None
    output_format, dpi = get_output_settings(config)
    date_title = _date_title(ds_l2)
    sel = dict(wavelength=wavelength)

    attempted = np.asarray(ds_l2["gluing_attempted_flag"].sel(**sel).values, dtype=bool)
    if not np.any(attempted):
        return None
    success = np.asarray(ds_l2["gluing_success_flag"].sel(**sel).values, dtype=bool)
    start = np.asarray(
        ds_l2["gluing_start_altitude_m"].sel(**sel).values, dtype=float
    ) / 1000.0
    split = np.asarray(
        ds_l2["gluing_split_altitude_m"].sel(**sel).values, dtype=float
    ) / 1000.0
    stop = np.asarray(
        ds_l2["gluing_stop_altitude_m"].sel(**sel).values, dtype=float
    ) / 1000.0
    corr = np.asarray(ds_l2["gluing_correlation"].sel(**sel).values, dtype=float)
    rmse = np.asarray(ds_l2["gluing_relative_rmse"].sel(**sel).values, dtype=float)
    bias = np.asarray(ds_l2["gluing_relative_bias"].sel(**sel).values, dtype=float)

    altitude_m = np.asarray(ds_l2["altitude"].values, dtype=np.float64)
    altitude_km = altitude_to_km(altitude_m)
    figure_cfg = config.get("visualization", {}).get("level2_figures", {}) or {}
    max_altitude_km = min(
        float(figure_cfg.get("max_altitude_km", 30.0)),
        float(np.nanmax(altitude_km)),
    )
    valid_alt = altitude_km <= max_altitude_km
    smooth_bins = int(figure_cfg.get("smooth_bins", 15))
    selected = smooth_for_plot(
        _finite_mean(
            np.asarray(ds_l2["range_corrected_signal_block"].sel(**sel).values)
        ),
        smooth_bins,
    )

    analog_profile = np.full_like(selected, np.nan)
    photon_profile = np.full_like(selected, np.nan)
    l1_altitude_km = altitude_km
    analog_channel = photon_channel = None
    scaling_note = "L1 channels unavailable"
    if ds_l1 is not None:
        analog_channel, photon_channel = infer_l1_channels_for_wavelength(
            ds_l1, wavelength
        )
        if "altitude" in ds_l1.coords:
            l1_altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
            if np.nanmax(l1_altitude_m) <= 100.0:
                l1_altitude_m = l1_altitude_m * 1000.0
            l1_altitude_km = l1_altitude_m / 1000.0
        else:
            l1_altitude_m = altitude_m

        if analog_channel is not None and photon_channel is not None:
            if "corrected_signal" in ds_l1:
                analog_signal = np.asarray(
                    safe_time_mean(
                        ds_l1["corrected_signal"].sel(channel=analog_channel)
                    ).values,
                    dtype=np.float64,
                )
                photon_signal = np.asarray(
                    safe_time_mean(
                        ds_l1["corrected_signal"].sel(channel=photon_channel)
                    ).values,
                    dtype=np.float64,
                )
                slope_value = _median_variable(ds_l2, "gluing_slope", wavelength)
                intercept = _median_variable(ds_l2, "gluing_intercept", wavelength)
                if (
                    np.isfinite(slope_value)
                    and slope_value > 0.0
                    and np.isfinite(intercept)
                ):
                    analog_signal = slope_value * analog_signal + intercept
                    scaling_note = "AN scaled with median operational coefficients"
                else:
                    factor, bins = display_scale_factor(analog_signal, photon_signal)
                    analog_signal = analog_signal * factor
                    scaling_note = f"AN display-scaled in bins {bins[0]}:{bins[1]}"
                analog_profile = smooth_for_plot(
                    analog_signal * l1_altitude_m**2, smooth_bins
                )
                photon_profile = smooth_for_plot(
                    photon_signal * l1_altitude_m**2, smooth_bins
                )
            elif "range_corrected_signal" in ds_l1:
                analog_profile = np.asarray(
                    safe_time_mean(
                        ds_l1["range_corrected_signal"].sel(channel=analog_channel)
                    ).values,
                    dtype=np.float64,
                )
                photon_profile = np.asarray(
                    safe_time_mean(
                        ds_l1["range_corrected_signal"].sel(channel=photon_channel)
                    ).values,
                    dtype=np.float64,
                )
                factor, bins = display_scale_factor(analog_profile, photon_profile)
                analog_profile = smooth_for_plot(
                    analog_profile * factor, smooth_bins
                )
                photon_profile = smooth_for_plot(photon_profile, smooth_bins)
                scaling_note = f"AN RCS display-scaled in bins {bins[0]}:{bins[1]}"

    median_start = _finite_median(start)
    median_split = _finite_median(split)
    median_stop = _finite_median(stop)
    color = channel_color(wavelength)

    fig, ax = plt.subplots(figsize=(9.2, 11.5))
    l1_valid = l1_altitude_km <= max_altitude_km
    if analog_profile.size == l1_altitude_km.size and np.any(np.isfinite(analog_profile)):
        ax.plot(
            analog_profile[l1_valid], l1_altitude_km[l1_valid], "--",
            color="tab:blue", linewidth=1.7,
            label=f"{analog_channel or 'AN'} mean (scaled)",
        )
    if photon_profile.size == l1_altitude_km.size and np.any(np.isfinite(photon_profile)):
        ax.plot(
            photon_profile[l1_valid], l1_altitude_km[l1_valid], ":",
            color="tab:orange", linewidth=1.9,
            label=f"{photon_channel or 'PC'} mean",
        )
    ax.plot(
        selected[valid_alt], altitude_km[valid_alt], color=color, linewidth=2.5,
        label="Background-corrected selected RCS",
    )
    if np.isfinite(median_start) and np.isfinite(median_stop):
        ax.axhspan(
            median_start, median_stop, color="gold", alpha=0.28,
            label=f"Median gluing window {median_start:.2f}–{median_stop:.2f} km",
        )
    if np.isfinite(median_split):
        ax.axhline(
            median_split, color="black", linestyle="-.", linewidth=1.35,
            label=f"Median split {median_split:.2f} km",
        )

    finite_abs = np.abs(selected[np.isfinite(selected) & (selected != 0.0)])
    linthresh = (
        max(float(np.percentile(finite_abs, 10.0)), 1.0e-12)
        if finite_abs.size
        else 1.0e-3
    )
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_ylim(0.0, max_altitude_km)
    ax.set_xlabel("Range-corrected signal [a.u.]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Altitude (km a.g.l.)", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.38)

    attempted_count = int(np.count_nonzero(attempted))
    success_count = int(np.count_nonzero(success & attempted))
    fallback = ds_l2.get(
        "single_channel_fallback_flag", xr.zeros_like(ds_l2["gluing_success_flag"])
    )
    fallback_count = int(
        np.count_nonzero(np.asarray(fallback.sel(**sel).values, dtype=bool))
    )
    summary = [
        f"gluing success = {success_count}/{attempted_count}",
        f"single-channel fallback = {fallback_count}",
        f"correlation med = {_finite_median(corr):.4g}",
        f"relative RMSE med = {_finite_median(rmse):.4g}",
        f"relative bias med = {_finite_median(bias):.4g}",
        scaling_note,
    ]
    ax.text(
        0.02, 0.985, "\n".join(summary), transform=ax.transAxes, va="top",
        fontsize=8.9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.65"},
    )

    if np.isfinite(median_start) and np.isfinite(median_stop):
        zoom_min = max(0.0, median_start - 0.5)
        zoom_max = min(max_altitude_km, median_stop + 0.5)
        zoom = (altitude_km >= zoom_min) & (altitude_km <= zoom_max)
        if np.count_nonzero(zoom & np.isfinite(selected)) >= 3:
            inset = ax.inset_axes([0.08, 0.10, 0.39, 0.29])
            inset.plot(selected[zoom], altitude_km[zoom], color=color, linewidth=1.5)
            inset.axhspan(median_start, median_stop, color="gold", alpha=0.28)
            if np.isfinite(median_split):
                inset.axhline(
                    median_split, color="black", linestyle="-.", linewidth=1.0
                )
            inset.set_xscale("symlog", linthresh=linthresh)
            inset.set_ylim(zoom_min, zoom_max)
            inset.set_title("Selected RCS near gluing", fontsize=8)
            inset.tick_params(labelsize=7)
            inset.grid(True, which="both", alpha=0.3)

    ax.legend(fontsize=8.4, loc="center right")
    fig.suptitle(
        f"MILGRAU Level 2 — Signal Profile and Gluing — {format_wavelength_label(wavelength)}\n{date_title}",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    add_footer_and_logos(fig, root_dir)
    return _save(
        fig,
        output_folder,
        f"{file_name_prefix}_L2_Gluing_{wavelength}nm.{output_format}",
        dpi,
    )


def plot_all_level2_figures(
    ds_l2: xr.Dataset,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
    ds_l1: xr.Dataset | None = None,
) -> list[Path]:
    """Generate the canonical Level 2 scientific/diagnostic figures."""
    generated: list[Path] = []
    figure_cfg = config.get("visualization", {}).get("level2_figures", {}) or {}
    for wavelength in get_wavelength_values(ds_l2):
        if bool(figure_cfg.get("generate_gluing", True)):
            path = plot_gluing(
                ds_l2,
                wavelength,
                output_folder,
                file_name_prefix,
                config,
                root_dir,
                ds_l1=ds_l1,
            )
            if path is not None:
                generated.append(path)
        if bool(figure_cfg.get("generate_molecular_reference", True)):
            path = plot_molecular_reference(ds_l2, wavelength, output_folder, file_name_prefix, config, root_dir)
            if path is not None:
                generated.append(path)
        if bool(figure_cfg.get("generate_optical_profiles", True)):
            path = plot_optical_profiles(ds_l2, wavelength, output_folder, file_name_prefix, config, root_dir)
            if path is not None:
                generated.append(path)
        if bool(figure_cfg.get("generate_mc_reference", True)):
            path = plot_mc_reference(ds_l2, wavelength, output_folder, file_name_prefix, config, root_dir)
            if path is not None:
                generated.append(path)
    return generated
