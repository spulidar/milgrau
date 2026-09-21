"""Support-aware Level 2 QA panels.

The productive dataset intentionally retains diagnostic quantities such as
scattering ratio outside accepted backward KFS support. These plots make that
semantic boundary explicit without masking, filling or changing any product.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from milgrau.viz.level2_qa import (
    _block_standard_error,
    _robust_centered_xlim,
    _robust_positive_xlim,
    _smooth_for_plot,
    _uncertainty_exceeds_xlim,
    altitude_to_km,
    format_wavelength_label,
    get_wavelength_values,
    plot_all_level2_qa,
)
from milgrau.viz.quicklooks import extract_datetime_strings
from milgrau.viz.level2_qa_v5 import is_method_v5_dataset, plot_all_method_v5_qa
from milgrau.viz.style import add_footer_and_logos, channel_color, get_output_settings


def retrieval_top_altitude_km(ds_l2: xr.Dataset, wavelength_nm: int | float) -> float:
    """Return the aggregate algorithmic backward-inversion top when available.

    The named persisted top is authoritative. The support-flag fallback exists
    only for compatible products that expose the altitude-resolved support but
    not the scalar convenience field. This value is algorithmic support; it is
    not a validated near-range overlap or instrument-support statement.
    """
    wavelength = int(wavelength_nm)
    if "retrieval_top_altitude_m" in ds_l2:
        try:
            value = float(ds_l2["retrieval_top_altitude_m"].sel(wavelength=wavelength).values)
            if np.isfinite(value):
                return value / 1000.0
        except Exception:
            pass

    if "retrieval_inversion_support_flag" in ds_l2 and "altitude" in ds_l2.coords:
        try:
            support = np.asarray(
                ds_l2["retrieval_inversion_support_flag"].sel(wavelength=wavelength).values,
                dtype=bool,
            )
            altitude_km = altitude_to_km(ds_l2["altitude"].values)
            if support.shape == altitude_km.shape and np.any(support):
                return float(np.nanmax(altitude_km[support]))
        except Exception:
            pass
    return np.nan


def _mark_algorithmic_support_top(ax: Any, top_km: float, max_alt_km: float) -> bool:
    """Mark the optical-support top while preserving diagnostics above it."""
    if not np.isfinite(top_km) or top_km <= 0.0 or top_km > max_alt_km:
        return False
    ax.axhline(
        top_km,
        color="black",
        linestyle=":",
        linewidth=1.6,
        label=f"Algorithmic optical top {top_km:.2f} km",
    )
    if top_km < max_alt_km:
        ax.axhspan(
            top_km,
            max_alt_km,
            color="gray",
            alpha=0.08,
            zorder=0,
            label="Outside supported aerosol retrieval",
        )
    return True


def plot_qa_scattering_ratio_with_support(
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
) -> Path | None:
    """Plot scattering ratio while explicitly marking optical retrieval support."""
    wavelength = int(wavelength_nm)
    if "scattering_ratio_mean" not in ds_l2:
        return None

    output_format, dpi = get_output_settings(config)
    date_title, _ = extract_datetime_strings(ds_l2)
    altitude_km = altitude_to_km(ds_l2["altitude"].values)
    max_alt_km = min(30.0, float(np.nanmax(altitude_km)))
    valid_alt = altitude_km <= max_alt_km
    smooth_bins = int(config.get("visualization", {}).get("level2_qa", {}).get("smooth_bins", 15))
    sr = _smooth_for_plot(
        ds_l2["scattering_ratio_mean"].sel(wavelength=wavelength).values,
        smooth_bins,
    )
    color = channel_color(wavelength)

    sr_sigma = np.full_like(sr, np.nan, dtype=np.float64)
    uncertainty_label = "Block SEM"
    if "scattering_ratio_error_mean" in ds_l2:
        sr_sigma = _smooth_for_plot(
            ds_l2["scattering_ratio_error_mean"].sel(wavelength=wavelength).values,
            smooth_bins,
        )
        uncertainty_label = "SR 1σ"
    elif "scattering_ratio_block" in ds_l2:
        valid_block = None
        if "retrieval_success_flag" in ds_l2:
            try:
                valid_block = np.asarray(
                    ds_l2["retrieval_success_flag"].sel(wavelength=wavelength).values,
                    dtype=bool,
                )
            except Exception:
                valid_block = None
        sr_sigma = _smooth_for_plot(
            _block_standard_error(
                ds_l2["scattering_ratio_block"].sel(wavelength=wavelength).values,
                valid_block,
            ),
            smooth_bins,
        )

    fig, ax = plt.subplots(figsize=(8.6, 9.4))
    fig.subplots_adjust(top=0.86, bottom=0.14)
    has_uncertainty = np.isfinite(sr_sigma).any()
    if has_uncertainty:
        ax.fill_betweenx(
            altitude_km[valid_alt],
            sr[valid_alt] - sr_sigma[valid_alt],
            sr[valid_alt] + sr_sigma[valid_alt],
            color=color,
            alpha=0.22,
            edgecolor="none",
            label=uncertainty_label,
        )
    ax.plot(
        sr[valid_alt],
        altitude_km[valid_alt],
        color=color,
        linewidth=2.2,
        label="Scattering ratio (diagnostic)",
    )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.4, label="Molecular reference SR=1")

    top_km = retrieval_top_altitude_km(ds_l2, wavelength)
    has_top = _mark_algorithmic_support_top(ax, top_km, max_alt_km)

    ax.set_title(f"Scattering Ratio - {format_wavelength_label(wavelength)}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Scattering ratio", fontsize=12, fontweight="bold")
    ax.set_ylabel("Altitude (km a.g.l.)", fontsize=12, fontweight="bold")
    xlim = _robust_positive_xlim(sr[valid_alt], default_max=6.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, max_alt_km)
    ax.grid(True, alpha=0.45)

    notes = [f"Savgol plot smoothing = {smooth_bins} bins"]
    if has_top:
        notes.insert(0, "Values above the optical top are diagnostic, not aerosol retrieval")
        notes.insert(0, f"Algorithmic backward-inversion top = {top_km:.2f} km")
    else:
        notes.insert(0, "Optical retrieval top unavailable in this product")
    notes.append("Near-range instrument/overlap support is not validated by this marker")
    if has_uncertainty and _uncertainty_exceeds_xlim(sr[valid_alt], sr_sigma[valid_alt], xlim):
        notes.append("Uncertainty band clipped by robust x-axis")
    ax.text(
        0.04,
        0.96,
        "\n".join(notes),
        transform=ax.transAxes,
        fontsize=9.4,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "gray"},
    )
    ax.legend(fontsize=8.6, loc="best")

    fig.suptitle(
        f"MILGRAU Level 2 QA - Scattering Ratio / Support - {format_wavelength_label(wavelength)}\n{date_title}",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    add_footer_and_logos(fig, root_dir)
    out_path = Path(output_folder) / f"QA_ScatteringRatio_{file_name_prefix}_{wavelength}nm.{output_format}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_qa_l2_kfs_with_support(
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
    max_altitude_km: float = 30.0,
) -> Path | None:
    """Render KFS optical QA with the algorithmic retrieval top made explicit."""
    wavelength = int(wavelength_nm)
    required = {
        "aerosol_backscatter_mean",
        "aerosol_backscatter_mean_error",
        "aerosol_extinction_mean",
        "aerosol_extinction_mean_error",
    }
    if not required.issubset(set(ds_l2.data_vars)):
        return None

    output_format, dpi = get_output_settings(config)
    date_title, _ = extract_datetime_strings(ds_l2)
    altitude_km = altitude_to_km(ds_l2["altitude"].values)
    max_alt_km = min(float(max_altitude_km), float(np.nanmax(altitude_km)))
    valid_alt = altitude_km <= max_alt_km
    smooth_bins = int(config.get("visualization", {}).get("level2_qa", {}).get("smooth_bins", 15))

    beta_mean = _smooth_for_plot(
        np.asarray(ds_l2["aerosol_backscatter_mean"].sel(wavelength=wavelength).values, dtype=np.float64),
        smooth_bins,
    )
    beta_sigma = _smooth_for_plot(
        np.asarray(ds_l2["aerosol_backscatter_mean_error"].sel(wavelength=wavelength).values, dtype=np.float64),
        smooth_bins,
    )
    alpha_mean = _smooth_for_plot(
        np.asarray(ds_l2["aerosol_extinction_mean"].sel(wavelength=wavelength).values, dtype=np.float64),
        smooth_bins,
    )
    alpha_sigma = _smooth_for_plot(
        np.asarray(ds_l2["aerosol_extinction_mean_error"].sel(wavelength=wavelength).values, dtype=np.float64),
        smooth_bins,
    )

    beta_plot = beta_mean * 1e6
    beta_sigma_plot = beta_sigma * 1e6
    alpha_plot = alpha_mean * 1e6
    alpha_sigma_plot = alpha_sigma * 1e6
    beta_xlim = _robust_centered_xlim(beta_plot[valid_alt], default_abs=5.0)
    alpha_xlim = _robust_centered_xlim(alpha_plot[valid_alt], default_abs=50.0)
    beta_clipped = _uncertainty_exceeds_xlim(beta_plot[valid_alt], beta_sigma_plot[valid_alt], beta_xlim)
    alpha_clipped = _uncertainty_exceeds_xlim(alpha_plot[valid_alt], alpha_sigma_plot[valid_alt], alpha_xlim)

    color = channel_color(wavelength)
    fig = plt.figure(figsize=(13.5, 8.5))
    grid = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)
    ax_beta = plt.subplot(grid[0])
    ax_alpha = plt.subplot(grid[1], sharey=ax_beta)

    ax_beta.plot(beta_plot[valid_alt], altitude_km[valid_alt], color=color, linewidth=2.2, label="Mean beta aer")
    ax_beta.fill_betweenx(
        altitude_km[valid_alt],
        beta_plot[valid_alt] - beta_sigma_plot[valid_alt],
        beta_plot[valid_alt] + beta_sigma_plot[valid_alt],
        color=color,
        alpha=0.25,
        edgecolor="none",
        label="MC 1σ",
    )
    ax_beta.axvline(0.0, color="black", linewidth=0.8)
    ax_beta.set_xlim(*beta_xlim)
    ax_beta.set_title("Aerosol backscatter", fontsize=14, fontweight="bold")
    ax_beta.set_xlabel(r"$\beta_{aer}$ [Mm$^{-1}$ sr$^{-1}$]", fontsize=12, fontweight="bold")
    ax_beta.set_ylabel("Altitude (km a.g.l.)", fontsize=12, fontweight="bold")
    ax_beta.set_ylim(0, max_alt_km)
    ax_beta.grid(True, alpha=0.45)

    ax_alpha.plot(alpha_plot[valid_alt], altitude_km[valid_alt], color=color, linewidth=2.2, label="Mean alpha aer")
    ax_alpha.fill_betweenx(
        altitude_km[valid_alt],
        alpha_plot[valid_alt] - alpha_sigma_plot[valid_alt],
        alpha_plot[valid_alt] + alpha_sigma_plot[valid_alt],
        color=color,
        alpha=0.25,
        edgecolor="none",
        label="MC 1σ",
    )
    ax_alpha.axvline(0.0, color="black", linewidth=0.8)
    ax_alpha.set_xlim(*alpha_xlim)
    ax_alpha.set_title("Aerosol extinction", fontsize=14, fontweight="bold")
    ax_alpha.set_xlabel(r"$\alpha_{aer}$ [Mm$^{-1}$]", fontsize=12, fontweight="bold")
    ax_alpha.grid(True, alpha=0.45)
    plt.setp(ax_alpha.get_yticklabels(), visible=False)

    top_km = retrieval_top_altitude_km(ds_l2, wavelength)
    for axis in (ax_beta, ax_alpha):
        _mark_algorithmic_support_top(axis, top_km, max_alt_km)
        axis.legend(fontsize=8.6, loc="best")

    if beta_clipped:
        ax_beta.text(
            0.04,
            0.96,
            "MC 1σ clipped by robust x-axis",
            transform=ax_beta.transAxes,
            fontsize=9,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "gray"},
        )
    if alpha_clipped:
        ax_alpha.text(
            0.04,
            0.96,
            "MC 1σ clipped by robust x-axis",
            transform=ax_alpha.transAxes,
            fontsize=9,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "gray"},
        )

    fig.text(
        0.5,
        0.075,
        "Support marker = algorithmic backward-inversion support; it does not validate near-range overlap/instrument support.",
        ha="center",
        fontsize=9.0,
    )
    fig.suptitle(
        f"MILGRAU Level 2 QA - KFS Optical Retrieval / Support - {format_wavelength_label(wavelength)}\n{date_title}",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    fig.subplots_adjust(top=0.84, bottom=0.14)
    add_footer_and_logos(fig, root_dir)
    out_path = Path(output_folder) / f"QA_L2_KFS_{file_name_prefix}_{wavelength}nm.{output_format}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_all_level2_qa_with_support(
    ds_l2: xr.Dataset,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
    ds_l1: xr.Dataset | None = None,
) -> list[Path]:
    """Generate normal QA plus support-aware replacements for SR and KFS panels."""
    if is_method_v5_dataset(ds_l2):
        return plot_all_method_v5_qa(
            ds_l2=ds_l2,
            output_folder=output_folder,
            file_name_prefix=file_name_prefix,
            config=config,
            root_dir=root_dir,
            ds_l1=ds_l1,
        )

    requested_cfg = config.get("visualization", {}).get("level2_qa", {}) or {}
    generate_sr = bool(requested_cfg.get("generate_scattering_ratio_qa", True))
    generate_kfs = bool(requested_cfg.get("generate_kfs_qa", True))

    base_config = deepcopy(config)
    qa_cfg = base_config.setdefault("visualization", {}).setdefault("level2_qa", {})
    qa_cfg["generate_scattering_ratio_qa"] = False
    qa_cfg["generate_kfs_qa"] = False

    generated = plot_all_level2_qa(
        ds_l2=ds_l2,
        output_folder=output_folder,
        file_name_prefix=file_name_prefix,
        config=base_config,
        root_dir=root_dir,
        ds_l1=ds_l1,
    )
    for wavelength in get_wavelength_values(ds_l2):
        if generate_sr:
            path = plot_qa_scattering_ratio_with_support(
                ds_l2,
                wavelength,
                output_folder,
                file_name_prefix,
                config,
                root_dir,
            )
            if path is not None:
                generated.append(path)
        if generate_kfs:
            path = plot_qa_l2_kfs_with_support(
                ds_l2,
                wavelength,
                output_folder,
                file_name_prefix,
                config,
                root_dir,
            )
            if path is not None:
                generated.append(path)
    return generated
