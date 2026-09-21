"""Method-v5-native Level 2 QA panels.

These plots consume schema-4/method-v5 variables directly.  They do not create
compatibility aliases for older Level-2 schemas, because several variable names
also changed scientific meaning when progressive-grid and selection-aware MC
were promoted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from milgrau.viz.level2_qa import altitude_to_km, format_wavelength_label, get_wavelength_values
from milgrau.viz.quicklooks import extract_datetime_strings
from milgrau.viz.style import add_footer_and_logos, channel_color, get_output_settings


def is_method_v5_dataset(ds_l2: xr.Dataset) -> bool:
    """Return whether the dataset exposes the productive method-v5 contract."""
    required = {
        "range_corrected_signal_block",
        "aerosol_backscatter_nominal_block",
        "rayleigh_reference_altitude_m_block",
        "selected_reference_altitude_m_mc",
        "period_support_fraction",
    }
    return required.issubset(set(ds_l2.data_vars))


def _date_title(ds_l2: xr.Dataset) -> str:
    """Return a readable observation interval for method-v5 block products."""
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


def plot_v5_reference_qa(
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
) -> Path | None:
    """Plot selected-reference altitude and persisted Rayleigh diagnostics."""
    wavelength = int(wavelength_nm)
    required = {
        "rayleigh_reference_altitude_m_block",
        "contiguous_path_top_altitude_m_block",
        "rayleigh_reference_relative_slope_block",
        "rayleigh_reference_relative_variance_block",
        "rayleigh_reference_snr_median_block",
        "rayleigh_reference_tier_min_altitude_m_block",
        "rayleigh_reference_fallback_used_block",
        "selection_success_fraction_block",
    }
    if not required.issubset(set(ds_l2.data_vars)):
        return None

    output_format, dpi = get_output_settings(config)
    date_title = _date_title(ds_l2)
    x, labels = _block_x(ds_l2)
    sel = dict(wavelength=wavelength)
    ref = np.asarray(ds_l2["rayleigh_reference_altitude_m_block"].sel(**sel).values, dtype=float) / 1000.0
    path_top = np.asarray(ds_l2["contiguous_path_top_altitude_m_block"].sel(**sel).values, dtype=float) / 1000.0
    tier = np.asarray(ds_l2["rayleigh_reference_tier_min_altitude_m_block"].sel(**sel).values, dtype=float) / 1000.0
    slope = np.asarray(ds_l2["rayleigh_reference_relative_slope_block"].sel(**sel).values, dtype=float)
    variance = np.asarray(ds_l2["rayleigh_reference_relative_variance_block"].sel(**sel).values, dtype=float)
    snr = np.asarray(ds_l2["rayleigh_reference_snr_median_block"].sel(**sel).values, dtype=float)
    selection_fraction = np.asarray(ds_l2["selection_success_fraction_block"].sel(**sel).values, dtype=float)
    fallback = np.asarray(ds_l2["rayleigh_reference_fallback_used_block"].sel(**sel).values, dtype=float)

    fit_cfg = config.get("inversion", {}).get("molecular_fit", {}) or {}
    max_slope = float(fit_cfg.get("max_relative_slope", np.nan))
    max_variance = float(fit_cfg.get("max_relative_variance", np.nan))
    color = channel_color(wavelength)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True)
    ax = axes[0, 0]
    ax.plot(x, path_top, "o-", label="Continuous KFS path top")
    ax.plot(x, ref, "o-", color=color, label="Selected reference")
    ax.plot(x, tier, "--", label="Winning tier floor")
    fb = np.isfinite(fallback) & (fallback > 0)
    if np.any(fb):
        ax.scatter(x[fb], ref[fb], marker="v", s=70, label="Tier fallback")
    ax.set_ylabel("Altitude (km a.g.l.)")
    ax.set_title("Reference / admissible path")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, slope, "o-", label="Relative slope")
    ax.plot(x, variance, "o-", label="Relative variance")
    if np.isfinite(max_slope):
        ax.axhline(max_slope, linestyle=":", label=f"Slope limit {max_slope:g}")
    if np.isfinite(max_variance):
        ax.axhline(max_variance, linestyle="--", label=f"Variance limit {max_variance:g}")
    ax.set_ylabel("Dimensionless")
    ax.set_title("Rayleigh shape diagnostics")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(x, snr, "o-", color=color)
    ax.set_ylabel("Median SNR")
    ax.set_title("Reference-window propagated SNR (diagnostic)")
    ax.grid(True, alpha=0.35)

    ax = axes[1, 1]
    ax.plot(x, selection_fraction, "o-", color=color)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("MC selection success fraction")
    ax.set_title("Selection robustness")
    ax.grid(True, alpha=0.35)

    step = max(1, int(np.ceil(max(len(x), 1) / 12)))
    for ax in axes[1]:
        ax.set_xticks(x[::step], labels[::step], rotation=45, ha="right")
        ax.set_xlabel("Block UTC")
    fig.suptitle(
        f"MILGRAU Level 2 QA - Method v5 Reference - {format_wavelength_label(wavelength)}\n{date_title}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    add_footer_and_logos(fig, root_dir)
    return _save(
        fig,
        output_folder,
        f"QA_V5_Reference_{file_name_prefix}_{wavelength}nm.{output_format}",
        dpi,
    )


def plot_v5_kfs_support_qa(
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
) -> Path | None:
    """Plot block-resolved nominal KFS profiles plus period support fraction."""
    wavelength = int(wavelength_nm)
    required = {
        "aerosol_backscatter_nominal_block",
        "aerosol_extinction_nominal_block",
        "aerosol_backscatter_mean",
        "aerosol_extinction_mean",
        "period_support_fraction",
    }
    if not required.issubset(set(ds_l2.data_vars)):
        return None

    output_format, dpi = get_output_settings(config)
    date_title = _date_title(ds_l2)
    altitude_km = altitude_to_km(ds_l2["altitude"].values)
    valid_alt = altitude_km <= min(30.0, float(np.nanmax(altitude_km)))
    sel = dict(wavelength=wavelength)
    beta_blocks = np.asarray(ds_l2["aerosol_backscatter_nominal_block"].sel(**sel).values, dtype=float) * 1e6
    alpha_blocks = np.asarray(ds_l2["aerosol_extinction_nominal_block"].sel(**sel).values, dtype=float) * 1e6
    beta_mean = np.asarray(ds_l2["aerosol_backscatter_mean"].sel(**sel).values, dtype=float) * 1e6
    alpha_mean = np.asarray(ds_l2["aerosol_extinction_mean"].sel(**sel).values, dtype=float) * 1e6
    support = np.asarray(ds_l2["period_support_fraction"].sel(**sel).values, dtype=float)
    color = channel_color(wavelength)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 8.6), sharey=True)
    for row in beta_blocks:
        axes[0].plot(row[valid_alt], altitude_km[valid_alt], color="0.75", linewidth=0.7, alpha=0.45)
    axes[0].plot(beta_mean[valid_alt], altitude_km[valid_alt], color=color, linewidth=2.2, label="Period nominal mean")
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel(r"$\beta_{aer}$ [Mm$^{-1}$ sr$^{-1}$]")
    axes[0].set_ylabel("Altitude (km a.g.l.)")
    axes[0].set_title("Backscatter")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(fontsize=8)

    for row in alpha_blocks:
        axes[1].plot(row[valid_alt], altitude_km[valid_alt], color="0.75", linewidth=0.7, alpha=0.45)
    axes[1].plot(alpha_mean[valid_alt], altitude_km[valid_alt], color=color, linewidth=2.2, label="Period nominal mean")
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel(r"$\alpha_{aer}$ [Mm$^{-1}$]")
    axes[1].set_title("Extinction (conditional on LR)")
    axes[1].grid(True, alpha=0.35)

    axes[2].plot(support[valid_alt], altitude_km[valid_alt], color=color, linewidth=2.2)
    axes[2].set_xlim(-0.03, 1.03)
    axes[2].set_xlabel("Supported block fraction")
    axes[2].set_title("Period inversion support")
    axes[2].grid(True, alpha=0.35)

    top = float(ds_l2["retrieval_top_altitude_m"].sel(**sel).values) / 1000.0 if "retrieval_top_altitude_m" in ds_l2 else np.nan
    if np.isfinite(top):
        for ax in axes:
            ax.axhline(top, linestyle=":", color="black", linewidth=1.2)
    axes[0].set_ylim(0, min(30.0, float(np.nanmax(altitude_km))))
    fig.suptitle(
        f"MILGRAU Level 2 QA - Method v5 KFS / Support - {format_wavelength_label(wavelength)}\n{date_title}",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.075,
        "Thin gray lines = successful nominal block retrievals; support fraction is algorithmic inversion support, not overlap validation.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    add_footer_and_logos(fig, root_dir)
    return _save(
        fig,
        output_folder,
        f"QA_V5_KFS_{file_name_prefix}_{wavelength}nm.{output_format}",
        dpi,
    )


def plot_v5_mc_reference_qa(
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
        f"MILGRAU Level 2 QA - Method v5 MC Reference - {format_wavelength_label(wavelength)}\n{date_title}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    add_footer_and_logos(fig, root_dir)
    return _save(
        fig,
        output_folder,
        f"QA_V5_MCReference_{file_name_prefix}_{wavelength}nm.{output_format}",
        dpi,
    )


def plot_v5_gluing_qa(
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
) -> Path | None:
    """Plot method-v5 persisted gluing geometry and fit diagnostics."""
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
    x, labels = _block_x(ds_l2)
    sel = dict(wavelength=wavelength)

    attempted = np.asarray(ds_l2["gluing_attempted_flag"].sel(**sel).values, dtype=bool)
    if not np.any(attempted):
        return None
    success = np.asarray(ds_l2["gluing_success_flag"].sel(**sel).values, dtype=bool)
    start = np.asarray(ds_l2["gluing_start_altitude_m"].sel(**sel).values, dtype=float) / 1000.0
    split = np.asarray(ds_l2["gluing_split_altitude_m"].sel(**sel).values, dtype=float) / 1000.0
    stop = np.asarray(ds_l2["gluing_stop_altitude_m"].sel(**sel).values, dtype=float) / 1000.0
    corr = np.asarray(ds_l2["gluing_correlation"].sel(**sel).values, dtype=float)
    rmse = np.asarray(ds_l2["gluing_relative_rmse"].sel(**sel).values, dtype=float)
    bias = np.asarray(ds_l2["gluing_relative_bias"].sel(**sel).values, dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.4), sharex=True)
    axes[0].plot(x, start, "o-", label="Start")
    axes[0].plot(x, split, "o-", label="Split")
    axes[0].plot(x, stop, "o-", label="Stop")
    failed = attempted & ~success
    if np.any(failed):
        axes[0].scatter(x[failed], split[failed], marker="x", s=70, label="Failed gluing")
    axes[0].set_ylabel("Altitude (km a.g.l.)")
    axes[0].set_title("Gluing geometry")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(fontsize=8)

    axes[1].plot(x, corr, "o-", label="Correlation")
    axes[1].plot(x, rmse, "o-", label="Relative RMSE")
    axes[1].plot(x, np.abs(bias), "o-", label="|Relative bias|")
    axes[1].set_ylabel("Fit diagnostic")
    axes[1].set_title("Gluing fit quality")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(fontsize=8)
    step = max(1, int(np.ceil(max(len(x), 1) / 12)))
    axes[1].set_xticks(x[::step], labels[::step], rotation=45, ha="right")
    axes[1].set_xlabel("Block UTC")

    fig.suptitle(
        f"MILGRAU Level 2 QA - Method v5 Gluing - {format_wavelength_label(wavelength)}\n{date_title}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    add_footer_and_logos(fig, root_dir)
    return _save(
        fig,
        output_folder,
        f"QA_V5_Gluing_{file_name_prefix}_{wavelength}nm.{output_format}",
        dpi,
    )


def plot_all_method_v5_qa(
    ds_l2: xr.Dataset,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
    ds_l1: xr.Dataset | None = None,
) -> list[Path]:
    """Generate schema-4/method-v5-native QA panels."""
    del ds_l1
    generated: list[Path] = []
    qa_cfg = config.get("visualization", {}).get("level2_qa", {}) or {}
    for wavelength in get_wavelength_values(ds_l2):
        if bool(qa_cfg.get("generate_gluing_qa", True)):
            path = plot_v5_gluing_qa(ds_l2, wavelength, output_folder, file_name_prefix, config, root_dir)
            if path is not None:
                generated.append(path)
        if bool(qa_cfg.get("generate_molecular_fit_qa", True)):
            path = plot_v5_reference_qa(ds_l2, wavelength, output_folder, file_name_prefix, config, root_dir)
            if path is not None:
                generated.append(path)
        if bool(qa_cfg.get("generate_kfs_qa", True)):
            path = plot_v5_kfs_support_qa(ds_l2, wavelength, output_folder, file_name_prefix, config, root_dir)
            if path is not None:
                generated.append(path)
        if bool(qa_cfg.get("generate_scattering_ratio_qa", True)):
            path = plot_v5_mc_reference_qa(ds_l2, wavelength, output_folder, file_name_prefix, config, root_dir)
            if path is not None:
                generated.append(path)
    return generated
