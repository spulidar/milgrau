"""Configuration helpers for Level 2 retrievals."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def incremental_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether incremental processing is enabled."""
    return bool(config.get("processing", {}).get("incremental", False))


def get_wavelengths_to_process(config: Mapping[str, Any]) -> list[int]:
    """Return configured wavelengths for Level 2 processing."""
    raw_values = config.get("inversion", {}).get("wavelengths_to_process", [532])
    wavelengths: list[int] = []
    for value in raw_values:
        try:
            wavelength = int(value)
        except (TypeError, ValueError):
            continue
        if wavelength > 0 and wavelength not in wavelengths:
            wavelengths.append(wavelength)
    return wavelengths or [532]


def get_lidar_ratio(config: Mapping[str, Any], wavelength_nm: int, measurement_time: Any) -> tuple[float, float]:
    """Return monthly aerosol lidar ratio and standard deviation for one wavelength."""
    month = pd.to_datetime(measurement_time).strftime("%m")
    inv_cfg = config.get("inversion", {})
    wavelength_key = str(int(wavelength_nm))
    ratios = inv_cfg.get("lidar_ratios_sr", inv_cfg.get("lidar_ratios", {}))
    lr_base = float(ratios.get(wavelength_key, {}).get(month, 60.0))
    lr_std = float(inv_cfg.get("lidar_ratio_std_sr", {}).get(wavelength_key, 10.0))
    return lr_base, lr_std


def get_gluing_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return gluing configuration with safe defaults."""
    gluing_cfg = config.get("inversion", {}).get("gluing", {}) or {}
    single_channel_priority = str(
        gluing_cfg.get("single_channel_priority", "photon_counting")
    ).strip().lower()
    if single_channel_priority not in {"photon_counting", "analog"}:
        raise ValueError("inversion.gluing.single_channel_priority must be 'photon_counting' or 'analog'.")
    return {
        "window_size": int(gluing_cfg.get("window_length_bins", 150)),
        "min_corr": float(gluing_cfg.get("correlation_threshold", 0.95)),
        "search_min_idx": int(gluing_cfg.get("search_min_idx", 200)),
        "search_max_idx": int(gluing_cfg.get("search_max_idx", 2000)),
        "intercept_threshold": float(gluing_cfg.get("intercept_threshold", 5.0)),
        "gaussian_threshold": float(gluing_cfg.get("gaussian_threshold", 0.1)),
        "minmax_threshold": float(gluing_cfg.get("minmax_threshold", 0.05)),
        "max_relative_rmse": float(gluing_cfg.get("max_relative_rmse", 0.08)),
        "max_relative_bias": float(gluing_cfg.get("max_relative_bias", 0.05)),
        "min_valid_fraction": float(gluing_cfg.get("min_valid_fraction", 0.80)),
        "max_saturation_fraction": float(gluing_cfg.get("max_saturation_fraction", 0.20)),
        "invalid_saturation_fraction": float(gluing_cfg.get("invalid_saturation_fraction", 1.0)),
        "allow_single_channel_fallback": bool(gluing_cfg.get("allow_single_channel_fallback", True)),
        "single_channel_priority": single_channel_priority,
    }


def get_molecular_fit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return molecular reference configuration with safe defaults."""
    fit_cfg = config.get("inversion", {}).get("molecular_fit", {}) or {}
    return {
        "ref_alt_min_m": float(fit_cfg.get("ref_alt_min_m", 5000.0)),
        "ref_alt_max_m": float(fit_cfg.get("ref_alt_max_m", 25000.0)),
        "ref_window_bins": int(fit_cfg.get("ref_window_bins", 2667)),
        "max_relative_slope": float(fit_cfg.get("max_relative_slope", 0.25)),
        "max_relative_variance": float(fit_cfg.get("max_relative_variance", 0.50)),
        "min_valid_fraction": float(fit_cfg.get("min_valid_fraction", 0.50)),
    }


def get_kfs_mode(config: Mapping[str, Any]) -> str:
    """Require the approved productive KFS two-sided integration mode."""
    mode = str(config.get("inversion", {}).get("kfs_mode", "two_sided")).strip().lower()
    if mode != "two_sided":
        raise ValueError("Level 2 KFS retrieval requires inversion.kfs_mode = 'two_sided'.")
    return "two_sided"


def kfs_mode_description(mode: str) -> str:
    """Return a human-readable description of the KFS mode."""
    if mode != "two_sided":
        raise ValueError("The productive Level 2 integration mode must be 'two_sided'.")
    return "Backward below and forward above one shared reference bin; the forward branch is mathematically validated but remains noise-sensitive."


def get_block_average_minutes(config: Mapping[str, Any]) -> int:
    """Return temporal block size used by LEBEAR retrievals."""
    inv_cfg = config.get("inversion", {})
    return max(int(inv_cfg.get("block_average_minutes", inv_cfg.get("temporal_average_minutes", 15))), 1)
