"""Strict configuration helpers for Level 2 retrievals.

Level 2 scientific settings are intentionally fail-fast: productive retrievals
must never manufacture aerosol, gluing, Rayleigh-reference, cloud-screening, or
Monte Carlo settings when configuration is missing.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Mapping

import pandas as pd


class Level2ConfigurationError(ValueError):
    """Raised when productive Level 2 configuration is incomplete or invalid."""


def _required_mapping(parent: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    if key not in parent:
        raise Level2ConfigurationError(f"Missing required configuration: {path}.{key}")
    value = parent[key]
    if not isinstance(value, Mapping):
        raise Level2ConfigurationError(f"Configuration {path}.{key} must be a mapping.")
    return value


def _required_value(parent: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in parent:
        raise Level2ConfigurationError(f"Missing required configuration: {path}.{key}")
    return parent[key]


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise Level2ConfigurationError(f"Configuration {path} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise Level2ConfigurationError(f"Configuration {path} must be finite.")
    return result


def _positive_number(value: Any, path: str) -> float:
    result = _finite_number(value, path)
    if result <= 0.0:
        raise Level2ConfigurationError(f"Configuration {path} must be positive.")
    return result


def _fraction(value: Any, path: str) -> float:
    result = _finite_number(value, path)
    if result < 0.0 or result > 1.0:
        raise Level2ConfigurationError(f"Configuration {path} must be between 0 and 1.")
    return result


def _nonnegative_number(value: Any, path: str) -> float:
    result = _finite_number(value, path)
    if result < 0.0:
        raise Level2ConfigurationError(f"Configuration {path} must be non-negative.")
    return result


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise Level2ConfigurationError(f"Configuration {path} must be an integer.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise Level2ConfigurationError(f"Configuration {path} must be at least {minimum}.")
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise Level2ConfigurationError(f"Configuration {path} must be a boolean.")
    return value


def _inversion(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise Level2ConfigurationError("MILGRAU configuration must be a mapping.")
    return _required_mapping(config, "inversion", "config")


def incremental_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether incremental processing is enabled.

    Runtime strictness is handled in the runtime/configuration refactor batch;
    this helper intentionally preserves current behavior until that batch.
    """
    return bool(config.get("processing", {}).get("incremental", False))


def _parse_wavelengths(config: Mapping[str, Any]) -> list[int]:
    """Parse explicitly configured productive Level 2 wavelengths."""
    inv_cfg = _inversion(config)
    raw_values = _required_value(inv_cfg, "wavelengths_to_process", "inversion")
    if not isinstance(raw_values, list) or not raw_values:
        raise Level2ConfigurationError(
            "Configuration inversion.wavelengths_to_process must be a non-empty list of positive integer wavelengths."
        )

    wavelengths: list[int] = []
    for index, value in enumerate(raw_values):
        wavelength = _integer(value, f"inversion.wavelengths_to_process[{index}]", minimum=1)
        if wavelength in wavelengths:
            raise Level2ConfigurationError(
                f"Configuration inversion.wavelengths_to_process contains duplicate wavelength {wavelength}."
            )
        wavelengths.append(wavelength)
    return wavelengths


def get_wavelengths_to_process(config: Mapping[str, Any]) -> list[int]:
    """Return productive wavelengths after validating the complete Level 2 recipe."""
    validate_level2_config(config)
    return _parse_wavelengths(config)


def get_lidar_ratio(config: Mapping[str, Any], wavelength_nm: int, measurement_time: Any) -> tuple[float, float]:
    """Return the explicitly configured monthly aerosol lidar ratio and uncertainty."""
    inv_cfg = _inversion(config)
    month = pd.to_datetime(measurement_time).strftime("%m")
    wavelength_key = str(int(wavelength_nm))

    ratios = _required_mapping(inv_cfg, "lidar_ratios_sr", "inversion")
    if wavelength_key not in ratios:
        raise Level2ConfigurationError(
            f"Missing required configuration: inversion.lidar_ratios_sr.{wavelength_key}"
        )
    wavelength_ratios = ratios[wavelength_key]
    if not isinstance(wavelength_ratios, Mapping):
        raise Level2ConfigurationError(
            f"Configuration inversion.lidar_ratios_sr.{wavelength_key} must be a month-to-value mapping."
        )
    if month not in wavelength_ratios:
        raise Level2ConfigurationError(
            f"Missing required configuration: inversion.lidar_ratios_sr.{wavelength_key}.{month}"
        )
    lr_base = _positive_number(
        wavelength_ratios[month],
        f"inversion.lidar_ratios_sr.{wavelength_key}.{month}",
    )

    uncertainties = _required_mapping(inv_cfg, "lidar_ratio_std_sr", "inversion")
    if wavelength_key not in uncertainties:
        raise Level2ConfigurationError(
            f"Missing required configuration: inversion.lidar_ratio_std_sr.{wavelength_key}"
        )
    lr_std = _nonnegative_number(
        uncertainties[wavelength_key],
        f"inversion.lidar_ratio_std_sr.{wavelength_key}",
    )
    return lr_base, lr_std


def get_gluing_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return complete explicitly configured analog/PC gluing settings."""
    gluing_cfg = _required_mapping(_inversion(config), "gluing", "inversion")
    required = {
        "window_length_bins",
        "correlation_threshold",
        "search_min_idx",
        "search_max_idx",
        "intercept_threshold",
        "gaussian_threshold",
        "minmax_threshold",
        "max_relative_rmse",
        "max_relative_bias",
        "min_valid_fraction",
        "max_saturation_fraction",
        "invalid_saturation_fraction",
        "allow_single_channel_fallback",
        "single_channel_priority",
    }
    missing = sorted(required - set(gluing_cfg))
    if missing:
        raise Level2ConfigurationError(
            "Missing required Level 2 gluing configuration: " + ", ".join(f"inversion.gluing.{key}" for key in missing)
        )

    single_channel_priority = str(gluing_cfg["single_channel_priority"]).strip().lower()
    if single_channel_priority not in {"photon_counting", "analog"}:
        raise Level2ConfigurationError(
            "inversion.gluing.single_channel_priority must be 'photon_counting' or 'analog'."
        )

    window_size = _integer(gluing_cfg["window_length_bins"], "inversion.gluing.window_length_bins", minimum=4)
    search_min_idx = _integer(gluing_cfg["search_min_idx"], "inversion.gluing.search_min_idx", minimum=0)
    search_max_idx = _integer(gluing_cfg["search_max_idx"], "inversion.gluing.search_max_idx", minimum=1)
    if search_max_idx <= search_min_idx:
        raise Level2ConfigurationError("inversion.gluing.search_max_idx must exceed search_min_idx.")
    if search_max_idx - search_min_idx < window_size:
        raise Level2ConfigurationError(
            "Configured inversion.gluing search interval must be at least window_length_bins wide."
        )

    correlation_threshold = _finite_number(
        gluing_cfg["correlation_threshold"], "inversion.gluing.correlation_threshold"
    )
    if correlation_threshold < -1.0 or correlation_threshold > 1.0:
        raise Level2ConfigurationError("inversion.gluing.correlation_threshold must be between -1 and 1.")

    return {
        "window_size": window_size,
        "min_corr": correlation_threshold,
        "search_min_idx": search_min_idx,
        "search_max_idx": search_max_idx,
        "intercept_threshold": _nonnegative_number(
            gluing_cfg["intercept_threshold"], "inversion.gluing.intercept_threshold"
        ),
        "gaussian_threshold": _nonnegative_number(
            gluing_cfg["gaussian_threshold"], "inversion.gluing.gaussian_threshold"
        ),
        "minmax_threshold": _nonnegative_number(
            gluing_cfg["minmax_threshold"], "inversion.gluing.minmax_threshold"
        ),
        "max_relative_rmse": _nonnegative_number(
            gluing_cfg["max_relative_rmse"], "inversion.gluing.max_relative_rmse"
        ),
        "max_relative_bias": _nonnegative_number(
            gluing_cfg["max_relative_bias"], "inversion.gluing.max_relative_bias"
        ),
        "min_valid_fraction": _fraction(
            gluing_cfg["min_valid_fraction"], "inversion.gluing.min_valid_fraction"
        ),
        "max_saturation_fraction": _fraction(
            gluing_cfg["max_saturation_fraction"], "inversion.gluing.max_saturation_fraction"
        ),
        "invalid_saturation_fraction": _fraction(
            gluing_cfg["invalid_saturation_fraction"], "inversion.gluing.invalid_saturation_fraction"
        ),
        "allow_single_channel_fallback": _boolean(
            gluing_cfg["allow_single_channel_fallback"], "inversion.gluing.allow_single_channel_fallback"
        ),
        "single_channel_priority": single_channel_priority,
    }


def get_molecular_fit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return complete explicitly configured Rayleigh-reference settings."""
    fit_cfg = _required_mapping(_inversion(config), "molecular_fit", "inversion")
    required = {
        "ref_alt_min_m",
        "ref_alt_max_m",
        "ref_window_bins",
        "max_relative_slope",
        "max_relative_variance",
        "min_valid_fraction",
    }
    missing = sorted(required - set(fit_cfg))
    if missing:
        raise Level2ConfigurationError(
            "Missing required Rayleigh-reference configuration: "
            + ", ".join(f"inversion.molecular_fit.{key}" for key in missing)
        )

    ref_alt_min_m = _nonnegative_number(fit_cfg["ref_alt_min_m"], "inversion.molecular_fit.ref_alt_min_m")
    ref_alt_max_m = _positive_number(fit_cfg["ref_alt_max_m"], "inversion.molecular_fit.ref_alt_max_m")
    if ref_alt_max_m <= ref_alt_min_m:
        raise Level2ConfigurationError("inversion.molecular_fit.ref_alt_max_m must exceed ref_alt_min_m.")

    return {
        "ref_alt_min_m": ref_alt_min_m,
        "ref_alt_max_m": ref_alt_max_m,
        "ref_window_bins": _integer(fit_cfg["ref_window_bins"], "inversion.molecular_fit.ref_window_bins", minimum=3),
        "max_relative_slope": _nonnegative_number(
            fit_cfg["max_relative_slope"], "inversion.molecular_fit.max_relative_slope"
        ),
        "max_relative_variance": _nonnegative_number(
            fit_cfg["max_relative_variance"], "inversion.molecular_fit.max_relative_variance"
        ),
        "min_valid_fraction": _fraction(
            fit_cfg["min_valid_fraction"], "inversion.molecular_fit.min_valid_fraction"
        ),
    }


def get_cloud_screening_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return explicit cloud-screening policy and, when enabled, detector settings."""
    cloud_cfg = _required_mapping(_inversion(config), "cloud_screening", "inversion")
    enabled = _boolean(
        _required_value(cloud_cfg, "enabled", "inversion.cloud_screening"),
        "inversion.cloud_screening.enabled",
    )
    if not enabled:
        return {"enabled": False}

    required = {
        "min_altitude_m",
        "max_altitude_m",
        "smooth_bins",
        "baseline_percentile",
        "robust_z_threshold",
        "min_cloud_bins",
        "vertical_dilation_bins",
        "exclude_clouds_from_reference_fit",
    }
    missing = sorted(required - set(cloud_cfg))
    if missing:
        raise Level2ConfigurationError(
            "Missing required cloud-screening configuration: "
            + ", ".join(f"inversion.cloud_screening.{key}" for key in missing)
        )

    min_altitude_m = _nonnegative_number(
        cloud_cfg["min_altitude_m"], "inversion.cloud_screening.min_altitude_m"
    )
    max_altitude_m = _positive_number(
        cloud_cfg["max_altitude_m"], "inversion.cloud_screening.max_altitude_m"
    )
    if max_altitude_m <= min_altitude_m:
        raise Level2ConfigurationError(
            "inversion.cloud_screening.max_altitude_m must exceed min_altitude_m."
        )
    baseline_percentile = _finite_number(
        cloud_cfg["baseline_percentile"], "inversion.cloud_screening.baseline_percentile"
    )
    if baseline_percentile < 0.0 or baseline_percentile > 100.0:
        raise Level2ConfigurationError(
            "inversion.cloud_screening.baseline_percentile must be between 0 and 100."
        )

    return {
        "enabled": True,
        "min_altitude_m": min_altitude_m,
        "max_altitude_m": max_altitude_m,
        "smooth_bins": _integer(cloud_cfg["smooth_bins"], "inversion.cloud_screening.smooth_bins", minimum=1),
        "baseline_percentile": baseline_percentile,
        "robust_z_threshold": _positive_number(
            cloud_cfg["robust_z_threshold"], "inversion.cloud_screening.robust_z_threshold"
        ),
        "min_cloud_bins": _integer(
            cloud_cfg["min_cloud_bins"], "inversion.cloud_screening.min_cloud_bins", minimum=1
        ),
        "vertical_dilation_bins": _integer(
            cloud_cfg["vertical_dilation_bins"], "inversion.cloud_screening.vertical_dilation_bins", minimum=0
        ),
        "exclude_clouds_from_reference_fit": _boolean(
            cloud_cfg["exclude_clouds_from_reference_fit"],
            "inversion.cloud_screening.exclude_clouds_from_reference_fit",
        ),
    }


def get_kfs_mode(config: Mapping[str, Any]) -> str:
    """Require the approved productive KFS two-sided integration mode."""
    mode = str(_required_value(_inversion(config), "kfs_mode", "inversion")).strip().lower()
    if mode != "two_sided":
        raise Level2ConfigurationError("Level 2 KFS retrieval requires inversion.kfs_mode = 'two_sided'.")
    return "two_sided"


def get_kfs_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return complete explicitly configured KFS/Monte Carlo settings."""
    inv_cfg = _inversion(config)
    required = {
        "monte_carlo_iterations",
        "random_seed",
        "beta_ref_relative_std",
        "aerosol_ref_fraction",
        "min_lidar_ratio_sr",
        "allow_negative_aerosol",
    }
    missing = sorted(required - set(inv_cfg))
    if missing:
        raise Level2ConfigurationError(
            "Missing required KFS configuration: " + ", ".join(f"inversion.{key}" for key in missing)
        )
    return {
        "monte_carlo_iterations": _integer(
            inv_cfg["monte_carlo_iterations"], "inversion.monte_carlo_iterations", minimum=1
        ),
        "random_seed": _integer(inv_cfg["random_seed"], "inversion.random_seed", minimum=0),
        "beta_ref_relative_std": _nonnegative_number(
            inv_cfg["beta_ref_relative_std"], "inversion.beta_ref_relative_std"
        ),
        "aerosol_ref_fraction": _fraction(
            inv_cfg["aerosol_ref_fraction"], "inversion.aerosol_ref_fraction"
        ),
        "min_lidar_ratio_sr": _positive_number(
            inv_cfg["min_lidar_ratio_sr"], "inversion.min_lidar_ratio_sr"
        ),
        "allow_negative_aerosol": _boolean(
            inv_cfg["allow_negative_aerosol"], "inversion.allow_negative_aerosol"
        ),
        "kfs_mode": get_kfs_mode(config),
    }


def kfs_mode_description(mode: str) -> str:
    """Return a human-readable description of the KFS mode."""
    if mode != "two_sided":
        raise ValueError("The productive Level 2 integration mode must be 'two_sided'.")
    return "Backward below and forward above one shared reference bin; the forward branch is mathematically validated but remains noise-sensitive."


def get_block_average_minutes(config: Mapping[str, Any]) -> int:
    """Return explicitly configured temporal block size used by LEBEAR retrievals."""
    inv_cfg = _inversion(config)
    return _integer(
        _required_value(inv_cfg, "block_average_minutes", "inversion"),
        "inversion.block_average_minutes",
        minimum=1,
    )


def validate_level2_config(config: Mapping[str, Any]) -> None:
    """Validate every productive Level 2 scientific setting before retrieval starts."""
    wavelengths = _parse_wavelengths(config)
    get_block_average_minutes(config)
    get_kfs_config(config)
    get_gluing_config(config)
    get_molecular_fit_config(config)
    get_cloud_screening_config(config)

    # Require complete monthly LR climatology for every productive wavelength so
    # processing cannot become date-dependent on whether a missing month is hit.
    for wavelength in wavelengths:
        for month in range(1, 13):
            get_lidar_ratio(config, wavelength, f"2000-{month:02d}-15T00:00:00")
