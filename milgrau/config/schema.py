"""Validation and unknown-key detection for MILGRAU configuration."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

REQUIRED_TOP_LEVEL_SECTIONS = ("directories", "processing", "physics", "hardware")

_KNOWN_KEYS_BY_PATH: dict[tuple[str, ...], set[str]] = {
    (): {"project", "processing", "directories", "site", "location", "physics", "hardware", "radiosonde", "surface_weather", "meteorology", "visualization", "inversion"},
    ("project",): {"name", "full_name", "station_name", "institution", "timezone"},
    ("processing",): {
        "incremental", "interactive_qa", "console_level", "file_level", "laser_shot_tolerance_fraction",
        "dark_current_max_association_hours", "spurious_extensions", "quarantine_dir", "raw_scan_ignore_dirs",
        "max_workers_io", "max_workers_cpu", "quarantine_spurious_files", "delete_spurious_files",
    },
    ("directories",): {"raw_data", "processed_data", "site_output", "log_dir"},
    ("site",): {"latitude", "longitude", "station_altitude_m", "timezone"},
    ("location",): {"latitude", "longitude", "station_altitude_m", "timezone"},
    ("physics",): {
        "vertical_resolution_m", "speed_of_light_m_s", "speed_of_light", "default_surface_temp_c",
        "default_surface_pressure_hpa", "background_start_m", "background_stop_m", "bg_start", "bg_stop",
        "bg_start_m", "bg_stop_m", "pbl_min_search_m", "pbl_max_search_m", "pbl_smooth_bins", "channels",
        "latitude", "longitude", "station_altitude_m",
    },
    ("hardware",): {"name_to_id"},
    ("radiosonde",): {"station_id", "station_name", "fallback_to_standard_atmosphere", "fallback_to_standard", "cache_dir"},
    ("surface_weather",): {"provider", "cache_dir", "fallback_to_config_defaults"},
    ("meteorology",): {
        "acquisition_mode", "cache_directory", "allow_era5t", "timeout_seconds",
        "max_retries", "contract_version", "radiosonde", "era5",
    },
    ("meteorology", "radiosonde"): {"provider", "station_id"},
    ("meteorology", "era5"): {
        "dataset", "vertical_coordinate", "levels", "variables", "grid_degrees",
        "spatial_sampling", "temporal_interpolation", "raw_format",
    },
    ("visualization",): {"output_format", "dpi", "altitude_ranges_km", "channels_to_plot", "quicklook", "level2_qa"},
    ("visualization", "quicklook"): {
        "show_pbl", "show_tropopause", "mean_profile_smooth_bins", "max_time_gap_minutes", "missing_data_color", "colormap",
    },
    ("visualization", "level2_qa"): {
        "enabled", "max_altitude_km", "smooth_bins", "generate_gluing_qa", "generate_molecular_fit_qa",
        "generate_scattering_ratio_qa", "generate_kfs_qa",
    },
    ("inversion",): {
        "enabled", "interactive_qa", "wavelengths_to_process", "block_average_minutes", "temporal_average_minutes",
        "kfs_mode", "products", "monte_carlo_iterations", "random_seed", "beta_ref_relative_std",
        "aerosol_ref_fraction", "min_lidar_ratio_sr", "allow_negative_aerosol", "molecular_fit", "gluing",
        "cloud_screening", "lidar_ratio_std_sr", "lidar_ratios_sr", "lidar_ratios",
    },
    ("inversion", "products"): {
        "save_glued_signal", "save_molecular_profile", "save_scattering_ratio", "save_aerosol_backscatter",
        "save_aerosol_extinction", "save_uncertainty", "save_quality_flags",
    },
    ("inversion", "molecular_fit"): {
        "ref_alt_min_m", "ref_alt_max_m", "ref_window_bins", "lidar_ratio_molecular_sr", "lidar_ratio_molecular",
        "max_relative_slope", "max_relative_variance", "min_valid_fraction",
    },
    ("inversion", "gluing"): {
        "window_length_bins", "correlation_threshold", "intercept_threshold", "gaussian_threshold", "minmax_threshold",
        "max_relative_rmse", "max_relative_bias", "min_valid_fraction", "max_saturation_fraction",
        "invalid_saturation_fraction", "search_min_idx", "search_max_idx", "allow_single_channel_fallback",
        "single_channel_priority",
    },
    ("inversion", "cloud_screening"): {
        "enabled", "min_altitude_m", "max_altitude_m", "smooth_bins", "snr_threshold", "robust_z_threshold",
        "min_cloud_bins", "vertical_dilation_bins", "exclude_clouds_from_reference_fit",
        "stop_kfs_below_optically_thick_clouds",
    },
}


def find_unknown_config_keys(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return unknown keys from fixed-schema sections; dynamic channel/ratio maps stay open."""
    unknown: list[str] = []

    def visit(value: Mapping[str, Any], path: tuple[str, ...]) -> None:
        known = _KNOWN_KEYS_BY_PATH.get(path)
        if known is None:
            return
        for key, child in value.items():
            key_text = str(key)
            child_path = (*path, key_text)
            if key_text not in known:
                unknown.append(".".join(child_path))
                continue
            if isinstance(child, Mapping) and child_path in _KNOWN_KEYS_BY_PATH:
                visit(child, child_path)

    visit(config, ())
    return tuple(sorted(unknown))


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise KeyError(f"Configuration {key} section is required and must be a mapping.")
    return value


def _optional_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration {key} must be a mapping when provided.")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Configuration {label} must be a finite number; booleans are not numeric values.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"Configuration {label} must be finite; got {converted}.")
    return converted


def _require_positive_number(section: Mapping[str, Any], key: str, label: str) -> float:
    if key not in section:
        raise KeyError(f"Configuration {label}.{key} must be a positive finite number.")
    value = _finite_number(section[key], f"{label}.{key}")
    if value <= 0.0:
        raise ValueError(f"Configuration {label}.{key} must be positive; got {value}.")
    return value


def _optional_finite_number(
    section: Mapping[str, Any], key: str, label: str, *, minimum: float | None = None, positive: bool = False
) -> float | None:
    if key not in section:
        return None
    value = _finite_number(section[key], f"{label}.{key}")
    if positive and value <= 0.0:
        raise ValueError(f"Configuration {label}.{key} must be positive; got {value}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"Configuration {label}.{key} must be at least {minimum}; got {value}.")
    return value


def _optional_integer(
    section: Mapping[str, Any], key: str, label: str, *, minimum: int | None = None
) -> int | None:
    if key not in section:
        return None
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"Configuration {label}.{key} must be an integer; booleans are not integers.")
    converted = int(value)
    if minimum is not None and converted < minimum:
        raise ValueError(f"Configuration {label}.{key} must be at least {minimum}; got {converted}.")
    return converted


def _optional_boolean(section: Mapping[str, Any], key: str, label: str) -> None:
    if key in section and not isinstance(section[key], bool):
        raise ValueError(f"Configuration {label}.{key} must be a boolean when provided.")


def _optional_string(section: Mapping[str, Any], key: str, label: str) -> None:
    if key not in section:
        return
    value = section[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration {label}.{key} must be a non-empty string when provided.")


def _string_list(section: Mapping[str, Any], key: str, label: str) -> None:
    if key not in section:
        return
    value = section[key]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"Configuration {label}.{key} must be a list of non-empty strings.")


def _positive_number_list(section: Mapping[str, Any], key: str, label: str) -> None:
    if key not in section:
        return
    value = section[key]
    if not isinstance(value, list) or not value:
        raise ValueError(f"Configuration {label}.{key} must be a non-empty list of positive finite numbers.")
    for index, item in enumerate(value):
        number = _finite_number(item, f"{label}.{key}[{index}]")
        if number <= 0.0:
            raise ValueError(f"Configuration {label}.{key}[{index}] must be positive; got {number}.")


def _validate_channels(channels: Mapping[str, Any]) -> None:
    for channel, constants in channels.items():
        label = f"physics.channels.{channel}"
        if not isinstance(channel, str) or not channel.strip():
            raise ValueError("Configuration physics.channels keys must be non-empty strings.")
        if isinstance(constants, Mapping):
            required = {"deadtime_us", "bin_shift_bins", "background_offset"}
            missing = sorted(required - set(constants))
            unknown = sorted(set(constants) - required)
            if missing or unknown:
                raise ValueError(
                    f"Configuration {label} must contain exactly {sorted(required)}; missing={missing}, unknown={unknown}."
                )
            deadtime = constants["deadtime_us"]
            shift = constants["bin_shift_bins"]
            background = constants["background_offset"]
        elif isinstance(constants, Sequence) and not isinstance(constants, (str, bytes)) and len(constants) == 3:
            deadtime, shift, background = constants
        else:
            raise ValueError(
                f"Configuration {label} must use named fields deadtime_us, bin_shift_bins, and background_offset "
                "(legacy three-item lists remain temporarily accepted)."
            )
        _finite_number(deadtime, f"{label}.deadtime_us")
        if isinstance(shift, bool) or not isinstance(shift, Integral):
            raise ValueError(f"Configuration {label}.bin_shift_bins must be an integer.")
        _finite_number(background, f"{label}.background_offset")


def _validate_numeric_leaves(mapping: Mapping[str, Any], label: str, *, positive: bool = False, integer: bool = False) -> None:
    for key, value in mapping.items():
        child_label = f"{label}.{key}"
        if isinstance(value, Mapping):
            _validate_numeric_leaves(value, child_label, positive=positive, integer=integer)
            continue
        if integer:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"Configuration {child_label} must be an integer.")
            if positive and int(value) <= 0:
                raise ValueError(f"Configuration {child_label} must be positive.")
        else:
            number = _finite_number(value, child_label)
            if positive and number <= 0.0:
                raise ValueError(f"Configuration {child_label} must be positive.")


def _validate_visualization(config: Mapping[str, Any]) -> None:
    visualization = _optional_mapping(config, "visualization")
    for key in ("output_format",):
        _optional_string(visualization, key, "visualization")
    _optional_integer(visualization, "dpi", "visualization", minimum=1)
    _positive_number_list(visualization, "altitude_ranges_km", "visualization")
    _string_list(visualization, "channels_to_plot", "visualization")

    quicklook = visualization.get("quicklook", {})
    if not isinstance(quicklook, Mapping):
        raise ValueError("Configuration visualization.quicklook must be a mapping.")
    for key in ("show_pbl", "show_tropopause"):
        _optional_boolean(quicklook, key, "visualization.quicklook")
    for key in ("mean_profile_smooth_bins", "max_time_gap_minutes"):
        _optional_integer(quicklook, key, "visualization.quicklook", minimum=1)
    for key in ("missing_data_color", "colormap"):
        _optional_string(quicklook, key, "visualization.quicklook")

    qa = visualization.get("level2_qa", {})
    if not isinstance(qa, Mapping):
        raise ValueError("Configuration visualization.level2_qa must be a mapping.")
    for key in ("enabled", "generate_gluing_qa", "generate_molecular_fit_qa", "generate_scattering_ratio_qa", "generate_kfs_qa"):
        _optional_boolean(qa, key, "visualization.level2_qa")
    _optional_finite_number(qa, "max_altitude_km", "visualization.level2_qa", positive=True)
    _optional_integer(qa, "smooth_bins", "visualization.level2_qa", minimum=1)


def _validate_inversion(config: Mapping[str, Any]) -> None:
    inversion = _optional_mapping(config, "inversion")
    for key in ("enabled", "interactive_qa", "allow_negative_aerosol"):
        _optional_boolean(inversion, key, "inversion")
    _positive_number_list(inversion, "wavelengths_to_process", "inversion")
    for key in ("block_average_minutes", "temporal_average_minutes", "monte_carlo_iterations"):
        _optional_integer(inversion, key, "inversion", minimum=1)
    _optional_integer(inversion, "random_seed", "inversion", minimum=0)
    if "kfs_mode" in inversion and inversion["kfs_mode"] != "two_sided":
        raise ValueError("Configuration inversion.kfs_mode must be 'two_sided' for Level 2 processing.")
    for key in ("beta_ref_relative_std", "aerosol_ref_fraction"):
        _optional_finite_number(inversion, key, "inversion", minimum=0.0)
    _optional_finite_number(inversion, "min_lidar_ratio_sr", "inversion", positive=True)

    products = inversion.get("products", {})
    if not isinstance(products, Mapping):
        raise ValueError("Configuration inversion.products must be a mapping.")
    for key in _KNOWN_KEYS_BY_PATH[("inversion", "products")]:
        _optional_boolean(products, key, "inversion.products")

    molecular = inversion.get("molecular_fit", {})
    if not isinstance(molecular, Mapping):
        raise ValueError("Configuration inversion.molecular_fit must be a mapping.")
    for key in ("ref_alt_min_m", "ref_alt_max_m", "lidar_ratio_molecular_sr", "lidar_ratio_molecular"):
        _optional_finite_number(molecular, key, "inversion.molecular_fit", positive=True)
    _optional_integer(molecular, "ref_window_bins", "inversion.molecular_fit", minimum=1)
    for key in ("max_relative_slope", "max_relative_variance", "min_valid_fraction"):
        _optional_finite_number(molecular, key, "inversion.molecular_fit", minimum=0.0)
    if molecular.get("ref_alt_min_m") is not None and molecular.get("ref_alt_max_m") is not None:
        if float(molecular["ref_alt_max_m"]) <= float(molecular["ref_alt_min_m"]):
            raise ValueError("Configuration inversion.molecular_fit.ref_alt_max_m must exceed ref_alt_min_m.")

    gluing = inversion.get("gluing", {})
    if not isinstance(gluing, Mapping):
        raise ValueError("Configuration inversion.gluing must be a mapping.")
    for key in ("window_length_bins", "search_min_idx", "search_max_idx"):
        _optional_integer(gluing, key, "inversion.gluing", minimum=1)
    for key in (
        "correlation_threshold", "intercept_threshold", "gaussian_threshold", "minmax_threshold", "max_relative_rmse",
        "max_relative_bias", "min_valid_fraction", "max_saturation_fraction", "invalid_saturation_fraction",
    ):
        _optional_finite_number(gluing, key, "inversion.gluing", minimum=0.0)
    _optional_boolean(gluing, "allow_single_channel_fallback", "inversion.gluing")
    _optional_string(gluing, "single_channel_priority", "inversion.gluing")
    if gluing.get("single_channel_priority", "photon_counting") not in {"photon_counting", "analog"}:
        raise ValueError(
            "Configuration inversion.gluing.single_channel_priority must be 'photon_counting' or 'analog'."
        )
    if gluing.get("search_min_idx") is not None and gluing.get("search_max_idx") is not None:
        if int(gluing["search_max_idx"]) <= int(gluing["search_min_idx"]):
            raise ValueError("Configuration inversion.gluing.search_max_idx must exceed search_min_idx.")

    cloud = inversion.get("cloud_screening", {})
    if not isinstance(cloud, Mapping):
        raise ValueError("Configuration inversion.cloud_screening must be a mapping.")
    for key in ("enabled", "exclude_clouds_from_reference_fit", "stop_kfs_below_optically_thick_clouds"):
        _optional_boolean(cloud, key, "inversion.cloud_screening")
    for key in ("min_altitude_m", "max_altitude_m", "snr_threshold", "robust_z_threshold"):
        _optional_finite_number(cloud, key, "inversion.cloud_screening", minimum=0.0)
    for key in ("smooth_bins", "min_cloud_bins", "vertical_dilation_bins"):
        _optional_integer(cloud, key, "inversion.cloud_screening", minimum=0)
    if cloud.get("min_altitude_m") is not None and cloud.get("max_altitude_m") is not None:
        if float(cloud["max_altitude_m"]) <= float(cloud["min_altitude_m"]):
            raise ValueError("Configuration inversion.cloud_screening.max_altitude_m must exceed min_altitude_m.")

    for key in ("lidar_ratio_std_sr", "lidar_ratios_sr", "lidar_ratios"):
        values = inversion.get(key, {})
        if not isinstance(values, Mapping):
            raise ValueError(f"Configuration inversion.{key} must be a mapping.")
        _validate_numeric_leaves(values, f"inversion.{key}", positive=True)


def validate_config_minimum(config: Mapping[str, Any]) -> None:
    """Validate public structure and finite runtime values without activating dormant controls."""
    unknown = find_unknown_config_keys(config)
    if unknown:
        raise ValueError("Unknown configuration key(s): " + ", ".join(unknown))

    missing = [section for section in REQUIRED_TOP_LEVEL_SECTIONS if section not in config]
    if missing:
        raise KeyError("Configuration file is missing required section(s): " + ", ".join(missing))

    project = _optional_mapping(config, "project")
    for key in _KNOWN_KEYS_BY_PATH[("project",)]:
        _optional_string(project, key, "project")

    directories = _require_mapping(config, "directories")
    for key in ("raw_data", "processed_data", "log_dir"):
        value = directories.get(key)
        if not isinstance(value, str) or not value.strip():
            raise KeyError(f"Configuration directories.{key} is required and must be a non-empty string.")
    _optional_string(directories, "site_output", "directories")

    processing = _require_mapping(config, "processing")
    for removed_key in ("quarantine_spurious_files", "delete_spurious_files"):
        if removed_key in processing:
            raise ValueError(
                f"Configuration processing.{removed_key} was removed; raw discovery is read-only and filesystem actions must be explicit."
            )
    for key in ("incremental", "interactive_qa"):
        _optional_boolean(processing, key, "processing")
    for key in ("console_level", "file_level"):
        _optional_string(processing, key, "processing")
        if key in processing and not isinstance(getattr(logging, str(processing[key]).strip().upper(), None), int):
            raise ValueError(f"Configuration processing.{key} is not a recognized logging level: {processing[key]!r}.")
    _optional_string(processing, "quarantine_dir", "processing")
    _optional_finite_number(processing, "laser_shot_tolerance_fraction", "processing", minimum=0.0)
    _optional_finite_number(processing, "dark_current_max_association_hours", "processing", minimum=0.0)
    _string_list(processing, "raw_scan_ignore_dirs", "processing")
    _string_list(processing, "spurious_extensions", "processing")
    for key in ("max_workers_io", "max_workers_cpu"):
        _optional_integer(processing, key, "processing", minimum=1)

    site = _optional_mapping(config, "site")
    for key in ("latitude", "longitude", "station_altitude_m"):
        _optional_finite_number(site, key, "site")
    _optional_string(site, "timezone", "site")
    location = _optional_mapping(config, "location")
    for key in ("latitude", "longitude", "station_altitude_m"):
        _optional_finite_number(location, key, "location")
    _optional_string(location, "timezone", "location")

    physics = _require_mapping(config, "physics")
    _require_positive_number(physics, "vertical_resolution_m", "physics")
    for key in ("speed_of_light_m_s", "speed_of_light", "default_surface_pressure_hpa", "background_start_m", "background_stop_m", "bg_start", "bg_stop", "bg_start_m", "bg_stop_m", "pbl_min_search_m", "pbl_max_search_m"):
        _optional_finite_number(physics, key, "physics", positive=True)
    for key in ("default_surface_temp_c", "latitude", "longitude", "station_altitude_m"):
        _optional_finite_number(physics, key, "physics")
    _optional_integer(physics, "pbl_smooth_bins", "physics", minimum=1)
    channels = physics.get("channels")
    if not isinstance(channels, Mapping) or not channels:
        raise KeyError("Configuration physics.channels is required and must be a non-empty mapping.")
    _validate_channels(channels)
    if "background_start_m" in physics and "background_stop_m" in physics:
        if float(physics["background_stop_m"]) <= float(physics["background_start_m"]):
            raise ValueError("Configuration physics.background_stop_m must be greater than background_start_m.")
    if "pbl_min_search_m" in physics and "pbl_max_search_m" in physics:
        if float(physics["pbl_max_search_m"]) <= float(physics["pbl_min_search_m"]):
            raise ValueError("Configuration physics.pbl_max_search_m must be greater than pbl_min_search_m.")

    hardware = _require_mapping(config, "hardware")
    name_to_id = hardware.get("name_to_id")
    if not isinstance(name_to_id, Mapping) or not name_to_id:
        raise KeyError("Configuration hardware.name_to_id is required and must be a non-empty mapping.")
    _validate_numeric_leaves(name_to_id, "hardware.name_to_id", positive=True, integer=True)

    radiosonde = _optional_mapping(config, "radiosonde")
    for key in ("station_id", "cache_dir", "station_name"):
        _optional_string(radiosonde, key, "radiosonde")
    for key in ("fallback_to_standard_atmosphere", "fallback_to_standard"):
        _optional_boolean(radiosonde, key, "radiosonde")

    surface_weather = _optional_mapping(config, "surface_weather")
    for key in ("provider", "cache_dir"):
        _optional_string(surface_weather, key, "surface_weather")
    _optional_boolean(surface_weather, "fallback_to_config_defaults", "surface_weather")

    meteorology = _optional_mapping(config, "meteorology")
    if meteorology:
        _optional_string(meteorology, "acquisition_mode", "meteorology")
        if meteorology.get("acquisition_mode") not in {None, "auto", "cache_only", "prefetch"}:
            raise ValueError("Configuration meteorology.acquisition_mode must be auto, cache_only or prefetch.")
        _optional_string(meteorology, "cache_directory", "meteorology")
        _optional_string(meteorology, "contract_version", "meteorology")
        _optional_boolean(meteorology, "allow_era5t", "meteorology")
        _optional_finite_number(
            meteorology, "timeout_seconds", "meteorology", positive=True
        )
        _optional_integer(meteorology, "max_retries", "meteorology", minimum=1)
        meteorology_radiosonde = _optional_mapping(meteorology, "radiosonde")
        _optional_string(meteorology_radiosonde, "provider", "meteorology.radiosonde")
        _optional_string(meteorology_radiosonde, "station_id", "meteorology.radiosonde")
        if meteorology_radiosonde.get("provider") not in {None, "wyoming_siphon"}:
            raise ValueError("Configuration meteorology.radiosonde.provider must be wyoming_siphon.")
        if "station_id" in meteorology_radiosonde and not str(
            meteorology_radiosonde["station_id"]
        ).isdigit():
            raise ValueError("Configuration meteorology.radiosonde.station_id must contain only digits.")
        era5 = _optional_mapping(meteorology, "era5")
        for key in (
            "dataset",
            "vertical_coordinate",
            "levels",
            "spatial_sampling",
            "temporal_interpolation",
            "raw_format",
        ):
            _optional_string(era5, key, "meteorology.era5")
        _optional_finite_number(era5, "grid_degrees", "meteorology.era5", positive=True)
        expected_fixed = {
            "dataset": "reanalysis-era5-complete",
            "vertical_coordinate": "model_levels",
            "levels": "1-137",
            "spatial_sampling": "surrounding_four_points",
            "temporal_interpolation": "linear",
            "raw_format": "grib",
        }
        for key, expected in expected_fixed.items():
            if key in era5 and era5[key] != expected:
                raise ValueError(f"Configuration meteorology.era5.{key} must be {expected!r}.")
        if "variables" in era5:
            variables = era5["variables"]
            if not isinstance(variables, Sequence) or isinstance(variables, (str, bytes)):
                raise ValueError("Configuration meteorology.era5.variables must be a list.")
            if tuple(variables) != (
                "temperature",
                "specific_humidity",
                "lnsp",
                "surface_geopotential",
            ):
                raise ValueError("Configuration meteorology.era5.variables must match the fixed SCI-004B contract.")

    _validate_visualization(config)
    _validate_inversion(config)
