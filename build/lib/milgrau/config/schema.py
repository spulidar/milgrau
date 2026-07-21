"""Lightweight configuration schema validation for MILGRAU.

The project is still evolving scientifically, so this module intentionally uses
focused validation instead of a rigid schema. It catches missing sections,
obvious type errors and invalid numerical values while keeping experimental
Level 2 settings flexible.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

REQUIRED_TOP_LEVEL_SECTIONS = ("directories", "processing", "physics", "hardware")


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a required mapping section from the configuration."""
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise KeyError(f"Configuration {key} section is required and must be a mapping.")
    return value


def _require_positive_number(section: Mapping[str, Any], key: str, label: str) -> float:
    """Validate that one key exists and stores a positive finite number."""
    value = section.get(key)
    if not isinstance(value, Real):
        raise KeyError(f"Configuration {label}.{key} must be a positive number.")
    value_float = float(value)
    if value_float <= 0.0:
        raise ValueError(f"Configuration {label}.{key} must be positive; got {value_float}.")
    return value_float


def _optional_finite_number(section: Mapping[str, Any], key: str, label: str) -> None:
    """Validate that an optional key is numeric when present."""
    if key not in section:
        return
    value = section[key]
    if not isinstance(value, Real):
        raise ValueError(f"Configuration {label}.{key} must be numeric when provided.")


def _optional_boolean(section: Mapping[str, Any], key: str, label: str) -> None:
    """Validate that an optional key is boolean when present."""
    if key in section and not isinstance(section[key], bool):
        raise ValueError(f"Configuration {label}.{key} must be a boolean when provided.")


def _optional_string(section: Mapping[str, Any], key: str, label: str) -> None:
    """Validate that an optional key is a non-empty string when present."""
    if key not in section:
        return
    value = section[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration {label}.{key} must be a non-empty string when provided.")


def validate_config_minimum(config: Mapping[str, Any]) -> None:
    """Validate the minimum required MILGRAU configuration structure."""
    missing = [section for section in REQUIRED_TOP_LEVEL_SECTIONS if section not in config]
    if missing:
        raise KeyError("Configuration file is missing required section(s): " + ", ".join(missing))

    directories = _require_mapping(config, "directories")
    for key in ("raw_data", "processed_data", "log_dir"):
        value = directories.get(key)
        if not isinstance(value, str) or not value.strip():
            raise KeyError(f"Configuration directories.{key} is required and must be a non-empty string.")

    processing = _require_mapping(config, "processing")
    if "incremental" in processing and not isinstance(processing["incremental"], bool):
        raise ValueError("Configuration processing.incremental must be a boolean.")
    for key in ("interactive_qa", "quarantine_spurious_files", "delete_spurious_files"):
        _optional_boolean(processing, key, "processing")
    for key in ("console_level", "file_level", "quarantine_dir"):
        _optional_string(processing, key, "processing")
    if "raw_scan_ignore_dirs" in processing and not isinstance(processing["raw_scan_ignore_dirs"], list):
        raise ValueError("Configuration processing.raw_scan_ignore_dirs must be a list when provided.")
    if "spurious_extensions" in processing and not isinstance(processing["spurious_extensions"], list):
        raise ValueError("Configuration processing.spurious_extensions must be a list when provided.")

    physics = _require_mapping(config, "physics")
    _require_positive_number(physics, "vertical_resolution_m", "physics")
    channels = physics.get("channels")
    if not isinstance(channels, Mapping) or not channels:
        raise KeyError("Configuration physics.channels is required and must be a non-empty mapping.")

    if "background_start_m" in physics and "background_stop_m" in physics:
        background_start = _require_positive_number(physics, "background_start_m", "physics")
        background_stop = _require_positive_number(physics, "background_stop_m", "physics")
        if background_stop <= background_start:
            raise ValueError("Configuration physics.background_stop_m must be greater than background_start_m.")

    site = config.get("site", {})
    if isinstance(site, Mapping):
        _optional_finite_number(site, "latitude", "site")
        _optional_finite_number(site, "longitude", "site")
        _optional_finite_number(site, "station_altitude_m", "site")

    hardware = _require_mapping(config, "hardware")
    if "name_to_id" not in hardware:
        raise KeyError("Configuration hardware.name_to_id is required.")

    radiosonde = config.get("radiosonde", {})
    if isinstance(radiosonde, Mapping):
        for key in ("station_id", "cache_dir", "station_name"):
            _optional_string(radiosonde, key, "radiosonde")
        for key in ("fallback_to_standard_atmosphere", "fallback_to_standard"):
            _optional_boolean(radiosonde, key, "radiosonde")

    surface_weather = config.get("surface_weather", {})
    if isinstance(surface_weather, Mapping):
        for key in ("provider", "cache_dir"):
            _optional_string(surface_weather, key, "surface_weather")
        _optional_boolean(surface_weather, "fallback_to_config_defaults", "surface_weather")
