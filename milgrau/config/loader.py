"""Configuration loader for MILGRAU."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import warnings

import yaml

from milgrau.config.schema import validate_config_minimum


def _project_root() -> Path:
    """Return the repository root inferred from this module location."""
    return Path(__file__).resolve().parents[2]


def _resolve_config_path(config_path: str | Path) -> Path:
    """Resolve one config path from cwd first, then fall back to the project root."""
    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()

    project_relative = _project_root() / path
    if project_relative.exists():
        return project_relative.resolve()
    return path.resolve()


def _copy_if_missing(mapping: dict[str, Any], canonical_key: str, alias_key: str) -> None:
    """Populate an alias key from a canonical key when the alias is absent."""
    if canonical_key in mapping and alias_key not in mapping:
        mapping[alias_key] = mapping[canonical_key]


def _normalize_physics_config(config: dict[str, Any]) -> None:
    """Normalize physics keys while preserving compatibility aliases."""
    physics = config.setdefault("physics", {})
    site = config.setdefault("site", {})

    _copy_if_missing(physics, "speed_of_light_m_s", "speed_of_light")
    _copy_if_missing(physics, "background_start_m", "bg_start")
    _copy_if_missing(physics, "background_stop_m", "bg_stop")
    _copy_if_missing(physics, "background_start_m", "bg_start_m")
    _copy_if_missing(physics, "background_stop_m", "bg_stop_m")

    if "latitude" in site and "latitude" not in physics:
        physics["latitude"] = site["latitude"]
    if "longitude" in site and "longitude" not in physics:
        physics["longitude"] = site["longitude"]
    if "station_altitude_m" in site and "station_altitude_m" not in physics:
        physics["station_altitude_m"] = site["station_altitude_m"]

    channels = physics.get("channels", {})
    if isinstance(channels, dict):
        for channel_name, constants in list(channels.items()):
            if isinstance(constants, (list, tuple)) and len(constants) == 3:
                warnings.warn(
                    f"Positional constants for physics.channels.{channel_name} are deprecated and will be removed in 0.2.0; "
                    "use deadtime_us, bin_shift_bins, and background_offset fields.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                channels[channel_name] = {
                    "deadtime_us": constants[0],
                    "bin_shift_bins": constants[1],
                    "background_offset": constants[2],
                }


def _normalize_radiosonde_config(config: dict[str, Any]) -> None:
    """Normalize radiosonde fallback aliases."""
    radiosonde = config.setdefault("radiosonde", {})
    _copy_if_missing(radiosonde, "fallback_to_standard_atmosphere", "fallback_to_standard")


def _normalize_inversion_config(config: dict[str, Any]) -> None:
    """Normalize inversion aliases used by older code paths."""
    inversion = config.setdefault("inversion", {})
    _copy_if_missing(inversion, "lidar_ratios_sr", "lidar_ratios")

    molecular_fit = inversion.setdefault("molecular_fit", {})
    _copy_if_missing(molecular_fit, "lidar_ratio_molecular_sr", "lidar_ratio_molecular")


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized copy of the MILGRAU configuration.

    The repository ``config.yaml`` uses canonical, non-duplicated key names.  This
    function injects legacy aliases in memory so existing pipeline code remains
    compatible while modules are migrated gradually.
    """
    normalized = deepcopy(config)
    _normalize_physics_config(normalized)
    _normalize_radiosonde_config(normalized)
    _normalize_inversion_config(normalized)
    validate_config_minimum(normalized)
    return normalized


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load, normalize and validate the MILGRAU YAML configuration file."""
    path = _resolve_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Error parsing YAML configuration: {exc}") from exc

    if config is None:
        raise RuntimeError(f"Configuration file is empty: {path}")
    if not isinstance(config, dict):
        raise RuntimeError(f"Configuration root must be a mapping: {path}")
    try:
        return normalize_config(config)
    except Exception as exc:
        raise type(exc)(f"{exc} [config: {path}]") from exc
