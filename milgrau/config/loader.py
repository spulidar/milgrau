"""Configuration loader for MILGRAU."""

from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
from typing import Any, Mapping
import warnings

import yaml

from milgrau.config.schema import validate_config_minimum
from milgrau.config.station import merge_station_defaults, validate_station_config


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


def _resolve_station_path(station_path: str | Path, config_path: Path) -> Path:
    """Resolve station.yaml relative to config.yaml before repository fallback."""
    path = Path(station_path).expanduser()
    if path.is_absolute():
        return path
    config_relative = config_path.parent / path
    if config_relative.exists():
        return config_relative.resolve()
    return _resolve_config_path(path)


def _read_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Error parsing {label} YAML: {exc}") from exc
    if payload is None:
        raise RuntimeError(f"{label} file is empty: {path}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} root must be a mapping: {path}")
    return payload


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


def _station_lidar_ratio_climatology(catalog: Mapping[str, Any]) -> Mapping[str, Any] | None:
    station = catalog.get("station")
    if not isinstance(station, Mapping):
        return None
    value = station.get("lidar_ratio_climatology")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("station.lidar_ratio_climatology must be a mapping.")
    if set(value) != {"provenance", "std_sr", "monthly_sr"}:
        raise ValueError(
            "station.lidar_ratio_climatology must contain exactly provenance, std_sr, and monthly_sr."
        )
    provenance = value["provenance"]
    if not isinstance(provenance, Mapping) or not isinstance(provenance.get("source"), str) or not provenance["source"].strip():
        raise ValueError("station.lidar_ratio_climatology.provenance.source must be a non-empty string.")
    std = value["std_sr"]
    monthly = value["monthly_sr"]
    if not isinstance(std, Mapping) or not std or not isinstance(monthly, Mapping) or not monthly:
        raise ValueError("station lidar-ratio climatology std_sr and monthly_sr must be non-empty mappings.")
    for wavelength, raw_std in std.items():
        if isinstance(raw_std, bool):
            raise ValueError(f"station lidar-ratio std for {wavelength} must be numeric.")
        resolved_std = float(raw_std)
        if not math.isfinite(resolved_std) or resolved_std < 0.0:
            raise ValueError(f"station lidar-ratio std for {wavelength} must be finite and non-negative.")
    for wavelength, raw_months in monthly.items():
        if not isinstance(raw_months, Mapping):
            raise ValueError(f"station lidar-ratio monthly values for {wavelength} must be a mapping.")
        for month, raw_value in raw_months.items():
            if str(month) not in {f"{index:02d}" for index in range(1, 13)} or isinstance(raw_value, bool):
                raise ValueError(f"Invalid station lidar-ratio entry {wavelength}.{month}.")
            resolved = float(raw_value)
            if not math.isfinite(resolved) or resolved <= 0.0:
                raise ValueError(f"station lidar-ratio value {wavelength}.{month} must be finite and positive.")
    return value


def _apply_station_lidar_ratio_climatology(config: dict[str, Any], catalog: Mapping[str, Any]) -> None:
    """Materialize station LR climatology into the transitional Level 2 config view.

    station.yaml is authoritative when the climatology exists. If it is absent,
    explicit inversion.lidar_ratios_sr / lidar_ratio_std_sr values are left
    untouched as a compatibility fallback. If neither source exists, strict
    Level 2 validation fails before retrieval.
    """
    climatology = _station_lidar_ratio_climatology(catalog)
    if climatology is None:
        return
    inversion = config.setdefault("inversion", {})
    if not isinstance(inversion, dict):
        raise ValueError("Configuration inversion must be a mapping.")
    inversion["lidar_ratios_sr"] = deepcopy(dict(climatology["monthly_sr"]))
    inversion["lidar_ratio_std_sr"] = deepcopy(dict(climatology["std_sr"]))


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized copy of the MILGRAU configuration.

    ``config.yaml`` contains algorithm and processing controls. Station-specific
    fields may already have been merged from station.yaml before this function is
    called. Legacy aliases remain temporarily available outside the new
    stage-specific sections.

    ``level0`` and ``level1`` are preserved unchanged and intentionally excluded
    from the legacy broad-schema validator until that validator is retired.
    Productive LIBIDS/LIPANCORA validate these sections through their own strict
    stage resolvers before discovery/processing, keeping dependency direction
    from stage -> config rather than loader -> stage.
    """
    normalized = deepcopy(config)
    _normalize_physics_config(normalized)
    _normalize_radiosonde_config(normalized)
    _normalize_inversion_config(normalized)

    legacy_validation = deepcopy(normalized)
    legacy_validation.pop("level0", None)
    legacy_validation.pop("level1", None)
    validate_config_minimum(legacy_validation)
    return normalized


def load_config(
    config_path: str | Path = "config.yaml",
    station_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load algorithm config and optionally merge one station catalog.

    Station selection precedence is:
    1. explicit ``station_config_path`` argument;
    2. ``MILGRAU_STATION_CONFIG`` environment variable;
    3. ``station_config`` path declared in config.yaml.

    Resolved config paths are kept only in the runtime mapping so the simple
    incremental rule can invalidate products when either YAML file changes.
    """
    path = _resolve_config_path(config_path)
    config = _read_yaml_mapping(path, "Configuration")

    declared_station_path = config.pop("station_config", None)
    requested_station_path = (
        station_config_path
        if station_config_path is not None
        else os.environ.get("MILGRAU_STATION_CONFIG") or declared_station_path
    )

    station_catalog: dict[str, Any] | None = None
    station_path: Path | None = None
    if requested_station_path is not None:
        station_path = _resolve_station_path(requested_station_path, path)
        station_catalog = _read_yaml_mapping(station_path, "Station configuration")
        try:
            validate_station_config(station_catalog)
            config = merge_station_defaults(config, station_catalog)
            _apply_station_lidar_ratio_climatology(config, station_catalog)
        except Exception as exc:
            raise type(exc)(f"{exc} [station config: {station_path}]") from exc

    try:
        normalized = normalize_config(config)
    except Exception as exc:
        raise type(exc)(f"{exc} [config: {path}]") from exc

    normalized["_config_file"] = str(path)
    if station_catalog is not None:
        normalized["_station_catalog"] = deepcopy(station_catalog)
        normalized["_station_config_file"] = station_path.name if station_path is not None else "station.yaml"
        if station_path is not None:
            normalized["_station_config_path"] = str(station_path)
    return normalized
