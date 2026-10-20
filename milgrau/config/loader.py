"""Configuration loader for MILGRAU."""

from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from milgrau.config.station import validate_station_config


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
    expected_months = {f"{index:02d}" for index in range(1, 13)}
    for wavelength, raw_months in monthly.items():
        if not isinstance(raw_months, Mapping):
            raise ValueError(f"station lidar-ratio monthly values for {wavelength} must be a mapping.")
        if set(map(str, raw_months)) != expected_months:
            raise ValueError(f"station lidar-ratio monthly values for {wavelength} must define all 12 months.")
        for month, raw_value in raw_months.items():
            if str(month) not in expected_months or isinstance(raw_value, bool):
                raise ValueError(f"Invalid station lidar-ratio entry {wavelength}.{month}.")
            resolved = float(raw_value)
            if not math.isfinite(resolved) or resolved <= 0.0:
                raise ValueError(f"station lidar-ratio value {wavelength}.{month} must be finite and positive.")
    if set(map(str, std)) != set(map(str, monthly)):
        raise ValueError("station lidar-ratio std_sr and monthly_sr must define the same wavelengths.")
    return value


def _apply_station_lidar_ratio_climatology(config: dict[str, Any], catalog: Mapping[str, Any]) -> None:
    """Materialize station-owned LR climatology into the Level 2 recipe view.

    This is not a legacy alias: the Level 2 strict resolver consumes the
    numerical monthly recipe from ``inversion`` while the authoritative source
    remains the station catalog and its provenance is recorded separately.
    """
    climatology = _station_lidar_ratio_climatology(catalog)
    if climatology is None:
        return
    inversion = config.setdefault("inversion", {})
    if not isinstance(inversion, dict):
        raise ValueError("Configuration inversion must be a mapping.")
    inversion["lidar_ratios_sr"] = deepcopy(dict(climatology["monthly_sr"]))
    inversion["lidar_ratio_std_sr"] = deepcopy(dict(climatology["std_sr"]))


def _scc_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the optional SCC feature switch without loading external metadata."""
    raw = config.get("scc")
    if raw is None:
        return {"enabled": False}
    if not isinstance(raw, Mapping):
        raise ValueError("config.scc must be a mapping.")
    allowed = {"enabled", "config_file", "state_dir", "upload"}
    if set(raw) - allowed:
        raise ValueError(f"Unknown config.scc keys: {sorted(set(raw) - allowed)}.")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("config.scc.enabled must be a boolean.")
    for name in ("config_file", "state_dir"):
        if name in raw and (not isinstance(raw[name], str) or not raw[name].strip()):
            raise ValueError(f"config.scc.{name} must be a non-empty path.")
    upload = raw.get("upload", {})
    if not isinstance(upload, Mapping) or set(upload) - {"automatic", "timeout_seconds", "retries"}:
        raise ValueError("config.scc.upload allows only automatic, timeout_seconds, retries.")
    if "automatic" in upload and not isinstance(upload["automatic"], bool):
        raise ValueError("config.scc.upload.automatic must be a boolean.")
    if "timeout_seconds" in upload and (
        isinstance(upload["timeout_seconds"], bool)
        or not isinstance(upload["timeout_seconds"], (float, int))
        or not math.isfinite(upload["timeout_seconds"])
        or upload["timeout_seconds"] <= 0
    ):
        raise ValueError("config.scc.upload.timeout_seconds must be finite and positive.")
    if "retries" in upload and (
        isinstance(upload["retries"], bool)
        or not isinstance(upload["retries"], int)
        or not 0 <= upload["retries"] <= 5
    ):
        raise ValueError("config.scc.upload.retries must be an integer 0..5.")
    if enabled and not raw.get("config_file"):
        raise ValueError("config.scc.config_file is required when SCC is enabled.")
    return raw


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a defensive copy without creating legacy compatibility aliases.

    Validation belongs to stage-specific strict resolvers. The loader only
    parses files, validates station-catalog structure, materializes the one
    intentional station-to-Level-2 recipe view, and records runtime source paths.
    """
    if not isinstance(config, dict):
        raise TypeError("MILGRAU configuration root must be a mapping.")
    return deepcopy(config)


def load_config(
    config_path: str | Path = "config.yaml",
    station_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the processing recipe plus an optional station catalog.

    Station selection precedence is:
    1. explicit ``station_config_path`` argument;
    2. ``MILGRAU_STATION_CONFIG`` environment variable;
    3. ``station_config`` path declared in config.yaml.

    Station metadata remains under ``_station_catalog``; the loader does not
    duplicate it into ``site``, ``radiosonde``, ``physics.channels`` or
    ``hardware.name_to_id`` compatibility structures.
    """
    path = _resolve_config_path(config_path)
    config = _read_yaml_mapping(path, "Configuration")
    scc_settings = _scc_settings(config)

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
            _apply_station_lidar_ratio_climatology(config, station_catalog)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{exc} [station config: {station_path}]") from exc

    normalized = normalize_config(config)
    normalized["_config_file"] = str(path)
    if scc_settings.get("enabled", False):
        if station_catalog is None:
            raise ValueError("SCC integration requires a loaded station.yaml.")
        # No SCC-specific imports, file reads or validation occur when disabled.
        from milgrau.config.scc import validate_scc_config

        scc_path = _resolve_station_path(str(scc_settings["config_file"]), path)
        scc_catalog = _read_yaml_mapping(scc_path, "SCC configuration")
        try:
            validate_scc_config(scc_catalog, station_catalog)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{exc} [SCC config: {scc_path}]") from exc
        normalized["_scc_catalog"] = deepcopy(scc_catalog)
        normalized["_scc_config_file"] = scc_path.name
        normalized["_scc_config_path"] = str(scc_path)
    if station_catalog is not None:
        normalized["_station_catalog"] = deepcopy(station_catalog)
        normalized["_station_config_file"] = station_path.name if station_path is not None else "station.yaml"
        if station_path is not None:
            normalized["_station_config_path"] = str(station_path)
    return normalized
