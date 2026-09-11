"""Strict configuration resolution for productive Level 0 processing."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

import numpy as np


class Level0ConfigurationError(ValueError):
    """Raised when productive Level 0 configuration is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class DirectoriesConfig:
    raw_data: str
    processed_data: str
    log_dir: str


@dataclass(frozen=True, slots=True)
class RawDiscoveryConfig:
    spurious_extensions: tuple[str, ...]
    raw_scan_ignore_dirs: tuple[str, ...]
    quarantine_dir: str


@dataclass(frozen=True, slots=True)
class AcquisitionQaConfig:
    laser_shot_tolerance_fraction: float
    licel_header_time_jitter_s: float


@dataclass(frozen=True, slots=True)
class DarkCurrentConfig:
    max_association_hours: float


@dataclass(frozen=True, slots=True)
class SurfaceWeatherPolicy:
    missing_policy: str


@dataclass(frozen=True, slots=True)
class Level0Config:
    directories: DirectoriesConfig
    discovery: RawDiscoveryConfig
    acquisition_qa: AcquisitionQaConfig
    dark_current: DarkCurrentConfig
    surface_weather: SurfaceWeatherPolicy


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Level0ConfigurationError(f"Configuration {label} must be a mapping.")
    return value


def _exact_keys(section: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(section))
    unknown = sorted(set(section) - required)
    if missing or unknown:
        raise Level0ConfigurationError(
            f"Configuration {label} must contain exactly {sorted(required)}; "
            f"missing={missing}, unknown={unknown}."
        )


def _finite(value: Any, label: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise Level0ConfigurationError(f"Configuration {label} must be a finite number.")
    number = float(value)
    if not np.isfinite(number):
        raise Level0ConfigurationError(f"Configuration {label} must be finite.")
    if positive and number <= 0.0:
        raise Level0ConfigurationError(f"Configuration {label} must be positive.")
    if nonnegative and number < 0.0:
        raise Level0ConfigurationError(f"Configuration {label} must be non-negative.")
    return number


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Level0ConfigurationError(f"Configuration {label} must be a non-empty string.")
    return value.strip()


def _string_list(value: Any, label: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise Level0ConfigurationError(f"Configuration {label} must be a list of strings.")
    if not allow_empty and not value:
        raise Level0ConfigurationError(f"Configuration {label} must not be empty.")
    result: list[str] = []
    for index, item in enumerate(value):
        text = _text(item, f"{label}[{index}]")
        if text in result:
            raise Level0ConfigurationError(f"Configuration {label} contains duplicate value {text!r}.")
        result.append(text)
    return tuple(result)


def _resolve_directories(config: Mapping[str, Any]) -> DirectoriesConfig:
    directories = _mapping(config.get("directories"), "directories")
    required = {"raw_data", "processed_data", "log_dir"}
    missing = sorted(required - set(directories))
    if missing:
        raise Level0ConfigurationError(
            "Missing required Level 0 directories: " + ", ".join(f"directories.{key}" for key in missing)
        )
    return DirectoriesConfig(
        raw_data=_text(directories["raw_data"], "directories.raw_data"),
        processed_data=_text(directories["processed_data"], "directories.processed_data"),
        log_dir=_text(directories["log_dir"], "directories.log_dir"),
    )


def _resolve_discovery(config: Mapping[str, Any]) -> RawDiscoveryConfig:
    processing = _mapping(config.get("processing"), "processing")
    required = {"spurious_extensions", "raw_scan_ignore_dirs", "quarantine_dir"}
    missing = sorted(required - set(processing))
    if missing:
        raise Level0ConfigurationError(
            "Missing required raw-discovery configuration: "
            + ", ".join(f"processing.{key}" for key in missing)
        )

    extensions = _string_list(
        processing["spurious_extensions"],
        "processing.spurious_extensions",
        allow_empty=True,
    )
    normalized_extensions: list[str] = []
    for extension in extensions:
        normalized = extension.lower()
        if not normalized.startswith(".") or normalized == ".":
            raise Level0ConfigurationError(
                "Configuration processing.spurious_extensions entries must be file suffixes beginning with '.'."
            )
        if normalized in normalized_extensions:
            raise Level0ConfigurationError(
                f"Configuration processing.spurious_extensions contains duplicate suffix {normalized!r}."
            )
        normalized_extensions.append(normalized)

    ignored = _string_list(
        processing["raw_scan_ignore_dirs"],
        "processing.raw_scan_ignore_dirs",
        allow_empty=True,
    )
    if any("/" in name or "\\" in name for name in ignored):
        raise Level0ConfigurationError(
            "Configuration processing.raw_scan_ignore_dirs entries must be directory basenames, not paths."
        )

    return RawDiscoveryConfig(
        spurious_extensions=tuple(normalized_extensions),
        raw_scan_ignore_dirs=ignored,
        quarantine_dir=_text(processing["quarantine_dir"], "processing.quarantine_dir"),
    )


def resolve_level0_config(config: Mapping[str, Any]) -> Level0Config:
    """Resolve the complete productive LIBIDS policy without semantic defaults."""
    directories = _resolve_directories(config)
    discovery = _resolve_discovery(config)

    level0 = _mapping(config.get("level0"), "level0")
    _exact_keys(level0, {"acquisition_qa", "dark_current", "surface_weather"}, "level0")

    acquisition = _mapping(level0["acquisition_qa"], "level0.acquisition_qa")
    _exact_keys(
        acquisition,
        {"laser_shot_tolerance_fraction", "licel_header_time_jitter_s"},
        "level0.acquisition_qa",
    )
    shot_tolerance = _finite(
        acquisition["laser_shot_tolerance_fraction"],
        "level0.acquisition_qa.laser_shot_tolerance_fraction",
        nonnegative=True,
    )
    if shot_tolerance >= 1.0:
        raise Level0ConfigurationError(
            "Configuration level0.acquisition_qa.laser_shot_tolerance_fraction must be < 1."
        )
    header_jitter = _finite(
        acquisition["licel_header_time_jitter_s"],
        "level0.acquisition_qa.licel_header_time_jitter_s",
        nonnegative=True,
    )

    dark_current = _mapping(level0["dark_current"], "level0.dark_current")
    _exact_keys(dark_current, {"max_association_hours"}, "level0.dark_current")
    max_association_hours = _finite(
        dark_current["max_association_hours"],
        "level0.dark_current.max_association_hours",
        nonnegative=True,
    )

    weather = _mapping(level0["surface_weather"], "level0.surface_weather")
    _exact_keys(weather, {"missing_policy"}, "level0.surface_weather")
    policy = weather["missing_policy"]
    if not isinstance(policy, str) or policy.strip().lower() not in {"nan", "fail"}:
        raise Level0ConfigurationError(
            "Configuration level0.surface_weather.missing_policy must be 'nan' or 'fail'."
        )

    return Level0Config(
        directories=directories,
        discovery=discovery,
        acquisition_qa=AcquisitionQaConfig(shot_tolerance, header_jitter),
        dark_current=DarkCurrentConfig(max_association_hours),
        surface_weather=SurfaceWeatherPolicy(policy.strip().lower()),
    )


def validate_level0_config(config: Mapping[str, Any]) -> None:
    """Validate the productive Level 0 recipe before discovery/processing."""
    resolve_level0_config(config)


def station_timezone(config: Mapping[str, Any]) -> str:
    """Resolve station timezone only from the validated station catalog."""
    catalog = _mapping(config.get("_station_catalog"), "_station_catalog")
    station = _mapping(catalog.get("station"), "_station_catalog.station")
    timezone = station.get("timezone")
    if not isinstance(timezone, str) or not timezone.strip():
        raise Level0ConfigurationError("station.timezone must be a non-empty string.")
    return timezone.strip()


def station_coordinates(config: Mapping[str, Any]) -> tuple[float, float]:
    """Resolve station latitude/longitude only from the validated station catalog."""
    catalog = _mapping(config.get("_station_catalog"), "_station_catalog")
    station = _mapping(catalog.get("station"), "_station_catalog.station")
    site = _mapping(station.get("site"), "_station_catalog.station.site")
    latitude = _finite(site.get("latitude"), "station.site.latitude")
    longitude = _finite(site.get("longitude"), "station.site.longitude")
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise Level0ConfigurationError("Station latitude/longitude are outside valid bounds.")
    return latitude, longitude
