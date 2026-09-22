"""Canonical path and identity builders for MILGRAU products."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CACHE_DIR = ".cache"
DEFAULT_SURFACE_WEATHER_CACHE_DIRNAME = "weather"
DEFAULT_RADIOSONDE_CACHE_DIRNAME = "radiosonde"

LEVEL0_SUFFIX = "_L0.nc"
LEVEL0_SCC_SUFFIX = "_L0_scc.nc"
LEVEL1_SUFFIX = "_L1.nc"
LEVEL1_SCC_SUFFIX = "_L1_scc.nc"
LEVEL2_SUFFIX = "_L2.nc"

LOCAL_PERIOD_STARTS = ("00", "06", "12", "18")
_STATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
MEASUREMENT_ID_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<station>[a-z0-9][a-z0-9-]*)_(?P<period>00|06|12|18)$",
    flags=re.IGNORECASE,
)
_PRODUCT_RE = re.compile(
    r"^(?P<measurement_id>\d{8}_[a-z0-9][a-z0-9-]*_(?:00|06|12|18))"
    r"(?P<suffix>_L0(?:_scc)?|_L1(?:_scc)?|(?:_[A-Za-z0-9_.-]+)?_L2(?:_scc)?)\.nc$",
    flags=re.IGNORECASE,
)


def project_root(root_dir: str | Path | None = None) -> Path:
    """Return the project root used to resolve relative configured paths."""
    return Path.cwd() if root_dir is None else Path(root_dir)


def resolve_project_path(path_value: str | Path, root_dir: str | Path | None = None) -> Path:
    """Resolve one configured path against the project root when relative."""
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else project_root(root_dir) / path


def _configured_directory(
    config: Mapping[str, Any],
    key: str,
    root_dir: str | Path | None = None,
) -> Path:
    directories = config.get("directories")
    if not isinstance(directories, Mapping):
        raise KeyError("Configuration directories section is required.")
    if key not in directories:
        raise KeyError(f"Configuration directories.{key} is required.")
    value = directories[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration directories.{key} must be a non-empty string.")
    return resolve_project_path(value.strip(), root_dir=root_dir)


def raw_data_root(config: Mapping[str, Any], root_dir: str | Path | None = None) -> Path:
    return _configured_directory(config, "raw_data", root_dir=root_dir)


def processed_data_root(config: Mapping[str, Any], root_dir: str | Path | None = None) -> Path:
    return _configured_directory(config, "processed_data", root_dir=root_dir)


def log_output_root(config: Mapping[str, Any], root_dir: str | Path | None = None) -> Path:
    return _configured_directory(config, "log_dir", root_dir=root_dir)


def surface_weather_cache_dir(
    config: Mapping[str, Any] | None = None,
    root_dir: str | Path | None = None,
) -> Path:
    if config:
        surface_weather = config.get("surface_weather", {})
        if isinstance(surface_weather, Mapping):
            cache_dir = surface_weather.get("cache_dir")
            if cache_dir:
                return resolve_project_path(str(cache_dir), root_dir=root_dir)
    return resolve_project_path(
        f"{DEFAULT_CACHE_DIR}/{DEFAULT_SURFACE_WEATHER_CACHE_DIRNAME}",
        root_dir=root_dir,
    )


def radiosonde_cache_dir(
    config: Mapping[str, Any] | None = None,
    root_dir: str | Path | None = None,
) -> Path:
    if config:
        radiosonde = config.get("radiosonde", {})
        if isinstance(radiosonde, Mapping):
            cache_dir = radiosonde.get("cache_dir")
            if cache_dir:
                return resolve_project_path(str(cache_dir), root_dir=root_dir)
    return resolve_project_path(
        f"{DEFAULT_CACHE_DIR}/{DEFAULT_RADIOSONDE_CACHE_DIRNAME}",
        root_dir=root_dir,
    )


def station_id(config: Mapping[str, Any]) -> str:
    """Return the canonical lowercase station identifier from station.yaml."""
    catalog = config.get("_station_catalog")
    if not isinstance(catalog, Mapping):
        raise KeyError("No station catalog is loaded; configure station_config in config.yaml.")
    station = catalog.get("station")
    if not isinstance(station, Mapping):
        raise KeyError("Station catalog must contain station metadata.")
    value = str(station.get("id", "")).strip().lower()
    if not _STATION_ID_RE.fullmatch(value):
        raise ValueError(f"station.id must be a lowercase filename-safe identifier; got {value!r}.")
    return value


def normalize_period_start(value: str | int) -> str:
    """Return one canonical local six-hour period start."""
    raw = str(value).strip()
    if not raw.isdigit():
        raise ValueError(f"Invalid local period start: {value!r}")
    normalized = f"{int(raw):02d}"
    if normalized not in LOCAL_PERIOD_STARTS:
        raise ValueError(
            f"Invalid local period start {value!r}; expected one of {', '.join(LOCAL_PERIOD_STARTS)}."
        )
    return normalized


def build_measurement_id(date_value: str, station: str, period_start: str | int) -> str:
    """Build YYYYMMDD_station_HH, where HH is the local six-hour period start."""
    date_text = str(date_value).strip()
    try:
        datetime.strptime(date_text, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"Invalid measurement date {date_value!r}; expected YYYYMMDD.") from exc
    station_text = str(station).strip().lower()
    if not _STATION_ID_RE.fullmatch(station_text):
        raise ValueError(f"Invalid station id: {station!r}")
    period = normalize_period_start(period_start)
    return f"{date_text}_{station_text}_{period}"


def measurement_id_parts(measurement_id: str) -> tuple[str, str, str]:
    """Return (YYYYMMDD, station, HH) for one canonical measurement ID."""
    value = str(measurement_id).strip().lower()
    match = MEASUREMENT_ID_RE.fullmatch(value)
    if match is None:
        raise ValueError(
            f"Invalid measurement_id {measurement_id!r}; expected YYYYMMDD_station_HH."
        )
    date_text = match.group("date")
    try:
        datetime.strptime(date_text, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"Invalid measurement date in {measurement_id!r}.") from exc
    return date_text, match.group("station"), match.group("period")


def is_measurement_id(value: str) -> bool:
    try:
        measurement_id_parts(value)
    except ValueError:
        return False
    return True


def validate_measurement_id_for_config(measurement_id: str, config: Mapping[str, Any]) -> str:
    """Validate a canonical ID and require its station to match station.yaml."""
    date_text, station, period = measurement_id_parts(measurement_id)
    expected_station = station_id(config)
    if station != expected_station:
        raise ValueError(
            f"Measurement ID station {station!r} does not match loaded station {expected_station!r}."
        )
    return build_measurement_id(date_text, station, period)


def measurement_day_dir(
    measurement_id: str,
    config: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> Path:
    """Return processed/station/YYYY/MM/YYYYMMDD for one canonical measurement."""
    value = validate_measurement_id_for_config(measurement_id, config)
    date_text, station, _period = measurement_id_parts(value)
    return (
        processed_data_root(config, root_dir=root_dir)
        / station
        / date_text[:4]
        / date_text[4:6]
        / date_text
    )


def product_measurement_id(product_path: str | Path) -> str:
    """Extract the canonical measurement ID from a MILGRAU product filename."""
    name = Path(product_path).name
    match = _PRODUCT_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"Unrecognized MILGRAU product filename: {name!r}")
    value = match.group("measurement_id").lower()
    measurement_id_parts(value)
    return value


def logging_measurement_id(product_path: str | Path) -> str:
    """Return canonical measurement ID for logging, or '-' for external inputs."""
    try:
        return product_measurement_id(product_path)
    except ValueError:
        return "-"


def level0_output_path(
    measurement_id: str,
    config: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> Path:
    value = validate_measurement_id_for_config(measurement_id, config)
    return measurement_day_dir(value, config, root_dir=root_dir) / f"{value}{LEVEL0_SUFFIX}"


def level0_scc_output_path(
    measurement_id: str,
    config: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> Path:
    value = validate_measurement_id_for_config(measurement_id, config)
    return measurement_day_dir(value, config, root_dir=root_dir) / f"{value}{LEVEL0_SCC_SUFFIX}"


def level1_output_path(
    level0_file: str | Path,
    config: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> Path:
    """Return the Level 1 path for canonical MILGRAU or explicit external Level 0 input."""
    source = Path(level0_file)
    name = source.name
    try:
        measurement_id = product_measurement_id(source)
    except ValueError:
        return source.with_name(f"{source.stem}{LEVEL1_SUFFIX}")

    if name.endswith(LEVEL0_SCC_SUFFIX):
        filename = f"{measurement_id}{LEVEL1_SCC_SUFFIX}"
    elif name.endswith(LEVEL0_SUFFIX):
        filename = f"{measurement_id}{LEVEL1_SUFFIX}"
    else:
        raise ValueError(f"Expected a Level 0 product, got {name!r}.")
    return measurement_day_dir(measurement_id, config, root_dir=root_dir) / filename


def level2_output_path(level1_file: str | Path, variant_tag: str | None = None) -> Path:
    """Return the Level 2 path for canonical MILGRAU or explicit external Level 1 input."""
    path = Path(level1_file)
    variant = ""
    if variant_tag:
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(variant_tag).strip()).strip("_")
        if safe_tag:
            variant = f"_{safe_tag}"

    try:
        measurement_id = product_measurement_id(path)
    except ValueError:
        stem = path.stem
        if stem.endswith("_L1"):
            stem = stem.removesuffix("_L1")
        return path.with_name(f"{stem}{variant}{LEVEL2_SUFFIX}")

    is_scc = path.name.endswith(LEVEL1_SCC_SUFFIX)
    if not is_scc and not path.name.endswith(LEVEL1_SUFFIX):
        raise ValueError(f"Expected a Level 1 file: {path}")
    scc_suffix = "_scc" if is_scc else ""
    return path.parent / f"{measurement_id}{variant}_L2{scc_suffix}.nc"


def quicklook_output_path(
    output_folder: str | Path,
    file_name_prefix: str,
    formatted_channel_name: str,
    max_altitude_km: float,
    output_format: str,
) -> Path:
    """Return an RCS quicklook path inside the day's quicklooks directory."""
    safe_channel = str(formatted_channel_name).replace(" ", "_")
    suffix = str(output_format).lstrip(".").lower()
    return (
        Path(output_folder)
        / f"rcs_{file_name_prefix}_{safe_channel}_{float(max_altitude_km):g}km.{suffix}"
    )


def global_mean_rcs_output_path(
    output_folder: str | Path,
    file_name_prefix: str,
    output_format: str,
) -> Path:
    suffix = str(output_format).lstrip(".").lower()
    return Path(output_folder) / f"rcs_{file_name_prefix}_mean.{suffix}"
