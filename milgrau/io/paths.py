"""Canonical path and session-identity builders for MILGRAU products."""

from __future__ import annotations

import re
from datetime import datetime, timezone
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

_STATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SESSION_ID_RE = re.compile(
    r"^(?P<station>[a-z0-9][a-z0-9-]*)_"
    r"(?P<start>\d{8}-\d{4}Z)_"
    r"(?P<end>\d{8}-\d{4}Z)$",
    flags=re.IGNORECASE,
)
_PRODUCT_RE = re.compile(
    r"^(?P<session_id>[a-z0-9][a-z0-9-]*_\d{8}-\d{4}Z_\d{8}-\d{4}Z)"
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


def _as_utc_datetime(value: Any, *, label: str) -> datetime:
    """Normalize a datetime-like value to an aware UTC datetime."""
    if isinstance(value, datetime):
        result = value
    else:
        text = str(value).strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            result = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"Invalid {label} {value!r}; expected an ISO-8601 datetime.") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware.")
    return result.astimezone(timezone.utc)


def _session_timestamp(value: Any, *, label: str) -> tuple[str, datetime]:
    exact = _as_utc_datetime(value, label=label)
    return exact.strftime("%Y%m%d-%H%MZ"), exact


def build_session_id(station: str, start_utc: Any, end_utc: Any) -> str:
    """Build station_YYYYMMDD-HHMMZ_YYYYMMDD-HHMMZ for one complete session."""
    station_text = str(station).strip().lower()
    if not _STATION_ID_RE.fullmatch(station_text):
        raise ValueError(f"Invalid station id: {station!r}")
    start_text, start_exact = _session_timestamp(start_utc, label="session start")
    end_text, end_exact = _session_timestamp(end_utc, label="session end")
    if end_exact <= start_exact:
        raise ValueError("Session end must be later than session start.")
    if end_text == start_text:
        raise ValueError(
            "Session start and end collapse to the same minute in the canonical ID; "
            "a session ID requires distinct start/end minutes."
        )
    return f"{station_text}_{start_text}_{end_text}"


def session_id_parts(session_id: str) -> tuple[str, datetime, datetime]:
    """Return (station, start_utc, end_utc) for one canonical session ID."""
    value = str(session_id).strip()
    match = SESSION_ID_RE.fullmatch(value)
    if match is None:
        raise ValueError(
            f"Invalid session_id {session_id!r}; expected "
            "station_YYYYMMDD-HHMMZ_YYYYMMDD-HHMMZ."
        )
    try:
        start = datetime.strptime(match.group("start").upper(), "%Y%m%d-%H%MZ").replace(
            tzinfo=timezone.utc
        )
        end = datetime.strptime(match.group("end").upper(), "%Y%m%d-%H%MZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"Invalid UTC timestamp in session_id {session_id!r}.") from exc
    if end <= start:
        raise ValueError(f"Invalid session_id {session_id!r}; end must be later than start.")
    return match.group("station").lower(), start, end


def is_session_id(value: str) -> bool:
    try:
        session_id_parts(value)
    except ValueError:
        return False
    return True


def validate_session_id_for_config(session_id: str, config: Mapping[str, Any]) -> str:
    """Validate a session ID and require its station to match station.yaml."""
    station, start, end = session_id_parts(session_id)
    expected_station = station_id(config)
    if station != expected_station:
        raise ValueError(
            f"Session ID station {station!r} does not match loaded station {expected_station!r}."
        )
    return build_session_id(station, start, end)


def session_dir(
    session_id: str,
    config: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> Path:
    """Return processed/station/YYYY/MM/session_id using the UTC session start."""
    value = validate_session_id_for_config(session_id, config)
    station, start, _end = session_id_parts(value)
    return (
        processed_data_root(config, root_dir=root_dir)
        / station
        / f"{start.year:04d}"
        / f"{start.month:02d}"
        / value
    )


def product_session_id(product_path: str | Path) -> str:
    """Extract the canonical session ID from a MILGRAU product filename."""
    name = Path(product_path).name
    match = _PRODUCT_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"Unrecognized MILGRAU product filename: {name!r}")
    station, start, end = session_id_parts(match.group("session_id"))
    return build_session_id(station, start, end)


def logging_session_id(product_path: str | Path) -> str:
    """Return canonical session ID for logging, or '-' for external inputs."""
    try:
        return product_session_id(product_path)
    except ValueError:
        return "-"


def level0_output_path(
    session_id: str,
    config: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> Path:
    value = validate_session_id_for_config(session_id, config)
    return session_dir(value, config, root_dir=root_dir) / f"{value}{LEVEL0_SUFFIX}"


def level0_scc_output_path(
    session_id: str,
    config: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> Path:
    value = validate_session_id_for_config(session_id, config)
    return session_dir(value, config, root_dir=root_dir) / f"{value}{LEVEL0_SCC_SUFFIX}"


def level1_output_path(
    level0_file: str | Path,
    config: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> Path:
    """Return Level 1 output for a canonical session or explicit external L0 input."""
    source = Path(level0_file)
    try:
        session_id = product_session_id(source)
    except ValueError:
        return source.with_name(f"{source.stem}{LEVEL1_SUFFIX}")

    if source.name.endswith(LEVEL0_SCC_SUFFIX):
        filename = f"{session_id}{LEVEL1_SCC_SUFFIX}"
    elif source.name.endswith(LEVEL0_SUFFIX):
        filename = f"{session_id}{LEVEL1_SUFFIX}"
    else:
        raise ValueError(f"Expected a Level 0 product, got {source.name!r}.")
    return session_dir(session_id, config, root_dir=root_dir) / filename


def level2_output_path(level1_file: str | Path, variant_tag: str | None = None) -> Path:
    """Return Level 2 output for a canonical session or explicit external L1 input."""
    path = Path(level1_file)
    variant = ""
    if variant_tag:
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(variant_tag).strip()).strip("_")
        if safe_tag:
            variant = f"_{safe_tag}"

    try:
        session_id = product_session_id(path)
    except ValueError:
        stem = path.stem.removesuffix("_L1")
        return path.with_name(f"{stem}{variant}{LEVEL2_SUFFIX}")

    is_scc = path.name.endswith(LEVEL1_SCC_SUFFIX)
    if not is_scc and not path.name.endswith(LEVEL1_SUFFIX):
        raise ValueError(f"Expected a Level 1 file: {path}")
    scc_suffix = "_scc" if is_scc else ""
    return path.parent / f"{session_id}{variant}_L2{scc_suffix}.nc"


def quicklook_output_path(
    output_folder: str | Path,
    file_name_prefix: str,
    formatted_channel_name: str,
    max_altitude_km: float,
    output_format: str,
) -> Path:
    """Return an RCS figure path."""
    safe_channel = str(formatted_channel_name).replace(" ", "_")
    suffix = str(output_format).lstrip(".").lower()
    return Path(output_folder) / (
        f"{file_name_prefix}_L1_RCS_{safe_channel}_{float(max_altitude_km):g}km.{suffix}"
    )


def global_mean_rcs_output_path(
    output_folder: str | Path,
    file_name_prefix: str,
    output_format: str,
) -> Path:
    suffix = str(output_format).lstrip(".").lower()
    return Path(output_folder) / f"{file_name_prefix}_L1_MeanRCS.{suffix}"
