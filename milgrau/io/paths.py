"""Canonical path builders for MILGRAU products.

This module centralizes file-name and directory conventions so pipeline stages do
not need to duplicate product layout logic. All functions are intentionally
small and side-effect free; directory creation remains the responsibility of the
calling pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CACHE_DIR = ".cache"
DEFAULT_SURFACE_WEATHER_CACHE_DIRNAME = "weather"
DEFAULT_RADIOSONDE_CACHE_DIRNAME = "radiosonde"
LEVEL0_SUFFIX = ".nc"
LEVEL0_SCC_SUFFIX = "_scc.nc"
LEVEL1_SUFFIX = "_level1_rcs.nc"
LEVEL2_SUFFIX = "_level2_optical.nc"

MEASUREMENT_ID_RE = re.compile(r"^\d{8}\d{2}(?:\d{2})?z$")
SAVE_ID_RE = re.compile(r"^\d{8}sa\d{2}(?:\d{2})?z$")


def is_measurement_id(value: str) -> bool:
    return MEASUREMENT_ID_RE.fullmatch(str(value).strip().lower()) is not None


def is_save_id(value: str) -> bool:
    return SAVE_ID_RE.fullmatch(str(value).strip().lower()) is not None


def measurement_id_from_save_id(save_id: str) -> str:
    value = str(save_id).strip().lower()
    if not is_save_id(value):
        raise ValueError(f"Invalid save_id: {save_id!r}")
    return value[:8] + value[10:]


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
    """Return one explicitly configured directory; never invent a production path."""
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


def surface_weather_cache_dir(config: Mapping[str, Any] | None = None, root_dir: str | Path | None = None) -> Path:
    if config:
        surface_weather = config.get("surface_weather", {})
        if isinstance(surface_weather, Mapping):
            cache_dir = surface_weather.get("cache_dir")
            if cache_dir:
                return resolve_project_path(str(cache_dir), root_dir=root_dir)
    return resolve_project_path(f"{DEFAULT_CACHE_DIR}/{DEFAULT_SURFACE_WEATHER_CACHE_DIRNAME}", root_dir=root_dir)


def radiosonde_cache_dir(config: Mapping[str, Any] | None = None, root_dir: str | Path | None = None) -> Path:
    if config:
        radiosonde = config.get("radiosonde", {})
        if isinstance(radiosonde, Mapping):
            cache_dir = radiosonde.get("cache_dir")
            if cache_dir:
                return resolve_project_path(str(cache_dir), root_dir=root_dir)
    return resolve_project_path(f"{DEFAULT_CACHE_DIR}/{DEFAULT_RADIOSONDE_CACHE_DIRNAME}", root_dir=root_dir)


def measurement_save_id(measurement_id: str) -> str:
    """Return the canonical SCC-style MILGRAU save ID for a measurement group."""
    value = str(measurement_id).strip().lower()
    if not is_measurement_id(value):
        raise ValueError(f"Invalid measurement_id: {measurement_id!r}")
    return f"{value[:8]}sa{value[8:]}"


def product_save_id(product_path: str | Path) -> str:
    """Extract the canonical save ID from a MILGRAU Level 0/1/2 product path."""
    name = Path(product_path).name
    for suffix in (LEVEL2_SUFFIX, LEVEL1_SUFFIX, LEVEL0_SCC_SUFFIX, LEVEL0_SUFFIX):
        if name.endswith(suffix):
            stem = name.removesuffix(suffix)
            save_id = stem.split("_", 1)[0].lower()
            if is_save_id(save_id):
                return save_id
            raise ValueError(f"Product name does not contain a canonical save_id: {name!r}")
    raise ValueError(f"Unrecognized MILGRAU product filename: {name!r}")


def logging_save_id(product_path: str | Path) -> str:
    """Return canonical save ID for logging, or '-' for a non-canonical input name.

    Product validation remains strict through :func:`product_save_id`; this helper
    exists only so error reporting itself never masks the underlying pipeline error.
    """
    try:
        return product_save_id(product_path)
    except ValueError:
        return "-"


def measurement_product_dir(
    save_id: str,
    config: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> Path:
    save_id = str(save_id).strip().lower()
    if not is_save_id(save_id):
        raise ValueError(f"Invalid save_id: {save_id!r}")
    return processed_data_root(config, root_dir=root_dir) / save_id[:4] / save_id[4:6] / save_id


def level0_output_path(
    measurement_id: str,
    config: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> Path:
    save_id = measurement_save_id(measurement_id)
    return measurement_product_dir(save_id, config, root_dir=root_dir) / f"{save_id}{LEVEL0_SUFFIX}"


def level0_scc_output_path(
    measurement_id: str,
    config: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> Path:
    save_id = measurement_save_id(measurement_id)
    return measurement_product_dir(save_id, config, root_dir=root_dir) / f"{save_id}{LEVEL0_SCC_SUFFIX}"


def level1_output_path(
    level0_file: str | Path,
    config: Mapping[str, Any],
    root_dir: str | Path | None = None,
) -> Path:
    """Return a predictable Level 1 path for canonical or explicit external Level 0 input.

    Canonical MILGRAU inputs, including ``*_scc.nc``, stay inside the canonical
    measurement directory while preserving their input stem in the output name.
    An explicitly supplied non-canonical external Level 0/SCC file is written
    beside that source file rather than inventing a date/product tree from its
    filename.
    """
    source = Path(level0_file)
    stem = source.stem
    try:
        save_id = product_save_id(source)
    except ValueError:
        return source.with_name(f"{stem}{LEVEL1_SUFFIX}")
    return measurement_product_dir(save_id, config, root_dir=root_dir) / f"{stem}{LEVEL1_SUFFIX}"


def level2_output_path(level1_file: str | Path, variant_tag: str | None = None) -> Path:
    path = Path(level1_file)
    if not path.name.endswith(LEVEL1_SUFFIX):
        raise ValueError(f"Expected a Level 1 file ending with {LEVEL1_SUFFIX}: {path}")
    stem = path.name.removesuffix(LEVEL1_SUFFIX)
    if variant_tag:
        safe_tag = str(variant_tag).strip().replace(" ", "_")
        if safe_tag:
            stem = f"{stem}_{safe_tag}"
    return path.parent / f"{stem}{LEVEL2_SUFFIX}"


def quicklook_output_path(
    output_folder: str | Path,
    file_name_prefix: str,
    formatted_channel_name: str,
    max_altitude_km: float,
    output_format: str,
) -> Path:
    safe_channel = str(formatted_channel_name).replace(" ", "_")
    suffix = str(output_format).lstrip(".").lower()
    return Path(output_folder) / f"Quicklook_{file_name_prefix}_{safe_channel}_{float(max_altitude_km):g}km.{suffix}"


def global_mean_rcs_output_path(
    output_folder: str | Path,
    file_name_prefix: str,
    output_format: str,
) -> Path:
    suffix = str(output_format).lstrip(".").lower()
    return Path(output_folder) / f"GlobalMeanRCS_{file_name_prefix}.{suffix}"
