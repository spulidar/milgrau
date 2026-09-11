"""Human-readable FAIR provenance helpers shared by MILGRAU products."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import netCDF4 as nc

from milgrau.version import __version__

PROVENANCE_ATTRS: tuple[str, ...] = (
    "software_name",
    "software_version",
    "processing_configuration_file",
    "station_configuration_file",
    "station_profile_id",
    "instrument_calibration_id",
)


def _source_path(config: Mapping[str, Any], key: str, label: str) -> Path | None:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Resolved {label} file no longer exists: {path}")
    return path


def configuration_provenance(config: Mapping[str, Any]) -> dict[str, str]:
    """Return concise provenance attributes intended to be read by humans."""
    attrs: dict[str, str] = {
        "software_name": "MILGRAU",
        "software_version": __version__,
    }
    config_path = _source_path(config, "_config_file", "processing configuration")
    if config_path is not None:
        attrs["processing_configuration_file"] = config_path.name
    station_path = _source_path(config, "_station_config_path", "station configuration")
    if station_path is not None:
        attrs["station_configuration_file"] = station_path.name

    resolved = config.get("_resolved_station")
    if isinstance(resolved, Mapping):
        if resolved.get("profile_id") is not None:
            attrs["station_profile_id"] = str(resolved["profile_id"])
        if resolved.get("calibration_id") is not None:
            attrs["instrument_calibration_id"] = str(resolved["calibration_id"])
    return attrs


def inherited_provenance(source_attrs: Mapping[str, Any]) -> dict[str, str]:
    """Copy stable human-readable MILGRAU provenance into a derived product."""
    result: dict[str, str] = {}
    for name in PROVENANCE_ATTRS:
        value = source_attrs.get(name)
        if value is not None and str(value).strip():
            result[name] = str(value)
    return result


def _write_yaml_variable(dataset: nc.Dataset, name: str, path: Path | None, description: str) -> None:
    if path is None:
        return
    text = path.read_text(encoding="utf-8")
    if name in dataset.variables:
        variable = dataset.variables[name]
    else:
        variable = dataset.createVariable(name, str)
    variable.assignValue(text)
    variable.setncattr("media_type", "application/yaml")
    variable.setncattr("description", description)
    variable.setncattr("source_filename", path.name)


def write_netcdf_provenance(
    path: str | Path,
    config: Mapping[str, Any],
    *,
    source_attrs: Mapping[str, Any] | None = None,
    extra_attrs: Mapping[str, str | int | float] | None = None,
) -> dict[str, str | int | float]:
    """Persist readable metadata and exact YAML recipes inside one NetCDF.

    SHA hashes and Git commit identifiers are deliberately not exposed in the
    scientific product. Reproducibility is provided by the release version,
    resolved station/calibration IDs, and exact YAML text used for processing.
    """
    attrs: dict[str, str | int | float] = inherited_provenance(source_attrs or {})
    attrs.update(configuration_provenance(config))
    if extra_attrs:
        for key, value in extra_attrs.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("NetCDF provenance attribute names must be non-empty strings.")
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise TypeError(f"NetCDF provenance attribute {key!r} must be a string or numeric scalar.")
            attrs[key.strip()] = value

    config_path = _source_path(config, "_config_file", "processing configuration")
    station_path = _source_path(config, "_station_config_path", "station configuration")
    with nc.Dataset(str(Path(path)), "a") as dataset:
        if attrs:
            dataset.setncatts(attrs)
        _write_yaml_variable(
            dataset,
            "processing_configuration_yaml",
            config_path,
            "Exact MILGRAU processing configuration used to generate this product.",
        )
        _write_yaml_variable(
            dataset,
            "station_configuration_yaml",
            station_path,
            "Exact station/instrument catalog used to resolve this product.",
        )
    return attrs
