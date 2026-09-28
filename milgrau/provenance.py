"""Human-readable FAIR provenance helpers shared by MILGRAU products."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
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
LEGACY_HASH_ATTRS: tuple[str, ...] = (
    "processing_config_sha256",
    "station_config_sha256",
)
YAML_DOCUMENT_DIMENSION = "milgrau_provenance_document"
YAML_DOCUMENT_VARIABLES: tuple[str, str] = (
    "processing_configuration_yaml",
    "station_configuration_yaml",
)
SOURCE_REPOSITORY = "https://github.com/spulidar/milgrau"
SOURCE_CODE_IDENTITY_SCOPE = "installed_milgrau_python_sources_normalized_lf"


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 identity of one file's exact byte content."""
    source = Path(path)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_source_sha256(package_root: str | Path | None = None) -> str:
    """Return a portable content identity for the installed MILGRAU Python source tree.

    Relative POSIX paths and UTF-8 source text are hashed in sorted order. Line
    endings are normalized to LF so the identity is stable across equivalent
    Unix/Windows checkouts while still changing for edited Python source.
    """
    root = (
        Path(package_root).expanduser().resolve()
        if package_root is not None
        else Path(__file__).resolve().parent
    )
    source_files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not source_files:
        raise FileNotFoundError(f"No Python sources found below {root}")

    digest = hashlib.sha256()
    for path in source_files:
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _repository_revision(package_root: Path) -> tuple[str, str] | None:
    """Resolve an optional repository/build revision without making Git mandatory."""
    explicit = os.environ.get("MILGRAU_SOURCE_REVISION", "").strip()
    if explicit:
        return explicit, "environment"

    repository_root = package_root.parent
    if not (repository_root / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return (revision, "git") if revision else None


def source_code_provenance(
    *,
    package_root: str | Path | None = None,
) -> dict[str, str]:
    """Return a code-state identity that remains meaningful outside Git checkouts.

    ``source_code_sha256`` is always derived from the installed Python source
    content, so development states sharing one package CalVer remain distinct.
    A Git/build revision is added when supplied through ``MILGRAU_SOURCE_REVISION``
    or discoverable from a source checkout; released wheels need not depend on a
    local ``.git`` directory for their primary content identity.
    """
    root = (
        Path(package_root).expanduser().resolve()
        if package_root is not None
        else Path(__file__).resolve().parent
    )
    source_hash = package_source_sha256(root)
    attrs = {
        "source_repository": SOURCE_REPOSITORY,
        "source_code_sha256": source_hash,
        "source_code_identity": f"sha256:{source_hash}",
        "source_code_identity_scope": SOURCE_CODE_IDENTITY_SCOPE,
    }
    revision = _repository_revision(root)
    if revision is not None:
        attrs["source_repository_revision"] = revision[0]
        attrs["source_repository_revision_source"] = revision[1]
    return attrs


def _source_path(config: Mapping[str, Any], key: str, label: str) -> Path | None:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Resolved {label} file no longer exists: {path}")
    return path


def _nonempty_source_text(source_attrs: Mapping[str, Any], *names: str) -> str | None:
    """Return the first non-empty textual source attribute from ``names``."""
    for name in names:
        value = source_attrs.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _station_lineage_provenance(
    config: Mapping[str, Any],
    source_attrs: Mapping[str, Any],
) -> dict[str, str]:
    """Return normalized station-profile/calibration lineage without date inference.

    Newer processing contexts may provide ``_resolved_station`` directly. Older
    MILGRAU products commonly expose the selected profile as ``Station_Profile``
    but not as the normalized FAIR attribute ``station_profile_id``. For a
    derived product, that explicit source profile may be normalized and its
    calibration identifier may be read from the currently loaded station
    catalog when the profile id is present there. No profile is selected from a
    measurement date in this provenance helper.
    """
    result: dict[str, str] = {}
    resolved = config.get("_resolved_station")
    if isinstance(resolved, Mapping):
        profile_id = _nonempty_source_text(resolved, "profile_id")
        calibration_id = _nonempty_source_text(resolved, "calibration_id")
    else:
        profile_id = _nonempty_source_text(
            source_attrs,
            "station_profile_id",
            "Station_Profile",
        )
        calibration_id = _nonempty_source_text(
            source_attrs,
            "instrument_calibration_id",
        )

    if profile_id is not None:
        result["station_profile_id"] = profile_id
    if calibration_id is not None:
        result["instrument_calibration_id"] = calibration_id
        return result

    catalog = config.get("_station_catalog")
    if profile_id is None or not isinstance(catalog, Mapping):
        return result
    profiles = catalog.get("profiles")
    if not isinstance(profiles, list):
        return result
    matches = [
        profile
        for profile in profiles
        if isinstance(profile, Mapping) and str(profile.get("id", "")).strip() == profile_id
    ]
    if len(matches) == 1:
        catalog_calibration = _nonempty_source_text(matches[0], "calibration_id")
        if catalog_calibration is not None:
            result["instrument_calibration_id"] = catalog_calibration
    return result


def configuration_provenance(
    config: Mapping[str, Any],
    *,
    source_attrs: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Return concise provenance attributes intended to be read by humans and tools."""
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

    attrs.update(_station_lineage_provenance(config, source_attrs or {}))
    return attrs


def inherited_provenance(source_attrs: Mapping[str, Any]) -> dict[str, str]:
    """Copy stable human-readable MILGRAU provenance into a derived product."""
    result: dict[str, str] = {}
    for name in PROVENANCE_ATTRS:
        value = source_attrs.get(name)
        if value is not None and str(value).strip():
            result[name] = str(value)
    return result


def _era5_dataset_from_config(config: Mapping[str, Any]) -> str:
    """Return the explicitly configured ERA5 dataset identifier when available."""
    level1 = config.get("level1")
    if not isinstance(level1, Mapping):
        return ""
    atmosphere = level1.get("atmosphere")
    if not isinstance(atmosphere, Mapping):
        return ""
    era5 = atmosphere.get("era5")
    if not isinstance(era5, Mapping):
        return ""
    dataset = era5.get("dataset")
    return str(dataset).strip() if isinstance(dataset, str) else ""


def thermodynamic_source_provenance(
    source_attrs: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Return a portable provider/product/release identity for the Level 1 atmosphere.

    The identity deliberately excludes cache filenames, download timestamps and
    host paths. ERA5 uses the configured CDS dataset plus its dataset DOI when
    available; radiosonde records the stable Wyoming upper-air service family;
    USSA76 records the standard-atmosphere edition. Source-specific timestamps,
    station identifiers and fallback fractions remain separate attributes.
    """
    source_type = str(source_attrs.get("thermodynamic_profile_source_type", "")).strip().lower()
    if not source_type:
        return {}

    doi = str(source_attrs.get("thermodynamic_profile_doi", "")).strip()
    if source_type == "era5":
        provider = "copernicus_climate_change_service"
        product = _era5_dataset_from_config(config) or "era5_pressure_levels"
        release = f"doi:{doi}" if doi else "dataset_family_unversioned"
    elif source_type == "radiosonde":
        provider = "university_of_wyoming"
        product = "upper_air_sounding"
        release = "service_unversioned"
    elif source_type == "ussa76":
        provider = "us_standard_atmosphere"
        product = "standard_atmosphere"
        release = "1976"
    else:
        return {}

    result = {
        "thermodynamic_profile_provider": provider,
        "thermodynamic_profile_product": product,
        "thermodynamic_profile_version_or_release": release,
        "thermodynamic_profile_source_id": f"{provider}/{product}/{release}",
    }
    if doi:
        result["thermodynamic_profile_doi"] = doi
    return result


def _yaml_document_variable(dataset: nc.Dataset, name: str) -> nc.Variable:
    """Return a one-element NC_STRING variable safe for netCDF4-python VLEN writes.

    netCDF4-python requires integer indexing for VLEN strings. A true scalar
    NC_STRING variable therefore cannot be populated reliably with assignValue()
    across supported versions. MILGRAU stores each YAML document as a one-element
    string vector and writes element 0 explicitly.
    """
    if YAML_DOCUMENT_DIMENSION not in dataset.dimensions:
        dataset.createDimension(YAML_DOCUMENT_DIMENSION, 1)
    elif len(dataset.dimensions[YAML_DOCUMENT_DIMENSION]) != 1:
        raise ValueError(
            f"NetCDF dimension {YAML_DOCUMENT_DIMENSION!r} must have length 1 for MILGRAU provenance."
        )

    if name in dataset.variables:
        variable = dataset.variables[name]
        if variable.dimensions != (YAML_DOCUMENT_DIMENSION,):
            raise ValueError(
                f"Existing provenance variable {name!r} has incompatible dimensions {variable.dimensions}; "
                "regenerate the product before writing the current provenance schema."
            )
        return variable
    return dataset.createVariable(name, str, (YAML_DOCUMENT_DIMENSION,))


def _write_yaml_variable(dataset: nc.Dataset, name: str, path: Path | None, description: str) -> None:
    if path is None:
        return
    text = path.read_text(encoding="utf-8")
    variable = _yaml_document_variable(dataset, name)
    variable[0] = text
    variable.setncattr("media_type", "application/yaml")
    variable.setncattr("description", description)
    variable.setncattr("source_filename", path.name)


def netcdf_provenance_is_complete(path: str | Path) -> bool:
    """Return whether a published NetCDF has the current readable provenance schema."""
    try:
        with nc.Dataset(str(Path(path))) as dataset:
            if str(dataset.getncattr("software_name")).strip() != "MILGRAU":
                return False
            if not str(dataset.getncattr("software_version")).strip():
                return False
            if YAML_DOCUMENT_DIMENSION not in dataset.dimensions:
                return False
            if len(dataset.dimensions[YAML_DOCUMENT_DIMENSION]) != 1:
                return False
            for name in YAML_DOCUMENT_VARIABLES:
                if name not in dataset.variables:
                    return False
                variable = dataset.variables[name]
                if variable.dimensions != (YAML_DOCUMENT_DIMENSION,):
                    return False
                if not str(variable[0]).strip():
                    return False
        return True
    except Exception:
        return False


def write_netcdf_provenance(
    path: str | Path,
    config: Mapping[str, Any],
    *,
    source_attrs: Mapping[str, Any] | None = None,
    extra_attrs: Mapping[str, str | int | float] | None = None,
) -> dict[str, str | int | float]:
    """Persist readable metadata, exact YAML recipes and named content identities.

    Exact YAML remains the human-readable configuration record, so legacy
    processing/station configuration hashes are not restored. Content hashes
    are used only with an explicit consumer: exact Level 1 bytes for scientific
    lineage/cache correctness and normalized installed Python sources for code
    state identity. Host-specific absolute paths are not written.
    """
    source = source_attrs or {}
    attrs: dict[str, str | int | float] = inherited_provenance(source)
    attrs.update(thermodynamic_source_provenance(source, config))
    attrs.update(configuration_provenance(config, source_attrs=source))
    attrs.update(source_code_provenance())
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
        for legacy_name in LEGACY_HASH_ATTRS:
            if legacy_name in dataset.ncattrs():
                dataset.delncattr(legacy_name)
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
