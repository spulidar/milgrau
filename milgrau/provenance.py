"""Compatibility helpers for the former provenance API.

The old SHA-256 provenance/sidecar system was intentionally retired. Existing
pipeline call sites may temporarily use these names, but they now delegate to the
simple timestamp-based incremental policy in :mod:`milgrau.incremental` and never
write ``*.provenance.json`` files.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from milgrau.incremental import output_is_current as _timestamp_output_is_current


@dataclass(frozen=True, slots=True)
class ProductProvenance:
    """Lightweight compatibility object carrying only incremental dependencies."""

    product: str
    input_paths: tuple[Path, ...]
    config: Mapping[str, Any]
    extra_dependencies: tuple[Path, ...] = ()


def file_signature(path: str | Path) -> dict[str, str]:
    """Compatibility representation of a dependency path; no content hash is computed."""
    return {"path": str(Path(path).expanduser())}


def build_product_provenance(
    product: str,
    input_paths: Iterable[str | Path],
    config: Mapping[str, Any],
    *,
    variant: Mapping[str, Any] | None = None,
) -> ProductProvenance:
    """Build a lightweight dependency set; ``variant`` is intentionally ignored."""
    del variant
    return ProductProvenance(
        product=product,
        input_paths=tuple(Path(path).expanduser() for path in input_paths),
        config=config,
    )


def build_product_provenance_from_signatures(
    product: str,
    input_signatures: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    variant: Mapping[str, Any] | None = None,
) -> ProductProvenance:
    """Build dependencies from legacy ``file_signature`` dictionaries."""
    del variant
    paths = []
    for signature in input_signatures:
        path = signature.get("path")
        if path:
            paths.append(Path(str(path)).expanduser())
    return ProductProvenance(product=product, input_paths=tuple(paths), config=config)


def provenance_manifest_path(output_path: str | Path) -> Path:
    """Return the former sidecar path for cleanup/backward compatibility only."""
    path = Path(output_path)
    return path.with_suffix(path.suffix + ".provenance.json")


def load_provenance_manifest(output_path: str | Path) -> dict[str, Any] | None:
    """Expose Level 2 completeness directly from its NetCDF, without a sidecar.

    This is a temporary compatibility bridge for the Level 2 incremental caller.
    Other products receive ``None`` because they no longer need manifest state.
    """
    path = Path(output_path)
    if not path.is_file() or path.suffix.lower() != ".nc":
        return None
    try:
        import numpy as np
        import xarray as xr

        with xr.open_dataset(path) as ds:
            if "requested_wavelengths" not in ds or "processed_wavelengths" not in ds or "failed_wavelengths" not in ds:
                return None
            result = {
                "product_completeness": str(ds.attrs.get("product_completeness", "")),
                "product_status": str(ds.attrs.get("product_status", "")),
                "requested_wavelengths": [int(value) for value in np.asarray(ds["requested_wavelengths"].values).tolist()],
                "processed_wavelengths": [int(value) for value in np.asarray(ds["processed_wavelengths"].values).tolist()],
                "failed_wavelengths": [int(value) for value in np.asarray(ds["failed_wavelengths"].values).tolist()],
            }
        return {"result": result}
    except Exception:
        return None


def write_provenance_manifest(
    output_path: str | Path,
    provenance: ProductProvenance,
    *,
    result_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Compatibility no-op: no sidecar is written."""
    del provenance, result_metadata
    return provenance_manifest_path(output_path)


def output_is_current(
    output_path: str | Path,
    expected: ProductProvenance,
    *,
    integrity_check: Callable[[Path], bool] | None = None,
) -> bool:
    """Delegate incremental reuse to the timestamp/contract policy."""
    return _timestamp_output_is_current(
        output_path,
        expected.input_paths,
        config=expected.config,
        extra_dependencies=expected.extra_dependencies,
        integrity_check=integrity_check,
    )


def relevant_configuration(product: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Deprecated compatibility helper; hashes/config fingerprints no longer exist."""
    del product, config
    return {}
