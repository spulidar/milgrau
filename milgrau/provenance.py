"""In-memory compatibility for the retired provenance API.

The former SHA-256 provenance/sidecar system is gone. Existing pipeline call
sites may temporarily use these names, but product reuse delegates to the simple
timestamp policy in :mod:`milgrau.incremental` and no ``*.provenance.json`` file
is created. A tiny process-local state exists only so older internal callers can
read result metadata during the same execution.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from milgrau.incremental import output_is_current as _timestamp_output_is_current

_RUNTIME_STATE: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True, slots=True)
class ProductProvenance:
    """Lightweight compatibility object carrying only incremental dependencies."""

    product: str
    input_paths: tuple[Path, ...]
    config: Mapping[str, Any]
    extra_dependencies: tuple[Path, ...] = ()


class _RuntimeManifestHandle:
    """Path-like test/backward-compatibility handle that never touches disk."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path).expanduser()

    def read_text(self, encoding: str = "utf-8") -> str:
        del encoding
        state = _RUNTIME_STATE.get(str(self.output_path.resolve()), {})
        payload = {"result": deepcopy(state.get("result"))} if "result" in state else {}
        return json.dumps(payload)

    def write_text(self, data: str, encoding: str = "utf-8") -> int:
        del encoding
        payload = json.loads(data)
        key = str(self.output_path.resolve())
        state = _RUNTIME_STATE.setdefault(key, {})
        if isinstance(payload, Mapping) and "result" in payload:
            state["result"] = deepcopy(payload["result"])
        else:
            state.pop("result", None)
        return len(data)

    def exists(self) -> bool:
        return False

    def __str__(self) -> str:
        return str(self.output_path.with_suffix(self.output_path.suffix + ".provenance.json"))


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


def provenance_manifest_path(output_path: str | Path) -> _RuntimeManifestHandle:
    """Return an in-memory handle for legacy callers; no filesystem path is created."""
    return _RuntimeManifestHandle(output_path)


def _netcdf_result_metadata(path: Path) -> dict[str, Any] | None:
    try:
        import numpy as np
        import xarray as xr

        with xr.open_dataset(path) as ds:
            if "requested_wavelengths" not in ds or "processed_wavelengths" not in ds or "failed_wavelengths" not in ds:
                return None
            return {
                "product_completeness": str(ds.attrs.get("product_completeness", "")),
                "product_status": str(ds.attrs.get("product_status", "")),
                "requested_wavelengths": [int(value) for value in np.asarray(ds["requested_wavelengths"].values).tolist()],
                "processed_wavelengths": [int(value) for value in np.asarray(ds["processed_wavelengths"].values).tolist()],
                "failed_wavelengths": [int(value) for value in np.asarray(ds["failed_wavelengths"].values).tolist()],
            }
    except Exception:
        return None


def load_provenance_manifest(output_path: str | Path) -> dict[str, Any] | None:
    """Return process-local result metadata, falling back to Level 2 NetCDF state."""
    path = Path(output_path).expanduser()
    state = _RUNTIME_STATE.get(str(path.resolve()))
    if state is not None and "result" in state:
        return {"result": deepcopy(state["result"])}
    if not path.is_file() or path.suffix.lower() != ".nc":
        return None
    result = _netcdf_result_metadata(path)
    return None if result is None else {"result": result}


def write_provenance_manifest(
    output_path: str | Path,
    provenance: ProductProvenance,
    *,
    result_metadata: Mapping[str, Any] | None = None,
) -> _RuntimeManifestHandle:
    """Record optional result state in memory only; never write a sidecar."""
    path = Path(output_path).expanduser()
    _RUNTIME_STATE[str(path.resolve())] = {
        "config": deepcopy(dict(provenance.config)),
        **({"result": deepcopy(dict(result_metadata))} if result_metadata is not None else {}),
    }
    return _RuntimeManifestHandle(path)


def output_is_current(
    output_path: str | Path,
    expected: ProductProvenance,
    *,
    integrity_check: Callable[[Path], bool] | None = None,
) -> bool:
    """Delegate reuse to timestamps and optionally compare same-process config state."""
    path = Path(output_path).expanduser()
    if not _timestamp_output_is_current(
        path,
        expected.input_paths,
        config=expected.config,
        extra_dependencies=expected.extra_dependencies,
        integrity_check=integrity_check,
    ):
        return False
    state = _RUNTIME_STATE.get(str(path.resolve()))
    if state is not None and "config" in state:
        return state["config"] == dict(expected.config)
    return True


def relevant_configuration(product: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Deprecated compatibility helper; hashes/config fingerprints no longer exist."""
    del product, config
    return {}
