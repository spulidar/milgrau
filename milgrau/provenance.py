"""Deterministic provenance fingerprints for incremental MILGRAU products."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final


PROVENANCE_FORMAT_VERSION: Final[int] = 1
PROVENANCE_MANIFEST_SUFFIX: Final[str] = ".provenance.json"
_PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parent

_LEVEL0_PHYSICS_KEYS: Final[tuple[str, ...]] = (
    "latitude",
    "longitude",
    "vertical_resolution_m",
    "default_surface_temp_c",
    "default_surface_pressure_hpa",
    "background_start_m",
    "background_stop_m",
    "bg_start",
    "bg_stop",
    "laser_pointing_angle_deg",
)
_LEVEL1_PHYSICS_KEYS: Final[tuple[str, ...]] = (
    "speed_of_light",
    "speed_of_light_m_s",
    "channels",
    "pbl_min_search_m",
    "pbl_max_search_m",
    "pbl_smooth_bins",
)
_DORMANT_INVERSION_KEYS: Final[frozenset[str]] = frozenset(
    {"enabled", "interactive_qa", "products", "cloud_screening"}
)
_DORMANT_QUICKLOOK_KEYS: Final[frozenset[str]] = frozenset(
    {"show_pbl", "show_tropopause", "mean_profile_smooth_bins"}
)
_DORMANT_LEVEL2_QA_KEYS: Final[frozenset[str]] = frozenset({"max_altitude_km"})


@dataclass(frozen=True, slots=True)
class ProductProvenance:
    """Expected provenance for one product before it is generated or reused."""

    product: str
    fingerprint: str
    payload: dict[str, Any]

    def manifest_payload(self, output_path: str | Path) -> dict[str, Any]:
        """Return the persisted manifest, including the generated-output digest."""
        return {
            "format_version": PROVENANCE_FORMAT_VERSION,
            "product": self.product,
            "fingerprint": self.fingerprint,
            "payload": self.payload,
            "output": file_signature(output_path),
        }


def _json_value(value: Any) -> Any:
    """Convert common Python and NumPy-like values to canonical JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_value(item) for item in value]
        return sorted(normalized, key=canonical_json)
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite_float__": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    """Serialize one value deterministically for hashing and persistence."""
    return json.dumps(_json_value(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(path: str | Path) -> dict[str, Any]:
    """Return a location-independent content signature for one regular file."""
    resolved = Path(path)
    stat = resolved.stat()
    if not resolved.is_file():
        raise ValueError(f"Provenance input is not a regular file: {resolved}")
    return {
        "name": resolved.name,
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(resolved),
    }


@lru_cache(maxsize=1)
def software_identity() -> dict[str, str]:
    """Return package version and a digest of all shipped Python sources."""
    try:
        package_version = version("milgrau")
    except PackageNotFoundError:
        package_version = "uninstalled"

    digest = hashlib.sha256()
    for source_path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        relative_path = source_path.relative_to(_PACKAGE_ROOT).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return {"package_version": package_version, "source_sha256": digest.hexdigest()}


def _selected(mapping: Any, keys: Iterable[str]) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        return {}
    return {key: mapping[key] for key in keys if key in mapping}


def relevant_configuration(product: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the code-audited configuration subset that can affect a product."""
    if product == "level0":
        return {
            "site": _selected(config.get("site"), ("timezone",)),
            "location": _selected(config.get("location"), ("timezone",)),
            "processing": _selected(
                config.get("processing"),
                ("laser_shot_tolerance_fraction", "dark_current_max_association_hours"),
            ),
            "physics": _selected(config.get("physics"), _LEVEL0_PHYSICS_KEYS),
            "hardware": _selected(config.get("hardware"), ("name_to_id",)),
        }
    if product == "level1":
        return {
            "physics": _selected(config.get("physics"), _LEVEL1_PHYSICS_KEYS),
            "radiosonde": _selected(config.get("radiosonde"), ("station_id",)),
            "location": _selected(config.get("location"), ("station_id",)),
        }
    if product == "level2":
        inversion = config.get("inversion", {})
        relevant_inversion = (
            {str(key): value for key, value in inversion.items() if key not in _DORMANT_INVERSION_KEYS}
            if isinstance(inversion, Mapping)
            else {}
        )
        return {
            "site": _selected(config.get("site"), ("station_altitude_m",)),
            "physics": _selected(config.get("physics"), ("station_altitude_m",)),
            "inversion": relevant_inversion,
        }
    if product in {"liracos.quicklook", "liracos.global_mean"}:
        visualization = config.get("visualization", {})
        if not isinstance(visualization, Mapping):
            return {"visualization": {}}
        quicklook = visualization.get("quicklook", {})
        relevant_quicklook = (
            {str(key): value for key, value in quicklook.items() if key not in _DORMANT_QUICKLOOK_KEYS}
            if isinstance(quicklook, Mapping)
            else {}
        )
        if product == "liracos.quicklook":
            return {
                "visualization": {
                    **_selected(visualization, ("output_format", "dpi")),
                    "quicklook": relevant_quicklook,
                }
            }
        return {
            "visualization": _selected(
                visualization,
                ("output_format", "dpi", "altitude_ranges_km", "channels_to_plot"),
            )
        }
    if product == "lebear.qa":
        visualization = config.get("visualization", {})
        if not isinstance(visualization, Mapping):
            return {"visualization": {}}
        qa = visualization.get("level2_qa", {})
        relevant_qa = (
            {str(key): value for key, value in qa.items() if key not in _DORMANT_LEVEL2_QA_KEYS}
            if isinstance(qa, Mapping)
            else {}
        )
        return {
            "visualization": {
                **_selected(visualization, ("output_format", "dpi")),
                "level2_qa": relevant_qa,
            }
        }
    raise ValueError(f"Unknown provenance product: {product}")


def build_product_provenance(
    product: str,
    input_paths: Iterable[str | Path],
    config: Mapping[str, Any],
    *,
    variant: Mapping[str, Any] | None = None,
) -> ProductProvenance:
    """Build one deterministic product fingerprint from inputs, config and software."""
    input_signatures = [file_signature(path) for path in input_paths]
    return build_product_provenance_from_signatures(
        product,
        input_signatures,
        config,
        variant=variant,
    )


def build_product_provenance_from_signatures(
    product: str,
    input_signatures: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    variant: Mapping[str, Any] | None = None,
) -> ProductProvenance:
    """Build a fingerprint while reusing input digests computed in the same run."""
    input_signatures = [dict(signature) for signature in input_signatures]
    input_signatures.sort(key=lambda item: (item["name"], item["sha256"], item["size_bytes"]))
    payload = _json_value(
        {
            "provenance_schema": PROVENANCE_FORMAT_VERSION,
            "product": product,
            "inputs": input_signatures,
            "configuration": relevant_configuration(product, config),
            "software": software_identity(),
            "variant": dict(variant or {}),
        }
    )
    fingerprint = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return ProductProvenance(product=product, fingerprint=fingerprint, payload=payload)


def provenance_manifest_path(output_path: str | Path) -> Path:
    """Return the sidecar path associated with one generated product."""
    path = Path(output_path)
    return path.with_suffix(path.suffix + PROVENANCE_MANIFEST_SUFFIX)


def load_provenance_manifest(output_path: str | Path) -> dict[str, Any] | None:
    """Load a product sidecar, returning ``None`` for absent or invalid manifests."""
    manifest_path = provenance_manifest_path(output_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_provenance_manifest(output_path: str | Path, provenance: ProductProvenance) -> Path:
    """Atomically write provenance only after a complete product exists."""
    output = Path(output_path)
    manifest_path = provenance_manifest_path(output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = provenance.manifest_payload(output)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=manifest_path.parent,
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return manifest_path


def output_is_current(
    output_path: str | Path,
    expected: ProductProvenance,
    *,
    integrity_check: Callable[[Path], bool] | None = None,
) -> bool:
    """Return whether output, digest, provenance and optional contract all match."""
    output = Path(output_path)
    try:
        if not output.is_file() or output.stat().st_size <= 0:
            return False
        manifest = load_provenance_manifest(output)
        if manifest is None:
            return False
        if manifest.get("format_version") != PROVENANCE_FORMAT_VERSION:
            return False
        if manifest.get("product") != expected.product or manifest.get("fingerprint") != expected.fingerprint:
            return False
        if manifest.get("output") != file_signature(output):
            return False
        return integrity_check(output) if integrity_check is not None else True
    except (OSError, ValueError):
        return False
