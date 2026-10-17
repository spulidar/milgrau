"""Offline validation of SCC submission packages before authentication.

No network client is imported here; dry-run works without the SCC extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import netCDF4 as nc
import numpy as np

from milgrau.level0.scc_staging import (
    scc_staging_root,
    manifest_raw_name,
    validate_pending_scc_package,
)
from milgrau.level0.scc_lidar_ratio import scc_lr_file_name
from milgrau.level0.scc_sounding import scc_sounding_name, validate_sounding


class SCCPackageError(ValueError):
    """A staged SCC submission failed local validation."""


def discover_pending(config: Mapping[str, Any], identity: str | None = None) -> list[Path]:
    """List only named complete package directories; never follow arbitrary paths."""
    root = scc_staging_root(config) / "pending"
    if identity is not None:
        from milgrau.level0.scc_staging import pending_scc_package

        path = pending_scc_package(config, identity)
        if not path.is_dir() or path.is_symlink():
            raise SCCPackageError(f"SCC Measurement_ID {identity!r} is not pending.")
        return [path]
    if not root.exists():
        return []
    return sorted(
        path for path in root.iterdir()
        if path.is_dir() and not path.is_symlink() and not path.name.startswith(".")
    )


def validate_upload_package(
    path: Path, *, allow_legacy_lr: bool = False,
) -> dict[str, Any]:
    """Verify checksums, raw SCC ID/config and climatological LR content."""
    try:
        manifest = validate_pending_scc_package(path)
        raw_name = manifest_raw_name(manifest)
        if path.is_symlink() or (path / raw_name).is_symlink():
            raise SCCPackageError("Symlinks are forbidden in SCC upload packages.")
        identity = str(manifest["measurement_id"])
        config_id = int(manifest["scc_configuration_id"])
        if config_id <= 0:
            raise SCCPackageError("SCC configuration ID must be positive.")
        product_ids = [int(pid) for pid in manifest["scc_product_ids"]]
        if len(product_ids) != len(set(product_ids)) or not product_ids:
            raise SCCPackageError("Invalid or duplicate SCC optical product IDs.")
        file_names = set(manifest["files"])
        with nc.Dataset(path / raw_name) as raw:
            if str(raw.getncattr("Measurement_ID")) != identity:
                raise SCCPackageError("Raw NetCDF Measurement_ID disagrees with manifest.")
            if int(raw.getncattr("SCC_Configuration_ID")) != config_id:
                raise SCCPackageError("Raw NetCDF SCC_Configuration_ID disagrees with manifest.")
            if "channel_ID" not in raw.variables:
                raise SCCPackageError("Raw NetCDF is missing SCC channel_ID.")
            lr_name = (
                str(raw.getncattr("LR_File_Name"))
                if "LR_File_Name" in raw.ncattrs() else None
            )
            sounding_name = (
                str(raw.getncattr("Sounding_File_Name"))
                if "Sounding_File_Name" in raw.ncattrs() else None
            )
            molecular_calc = (
                int(raw.variables["Molecular_Calc"].getValue())
                if "Molecular_Calc" in raw.variables else None
            )
        expected_lr = scc_lr_file_name(identity)
        if allow_legacy_lr and lr_name == f"lr_{identity}.nc":
            expected_lr = lr_name
        expected_files = {raw_name}
        if lr_name is not None:
            if lr_name != expected_lr:
                raise SCCPackageError("LR_File_Name does not match staged auxiliary.")
            expected_files.add(expected_lr)
        if sounding_name is not None:
            if sounding_name != scc_sounding_name(identity) or molecular_calc != 1:
                raise SCCPackageError(
                    "Sounding_File_Name requires rs_ID.nc and Molecular_Calc=1."
                )
            expected_files.add(sounding_name)
            validate_sounding(path / sounding_name, identity=identity)
        elif molecular_calc == 1:
            raise SCCPackageError(
                "Molecular_Calc=1 requires a referenced observed SCC sounding."
            )
        if file_names != expected_files:
            raise SCCPackageError("SCC staged auxiliary filenames do not match raw references.")
        if lr_name is not None:
            lr_file = path / expected_lr
            if lr_file.is_symlink():
                raise SCCPackageError("Symlinked SCC lidar-ratio file is forbidden.")
            with nc.Dataset(lr_file) as lr:
                if lr.variables["Altitude"].dimensions != ("points",):
                    raise SCCPackageError("Invalid SCC LR altitude dimension.")
                if lr.variables["Lidar_Ratio"].dimensions != ("products", "points"):
                    raise SCCPackageError("Invalid SCC LR profile dimensions.")
                if lr.variables["product_ID"].dimensions != ("products",):
                    raise SCCPackageError("Invalid SCC LR product_ID dimension.")
                ids = np.asarray(lr.variables["product_ID"][:], dtype=np.int64).tolist()
                if not ids or len(ids) != len(set(ids)) or not set(ids).issubset(product_ids):
                    raise SCCPackageError("SCC LR has unknown or duplicate optical product IDs.")
                altitude = np.asarray(lr.variables["Altitude"][:], dtype=np.float64)
                values = np.asarray(lr.variables["Lidar_Ratio"][:], dtype=np.float64)
                if (
                    altitude.size < 2 or not np.isfinite(altitude).all()
                    or not (np.diff(altitude) > 0).all()
                    or values.shape != (len(ids), len(altitude))
                    or not np.isfinite(values).all() or (values <= 0).any()
                ):
                    raise SCCPackageError("Invalid SCC LR altitude or numeric profile.")
        return manifest
    except (OSError, KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        raise SCCPackageError(f"Invalid SCC package {path.name}: {exc}") from exc
