"""Offline, crash-recoverable migration of SCC staged raw.nc to Measurement_ID.nc.

Only local disposable pending directories change. Immutable original raw bytes,
LR bytes, SHA-256 checksums, durable submission state and SCC remain unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from milgrau.level0.scc_staging import (
    manifest_raw_name, pending_scc_package, scc_raw_filename,
)
from milgrau.provenance import file_sha256
from milgrau.scc.packages import validate_upload_package
from milgrau.scc.state import operation_lock, state_root


def migrate_pending(
    config: Mapping[str, Any], identity: str,
) -> tuple[Path, bool]:
    """Migrate one existing package without ever issuing an SCC API request.

    On interruption, a complete old manifest remains valid until the new
    manifest is atomically published; an extra, same-hash ID.nc can be reused.
    Never overwrite a differing destination NetCDF or follow symlinks.
    """
    path = pending_scc_package(config, identity)
    root = state_root(config)
    with operation_lock(root, identity):
        manifest = validate_upload_package(path)
        new_name = scc_raw_filename(identity)
        old_path = path / "raw.nc"
        new_path = path / new_name
        raw_name = manifest_raw_name(manifest)
        if raw_name == new_name:
            if old_path.exists() or old_path.is_symlink():
                if old_path.is_symlink() or (
                    file_sha256(old_path) != manifest["files"][new_name]["sha256"]
                ):
                    raise ValueError("Unexpected old raw.nc conflicts with migrated package.")
                old_path.unlink()
            return path, False
        if raw_name != "raw.nc":
            raise ValueError("Unknown source file in SCC staging manifest.")
        expected = manifest["files"]["raw.nc"]
        if old_path.is_symlink():
            raise ValueError("Unsafe symlink in legacy SCC staging.")
        if new_path.exists() or new_path.is_symlink():
            if new_path.is_symlink() or not new_path.is_file() or (
                file_sha256(new_path) != expected["sha256"]
                or new_path.stat().st_size != expected["size_bytes"]
            ):
                raise ValueError("Conflicting SCC ID-named NetCDF: migration aborted.")
        else:
            fd, temporary = tempfile.mkstemp(prefix=".raw-migrate-", dir=path)
            try:
                with os.fdopen(fd, "wb") as output, old_path.open("rb") as source:
                    shutil.copyfileobj(source, output)
                    output.flush()
                    os.fsync(output.fileno())
                if (
                    file_sha256(temporary) != expected["sha256"]
                    or Path(temporary).stat().st_size != expected["size_bytes"]
                ):
                    raise ValueError("Staged raw bytes changed while copying.")
                # Same-directory atomic rename; NEVER rewrite raw NetCDF bytes.
                os.replace(temporary, new_path)
            finally:
                if Path(temporary).exists():
                    Path(temporary).unlink()

        if file_sha256(old_path) != expected["sha256"]:
            raise ValueError("Legacy SCC raw bytes changed during migration.")
        revised = dict(manifest)
        revised["files"] = dict(manifest["files"])
        revised["files"][new_name] = revised["files"].pop("raw.nc")

        fd, temporary = tempfile.mkstemp(prefix=".manifest-migrate-", dir=path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                output.write(json.dumps(revised, indent=2, sort_keys=True) + "\n")
                output.flush()
                os.fsync(output.fileno())
            # Publication of new manifest is atomic. Existing journal raw_SHA
            # remains valid, because the NetCDF bytes have not been altered.
            os.replace(temporary, path / "manifest.json")
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()

        verified = validate_upload_package(path)
        if (
            manifest_raw_name(verified) != new_name
            or verified["files"][new_name] != expected
        ):
            raise ValueError("SCC staged manifest migration failed integrity checks.")
        if old_path.is_symlink() or file_sha256(old_path) != expected["sha256"]:
            raise ValueError("Legacy SCC raw changed before duplicate removal.")
        old_path.unlink()
        return path, True
