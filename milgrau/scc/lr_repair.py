"""Explicit offline repair of SCC's legacy lidar-ratio ancillary filename.

The deployed SCC Quick Upload form accepts lr_YYYYMMDDcccNN.nc, while older
MILGRAU exports included four HHMM digits. An on-disk LR rename must ALSO
update LR_File_Name within the raw NetCDF, and therefore its SHA-256. Keep
original bytes in the disposable cache, never modify the original remote
backup or silently reset an uncertain upload's journal.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import netCDF4 as nc

from milgrau.level0.scc_lidar_ratio import scc_lr_file_name
from milgrau.level0.scc_staging import pending_scc_package, scc_raw_filename
from milgrau.provenance import file_sha256
from milgrau.scc.packages import validate_upload_package
from milgrau.scc.state import SubmissionStore, operation_lock, state_root


def _check_one_folder(stage: Path, identity: str) -> None:
    if stage.is_symlink() or not stage.is_dir() or stage.name != identity:
        raise ValueError("Unsafe or incorrectly identified SCC cache folder.")
    for entry in stage.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"Unsafe SCC cache member: {entry.name}.")


def repair_lr_cache(config: Mapping[str, Any], identity: str) -> tuple[Path, bool]:
    """Fix only filename metadata without changing scientific arrays.

    Original cache is moved to .cache/scc/legacy-lr/ID and retained.
    A crash between publication and journal update is reconciled by rerunning
    this offline command. A crash during the directory swap restores original
    directory on the next invocation, with no remote action.
    """
    stage = pending_scc_package(config, identity)
    archive = stage.parent.parent / "legacy-lr" / identity
    root = state_root(config)
    new_lr = scc_lr_file_name(identity)
    old_lr = f"lr_{identity}.nc"
    raw_name = scc_raw_filename(identity)

    with operation_lock(root, identity):
        if not stage.exists() and archive.is_dir() and not archive.is_symlink():
            archive.rename(stage)
        _check_one_folder(stage, identity)
        if (stage / new_lr).is_file() and not (stage / old_lr).exists():
            current = validate_upload_package(stage)
            if archive.is_dir():
                original = validate_upload_package(archive, allow_legacy_lr=True)
                _reconcile_journal(root, identity, original, current)
            return stage, False

        original = validate_upload_package(stage, allow_legacy_lr=True)
        if old_lr not in original["files"] or new_lr in original["files"]:
            raise ValueError(
                f"SCC LR repair expected original {old_lr} and no {new_lr}."
            )
        if archive.exists() or archive.is_symlink():
            raise ValueError(
                f"SCC original LR cache archive already exists: {archive}; "
                "review the interrupted migration manually."
            )
        # A 13-character SCC LR filename cannot distinguish all 15-character
        # measurement IDs. Fail if another staged packet has the same LR name.
        for other in stage.parent.iterdir():
            if other == stage or other.is_symlink() or not other.is_dir():
                continue
            if other.name[:13] == identity[:13] and other.name != identity:
                raise ValueError(
                    f"LR filename {new_lr} would collide with {other.name}. "
                    "Do not silently reuse a lidar-ratio ancillary."
                )

        tmp = Path(tempfile.mkdtemp(prefix=f".{identity}.lr-", dir=stage.parent))
        try:
            for member in stage.iterdir():
                shutil.copy2(member, tmp / member.name)
            (tmp / old_lr).rename(tmp / new_lr)
            with nc.Dataset(tmp / raw_name, "r+") as raw:
                if raw.getncattr("LR_File_Name") != old_lr:
                    raise ValueError("Unexpected original LR_File_Name in raw NetCDF.")
                raw.setncattr("LR_File_Name", new_lr)
            revised = dict(original)
            revised["files"] = dict(original["files"])
            revised["files"][new_lr] = revised["files"].pop(old_lr)
            raw_file = tmp / raw_name
            revised["files"][raw_name] = {
                "sha256": file_sha256(raw_file),
                "size_bytes": raw_file.stat().st_size,
            }
            # The LR content remains byte-for-byte unchanged; only its name
            # and the raw file's global attribute change.
            if file_sha256(tmp / new_lr) != original["files"][old_lr]["sha256"]:
                raise ValueError("SCC LR bytes changed unexpectedly.")
            (tmp / "manifest.json").write_text(
                json.dumps(revised, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            archive.parent.mkdir(parents=True, exist_ok=True)
            stage.rename(archive)
            try:
                tmp.rename(stage)
            except Exception:
                archive.rename(stage)
                raise
            try:
                current = validate_upload_package(stage)
                _reconcile_journal(root, identity, original, current)
            except Exception:
                # Retain the original evidence, roll back the corrected cache
                # if durable journal reconciliation was not possible.
                if archive.is_dir() and not archive.is_symlink():
                    failed = stage.parent / f".{identity}.repair-failed"
                    if failed.exists():
                        raise
                    stage.rename(failed)
                    archive.rename(stage)
                    shutil.rmtree(failed)
                raise
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        return stage, True


def _reconcile_journal(
    root: Path, identity: str,
    original: Mapping[str, Any], revised: Mapping[str, Any],
) -> None:
    with SubmissionStore(root) as journal:
        journal.record_local_lr_repair(identity, original, revised)
