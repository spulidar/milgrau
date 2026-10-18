"""Offline single-ID repair of historical SCC sounding filename references.

A legacy stage may reference rs_<15-character Measurement_ID>.nc even
though the SCC Quick Upload form requires rs_<date><station><2 digits>.nc.
Only an unjournalled package is repairable. Never contact SCC or silently
reset an uncertain/accepted submission. Preserve the preimage for audit.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import netCDF4 as nc

from milgrau.level0.scc_sounding import scc_sounding_name
from milgrau.level0.scc_staging import pending_scc_package, scc_raw_filename
from milgrau.provenance import file_sha256
from milgrau.scc.packages import validate_upload_package
from milgrau.scc.state import SubmissionStore, operation_lock, state_root


def _validate_legacy(path: Path, identity: str) -> tuple[dict[str, Any], str, str]:
    """Accept only the known old name and exact recorded hashes/byte counts.

    The sounding file itself may have already been manually renamed to the
    correct name; its original digest must still match the manifest.
    """
    if path.is_symlink() or path.name != identity or not path.is_dir():
        raise ValueError("SCC pending package identity/path is invalid.")
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("SCC pending manifest missing or unsafe.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("measurement_id") != identity
        or manifest.get("state") != "pending"
    ):
        raise ValueError("SCC pending manifest identity/state mismatch.")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("SCC pending files mapping missing.")
    old = f"rs_{identity}.nc"
    new = scc_sounding_name(identity)
    raw = scc_raw_filename(identity)
    if old not in files or new in files or raw not in files:
        raise ValueError("Expected one legacy rs_<Measurement_ID>.nc manifest entry.")
    allowed = {old, raw} | {x for x in files if x.startswith("lr_")}
    if set(files) != allowed or len(allowed) not in {2, 3}:
        raise ValueError("Unexpected files in legacy SCC pending manifest.")
    extras = {p.name for p in path.iterdir()} - (
        set(files) - {old} | {old, new, "manifest.json"}
    )
    if extras or any(p.is_symlink() or not p.is_file() for p in path.iterdir()):
        raise ValueError("Unexpected or unsafe physical SCC stage file.")
    has_old = (path / old).is_file()
    has_new = (path / new).is_file()
    if has_old == has_new:
        raise ValueError(
            "Expected exactly one sounding, with the legacy or correct name."
        )
    actual = old if has_old else new
    for filename, entry in files.items():
        source = path / (actual if filename == old else filename)
        if (
            not source.is_file()
            or file_sha256(source) != entry["sha256"]
            or source.stat().st_size != entry["size_bytes"]
        ):
            raise ValueError(
                f"Original SCC staged checksum/size mismatch: {filename}. "
                "Do not overwrite provenance after manual NetCDF changes."
            )
    with nc.Dataset(path / raw, "r") as ds:
        if str(ds.getncattr("Measurement_ID")) != identity:
            raise ValueError("Raw NetCDF Measurement_ID mismatch.")
        if int(ds.variables["Molecular_Calc"].getValue()) != 1:
            raise ValueError("SCC sounding referenced but Molecular_Calc is not 1.")
        if ds.getncattr("Sounding_File_Name") not in {old, new}:
            raise ValueError("Unexpected raw NetCDF Sounding_File_Name.")
    return manifest, actual, new


def repair_sounding_name(
    config: Mapping[str, Any], identity: str, *, confirm: bool = False,
) -> str:
    """Inspect/repair one pending packet; never make any SCC HTTP request.

    For corrections a complete original directory is retained in
    .cache/scc/legacy-sounding/<ID>. Crash/failed validation restores the old
    stage and never changes the SQLite submissions journal.
    """
    stage = pending_scc_package(config, identity)
    root = state_root(config)
    archive = stage.parent.parent / "legacy-sounding" / identity
    with operation_lock(root, identity), SubmissionStore(root) as journal:
        if journal.get(identity) is not None:
            raise ValueError(
                f"{identity} already has SCC submission history. "
                "Do not revise its pending files until the remote POST "
                "and journal are reconciled separately."
            )
        try:
            validate_upload_package(stage)
        except Exception:
            pass
        else:
            return f"CURRENT {identity}: sounding filename and hashes already valid"
        original, actual, correct = _validate_legacy(stage, identity)
        if not confirm:
            return (
                f"REPAIRABLE {identity}: {actual} -> {correct}; "
                "all original recorded hashes match; no files changed"
            )
        if archive.exists() or archive.is_symlink():
            raise ValueError(
                f"Original sounding archive already exists: {archive}. "
                "Inspect previous repair before retry."
            )
        tmp_root = Path(tempfile.mkdtemp(
            prefix=f".{identity}.sounding-rename-", dir=stage.parent,
        ))
        candidate = tmp_root / identity
        try:
            shutil.copytree(stage, candidate)
            old = f"rs_{identity}.nc"
            if actual == old:
                (candidate / old).rename(candidate / correct)
            raw_name = scc_raw_filename(identity)
            with nc.Dataset(candidate / raw_name, "r+") as ds:
                if ds.getncattr("Sounding_File_Name") != correct:
                    ds.setncattr("Sounding_File_Name", correct)
            revised = dict(original)
            revised["files"] = dict(original["files"])
            revised["files"][correct] = revised["files"].pop(old)
            raw_path = candidate / raw_name
            revised["files"][raw_name] = {
                "sha256": file_sha256(raw_path),
                "size_bytes": raw_path.stat().st_size,
            }
            (candidate / "manifest.json").write_text(
                json.dumps(revised, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            validate_upload_package(candidate)
            if file_sha256(candidate / correct) != original["files"][old]["sha256"]:
                raise ValueError("Sounding bytes changed during filename repair.")
            # One last check under the same ID lock before publishing.
            if journal.get(identity) is not None:
                raise ValueError("Submission journal changed during local repair.")
            archive.parent.mkdir(parents=True, exist_ok=True)
            stage.rename(archive)
            try:
                candidate.rename(stage)
            except Exception:
                archive.rename(stage)
                raise
            try:
                validate_upload_package(stage)
            except Exception:
                failed = stage.parent / f".{identity}.sounding-invalid"
                if failed.exists():
                    raise ValueError("Cannot roll back safely; original archive preserved.")
                stage.rename(failed)
                archive.rename(stage)
                shutil.rmtree(failed)
                raise
            return (
                f"REPAIRED {identity}: sounding={correct}; original stage "
                f"preserved at {archive}; no SCC request or journal changes"
            )
        finally:
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
