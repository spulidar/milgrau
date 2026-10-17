"""Finalize LIBIDS SCC packets from independently validated LIPANCORA observations.

SCC transport and scientific Level 1 remain separate. No network requests, no
retroactive changes to already journalled/accepted measurements, and no fake
sounding from modelled temperature or pressure data.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import netCDF4 as nc

from milgrau.level0.scc_sounding import stage_cached_sounding
from milgrau.level0.scc_staging import (
    manifest_raw_name, scc_staging_root, validate_pending_scc_package,
)
from milgrau.provenance import file_sha256
from milgrau.scc.packages import validate_upload_package
from milgrau.scc.state import SubmissionStore, operation_lock, state_root


def _enabled(config: Mapping[str, Any]) -> bool:
    return (
        config.get("scc", {}).get("enabled", False) is True
        and config.get("_scc_catalog", {}).get("scc_policy", {})
        .get("sounding", {}).get("enabled", False) is True
    )


def finalize_session_soundings(
    level0_path: Path, level1_path: Path,
    config: Mapping[str, Any], logger: logging.Logger,
) -> list[str]:
    """Complete all eligible SCC partitions of a successfully produced L1.

    Only packages created with sounding_stage=awaiting_lipancora and NO prior
    SCC journal row may be revised. Returned descriptions are useful in logs
    and tests. Missing sounding with skip policy is finalized without changing
    the raw NetCDF's Molecular_Calc=0.
    """
    if not _enabled(config):
        return []
    if not level1_path.is_file():
        raise ValueError("Cannot finalize SCC sounding before a valid L1 file exists.")
    with nc.Dataset(level1_path) as ds:
        available = str(getattr(ds, "radiosonde_available", "false")).lower() == "true"
        qa_target = str(getattr(ds, "radiosonde_qa_target_datetime_utc", ""))
        l1_session = str(getattr(ds, "Session_ID", ""))
    packets = scc_staging_root(config) / "pending"
    if not packets.is_dir():
        return []
    results: list[str] = []
    for path in sorted(packets.iterdir()):
        if not path.is_dir() or path.is_symlink() or path.name.startswith("."):
            continue
        manifest = validate_pending_scc_package(path)
        if Path(str(manifest["canonical_level0_path"])).resolve() != level0_path.resolve():
            continue
        if l1_session and l1_session != manifest["scientific_session_id"]:
            raise ValueError(f"SCC {path.name}: L1 Session_ID does not match SCC scientific source.")
        if manifest.get("sounding_stage") != "awaiting_lipancora":
            continue
        result = _finalize_one(
            config, path, available=available, qa_target=qa_target, logger=logger,
        )
        results.append(result)
    return results


def _finalize_one(
    config: Mapping[str, Any], path: Path, *,
    available: bool, qa_target: str, logger: logging.Logger,
) -> str:
    identity = path.name
    root = state_root(config)
    backup = path.parent / f".{identity}.before-sounding"
    policy = config["_scc_catalog"]["scc_policy"]["sounding"]
    missing = policy.get("missing_policy", "skip")
    with operation_lock(root, identity), SubmissionStore(root) as journal:
        # Even a pending/failed/uncertain journal row is evidence of a prior
        # SCC operation. Finalization is NEVER an implicit replacement.
        if journal.get(identity) is not None:
            raise ValueError(
                f"SCC {identity} already has durable submission history; "
                "refusing to revise source checksums or sounding."
            )
        manifest = validate_pending_scc_package(path)
        if manifest.get("sounding_stage") != "awaiting_lipancora":
            return f"{identity}: already finalized"
        catalog_path = config.get("_scc_config_path")
        if (catalog_path and manifest.get("scc_config_sha256") != file_sha256(catalog_path)):
            raise ValueError(
                f"SCC {identity}: scc.yaml changed between LIBIDS and LIPANCORA; "
                "the pending package cannot be silently rebound to a new configuration."
            )
        if backup.exists() or backup.is_symlink():
            raise ValueError(
                f"Interrupted SCC sounding transaction for {identity}: "
                f"inspect retained backup {backup} before any further operation."
            )
        temp = Path(tempfile.mkdtemp(prefix=f".{identity}.sounding-tmp-", dir=path.parent))
        published = False
        try:
            for file in path.iterdir():
                if file.is_symlink() or not file.is_file():
                    raise ValueError("SCC sounding transaction refuses non-file or symlink cache members.")
                shutil.copy2(file, temp / file.name)

            sounding = None
            if available and qa_target:
                sounding = stage_cached_sounding(
                    temp, identity, manifest["measurement_start_utc"], config,
                    observed_target_utc=qa_target,
                )
            if sounding is None and missing == "fail":
                raise ValueError(
                    f"SCC {identity} has no validated observed sounding from "
                    "LIPANCORA; policy missing_policy=fail."
                )
            revised = dict(manifest)
            revised["files"] = dict(manifest["files"])
            raw_name = manifest_raw_name(manifest)
            if sounding is not None:
                with nc.Dataset(temp / raw_name, "r+") as raw:
                    raw.setncattr("Sounding_File_Name", sounding.name)
                    raw.variables["Molecular_Calc"].assignValue(1)
                revised["files"][raw_name] = {
                    "sha256": file_sha256(temp / raw_name),
                    "size_bytes": (temp / raw_name).stat().st_size,
                }
                revised["files"][sounding.name] = {
                    "sha256": file_sha256(sounding),
                    "size_bytes": sounding.stat().st_size,
                }
                revised["sounding_stage"] = "ready"
                revised["sounding_source"] = "lipancora_wyoming_observation"
            else:
                revised["sounding_stage"] = "skipped_missing"
            (temp / "manifest.json").write_text(
                json.dumps(revised, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            # All in-memory/metadata checks are done before touching the
            # original pending directory. A complete preimage is retained
            # during the two-directory swap.
            path.rename(backup)
            try:
                temp.rename(path)
                published = True
                validate_upload_package(path)
            except Exception:
                if published:
                    failed = path.parent / f".{identity}.invalid-sounding"
                    if failed.exists():
                        raise ValueError("SCC sounding rollback collision; original backup retained.")
                    path.rename(failed)
                    backup.rename(path)
                    shutil.rmtree(failed)
                else:
                    backup.rename(path)
                raise
            shutil.rmtree(backup)
            label = "observed sounding attached" if sounding is not None else "no observed sounding (allowed)"
            logger.info("SCC %s | %s", identity, label)
            return f"{identity}: {label}"
        finally:
            if temp.exists():
                shutil.rmtree(temp)
