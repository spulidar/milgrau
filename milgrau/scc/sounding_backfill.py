"""Explicit, offline SCC sounding backfill for pre-handoff pending packets.

Only complete observations validated by LIPANCORA are considered. No SCC
HTTP requests, no LIBIDS/LIPANCORA reprocessing, and no edits to journalled
submissions, original raw Licel data, or canonical L0/L1 products.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import netCDF4 as nc

from milgrau.io.paths import level1_output_path
from milgrau.level0.scc_sounding import stage_cached_sounding
from milgrau.level0.scc_staging import validate_pending_scc_package
from milgrau.scc.packages import discover_pending, validate_upload_package
from milgrau.scc.sounding_finalize import _enabled, _finalize_one
from milgrau.scc.state import SubmissionStore, state_root


@dataclass(frozen=True)
class SoundingCandidate:
    identity: str
    status: str
    reason: str
    source_level1: Path | None = None
    observed_target: str = ""


def inspect_legacy_sounding(
    config: Mapping[str, Any], path: Path,
    *,
    journal: SubmissionStore,
) -> SoundingCandidate:
    """Classify a historical packet using only local artifacts and provenance."""
    identity = path.name
    try:
        if path.is_symlink() or not path.is_dir():
            raise ValueError("Unsafe staged SCC package directory.")
        manifest = validate_pending_scc_package(path)
        if journal.get(identity) is not None:
            return SoundingCandidate(identity, "BLOCKED", "SCC journal already has this ID")
        stage = manifest.get("sounding_stage", "not_requested")
        if stage != "not_requested":
            return SoundingCandidate(identity, "SKIP", f"sounding_stage={stage}")
        # Original data and LR must be valid before augmenting anything.
        validate_upload_package(path)
        original_l0 = Path(str(manifest["canonical_level0_path"]))
        if not original_l0.is_file():
            raise ValueError("Referenced scientific L0 is missing.")
        l1 = level1_output_path(original_l0, config)
        if not l1.is_file() or l1.is_symlink():
            return SoundingCandidate(identity, "SKIP", "Scientific L1 is missing")
        with nc.Dataset(l1) as ds:
            session_id = str(getattr(ds, "Session_ID", ""))
            if session_id != manifest["scientific_session_id"]:
                raise ValueError("L1 Session_ID does not match SCC scientific session.")
            source_station = str(getattr(ds, "radiosonde_station_id", ""))
            station_id = str(config["_station_catalog"]["station"]["radiosonde"]["station_id"])
            if source_station and source_station != station_id:
                raise ValueError("L1 radiosonde station differs from SCC station.yaml.")
            available = str(getattr(ds, "radiosonde_available", "false")).lower() == "true"
            selected = str(getattr(ds, "radiosonde_qa_target_datetime_utc", ""))
        if not available or not selected:
            return SoundingCandidate(identity, "SKIP", "No observed sounding QA in LIPANCORA")
        # Probe in a disposable directory. In particular, do NOT alter the
        # historical raw NetCDF or journal during --dry-run.
        with tempfile.TemporaryDirectory(prefix="scc-sounding-preview-") as tmp:
            try:
                sounding = stage_cached_sounding(
                    Path(tmp), identity, manifest["measurement_start_utc"],
                    config, observed_target_utc=selected,
                )
            except (ValueError, KeyError, OSError) as exc:
                return SoundingCandidate(identity, "SKIP", f"Observed cache unusable: {exc}")
            if sounding is None:
                return SoundingCandidate(
                    identity, "SKIP", "No matching observed Wyoming CSV/JSON cache"
                )
        return SoundingCandidate(identity, "READY", "Observed radiosonde verified", l1, selected)
    except Exception as exc:
        return SoundingCandidate(identity, "ERROR", f"{type(exc).__name__}: {exc}")


def backfill_pending_soundings(
    config: Mapping[str, Any], *,
    identity: str | None = None,
    confirm: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[list[SoundingCandidate], int]:
    """Preview or explicitly enrich older cache packets; no SCC network calls.

    Applies independently per package, keeps a restorable preimage during
    writes, and refuses IDs with any existing submission journal row.
    """
    if not _enabled(config):
        raise ValueError(
            "Sounding backfill requires scc.enabled and scc_policy.sounding.enabled: true."
        )
    logger = logger or logging.getLogger("milgrau.scc.sounding")
    paths = discover_pending(config, identity)
    with SubmissionStore(state_root(config)) as journal:
        report = [inspect_legacy_sounding(config, path, journal=journal) for path in paths]

    # Abort modifications if any eligible package is structurally invalid.
    if confirm and any(item.status == "ERROR" for item in report):
        return report, 1

    if confirm:
        revised: list[SoundingCandidate] = []
        for item, path in zip(report, paths):
            if item.status != "READY":
                revised.append(item)
                continue
            # The finalizer independently rechecks journal + bytes and
            # implements the original-packet rollback. Preflight alone
            # is never trusted as upload authorization.
            try:
                result = _finalize_one(
                    config, path, available=True,
                    qa_target=item.observed_target, logger=logger,
                    allow_legacy=True,
                )
                # Guarantee the resulting raw/LR/sounding package is valid.
                new_manifest = validate_upload_package(path)
                if (
                    new_manifest.get("sounding_stage") != "ready"
                    or not new_manifest.get("sounding_backfilled")
                ):
                    raise ValueError("Sounding preparation did not produce a ready package.")
                revised.append(SoundingCandidate(
                    item.identity, "UPDATED", result, item.source_level1,
                    item.observed_target,
                ))
            except Exception as exc:
                revised.append(SoundingCandidate(
                    item.identity, "ERROR", f"Backfill failed: {exc}",
                ))
                # Stop rather than continue through a potentially interrupted
                # file transaction and notify the operator.
                revised.extend(report[len(revised):])
                return revised, 1
        report = revised
    return report, (1 if any(item.status in {"ERROR", "BLOCKED"} for item in report) else 0)
