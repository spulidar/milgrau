"""Durable per-Measurement_ID SCC operations journal; never stored under .cache.

A local exclusive O_EXCL lock prevents two MILGRAU processes racing for the same
ID. Old locks require manual inspection; they are intentionally not auto-stolen.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from milgrau.io.paths import resolve_project_path

STATES = frozenset({
    "pending", "remote_exists", "preflight", "archiving", "backed_up",
    "deleting", "remote_deleted", "uploading", "accepted", "processing",
    "completed", "downloaded", "cleaned", "failed", "uncertain",
})
# If an operation stopped after a destructive/network action, an operator must
# inspect server and journal before any repeat. Never automatically retry.
BLOCKED_RESTART = frozenset({"deleting", "remote_deleted", "uploading", "uncertain"})


def state_root(config: Mapping[str, Any]) -> Path:
    settings = config.get("scc", {})
    directory = settings.get("state_dir", ".state/scc")
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("scc.state_dir must be a nonempty path.")
    root = resolve_project_path(directory).resolve()
    # Input packets live inside scientific sessions, not below a global cache.
    from milgrau.io.paths import processed_data_root
    if isinstance(config.get("directories"), Mapping) and "processed_data" in config["directories"]:
        products = processed_data_root(config).resolve()
        if root == products or products in root.parents or root in products.parents:
            raise ValueError("Durable SCC journal must remain outside processed session products.")
    return root


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(identity: str) -> str:
    from milgrau.level0.scc_staging import _MEASUREMENT_ID

    if not _MEASUREMENT_ID.fullmatch(identity):
        raise ValueError(f"Invalid SCC Measurement_ID: {identity!r}.")
    return identity



def manifest_identity_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical immutable SCC input identity shared by staging and recovery.

    The stored fields intentionally retain the existing SQLite schema so that
    historical journals remain readable without migration or state resets.
    """
    from milgrau.level0.scc_staging import manifest_raw_name

    files = manifest["files"]
    raw_name = manifest_raw_name(manifest)
    return {
        "scientific_session_id": str(manifest["scientific_session_id"]),
        "configuration_id": int(manifest["scc_configuration_id"]),
        "raw_sha256": str(files[raw_name]["sha256"]),
        "lr_sha256": next(
            (str(v["sha256"]) for k, v in files.items() if k.startswith("lr_")), None,
        ),
        "sounding_sha256": next(
            (str(v["sha256"]) for k, v in files.items() if k.startswith("rs_")), None,
        ),
        "product_ids_json": json.dumps(
            [int(x) for x in manifest["scc_product_ids"]], separators=(",", ":"),
        ),
        "source_segments_json": json.dumps(
            manifest.get("source_segments", []), separators=(",", ":"),
        ),
        "scc_config_sha256": manifest.get("scc_config_sha256"),
        "start_utc": str(manifest["measurement_start_utc"]),
        "stop_utc": str(manifest["measurement_stop_utc"]),
    }


@contextmanager
def operation_lock(root: Path, identity: str) -> Iterator[None]:
    """OS-exclusive lock file: no implicit deletion of crash remnants."""
    identity = _id(identity)
    locks = root / "locks"
    locks.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = locks / f"{identity}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"SCC {identity} has an active/stale lock at {lock}. "
            "Investigate previous process before manually clearing it."
        ) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()} started={_utcnow()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        lock.unlink(missing_ok=True)


class SubmissionStore:
    """SQLite WAL journal with one current entry per identity and append-only events."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = self.root / "submissions.sqlite"
        self._conn = sqlite3.connect(self.path, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                measurement_id TEXT PRIMARY KEY,
                scientific_session_id TEXT NOT NULL,
                configuration_id INTEGER NOT NULL,
                raw_sha256 TEXT NOT NULL,
                lr_sha256 TEXT,
                sounding_sha256 TEXT,
                product_ids_json TEXT NOT NULL,
                source_segments_json TEXT NOT NULL,
                scc_config_sha256 TEXT,
                start_utc TEXT NOT NULL,
                stop_utc TEXT NOT NULL,
                state TEXT NOT NULL,
                remote_creation_date TEXT,
                archive_path TEXT,
                archive_manifest_sha256 TEXT,
                results_path TEXT,
                results_manifest_sha256 TEXT,
                last_error TEXT,
                updated_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                measurement_id TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                action TEXT NOT NULL,
                state TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        # Open earlier 12.5 journals in place (no destructive migration).
        existing = {
            str(row[1]) for row in self._conn.execute("PRAGMA table_info(submissions)")
        }
        for column in ("results_path", "results_manifest_sha256", "sounding_sha256"):
            if column not in existing:
                self._conn.execute(
                    f"ALTER TABLE submissions ADD COLUMN {column} TEXT"
                )
        self._conn.commit()
        # chmod is best-effort on Windows; POSIX storage must remain private.
        if os.name == "posix":
            self.path.chmod(0o600)

    def __enter__(self) -> "SubmissionStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self._conn.close()

    def get(self, identity: str) -> dict[str, Any] | None:
        result = self._conn.execute(
            "SELECT * FROM submissions WHERE measurement_id = ?", (_id(identity),),
        ).fetchone()
        return dict(result) if result is not None else None

    def list(self) -> list[dict[str, Any]]:
        return [dict(x) for x in self._conn.execute(
            "SELECT * FROM submissions ORDER BY measurement_id",
        ).fetchall()]

    def events(self, identity: str) -> list[dict[str, Any]]:
        return [dict(x) for x in self._conn.execute(
            "SELECT * FROM events WHERE measurement_id = ? ORDER BY seq",
            (_id(identity),),
        ).fetchall()]

    def matches_manifest(self, identity: str, manifest: Mapping[str, Any]) -> bool:
        """Require every immutable field to match the existing journal revision."""
        row = self.get(identity)
        if row is None or str(manifest["measurement_id"]) != identity:
            return False
        expected = manifest_identity_fields(manifest)
        return all(row[field] == value for field, value in expected.items())

    def current_revision_events(self, identity: str) -> list[dict[str, Any]]:
        """Exclude events from older accepted revisions when reconciling uncertainty.

        Historical event rows remain untouched; a new explicit revision has
        its own initial/new revision marker and its own audit evidence.
        """
        events = self.events(identity)
        for index in range(len(events) - 1, -1, -1):
            if events[index]["action"] in {"initial_stage", "new_explicit_revision"}:
                return events[index:]
        return events

    def begin(
        self, manifest: Mapping[str, Any], *, allow_new_revision: bool = False,
        allow_same_revision: bool = False,
    ) -> dict[str, Any]:
        """Insert once; prevent changed cached bytes from silently resetting state."""
        identity = _id(str(manifest["measurement_id"]))
        expected = manifest_identity_fields(manifest)
        row = self.get(identity)
        if row is not None:
            if row["state"] in BLOCKED_RESTART:
                raise RuntimeError(
                    f"SCC {identity} recorded as {row['state']}; manual remote "
                    "reconciliation is required before any new operation."
                )
            if row["state"] in {"accepted", "processing", "completed", "downloaded", "cleaned"}:
                if not allow_new_revision:
                    raise RuntimeError(
                        f"SCC {identity} was already submitted; use an explicit "
                        "replacement workflow for new contents."
                    )
            changed = any(row[key] != value for key, value in expected.items())
            if changed or (
                allow_new_revision and allow_same_revision
                and row["state"] in {
                    "accepted", "processing", "completed", "downloaded", "cleaned",
                }
            ):
                if not allow_new_revision:
                    raise RuntimeError(
                        f"SCC {identity} has different source bytes/period from "
                        "the durable journal; explicit replacement is required."
                    )
                previous = {key: row[key] for key in expected}
                with self._conn:
                    self._conn.execute(
                        """
                        UPDATE submissions SET
                            scientific_session_id=?, configuration_id=?,
                            raw_sha256=?, lr_sha256=?, sounding_sha256=?, product_ids_json=?,
                            source_segments_json=?, scc_config_sha256=?,
                            start_utc=?, stop_utc=?, state='pending',
                            archive_path=NULL, archive_manifest_sha256=NULL,
                            remote_creation_date=NULL, results_path=NULL,
                            results_manifest_sha256=NULL, last_error=NULL,
                            updated_utc=? WHERE measurement_id=?
                        """,
                        (*expected.values(), _utcnow(), identity),
                    )
                    self._add_event(identity, "new_explicit_revision", "pending", {
                        "prior": previous, "new": expected,
                    })
                return self.get(identity) or {}
            if row["state"] in {"accepted", "processing", "completed", "downloaded", "cleaned"}:
                raise RuntimeError(
                    f"SCC {identity} local content already submitted; "
                    "no duplicate upload without a new revision."
                )
            return row
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO submissions(
                    measurement_id, scientific_session_id, configuration_id,
                    raw_sha256, lr_sha256, sounding_sha256, product_ids_json,
                    source_segments_json, scc_config_sha256,
                    start_utc, stop_utc, state, updated_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (identity, *expected.values(), _utcnow()),
            )
            self._add_event(identity, "initial_stage", "pending", expected)
        return self.get(identity) or {}

    def record_local_lr_repair(
        self, identity: str,
        original: Mapping[str, Any], revised: Mapping[str, Any],
    ) -> None:
        """Audit an explicitly requested offline LR filename correction.

        Preserve current SCC state, archive, remote creation, error and prior
        events. Never turn an uncertain/accepted remote POST into a new
        upload opportunity merely because its local NetCDF was revised.
        """
        from milgrau.level0.scc_lidar_ratio import scc_lr_file_name
        from milgrau.level0.scc_staging import manifest_raw_name

        row = self.get(identity)
        if row is None:
            return  # unsent pending cache; no durable upload ownership
        new_lr = scc_lr_file_name(identity)
        old_lr = f"lr_{identity}.nc"
        raw_name = manifest_raw_name(original)
        revised_name = manifest_raw_name(revised)
        for key in (
            "scientific_session_id", "scc_configuration_id",
            "measurement_start_utc", "measurement_stop_utc",
            "scc_product_ids", "source_segments", "scc_config_sha256",
        ):
            if original.get(key) != revised.get(key):
                raise ValueError(f"Scientific provenance changed during LR repair: {key}")
        if (
            old_lr not in original["files"]
            or new_lr not in revised["files"]
            or original["files"][old_lr] != revised["files"][new_lr]
        ):
            raise ValueError("SCC LR source bytes changed during filename repair.")
        if revised_name != raw_name or revised_name != f"{identity}.nc":
            raise ValueError("SCC staged raw identity changed during LR repair.")
        new_digest = revised["files"][revised_name]["sha256"]
        old_digest = original["files"][raw_name]["sha256"]
        lr_digest = original["files"][old_lr]["sha256"]
        if row["raw_sha256"] == new_digest and row["lr_sha256"] == lr_digest:
            return
        if row["raw_sha256"] != old_digest or row["lr_sha256"] != lr_digest:
            raise ValueError(
                "SCC journal source hashes are not the documented original "
                "LR-rename revision; will not rewrite provenance."
            )
        if row["state"] not in {"pending", "remote_exists", "failed", "uncertain"}:
            raise ValueError("SCC submission state is not eligible for local LR repair.")
        events = self.events(identity)
        last_action = events[-1]["action"] if events else None
        if row["state"] == "uncertain" and last_action != "manual_corrected_upload_uncertain":
            raise ValueError(
                "Cannot revise uncertain SCC input after an unrecognized remote "
                f"event: {last_action}. Reconcile manually."
            )
        with self._conn:
            self._conn.execute(
                """UPDATE submissions SET raw_sha256=?, lr_sha256=?, updated_utc=?
                   WHERE measurement_id=?""",
                (new_digest, lr_digest, _utcnow(), identity),
            )
            self._add_event(
                identity, "local_lr_filename_repair", row["state"],
                {
                    "previous_action": last_action,
                    "old_lr_filename": old_lr, "new_lr_filename": new_lr,
                    "old_raw_sha256": old_digest, "new_raw_sha256": new_digest,
                    "unchanged_lr_sha256": lr_digest,
                    "remote_post": False, "remote_delete": False,
                },
            )

    def _add_event(
        self, identity: str, action: str, state: str, payload: Mapping[str, Any],
    ) -> None:
        self._conn.execute(
            """INSERT INTO events(measurement_id, timestamp_utc, action, state, payload_json)
            VALUES(?, ?, ?, ?, ?)""",
            (identity, _utcnow(), action, state, json.dumps(dict(payload), sort_keys=True)),
        )

    def transition(
        self, identity: str, state: str, *, action: str,
        detail: Mapping[str, Any] | None = None,
        archive_path: str | None = None,
        archive_manifest_sha256: str | None = None,
        remote_creation_date: str | None = None,
        results_path: str | None = None,
        results_manifest_sha256: str | None = None,
        error: str | None = None,
    ) -> None:
        identity = _id(identity)
        if state not in STATES:
            raise ValueError(f"Unknown SCC submission state {state!r}.")
        if self.get(identity) is None:
            raise ValueError(f"Unknown SCC submission {identity}.")
        with self._conn:
            self._conn.execute(
                """
                UPDATE submissions SET state = ?, updated_utc = ?,
                archive_path = COALESCE(?, archive_path),
                archive_manifest_sha256 = COALESCE(?, archive_manifest_sha256),
                remote_creation_date = COALESCE(?, remote_creation_date),
                results_path = COALESCE(?, results_path),
                results_manifest_sha256 = COALESCE(?, results_manifest_sha256),
                last_error = ? WHERE measurement_id = ?
                """,
                (
                    state, _utcnow(), archive_path, archive_manifest_sha256,
                    remote_creation_date, results_path, results_manifest_sha256,
                    error, identity,
                ),
            )
            self._add_event(identity, action, state, detail or {})
