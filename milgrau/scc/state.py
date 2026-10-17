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
    "completed", "failed", "uncertain",
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
    staging = settings.get("staging_dir", ".cache/scc")
    cache = resolve_project_path(staging).resolve()
    if root == cache or cache in root.parents or root in cache.parents:
        raise ValueError("Durable SCC state directory must be independent of staging cache.")
    return root


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(identity: str) -> str:
    from milgrau.level0.scc_staging import _MEASUREMENT_ID

    if not _MEASUREMENT_ID.fullmatch(identity):
        raise ValueError(f"Invalid SCC Measurement_ID: {identity!r}.")
    return identity


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
                product_ids_json TEXT NOT NULL,
                source_segments_json TEXT NOT NULL,
                scc_config_sha256 TEXT,
                start_utc TEXT NOT NULL,
                stop_utc TEXT NOT NULL,
                state TEXT NOT NULL,
                remote_creation_date TEXT,
                archive_path TEXT,
                archive_manifest_sha256 TEXT,
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

    def begin(
        self, manifest: Mapping[str, Any], *, allow_new_revision: bool = False,
    ) -> dict[str, Any]:
        """Insert once; prevent changed cached bytes from silently resetting state."""
        identity = _id(str(manifest["measurement_id"]))
        files = manifest["files"]
        expected = {
            "scientific_session_id": str(manifest["scientific_session_id"]),
            "configuration_id": int(manifest["scc_configuration_id"]),
            "raw_sha256": str(files["raw.nc"]["sha256"]),
            "lr_sha256": next(
                (str(v["sha256"]) for k, v in files.items() if k != "raw.nc"), None,
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
        row = self.get(identity)
        if row is not None:
            if row["state"] in BLOCKED_RESTART:
                raise RuntimeError(
                    f"SCC {identity} recorded as {row['state']}; manual remote "
                    "reconciliation is required before any new operation."
                )
            if row["state"] in {"accepted", "processing", "completed"}:
                if not allow_new_revision:
                    raise RuntimeError(
                        f"SCC {identity} was already submitted; use an explicit "
                        "replacement workflow for new contents."
                    )
            changed = any(row[key] != value for key, value in expected.items())
            if changed:
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
                            raw_sha256=?, lr_sha256=?, product_ids_json=?,
                            source_segments_json=?, scc_config_sha256=?,
                            start_utc=?, stop_utc=?, state='pending',
                            archive_path=NULL, archive_manifest_sha256=NULL,
                            remote_creation_date=NULL, last_error=NULL,
                            updated_utc=? WHERE measurement_id=?
                        """,
                        (*expected.values(), _utcnow(), identity),
                    )
                    self._add_event(identity, "new_explicit_revision", "pending", {
                        "prior": previous, "new": expected,
                    })
                return self.get(identity) or {}
            if row["state"] in {"accepted", "processing", "completed"}:
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
                    raw_sha256, lr_sha256, product_ids_json,
                    source_segments_json, scc_config_sha256,
                    start_utc, stop_utc, state, updated_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (identity, *expected.values(), _utcnow()),
            )
            self._add_event(identity, "initial_stage", "pending", expected)
        return self.get(identity) or {}

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
                last_error = ? WHERE measurement_id = ?
                """,
                (
                    state, _utcnow(), archive_path, archive_manifest_sha256,
                    remote_creation_date, error, identity,
                ),
            )
            self._add_event(identity, action, state, detail or {})
