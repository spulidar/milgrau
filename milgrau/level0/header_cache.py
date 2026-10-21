"""Rebuildable Licel *header* cache, never a source of scientific session identity.

The canonical inventory still resolves station profiles, session boundaries,
dark-current association and QA on every run from cached header facts.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable

# Increment after any change to read_licel_header's interpretation.
HEADER_PARSER_REVISION = 1
Header = tuple[datetime | None, datetime | None, float | None, int | None, int | None]


class LicelHeaderCache:
    """File-signature-indexed cache; a failed cache never prevents raw parsing."""

    def __init__(self, directory: Path, logger: logging.Logger) -> None:
        self.logger = logger
        self.connection: sqlite3.Connection | None = None
        self.hits = 0
        self.misses = 0
        self.directory = directory

    def __enter__(self) -> "LicelHeaderCache":
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.directory / "inventory.sqlite")
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS headers ("
                "path TEXT PRIMARY KEY, size INTEGER NOT NULL, "
                "mtime_ns INTEGER NOT NULL, ctime_ns INTEGER NOT NULL, "
                "parser_revision INTEGER NOT NULL, payload TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE TEMP TABLE seen (path TEXT PRIMARY KEY)"
            )
        except (OSError, sqlite3.Error) as exc:
            self.logger.warning("Licel header cache unavailable; using raw headers: %s", exc)
            if self.connection is not None:
                self.connection.close()
            self.connection = None
        return self

    def header(
        self,
        filepath: str,
        file_type: str,
        read_header: Callable[[str], Header],
    ) -> Header:
        del file_type  # Classification is derived from each current filesystem scan.
        if self.connection is None:
            return read_header(filepath)
        try:
            path = Path(filepath).resolve()
            stat = path.stat()
            key = str(path)
            self.connection.execute(
                "INSERT OR IGNORE INTO seen (path) VALUES (?)", (key,)
            )
            row = self.connection.execute(
                "SELECT payload FROM headers WHERE path=? AND size=? "
                "AND mtime_ns=? AND ctime_ns=? AND parser_revision=?",
                (key, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                 HEADER_PARSER_REVISION),
            ).fetchone()
            if row is not None:
                fields = json.loads(row[0])
                self.hits += 1
                return (
                    *(datetime.fromisoformat(v) if v is not None else None
                      for v in fields[:2]),
                    *fields[2:],
                )
        except (OSError, sqlite3.Error, ValueError, TypeError, json.JSONDecodeError):
            # Unsupported tests/nonexistent paths and cache failures use the
            # authoritative parser. Never infer scientific values from filenames.
            return read_header(filepath)

        parsed = read_header(filepath)
        self.misses += 1
        try:
            fields = [
                value.isoformat() if index < 2 and value is not None else value
                for index, value in enumerate(parsed)
            ]
            self.connection.execute(
                "INSERT OR REPLACE INTO headers "
                "(path,size,mtime_ns,ctime_ns,parser_revision,payload) "
                "VALUES (?,?,?,?,?,?)",
                (key, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                 HEADER_PARSER_REVISION, json.dumps(fields)),
            )
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            self.logger.debug("Licel header cache write ignored: %s", exc)
        return parsed

    def __exit__(self, _type, _value, _traceback) -> None:
        if self.connection is not None:
            try:
                # A complete filesystem walk supplies the set of surviving files.
                # Do not prune on an interrupted/failed walk.
                if _type is None:
                    self.connection.execute(
                        "DELETE FROM headers WHERE path NOT IN (SELECT path FROM seen)"
                    )
                    self.connection.commit()
                else:
                    self.connection.rollback()
            except sqlite3.Error as exc:
                self.logger.warning("Licel header cache commit failed: %s", exc)
            finally:
                self.connection.close()
        self.logger.info(
            "Licel header cache | %d reused | %d parsed",
            self.hits, self.misses,
        )
