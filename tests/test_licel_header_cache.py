"""Rebuildable Licel header cache: no scientific identity persisted."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from milgrau.level0.header_cache import LicelHeaderCache


def test_header_cache_reuses_unchanged_files_and_detects_modified_inputs(tmp_path: Path) -> None:
    file = tmp_path / "raw.dat"
    file.write_bytes(b"initial")
    parsed = []
    header = (datetime(2020, 1, 7, 11), datetime(2020, 1, 7, 12), 3600., 120, 20)

    def read(path):
        parsed.append(path)
        return header

    root = tmp_path / "cache"
    for _ in range(2):
        with LicelHeaderCache(root, logging.getLogger("test.cache")) as cache:
            assert cache.header(str(file), "measurements", read) == header
    assert len(parsed) == 1
    file.write_bytes(b"changed size")
    with LicelHeaderCache(root, logging.getLogger("test.cache")) as cache:
        assert cache.header(str(file), "measurements", read) == header
    assert len(parsed) == 2


def test_header_cache_remembers_invalid_and_prunes_removed_files(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid"
    invalid.write_bytes(b"not licel")
    reads = []

    def read(path):
        reads.append(path)
        return None, None, None, None, None

    root = tmp_path / "cache"
    for _ in range(2):
        with LicelHeaderCache(root, logging.getLogger("test.cache.invalid")) as cache:
            assert cache.header(str(invalid), "measurements", read)[0] is None
    assert len(reads) == 1
    invalid.unlink()
    with LicelHeaderCache(root, logging.getLogger("test.cache.invalid")):
        pass
    invalid.write_bytes(b"recreated")
    with LicelHeaderCache(root, logging.getLogger("test.cache.invalid")) as cache:
        assert cache.header(str(invalid), "measurements", read)[0] is None
    assert len(reads) == 2


def test_header_cache_keeps_current_classification_outside_cache(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.write_bytes(b"file")
    events = []

    def reader(path):
        events.append(path)
        return datetime(2020, 1, 1), datetime(2020, 1, 1, 1), 3600., 1, 1

    with LicelHeaderCache(tmp_path / "cache", logging.getLogger("test.cache.kind")) as cache:
        first = cache.header(str(raw), "measurements", reader)
        second = cache.header(str(raw), "dark_current", reader)
    assert first == second
    assert len(events) == 1
