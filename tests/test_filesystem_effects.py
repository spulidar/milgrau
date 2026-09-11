"""Tests for read-only discovery and explicit filesystem actions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from milgrau.io.filesystem import (
    RawFileKind,
    delete_file,
    discover_raw_files,
    quarantine_file,
    quarantine_files,
    scan_raw_files,
)
from milgrau.operations import ExecutionStatus


class _ListLogger:
    """Capture filesystem messages without configuring handlers."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, message: str) -> None:
        self.messages.append(f"DEBUG: {message}")

    def info(self, message: str) -> None:
        self.messages.append(f"INFO: {message}")

    def warning(self, message: str) -> None:
        self.messages.append(f"WARNING: {message}")

    def error(self, message: str) -> None:
        self.messages.append(f"ERROR: {message}")


def _scan_kwargs(raw_root: Path) -> dict:
    return {
        "spurious_extensions": [".zip"],
        "raw_scan_ignore_dirs": [],
        "quarantine_dir": raw_root.parent / "quarantine",
    }


def test_scan_is_read_only_and_reports_spurious_detection(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    measurement = raw_root / "measurement_001"
    dark_current = raw_root / "dark_current_001"
    spurious = raw_root / "archive.zip"
    measurement.write_text("measurement", encoding="utf-8")
    dark_current.write_text("dark", encoding="utf-8")
    spurious.write_text("archive", encoding="utf-8")
    logger = _ListLogger()

    paths, types = scan_raw_files(raw_root, logger=logger, **_scan_kwargs(raw_root))

    assert list(zip(map(Path, paths), types)) == [
        (dark_current, "dark_current"),
        (measurement, "measurements"),
    ]
    assert spurious.read_text(encoding="utf-8") == "archive"
    assert not (tmp_path / "quarantine").exists()
    assert any("Spurious file detected; no action taken" in message for message in logger.messages)


def test_discovery_exposes_validation_classification_without_side_effects(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    spurious = raw_root / "archive.zip"
    measurement = raw_root / "measurement"
    spurious.write_text("archive", encoding="utf-8")
    measurement.write_text("measurement", encoding="utf-8")

    candidates = discover_raw_files(raw_root, **_scan_kwargs(raw_root))

    assert [(candidate.path, candidate.kind) for candidate in candidates] == [
        (spurious, RawFileKind.SPURIOUS),
        (measurement, RawFileKind.MEASUREMENT),
    ]
    assert "configured as spurious" in candidates[0].reason
    assert spurious.exists() and measurement.exists()


def test_discovery_honors_explicit_ignore_dirs_and_quarantine_path(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    ignored = raw_root / "ignore-me"
    ignored.mkdir()
    ignored_file = ignored / "hidden"
    ignored_file.write_text("hidden", encoding="utf-8")
    visible = raw_root / "visible"
    visible.write_text("visible", encoding="utf-8")

    quarantine_root = raw_root / "quarantine"
    quarantine_root.mkdir()
    quarantined = quarantine_root / "old.zip"
    quarantined.write_text("old", encoding="utf-8")

    candidates = discover_raw_files(
        raw_root,
        spurious_extensions=[".zip"],
        raw_scan_ignore_dirs=["ignore-me"],
        quarantine_dir=quarantine_root,
    )

    assert [candidate.path for candidate in candidates] == [visible]


def test_discovery_ignores_legacy_weather_cache_and_json_artifacts(tmp_path: Path) -> None:
    raw_root = tmp_path / "01-data"
    raw_root.mkdir()
    legacy_cache = raw_root / "openmeteo_cache" / "2025" / "07"
    legacy_cache.mkdir(parents=True)
    (legacy_cache / "openmeteo.json").write_text("{}", encoding="utf-8")
    raw_measurement = raw_root / "measurement_001"
    raw_measurement.write_text("measurement", encoding="utf-8")
    loose_json = raw_root / "metadata.json"
    loose_json.write_text("{}", encoding="utf-8")

    candidates = discover_raw_files(
        raw_root,
        spurious_extensions=[".zip", ".json"],
        raw_scan_ignore_dirs=["openmeteo_cache", "wyoming_cache"],
        quarantine_dir=tmp_path / "quarantine",
    )

    assert [(candidate.path, candidate.kind) for candidate in candidates] == [
        (raw_measurement, RawFileKind.MEASUREMENT),
        (loose_json, RawFileKind.SPURIOUS),
    ]


def test_explicit_quarantine_uses_dated_reason_bucket_and_audit_sidecar(tmp_path: Path) -> None:
    first = tmp_path / "one" / "archive.zip"
    second = tmp_path / "two" / "archive.zip"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    quarantine_root = tmp_path / "quarantine"
    logger = _ListLogger()
    when = datetime(2026, 9, 11, 12, 30, tzinfo=timezone.utc)

    summary = quarantine_files(
        [first, second],
        quarantine_root,
        logger,
        reason="invalid Licel header",
        stage="level0.inventory",
        measurement_id="20260911am",
        quarantined_at_utc=when,
    )
    repeat = quarantine_files(
        [first, second],
        quarantine_root,
        logger,
        reason="invalid Licel header",
        stage="level0.inventory",
        measurement_id="20260911am",
        quarantined_at_utc=when,
    )

    bucket = quarantine_root / "2026" / "09" / "11" / "invalid-licel-header"
    digest = sha256(str(second.absolute()).encode("utf-8")).hexdigest()[:12]
    first_destination = bucket / "archive.zip"
    second_destination = bucket / f"archive_{digest}.zip"
    assert [result.status for result in summary.results] == [ExecutionStatus.OK, ExecutionStatus.OK]
    assert summary.results[0].output_path == first_destination
    assert summary.results[1].output_path == second_destination
    assert first_destination.read_text(encoding="utf-8") == "first"
    assert second_destination.read_text(encoding="utf-8") == "second"

    first_sidecar = json.loads((bucket / "archive.zip.json").read_text(encoding="utf-8"))
    second_sidecar = json.loads((bucket / f"archive_{digest}.zip.json").read_text(encoding="utf-8"))
    assert first_sidecar["schema_version"] == 1
    assert first_sidecar["reason"] == "invalid Licel header"
    assert first_sidecar["stage"] == "level0.inventory"
    assert first_sidecar["measurement_id"] == "20260911am"
    assert first_sidecar["sha256"] == sha256(b"first").hexdigest()
    assert first_sidecar["size_bytes"] == 5
    assert second_sidecar["sha256"] == sha256(b"second").hexdigest()
    assert all(result.status is ExecutionStatus.SKIPPED for result in repeat.results)
    assert any("audit sidecar" in message for message in logger.messages)


def test_quarantine_requires_explicit_reason(tmp_path: Path) -> None:
    target = tmp_path / "archive.zip"
    target.write_text("archive", encoding="utf-8")

    try:
        quarantine_file(target, tmp_path / "quarantine", reason="")
    except ValueError as exc:
        assert "reason" in str(exc).lower()
    else:
        raise AssertionError("Empty quarantine reason should fail.")
    assert target.exists()


def test_explicit_delete_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "archive.zip"
    target.write_text("archive", encoding="utf-8")

    deleted = delete_file(target)
    repeated = delete_file(target)

    assert deleted.status is ExecutionStatus.OK
    assert repeated.status is ExecutionStatus.SKIPPED
    assert not target.exists()


def test_action_permission_failure_is_structured_and_leaves_file(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "archive.zip"
    target.write_text("archive", encoding="utf-8")

    def deny_unlink(_path: Path) -> None:
        raise PermissionError("synthetic permission denial")

    monkeypatch.setattr(Path, "unlink", deny_unlink)

    result = delete_file(target)

    assert result.status is ExecutionStatus.ERROR
    assert isinstance(result.cause, PermissionError)
    assert target.exists()


def test_actions_reject_directories_without_recursive_mutation(tmp_path: Path) -> None:
    target = tmp_path / "directory"
    target.mkdir()
    child = target / "child"
    child.write_text("keep", encoding="utf-8")

    quarantine_result = quarantine_file(
        target,
        tmp_path / "quarantine",
        reason="not a regular file",
    )
    delete_result = delete_file(target)

    assert quarantine_result.status is ExecutionStatus.ERROR
    assert delete_result.status is ExecutionStatus.ERROR
    assert child.read_text(encoding="utf-8") == "keep"


def test_scan_raw_files_missing_root_has_no_side_effects(tmp_path: Path) -> None:
    raw_root = tmp_path / "missing"
    logger = _ListLogger()

    paths, types = scan_raw_files(raw_root, logger=logger, **_scan_kwargs(raw_root))

    assert paths == []
    assert types == []
    assert not raw_root.exists()
    assert logger.messages == [f"ERROR: Raw data directory not found: {raw_root}"]
