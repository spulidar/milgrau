"""Tests for read-only discovery and explicit filesystem actions."""

from __future__ import annotations

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


def test_explicit_quarantine_is_idempotent_and_collision_destination_is_auditable(tmp_path: Path) -> None:
    first = tmp_path / "one" / "archive.zip"
    second = tmp_path / "two" / "archive.zip"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    quarantine_root = tmp_path / "quarantine"
    logger = _ListLogger()

    summary = quarantine_files([first, second], quarantine_root, logger)
    repeat = quarantine_files([first, second], quarantine_root, logger)

    digest = sha256(str(second.absolute()).encode("utf-8")).hexdigest()[:12]
    assert [result.status for result in summary.results] == [ExecutionStatus.SUCCESS, ExecutionStatus.SUCCESS]
    assert summary.results[0].output_path == quarantine_root / "archive.zip"
    assert summary.results[1].output_path == quarantine_root / f"archive_{digest}.zip"
    assert (quarantine_root / "archive.zip").read_text(encoding="utf-8") == "first"
    assert (quarantine_root / f"archive_{digest}.zip").read_text(encoding="utf-8") == "second"
    assert all(result.status is ExecutionStatus.SKIPPED for result in repeat.results)
    assert any("File quarantined" in message for message in logger.messages)


def test_explicit_delete_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "archive.zip"
    target.write_text("archive", encoding="utf-8")

    deleted = delete_file(target)
    repeated = delete_file(target)

    assert deleted.status is ExecutionStatus.SUCCESS
    assert repeated.status is ExecutionStatus.SKIPPED
    assert not target.exists()


def test_action_permission_failure_is_structured_and_leaves_file(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "archive.zip"
    target.write_text("archive", encoding="utf-8")

    def deny_unlink(_path: Path) -> None:
        raise PermissionError("synthetic permission denial")

    monkeypatch.setattr(Path, "unlink", deny_unlink)

    result = delete_file(target)

    assert result.status is ExecutionStatus.RECOVERABLE_FAILURE
    assert isinstance(result.cause, PermissionError)
    assert target.exists()


def test_actions_reject_directories_without_recursive_mutation(tmp_path: Path) -> None:
    target = tmp_path / "directory"
    target.mkdir()
    child = target / "child"
    child.write_text("keep", encoding="utf-8")

    quarantine_result = quarantine_file(target, tmp_path / "quarantine")
    delete_result = delete_file(target)

    assert quarantine_result.status is ExecutionStatus.RECOVERABLE_FAILURE
    assert delete_result.status is ExecutionStatus.RECOVERABLE_FAILURE
    assert child.read_text(encoding="utf-8") == "keep"


def test_scan_raw_files_missing_root_has_no_side_effects(tmp_path: Path) -> None:
    raw_root = tmp_path / "missing"
    logger = _ListLogger()

    paths, types = scan_raw_files(raw_root, logger=logger, **_scan_kwargs(raw_root))

    assert paths == []
    assert types == []
    assert not raw_root.exists()
    assert logger.messages == [f"ERROR: Raw data directory not found: {raw_root}"]
