"""Regression tests for portable MILGRAU source-code provenance."""

from __future__ import annotations

from pathlib import Path

from milgrau.provenance import (
    SOURCE_CODE_IDENTITY_SCOPE,
    package_source_sha256,
    source_code_provenance,
)


def _write_source_tree(root: Path, *, line_ending: str = "\n") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.py").write_text(
        f"VALUE = 1{line_ending}", encoding="utf-8", newline=""
    )
    nested = root / "sub"
    nested.mkdir()
    (nested / "b.py").write_text(
        f"def f():{line_ending}    return 2{line_ending}",
        encoding="utf-8",
        newline="",
    )


def test_package_source_hash_is_stable_across_equivalent_line_endings(tmp_path: Path) -> None:
    unix_root = tmp_path / "unix"
    windows_root = tmp_path / "windows"
    _write_source_tree(unix_root, line_ending="\n")
    _write_source_tree(windows_root, line_ending="\r\n")

    assert package_source_sha256(unix_root) == package_source_sha256(windows_root)


def test_package_source_hash_changes_when_python_source_changes(tmp_path: Path) -> None:
    root = tmp_path / "package"
    _write_source_tree(root)
    first = package_source_sha256(root)

    (root / "a.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert package_source_sha256(root) != first


def test_source_code_provenance_uses_content_identity_without_git(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "package"
    _write_source_tree(root)
    monkeypatch.setenv("MILGRAU_SOURCE_REVISION", "abc123")

    result = source_code_provenance(package_root=root)

    assert result["source_code_identity_scope"] == SOURCE_CODE_IDENTITY_SCOPE
    assert result["source_code_identity"] == (
        f"sha256:{result['source_code_sha256']}"
    )
    assert result["source_repository_revision"] == "abc123"
    assert result["source_repository_revision_source"] == "environment"
