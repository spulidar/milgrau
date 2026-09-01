"""Tests for simple timestamp-based incremental reuse."""

from __future__ import annotations

import os
from pathlib import Path

from milgrau.incremental import output_is_current


def _set_mtime(path: Path, ns: int) -> None:
    os.utime(path, ns=(ns, ns))


def test_output_must_exist_and_be_nonempty(tmp_path: Path) -> None:
    source = tmp_path / "input.dat"
    source.write_text("input", encoding="utf-8")
    output = tmp_path / "output.dat"

    assert not output_is_current(output, [source], include_code=False)
    output.write_bytes(b"")
    assert not output_is_current(output, [source], include_code=False)


def test_newer_valid_output_is_current(tmp_path: Path) -> None:
    source = tmp_path / "input.dat"
    output = tmp_path / "output.dat"
    source.write_text("input", encoding="utf-8")
    output.write_text("output", encoding="utf-8")
    _set_mtime(source, 1_000_000_000)
    _set_mtime(output, 2_000_000_000)

    assert output_is_current(output, [source], include_code=False)


def test_newer_input_invalidates_output(tmp_path: Path) -> None:
    source = tmp_path / "input.dat"
    output = tmp_path / "output.dat"
    source.write_text("input", encoding="utf-8")
    output.write_text("output", encoding="utf-8")
    _set_mtime(output, 1_000_000_000)
    _set_mtime(source, 2_000_000_000)

    assert not output_is_current(output, [source], include_code=False)


def test_config_file_change_invalidates_output(tmp_path: Path) -> None:
    source = tmp_path / "input.dat"
    config_file = tmp_path / "config.yaml"
    output = tmp_path / "output.dat"
    source.write_text("input", encoding="utf-8")
    config_file.write_text("processing: {}", encoding="utf-8")
    output.write_text("output", encoding="utf-8")
    _set_mtime(source, 1_000_000_000)
    _set_mtime(output, 2_000_000_000)
    _set_mtime(config_file, 3_000_000_000)

    assert not output_is_current(
        output,
        [source],
        config={"_config_file": str(config_file)},
        include_code=False,
    )


def test_integrity_check_can_force_reprocessing(tmp_path: Path) -> None:
    source = tmp_path / "input.dat"
    output = tmp_path / "output.dat"
    source.write_text("input", encoding="utf-8")
    output.write_text("output", encoding="utf-8")
    _set_mtime(source, 1_000_000_000)
    _set_mtime(output, 2_000_000_000)

    assert not output_is_current(
        output,
        [source],
        integrity_check=lambda _path: False,
        include_code=False,
    )
