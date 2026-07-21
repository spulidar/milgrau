"""Tests for Licel header and payload parsing helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from milgrau.io.licel import parse_licel_group, parse_single_licel_file, read_licel_header


def _write_synthetic_licel_file(
    path: Path,
    *,
    n_shots: int = 100,
    laser_freq: int = 20,
    channels: list[dict] | None = None,
) -> Path:
    """Write a minimal synthetic Licel-like file for parser tests."""
    if channels is None:
        channels = [
            {
                "active": 1,
                "is_pc": 0,
                "points": 4,
                "wavelength": "532.0",
                "adc_bits": 12,
                "adc_range_v": 0.5,
                "payload": [100, 200, 300, 400],
            },
            {
                "active": 1,
                "is_pc": 1,
                "points": 4,
                "wavelength": "532.0",
                "adc_bits": 0,
                "adc_range_v": 0.0,
                "payload": [1, 2, 3, 4],
            },
        ]

    line1 = "Synthetic Licel header\n"
    line2 = "HEADER 01/01/2024 00:00:00 01/01/2024 00:05:00\n"
    line3 = f"SYNTH SYS {n_shots} {laser_freq} {len(channels)}\n"

    with path.open("wb") as handle:
        handle.write(line1.encode("utf-8"))
        handle.write(line2.encode("utf-8"))
        handle.write(line3.encode("utf-8"))
        for channel in channels:
            fields = [
                str(channel["active"]),
                str(channel["is_pc"]),
                "0",
                str(channel["points"]),
                "0",
                "0",
                "0",
                str(channel["wavelength"]),
                "0",
                "0",
                "0",
                "0",
                str(channel["adc_bits"]),
                "0",
                str(channel["adc_range_v"]),
            ]
            handle.write((" ".join(fields) + "\n").encode("utf-8"))
        handle.write(b"\n")
        payload = np.concatenate([np.asarray(channel["payload"], dtype=np.int32) for channel in channels if channel["active"]])
        payload.tofile(handle)

    return path


def test_read_licel_header_reads_basic_metadata(tmp_path: Path) -> None:
    """The lightweight header reader should extract times, duration and laser settings."""
    path = _write_synthetic_licel_file(tmp_path / "basic.licel")

    start, stop, duration, n_shots, laser_freq = read_licel_header(str(path), logger=logging.getLogger("test"))

    assert start is not None
    assert stop is not None
    assert duration == 300.0
    assert n_shots == 100
    assert laser_freq == 20


def test_parse_single_licel_file_converts_analog_and_pc_channels(tmp_path: Path) -> None:
    """Analog channels should be converted to mV/shot while PC channels stay in counts."""
    path = _write_synthetic_licel_file(tmp_path / "single.licel")

    parsed = parse_single_licel_file(str(path))

    assert parsed["laser_freq"] == 20
    assert parsed["shots"] == 100
    assert parsed["payload_samples_used"] == 8
    assert parsed["extra_payload_samples"] == 0
    assert set(parsed["data"]) == {"532.AN", "532.PC"}
    np.testing.assert_allclose(parsed["data"]["532.AN"], np.array([0.12207031, 0.24414062, 0.36621094, 0.48828125]))
    np.testing.assert_array_equal(parsed["data"]["532.PC"], np.array([1.0, 2.0, 3.0, 4.0]))


def test_parse_licel_group_skips_incompatible_files(tmp_path: Path) -> None:
    """Group parsing should keep compatible files and skip files with mismatched channels."""
    valid_a = _write_synthetic_licel_file(tmp_path / "a.licel")
    valid_b = _write_synthetic_licel_file(tmp_path / "b.licel")
    invalid = _write_synthetic_licel_file(
        tmp_path / "c.licel",
        channels=[
            {
                "active": 1,
                "is_pc": 0,
                "points": 4,
                "wavelength": "355.0",
                "adc_bits": 12,
                "adc_range_v": 0.5,
                "payload": [1, 2, 3, 4],
            }
        ],
    )

    parsed = parse_licel_group([str(valid_a), str(invalid), str(valid_b)], logging.getLogger("test"))

    assert parsed["shots"] == 100
    assert parsed["channels"] == ["532.AN", "532.PC"]
    assert parsed["tensors"]["532.AN"].shape == (2, 4)
    assert parsed["tensors"]["532.PC"].shape == (2, 4)
