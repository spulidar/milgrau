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
    channel_separators: bool = False,
    standard_global_header: bool = True,
) -> Path:
    if channels is None:
        channels = [
            {"active": 1, "is_pc": 0, "laser_used": 1, "points": 4, "bin_width_m": 7.5, "wavelength": "532.0", "adc_bits": 12, "shots": n_shots, "adc_range_v": 0.5, "payload": [100, 200, 300, 400]},
            {"active": 1, "is_pc": 1, "laser_used": 1, "points": 4, "bin_width_m": 7.5, "wavelength": "532.0", "adc_bits": 0, "shots": n_shots, "adc_range_v": 0.0, "payload": [1, 2, 3, 4]},
        ]
    line1 = "Synthetic Licel header\n"
    line2 = "HEADER 01/01/2024 00:00:00 01/01/2024 00:05:00\n"
    line3 = f"1 {laser_freq} 0 0 {len(channels)}\n" if standard_global_header else f"SYNTH SYS {n_shots} {laser_freq} {len(channels)}\n"
    with path.open("wb") as handle:
        handle.write(line1.encode())
        handle.write(line2.encode())
        handle.write(line3.encode())
        for channel in channels:
            fields = [
                str(channel["active"]), str(channel["is_pc"]), str(channel.get("laser_used", 1)),
                str(channel["points"]), "1", "800", str(channel.get("bin_width_m", 7.5)),
                str(channel["wavelength"]), "0", "0", "0", "0", str(channel["adc_bits"]),
                str(channel.get("shots", n_shots)), str(channel["adc_range_v"]), "BT00",
            ]
            handle.write((" ".join(fields) + "\n").encode())
        handle.write(b"\n")
        active_channels = [channel for channel in channels if channel["active"]]
        if channel_separators:
            for channel in active_channels:
                np.asarray(channel["payload"], dtype="<i4").tofile(handle)
                handle.write(b"\r\n")
        else:
            np.concatenate([np.asarray(channel["payload"], dtype="<i4") for channel in active_channels]).tofile(handle)
    return path


def test_read_licel_header_reads_standard_channel_shots_and_laser_rate(tmp_path: Path) -> None:
    path = _write_synthetic_licel_file(tmp_path / "basic.licel")
    start, stop, duration, n_shots, laser_freq = read_licel_header(str(path), logger=logging.getLogger("test"))
    assert start is not None and stop is not None
    assert duration == 300.0
    assert n_shots == 100
    assert laser_freq == 20


def test_read_licel_header_keeps_legacy_global_shot_fallback(tmp_path: Path) -> None:
    channels = [{"active": 1, "is_pc": 1, "points": 4, "bin_width_m": 7.5, "wavelength": "532.0", "adc_bits": 0, "shots": 0, "adc_range_v": 0.0, "payload": [1, 2, 3, 4]}]
    path = _write_synthetic_licel_file(tmp_path / "legacy.licel", n_shots=100, channels=channels, standard_global_header=False)
    _, _, _, n_shots, laser_freq = read_licel_header(str(path), logger=logging.getLogger("test"))
    assert n_shots == 100
    assert laser_freq == 20


def test_parse_single_licel_file_converts_analog_with_full_adc_span(tmp_path: Path) -> None:
    path = _write_synthetic_licel_file(tmp_path / "single.licel")
    parsed = parse_single_licel_file(str(path))
    assert parsed["shots_by_channel"] == {"532.AN": 100, "532.PC": 100}
    assert parsed["payload_samples_used"] == 8
    expected = np.array([100, 200, 300, 400], dtype=float) / 100.0 * (500.0 / 4095.0)
    np.testing.assert_allclose(parsed["data"]["532.AN"], expected)
    np.testing.assert_array_equal(parsed["data"]["532.PC"], np.array([1.0, 2.0, 3.0, 4.0]))
    analog = next(channel for channel in parsed["channels_meta"] if channel["name"] == "532.AN")
    pc = next(channel for channel in parsed["channels_meta"] if channel["name"] == "532.PC")
    assert analog["daq_range_mV"] == 500.0
    assert analog["bin_width_m"] == 7.5
    assert analog["acquisition_mode"] == 0
    assert np.isnan(pc["daq_range_mV"])
    assert pc["acquisition_mode"] == 1


def test_parse_single_licel_file_consumes_channel_separators(tmp_path: Path) -> None:
    path = _write_synthetic_licel_file(tmp_path / "separated.licel", channel_separators=True)
    parsed = parse_single_licel_file(str(path))
    assert parsed["extra_payload_samples"] == 0
    expected = np.array([100, 200, 300, 400], dtype=float) / 100.0 * (500.0 / 4095.0)
    np.testing.assert_allclose(parsed["data"]["532.AN"], expected)
    np.testing.assert_array_equal(parsed["data"]["532.PC"], np.array([1.0, 2.0, 3.0, 4.0]))


def test_parse_licel_group_preserves_profile_channel_shots_and_metadata(tmp_path: Path) -> None:
    first = _write_synthetic_licel_file(tmp_path / "a.licel", n_shots=100)
    second_channels = [
        {"active": 1, "is_pc": 0, "points": 4, "bin_width_m": 7.5, "wavelength": "532.0", "adc_bits": 12, "shots": 120, "adc_range_v": 0.5, "payload": [100, 200, 300, 400]},
        {"active": 1, "is_pc": 1, "points": 4, "bin_width_m": 7.5, "wavelength": "532.0", "adc_bits": 0, "shots": 240, "adc_range_v": 0.0, "payload": [1, 2, 3, 4]},
    ]
    second = _write_synthetic_licel_file(tmp_path / "b.licel", channels=second_channels)
    parsed = parse_licel_group([str(first), str(second)], logging.getLogger("test"))
    assert parsed["channels"] == ["532.AN", "532.PC"]
    np.testing.assert_array_equal(parsed["laser_shots"], np.array([[100, 100], [120, 240]], dtype=np.int32))
    assert parsed["channel_metadata"]["532.AN"]["daq_range_mV"] == 500.0
    assert parsed["channel_metadata"]["532.AN"]["bin_width_m"] == 7.5


def test_parse_licel_group_skips_incompatible_files(tmp_path: Path) -> None:
    valid_a = _write_synthetic_licel_file(tmp_path / "a.licel")
    valid_b = _write_synthetic_licel_file(tmp_path / "b.licel")
    invalid = _write_synthetic_licel_file(
        tmp_path / "c.licel",
        channels=[{"active": 1, "is_pc": 0, "points": 4, "bin_width_m": 7.5, "wavelength": "355.0", "adc_bits": 12, "shots": 100, "adc_range_v": 0.5, "payload": [1, 2, 3, 4]}],
    )
    parsed = parse_licel_group([str(valid_a), str(invalid), str(valid_b)], logging.getLogger("test"))
    assert parsed["shots"] == 100
    assert parsed["channels"] == ["532.AN", "532.PC"]
    assert parsed["tensors"]["532.AN"].shape == (2, 4)
    assert parsed["laser_shots"].shape == (2, 2)
