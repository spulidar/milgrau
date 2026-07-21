"""Licel binary header and payload parsing utilities for SPU-Lidar."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mode
from typing import Final, Optional

import numpy as np

LICEL_DATETIME_PATTERN: Final[str] = r"\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}"
DEFAULT_ANALOG_ADC_BITS: Final[int] = 12
DEFAULT_ANALOG_ADC_RANGE_V: Final[float] = 0.5
BYTES_PER_PAYLOAD_SAMPLE: Final[int] = 4


def _split_header_parts(line: str, minimum_parts: int, label: str) -> list[str]:
    """Split one Licel header line and validate its minimum field count."""
    parts = line.split()
    if len(parts) < minimum_parts:
        raise ValueError(f"{label} is malformed: expected at least {minimum_parts} fields, got {len(parts)}.")
    return parts


def _extract_licel_datetimes(line2: str) -> tuple[datetime, datetime]:
    """Extract start/stop datetimes from the second Licel header line."""
    matches = re.findall(LICEL_DATETIME_PATTERN, line2)
    if len(matches) >= 2:
        start_time_str, stop_time_str = matches[0], matches[1]
    else:
        start_time_str = line2[8:27].strip()
        stop_time_str = line2[28:47].strip()
    return (
        datetime.strptime(start_time_str, "%d/%m/%Y %H:%M:%S"),
        datetime.strptime(stop_time_str, "%d/%m/%Y %H:%M:%S"),
    )


def _parse_global_header_line(line3: str, filepath: str) -> tuple[int, int, int]:
    """Parse the third Licel header line shared by all channels."""
    parts = _split_header_parts(line3, 5, f"Licel global header in {filepath}")
    n_shots = int(parts[2])
    laser_freq = int(parts[3])
    num_channels = int(parts[4])
    if n_shots <= 0:
        raise ValueError(f"Invalid number of laser shots in {filepath}: {n_shots}")
    if num_channels < 0:
        raise ValueError(f"Invalid number of channels in {filepath}: {num_channels}")
    return n_shots, laser_freq, num_channels


def _channel_name(raw_wavelength: str, is_photon_counting: bool) -> str:
    """Normalize one Licel wavelength token into the project channel naming scheme."""
    clean_wl = re.sub(r"[^0-9]", "", raw_wavelength.split(".")[0]).lstrip("0") or "0"
    return f"{clean_wl}.{'PC' if is_photon_counting else 'AN'}"


def _parse_channel_metadata(line: str, filepath: str) -> dict[str, int | float | bool | str]:
    """Parse one Licel channel header line into typed metadata."""
    parts = _split_header_parts(line, 8, f"Licel channel header in {filepath}")
    active = int(parts[0])
    is_photon_counting = bool(int(parts[1]))
    num_points = int(parts[3])
    if num_points <= 0:
        raise ValueError(f"Invalid point count in {filepath}: {num_points}")

    adc_bits = int(parts[12]) if len(parts) > 12 else DEFAULT_ANALOG_ADC_BITS
    if adc_bits <= 0 and not is_photon_counting:
        adc_bits = DEFAULT_ANALOG_ADC_BITS
    adc_range_v = float(parts[14]) if len(parts) > 14 else DEFAULT_ANALOG_ADC_RANGE_V

    return {
        "name": _channel_name(parts[7], is_photon_counting),
        "active": active,
        "is_pc": is_photon_counting,
        "points": num_points,
        "adc_range": adc_range_v * 1000.0,
        "adc_bits": adc_bits,
    }


def read_licel_header(
    filepath: str,
    logger: Optional[logging.Logger] = None,
) -> tuple[Optional[datetime], Optional[datetime], Optional[float], Optional[int], Optional[int]]:
    """Read global metadata from a Licel binary header."""
    try:
        with open(filepath, "rb") as file:
            _ = file.readline()
            line2 = file.readline().decode("utf-8", errors="ignore").strip()
            line3 = file.readline().decode("utf-8", errors="ignore").strip()

        start_time_utc, stop_time_utc = _extract_licel_datetimes(line2)
        duration = (stop_time_utc - start_time_utc).total_seconds()
        n_shots, laser_freq, _ = _parse_global_header_line(line3, filepath)
        return start_time_utc, stop_time_utc, duration, n_shots, laser_freq
    except Exception as exc:
        if logger:
            logger.warning(f"  -> Invalid Licel header skipped: {filepath} ({exc})")
        return None, None, None, None, None


def parse_single_licel_file(filepath: str) -> dict:
    """Read one SPU-Lidar Licel binary file into physical channel arrays."""
    with open(filepath, "rb") as file:
        _ = file.readline()
        _ = file.readline()
        line3 = file.readline().decode("utf-8", errors="ignore").strip()
        n_shots, laser_freq, num_channels = _parse_global_header_line(line3, filepath)

        channels_meta = []
        for _ in range(num_channels):
            ch_line = file.readline().decode("utf-8", errors="ignore").strip()
            channels_meta.append(_parse_channel_metadata(ch_line, filepath))

        file.readline()
        binary_payload = np.fromfile(file, dtype=np.int32)

    expected_payload_points = sum(ch["points"] for ch in channels_meta if ch["active"] != 0)
    if binary_payload.size < expected_payload_points:
        raise ValueError(
            f"Binary payload too short in {filepath}: expected {expected_payload_points}, "
            f"found {binary_payload.size}"
        )

    data_dict: dict[str, np.ndarray] = {}
    cursor = 0
    for ch in channels_meta:
        if ch["active"] == 0:
            continue
        ch_name = ch["name"]
        if ch_name in data_dict:
            raise ValueError(f"Duplicated active channel name in {filepath}: {ch_name}")

        raw_int_array = binary_payload[cursor : cursor + ch["points"]]
        cursor += ch["points"]
        if ch["is_pc"]:
            physical_array = raw_int_array.astype(np.float64)
        else:
            adc_factor = ch["adc_range"] / (2 ** ch["adc_bits"])
            physical_array = (raw_int_array.astype(np.float64) / n_shots) * adc_factor
        data_dict[ch_name] = physical_array

    payload_samples_used = cursor
    extra_payload_samples = int(binary_payload.size - payload_samples_used)
    return {
        "data": data_dict,
        "shots": n_shots,
        "laser_freq": laser_freq,
        "channels_meta": channels_meta,
        "payload_samples_used": payload_samples_used,
        "extra_payload_samples": extra_payload_samples,
    }


def parse_licel_group(filepaths: list[str], logger: logging.Logger) -> dict:
    """Parse multiple Licel files into time x range tensors."""
    logger.info(f"    -> Parsing {len(filepaths)} raw binary files...")
    time_series: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    global_shots: list[int] = []

    baseline_channels: Optional[tuple[str, ...]] = None
    baseline_points: Optional[dict[str, int]] = None
    baseline_laser_freq: Optional[int] = None

    for filepath in sorted(filepaths):
        try:
            parsed = parse_single_licel_file(filepath)
            data = parsed["data"]
            channels = tuple(sorted(data.keys()))
            points = {ch_name: int(array.shape[0]) for ch_name, array in data.items()}
            laser_freq = int(parsed["laser_freq"])

            if baseline_channels is None:
                baseline_channels = channels
                baseline_points = points
                baseline_laser_freq = laser_freq
            else:
                if channels != baseline_channels:
                    logger.warning(
                        f"    -> Skipping incompatible file {Path(filepath).name}: "
                        f"channel set {channels} differs from baseline {baseline_channels}."
                    )
                    continue
                if points != baseline_points:
                    logger.warning(
                        f"    -> Skipping incompatible file {Path(filepath).name}: "
                        "range-bin count differs from baseline."
                    )
                    continue
                if laser_freq != baseline_laser_freq:
                    logger.warning(
                        f"    -> Skipping incompatible file {Path(filepath).name}: "
                        f"laser frequency {laser_freq} differs from baseline {baseline_laser_freq}."
                    )
                    continue

            global_shots.append(int(parsed["shots"]))
            for ch_name, array in data.items():
                time_series[ch_name].append(array)
        except Exception as exc:
            logger.warning(f"    -> Failed to read {Path(filepath).name}: {exc}")
            continue

    if not time_series:
        return {"tensors": {}, "shots": 0, "channels": []}

    tensor_dict = {ch_name: np.vstack(arrays) for ch_name, arrays in time_series.items()}
    return {
        "tensors": tensor_dict,
        "shots": int(mode(global_shots)) if global_shots else 0,
        "channels": sorted(tensor_dict.keys()),
    }
