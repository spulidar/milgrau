"""Licel binary header and payload parsing utilities for SPU-Lidar."""

from __future__ import annotations

import logging
import math
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
CHANNEL_PAYLOAD_SEPARATOR: Final[bytes] = b"\r\n"


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


def _finite_float(value: str, default: float = math.nan) -> float:
    """Parse one finite float token, returning a default for malformed values."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _parse_global_header_line(line3: str, filepath: str) -> tuple[int, int, int]:
    """Parse the third Licel header line.

    Standard Licel files use ``LS1 Rate1 LS2 Rate2 DataSets`` and store laser
    shots per channel in the channel header. Older MILGRAU fixtures/custom
    exports used the third token as a file-level shot fallback. The returned
    first value is therefore only a compatibility fallback; channel ``NShots``
    always takes precedence.
    """
    parts = _split_header_parts(line3, 5, f"Licel global header in {filepath}")
    try:
        num_channels = int(float(parts[4]))
    except ValueError as exc:
        raise ValueError(f"Invalid Licel dataset count in {filepath}: {parts[4]!r}") from exc
    if num_channels < 0:
        raise ValueError(f"Invalid number of channels in {filepath}: {num_channels}")

    standard_numeric = all(math.isfinite(_finite_float(token)) for token in parts[:4])
    if standard_numeric:
        fallback_shots = 0
        rates = [int(round(_finite_float(parts[1], 0.0))), int(round(_finite_float(parts[3], 0.0)))]
        laser_freq = next((rate for rate in rates if rate > 0), 0)
    else:
        try:
            fallback_shots = int(float(parts[2]))
            laser_freq = int(float(parts[3]))
        except ValueError as exc:
            raise ValueError(f"Invalid legacy Licel global header in {filepath}: {line3!r}") from exc
        if fallback_shots < 0:
            raise ValueError(f"Invalid fallback laser-shot count in {filepath}: {fallback_shots}")

    return fallback_shots, laser_freq, num_channels


def _channel_name(raw_wavelength: str, is_photon_counting: bool) -> str:
    """Normalize one Licel wavelength token into the project channel naming scheme."""
    clean_wl = re.sub(r"[^0-9]", "", raw_wavelength.split(".")[0]).lstrip("0") or "0"
    return f"{clean_wl}.{'PC' if is_photon_counting else 'AN'}"


def _parse_channel_metadata(line: str, filepath: str) -> dict[str, int | float | bool | str]:
    """Parse one Licel channel header into typed acquisition metadata.

    The canonical fields are ``Active AnalogPhoton LaserUsed DataPoints 1 HV
    BinW Wavelength d1 d2 d3 d4 ADCbits NShots Discriminator ID``.
    ``Discriminator`` is exposed in mV for the SCC ``DAQ_Range`` variable.
    """
    parts = _split_header_parts(line, 8, f"Licel channel header in {filepath}")
    active = int(parts[0])
    is_photon_counting = bool(int(parts[1]))
    laser_used = int(parts[2]) if len(parts) > 2 else 0
    num_points = int(parts[3])
    if num_points <= 0:
        raise ValueError(f"Invalid point count in {filepath}: {num_points}")

    bin_width_m = _finite_float(parts[6], math.nan) if len(parts) > 6 else math.nan
    if bin_width_m <= 0.0:
        bin_width_m = math.nan

    adc_bits = int(float(parts[12])) if len(parts) > 12 else DEFAULT_ANALOG_ADC_BITS
    if adc_bits <= 0 and not is_photon_counting:
        adc_bits = DEFAULT_ANALOG_ADC_BITS
    number_of_shots = int(float(parts[13])) if len(parts) > 13 else 0
    discriminator = _finite_float(parts[14], DEFAULT_ANALOG_ADC_RANGE_V) if len(parts) > 14 else DEFAULT_ANALOG_ADC_RANGE_V
    daq_range_mv = discriminator * 1000.0 if not is_photon_counting else math.nan
    channel_id = parts[15] if len(parts) > 15 else ""

    return {
        "name": _channel_name(parts[7], is_photon_counting),
        "active": active,
        "is_pc": is_photon_counting,
        "acquisition_mode": 1 if is_photon_counting else 0,
        "laser_used": laser_used,
        "points": num_points,
        "bin_width_m": bin_width_m,
        "shots": number_of_shots,
        "daq_range_mV": daq_range_mv,
        "adc_range": daq_range_mv,
        "adc_bits": adc_bits,
        "channel_id": channel_id,
    }


def _split_channel_payloads(
    payload: bytes,
    channels_meta: list[dict[str, int | float | bool | str]],
    filepath: str,
) -> tuple[list[np.ndarray], int]:
    """Decode active channel blocks, consuming optional Licel CRLF separators."""
    active_channels = [channel for channel in channels_meta if int(channel["active"]) != 0]
    expected_data_bytes = sum(int(channel["points"]) * BYTES_PER_PAYLOAD_SAMPLE for channel in active_channels)
    if len(payload) < expected_data_bytes:
        raise ValueError(
            f"Binary payload too short in {filepath}: expected {expected_data_bytes // BYTES_PER_PAYLOAD_SAMPLE} "
            f"samples, found {len(payload) // BYTES_PER_PAYLOAD_SAMPLE}"
        )

    separated_blocks: list[np.ndarray] = []
    separated_cursor = 0
    for channel in active_channels:
        block_bytes = int(channel["points"]) * BYTES_PER_PAYLOAD_SAMPLE
        block_stop = separated_cursor + block_bytes
        separator_stop = block_stop + len(CHANNEL_PAYLOAD_SEPARATOR)
        if separator_stop > len(payload) or payload[block_stop:separator_stop] != CHANNEL_PAYLOAD_SEPARATOR:
            separated_blocks = []
            break
        separated_blocks.append(
            np.frombuffer(payload[separated_cursor:block_stop], dtype="<i4").astype(np.int32, copy=True)
        )
        separated_cursor = separator_stop

    if separated_blocks:
        extra_payload_samples = max(len(payload) - separated_cursor, 0) // BYTES_PER_PAYLOAD_SAMPLE
        return separated_blocks, int(extra_payload_samples)

    contiguous_payload = payload[:expected_data_bytes]
    contiguous_samples = np.frombuffer(contiguous_payload, dtype="<i4")
    blocks: list[np.ndarray] = []
    sample_cursor = 0
    for channel in active_channels:
        points = int(channel["points"])
        blocks.append(contiguous_samples[sample_cursor : sample_cursor + points].astype(np.int32, copy=True))
        sample_cursor += points
    extra_payload_samples = max(len(payload) - expected_data_bytes, 0) // BYTES_PER_PAYLOAD_SAMPLE
    return blocks, int(extra_payload_samples)


def _resolved_channel_shots(channel: dict[str, int | float | bool | str], fallback_shots: int, filepath: str) -> int:
    """Return channel ``NShots`` with a compatibility fallback for old files."""
    channel_shots = int(channel.get("shots", 0))
    shots = channel_shots if channel_shots > 0 else int(fallback_shots)
    if int(channel["active"]) != 0 and shots <= 0:
        raise ValueError(f"Active channel {channel['name']} in {filepath} has no valid NShots value.")
    return shots


def read_licel_header(
    filepath: str,
    logger: Optional[logging.Logger] = None,
) -> tuple[Optional[datetime], Optional[datetime], Optional[float], Optional[int], Optional[int]]:
    """Read inventory metadata from a Licel binary header."""
    try:
        with open(filepath, "rb") as file:
            _ = file.readline()
            line2 = file.readline().decode("utf-8", errors="ignore").strip()
            line3 = file.readline().decode("utf-8", errors="ignore").strip()
            fallback_shots, laser_freq, num_channels = _parse_global_header_line(line3, filepath)
            channels_meta = [
                _parse_channel_metadata(file.readline().decode("utf-8", errors="ignore").strip(), filepath)
                for _ in range(num_channels)
            ]

        start_time_utc, stop_time_utc = _extract_licel_datetimes(line2)
        duration = (stop_time_utc - start_time_utc).total_seconds()
        active_shots = [
            _resolved_channel_shots(channel, fallback_shots, filepath)
            for channel in channels_meta
            if int(channel["active"]) != 0
        ]
        representative_shots = int(mode(active_shots)) if active_shots else int(fallback_shots)
        if representative_shots <= 0:
            raise ValueError(f"No positive laser-shot count found in {filepath}.")
        return start_time_utc, stop_time_utc, duration, representative_shots, laser_freq
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
        fallback_shots, laser_freq, num_channels = _parse_global_header_line(line3, filepath)
        channels_meta = []
        for _ in range(num_channels):
            ch_line = file.readline().decode("utf-8", errors="ignore").strip()
            channels_meta.append(_parse_channel_metadata(ch_line, filepath))
        file.readline()
        binary_payload = file.read()

    channel_payloads, extra_payload_samples = _split_channel_payloads(binary_payload, channels_meta, filepath)
    data_dict: dict[str, np.ndarray] = {}
    shots_by_channel: dict[str, int] = {}
    active_channels = [channel for channel in channels_meta if int(channel["active"]) != 0]
    for ch, raw_int_array in zip(active_channels, channel_payloads, strict=True):
        ch_name = str(ch["name"])
        if ch_name in data_dict:
            raise ValueError(f"Duplicated active channel name in {filepath}: {ch_name}")
        channel_shots = _resolved_channel_shots(ch, fallback_shots, filepath)
        ch["shots"] = channel_shots
        shots_by_channel[ch_name] = channel_shots
        if bool(ch["is_pc"]):
            physical_array = raw_int_array.astype(np.float64)
        else:
            adc_bits = int(ch["adc_bits"])
            adc_span_counts = (2**adc_bits) - 1
            if adc_span_counts <= 0:
                raise ValueError(f"Invalid ADC bit depth for analog channel {ch_name}: {adc_bits}")
            daq_range_mv = float(ch["daq_range_mV"])
            if not np.isfinite(daq_range_mv) or daq_range_mv <= 0.0:
                raise ValueError(f"Invalid analog DAQ range for channel {ch_name}: {daq_range_mv}")
            physical_array = (raw_int_array.astype(np.float64) / channel_shots) * (daq_range_mv / adc_span_counts)
        data_dict[ch_name] = physical_array

    payload_samples_used = sum(int(channel["points"]) for channel in active_channels)
    representative_shots = int(mode(shots_by_channel.values())) if shots_by_channel else int(fallback_shots)
    return {
        "data": data_dict,
        "shots": representative_shots,
        "shots_by_channel": shots_by_channel,
        "laser_freq": laser_freq,
        "channels_meta": channels_meta,
        "payload_samples_used": payload_samples_used,
        "extra_payload_samples": extra_payload_samples,
    }


def _active_metadata_by_name(parsed: dict) -> dict[str, dict[str, int | float | bool | str]]:
    """Return active channel metadata indexed by normalized channel name."""
    return {
        str(channel["name"]): channel
        for channel in parsed.get("channels_meta", [])
        if int(channel["active"]) != 0
    }


def _metadata_signature(metadata: dict[str, int | float | bool | str]) -> tuple[object, ...]:
    """Return acquisition fields that must remain stable inside one Level 0 group."""
    is_pc = bool(metadata["is_pc"])
    daq_range = math.nan if is_pc else float(metadata["daq_range_mV"])
    return (
        is_pc,
        int(metadata["points"]),
        int(metadata["laser_used"]),
        float(metadata["bin_width_m"]),
        int(metadata["adc_bits"]),
        daq_range,
    )


def _metadata_compatible(
    baseline: dict[str, dict[str, int | float | bool | str]],
    current: dict[str, dict[str, int | float | bool | str]],
) -> bool:
    """Return whether SCC-relevant channel metadata are stable across files."""
    if set(baseline) != set(current):
        return False
    for channel_name in baseline:
        left = _metadata_signature(baseline[channel_name])
        right = _metadata_signature(current[channel_name])
        for left_value, right_value in zip(left, right, strict=True):
            if isinstance(left_value, float) and isinstance(right_value, float):
                if math.isnan(left_value) and math.isnan(right_value):
                    continue
                if not math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=1e-9):
                    return False
            elif left_value != right_value:
                return False
    return True


def parse_licel_group(filepaths: list[str], logger: logging.Logger) -> dict:
    """Parse multiple Licel files into time x range tensors and SCC metadata."""
    logger.info(f"    -> Parsing {len(filepaths)} raw binary files...")
    time_series: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    global_shots: list[int] = []
    laser_shot_rows: list[list[int]] = []
    baseline_channels: Optional[tuple[str, ...]] = None
    baseline_points: Optional[dict[str, int]] = None
    baseline_laser_freq: Optional[int] = None
    baseline_metadata: Optional[dict[str, dict[str, int | float | bool | str]]] = None

    for filepath in sorted(filepaths):
        try:
            parsed = parse_single_licel_file(filepath)
            data = parsed["data"]
            channels = tuple(sorted(data.keys()))
            points = {ch_name: int(array.shape[0]) for ch_name, array in data.items()}
            laser_freq = int(parsed["laser_freq"])
            metadata_by_name = _active_metadata_by_name(parsed)
            if baseline_channels is None:
                baseline_channels = channels
                baseline_points = points
                baseline_laser_freq = laser_freq
                baseline_metadata = metadata_by_name
            else:
                if channels != baseline_channels:
                    logger.warning(
                        f"    -> Skipping incompatible file {Path(filepath).name}: channel set {channels} differs from baseline {baseline_channels}."
                    )
                    continue
                if points != baseline_points:
                    logger.warning(f"    -> Skipping incompatible file {Path(filepath).name}: range-bin count differs from baseline.")
                    continue
                if laser_freq != baseline_laser_freq:
                    logger.warning(
                        f"    -> Skipping incompatible file {Path(filepath).name}: laser frequency {laser_freq} differs from baseline {baseline_laser_freq}."
                    )
                    continue
                if baseline_metadata is None or not _metadata_compatible(baseline_metadata, metadata_by_name):
                    logger.warning(
                        f"    -> Skipping incompatible file {Path(filepath).name}: SCC acquisition metadata differ from baseline."
                    )
                    continue

            shot_row = [int(parsed["shots_by_channel"][ch_name]) for ch_name in channels]
            if any(value <= 0 for value in shot_row):
                raise ValueError(f"One or more channels in {filepath} have invalid laser shots: {shot_row}")
            laser_shot_rows.append(shot_row)
            global_shots.extend(shot_row)
            for ch_name, array in data.items():
                time_series[ch_name].append(array)
        except Exception as exc:
            logger.warning(f"    -> Failed to read {Path(filepath).name}: {exc}")
            continue

    if not time_series or baseline_channels is None or baseline_metadata is None:
        return {
            "tensors": {},
            "shots": 0,
            "channels": [],
            "laser_shots": np.empty((0, 0), dtype=np.int32),
            "channel_metadata": {},
        }

    tensor_dict = {ch_name: np.vstack(time_series[ch_name]) for ch_name in baseline_channels}
    channel_metadata = {channel_name: dict(baseline_metadata[channel_name]) for channel_name in baseline_channels}
    return {
        "tensors": tensor_dict,
        "shots": int(mode(global_shots)) if global_shots else 0,
        "channels": list(baseline_channels),
        "laser_shots": np.asarray(laser_shot_rows, dtype=np.int32),
        "channel_metadata": channel_metadata,
    }
