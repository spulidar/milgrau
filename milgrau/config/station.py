"""Station metadata, temporal profile resolution, calibration, and SCC mapping."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from milgrau.config.overlap import resolve_overlap_context, validate_overlap_catalog


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return int(value)


def _date(value: Any, label: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{label} must use ISO date YYYY-MM-DD.") from exc
    raise ValueError(f"{label} must use ISO date YYYY-MM-DD.")


def _channel_wavelength_nm(channel_name: str) -> int | None:
    prefix = str(channel_name).split(".", 1)[0].strip()
    try:
        return int(prefix)
    except ValueError:
        return None


def _detector_mode(channel_name: str) -> str:
    suffix = str(channel_name).split(".")[-1].upper()
    if suffix == "PC":
        return "photon_counting"
    if suffix == "AN":
        return "analog"
    raise ValueError(f"Cannot infer detector mode from canonical channel name {channel_name!r}.")


def _validate_saturation(channel: str, saturation: Mapping[str, Any]) -> None:
    label = f"calibrations channel {channel}.saturation"
    status = _text(saturation.get("status"), f"{label}.status")
    if status not in {"characterized", "not_characterized"}:
        raise ValueError(f"{label}.status must be 'characterized' or 'not_characterized'.")
    if status == "characterized":
        allowed = {"status", "max_rate_mhz"}
        if set(saturation) != allowed:
            raise ValueError(f"{label} must contain exactly {sorted(allowed)} when characterized.")
        max_rate = _number(saturation.get("max_rate_mhz"), f"{label}.max_rate_mhz")
        if max_rate <= 0.0:
            raise ValueError(f"{label}.max_rate_mhz must be positive.")
    elif set(saturation) != {"status"}:
        raise ValueError(f"{label} must contain only status when not characterized.")


def _validate_calibrations(catalog: Mapping[str, Any]) -> None:
    calibrations = _mapping(catalog.get("calibrations"), "calibrations")
    if not calibrations:
        raise ValueError("calibrations must not be empty.")
    for calibration_id, raw in calibrations.items():
        calibration_id = _text(calibration_id, "calibration id")
        calibration = _mapping(raw, f"calibrations.{calibration_id}")
        unknown = sorted(set(calibration) - {"provenance", "channels"})
        if unknown:
            raise ValueError(f"Unknown calibrations.{calibration_id} key(s): {unknown}")
        provenance = _mapping(calibration.get("provenance"), f"calibrations.{calibration_id}.provenance")
        _text(provenance.get("source"), f"calibrations.{calibration_id}.provenance.source")
        channels = _mapping(calibration.get("channels"), f"calibrations.{calibration_id}.channels")
        if not channels:
            raise ValueError(f"calibrations.{calibration_id}.channels must not be empty.")
        for channel, raw_channel in channels.items():
            _text(channel, f"calibrations.{calibration_id} channel")
            values = _mapping(raw_channel, f"calibrations.{calibration_id}.channels.{channel}")
            expected_mode = _detector_mode(channel)
            mode = _text(values.get("detector_mode"), f"calibrations.{calibration_id}.channels.{channel}.detector_mode")
            if mode != expected_mode:
                raise ValueError(
                    f"calibrations.{calibration_id}.channels.{channel}.detector_mode must be {expected_mode!r}."
                )
            required = {"detector_mode", "deadtime_us", "bin_shift_bins", "background_offset"}
            if mode == "photon_counting":
                required.add("saturation")
            if set(values) != required:
                raise ValueError(
                    f"calibrations.{calibration_id}.channels.{channel} must contain exactly {sorted(required)}."
                )
            deadtime = _number(values["deadtime_us"], f"calibrations.{calibration_id}.channels.{channel}.deadtime_us")
            if deadtime < 0.0:
                raise ValueError(f"calibrations.{calibration_id}.channels.{channel}.deadtime_us must be non-negative.")
            if isinstance(values["bin_shift_bins"], bool) or not isinstance(values["bin_shift_bins"], Integral):
                raise ValueError(f"calibrations.{calibration_id}.channels.{channel}.bin_shift_bins must be an integer.")
            _number(values["background_offset"], f"calibrations.{calibration_id}.channels.{channel}.background_offset")
            if mode == "photon_counting":
                _validate_saturation(
                    channel,
                    _mapping(values["saturation"], f"calibrations.{calibration_id}.channels.{channel}.saturation"),
                )


def validate_station_config(catalog: Mapping[str, Any]) -> None:
    """Validate station metadata, geometry, calibrations, temporal profiles, and SCC policy."""
    allowed_root = {"station", "calibrations", "profiles"}
    unknown_root = sorted(set(catalog) - allowed_root)
    if unknown_root:
        raise ValueError(f"Unknown station catalog key(s): {unknown_root}")

    station = _mapping(catalog.get("station"), "station")
    required_station = {"id", "name", "institution", "timezone", "site", "radiosonde", "lidar_geometry"}
    optional_station = {"lidar_ratio_climatology", "overlap"}
    missing_station = sorted(required_station - set(station))
    unknown_station = sorted(set(station) - required_station - optional_station)
    if missing_station or unknown_station:
        raise ValueError(
            f"station keys invalid; missing={missing_station}, unknown={unknown_station}."
        )
    for key in ("id", "name", "institution"):
        _text(station[key], f"station.{key}")
    timezone_name = _text(station["timezone"], "station.timezone")
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"station.timezone is not a valid IANA timezone: {timezone_name!r}.") from exc

    site = _mapping(station["site"], "station.site")
    required_site = {"latitude", "longitude", "station_altitude_m"}
    if set(site) != required_site:
        raise ValueError(f"station.site must contain exactly {sorted(required_site)}.")
    lat = _number(site["latitude"], "station.site.latitude")
    lon = _number(site["longitude"], "station.site.longitude")
    _number(site["station_altitude_m"], "station.site.station_altitude_m")
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Station latitude/longitude are outside valid bounds.")

    geometry = _mapping(station["lidar_geometry"], "station.lidar_geometry")
    if set(geometry) != {"pointing_angle_deg_from_zenith"}:
        raise ValueError("station.lidar_geometry must contain exactly pointing_angle_deg_from_zenith.")
    pointing = _number(
        geometry["pointing_angle_deg_from_zenith"],
        "station.lidar_geometry.pointing_angle_deg_from_zenith",
    )
    if not 0.0 <= pointing <= 180.0:
        raise ValueError("station.lidar_geometry.pointing_angle_deg_from_zenith must be within 0..180 degrees.")

    radiosonde = _mapping(station["radiosonde"], "station.radiosonde")
    if set(radiosonde) != {"station_id", "station_name"}:
        raise ValueError("station.radiosonde must contain exactly station_id and station_name.")
    _text(radiosonde["station_id"], "station.radiosonde.station_id")
    _text(radiosonde["station_name"], "station.radiosonde.station_name")

    validate_overlap_catalog(catalog)
    _validate_calibrations(catalog)
    calibration_ids = set(catalog["calibrations"])

    profiles = catalog.get("profiles")
    if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)) or not profiles:
        raise ValueError("profiles must be a non-empty list.")
    intervals: list[tuple[date, date | None, str]] = []
    ids: set[str] = set()
    for index, raw in enumerate(profiles):
        profile = _mapping(raw, f"profiles[{index}]")
        profile_id = _text(profile.get("id"), f"profiles[{index}].id")
        if profile_id in ids:
            raise ValueError(f"Duplicate station profile id: {profile_id}")
        ids.add(profile_id)
        calibration_id = _text(profile.get("calibration_id"), f"profiles.{profile_id}.calibration_id")
        if calibration_id not in calibration_ids:
            raise ValueError(f"profiles.{profile_id}.calibration_id references unknown calibration {calibration_id!r}.")
        start = _date(profile.get("valid_from"), f"profiles.{profile_id}.valid_from")
        end = None if profile.get("valid_to") is None else _date(profile["valid_to"], f"profiles.{profile_id}.valid_to")
        if end is not None and end < start:
            raise ValueError(f"profiles.{profile_id}.valid_to precedes valid_from.")
        intervals.append((start, end, profile_id))
        if "scc" in profile:
            raise ValueError(
                f"profiles.{profile_id}.scc is SCC-only metadata; move it to scc.yaml."
            )

    intervals.sort(key=lambda item: item[0])
    for left, right in zip(intervals, intervals[1:]):
        if left[1] is None or right[0] <= left[1]:
            raise ValueError(f"Station profile validity overlaps: {left[2]} and {right[2]}.")


def _local_measurement_time(station: Mapping[str, Any], measurement_time: datetime) -> datetime:
    """Convert an aware measurement timestamp to the station IANA timezone."""
    if measurement_time.tzinfo is None or measurement_time.utcoffset() is None:
        raise ValueError("measurement_time must be timezone-aware.")
    timezone_name = _text(station.get("timezone"), "station.timezone")
    try:
        station_timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"station.timezone is not a valid IANA timezone: {timezone_name!r}.") from exc
    return measurement_time.astimezone(station_timezone)


def _scc_mode(value: str | None) -> str | None:
    """Validate an explicitly resolved solar SCC mode."""
    if value is None:
        return None
    mode = str(value).strip().lower()
    if mode not in {"day", "night"}:
        raise ValueError("SCC mode must be 'day' or 'night'.")
    return mode


def resolve_station_context(
    config: Mapping[str, Any],
    measurement_time: datetime,
    available_channels: Sequence[str],
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    """Resolve instrument science; add SCC mapping only when explicitly loaded."""
    catalog = config.get("_station_catalog")
    if not isinstance(catalog, Mapping):
        raise KeyError("No station catalog is loaded; configure station_config in config.yaml.")
    station = catalog["station"]
    local_measurement_time = _local_measurement_time(station, measurement_time)
    when = local_measurement_time.date()
    matches = []
    for profile in catalog["profiles"]:
        start = _date(profile["valid_from"], f"profiles.{profile['id']}.valid_from")
        end = None if profile.get("valid_to") is None else _date(profile["valid_to"], f"profiles.{profile['id']}.valid_to")
        if when >= start and (end is None or when <= end):
            matches.append(profile)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one station profile for {when.isoformat()}, found {[p['id'] for p in matches]}.")

    profile = matches[0]
    resolved_mode = _scc_mode(mode)
    available = [str(channel) for channel in available_channels]
    resolved_site = deepcopy(station["site"])
    resolved_site.update(profile.get("site", {}))
    calibration_id = str(profile["calibration_id"])
    calibration = catalog["calibrations"][calibration_id]
    channel_calibrations = deepcopy(dict(calibration["channels"]))
    common = {
        "station_id": station["id"],
        "station_name": station["name"],
        "profile_id": profile["id"],
        "calibration_id": calibration_id,
        "calibration_provenance": deepcopy(dict(calibration["provenance"])),
        "channel_calibrations": channel_calibrations,
        "valid_from": profile["valid_from"],
        "valid_to": profile.get("valid_to"),
        "mode": resolved_mode,
        "timezone": str(station["timezone"]),
        "site": resolved_site,
        "laser": deepcopy(profile.get("laser", {})),
        "overlap": resolve_overlap_context(catalog, profile),
        "selected_channels": available,
    }
    scc_catalog = config.get("_scc_catalog")
    if not isinstance(scc_catalog, Mapping):
        # Disabled/unconfigured SCC is completely absent from scientific execution.
        return {
            **common,
            "scc_available": False,
            "scc_export_ready": False,
            "scc_configuration_id": None,
            "scc_configuration_name": None,
            "channel_ids": {},
            "lr_input": {},
            "elastic_product_ids": {},
            "scc_channels": [],
            "missing_scc_channels": [],
            "extra_channels": available,
        }

    # Import the SCC adapter only for explicitly enabled integration.
    from milgrau.config.scc import resolve_scc_context

    return {
        **common,
        **resolve_scc_context(scc_catalog, str(profile["id"]), available, resolved_mode),
    }


def select_lidar_channels(lidar_data: Mapping[str, Any], selected_channels: Sequence[str]) -> dict[str, Any]:
    """Subset parsed Licel data for a derived product such as an SCC export."""
    original = [str(channel) for channel in lidar_data.get("channels", [])]
    selected = [str(channel) for channel in selected_channels]
    missing = [channel for channel in selected if channel not in original]
    if missing:
        raise ValueError(f"Cannot select missing parsed lidar channels: {missing}")
    indices = [original.index(channel) for channel in selected]
    result = dict(lidar_data)
    result["channels"] = selected
    tensors = lidar_data.get("tensors", {})
    result["tensors"] = {channel: tensors[channel] for channel in selected}
    metadata = lidar_data.get("channel_metadata", {})
    if isinstance(metadata, Mapping):
        result["channel_metadata"] = {
            channel: deepcopy(metadata[channel]) for channel in selected if channel in metadata
        }
    if "laser_shots" in lidar_data:
        shots = np.asarray(lidar_data["laser_shots"])
        if shots.ndim != 2 or shots.shape[1] != len(original):
            raise ValueError(
                f"Parsed laser_shots is not conformable with parsed channel order: shape={shots.shape}, channels={len(original)}."
            )
        result["laser_shots"] = shots[:, indices]
    return result
