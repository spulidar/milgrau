"""Station metadata, profile resolution, and SCC channel selection."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _as_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _as_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return int(value)


def _as_finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number.")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{label} must be finite.")
    return converted


def _as_date(value: Any, label: str) -> date:
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


def _validate_channel_corrections(corrections: Mapping[str, Any]) -> None:
    required = {"deadtime_us", "bin_shift_bins", "background_offset"}
    for channel_name, raw_values in corrections.items():
        _as_nonempty_string(channel_name, "channel_corrections channel name")
        values = _as_mapping(raw_values, f"channel_corrections.{channel_name}")
        missing = sorted(required - set(values))
        unknown = sorted(set(values) - required)
        if missing or unknown:
            raise ValueError(
                f"channel_corrections.{channel_name} must contain exactly {sorted(required)}; "
                f"missing={missing}, unknown={unknown}."
            )
        _as_finite_float(values["deadtime_us"], f"channel_corrections.{channel_name}.deadtime_us")
        shift = values["bin_shift_bins"]
        if isinstance(shift, bool) or not isinstance(shift, Integral):
            raise ValueError(f"channel_corrections.{channel_name}.bin_shift_bins must be an integer.")
        _as_finite_float(values["background_offset"], f"channel_corrections.{channel_name}.background_offset")


def _validate_scc_configuration(profile_id: str, mode: str, raw_config: Any) -> None:
    config = _as_mapping(raw_config, f"profiles.{profile_id}.scc.{mode}")
    _as_positive_int(config.get("configuration_id"), f"profiles.{profile_id}.scc.{mode}.configuration_id")
    _as_nonempty_string(config.get("name"), f"profiles.{profile_id}.scc.{mode}.name")
    channels = _as_mapping(config.get("channels"), f"profiles.{profile_id}.scc.{mode}.channels")
    if not channels:
        raise ValueError(f"profiles.{profile_id}.scc.{mode}.channels must not be empty.")
    seen_ids: set[int] = set()
    for channel_name, channel_id in channels.items():
        _as_nonempty_string(channel_name, f"profiles.{profile_id}.scc.{mode} channel name")
        resolved_id = _as_positive_int(channel_id, f"profiles.{profile_id}.scc.{mode}.channels.{channel_name}")
        if resolved_id in seen_ids:
            raise ValueError(
                f"profiles.{profile_id}.scc.{mode}.channels contains duplicate SCC channel ID {resolved_id}."
            )
        seen_ids.add(resolved_id)


def validate_station_config(catalog: Mapping[str, Any]) -> None:
    """Validate the compact station.yaml structure."""
    station = _as_mapping(catalog.get("station"), "station")
    for key in ("id", "name", "institution", "timezone"):
        _as_nonempty_string(station.get(key), f"station.{key}")

    site = _as_mapping(station.get("site"), "station.site")
    latitude = _as_finite_float(site.get("latitude"), "station.site.latitude")
    longitude = _as_finite_float(site.get("longitude"), "station.site.longitude")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("station.site.latitude must be within [-90, 90].")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError("station.site.longitude must be within [-180, 180].")
    altitude = _as_finite_float(site.get("station_altitude_m"), "station.site.station_altitude_m")
    if altitude < -500.0:
        raise ValueError("station.site.station_altitude_m is implausibly low.")

    radiosonde = station.get("radiosonde", {})
    if radiosonde:
        radiosonde = _as_mapping(radiosonde, "station.radiosonde")
        if "station_id" in radiosonde:
            _as_nonempty_string(radiosonde["station_id"], "station.radiosonde.station_id")
        if "station_name" in radiosonde:
            _as_nonempty_string(radiosonde["station_name"], "station.radiosonde.station_name")
        if "fallback_to_standard_atmosphere" in radiosonde and not isinstance(
            radiosonde["fallback_to_standard_atmosphere"], bool
        ):
            raise ValueError("station.radiosonde.fallback_to_standard_atmosphere must be boolean.")

    corrections = _as_mapping(catalog.get("channel_corrections"), "channel_corrections")
    if not corrections:
        raise ValueError("channel_corrections must not be empty.")
    _validate_channel_corrections(corrections)

    profiles = catalog.get("profiles")
    if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)) or not profiles:
        raise ValueError("profiles must be a non-empty list.")

    profile_ids: set[str] = set()
    intervals: list[tuple[date, date | None, str]] = []
    for index, raw_profile in enumerate(profiles):
        profile = _as_mapping(raw_profile, f"profiles[{index}]")
        profile_id = _as_nonempty_string(profile.get("id"), f"profiles[{index}].id")
        if profile_id in profile_ids:
            raise ValueError(f"Duplicate station profile id: {profile_id}")
        profile_ids.add(profile_id)
        valid_from = _as_date(profile.get("valid_from"), f"profiles.{profile_id}.valid_from")
        valid_to_raw = profile.get("valid_to")
        valid_to = None if valid_to_raw is None else _as_date(valid_to_raw, f"profiles.{profile_id}.valid_to")
        if valid_to is not None and valid_to < valid_from:
            raise ValueError(f"profiles.{profile_id}.valid_to precedes valid_from.")
        intervals.append((valid_from, valid_to, profile_id))

        profile_site = profile.get("site", {})
        if profile_site:
            profile_site = _as_mapping(profile_site, f"profiles.{profile_id}.site")
            if "station_altitude_m" in profile_site:
                _as_finite_float(
                    profile_site["station_altitude_m"],
                    f"profiles.{profile_id}.site.station_altitude_m",
                )

        laser = profile.get("laser", {})
        if laser:
            laser = _as_mapping(laser, f"profiles.{profile_id}.laser")
            for key in ("manufacturer", "model"):
                if key in laser:
                    _as_nonempty_string(laser[key], f"profiles.{profile_id}.laser.{key}")
            if "repetition_rate_hz" in laser:
                repetition_rate = _as_finite_float(
                    laser["repetition_rate_hz"],
                    f"profiles.{profile_id}.laser.repetition_rate_hz",
                )
                if repetition_rate <= 0.0:
                    raise ValueError(f"profiles.{profile_id}.laser.repetition_rate_hz must be positive.")

        scc = _as_mapping(profile.get("scc"), f"profiles.{profile_id}.scc")
        for mode in ("day", "night"):
            if mode not in scc:
                raise ValueError(f"profiles.{profile_id}.scc.{mode} is required.")
            _validate_scc_configuration(profile_id, mode, scc[mode])

    intervals.sort(key=lambda item: item[0])
    for left, right in zip(intervals, intervals[1:]):
        left_end = left[1]
        if left_end is None or right[0] <= left_end:
            raise ValueError(
                f"Station profile validity overlaps: {left[2]} and {right[2]}."
            )


def merge_station_defaults(config: Mapping[str, Any], catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Merge station-wide defaults into the algorithm config before validation."""
    validate_station_config(catalog)
    merged = deepcopy(dict(config))
    station = catalog["station"]

    project = merged.setdefault("project", {})
    project.setdefault("station_name", station["name"])
    project.setdefault("institution", station["institution"])
    project.setdefault("timezone", station["timezone"])

    site = merged.setdefault("site", {})
    station_site = station["site"]
    for key in ("latitude", "longitude", "station_altitude_m"):
        site.setdefault(key, station_site[key])
    site.setdefault("timezone", station["timezone"])

    radiosonde = merged.setdefault("radiosonde", {})
    for key, value in station.get("radiosonde", {}).items():
        radiosonde.setdefault(key, value)

    physics = merged.setdefault("physics", {})
    physics["channels"] = deepcopy(catalog["channel_corrections"])

    profile_maps: dict[str, Any] = {}
    for profile in catalog["profiles"]:
        profile_maps[profile["id"]] = {
            mode: deepcopy(profile["scc"][mode]["channels"])
            for mode in ("day", "night")
        }
    merged["hardware"] = {"name_to_id": {"profiles": profile_maps}}
    return merged


def _period_mode(period: str) -> str:
    value = str(period).strip().lower()
    if value in {"nt", "night", "nighttime"}:
        return "night"
    if value in {"am", "pm", "day", "daytime"}:
        return "day"
    raise ValueError(f"Unknown measurement period {period!r}; expected am, pm, or nt.")


def resolve_station_context(
    config: Mapping[str, Any],
    measurement_time: datetime,
    period: str,
    available_channels: Sequence[str],
) -> dict[str, Any]:
    """Resolve one temporal station profile and SCC configuration."""
    catalog = config.get("_station_catalog")
    if not isinstance(catalog, Mapping):
        raise KeyError("No station catalog is loaded; configure station_config in config.yaml.")

    when = measurement_time.date()
    matching_profiles = []
    for profile in catalog["profiles"]:
        start = _as_date(profile["valid_from"], f"profiles.{profile['id']}.valid_from")
        end = None if profile.get("valid_to") is None else _as_date(
            profile["valid_to"], f"profiles.{profile['id']}.valid_to"
        )
        if when >= start and (end is None or when <= end):
            matching_profiles.append(profile)
    if len(matching_profiles) != 1:
        raise ValueError(
            f"Expected exactly one station profile for {when.isoformat()}, found "
            f"{[profile['id'] for profile in matching_profiles]}."
        )

    profile = matching_profiles[0]
    mode = _period_mode(period)
    scc = profile["scc"][mode]
    channel_ids = {str(name): int(value) for name, value in scc["channels"].items()}
    available = [str(channel) for channel in available_channels]
    available_set = set(available)
    expected_set = set(channel_ids)
    missing = sorted(expected_set - available_set)
    if missing:
        raise ValueError(
            f"SCC configuration {scc['configuration_id']} ({mode}) requires channels missing from "
            f"the Licel group: {missing}. Available channels: {sorted(available_set)}"
        )

    selected_channels = [name for name in channel_ids if name in available_set]
    extra_channels = [name for name in available if name not in expected_set]
    station = catalog["station"]
    resolved_site = deepcopy(station["site"])
    resolved_site.update(profile.get("site", {}))
    return {
        "station_id": station["id"],
        "station_name": station["name"],
        "profile_id": profile["id"],
        "valid_from": profile["valid_from"],
        "valid_to": profile.get("valid_to"),
        "mode": mode,
        "scc_configuration_id": int(scc["configuration_id"]),
        "scc_configuration_name": str(scc["name"]),
        "channel_ids": channel_ids,
        "selected_channels": selected_channels,
        "extra_channels": extra_channels,
        "site": resolved_site,
        "laser": deepcopy(profile.get("laser", {})),
    }


def apply_station_context(config: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    """Return a group-specific config with resolved station metadata applied."""
    resolved = deepcopy(dict(config))
    channel_ids = deepcopy(dict(context["channel_ids"]))
    resolved["hardware"] = {"name_to_id": channel_ids}

    site = resolved.setdefault("site", {})
    site.update(deepcopy(dict(context["site"])))
    physics = resolved.setdefault("physics", {})
    for key in ("latitude", "longitude", "station_altitude_m"):
        if key in site:
            physics[key] = site[key]
    resolved["_resolved_station"] = deepcopy(dict(context))
    return resolved


def select_lidar_channels(lidar_data: Mapping[str, Any], selected_channels: Sequence[str]) -> dict[str, Any]:
    """Subset parsed Licel tensors and per-channel metadata to an SCC configuration."""
    original_channels = [str(channel) for channel in lidar_data.get("channels", [])]
    selected = [str(channel) for channel in selected_channels]
    missing = [channel for channel in selected if channel not in original_channels]
    if missing:
        raise ValueError(f"Cannot select missing parsed lidar channels: {missing}")

    indices = [original_channels.index(channel) for channel in selected]
    result = dict(lidar_data)
    result["channels"] = selected
    tensors = lidar_data.get("tensors", {})
    result["tensors"] = {channel: tensors[channel] for channel in selected}

    metadata = lidar_data.get("channel_metadata", {})
    if isinstance(metadata, Mapping):
        result["channel_metadata"] = {
            channel: deepcopy(metadata[channel])
            for channel in selected
            if channel in metadata
        }

    if "laser_shots" in lidar_data:
        shots = np.asarray(lidar_data["laser_shots"])
        if shots.ndim != 2 or shots.shape[1] != len(original_channels):
            raise ValueError(
                "Parsed laser_shots is not conformable with parsed channel order: "
                f"shape={shots.shape}, channels={len(original_channels)}."
            )
        result["laser_shots"] = shots[:, indices]
    return result
