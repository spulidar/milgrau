"""Station metadata, temporal profile resolution, and optional SCC mapping."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np


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
    """Return the integer wavelength prefix from canonical names such as 532.PC."""
    prefix = str(channel_name).split(".", 1)[0].strip()
    try:
        return int(prefix)
    except ValueError:
        return None


def _validate_corrections(corrections: Mapping[str, Any]) -> None:
    required = {"deadtime_us", "bin_shift_bins", "background_offset"}
    for channel, raw in corrections.items():
        values = _mapping(raw, f"channel_corrections.{channel}")
        if set(values) != required:
            raise ValueError(f"channel_corrections.{channel} must contain exactly {sorted(required)}.")
        _number(values["deadtime_us"], f"channel_corrections.{channel}.deadtime_us")
        if isinstance(values["bin_shift_bins"], bool) or not isinstance(values["bin_shift_bins"], Integral):
            raise ValueError(f"channel_corrections.{channel}.bin_shift_bins must be an integer.")
        _number(values["background_offset"], f"channel_corrections.{channel}.background_offset")


def _validate_lr_input_map(lr_input: Mapping[str, Any], channels: Mapping[str, Any], label: str) -> None:
    unknown_channels = sorted(set(lr_input) - set(channels))
    if unknown_channels:
        raise ValueError(f"{label} references channels outside the SCC configuration: {unknown_channels}.")
    for channel, value in lr_input.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in {0, 1}:
            raise ValueError(f"{label}.{channel} must be integer 0 or 1.")


def _validate_scc_defaults(catalog: Mapping[str, Any]) -> None:
    raw_defaults = catalog.get("scc_defaults")
    if raw_defaults is None:
        return
    defaults = _mapping(raw_defaults, "scc_defaults")
    unknown = sorted(set(defaults) - {"lr_input"})
    if unknown:
        raise ValueError(f"Unknown scc_defaults key(s): {unknown}")
    if "lr_input" not in defaults:
        return
    lr_default = _mapping(defaults["lr_input"], "scc_defaults.lr_input")
    required = {"value", "elastic_wavelengths"}
    if set(lr_default) != required:
        raise ValueError(f"scc_defaults.lr_input must contain exactly {sorted(required)}.")
    value = lr_default["value"]
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in {0, 1}:
        raise ValueError("scc_defaults.lr_input.value must be integer 0 or 1.")
    elastic = _mapping(lr_default["elastic_wavelengths"], "scc_defaults.lr_input.elastic_wavelengths")
    if not elastic:
        raise ValueError("scc_defaults.lr_input.elastic_wavelengths must not be empty.")
    for raw_wavelength, raw_companions in elastic.items():
        try:
            wavelength = int(raw_wavelength)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid elastic wavelength in SCC LR defaults: {raw_wavelength!r}.") from exc
        if wavelength <= 0:
            raise ValueError("SCC LR default elastic wavelengths must be positive integers.")
        if not isinstance(raw_companions, Sequence) or isinstance(raw_companions, (str, bytes)):
            raise ValueError(f"scc_defaults.lr_input.elastic_wavelengths.{raw_wavelength} must be a list.")
        for companion in raw_companions:
            if isinstance(companion, bool) or not isinstance(companion, Integral) or int(companion) <= 0:
                raise ValueError(
                    f"scc_defaults.lr_input.elastic_wavelengths.{raw_wavelength} companions must be positive integers."
                )


def _validate_scc(profile_id: str, scc: Mapping[str, Any]) -> None:
    for mode in ("day", "night"):
        if mode not in scc:
            raise ValueError(f"profiles.{profile_id}.scc.{mode} is required when scc is configured.")
        config = _mapping(scc[mode], f"profiles.{profile_id}.scc.{mode}")
        _positive_int(config.get("configuration_id"), f"profiles.{profile_id}.scc.{mode}.configuration_id")
        _text(config.get("name"), f"profiles.{profile_id}.scc.{mode}.name")
        channels = _mapping(config.get("channels"), f"profiles.{profile_id}.scc.{mode}.channels")
        if not channels:
            raise ValueError(f"profiles.{profile_id}.scc.{mode}.channels must not be empty.")
        ids: set[int] = set()
        for channel, channel_id in channels.items():
            _text(channel, f"profiles.{profile_id}.scc.{mode} channel")
            resolved = _positive_int(channel_id, f"profiles.{profile_id}.scc.{mode}.channels.{channel}")
            if resolved in ids:
                raise ValueError(f"profiles.{profile_id}.scc.{mode} duplicates SCC channel ID {resolved}.")
            ids.add(resolved)

        if "lr_input" in config:
            lr_input = _mapping(config["lr_input"], f"profiles.{profile_id}.scc.{mode}.lr_input")
            _validate_lr_input_map(lr_input, channels, f"profiles.{profile_id}.scc.{mode}.lr_input")


def validate_station_config(catalog: Mapping[str, Any]) -> None:
    """Validate station.yaml; a profile may omit SCC metadata entirely."""
    station = _mapping(catalog.get("station"), "station")
    for key in ("id", "name", "institution", "timezone"):
        _text(station.get(key), f"station.{key}")
    site = _mapping(station.get("site"), "station.site")
    lat = _number(site.get("latitude"), "station.site.latitude")
    lon = _number(site.get("longitude"), "station.site.longitude")
    _number(site.get("station_altitude_m"), "station.site.station_altitude_m")
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Station latitude/longitude are outside valid bounds.")

    radiosonde = station.get("radiosonde", {})
    if radiosonde:
        radiosonde = _mapping(radiosonde, "station.radiosonde")
        for key in ("station_id", "station_name"):
            if key in radiosonde:
                _text(radiosonde[key], f"station.radiosonde.{key}")
        if "fallback_to_standard_atmosphere" in radiosonde and not isinstance(radiosonde["fallback_to_standard_atmosphere"], bool):
            raise ValueError("station.radiosonde.fallback_to_standard_atmosphere must be boolean.")

    _validate_scc_defaults(catalog)

    corrections = _mapping(catalog.get("channel_corrections"), "channel_corrections")
    if not corrections:
        raise ValueError("channel_corrections must not be empty.")
    _validate_corrections(corrections)

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
        start = _date(profile.get("valid_from"), f"profiles.{profile_id}.valid_from")
        end = None if profile.get("valid_to") is None else _date(profile["valid_to"], f"profiles.{profile_id}.valid_to")
        if end is not None and end < start:
            raise ValueError(f"profiles.{profile_id}.valid_to precedes valid_from.")
        intervals.append((start, end, profile_id))
        if "scc" in profile:
            _validate_scc(profile_id, _mapping(profile["scc"], f"profiles.{profile_id}.scc"))

    intervals.sort(key=lambda item: item[0])
    for left, right in zip(intervals, intervals[1:]):
        if left[1] is None or right[0] <= left[1]:
            raise ValueError(f"Station profile validity overlaps: {left[2]} and {right[2]}.")


def merge_station_defaults(config: Mapping[str, Any], catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Merge station-wide defaults into the algorithm configuration."""
    validate_station_config(catalog)
    merged = deepcopy(dict(config))
    station = catalog["station"]
    project = merged.setdefault("project", {})
    project.setdefault("station_name", station["name"])
    project.setdefault("institution", station["institution"])
    project.setdefault("timezone", station["timezone"])
    site = merged.setdefault("site", {})
    for key, value in station["site"].items():
        site.setdefault(key, value)
    site.setdefault("timezone", station["timezone"])
    radiosonde = merged.setdefault("radiosonde", {})
    for key, value in station.get("radiosonde", {}).items():
        radiosonde.setdefault(key, value)
    merged.setdefault("physics", {})["channels"] = deepcopy(catalog["channel_corrections"])

    profile_maps: dict[str, Any] = {}
    for profile in catalog["profiles"]:
        if "scc" not in profile:
            profile_maps[profile["id"]] = {}
        else:
            profile_maps[profile["id"]] = {mode: deepcopy(profile["scc"][mode]["channels"]) for mode in ("day", "night")}
    merged["hardware"] = {"name_to_id": {"profiles": profile_maps}}
    return merged


def _period_mode(period: str) -> str:
    value = str(period).strip().lower()
    if value in {"nt", "night", "nighttime"}:
        return "night"
    if value in {"am", "pm", "day", "daytime"}:
        return "day"
    raise ValueError(f"Unknown measurement period {period!r}; expected am, pm, or nt.")


def _default_lr_input(catalog: Mapping[str, Any], scc_config: Mapping[str, Any]) -> dict[str, int]:
    """Resolve station-wide LR_Input defaults for one concrete SCC configuration."""
    defaults = catalog.get("scc_defaults", {})
    if not isinstance(defaults, Mapping):
        return {}
    lr_default = defaults.get("lr_input", {})
    if not isinstance(lr_default, Mapping):
        return {}
    elastic = lr_default.get("elastic_wavelengths", {})
    if not isinstance(elastic, Mapping):
        return {}
    value = int(lr_default.get("value", 1))
    channels = [str(name) for name in scc_config.get("channels", {})]
    wavelengths_present = {
        wavelength for name in channels if (wavelength := _channel_wavelength_nm(name)) is not None
    }
    result: dict[str, int] = {}
    for raw_elastic, raw_companions in elastic.items():
        elastic_nm = int(raw_elastic)
        companions = {int(item) for item in raw_companions}
        if companions & wavelengths_present:
            continue
        for channel_name in channels:
            if _channel_wavelength_nm(channel_name) == elastic_nm:
                result[channel_name] = value
    return result


def _resolve_lr_input(catalog: Mapping[str, Any], scc_config: Mapping[str, Any]) -> dict[str, int]:
    """Return explicit per-configuration LR_Input or the station-wide default."""
    explicit = scc_config.get("lr_input")
    if isinstance(explicit, Mapping):
        return {str(name): int(value) for name, value in explicit.items()}
    return _default_lr_input(catalog, scc_config)


def resolve_station_context(config: Mapping[str, Any], measurement_time: datetime, period: str, available_channels: Sequence[str]) -> dict[str, Any]:
    """Resolve station/SCC metadata without discarding any Licel channels."""
    catalog = config.get("_station_catalog")
    if not isinstance(catalog, Mapping):
        raise KeyError("No station catalog is loaded; configure station_config in config.yaml.")
    when = measurement_time.date()
    matches = []
    for profile in catalog["profiles"]:
        start = _date(profile["valid_from"], f"profiles.{profile['id']}.valid_from")
        end = None if profile.get("valid_to") is None else _date(profile["valid_to"], f"profiles.{profile['id']}.valid_to")
        if when >= start and (end is None or when <= end):
            matches.append(profile)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one station profile for {when.isoformat()}, found {[p['id'] for p in matches]}.")

    profile = matches[0]
    station = catalog["station"]
    mode = _period_mode(period)
    available = [str(channel) for channel in available_channels]
    available_set = set(available)
    resolved_site = deepcopy(station["site"])
    resolved_site.update(profile.get("site", {}))
    common = {
        "station_id": station["id"],
        "station_name": station["name"],
        "profile_id": profile["id"],
        "valid_from": profile["valid_from"],
        "valid_to": profile.get("valid_to"),
        "mode": mode,
        "site": resolved_site,
        "laser": deepcopy(profile.get("laser", {})),
        "selected_channels": available,
    }
    if "scc" not in profile:
        return {
            **common,
            "scc_available": False,
            "scc_export_ready": False,
            "scc_configuration_id": None,
            "scc_configuration_name": None,
            "channel_ids": {},
            "lr_input": {},
            "scc_channels": [],
            "missing_scc_channels": [],
            "extra_channels": available,
        }

    scc = profile["scc"][mode]
    channel_ids = {str(name): int(value) for name, value in scc["channels"].items()}
    lr_input = _resolve_lr_input(catalog, scc)
    missing = [name for name in channel_ids if name not in available_set]
    scc_channels = [name for name in channel_ids if name in available_set]
    extra = [name for name in available if name not in channel_ids]
    return {
        **common,
        "scc_available": True,
        "scc_export_ready": not missing,
        "scc_configuration_id": int(scc["configuration_id"]),
        "scc_configuration_name": str(scc["name"]),
        "channel_ids": channel_ids,
        "lr_input": lr_input,
        "scc_channels": scc_channels,
        "missing_scc_channels": missing,
        "extra_channels": extra,
    }


def apply_station_context(config: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    """Return a group-specific config with station metadata applied."""
    resolved = deepcopy(dict(config))
    resolved["hardware"] = {"name_to_id": deepcopy(dict(context.get("channel_ids", {})))}
    site = resolved.setdefault("site", {})
    site.update(deepcopy(dict(context["site"])))
    physics = resolved.setdefault("physics", {})
    for key in ("latitude", "longitude", "station_altitude_m"):
        if key in site:
            physics[key] = site[key]
    resolved["_resolved_station"] = deepcopy(dict(context))
    return resolved


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
        result["channel_metadata"] = {channel: deepcopy(metadata[channel]) for channel in selected if channel in metadata}
    if "laser_shots" in lidar_data:
        shots = np.asarray(lidar_data["laser_shots"])
        if shots.ndim != 2 or shots.shape[1] != len(original):
            raise ValueError(f"Parsed laser_shots is not conformable with parsed channel order: shape={shots.shape}, channels={len(original)}.")
        result["laser_shots"] = shots[:, indices]
    return result
