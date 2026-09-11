"""Station metadata, temporal profile resolution, calibration, and SCC mapping."""

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


def _validate_scc_policy(catalog: Mapping[str, Any]) -> None:
    policy = _mapping(catalog.get("scc_policy"), "scc_policy")
    if set(policy) != {"lr_input"}:
        raise ValueError("scc_policy must contain exactly ['lr_input'].")
    lr_policy = _mapping(policy["lr_input"], "scc_policy.lr_input")
    required = {"fixed_value", "raman_companions_nm"}
    if set(lr_policy) != required:
        raise ValueError(f"scc_policy.lr_input must contain exactly {sorted(required)}.")
    fixed = lr_policy["fixed_value"]
    if isinstance(fixed, bool) or not isinstance(fixed, Integral) or int(fixed) not in {0, 1}:
        raise ValueError("scc_policy.lr_input.fixed_value must be integer 0 or 1.")
    companions = _mapping(lr_policy["raman_companions_nm"], "scc_policy.lr_input.raman_companions_nm")
    if not companions:
        raise ValueError("scc_policy.lr_input.raman_companions_nm must not be empty.")
    for raw_wavelength, raw_companions in companions.items():
        try:
            wavelength = int(raw_wavelength)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid elastic wavelength in SCC policy: {raw_wavelength!r}.") from exc
        if wavelength <= 0:
            raise ValueError("SCC elastic wavelengths must be positive integers.")
        if not isinstance(raw_companions, Sequence) or isinstance(raw_companions, (str, bytes)):
            raise ValueError(f"scc_policy.lr_input.raman_companions_nm.{raw_wavelength} must be a list.")
        for companion in raw_companions:
            if isinstance(companion, bool) or not isinstance(companion, Integral) or int(companion) <= 0:
                raise ValueError(
                    f"scc_policy.lr_input.raman_companions_nm.{raw_wavelength} companions must be positive integers."
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


def validate_station_config(catalog: Mapping[str, Any]) -> None:
    """Validate station metadata, geometry, calibrations, temporal profiles, and SCC policy."""
    allowed_root = {"station", "scc_policy", "calibrations", "profiles"}
    unknown_root = sorted(set(catalog) - allowed_root)
    if unknown_root:
        raise ValueError(f"Unknown station catalog key(s): {unknown_root}")

    station = _mapping(catalog.get("station"), "station")
    required_station = {"id", "name", "institution", "timezone", "site", "radiosonde", "lidar_geometry"}
    optional_station = {"lidar_ratio_climatology"}
    missing_station = sorted(required_station - set(station))
    unknown_station = sorted(set(station) - required_station - optional_station)
    if missing_station or unknown_station:
        raise ValueError(
            f"station keys invalid; missing={missing_station}, unknown={unknown_station}."
        )
    for key in ("id", "name", "institution", "timezone"):
        _text(station[key], f"station.{key}")

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

    _validate_scc_policy(catalog)
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
            _validate_scc(profile_id, _mapping(profile["scc"], f"profiles.{profile_id}.scc"))

    intervals.sort(key=lambda item: item[0])
    for left, right in zip(intervals, intervals[1:]):
        if left[1] is None or right[0] <= left[1]:
            raise ValueError(f"Station profile validity overlaps: {left[2]} and {right[2]}.")


def _period_mode(period: str) -> str:
    value = str(period).strip().lower()
    if value in {"nt", "night", "nighttime"}:
        return "night"
    if value in {"am", "pm", "day", "daytime"}:
        return "day"
    raise ValueError(f"Unknown measurement period {period!r}; expected am, pm, or nt.")


def _default_lr_input(catalog: Mapping[str, Any], scc_config: Mapping[str, Any]) -> dict[str, int]:
    """Resolve station-wide LR_Input policy for one concrete SCC configuration."""
    lr_policy = catalog["scc_policy"]["lr_input"]
    companions_by_elastic = lr_policy["raman_companions_nm"]
    value = int(lr_policy["fixed_value"])
    channels = [str(name) for name in scc_config["channels"]]
    wavelengths_present = {
        wavelength for name in channels if (wavelength := _channel_wavelength_nm(name)) is not None
    }
    result: dict[str, int] = {}
    for raw_elastic, raw_companions in companions_by_elastic.items():
        elastic_nm = int(raw_elastic)
        companions = {int(item) for item in raw_companions}
        if companions & wavelengths_present:
            continue
        for channel_name in channels:
            if _channel_wavelength_nm(channel_name) == elastic_nm:
                result[channel_name] = value
    return result


def _resolve_lr_input(catalog: Mapping[str, Any], scc_config: Mapping[str, Any]) -> dict[str, int]:
    explicit = scc_config.get("lr_input")
    if isinstance(explicit, Mapping):
        return {str(name): int(value) for name, value in explicit.items()}
    return _default_lr_input(catalog, scc_config)


def resolve_station_context(
    config: Mapping[str, Any],
    measurement_time: datetime,
    period: str,
    available_channels: Sequence[str],
) -> dict[str, Any]:
    """Resolve one temporal station profile, calibration set, and optional SCC map."""
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
