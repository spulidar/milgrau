"""SCC-specific configuration schema and mapping.

Only used when the SCC feature flag is enabled. Never required for
scientific station profile/calibration resolution.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from typing import Any, Mapping, Sequence

from milgrau.config.station import (
    _channel_wavelength_nm,
    _mapping,
    _positive_int,
    _text,
)

def _validate_scc_policy(
    catalog: Mapping[str, Any], station_catalog: Mapping[str, Any],
) -> None:
    policy = _mapping(catalog.get("scc_policy"), "scc_policy")
    if set(policy) != {"lr_input"}:
        raise ValueError("scc_policy must contain exactly ['lr_input'].")
    lr_policy = _mapping(policy["lr_input"], "scc_policy.lr_input")
    required = {"fixed_value", "raman_companions_nm"}
    allowed = required | {"source"}
    if set(lr_policy) - allowed or required - set(lr_policy):
        raise ValueError(
            f"scc_policy.lr_input requires {sorted(required)} and allows only {sorted(allowed)}."
        )
    source = lr_policy.get("source", "scc_db")
    if source not in {"scc_db", "climatology_ancillary"}:
        raise ValueError("scc_policy.lr_input.source must be scc_db or climatology_ancillary.")
    if source == "climatology_ancillary":
        station = _mapping(station_catalog.get("station"), "station")
        if not isinstance(station.get("lidar_ratio_climatology"), Mapping):
            raise ValueError("SCC climatology_ancillary requires station.lidar_ratio_climatology.")
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
        required = {"configuration_id", "name", "channels"}
        allowed = required | {"elastic_product_ids", "lr_input"}
        if set(config) - allowed or required - set(config):
            raise ValueError(
                f"SCC profile {profile_id!r}/{mode} requires {sorted(required)}; "
                f"allowed={sorted(allowed)}."
            )
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
        flags = config.get("lr_input", {})
        if not isinstance(flags, Mapping) or set(flags) - set(channels):
            raise ValueError(f"SCC {profile_id}/{mode} lr_input must map known channels.")
        for channel, flag in flags.items():
            if isinstance(flag, bool) or not isinstance(flag, Integral) or flag not in {0, 1}:
                raise ValueError(f"Invalid SCC LR_Input flag for {channel}: {flag!r}.")

        # Product IDs are not channel IDs or configuration IDs.
        product_map = config.get("elastic_product_ids", {})
        if not isinstance(product_map, Mapping):
            raise ValueError(
                f"profiles.{profile_id}.scc.{mode}.elastic_product_ids must be a mapping."
            )
        known_wavelengths = {_channel_wavelength_nm(name) for name in channels}
        product_ids: set[int] = set()
        for raw_wavelength, raw_ids in product_map.items():
            try:
                wavelength = int(raw_wavelength)
            except (TypeError, ValueError) as exc:
                raise ValueError("SCC elastic product wavelength must be numeric.") from exc
            if wavelength not in known_wavelengths or wavelength <= 0:
                raise ValueError(
                    f"profiles.{profile_id}.scc.{mode} product wavelength {wavelength} has no channel."
                )
            values = raw_ids if isinstance(raw_ids, list) else [raw_ids]
            if not values:
                raise ValueError("SCC elastic_product_ids must not contain an empty list.")
            for raw_id in values:
                product_id = _positive_int(raw_id, "SCC elastic product_ID")
                if product_id in product_ids:
                    raise ValueError(f"Duplicate SCC product_ID {product_id}.")
                product_ids.add(product_id)


def _default_lr_input(catalog: Mapping[str, Any], scc_config: Mapping[str, Any]) -> dict[str, int]:
    """Resolve station-wide LR_Input policy for one concrete SCC configuration."""
    lr_policy = catalog["scc_policy"]["lr_input"]
    companions_by_elastic = lr_policy["raman_companions_nm"]
    value = (
        0 if lr_policy.get("source", "scc_db") == "climatology_ancillary"
        else int(lr_policy["fixed_value"])
    )
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


def validate_scc_config(catalog: Mapping[str, Any], station_catalog: Mapping[str, Any]) -> None:
    """Validate SCC database references against the independent station catalog."""
    if not isinstance(catalog, Mapping):
        raise ValueError("SCC configuration must be a mapping.")
    required = {"station_id", "scc_policy", "profiles"}
    allowed = required | {"raman_channel_overrides"}
    if set(catalog) - allowed or required - set(catalog):
        raise ValueError(f"SCC configuration requires {sorted(required)}; allowed={sorted(allowed)}.")
    station_id = _text(catalog["station_id"], "scc.station_id")
    if station_id != station_catalog["station"]["id"]:
        raise ValueError(
            f"SCC station_id {station_id!r} does not match station.yaml "
            f"station.id {station_catalog['station']['id']!r}."
        )
    _validate_scc_policy(catalog, station_catalog)
    profiles = _mapping(catalog["profiles"], "scc.profiles")
    known = {str(profile["id"]) for profile in station_catalog["profiles"]}
    if not profiles:
        raise ValueError("SCC profiles mapping cannot be empty.")
    for profile_id, context in profiles.items():
        if profile_id not in known:
            raise ValueError(f"SCC references unknown station profile {profile_id!r}.")
        settings = _mapping(context, f"scc.profiles.{profile_id}")
        if set(settings) != {"day", "night"}:
            raise ValueError(f"SCC profile {profile_id!r} must define day and night.")
        _validate_scc(str(profile_id), settings)
    overrides = catalog.get("raman_channel_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("scc.raman_channel_overrides must be a mapping.")
    for profile_id, wave_map in overrides.items():
        if profile_id not in known or profile_id not in profiles:
            raise ValueError(f"Unknown SCC Raman mapping profile {profile_id!r}.")
        wave_map = _mapping(wave_map, f"scc.raman_channel_overrides.{profile_id}")
        for wavelength, fields in wave_map.items():
            if not str(wavelength).isdigit():
                raise ValueError("SCC Raman wavelength must be numeric.")
            fields = _mapping(fields, "SCC Raman wavelength metadata")
            if set(fields) - {"scc_night_channel_ids", "scc_mapping_status"}:
                raise ValueError("Unsupported SCC-only Raman metadata field.")
            if "scc_night_channel_ids" in fields:
                ids = _mapping(fields["scc_night_channel_ids"], "scc_night_channel_ids")
                for _detector, channel_id in ids.items():
                    _positive_int(channel_id, "scc_night_channel_ids")
            if "scc_mapping_status" in fields:
                _text(fields["scc_mapping_status"], "scc_mapping_status")


def resolve_scc_context(
    catalog: Mapping[str, Any],
    profile_id: str,
    available_channels: Sequence[str],
    mode: str | None,
) -> dict[str, Any]:
    """Resolve a profile/day-night SCC subset independent from scientific station metadata."""
    available = [str(channel) for channel in available_channels]
    mapping = catalog["profiles"].get(profile_id)
    if mapping is None:
        return {
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
    if mode is None:
        return {
            "scc_available": True,
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
    if mode not in {"day", "night"}:
        raise ValueError("SCC mode must be 'day' or 'night'.")
    scc = mapping[mode]
    channel_ids = {str(name): int(value) for name, value in scc["channels"].items()}
    lr_input = _resolve_lr_input(catalog, scc)
    available_set = set(available)
    missing = [name for name in channel_ids if name not in available_set]
    return {
        "scc_available": True,
        "scc_export_ready": not missing,
        "scc_configuration_id": int(scc["configuration_id"]),
        "scc_configuration_name": str(scc["name"]),
        "channel_ids": channel_ids,
        "lr_input": lr_input,
        "elastic_product_ids": deepcopy(dict(scc.get("elastic_product_ids", {}))),
        "scc_channels": [name for name in channel_ids if name in available_set],
        "missing_scc_channels": missing,
        "extra_channels": [name for name in available if name not in channel_ids],
    }
