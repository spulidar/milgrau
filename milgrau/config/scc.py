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
    if "lr_input" not in policy or set(policy) - {"lr_input", "sounding"}:
        raise ValueError("scc_policy requires lr_input; optional sounding policy allowed.")
    sounding = policy.get("sounding", {"enabled": False})
    sounding = _mapping(sounding, "scc_policy.sounding")
    if set(sounding) - {"enabled", "source", "missing_policy"}:
        raise ValueError("scc_policy.sounding has unknown keys.")
    if not isinstance(sounding.get("enabled", False), bool):
        raise ValueError("scc_policy.sounding.enabled must be boolean.")
    if sounding.get("source", "cached_wyoming") != "cached_wyoming":
        raise ValueError("scc_policy.sounding.source must be cached_wyoming.")
    if sounding.get("missing_policy", "skip") not in {"skip", "fail"}:
        raise ValueError("scc_policy.sounding.missing_policy must be skip or fail.")
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


PRODUCT_TYPES = frozenset({"elastic_backscatter", "raman_backscatter", "extinction_only", "lidar_ratio_extinction"})


def _validate_product_registry(products: Mapping[str | int, Any]) -> dict[int, dict[str, Any]]:
    """Validate original SCC product identities; cross-system ID reuse is legitimate."""
    if not products:
        raise ValueError("SCC product registry must not be empty.")
    validated: dict[int, dict[str, Any]] = {}
    for raw_id, value in products.items():
        pid = _positive_int(raw_id, "SCC product_ID")
        if pid in validated:
            raise ValueError(f"Duplicate normalized SCC product_ID {pid}.")
        spec = _mapping(value, f"SCC products.{pid}")
        required = {"type", "wavelength_nm", "usecase", "channel_count"}
        if required - set(spec) or set(spec) - (required | {"parent_product_id"}):
            raise ValueError(f"Invalid metadata fields for SCC product {pid}.")
        kind = _text(spec["type"], f"SCC product {pid} type")
        if kind not in PRODUCT_TYPES:
            raise ValueError(f"Unknown SCC product type {kind!r} for {pid}.")
        wl = _positive_int(spec["wavelength_nm"], f"SCC product {pid} wavelength")
        usecase = spec["usecase"]
        if isinstance(usecase, bool) or not isinstance(usecase, Integral) or usecase < 0:
            raise ValueError(f"SCC product {pid} usecase must be a nonnegative integer.")
        count = _positive_int(spec["channel_count"], f"SCC product {pid} channel_count")
        record = {"type": kind, "wavelength_nm": wl, "usecase": int(usecase), "channel_count": count}
        if "parent_product_id" in spec:
            record["parent_product_id"] = _positive_int(spec["parent_product_id"], f"SCC product {pid} parent")
        validated[pid] = record
    for pid, item in validated.items():
        parent = item.get("parent_product_id")
        if parent is not None:
            if parent == pid or parent not in validated:
                raise ValueError(f"SCC product {pid} has missing/self parent {parent}.")
            if validated[parent]["wavelength_nm"] != item["wavelength_nm"]:
                raise ValueError(f"SCC product {pid} parent has a different wavelength.")
            if validated[parent]["type"] != "lidar_ratio_extinction":
                raise ValueError(f"SCC product {pid} parent is not a lidar_ratio_extinction product.")
    return validated


def _elastic_products(
    system: Mapping[str, Any], registry: Mapping[int, Mapping[str, Any]],
) -> dict[str, list[int]]:
    """Select all eligible elastic products; preserve distinct IDs at the same wavelength."""
    groups: dict[str, list[int]] = {}
    for raw_id in system["product_ids"]:
        pid = int(raw_id)
        if registry[pid]["type"] == "elastic_backscatter":
            groups.setdefault(str(registry[pid]["wavelength_nm"]), []).append(pid)
    return groups


def _validate_scc(
    profile_id: str, scc: Mapping[str, Any],
    registry: Mapping[int, Mapping[str, Any]], policy: Mapping[str, Any],
) -> None:
    for mode in ("day", "night"):
        if mode not in scc:
            raise ValueError(f"profiles.{profile_id}.scc.{mode} is required when scc is configured.")
        config = _mapping(scc[mode], f"profiles.{profile_id}.scc.{mode}")
        required = {"configuration_id", "name", "channels", "product_ids"}
        allowed = required | {"lr_input"}
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
            if flag == 0:
                if policy["lr_input"].get("source", "scc_db") != "climatology_ancillary":
                    raise ValueError("SCC LR_Input=0 requires climatology_ancillary source.")
                wavelength = _channel_wavelength_nm(channel)
                companions = policy["lr_input"]["raman_companions_nm"].get(str(wavelength), [])
                channel_wavelengths = {_channel_wavelength_nm(c) for c in channels}
                if set(int(value) for value in companions) & channel_wavelengths:
                    raise ValueError(
                        f"SCC {profile_id}/{mode} cannot override Raman "
                        f"retrieval at {wavelength} nm with external LR."
                    )

        # System membership references actual optical products in the registry.
        linked = config["product_ids"]
        if not isinstance(linked, list) or not linked:
            raise ValueError(f"SCC {profile_id}/{mode} product_ids must be a nonempty list.")
        selected: set[int] = set()
        wavelengths = {_channel_wavelength_nm(name) for name in channels}
        for raw_id in linked:
            pid = _positive_int(raw_id, "SCC product_ID")
            if pid in selected:
                raise ValueError(f"Repeated SCC product_ID {pid} in {profile_id}/{mode}.")
            if pid not in registry:
                raise ValueError(f"Unknown SCC product_ID {pid} in {profile_id}/{mode}.")
            if registry[pid]["wavelength_nm"] not in wavelengths:
                raise ValueError(f"SCC product {pid} wavelength absent from {profile_id}/{mode} channels.")
            selected.add(pid)
        if policy["lr_input"].get("source", "scc_db") == "climatology_ancillary":
            elastic = _elastic_products(config, registry)
            flags = _resolve_lr_input({"scc_policy": policy}, config)
            for channel, flag in flags.items():
                if flag == 0 and str(_channel_wavelength_nm(channel)) not in elastic:
                    raise ValueError(
                        f"SCC {profile_id}/{mode} requests an external LR for {channel} "
                        "without a registered elastic_backscatter product."
                    )


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
    required = {"station_id", "scc_policy", "profiles", "products"}
    allowed = required | {"raman_channel_overrides", "server"}
    if set(catalog) - allowed or required - set(catalog):
        raise ValueError(f"SCC configuration requires {sorted(required)}; allowed={sorted(allowed)}.")
    station_id = _text(catalog["station_id"], "scc.station_id")
    if station_id != station_catalog["station"]["id"]:
        raise ValueError(
            f"SCC station_id {station_id!r} does not match station.yaml "
            f"station.id {station_catalog['station']['id']!r}."
        )
    server = catalog.get("server", {})
    if not isinstance(server, Mapping) or set(server) - {"base_url"}:
        raise ValueError("scc.server permits only base_url.")
    if "base_url" in server:
        from urllib.parse import urlsplit

        url = urlsplit(_text(server["base_url"], "scc.server.base_url"))
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.query or url.fragment or url.path not in {"", "/"}):
            raise ValueError("scc.server.base_url must be an HTTPS origin with no credentials.")
    _validate_scc_policy(catalog, station_catalog)
    registry = _validate_product_registry(_mapping(catalog["products"], "scc.products"))
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
        _validate_scc(str(profile_id), settings, registry, catalog["scc_policy"])
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
    registry = _validate_product_registry(_mapping(catalog["products"], "scc.products"))
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
        "elastic_product_ids": _elastic_products(scc, registry),
        "scc_product_ids": list(scc["product_ids"]),
        "scc_products": {int(pid): deepcopy(registry[int(pid)]) for pid in scc["product_ids"]},
        "scc_channels": [name for name in channel_ids if name in available_set],
        "missing_scc_channels": missing,
        "extra_channels": [name for name in available if name not in channel_ids],
    }
