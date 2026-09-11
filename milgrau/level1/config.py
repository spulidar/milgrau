"""Strict configuration, atmosphere policy, and calibration resolution for Level 1."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr


class Level1ConfigurationError(ValueError):
    """Raised when productive Level 1 configuration is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class BackgroundConfig:
    start_altitude_m: float
    stop_altitude_m: float


@dataclass(frozen=True, slots=True)
class PhotonCountingConfig:
    deadtime_min_denominator: float


@dataclass(frozen=True, slots=True)
class PblConfig:
    reference_channel: str
    min_search_altitude_m: float
    max_search_altitude_m: float
    smooth_bins: int


@dataclass(frozen=True, slots=True)
class RadiosondeConfig:
    cache_dir: str
    synoptic_hours_utc: tuple[int, ...]
    selection: str
    max_time_delta_hours: float

    def as_io_mapping(self) -> dict[str, Any]:
        return {
            "cache_dir": self.cache_dir,
            "synoptic_hours_utc": list(self.synoptic_hours_utc),
            "selection": self.selection,
            "max_time_delta_hours": self.max_time_delta_hours,
        }


@dataclass(frozen=True, slots=True)
class Era5Config:
    cache_dir: str
    dataset: str
    pressure_levels_hpa: tuple[int, ...]
    grid_deg: float
    area_half_width_deg: float

    def as_io_mapping(self) -> dict[str, Any]:
        return {
            "cache_dir": self.cache_dir,
            "dataset": self.dataset,
            "pressure_levels_hpa": list(self.pressure_levels_hpa),
            "grid_deg": self.grid_deg,
            "area_half_width_deg": self.area_half_width_deg,
        }


@dataclass(frozen=True, slots=True)
class AtmosphereConfig:
    source_priority: tuple[str, ...]
    external_profile_outside_coverage: str
    radiosonde: RadiosondeConfig | None
    era5: Era5Config | None


@dataclass(frozen=True, slots=True)
class Level1Config:
    background: BackgroundConfig
    photon_counting: PhotonCountingConfig
    pbl: PblConfig
    atmosphere: AtmosphereConfig


@dataclass(frozen=True, slots=True)
class ChannelCalibration:
    calibration_id: str
    channel: str
    detector_mode: str
    deadtime_us: float
    bin_shift_bins: int
    background_offset: float
    saturation_status: str
    saturation_max_rate_mhz: float | None

    @property
    def saturation_characterized(self) -> bool:
        return self.saturation_status == "characterized"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Level1ConfigurationError(f"Configuration {label} must be a mapping.")
    return value


def _exact_keys(section: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(section))
    unknown = sorted(set(section) - required)
    if missing or unknown:
        raise Level1ConfigurationError(
            f"Configuration {label} must contain exactly {sorted(required)}; missing={missing}, unknown={unknown}."
        )


def _allowed_keys(
    section: Mapping[str, Any], *, required: set[str], allowed: set[str], label: str
) -> None:
    missing = sorted(required - set(section))
    unknown = sorted(set(section) - allowed)
    if missing or unknown:
        raise Level1ConfigurationError(
            f"Configuration {label} has invalid structure; missing={missing}, unknown={unknown}."
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Level1ConfigurationError(f"Configuration {label} must be a non-empty string.")
    return value.strip()


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise Level1ConfigurationError(f"Configuration {label} must be a finite number.")
    number = float(value)
    if not np.isfinite(number):
        raise Level1ConfigurationError(f"Configuration {label} must be finite.")
    if positive and number <= 0.0:
        raise Level1ConfigurationError(f"Configuration {label} must be positive.")
    return number


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise Level1ConfigurationError(f"Configuration {label} must be an integer.")
    converted = int(value)
    if converted < minimum:
        raise Level1ConfigurationError(f"Configuration {label} must be at least {minimum}.")
    return converted


def _resolve_radiosonde_config(section: Mapping[str, Any]) -> RadiosondeConfig:
    label = "level1.atmosphere.radiosonde"
    _exact_keys(
        section,
        {"cache_dir", "synoptic_hours_utc", "selection", "max_time_delta_hours"},
        label,
    )
    cache_dir = _text(section["cache_dir"], f"{label}.cache_dir")
    selection = _text(section["selection"], f"{label}.selection").lower()
    if selection != "nearest":
        raise Level1ConfigurationError(
            f"Configuration {label}.selection currently supports only 'nearest'."
        )
    raw_hours = section["synoptic_hours_utc"]
    if not isinstance(raw_hours, list) or not raw_hours:
        raise Level1ConfigurationError(f"Configuration {label}.synoptic_hours_utc must be a non-empty list.")
    hours: list[int] = []
    for index, raw_hour in enumerate(raw_hours):
        hour = _integer(raw_hour, f"{label}.synoptic_hours_utc[{index}]")
        if hour > 23:
            raise Level1ConfigurationError(
                f"Configuration {label}.synoptic_hours_utc[{index}] must be between 0 and 23."
            )
        if hour in hours:
            raise Level1ConfigurationError(f"Configuration {label}.synoptic_hours_utc contains duplicate hour {hour}.")
        hours.append(hour)
    max_delta = _finite(section["max_time_delta_hours"], f"{label}.max_time_delta_hours", positive=True)
    return RadiosondeConfig(cache_dir, tuple(sorted(hours)), selection, max_delta)


def _resolve_era5_config(section: Mapping[str, Any]) -> Era5Config:
    label = "level1.atmosphere.era5"
    _exact_keys(
        section,
        {"cache_dir", "dataset", "pressure_levels_hpa", "grid_deg", "area_half_width_deg"},
        label,
    )
    cache_dir = _text(section["cache_dir"], f"{label}.cache_dir")
    dataset = _text(section["dataset"], f"{label}.dataset")
    grid_deg = _finite(section["grid_deg"], f"{label}.grid_deg", positive=True)
    area_half_width_deg = _finite(
        section["area_half_width_deg"], f"{label}.area_half_width_deg", positive=True
    )
    if grid_deg > 5.0:
        raise Level1ConfigurationError(f"Configuration {label}.grid_deg must be <= 5 degrees.")
    if area_half_width_deg > 10.0:
        raise Level1ConfigurationError(f"Configuration {label}.area_half_width_deg must be <= 10 degrees.")

    raw_levels = section["pressure_levels_hpa"]
    if not isinstance(raw_levels, list) or not raw_levels:
        raise Level1ConfigurationError(f"Configuration {label}.pressure_levels_hpa must be a non-empty list.")
    levels: list[int] = []
    for index, raw_level in enumerate(raw_levels):
        level = _integer(raw_level, f"{label}.pressure_levels_hpa[{index}]", minimum=1)
        if level in levels:
            raise Level1ConfigurationError(
                f"Configuration {label}.pressure_levels_hpa contains duplicate pressure level {level}."
            )
        levels.append(level)
    return Era5Config(cache_dir, dataset, tuple(levels), grid_deg, area_half_width_deg)


def _resolve_atmosphere_config(level1: Mapping[str, Any]) -> AtmosphereConfig:
    atmosphere = _mapping(level1.get("atmosphere"), "level1.atmosphere")
    allowed = {"source_priority", "external_profile_outside_coverage", "radiosonde", "era5"}
    _allowed_keys(
        atmosphere,
        required={"source_priority", "external_profile_outside_coverage"},
        allowed=allowed,
        label="level1.atmosphere",
    )

    raw_priority = atmosphere["source_priority"]
    if not isinstance(raw_priority, list) or not raw_priority:
        raise Level1ConfigurationError("Configuration level1.atmosphere.source_priority must be a non-empty list.")
    valid_sources = {"radiosonde", "era5", "ussa76"}
    priority: list[str] = []
    for index, raw_source in enumerate(raw_priority):
        source = _text(raw_source, f"level1.atmosphere.source_priority[{index}]").lower()
        if source not in valid_sources:
            raise Level1ConfigurationError(
                f"Configuration level1.atmosphere.source_priority[{index}] must be one of {sorted(valid_sources)}."
            )
        if source in priority:
            raise Level1ConfigurationError(
                f"Configuration level1.atmosphere.source_priority contains duplicate source {source!r}."
            )
        priority.append(source)

    extension = _text(
        atmosphere["external_profile_outside_coverage"],
        "level1.atmosphere.external_profile_outside_coverage",
    ).lower()
    if extension not in {"ussa76", "fail"}:
        raise Level1ConfigurationError(
            "Configuration level1.atmosphere.external_profile_outside_coverage must be 'ussa76' or 'fail'."
        )

    radiosonde = None
    if "radiosonde" in priority:
        radiosonde = _resolve_radiosonde_config(
            _mapping(atmosphere.get("radiosonde"), "level1.atmosphere.radiosonde")
        )
    elif "radiosonde" in atmosphere:
        raise Level1ConfigurationError(
            "Configuration level1.atmosphere.radiosonde is present but radiosonde is absent from source_priority."
        )

    era5 = None
    if "era5" in priority:
        era5 = _resolve_era5_config(_mapping(atmosphere.get("era5"), "level1.atmosphere.era5"))
    elif "era5" in atmosphere:
        raise Level1ConfigurationError(
            "Configuration level1.atmosphere.era5 is present but era5 is absent from source_priority."
        )

    return AtmosphereConfig(tuple(priority), extension, radiosonde, era5)


def resolve_level1_config(config: Mapping[str, Any]) -> Level1Config:
    """Resolve the complete productive LIPANCORA configuration without defaults."""
    level1 = _mapping(config.get("level1"), "level1")
    _exact_keys(level1, {"background", "photon_counting", "pbl", "atmosphere"}, "level1")

    background = _mapping(level1["background"], "level1.background")
    _exact_keys(background, {"start_altitude_m", "stop_altitude_m"}, "level1.background")
    background_start = _finite(background["start_altitude_m"], "level1.background.start_altitude_m", positive=True)
    background_stop = _finite(background["stop_altitude_m"], "level1.background.stop_altitude_m", positive=True)
    if background_stop <= background_start:
        raise Level1ConfigurationError(
            "Configuration level1.background.stop_altitude_m must exceed start_altitude_m."
        )

    photon = _mapping(level1["photon_counting"], "level1.photon_counting")
    _exact_keys(photon, {"deadtime_min_denominator"}, "level1.photon_counting")
    minimum_denominator = _finite(
        photon["deadtime_min_denominator"],
        "level1.photon_counting.deadtime_min_denominator",
        positive=True,
    )
    if minimum_denominator > 1.0:
        raise Level1ConfigurationError(
            "Configuration level1.photon_counting.deadtime_min_denominator must be <= 1."
        )

    pbl = _mapping(level1["pbl"], "level1.pbl")
    _exact_keys(
        pbl,
        {"reference_channel", "min_search_altitude_m", "max_search_altitude_m", "smooth_bins"},
        "level1.pbl",
    )
    reference_channel = _text(pbl["reference_channel"], "level1.pbl.reference_channel")
    min_search = _finite(pbl["min_search_altitude_m"], "level1.pbl.min_search_altitude_m", positive=True)
    max_search = _finite(pbl["max_search_altitude_m"], "level1.pbl.max_search_altitude_m", positive=True)
    if max_search <= min_search:
        raise Level1ConfigurationError(
            "Configuration level1.pbl.max_search_altitude_m must exceed min_search_altitude_m."
        )
    smooth_bins = _integer(pbl["smooth_bins"], "level1.pbl.smooth_bins", minimum=3)
    if smooth_bins % 2 == 0:
        raise Level1ConfigurationError("Configuration level1.pbl.smooth_bins must be an odd integer.")

    return Level1Config(
        background=BackgroundConfig(background_start, background_stop),
        photon_counting=PhotonCountingConfig(minimum_denominator),
        pbl=PblConfig(reference_channel, min_search, max_search, smooth_bins),
        atmosphere=_resolve_atmosphere_config(level1),
    )


def validate_level1_config(config: Mapping[str, Any]) -> None:
    """Validate the complete productive Level 1 recipe."""
    resolve_level1_config(config)


def _profile_for_dataset(config: Mapping[str, Any], ds: xr.Dataset) -> Mapping[str, Any]:
    catalog = _mapping(config.get("_station_catalog"), "_station_catalog")
    profiles = catalog.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise Level1ConfigurationError("Station catalog profiles must be a non-empty list for Level 1 processing.")

    stored_profile_id = str(ds.attrs.get("Station_Profile", "")).strip()
    if stored_profile_id:
        matches = [profile for profile in profiles if str(profile.get("id", "")) == stored_profile_id]
        if len(matches) != 1:
            raise Level1ConfigurationError(
                f"Level 0 Station_Profile={stored_profile_id!r} does not resolve uniquely in station.yaml."
            )
        return _mapping(matches[0], f"station profile {stored_profile_id}")

    if "time" not in ds.coords or ds.sizes.get("time", 0) <= 0:
        raise Level1ConfigurationError("Station-profile resolution requires a non-empty Level 0 time coordinate.")
    measurement_date = pd.Timestamp(ds["time"].values[0]).date()
    matches = []
    for raw_profile in profiles:
        profile = _mapping(raw_profile, "station profile")
        start = pd.Timestamp(profile["valid_from"]).date()
        end_raw = profile.get("valid_to")
        end = None if end_raw is None else pd.Timestamp(end_raw).date()
        if measurement_date >= start and (end is None or measurement_date <= end):
            matches.append(profile)
    if len(matches) != 1:
        raise Level1ConfigurationError(
            f"Expected exactly one station profile for {measurement_date.isoformat()}, found "
            f"{[str(profile.get('id', '')) for profile in matches]}."
        )
    return matches[0]


def resolve_station_site(config: Mapping[str, Any], ds: xr.Dataset) -> dict[str, float]:
    """Resolve profile-aware station coordinates/altitude without scientific fallbacks."""
    catalog = _mapping(config.get("_station_catalog"), "_station_catalog")
    station = _mapping(catalog.get("station"), "_station_catalog.station")
    base_site = dict(_mapping(station.get("site"), "_station_catalog.station.site"))
    profile = _profile_for_dataset(config, ds)
    profile_site = profile.get("site", {})
    if profile_site is not None:
        base_site.update(dict(_mapping(profile_site, f"profiles.{profile.get('id', '')}.site")))

    latitude = _finite(base_site.get("latitude"), "resolved station latitude")
    longitude = _finite(base_site.get("longitude"), "resolved station longitude")
    altitude = _finite(base_site.get("station_altitude_m"), "resolved station altitude")
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise Level1ConfigurationError("Resolved station latitude/longitude are outside valid bounds.")
    return {
        "latitude": latitude,
        "longitude": longitude,
        "station_altitude_m": altitude,
    }


def resolve_radiosonde_station(config: Mapping[str, Any]) -> tuple[str, str]:
    """Return radiosonde station identity only from station.yaml."""
    catalog = _mapping(config.get("_station_catalog"), "_station_catalog")
    station = _mapping(catalog.get("station"), "_station_catalog.station")
    radiosonde = _mapping(station.get("radiosonde"), "_station_catalog.station.radiosonde")
    return (
        _text(radiosonde.get("station_id"), "station.radiosonde.station_id"),
        _text(radiosonde.get("station_name"), "station.radiosonde.station_name"),
    )


def resolve_channel_calibration(
    config: Mapping[str, Any],
    ds: xr.Dataset,
    channel_name: str,
) -> ChannelCalibration:
    """Resolve one channel's full instrument calibration for this Level 0 dataset."""
    resolved_station = config.get("_resolved_station")
    if isinstance(resolved_station, Mapping) and isinstance(resolved_station.get("channel_calibrations"), Mapping):
        calibration_id = str(resolved_station.get("calibration_id", "")).strip()
        channels = resolved_station["channel_calibrations"]
    else:
        catalog = _mapping(config.get("_station_catalog"), "_station_catalog")
        profile = _profile_for_dataset(config, ds)
        calibration_id = str(profile.get("calibration_id", "")).strip()
        if not calibration_id:
            raise Level1ConfigurationError(f"Station profile {profile.get('id', '')!r} lacks calibration_id.")
        calibrations = _mapping(catalog.get("calibrations"), "_station_catalog.calibrations")
        calibration = _mapping(calibrations.get(calibration_id), f"_station_catalog.calibrations.{calibration_id}")
        channels = _mapping(calibration.get("channels"), f"_station_catalog.calibrations.{calibration_id}.channels")

    if channel_name not in channels:
        raise Level1ConfigurationError(
            f"Channel {channel_name} has no calibration in instrument calibration {calibration_id!r}."
        )
    values = _mapping(channels[channel_name], f"calibration {calibration_id}.{channel_name}")
    detector_mode = str(values.get("detector_mode", "")).strip()
    if detector_mode not in {"analog", "photon_counting"}:
        raise Level1ConfigurationError(
            f"Calibration {calibration_id}.{channel_name}.detector_mode is invalid: {detector_mode!r}."
        )
    deadtime = _finite(values.get("deadtime_us"), f"calibration {calibration_id}.{channel_name}.deadtime_us")
    if deadtime < 0.0:
        raise Level1ConfigurationError(f"Calibration {calibration_id}.{channel_name}.deadtime_us must be non-negative.")
    shift = _integer(
        values.get("bin_shift_bins"),
        f"calibration {calibration_id}.{channel_name}.bin_shift_bins",
        minimum=-10**9,
    )
    background_offset = _finite(
        values.get("background_offset"),
        f"calibration {calibration_id}.{channel_name}.background_offset",
    )

    if detector_mode == "photon_counting":
        saturation = _mapping(values.get("saturation"), f"calibration {calibration_id}.{channel_name}.saturation")
        status = str(saturation.get("status", "")).strip()
        if status not in {"characterized", "not_characterized"}:
            raise Level1ConfigurationError(
                f"Calibration {calibration_id}.{channel_name}.saturation.status is invalid: {status!r}."
            )
        if status == "characterized":
            max_rate = _finite(
                saturation.get("max_rate_mhz"),
                f"calibration {calibration_id}.{channel_name}.saturation.max_rate_mhz",
                positive=True,
            )
        else:
            max_rate = None
    else:
        status = "not_applicable"
        max_rate = None

    return ChannelCalibration(
        calibration_id=calibration_id,
        channel=str(channel_name),
        detector_mode=detector_mode,
        deadtime_us=deadtime,
        bin_shift_bins=shift,
        background_offset=background_offset,
        saturation_status=status,
        saturation_max_rate_mhz=max_rate,
    )
