"""Strict configuration and calibration resolution for Level 1 processing."""

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
class Level1Config:
    background: BackgroundConfig
    photon_counting: PhotonCountingConfig
    pbl: PblConfig


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


def resolve_level1_config(config: Mapping[str, Any]) -> Level1Config:
    """Resolve the complete productive LIPANCORA configuration without defaults."""
    level1 = _mapping(config.get("level1"), "level1")
    _exact_keys(level1, {"background", "photon_counting", "pbl"}, "level1")

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
    reference_channel = pbl["reference_channel"]
    if not isinstance(reference_channel, str) or not reference_channel.strip():
        raise Level1ConfigurationError("Configuration level1.pbl.reference_channel must be a non-empty string.")
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
        pbl=PblConfig(reference_channel.strip(), min_search, max_search, smooth_bins),
    )


def validate_level1_config(config: Mapping[str, Any]) -> None:
    """Validate the complete productive Level 1 recipe."""
    resolve_level1_config(config)


def _profile_for_dataset(config: Mapping[str, Any], ds: xr.Dataset) -> Mapping[str, Any]:
    catalog = _mapping(config.get("_station_catalog"), "_station_catalog")
    profiles = catalog.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise Level1ConfigurationError("Station catalog profiles must be a non-empty list for Level 1 calibration.")

    stored_profile_id = str(ds.attrs.get("Station_Profile", "")).strip()
    if stored_profile_id:
        matches = [profile for profile in profiles if str(profile.get("id", "")) == stored_profile_id]
        if len(matches) != 1:
            raise Level1ConfigurationError(
                f"Level 0 Station_Profile={stored_profile_id!r} does not resolve uniquely in station.yaml."
            )
        return _mapping(matches[0], f"station profile {stored_profile_id}")

    if "time" not in ds.coords or ds.sizes.get("time", 0) <= 0:
        raise Level1ConfigurationError("Level 1 calibration resolution requires a non-empty Level 0 time coordinate.")
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
            raise Level1ConfigurationError(
                f"Station profile {profile.get('id', '')!r} lacks calibration_id."
            )
        calibrations = _mapping(catalog.get("calibrations"), "_station_catalog.calibrations")
        calibration = _mapping(
            calibrations.get(calibration_id),
            f"_station_catalog.calibrations.{calibration_id}",
        )
        channels = _mapping(
            calibration.get("channels"),
            f"_station_catalog.calibrations.{calibration_id}.channels",
        )

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
        raise Level1ConfigurationError(
            f"Calibration {calibration_id}.{channel_name}.deadtime_us must be non-negative."
        )
    shift = _integer(values.get("bin_shift_bins"), f"calibration {calibration_id}.{channel_name}.bin_shift_bins", minimum=-10**9)
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
