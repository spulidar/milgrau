"""NetCDF serialization for frozen normalized atmospheric snapshots."""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import xarray as xr

from milgrau.meteorology.contracts import (
    AtmosphericProfile,
    ProfileQuality,
    create_atmospheric_profile,
)

_FLOAT_FIELDS = (
    "geometric_altitude_m",
    "geopotential_m2_s2",
    "pressure_pa",
    "temperature_k",
    "specific_humidity_kg_kg",
    "radiosonde_weight",
)
_FLAG_FIELDS = (
    "primary_source_flag",
    "interpolation_flag",
    "fallback_flag",
    "humidity_flag",
    "quality_flag",
)


def profiles_to_netcdf_bytes(
    profiles: tuple[AtmosphericProfile, ...],
    *,
    cache_metadata: dict[str, object],
) -> bytes:
    """Serialize equal-length profiles without relying on ecCodes."""
    if not profiles:
        raise ValueError("At least one atmospheric profile is required.")
    level_count = profiles[0].geometric_altitude_m.size
    if any(profile.geometric_altitude_m.size != level_count for profile in profiles):
        raise ValueError("Profiles in one normalized artifact must have equal level counts.")
    data_vars: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    for name in (*_FLOAT_FIELDS, *_FLAG_FIELDS):
        data_vars[name] = (
            ("profile", "level"),
            np.stack([getattr(profile, name) for profile in profiles]),
        )
    text_fields = {
        "nominal_time_utc": [profile.nominal_time.isoformat() for profile in profiles],
        "observation_time_utc": [profile.observation_time.isoformat() for profile in profiles],
        "provider": [profile.provider for profile in profiles],
        "station_or_dataset_id": [profile.station_or_dataset_id for profile in profiles],
        "raw_snapshot_sha256": [profile.raw_snapshot_sha256 for profile in profiles],
        "normalizer_version": [profile.normalizer_version for profile in profiles],
        "profile_quality": [profile.profile_quality.value for profile in profiles],
    }
    for name, values in text_fields.items():
        data_vars[name] = (("profile",), np.asarray(values, dtype=str))
    for name, values in {
        "latitude_deg_north": [profile.latitude_deg_north for profile in profiles],
        "longitude_deg_east": [profile.longitude_deg_east for profile in profiles],
        "quantitative_retrieval_allowed": [
            int(profile.quantitative_retrieval_allowed) for profile in profiles
        ],
    }.items():
        data_vars[name] = (("profile",), np.asarray(values))
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "profile": np.arange(len(profiles), dtype=np.int32),
            "level": np.arange(level_count, dtype=np.int32),
        },
        attrs={str(key): value for key, value in cache_metadata.items()},
    )
    payload = dataset.to_netcdf(path=None, engine="scipy")
    return bytes(payload)


def profiles_from_netcdf_bytes(payload: bytes) -> tuple[AtmosphericProfile, ...]:
    """Reconstruct validated kernel profiles from one normalized cache artifact."""
    with xr.open_dataset(io.BytesIO(payload), engine="scipy") as opened:
        dataset = opened.load()
    profiles: list[AtmosphericProfile] = []
    for index in range(dataset.sizes["profile"]):
        selected = dataset.isel(profile=index)
        finite = np.isfinite(selected["pressure_pa"].values)
        altitude = np.asarray(selected["geometric_altitude_m"].values, dtype=np.float64)
        profiles.append(
            create_atmospheric_profile(
                geometric_altitude_m=altitude,
                geopotential_m2_s2=np.asarray(
                    selected["geopotential_m2_s2"].values, dtype=np.float64
                ),
                pressure_pa=np.asarray(selected["pressure_pa"].values, dtype=np.float64),
                temperature_k=np.asarray(
                    selected["temperature_k"].values, dtype=np.float64
                ),
                specific_humidity_kg_kg=np.asarray(
                    selected["specific_humidity_kg_kg"].values, dtype=np.float64
                ),
                primary_source_flag=np.asarray(
                    selected["primary_source_flag"].values, dtype=np.int8
                ),
                interpolation_flag=np.asarray(
                    selected["interpolation_flag"].values, dtype=np.int8
                ),
                fallback_flag=np.asarray(
                    selected["fallback_flag"].values, dtype=np.int8
                ),
                humidity_flag=np.asarray(selected["humidity_flag"].values, dtype=np.int8),
                radiosonde_weight=np.asarray(
                    selected["radiosonde_weight"].values, dtype=np.float64
                ),
                quality_flag=np.asarray(selected["quality_flag"].values, dtype=np.int8),
                nominal_time=datetime.fromisoformat(str(selected["nominal_time_utc"].item())),
                observation_time=datetime.fromisoformat(
                    str(selected["observation_time_utc"].item())
                ),
                latitude_deg_north=float(selected["latitude_deg_north"].item()),
                longitude_deg_east=float(selected["longitude_deg_east"].item()),
                provider=str(selected["provider"].item()),
                station_or_dataset_id=str(selected["station_or_dataset_id"].item()),
                raw_snapshot_sha256=str(selected["raw_snapshot_sha256"].item()),
                normalizer_version=str(selected["normalizer_version"].item()),
                vertical_coverage_m=(float(altitude[finite][0]), float(altitude[finite][-1])),
                profile_quality=ProfileQuality(str(selected["profile_quality"].item())),
                quantitative_retrieval_allowed=bool(
                    selected["quantitative_retrieval_allowed"].item()
                ),
            )
        )
    return tuple(profiles)
