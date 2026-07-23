"""Pure vertical, bilinear-spatial and linear-temporal profile interpolation."""

from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np

from milgrau.meteorology.contracts import (
    AtmosphericProfile,
    FallbackFlag,
    HumidityFlag,
    InterpolationFlag,
    PrimarySource,
    ProfileQuality,
    QualityFlag,
    create_atmospheric_profile,
)


def bilinear_interpolate_four_points(
    coordinates_lat_lon: np.ndarray,
    values: np.ndarray,
    target_latitude: float,
    target_longitude: float,
) -> np.ndarray:
    """Interpolate four rectangular corner values; input point order is irrelevant."""
    coordinates = np.asarray(coordinates_lat_lon, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    if coordinates.shape != (4, 2) or data.shape[0] != 4:
        raise ValueError("Expected four (latitude, longitude) corners and four value rows.")
    latitudes = np.unique(coordinates[:, 0])
    longitudes = np.unique(coordinates[:, 1])
    if latitudes.size != 2 or longitudes.size != 2:
        raise ValueError("The four coordinates must form a two-by-two latitude/longitude rectangle.")
    expected = {(lat, lon) for lat in latitudes for lon in longitudes}
    actual = {(lat, lon) for lat, lon in coordinates}
    if actual != expected:
        raise ValueError("The four coordinates do not contain every rectangle corner exactly once.")
    latitude = float(target_latitude)
    longitude = float(target_longitude)
    if not (latitudes[0] <= latitude <= latitudes[1]) or not (
        longitudes[0] <= longitude <= longitudes[1]
    ):
        raise ValueError("Target coordinate lies outside the four-point interpolation box.")
    by_corner = {
        (lat, lon): data[index]
        for index, (lat, lon) in enumerate(coordinates)
    }
    y = (latitude - latitudes[0]) / (latitudes[1] - latitudes[0])
    x = (longitude - longitudes[0]) / (longitudes[1] - longitudes[0])
    southwest = by_corner[(latitudes[0], longitudes[0])]
    southeast = by_corner[(latitudes[0], longitudes[1])]
    northwest = by_corner[(latitudes[1], longitudes[0])]
    northeast = by_corner[(latitudes[1], longitudes[1])]
    return (
        southwest * (1.0 - x) * (1.0 - y)
        + southeast * x * (1.0 - y)
        + northwest * (1.0 - x) * y
        + northeast * x * y
    )


def _source_between(left: int, right: int, weight: float) -> int:
    if left == right:
        return left
    if left == int(PrimarySource.INVALID) or right == int(PrimarySource.INVALID):
        return int(PrimarySource.INVALID)
    if weight <= 0.0:
        return left
    if weight >= 1.0:
        return right
    return int(PrimarySource.BLENDED)


def interpolate_profile_to_altitudes(
    profile: AtmosphericProfile,
    target_geometric_altitude_m: np.ndarray,
    *,
    allow_extrapolation: bool = False,
    maximum_gap_m: float | None = None,
) -> AtmosphericProfile:
    """Interpolate P in log-space and T/q/geopotential linearly onto an MSL grid."""
    if not isinstance(profile, AtmosphericProfile):
        raise TypeError("profile must be AtmosphericProfile.")
    target = np.asarray(target_geometric_altitude_m, dtype=np.float64)
    if target.ndim != 1 or target.size < 2 or np.any(~np.isfinite(target)):
        raise ValueError("target_geometric_altitude_m must be a finite 1D array.")
    if not np.all(np.diff(target) > 0.0):
        raise ValueError("The target lidar grid must be strictly increasing.")
    if maximum_gap_m is not None and (not np.isfinite(maximum_gap_m) or maximum_gap_m <= 0.0):
        raise ValueError("maximum_gap_m must be positive when supplied.")

    source_altitude = profile.geometric_altitude_m
    count = target.size
    pressure = np.full(count, np.nan, dtype=np.float64)
    temperature = np.full(count, np.nan, dtype=np.float64)
    humidity = np.full(count, np.nan, dtype=np.float64)
    geopotential = np.interp(target, source_altitude, profile.geopotential_m2_s2)
    source_flag = np.full(count, int(PrimarySource.INVALID), dtype=np.int8)
    operation_flag = np.full(count, int(InterpolationFlag.INVALID), dtype=np.int8)
    fallback_flag = np.full(count, int(FallbackFlag.NONE), dtype=np.int8)
    humidity_flag = np.full(count, int(HumidityFlag.MISSING), dtype=np.int8)
    weight = np.zeros(count, dtype=np.float64)
    quality = np.full(count, int(QualityFlag.INVALID), dtype=np.int8)

    for output_index, altitude in enumerate(target):
        exact = np.flatnonzero(np.isclose(source_altitude, altitude, rtol=0.0, atol=1e-9))
        if exact.size:
            index = int(exact[0])
            if np.isfinite(profile.pressure_pa[index]) and np.isfinite(profile.temperature_k[index]):
                pressure[output_index] = profile.pressure_pa[index]
                temperature[output_index] = profile.temperature_k[index]
                humidity[output_index] = profile.specific_humidity_kg_kg[index]
                geopotential[output_index] = profile.geopotential_m2_s2[index]
                source_flag[output_index] = profile.primary_source_flag[index]
                operation_flag[output_index] = int(InterpolationFlag.DIRECT)
                fallback_flag[output_index] = profile.fallback_flag[index]
                humidity_flag[output_index] = profile.humidity_flag[index]
                weight[output_index] = profile.radiosonde_weight[index]
                quality[output_index] = profile.quality_flag[index]
            continue

        insertion = int(np.searchsorted(source_altitude, altitude))
        outside = insertion == 0 or insertion == source_altitude.size
        if outside and not allow_extrapolation:
            geopotential[output_index] = np.interp(
                altitude,
                source_altitude,
                profile.geopotential_m2_s2,
                left=profile.geopotential_m2_s2[0],
                right=profile.geopotential_m2_s2[-1],
            )
            continue
        if insertion == 0:
            left, right = 0, 1
        elif insertion == source_altitude.size:
            left, right = source_altitude.size - 2, source_altitude.size - 1
        else:
            left, right = insertion - 1, insertion
        gap = source_altitude[right] - source_altitude[left]
        if maximum_gap_m is not None and gap > maximum_gap_m:
            continue
        if not (
            np.isfinite(profile.pressure_pa[[left, right]]).all()
            and np.isfinite(profile.temperature_k[[left, right]]).all()
        ):
            continue
        fraction = (altitude - source_altitude[left]) / gap
        pressure[output_index] = np.exp(
            (1.0 - fraction) * np.log(profile.pressure_pa[left])
            + fraction * np.log(profile.pressure_pa[right])
        )
        temperature[output_index] = (
            (1.0 - fraction) * profile.temperature_k[left]
            + fraction * profile.temperature_k[right]
        )
        geopotential[output_index] = (
            (1.0 - fraction) * profile.geopotential_m2_s2[left]
            + fraction * profile.geopotential_m2_s2[right]
        )
        if np.isfinite(profile.specific_humidity_kg_kg[[left, right]]).all():
            humidity[output_index] = np.clip(
                (1.0 - fraction) * profile.specific_humidity_kg_kg[left]
                + fraction * profile.specific_humidity_kg_kg[right],
                0.0,
                0.1,
            )
            if profile.humidity_flag[left] == profile.humidity_flag[right]:
                humidity_flag[output_index] = profile.humidity_flag[left]
            else:
                humidity_flag[output_index] = int(HumidityFlag.MEASURED)
        source_flag[output_index] = _source_between(
            int(profile.primary_source_flag[left]),
            int(profile.primary_source_flag[right]),
            float(fraction),
        )
        operation_flag[output_index] = int(
            InterpolationFlag.EXTRAPOLATED if outside else InterpolationFlag.INTERPOLATED
        )
        if profile.fallback_flag[left] == profile.fallback_flag[right]:
            fallback_flag[output_index] = profile.fallback_flag[left]
        elif source_flag[output_index] == int(PrimarySource.STANDARD_ATMOSPHERE):
            fallback_flag[output_index] = int(FallbackFlag.STANDARD_ATMOSPHERE)
        weight[output_index] = np.clip(
            (1.0 - fraction) * profile.radiosonde_weight[left]
            + fraction * profile.radiosonde_weight[right],
            0.0,
            1.0,
        )
        if fallback_flag[output_index] == int(FallbackFlag.STANDARD_ATMOSPHERE):
            quality[output_index] = int(QualityFlag.FALLBACK_DIAGNOSTIC)
        elif np.isfinite(humidity[output_index]):
            quality[output_index] = int(QualityFlag.VALID)
        else:
            quality[output_index] = int(QualityFlag.MISSING_HUMIDITY)

    valid = np.isfinite(pressure) & np.isfinite(temperature)
    if valid.sum() < 2:
        raise ValueError("Target grid has fewer than two covered atmospheric bins.")
    has_fallback = np.any(fallback_flag == int(FallbackFlag.STANDARD_ATMOSPHERE))
    incomplete = np.any(~valid) or np.any(~np.isfinite(humidity))
    profile_quality = (
        ProfileQuality.FALLBACK_DIAGNOSTIC
        if has_fallback
        else ProfileQuality.INCOMPLETE
        if incomplete
        else ProfileQuality.QUANTITATIVE
    )
    return create_atmospheric_profile(
        geometric_altitude_m=target,
        geopotential_m2_s2=geopotential,
        pressure_pa=pressure,
        temperature_k=temperature,
        specific_humidity_kg_kg=humidity,
        primary_source_flag=source_flag,
        interpolation_flag=operation_flag,
        fallback_flag=fallback_flag,
        humidity_flag=humidity_flag,
        radiosonde_weight=weight,
        quality_flag=quality,
        nominal_time=profile.nominal_time,
        observation_time=profile.observation_time,
        latitude_deg_north=profile.latitude_deg_north,
        longitude_deg_east=profile.longitude_deg_east,
        provider=profile.provider,
        station_or_dataset_id=profile.station_or_dataset_id,
        raw_snapshot_sha256=profile.raw_snapshot_sha256,
        normalizer_version=f"{profile.normalizer_version}+vertical-logp-v1",
        vertical_coverage_m=(float(target[valid][0]), float(target[valid][-1])),
        profile_quality=profile_quality,
        quantitative_retrieval_allowed=bool(
            profile.quantitative_retrieval_allowed and not has_fallback and not incomplete
        ),
        blend_method=profile.blend_method,
        blend_parameters=profile.blend_parameters,
    )


def interpolate_profiles_in_time(
    before: AtmosphericProfile,
    after: AtmosphericProfile,
    target_time: datetime,
) -> AtmosphericProfile:
    """Linearly interpolate two conformable analyses without temporal extrapolation."""
    if not isinstance(before, AtmosphericProfile) or not isinstance(after, AtmosphericProfile):
        raise TypeError("before and after must be AtmosphericProfile instances.")
    if not np.array_equal(before.geometric_altitude_m, after.geometric_altitude_m):
        raise ValueError("Temporal interpolation requires identical geometric altitude grids.")
    start = before.observation_time
    stop = after.observation_time
    if stop <= start:
        raise ValueError("The after profile must have a later observation time.")
    if target_time.tzinfo is None:
        raise TypeError("target_time must be timezone-aware.")
    if target_time == start:
        return before
    if target_time == stop:
        return after
    if not start < target_time < stop:
        raise ValueError("Temporal extrapolation is not allowed.")
    fraction = (target_time - start).total_seconds() / (stop - start).total_seconds()
    pressure = (1.0 - fraction) * before.pressure_pa + fraction * after.pressure_pa
    temperature = (1.0 - fraction) * before.temperature_k + fraction * after.temperature_k
    humidity = (1.0 - fraction) * before.specific_humidity_kg_kg + fraction * after.specific_humidity_kg_kg
    geopotential = (1.0 - fraction) * before.geopotential_m2_s2 + fraction * after.geopotential_m2_s2
    finite = np.isfinite(pressure) & np.isfinite(temperature)
    source = np.where(
        before.primary_source_flag == after.primary_source_flag,
        before.primary_source_flag,
        int(PrimarySource.BLENDED),
    ).astype(np.int8)
    source[~finite] = int(PrimarySource.INVALID)
    operation = np.full(source.shape, int(InterpolationFlag.INTERPOLATED), dtype=np.int8)
    operation[~finite] = int(InterpolationFlag.INVALID)
    fallback = np.maximum(before.fallback_flag, after.fallback_flag).astype(np.int8)
    fallback[~finite] = int(FallbackFlag.NONE)
    humidity_known = np.isfinite(humidity)
    humidity_flag = np.where(
        before.humidity_flag == after.humidity_flag,
        before.humidity_flag,
        int(HumidityFlag.MEASURED),
    ).astype(np.int8)
    humidity_flag[~humidity_known] = int(HumidityFlag.MISSING)
    weight = np.clip(
        (1.0 - fraction) * before.radiosonde_weight + fraction * after.radiosonde_weight,
        0.0,
        1.0,
    )
    quality = np.full(source.shape, int(QualityFlag.VALID), dtype=np.int8)
    quality[~humidity_known & finite] = int(QualityFlag.MISSING_HUMIDITY)
    quality[fallback == int(FallbackFlag.STANDARD_ATMOSPHERE)] = int(
        QualityFlag.FALLBACK_DIAGNOSTIC
    )
    quality[~finite] = int(QualityFlag.INVALID)
    digest = hashlib.sha256(
        f"{before.raw_snapshot_sha256}:{after.raw_snapshot_sha256}:{fraction:.17g}".encode()
    ).hexdigest()
    has_fallback = np.any(fallback == int(FallbackFlag.STANDARD_ATMOSPHERE))
    incomplete = np.any(~finite) or np.any(~humidity_known)
    profile_quality = (
        ProfileQuality.FALLBACK_DIAGNOSTIC
        if has_fallback
        else ProfileQuality.INCOMPLETE
        if incomplete
        else ProfileQuality.QUANTITATIVE
    )
    return create_atmospheric_profile(
        geometric_altitude_m=before.geometric_altitude_m,
        geopotential_m2_s2=geopotential,
        pressure_pa=pressure,
        temperature_k=temperature,
        specific_humidity_kg_kg=humidity,
        primary_source_flag=source,
        interpolation_flag=operation,
        fallback_flag=fallback,
        humidity_flag=humidity_flag,
        radiosonde_weight=weight,
        quality_flag=quality,
        nominal_time=target_time,
        observation_time=target_time,
        latitude_deg_north=(1.0 - fraction) * before.latitude_deg_north
        + fraction * after.latitude_deg_north,
        longitude_deg_east=(1.0 - fraction) * before.longitude_deg_east
        + fraction * after.longitude_deg_east,
        provider=f"temporal interpolation: {before.provider} | {after.provider}",
        station_or_dataset_id=before.station_or_dataset_id,
        raw_snapshot_sha256=digest,
        normalizer_version="linear-time-v1",
        vertical_coverage_m=(
            float(before.geometric_altitude_m[finite][0]),
            float(before.geometric_altitude_m[finite][-1]),
        ),
        profile_quality=profile_quality,
        quantitative_retrieval_allowed=bool(
            before.quantitative_retrieval_allowed
            and after.quantitative_retrieval_allowed
            and not has_fallback
            and not incomplete
        ),
    )
