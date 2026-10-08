"""Deterministic solar-position utilities used by MILGRAU scientific regimes.

The implementation follows the standard NOAA solar-position equations for the
solar center. Elevation is geometric: no atmospheric-refraction correction is
applied, so regime classification is independent of surface meteorology.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SOLAR_POSITION_ALGORITHM = "NOAA solar position equations; geometric solar-center elevation"


def _as_utc_index(times_utc: Any) -> pd.DatetimeIndex:
    values = pd.to_datetime(times_utc, utc=True, errors="coerce")
    if isinstance(values, pd.Timestamp):
        values = pd.DatetimeIndex([values])
    else:
        values = pd.DatetimeIndex(values)
    if values.isna().any():
        raise ValueError("Solar-position times must be valid UTC datetimes.")
    return values


def solar_elevation_deg(
    times_utc: Any,
    latitude_deg: float,
    longitude_deg: float,
) -> np.ndarray:
    """Return geometric solar-center elevation in degrees for UTC timestamps."""
    latitude = float(latitude_deg)
    longitude = float(longitude_deg)
    if not np.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise ValueError("latitude_deg must be finite and within [-90, 90].")
    if not np.isfinite(longitude) or not -180.0 <= longitude <= 180.0:
        raise ValueError("longitude_deg must be finite and within [-180, 180].")

    times = _as_utc_index(times_utc)
    unix_seconds = times.asi8.astype(np.float64) / 1.0e9
    julian_day = unix_seconds / 86400.0 + 2440587.5
    century = (julian_day - 2451545.0) / 36525.0

    geom_mean_long = np.mod(
        280.46646 + century * (36000.76983 + 0.0003032 * century),
        360.0,
    )
    geom_mean_anomaly = (
        357.52911 + century * (35999.05029 - 0.0001537 * century)
    )
    eccentricity = (
        0.016708634
        - century * (0.000042037 + 0.0000001267 * century)
    )

    anomaly_rad = np.deg2rad(geom_mean_anomaly)
    equation_center = (
        np.sin(anomaly_rad)
        * (1.914602 - century * (0.004817 + 0.000014 * century))
        + np.sin(2.0 * anomaly_rad) * (0.019993 - 0.000101 * century)
        + np.sin(3.0 * anomaly_rad) * 0.000289
    )
    true_longitude = geom_mean_long + equation_center
    omega = 125.04 - 1934.136 * century
    apparent_longitude = (
        true_longitude
        - 0.00569
        - 0.00478 * np.sin(np.deg2rad(omega))
    )

    mean_obliquity = (
        23.0
        + (
            26.0
            + (
                21.448
                - century
                * (46.815 + century * (0.00059 - century * 0.001813))
            )
            / 60.0
        )
        / 60.0
    )
    corrected_obliquity = (
        mean_obliquity
        + 0.00256 * np.cos(np.deg2rad(omega))
    )

    obliquity_rad = np.deg2rad(corrected_obliquity)
    apparent_longitude_rad = np.deg2rad(apparent_longitude)
    declination = np.arcsin(
        np.sin(obliquity_rad) * np.sin(apparent_longitude_rad)
    )

    y = np.tan(obliquity_rad / 2.0) ** 2
    long_rad = np.deg2rad(geom_mean_long)
    equation_time_minutes = 4.0 * np.rad2deg(
        y * np.sin(2.0 * long_rad)
        - 2.0 * eccentricity * np.sin(anomaly_rad)
        + 4.0
        * eccentricity
        * y
        * np.sin(anomaly_rad)
        * np.cos(2.0 * long_rad)
        - 0.5 * y * y * np.sin(4.0 * long_rad)
        - 1.25
        * eccentricity
        * eccentricity
        * np.sin(2.0 * anomaly_rad)
    )

    minutes_utc = (
        times.hour.to_numpy(dtype=np.float64) * 60.0
        + times.minute.to_numpy(dtype=np.float64)
        + times.second.to_numpy(dtype=np.float64) / 60.0
        + times.microsecond.to_numpy(dtype=np.float64) / 60.0e6
    )
    true_solar_minutes = np.mod(
        minutes_utc + equation_time_minutes + 4.0 * longitude,
        1440.0,
    )
    hour_angle_deg = true_solar_minutes / 4.0 - 180.0
    hour_angle = np.deg2rad(hour_angle_deg)
    latitude_rad = np.deg2rad(latitude)

    cos_zenith = (
        np.sin(latitude_rad) * np.sin(declination)
        + np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    zenith = np.arccos(np.clip(cos_zenith, -1.0, 1.0))
    return 90.0 - np.rad2deg(zenith)


def solar_regime(
    elevation_deg: Any,
    *,
    day_night_threshold_deg: float,
) -> np.ndarray:
    """Classify geometric solar elevation as day/night using one explicit threshold."""
    threshold = float(day_night_threshold_deg)
    if not np.isfinite(threshold) or not -90.0 <= threshold <= 90.0:
        raise ValueError("day_night_threshold_deg must be finite and within [-90, 90].")
    elevation = np.asarray(elevation_deg, dtype=np.float64)
    if not np.all(np.isfinite(elevation)):
        raise ValueError("Solar elevation must be finite for regime classification.")
    return np.where(elevation >= threshold, "day", "night").astype(object)


__all__ = ["SOLAR_POSITION_ALGORITHM", "solar_elevation_deg", "solar_regime"]
