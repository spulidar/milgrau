"""Immutable acquisition request and minimal ERA5-hour planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Iterable


class AcquisitionMode(StrEnum):
    """Approved network/cache behavior."""

    AUTO = "auto"
    CACHE_ONLY = "cache_only"
    PREFETCH = "prefetch"


class MeteorologyProvider(StrEnum):
    """Providers requested by one acquisition operation."""

    RADIOSONDE = "radiosonde"
    ERA5 = "era5"
    BOTH = "both"


ERA5_VARIABLES = ("temperature", "specific_humidity", "lnsp", "surface_geopotential")
ERA5_MODEL_LEVELS = tuple(range(1, 138))
METEOROLOGY_REQUEST_VERSION = "meteorology-request-v1"


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TypeError(f"{label} must be timezone-aware.")
    return value.astimezone(UTC)


def _ordered_unique_utc(values: Iterable[datetime], label: str) -> tuple[datetime, ...]:
    normalized = {_utc(value, label) for value in values}
    if not normalized:
        raise ValueError(f"{label} must contain at least one timestamp.")
    return tuple(sorted(normalized))


def plan_era5_hours(measurement_timestamps: Iterable[datetime]) -> tuple[datetime, ...]:
    """Return only hourly analyses needed to bracket the supplied measurements."""
    measurements = _ordered_unique_utc(measurement_timestamps, "measurement_timestamps")
    planned: set[datetime] = set()
    for timestamp in measurements:
        before = timestamp.replace(minute=0, second=0, microsecond=0)
        planned.add(before)
        if timestamp != before:
            planned.add(before + timedelta(hours=1))
    return tuple(sorted(planned))


def hourly_timestamps_for_interval(start: datetime, end: datetime) -> tuple[datetime, ...]:
    """Expand a prefetch interval to every analysis hour that brackets it."""
    start_utc = _utc(start, "start")
    end_utc = _utc(end, "end")
    if end_utc < start_utc:
        raise ValueError("end must not precede start.")
    first = start_utc.replace(minute=0, second=0, microsecond=0)
    final = end_utc.replace(minute=0, second=0, microsecond=0)
    if end_utc != final:
        final += timedelta(hours=1)
    count = int((final - first).total_seconds() // 3600) + 1
    return tuple(first + timedelta(hours=index) for index in range(count))


def group_era5_hours_by_month(
    hours: Iterable[datetime],
) -> tuple[tuple[tuple[int, int], tuple[datetime, ...]], ...]:
    """Group sorted unique UTC analyses by calendar month."""
    normalized = _ordered_unique_utc(hours, "era5_hours")
    groups: dict[tuple[int, int], list[datetime]] = {}
    for timestamp in normalized:
        groups.setdefault((timestamp.year, timestamp.month), []).append(timestamp)
    return tuple((key, tuple(groups[key])) for key in sorted(groups))


def surrounding_grid_points(
    latitude_deg_north: float,
    longitude_deg_east: float,
    grid_degrees: float = 0.25,
) -> tuple[tuple[float, float], ...]:
    """Return the deterministic southwest/southeast/northwest/northeast box."""
    latitude = float(latitude_deg_north)
    longitude = float(longitude_deg_east)
    grid = float(grid_degrees)
    if not all(math.isfinite(value) for value in (latitude, longitude, grid)):
        raise ValueError("Site coordinates and ERA5 grid spacing must be finite.")
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise ValueError("Site coordinates are outside latitude/longitude bounds.")
    if grid <= 0.0 or grid > 180.0:
        raise ValueError("ERA5 grid spacing must be positive and no larger than 180 degrees.")

    south = math.floor(latitude / grid) * grid
    north = math.ceil(latitude / grid) * grid
    west = math.floor(longitude / grid) * grid
    east = math.ceil(longitude / grid) * grid
    if math.isclose(south, north, abs_tol=1e-12):
        south -= grid
        north += grid
    if math.isclose(west, east, abs_tol=1e-12):
        west -= grid
        east += grid
    return (
        (round(south, 10), round(west, 10)),
        (round(south, 10), round(east, 10)),
        (round(north, 10), round(west, 10)),
        (round(north, 10), round(east, 10)),
    )


@dataclass(frozen=True, slots=True)
class MeteorologyRequest:
    """Validated, credential-free contract for one acquisition operation."""

    site_id: str
    latitude_deg_north: float
    longitude_deg_east: float
    station_altitude_m: float
    measurement_timestamps: tuple[datetime, ...]
    provider: MeteorologyProvider
    mode: AcquisitionMode
    cache_directory: Path
    radiosonde_station_id: str = "83779"
    radiosonde_nominal_times: tuple[datetime, ...] = ()
    era5_variables: tuple[str, ...] = ERA5_VARIABLES
    era5_model_levels: tuple[int, ...] = ERA5_MODEL_LEVELS
    era5_grid_degrees: float = 0.25
    allow_era5t: bool = True
    timeout_seconds: float = 300.0
    max_retries: int = 3
    contract_version: str = METEOROLOGY_REQUEST_VERSION
    fallback_altitudes_m: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in ("site_id", "radiosonde_station_id", "contract_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, value.strip())
        if not self.radiosonde_station_id.isdigit():
            raise ValueError("radiosonde_station_id must contain only digits.")
        if not isinstance(self.provider, MeteorologyProvider):
            raise TypeError("provider must be MeteorologyProvider.")
        if not isinstance(self.mode, AcquisitionMode):
            raise TypeError("mode must be AcquisitionMode.")
        if not isinstance(self.allow_era5t, bool):
            raise TypeError("allow_era5t must be boolean.")

        latitude = float(self.latitude_deg_north)
        longitude = float(self.longitude_deg_east)
        altitude = float(self.station_altitude_m)
        grid = float(self.era5_grid_degrees)
        timeout = float(self.timeout_seconds)
        if not all(math.isfinite(value) for value in (latitude, longitude, altitude, grid, timeout)):
            raise ValueError("Coordinates, altitude, grid and timeout must be finite.")
        if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
            raise ValueError("Site coordinates are outside latitude/longitude bounds.")
        if grid <= 0.0 or timeout <= 0.0:
            raise ValueError("ERA5 grid and timeout must be positive.")
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 1
        ):
            raise ValueError("max_retries must be a positive integer.")
        object.__setattr__(self, "latitude_deg_north", latitude)
        object.__setattr__(self, "longitude_deg_east", longitude)
        object.__setattr__(self, "station_altitude_m", altitude)
        object.__setattr__(self, "era5_grid_degrees", grid)
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "cache_directory", Path(self.cache_directory).expanduser())
        object.__setattr__(
            self,
            "measurement_timestamps",
            _ordered_unique_utc(self.measurement_timestamps, "measurement_timestamps"),
        )

        radiosonde_times = tuple(
            sorted({_utc(value, "radiosonde_nominal_times") for value in self.radiosonde_nominal_times})
        )
        if self.provider in {MeteorologyProvider.RADIOSONDE, MeteorologyProvider.BOTH}:
            if not radiosonde_times:
                raise ValueError(
                    "radiosonde_nominal_times must explicitly select previous, next or nominal soundings."
                )
            if any(
                value.minute != 0
                or value.second != 0
                or value.microsecond != 0
                or value.hour not in {0, 12}
                for value in radiosonde_times
            ):
                raise ValueError("Radiosonde nominal times must be exact 00 or 12 UTC soundings.")
        object.__setattr__(self, "radiosonde_nominal_times", radiosonde_times)

        variables = tuple(str(value) for value in self.era5_variables)
        levels = tuple(int(value) for value in self.era5_model_levels)
        if variables != ERA5_VARIABLES:
            raise ValueError(f"era5_variables must be exactly {ERA5_VARIABLES}.")
        if levels != ERA5_MODEL_LEVELS:
            raise ValueError("era5_model_levels must be exactly 1 through 137.")
        object.__setattr__(self, "era5_variables", variables)
        object.__setattr__(self, "era5_model_levels", levels)

        fallback = tuple(float(value) for value in self.fallback_altitudes_m)
        if not fallback:
            fallback = (altitude, altitude + 30_000.0)
        if (
            len(fallback) < 2
            or any(not math.isfinite(value) or value < 0.0 for value in fallback)
            or any(right <= left for left, right in zip(fallback, fallback[1:], strict=False))
        ):
            raise ValueError("fallback_altitudes_m must be finite, non-negative and strictly increasing.")
        object.__setattr__(self, "fallback_altitudes_m", fallback)

        points = self.era5_grid_points
        latitudes = sorted({point[0] for point in points})
        longitudes = sorted({point[1] for point in points})
        if len(latitudes) != 2 or len(longitudes) != 2:
            raise ValueError("ERA5 area must contain exactly four surrounding grid points.")
        if not (
            latitudes[0] <= latitude <= latitudes[1]
            and longitudes[0] <= longitude <= longitudes[1]
        ):
            raise ValueError("ERA5 four-point area must contain the site.")

    @property
    def era5_hours(self) -> tuple[datetime, ...]:
        return plan_era5_hours(self.measurement_timestamps)

    @property
    def era5_month_groups(
        self,
    ) -> tuple[tuple[tuple[int, int], tuple[datetime, ...]], ...]:
        return group_era5_hours_by_month(self.era5_hours)

    @property
    def era5_grid_points(self) -> tuple[tuple[float, float], ...]:
        return surrounding_grid_points(
            self.latitude_deg_north,
            self.longitude_deg_east,
            self.era5_grid_degrees,
        )

    @property
    def era5_area_north_west_south_east(self) -> tuple[float, float, float, float]:
        points = self.era5_grid_points
        latitudes = [point[0] for point in points]
        longitudes = [point[1] for point in points]
        return (max(latitudes), min(longitudes), min(latitudes), max(longitudes))

    def canonical_payload(
        self,
        *,
        provider: MeteorologyProvider | None = None,
        timestamps: Iterable[datetime] | None = None,
        include_cache_directory: bool = False,
        include_acquisition_mode: bool = True,
    ) -> dict[str, object]:
        """Return a canonical secret-free request suitable for manifests."""
        selected_provider = self.provider if provider is None else provider
        selected_timestamps = (
            self.measurement_timestamps
            if timestamps is None
            else _ordered_unique_utc(timestamps, "timestamps")
        )
        payload: dict[str, object] = {
            "contract_version": self.contract_version,
            "site_id": self.site_id,
            "latitude_deg_north": self.latitude_deg_north,
            "longitude_deg_east": self.longitude_deg_east,
            "station_altitude_m": self.station_altitude_m,
            "measurement_timestamps_utc": [
                value.isoformat() for value in selected_timestamps
            ],
            "provider": selected_provider.value,
            "radiosonde_station_id": self.radiosonde_station_id,
            "radiosonde_nominal_times_utc": [
                value.isoformat() for value in self.radiosonde_nominal_times
            ],
            "era5_variables": list(self.era5_variables),
            "era5_model_levels": list(self.era5_model_levels),
            "era5_grid_degrees": self.era5_grid_degrees,
            "era5_grid_points": [list(point) for point in self.era5_grid_points],
            "era5_area_north_west_south_east": list(
                self.era5_area_north_west_south_east
            ),
            "era5_hours_utc": [value.isoformat() for value in self.era5_hours],
            "allow_era5t": self.allow_era5t,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }
        if include_acquisition_mode:
            payload["acquisition_mode"] = self.mode.value
        if include_cache_directory:
            payload["cache_directory"] = self.cache_directory.as_posix()
        return payload

    def artifact_request_payload(
        self,
        *,
        provider: MeteorologyProvider,
        timestamps: Iterable[datetime],
    ) -> dict[str, object]:
        """Return the data identity shared by auto, cache_only and prefetch."""
        selected_timestamps = _ordered_unique_utc(timestamps, "timestamps")
        payload: dict[str, object] = {
            "contract_version": self.contract_version,
            "provider": provider.value,
            "site_id": self.site_id,
            "latitude_deg_north": self.latitude_deg_north,
            "longitude_deg_east": self.longitude_deg_east,
            "station_altitude_m": self.station_altitude_m,
            "timestamps_utc": [
                value.isoformat() for value in selected_timestamps
            ],
        }
        if provider is MeteorologyProvider.RADIOSONDE:
            payload["radiosonde_station_id"] = self.radiosonde_station_id
        elif provider is MeteorologyProvider.ERA5:
            payload.update(
                {
                    "era5_variables": list(self.era5_variables),
                    "era5_model_levels": list(self.era5_model_levels),
                    "era5_grid_degrees": self.era5_grid_degrees,
                    "era5_grid_points": [
                        list(point) for point in self.era5_grid_points
                    ],
                    "era5_area_north_west_south_east": list(
                        self.era5_area_north_west_south_east
                    ),
                }
            )
        else:
            raise ValueError("Artifact identities require one concrete provider.")
        return payload
