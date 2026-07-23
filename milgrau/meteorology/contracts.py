"""Validated, immutable SI contract for one atmospheric vertical profile."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr


class PrimarySource(IntEnum):
    """Physical source of the state in one bin."""

    INVALID = 0
    RADIOSONDE = 1
    ERA5 = 2
    BLENDED = 3
    STANDARD_ATMOSPHERE = 4


class InterpolationFlag(IntEnum):
    """Operation that placed the state on its current vertical grid."""

    INVALID = 0
    DIRECT = 1
    INTERPOLATED = 2
    EXTRAPOLATED = 3


class FallbackFlag(IntEnum):
    """Fallback status, kept independent of physical source and interpolation."""

    NONE = 0
    STANDARD_ATMOSPHERE = 1


class HumidityFlag(IntEnum):
    """Meaning of specific humidity in one bin."""

    MEASURED = 1
    DERIVED_FROM_DEWPOINT = 2
    DRY_AIR_ASSUMED = 3
    MISSING = 4


class QualityFlag(IntEnum):
    """Per-bin physical usability."""

    INVALID = 0
    VALID = 1
    FALLBACK_DIAGNOSTIC = 2
    MISSING_HUMIDITY = 3


class ProfileQuality(StrEnum):
    """Profile-level authorization semantics."""

    QUANTITATIVE = "quantitative"
    FALLBACK_DIAGNOSTIC = "fallback_diagnostic"
    INCOMPLETE = "incomplete"


_FLOAT_ARRAY_FIELDS = (
    "geometric_altitude_m",
    "geopotential_m2_s2",
    "pressure_pa",
    "temperature_k",
    "specific_humidity_kg_kg",
    "virtual_temperature_k",
    "air_density_kg_m3",
    "molecular_number_density_m3",
    "dry_air_mass_density_kg_m3",
    "water_vapor_mass_density_kg_m3",
    "dry_air_number_density_m3",
    "water_vapor_number_density_m3",
    "radiosonde_weight",
)
_FLAG_ARRAY_FIELDS = (
    "primary_source_flag",
    "interpolation_flag",
    "fallback_flag",
    "humidity_flag",
    "quality_flag",
)


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TypeError(f"{label} must be a timezone-aware datetime.")
    return value.astimezone(UTC)


def _frozen_array(value: object, dtype: object, label: str) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional array.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class AtmosphericProfile:
    """One vertical atmospheric state on geometric altitude above mean sea level.

    Every numeric field uses SI. Geopotential is retained alongside the canonical
    geometric altitude so ERA5 hydrostatic reconstruction is reproducible.
    """

    geometric_altitude_m: np.ndarray
    geopotential_m2_s2: np.ndarray
    pressure_pa: np.ndarray
    temperature_k: np.ndarray
    specific_humidity_kg_kg: np.ndarray
    virtual_temperature_k: np.ndarray
    air_density_kg_m3: np.ndarray
    molecular_number_density_m3: np.ndarray
    dry_air_mass_density_kg_m3: np.ndarray
    water_vapor_mass_density_kg_m3: np.ndarray
    dry_air_number_density_m3: np.ndarray
    water_vapor_number_density_m3: np.ndarray
    primary_source_flag: np.ndarray
    interpolation_flag: np.ndarray
    fallback_flag: np.ndarray
    humidity_flag: np.ndarray
    radiosonde_weight: np.ndarray
    quality_flag: np.ndarray
    nominal_time: datetime
    observation_time: datetime
    latitude_deg_north: float
    longitude_deg_east: float
    provider: str
    station_or_dataset_id: str
    raw_snapshot_sha256: str
    normalizer_version: str
    thermodynamic_formula_version: str
    vertical_coverage_m: tuple[float, float]
    profile_quality: ProfileQuality = ProfileQuality.QUANTITATIVE
    quantitative_retrieval_allowed: bool = True
    blend_method: str = "none"
    blend_parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in _FLOAT_ARRAY_FIELDS:
            object.__setattr__(self, name, _frozen_array(getattr(self, name), np.float64, name))
        for name in _FLAG_ARRAY_FIELDS:
            object.__setattr__(self, name, _frozen_array(getattr(self, name), np.int8, name))

        size = self.geometric_altitude_m.size
        if size < 2:
            raise ValueError("AtmosphericProfile requires at least two vertical levels.")
        for name in (*_FLOAT_ARRAY_FIELDS, *_FLAG_ARRAY_FIELDS):
            if getattr(self, name).shape != (size,):
                raise ValueError(f"{name} must have shape ({size},).")
        if not np.isfinite(self.geometric_altitude_m).all():
            raise ValueError("geometric_altitude_m must be finite.")
        if not np.all(np.diff(self.geometric_altitude_m) > 0.0):
            raise ValueError("geometric_altitude_m must be strictly increasing.")
        if not np.isfinite(self.geopotential_m2_s2).all():
            raise ValueError("geopotential_m2_s2 must be finite.")

        finite_pressure = np.isfinite(self.pressure_pa)
        finite_temperature = np.isfinite(self.temperature_k)
        if np.any(self.pressure_pa[finite_pressure] <= 0.0):
            raise ValueError("Finite pressure_pa values must be positive.")
        if np.any(self.temperature_k[finite_temperature] <= 0.0):
            raise ValueError("Finite temperature_k values must be positive.")
        paired = finite_pressure & finite_temperature
        if np.any(finite_pressure != finite_temperature):
            raise ValueError("Pressure and temperature must be missing in the same bins.")
        if paired.sum() < 2:
            raise ValueError("AtmosphericProfile requires at least two finite pressure/temperature bins.")
        if paired.sum() >= 2 and not np.all(np.diff(self.pressure_pa[paired]) < 0.0):
            raise ValueError("pressure_pa must be strictly decreasing with altitude.")

        finite_q = np.isfinite(self.specific_humidity_kg_kg)
        if np.any(
            (self.specific_humidity_kg_kg[finite_q] < 0.0)
            | (self.specific_humidity_kg_kg[finite_q] > 0.1)
        ):
            raise ValueError("specific_humidity_kg_kg must be within [0, 0.1].")
        missing_humidity = self.humidity_flag == int(HumidityFlag.MISSING)
        if np.any(finite_q & missing_humidity) or np.any(~finite_q & ~missing_humidity):
            raise ValueError("Humidity flags must distinguish missing values from explicit/assumed humidity.")

        dependent_names = (
            "virtual_temperature_k",
            "air_density_kg_m3",
            "molecular_number_density_m3",
            "dry_air_mass_density_kg_m3",
            "water_vapor_mass_density_kg_m3",
            "dry_air_number_density_m3",
            "water_vapor_number_density_m3",
        )
        thermo_valid = paired & finite_q
        for name in dependent_names:
            values = getattr(self, name)
            if np.any(~np.isfinite(values[thermo_valid])) or np.any(values[thermo_valid] <= 0.0):
                if name in {"water_vapor_mass_density_kg_m3", "water_vapor_number_density_m3"}:
                    if np.any(~np.isfinite(values[thermo_valid])) or np.any(values[thermo_valid] < 0.0):
                        raise ValueError(f"{name} must be finite and non-negative where humidity is known.")
                else:
                    raise ValueError(f"{name} must be finite and positive where humidity is known.")
            if np.any(np.isfinite(values[~thermo_valid])):
                raise ValueError(f"{name} must be missing where the thermodynamic state is incomplete.")

        self._validate_flags(paired, finite_q)
        if np.any((self.radiosonde_weight < 0.0) | (self.radiosonde_weight > 1.0)):
            raise ValueError("radiosonde_weight must be within [0, 1].")

        object.__setattr__(self, "nominal_time", _utc(self.nominal_time, "nominal_time"))
        object.__setattr__(self, "observation_time", _utc(self.observation_time, "observation_time"))
        for name in ("latitude_deg_north", "longitude_deg_east"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if not -90.0 <= self.latitude_deg_north <= 90.0:
            raise ValueError("latitude_deg_north must be within [-90, 90].")
        if not -180.0 <= self.longitude_deg_east <= 180.0:
            raise ValueError("longitude_deg_east must be within [-180, 180].")
        for name in (
            "provider",
            "station_or_dataset_id",
            "normalizer_version",
            "thermodynamic_formula_version",
            "blend_method",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise TypeError(f"{name} must be a non-empty string.")
        digest = str(self.raw_snapshot_sha256).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("raw_snapshot_sha256 must be a lowercase SHA-256 hex digest.")
        object.__setattr__(self, "raw_snapshot_sha256", digest)
        if not isinstance(self.profile_quality, ProfileQuality):
            raise TypeError("profile_quality must be ProfileQuality.")
        if not isinstance(self.quantitative_retrieval_allowed, bool):
            raise TypeError("quantitative_retrieval_allowed must be boolean.")

        coverage = tuple(float(value) for value in self.vertical_coverage_m)
        if len(coverage) != 2 or not np.isfinite(coverage).all() or coverage[1] < coverage[0]:
            raise ValueError("vertical_coverage_m must be a finite ordered pair.")
        if paired.any():
            actual = (
                float(self.geometric_altitude_m[paired][0]),
                float(self.geometric_altitude_m[paired][-1]),
            )
            if not np.allclose(coverage, actual, rtol=0.0, atol=1e-9):
                raise ValueError("vertical_coverage_m must match finite pressure/temperature coverage.")
        object.__setattr__(self, "vertical_coverage_m", coverage)
        parameters = tuple((str(key), str(value)) for key, value in self.blend_parameters)
        object.__setattr__(self, "blend_parameters", parameters)

        contains_fallback = np.any(self.fallback_flag == int(FallbackFlag.STANDARD_ATMOSPHERE))
        if contains_fallback and self.quantitative_retrieval_allowed:
            raise ValueError("Standard-atmosphere fallback cannot authorize quantitative retrieval.")
        incomplete_state = np.any(~paired) or np.any(~finite_q)
        if self.profile_quality is ProfileQuality.QUANTITATIVE and incomplete_state:
            raise ValueError("Quantitative profile quality requires complete pressure, temperature and humidity.")
        if self.quantitative_retrieval_allowed and self.profile_quality is not ProfileQuality.QUANTITATIVE:
            raise ValueError("Quantitative retrieval requires quantitative profile quality.")
        if self.profile_quality is ProfileQuality.FALLBACK_DIAGNOSTIC:
            if self.quantitative_retrieval_allowed or not contains_fallback:
                raise ValueError("Fallback profile quality requires flagged fallback and retrieval disallowed.")

    def _validate_flags(self, paired: np.ndarray, finite_q: np.ndarray) -> None:
        enum_fields = {
            "primary_source_flag": PrimarySource,
            "interpolation_flag": InterpolationFlag,
            "fallback_flag": FallbackFlag,
            "humidity_flag": HumidityFlag,
            "quality_flag": QualityFlag,
        }
        for name, enum_type in enum_fields.items():
            allowed = np.array([int(member) for member in enum_type], dtype=np.int8)
            if not np.isin(getattr(self, name), allowed).all():
                raise ValueError(f"{name} contains an unknown code.")

        invalid = ~paired
        if np.any(invalid & (self.primary_source_flag != int(PrimarySource.INVALID))):
            raise ValueError("Missing pressure/temperature bins require primary source invalid.")
        if np.any(invalid & (self.interpolation_flag != int(InterpolationFlag.INVALID))):
            raise ValueError("Missing pressure/temperature bins require interpolation flag invalid.")
        if np.any(invalid & (self.quality_flag != int(QualityFlag.INVALID))):
            raise ValueError("Missing pressure/temperature bins require quality flag invalid.")
        if np.any(invalid & (self.fallback_flag != int(FallbackFlag.NONE))):
            raise ValueError("Missing pressure/temperature bins cannot carry a fallback source.")
        if np.any(invalid & (self.radiosonde_weight != 0.0)):
            raise ValueError("Missing pressure/temperature bins require radiosonde_weight=0.")
        if np.any(paired & (self.primary_source_flag == int(PrimarySource.INVALID))):
            raise ValueError("Finite bins require a physical primary source.")
        if np.any(paired & (self.interpolation_flag == int(InterpolationFlag.INVALID))):
            raise ValueError("Finite bins require a valid interpolation operation.")
        standard = self.primary_source_flag == int(PrimarySource.STANDARD_ATMOSPHERE)
        fallback = self.fallback_flag == int(FallbackFlag.STANDARD_ATMOSPHERE)
        if np.any(standard != fallback):
            raise ValueError("Standard-atmosphere source and fallback flag must agree.")
        if np.any(standard & (self.quality_flag != int(QualityFlag.FALLBACK_DIAGNOSTIC))):
            raise ValueError("Standard-atmosphere bins require fallback diagnostic quality.")
        if np.any(
            (self.primary_source_flag == int(PrimarySource.RADIOSONDE))
            & ~np.isclose(self.radiosonde_weight, 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("Pure radiosonde bins require radiosonde_weight=1.")
        non_radiosonde = np.isin(
            self.primary_source_flag,
            [int(PrimarySource.ERA5), int(PrimarySource.STANDARD_ATMOSPHERE)],
        )
        if np.any(non_radiosonde & (self.radiosonde_weight != 0.0)):
            raise ValueError("Pure ERA5/standard bins require radiosonde_weight=0.")
        blended = self.primary_source_flag == int(PrimarySource.BLENDED)
        if np.any(blended & ((self.radiosonde_weight <= 0.0) | (self.radiosonde_weight >= 1.0))):
            raise ValueError("Blended bins require radiosonde_weight strictly between zero and one.")
        if np.any(
            paired
            & ~finite_q
            & (self.quality_flag != int(QualityFlag.MISSING_HUMIDITY))
        ):
            raise ValueError("Finite P/T with missing humidity requires missing-humidity quality.")
        nonstandard_complete = paired & finite_q & ~standard
        if np.any(nonstandard_complete & (self.quality_flag != int(QualityFlag.VALID))):
            raise ValueError("Complete non-fallback bins require valid quality.")

    @property
    def height_above_station_m(self) -> np.ndarray:
        """This coordinate needs an explicit station datum; use the method below."""
        raise AttributeError("Use height_above_station(station_altitude_m) with an explicit datum.")

    def height_above_station(self, station_altitude_m: float) -> np.ndarray:
        values = np.array(self.geometric_altitude_m - float(station_altitude_m), copy=True)
        values.setflags(write=False)
        return values

    def to_xarray(self) -> xr.Dataset:
        """Explicit persistence adapter; the physical kernel itself is xarray-free."""
        import xarray as xr

        data_vars = {
            field.name: (("geometric_altitude_m",), getattr(self, field.name))
            for field in fields(self)
            if field.name in {*_FLOAT_ARRAY_FIELDS, *_FLAG_ARRAY_FIELDS}
            and field.name != "geometric_altitude_m"
        }
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords={"geometric_altitude_m": self.geometric_altitude_m},
            attrs={
                "vertical_coordinate": "geometric altitude above mean sea level",
                "nominal_time_utc": self.nominal_time.isoformat(),
                "observation_time_utc": self.observation_time.isoformat(),
                "latitude_deg_north": self.latitude_deg_north,
                "longitude_deg_east": self.longitude_deg_east,
                "provider": self.provider,
                "station_or_dataset_id": self.station_or_dataset_id,
                "raw_snapshot_sha256": self.raw_snapshot_sha256,
                "normalizer_version": self.normalizer_version,
                "thermodynamic_formula_version": self.thermodynamic_formula_version,
                "profile_quality": self.profile_quality.value,
                "quantitative_retrieval_allowed": int(self.quantitative_retrieval_allowed),
                "blend_method": self.blend_method,
                "vertical_coverage_m": list(self.vertical_coverage_m),
            },
        )
        units = {
            "geometric_altitude_m": "m",
            "geopotential_m2_s2": "m2 s-2",
            "pressure_pa": "Pa",
            "temperature_k": "K",
            "specific_humidity_kg_kg": "kg kg-1",
            "virtual_temperature_k": "K",
            "air_density_kg_m3": "kg m-3",
            "molecular_number_density_m3": "m-3",
            "dry_air_mass_density_kg_m3": "kg m-3",
            "water_vapor_mass_density_kg_m3": "kg m-3",
            "dry_air_number_density_m3": "m-3",
            "water_vapor_number_density_m3": "m-3",
            "radiosonde_weight": "1",
        }
        for name, unit in units.items():
            dataset[name].attrs["units"] = unit
        return dataset


def create_atmospheric_profile(
    *,
    geometric_altitude_m: np.ndarray,
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    specific_humidity_kg_kg: np.ndarray,
    primary_source_flag: np.ndarray,
    interpolation_flag: np.ndarray,
    fallback_flag: np.ndarray,
    humidity_flag: np.ndarray,
    radiosonde_weight: np.ndarray,
    quality_flag: np.ndarray,
    nominal_time: datetime,
    observation_time: datetime,
    latitude_deg_north: float,
    longitude_deg_east: float,
    provider: str,
    station_or_dataset_id: str,
    raw_snapshot_sha256: str,
    normalizer_version: str,
    vertical_coverage_m: tuple[float, float],
    geopotential_m2_s2: np.ndarray | None = None,
    profile_quality: ProfileQuality = ProfileQuality.QUANTITATIVE,
    quantitative_retrieval_allowed: bool = True,
    blend_method: str = "none",
    blend_parameters: tuple[tuple[str, str], ...] = (),
) -> AtmosphericProfile:
    """Construct the contract and derive all moist-air density components."""
    from milgrau.meteorology.thermodynamics import (
        THERMODYNAMIC_FORMULA_VERSION,
        geopotential_from_geometric_altitude,
        thermodynamic_state,
    )

    altitude = np.asarray(geometric_altitude_m, dtype=np.float64)
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    humidity = np.asarray(specific_humidity_kg_kg, dtype=np.float64)
    state = thermodynamic_state(pressure, temperature, humidity)
    geopotential = (
        geopotential_from_geometric_altitude(altitude)
        if geopotential_m2_s2 is None
        else np.asarray(geopotential_m2_s2, dtype=np.float64)
    )
    return AtmosphericProfile(
        geometric_altitude_m=altitude,
        geopotential_m2_s2=geopotential,
        pressure_pa=pressure,
        temperature_k=temperature,
        specific_humidity_kg_kg=humidity,
        virtual_temperature_k=state.virtual_temperature_k,
        air_density_kg_m3=state.air_density_kg_m3,
        molecular_number_density_m3=state.molecular_number_density_m3,
        dry_air_mass_density_kg_m3=state.dry_air_mass_density_kg_m3,
        water_vapor_mass_density_kg_m3=state.water_vapor_mass_density_kg_m3,
        dry_air_number_density_m3=state.dry_air_number_density_m3,
        water_vapor_number_density_m3=state.water_vapor_number_density_m3,
        primary_source_flag=primary_source_flag,
        interpolation_flag=interpolation_flag,
        fallback_flag=fallback_flag,
        humidity_flag=humidity_flag,
        radiosonde_weight=radiosonde_weight,
        quality_flag=quality_flag,
        nominal_time=nominal_time,
        observation_time=observation_time,
        latitude_deg_north=latitude_deg_north,
        longitude_deg_east=longitude_deg_east,
        provider=provider,
        station_or_dataset_id=station_or_dataset_id,
        raw_snapshot_sha256=raw_snapshot_sha256,
        normalizer_version=normalizer_version,
        thermodynamic_formula_version=THERMODYNAMIC_FORMULA_VERSION,
        vertical_coverage_m=vertical_coverage_m,
        profile_quality=profile_quality,
        quantitative_retrieval_allowed=quantitative_retrieval_allowed,
        blend_method=blend_method,
        blend_parameters=blend_parameters,
    )
