"""Offline smooth radiosonde/ERA5 fusion with exposed quantitative diagnostics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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
from milgrau.meteorology.interpolation import interpolate_profile_to_altitudes
from milgrau.meteorology.thermodynamics import hydrostatic_pressure_profile, virtual_temperature


@dataclass(frozen=True, slots=True)
class BlendDiagnostics:
    overlap_start_m: float
    overlap_end_m: float
    overlap_thickness_m: float
    blend_start_m: float
    blend_end_m: float
    mean_temperature_difference_k: float
    maximum_temperature_difference_k: float
    mean_virtual_temperature_difference_k: float
    maximum_virtual_temperature_difference_k: float
    mean_absolute_pressure_difference_pa: float
    maximum_absolute_pressure_difference_pa: float
    mean_relative_pressure_difference: float
    maximum_relative_pressure_difference: float
    mean_molecular_number_density_difference_m3: float
    maximum_molecular_number_density_difference_m3: float
    maximum_temperature_jump_before_k: float
    maximum_temperature_jump_after_k: float
    maximum_relative_pressure_jump_before: float
    maximum_relative_pressure_jump_after: float
    maximum_relative_number_density_jump_before: float
    maximum_relative_number_density_jump_after: float
    radiosonde_weight: np.ndarray

    def __post_init__(self) -> None:
        weight = np.array(self.radiosonde_weight, dtype=np.float64, copy=True)
        weight.setflags(write=False)
        object.__setattr__(self, "radiosonde_weight", weight)


@dataclass(frozen=True, slots=True)
class BlendResult:
    profile: AtmosphericProfile
    diagnostics: BlendDiagnostics


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _cosine_radiosonde_weight(
    altitude: np.ndarray,
    radiosonde_valid: np.ndarray,
    era5_valid: np.ndarray,
    blend_width_m: float,
) -> np.ndarray:
    weight = np.zeros(altitude.shape, dtype=np.float64)
    weight[radiosonde_valid & ~era5_valid] = 1.0
    for start, stop in _runs(radiosonde_valid & era5_valid):
        weight[start : stop + 1] = 1.0
        run_thickness = float(altitude[stop] - altitude[start])
        left_transition = start > 0 and era5_valid[start - 1]
        right_transition = stop < altitude.size - 1 and era5_valid[stop + 1]
        transition_count = int(left_transition) + int(right_transition)
        width = float(blend_width_m)
        if transition_count == 2:
            width = min(width, max(run_thickness / 2.0, np.finfo(float).eps))
        elif transition_count == 1:
            width = min(width, max(run_thickness, np.finfo(float).eps))
        if left_transition:
            distance = altitude[start : stop + 1] - altitude[start]
            left = distance <= width
            ramp = 0.5 * (1.0 - np.cos(np.pi * distance[left] / width))
            weight[start : stop + 1][left] = np.minimum(
                weight[start : stop + 1][left], ramp
            )
        if right_transition:
            distance = altitude[stop] - altitude[start : stop + 1]
            right = distance <= width
            ramp = 0.5 * (1.0 - np.cos(np.pi * distance[right] / width))
            weight[start : stop + 1][right] = np.minimum(
                weight[start : stop + 1][right], ramp
            )
    return np.clip(weight, 0.0, 1.0)


def _maximum_transition_jump(values: np.ndarray, transition_indices: np.ndarray) -> float:
    if transition_indices.size == 0:
        return 0.0
    return float(np.max(np.abs(values[transition_indices + 1] - values[transition_indices])))


def _maximum_relative_transition_jump(values: np.ndarray, transition_indices: np.ndarray) -> float:
    if transition_indices.size == 0:
        return 0.0
    left = values[transition_indices]
    right = values[transition_indices + 1]
    scale = np.maximum(np.abs(left), np.abs(right))
    return float(np.max(np.abs(right - left) / scale))


def _empty_diagnostics(weight: np.ndarray) -> BlendDiagnostics:
    return BlendDiagnostics(
        overlap_start_m=np.nan,
        overlap_end_m=np.nan,
        overlap_thickness_m=0.0,
        blend_start_m=np.nan,
        blend_end_m=np.nan,
        mean_temperature_difference_k=np.nan,
        maximum_temperature_difference_k=np.nan,
        mean_virtual_temperature_difference_k=np.nan,
        maximum_virtual_temperature_difference_k=np.nan,
        mean_absolute_pressure_difference_pa=np.nan,
        maximum_absolute_pressure_difference_pa=np.nan,
        mean_relative_pressure_difference=np.nan,
        maximum_relative_pressure_difference=np.nan,
        mean_molecular_number_density_difference_m3=np.nan,
        maximum_molecular_number_density_difference_m3=np.nan,
        maximum_temperature_jump_before_k=0.0,
        maximum_temperature_jump_after_k=0.0,
        maximum_relative_pressure_jump_before=0.0,
        maximum_relative_pressure_jump_after=0.0,
        maximum_relative_number_density_jump_before=0.0,
        maximum_relative_number_density_jump_after=0.0,
        radiosonde_weight=weight,
    )


def blend_radiosonde_and_era5(
    radiosonde: AtmosphericProfile | None,
    era5: AtmosphericProfile,
    target_geometric_altitude_m: np.ndarray,
    *,
    blend_width_m: float,
    maximum_radiosonde_gap_m: float | None = None,
    require_overlap: bool = True,
) -> BlendResult:
    """Fuse on one grid with cosine weights and hydrostatically integrated pressure."""
    if not isinstance(era5, AtmosphericProfile):
        raise TypeError("era5 must be AtmosphericProfile.")
    if not np.isfinite(blend_width_m) or blend_width_m <= 0.0:
        raise ValueError("blend_width_m must be finite and greater than zero.")
    target = np.asarray(target_geometric_altitude_m, dtype=np.float64)
    era_grid = interpolate_profile_to_altitudes(era5, target, allow_extrapolation=False)
    era_valid = (
        np.isfinite(era_grid.pressure_pa)
        & np.isfinite(era_grid.temperature_k)
        & np.isfinite(era_grid.specific_humidity_kg_kg)
    )
    if not era_valid.all():
        raise ValueError("ERA5 must cover the complete requested blend grid.")
    if radiosonde is None:
        return BlendResult(profile=era_grid, diagnostics=_empty_diagnostics(np.zeros(target.shape)))
    if not isinstance(radiosonde, AtmosphericProfile):
        raise TypeError("radiosonde must be AtmosphericProfile or None.")
    overlap_floor = max(
        float(target[0]),
        radiosonde.vertical_coverage_m[0],
        era5.vertical_coverage_m[0],
    )
    overlap_ceiling = min(
        float(target[-1]),
        radiosonde.vertical_coverage_m[1],
        era5.vertical_coverage_m[1],
    )
    if require_overlap and overlap_ceiling < overlap_floor:
        raise ValueError("Radiosonde/ERA5 fusion requires overlapping vertical coverage.")
    radio_grid = interpolate_profile_to_altitudes(
        radiosonde,
        target,
        allow_extrapolation=False,
        maximum_gap_m=maximum_radiosonde_gap_m,
    )
    radio_valid = (
        np.isfinite(radio_grid.pressure_pa)
        & np.isfinite(radio_grid.temperature_k)
        & np.isfinite(radio_grid.specific_humidity_kg_kg)
    )
    overlap = radio_valid & era_valid
    if require_overlap and not overlap.any():
        raise ValueError("Radiosonde/ERA5 fusion requires at least one overlapping valid bin.")
    if overlap.sum() < 2 and require_overlap:
        raise ValueError("Radiosonde/ERA5 fusion requires at least two overlapping valid bins.")

    weight = _cosine_radiosonde_weight(target, radio_valid, era_valid, blend_width_m)
    only_radio = radio_valid & ~era_valid
    weight[only_radio] = 1.0
    temperature = weight * np.where(radio_valid, radio_grid.temperature_k, 0.0) + (
        1.0 - weight
    ) * era_grid.temperature_k
    humidity = weight * np.where(
        radio_valid, radio_grid.specific_humidity_kg_kg, 0.0
    ) + (1.0 - weight) * era_grid.specific_humidity_kg_kg
    preliminary_log_pressure = weight * np.where(
        radio_valid, np.log(radio_grid.pressure_pa), 0.0
    ) + (1.0 - weight) * np.log(era_grid.pressure_pa)
    preliminary_pressure = np.exp(preliminary_log_pressure)
    tv = virtual_temperature(temperature, humidity)
    pressure = hydrostatic_pressure_profile(target, tv, float(preliminary_pressure[0]))

    source = np.full(target.shape, int(PrimarySource.ERA5), dtype=np.int8)
    source[weight == 1.0] = int(PrimarySource.RADIOSONDE)
    source[(weight > 0.0) & (weight < 1.0)] = int(PrimarySource.BLENDED)
    operation = np.full(target.shape, int(InterpolationFlag.INTERPOLATED), dtype=np.int8)
    fallback = np.full(target.shape, int(FallbackFlag.NONE), dtype=np.int8)
    humidity_flag = np.full(target.shape, int(HumidityFlag.MEASURED), dtype=np.int8)
    quality = np.full(target.shape, int(QualityFlag.VALID), dtype=np.int8)
    digest = hashlib.sha256(
        (
            f"{radiosonde.raw_snapshot_sha256}:{era5.raw_snapshot_sha256}:"
            f"cosine:{float(blend_width_m):.17g}"
        ).encode()
    ).hexdigest()
    profile = create_atmospheric_profile(
        geometric_altitude_m=target,
        pressure_pa=pressure,
        temperature_k=temperature,
        specific_humidity_kg_kg=humidity,
        primary_source_flag=source,
        interpolation_flag=operation,
        fallback_flag=fallback,
        humidity_flag=humidity_flag,
        radiosonde_weight=weight,
        quality_flag=quality,
        nominal_time=era5.nominal_time,
        observation_time=era5.observation_time,
        latitude_deg_north=era5.latitude_deg_north,
        longitude_deg_east=era5.longitude_deg_east,
        provider="offline radiosonde + ERA5 hybrid",
        station_or_dataset_id=f"{radiosonde.station_or_dataset_id}+{era5.station_or_dataset_id}",
        raw_snapshot_sha256=digest,
        normalizer_version="cosine-hydrostatic-blend-v1",
        vertical_coverage_m=(float(target[0]), float(target[-1])),
        profile_quality=ProfileQuality.QUANTITATIVE,
        quantitative_retrieval_allowed=bool(
            radiosonde.quantitative_retrieval_allowed and era5.quantitative_retrieval_allowed
        ),
        blend_method="cosine weight plus hydrostatic pressure reintegration",
        blend_parameters=(("blend_width_m", f"{float(blend_width_m):.17g}"),),
    )

    overlap_altitude = target[overlap]
    temperature_difference = np.abs(
        radio_grid.temperature_k[overlap] - era_grid.temperature_k[overlap]
    )
    virtual_temperature_difference = np.abs(
        radio_grid.virtual_temperature_k[overlap] - era_grid.virtual_temperature_k[overlap]
    )
    pressure_difference = np.abs(radio_grid.pressure_pa[overlap] - era_grid.pressure_pa[overlap])
    pressure_relative = pressure_difference / era_grid.pressure_pa[overlap]
    number_difference = np.abs(
        radio_grid.molecular_number_density_m3[overlap]
        - era_grid.molecular_number_density_m3[overlap]
    )
    hard_radio = radio_valid
    hard_temperature = np.where(hard_radio, radio_grid.temperature_k, era_grid.temperature_k)
    hard_pressure = np.where(hard_radio, radio_grid.pressure_pa, era_grid.pressure_pa)
    hard_number = np.where(
        hard_radio,
        radio_grid.molecular_number_density_m3,
        era_grid.molecular_number_density_m3,
    )
    transition_indices = np.flatnonzero(np.diff(hard_radio.astype(np.int8)) != 0)
    blended_bins = (weight > 0.0) & (weight < 1.0)
    diagnostics = BlendDiagnostics(
        overlap_start_m=float(overlap_altitude[0]),
        overlap_end_m=float(overlap_altitude[-1]),
        overlap_thickness_m=float(overlap_altitude[-1] - overlap_altitude[0]),
        blend_start_m=float(target[blended_bins][0]) if blended_bins.any() else np.nan,
        blend_end_m=float(target[blended_bins][-1]) if blended_bins.any() else np.nan,
        mean_temperature_difference_k=float(np.mean(temperature_difference)),
        maximum_temperature_difference_k=float(np.max(temperature_difference)),
        mean_virtual_temperature_difference_k=float(np.mean(virtual_temperature_difference)),
        maximum_virtual_temperature_difference_k=float(np.max(virtual_temperature_difference)),
        mean_absolute_pressure_difference_pa=float(np.mean(pressure_difference)),
        maximum_absolute_pressure_difference_pa=float(np.max(pressure_difference)),
        mean_relative_pressure_difference=float(np.mean(pressure_relative)),
        maximum_relative_pressure_difference=float(np.max(pressure_relative)),
        mean_molecular_number_density_difference_m3=float(np.mean(number_difference)),
        maximum_molecular_number_density_difference_m3=float(np.max(number_difference)),
        maximum_temperature_jump_before_k=_maximum_transition_jump(
            hard_temperature, transition_indices
        ),
        maximum_temperature_jump_after_k=_maximum_transition_jump(
            profile.temperature_k, transition_indices
        ),
        maximum_relative_pressure_jump_before=_maximum_relative_transition_jump(
            hard_pressure, transition_indices
        ),
        maximum_relative_pressure_jump_after=_maximum_relative_transition_jump(
            profile.pressure_pa, transition_indices
        ),
        maximum_relative_number_density_jump_before=_maximum_relative_transition_jump(
            hard_number, transition_indices
        ),
        maximum_relative_number_density_jump_after=_maximum_relative_transition_jump(
            profile.molecular_number_density_m3, transition_indices
        ),
        radiosonde_weight=weight,
    )
    return BlendResult(profile=profile, diagnostics=diagnostics)
