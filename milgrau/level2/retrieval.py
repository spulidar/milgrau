"""Signal, gluing and molecular-state preparation for Level 2 retrieval."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import logging
from typing import Any, Mapping, TypeVar

import numpy as np
import xarray as xr

from milgrau.level2.config import (
    get_kfs_mode,
    get_lidar_ratio,
    get_molecular_fit_config,
)
from milgrau.level2.gluing import propagate_glued_error
from milgrau.level2.molecular import (
    calculate_molecular_profile,
    calculate_simulated_molecular_signal,
)
from milgrau.level2.rayleigh_window import rayleigh_window_bins
from milgrau.level2.signal_selection import (
    BlockGluingResult,
    WavelengthBlockInputs,
    glue_signal_blocks,
    prepare_wavelength_blocks,
)

_StageResult = TypeVar("_StageResult")

# Temporary operational policy for legacy SPU photon-counting channels whose
# physical saturation limit is not yet characterized. This is deliberately not
# called a detector saturation limit: it is a conservative dead-time occupancy
# guard used only to decide whether an uncharacterized PC sample may participate
# in Level 2.
PROVISIONAL_PC_MAX_DEADTIME_OCCUPANCY = 0.10


@dataclass(frozen=True, slots=True)
class MolecularModel:
    """Molecular atmosphere and retrieval assumptions for one wavelength."""

    source: str
    backscatter: np.ndarray
    extinction: np.ndarray
    transmission: np.ndarray
    simulated_signal: np.ndarray
    simulated_range_corrected_signal: np.ndarray
    fit_config: dict[str, Any]
    lidar_ratio_assumed_sr: float
    lidar_ratio_std_sr: float
    kfs_mode: str


class RetrievalStageError(RuntimeError):
    """Identify the stable retrieval stage that raised an underlying exception."""

    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = stage
        super().__init__(f"[{stage}] {cause}")


def _run_retrieval_stage(
    stage: str,
    operation: Callable[[], _StageResult],
) -> _StageResult:
    """Run one retrieval stage and attach its stable name to any failure."""
    try:
        return operation()
    except RetrievalStageError:
        raise
    except Exception as exc:
        raise RetrievalStageError(stage, exc) from exc


def build_thermodynamic_profile(
    ds_l1: xr.Dataset,
    altitude_agl_m: np.ndarray,
    config: Mapping[str, Any],
    *,
    target_time: Any,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Interpolate the canonical Level 1 atmosphere to one Level 2 block time."""
    del config
    required = ("Atmospheric_Temperature_K", "Atmospheric_Pressure_hPa")
    missing = [name for name in required if name not in ds_l1]
    if missing:
        raise KeyError(
            "Level 1 product lacks canonical thermodynamic variable(s) "
            f"{missing}; reprocess Level 1 with the current LIPANCORA pipeline."
        )
    if "atmosphere_time" not in ds_l1.coords:
        raise KeyError(
            "Level 1 product lacks atmosphere_time; reprocess Level 1 with the "
            "time-resolved LIPANCORA atmosphere."
        )

    altitude = np.asarray(altitude_agl_m, dtype=np.float64)
    expected_dims = ("atmosphere_time", "altitude")
    for name in required:
        if ds_l1[name].dims != expected_dims:
            raise ValueError(f"{name} must have dimensions {expected_dims}.")

    atmosphere_time = np.asarray(ds_l1["atmosphere_time"].values).astype("datetime64[ns]")
    if atmosphere_time.ndim != 1 or atmosphere_time.size == 0:
        raise ValueError("Level 1 atmosphere_time must be a non-empty one-dimensional coordinate.")
    if np.any(np.diff(atmosphere_time.astype("int64")) <= 0):
        raise ValueError("Level 1 atmosphere_time must be strictly increasing.")

    target = np.datetime64(target_time, "ns")
    if target < atmosphere_time[0] or target > atmosphere_time[-1]:
        raise ValueError(
            f"Level 2 block time {target} lies outside the materialized Level 1 atmosphere "
            f"[{atmosphere_time[0]}, {atmosphere_time[-1]}]."
        )

    source_seconds = atmosphere_time.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    target_seconds = float(target.astype("datetime64[s]").astype(np.int64))
    temperature_source = np.asarray(ds_l1["Atmospheric_Temperature_K"].values, dtype=np.float64)
    pressure_source = np.asarray(ds_l1["Atmospheric_Pressure_hPa"].values, dtype=np.float64)
    if temperature_source.shape != (atmosphere_time.size, altitude.size):
        raise ValueError("Stored Level 1 temperature grid is inconsistent with atmosphere_time/altitude.")
    if pressure_source.shape != temperature_source.shape:
        raise ValueError("Stored Level 1 pressure grid is inconsistent with temperature.")

    temperature_k = np.asarray(
        [
            np.interp(target_seconds, source_seconds, temperature_source[:, index])
            for index in range(altitude.size)
        ],
        dtype=np.float64,
    )
    log_pressure = np.log(pressure_source)
    pressure_hpa = np.exp(
        np.asarray(
            [
                np.interp(target_seconds, source_seconds, log_pressure[:, index])
                for index in range(altitude.size)
            ],
            dtype=np.float64,
        )
    )
    if not np.all(np.isfinite(temperature_k)) or np.any(temperature_k <= 0.0):
        raise ValueError("Interpolated atmospheric temperature is not finite and positive.")
    if not np.all(np.isfinite(pressure_hpa)) or np.any(pressure_hpa <= 0.0):
        raise ValueError("Interpolated atmospheric pressure is not finite and positive.")

    source = str(ds_l1.attrs.get("thermodynamic_profile_source_type", "")).strip()
    if source != "time_resolved":
        raise ValueError(
            "Level 1 thermodynamic_profile_source_type must be 'time_resolved'."
        )
    return pressure_hpa, temperature_k, source


def build_molecular_model(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
    *,
    target_time: Any,
) -> MolecularModel:
    """Build the molecular atmosphere at one target block time."""
    pressure_hpa, temperature_k, source = build_thermodynamic_profile(
        ds_l1, altitude_m, config, target_time=target_time
    )
    backscatter, extinction = calculate_molecular_profile(
        temperature_k, pressure_hpa, wavelength_nm
    )
    simulated_signal, transmission = calculate_simulated_molecular_signal(
        backscatter, extinction, altitude_m
    )
    positive_altitudes = altitude_m[altitude_m > 0.0]
    safe_altitude = np.where(
        altitude_m > 0.0,
        altitude_m,
        positive_altitudes[0] if positive_altitudes.size else 1.0,
    )
    lidar_ratio, lidar_ratio_std = get_lidar_ratio(
        config, wavelength_nm, target_time
    )
    fit_config = get_molecular_fit_config(config)
    fit_config["ref_window_bins"] = rayleigh_window_bins(
        altitude_m, fit_config["ref_window_m"]
    )
    return MolecularModel(
        source=source,
        backscatter=backscatter,
        extinction=extinction,
        transmission=transmission,
        simulated_signal=simulated_signal,
        simulated_range_corrected_signal=simulated_signal * safe_altitude**2,
        fit_config=fit_config,
        lidar_ratio_assumed_sr=lidar_ratio,
        lidar_ratio_std_sr=lidar_ratio_std,
        kfs_mode=get_kfs_mode(config),
    )


def _enforce_pc_saturation_characterization(
    ds_l1: xr.Dataset,
    inputs: WavelengthBlockInputs,
) -> WavelengthBlockInputs:
    """Mark uncharacterized PC input unavailable before any provisional guard."""
    if inputs.photon_channel is None:
        return inputs
    characterized = False
    if "pc_saturation_characterized" in ds_l1:
        try:
            characterized = bool(
                int(
                    ds_l1["pc_saturation_characterized"]
                    .sel(channel=inputs.photon_channel)
                    .item()
                )
                == 1
            )
        except Exception:
            characterized = False
    if characterized:
        return inputs
    return replace(inputs, photon_correction_valid=False)


def _channel_calibration_mapping(
    ds_l1: xr.Dataset,
    config: Mapping[str, Any],
    channel: str,
) -> Mapping[str, Any] | None:
    """Return the traceable station calibration mapping for one Level 1 channel."""
    resolved = config.get("_resolved_station")
    if isinstance(resolved, Mapping):
        channels = resolved.get("channel_calibrations")
        if isinstance(channels, Mapping) and isinstance(channels.get(channel), Mapping):
            return channels[channel]

    catalog = config.get("_station_catalog")
    if not isinstance(catalog, Mapping):
        return None
    calibrations = catalog.get("calibrations")
    if not isinstance(calibrations, Mapping):
        return None

    calibration_id = str(ds_l1.attrs.get("instrument_calibration_id", "")).strip()
    if not calibration_id:
        profile_id = str(ds_l1.attrs.get("station_profile_id", "")).strip()
        profiles = catalog.get("profiles")
        if profile_id and isinstance(profiles, list):
            matches = [
                profile
                for profile in profiles
                if isinstance(profile, Mapping)
                and str(profile.get("id", "")).strip() == profile_id
            ]
            if len(matches) == 1:
                calibration_id = str(matches[0].get("calibration_id", "")).strip()
    if not calibration_id:
        return None

    calibration = calibrations.get(calibration_id)
    if not isinstance(calibration, Mapping):
        return None
    channels = calibration.get("channels")
    if not isinstance(channels, Mapping):
        return None
    values = channels.get(channel)
    return values if isinstance(values, Mapping) else None


def _apply_provisional_pc_deadtime_guard(
    ds_l1: xr.Dataset,
    inputs: WavelengthBlockInputs,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> WavelengthBlockInputs:
    """Allow guarded use of uncharacterized PC when dead-time is traceable.

    This remains an operational QA rule, not a physical detector saturation
    characterization. It maps corrected Level 1 PC rate back through the
    non-paralyzable relation as a temporary observed-rate occupancy proxy.
    """
    channel = inputs.photon_channel
    if channel is None or inputs.photon_block is None:
        return inputs

    if "pc_saturation_characterized" not in ds_l1:
        return inputs
    try:
        characterized = bool(
            int(ds_l1["pc_saturation_characterized"].sel(channel=channel).item())
            == 1
        )
    except Exception:
        return inputs
    if characterized:
        return inputs

    try:
        correction_valid = bool(
            "channel_correction_success" in ds_l1
            and int(ds_l1["channel_correction_success"].sel(channel=channel).item())
            == 1
        )
    except Exception:
        correction_valid = False
    if not correction_valid:
        return inputs

    calibration = _channel_calibration_mapping(ds_l1, config, channel)
    if calibration is None:
        return inputs
    try:
        deadtime_us = float(calibration["deadtime_us"])
    except (KeyError, TypeError, ValueError):
        return inputs
    if not np.isfinite(deadtime_us) or deadtime_us <= 0.0:
        return inputs

    saturation = calibration.get("saturation")
    if not isinstance(saturation, Mapping):
        return inputs
    saturation_status = str(saturation.get("status", "")).strip()
    if saturation_status != "not_characterized":
        return inputs

    corrected_rate = np.asarray(inputs.photon_block, dtype=np.float64)
    positive_rate = np.where(
        np.isfinite(corrected_rate) & (corrected_rate > 0.0),
        corrected_rate,
        0.0,
    )
    observed_rate_proxy = positive_rate / (1.0 + positive_rate * deadtime_us)
    occupancy_proxy = observed_rate_proxy * deadtime_us
    provisional_mask = occupancy_proxy >= PROVISIONAL_PC_MAX_DEADTIME_OCCUPANCY

    if inputs.photon_mask_block is None:
        combined_mask = provisional_mask.astype(np.float64)
    else:
        combined_mask = np.maximum(
            np.asarray(inputs.photon_mask_block, dtype=np.float64),
            provisional_mask.astype(np.float64),
        )

    rate_proxy_limit_mhz = PROVISIONAL_PC_MAX_DEADTIME_OCCUPANCY / deadtime_us
    logger.warning(
        "  -> %d nm provisional PC guard active for %s: max dead-time occupancy %.3f "
        "(nominal observed-rate proxy %.2f MHz); physical saturation remains uncharacterized.",
        inputs.wavelength_nm,
        channel,
        PROVISIONAL_PC_MAX_DEADTIME_OCCUPANCY,
        rate_proxy_limit_mhz,
    )
    return replace(
        inputs,
        photon_correction_valid=True,
        photon_mask_block=combined_mask,
    )


def prepare_wavelength_state(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> tuple[WavelengthBlockInputs, BlockGluingResult, MolecularModel]:
    """Prepare signal, gluing and molecular state before reference selection."""
    inputs = _run_retrieval_stage(
        "selection_and_blocking",
        lambda: _apply_provisional_pc_deadtime_guard(
            ds_l1,
            _enforce_pc_saturation_characterization(
                ds_l1,
                prepare_wavelength_blocks(ds_l1, wavelength_nm, altitude_m, config),
            ),
            config,
            logger,
        ),
    )
    glued = _run_retrieval_stage(
        "gluing",
        lambda: glue_signal_blocks(inputs, altitude_m, logger),
    )
    if np.asarray(inputs.block_time).size == 0:
        raise ValueError("No Level 2 temporal block is available for molecular-state preparation.")
    molecular_model = _run_retrieval_stage(
        "molecular_model",
        lambda: build_molecular_model(
            ds_l1,
            wavelength_nm,
            altitude_m,
            config,
            target_time=np.asarray(inputs.block_time)[0],
        ),
    )
    return inputs, glued, molecular_model


__all__ = [
    "BlockGluingResult",
    "MolecularModel",
    "PROVISIONAL_PC_MAX_DEADTIME_OCCUPANCY",
    "RetrievalStageError",
    "WavelengthBlockInputs",
    "build_molecular_model",
    "build_thermodynamic_profile",
    "glue_signal_blocks",
    "prepare_wavelength_blocks",
    "prepare_wavelength_state",
    "propagate_glued_error",
]
