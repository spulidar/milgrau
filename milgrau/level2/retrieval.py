"""Public Level 2 retrieval API with explicit productive orchestration.

Level 2 consumes the canonical thermodynamic atmosphere materialized by Level 1.
Productive signal selection and backward optical aggregation are wired directly;
package import order does not change scientific behavior.  Shared dataclasses
and a few low-level helpers remain in ``_retrieval_impl`` until the next
decomposition batch.
"""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import Any, Mapping

import numpy as np
import xarray as xr

from milgrau.level2._retrieval_impl import (
    BlockGluingResult,
    MolecularModel,
    RetrievalStageError,
    WavelengthBlockInputs,
    _run_retrieval_stage,
    assemble_wavelength_result,
    evaluate_rayleigh_reference,
    prepare_wavelength_blocks,
)
from milgrau.level2.config import get_kfs_mode, get_lidar_ratio, get_molecular_fit_config
from milgrau.level2.contracts import WavelengthRetrievalResult
from milgrau.level2.gluing import propagate_glued_error
from milgrau.level2.molecular import (
    calculate_molecular_profile,
    calculate_simulated_molecular_signal,
)
from milgrau.level2.optical_retrieval import retrieve_optical_blocks
from milgrau.level2.rayleigh_window import rayleigh_window_bins
from milgrau.level2.signal_selection import (
    _apply_single_channel_fallback_after_input_qa,
    glue_signal_blocks,
)


# Temporary operational policy for legacy SPU photon-counting channels whose
# physical saturation limit is not yet characterized. This is deliberately not
# called a detector saturation limit: it is a conservative dead-time occupancy
# guard used only to decide whether an uncharacterized PC sample may participate
# in Level 2.
PROVISIONAL_PC_MAX_DEADTIME_OCCUPANCY = 0.10


def build_thermodynamic_profile(
    ds_l1: xr.Dataset,
    altitude_agl_m: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Read the complete canonical atmosphere materialized by Level 1."""
    del config
    required = ("Atmospheric_Temperature_K", "Atmospheric_Pressure_hPa")
    missing = [name for name in required if name not in ds_l1]
    if missing:
        raise KeyError(
            "Level 1 product lacks canonical thermodynamic variable(s) "
            f"{missing}; reprocess Level 1 with the current LIPANCORA pipeline."
        )

    altitude = np.asarray(altitude_agl_m, dtype=np.float64)
    temperature_k = np.asarray(
        ds_l1["Atmospheric_Temperature_K"].values, dtype=np.float64
    )
    pressure_hpa = np.asarray(
        ds_l1["Atmospheric_Pressure_hPa"].values, dtype=np.float64
    )
    if ds_l1["Atmospheric_Temperature_K"].dims != ("altitude",):
        raise ValueError(
            "Atmospheric_Temperature_K must have dimensions ('altitude',)."
        )
    if ds_l1["Atmospheric_Pressure_hPa"].dims != ("altitude",):
        raise ValueError(
            "Atmospheric_Pressure_hPa must have dimensions ('altitude',)."
        )
    if temperature_k.shape != altitude.shape or pressure_hpa.shape != altitude.shape:
        raise ValueError(
            "Stored Level 1 atmosphere must match the Level 2 lidar altitude grid exactly."
        )
    if not np.all(np.isfinite(temperature_k)) or np.any(temperature_k <= 0.0):
        raise ValueError(
            "Atmospheric_Temperature_K must be finite and positive on every altitude bin."
        )
    if not np.all(np.isfinite(pressure_hpa)) or np.any(pressure_hpa <= 0.0):
        raise ValueError(
            "Atmospheric_Pressure_hPa must be finite and positive on every altitude bin."
        )

    source = str(ds_l1.attrs.get("thermodynamic_profile_source_type", "")).strip()
    if not source:
        raise ValueError(
            "Level 1 product lacks thermodynamic_profile_source_type provenance."
        )
    return pressure_hpa, temperature_k, source


def build_molecular_model(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
) -> MolecularModel:
    """Build the molecular atmosphere from canonical Level 1 thermodynamics."""
    pressure_hpa, temperature_k, source = build_thermodynamic_profile(
        ds_l1, altitude_m, config
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
        config, wavelength_nm, ds_l1["time"].values[0]
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

    calibration_id = str(
        ds_l1.attrs.get("instrument_calibration_id", "")
    ).strip()
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
                calibration_id = str(
                    matches[0].get("calibration_id", "")
                ).strip()
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
    characterization.  It maps corrected Level 1 PC rate back through the
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
            and int(
                ds_l1["channel_correction_success"].sel(channel=channel).item()
            )
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
    provisional_mask = (
        occupancy_proxy >= PROVISIONAL_PC_MAX_DEADTIME_OCCUPANCY
    )

    if inputs.photon_mask_block is None:
        combined_mask = provisional_mask.astype(np.float64)
    else:
        combined_mask = np.maximum(
            np.asarray(inputs.photon_mask_block, dtype=np.float64),
            provisional_mask.astype(np.float64),
        )

    rate_proxy_limit_mhz = (
        PROVISIONAL_PC_MAX_DEADTIME_OCCUPANCY / deadtime_us
    )
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


def process_wavelength(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> WavelengthRetrievalResult:
    """Process one wavelength through the explicit productive Level 2 path."""
    inputs = _run_retrieval_stage(
        "selection_and_blocking",
        lambda: _apply_provisional_pc_deadtime_guard(
            ds_l1,
            _enforce_pc_saturation_characterization(
                ds_l1,
                prepare_wavelength_blocks(
                    ds_l1, wavelength_nm, altitude_m, config
                ),
            ),
            config,
            logger,
        ),
    )
    glued = _run_retrieval_stage(
        "gluing",
        lambda: glue_signal_blocks(inputs, altitude_m, logger),
    )
    molecular_model = _run_retrieval_stage(
        "molecular_model",
        lambda: build_molecular_model(
            ds_l1, wavelength_nm, altitude_m, config
        ),
    )
    molecular, optical, rayleigh, kfs = _run_retrieval_stage(
        "rayleigh_kfs",
        lambda: retrieve_optical_blocks(
            inputs,
            glued,
            molecular_model,
            altitude_m,
            config,
            logger,
        ),
    )
    return _run_retrieval_stage(
        "result_assembly",
        lambda: assemble_wavelength_result(
            inputs, glued, molecular, optical, rayleigh, kfs
        ),
    )


__all__ = [
    "BlockGluingResult",
    "MolecularModel",
    "PROVISIONAL_PC_MAX_DEADTIME_OCCUPANCY",
    "RetrievalStageError",
    "WavelengthBlockInputs",
    "build_molecular_model",
    "build_thermodynamic_profile",
    "evaluate_rayleigh_reference",
    "glue_signal_blocks",
    "prepare_wavelength_blocks",
    "process_wavelength",
    "propagate_glued_error",
    "retrieve_optical_blocks",
]
