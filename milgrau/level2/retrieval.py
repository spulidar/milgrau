"""Public Level 2 retrieval API with a strict Level 1 atmosphere contract.

The numerical retrieval implementation remains in ``_retrieval_impl`` while
this module owns the Level 1 -> Level 2 boundary. Thermodynamic source
selection, interpolation and fallback are completed by Level 1; Level 2 only
consumes the canonical atmosphere stored in the Level 1 NetCDF.
"""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import Any, Mapping

import numpy as np
import xarray as xr

# Re-export the existing numerical retrieval API while the monolithic module is
# progressively split into smaller scientific components.
from milgrau.level2._retrieval_impl import *  # noqa: F401,F403
import milgrau.level2._retrieval_impl as _impl
from milgrau.level2._retrieval_impl import _run_retrieval_stage
from milgrau.level2.config import get_kfs_mode, get_lidar_ratio, get_molecular_fit_config
from milgrau.level2.molecular import calculate_molecular_profile, calculate_simulated_molecular_signal
from milgrau.level2.rayleigh_window import rayleigh_window_bins


# Temporary operational policy for legacy SPU photon-counting channels whose
# physical saturation limit is not yet characterized. This is deliberately not
# called a detector saturation limit: it is a conservative dead-time occupancy
# guard used only to decide whether an uncharacterized PC sample may participate
# in Level 2. Products remain traceable to the software version and the station
# calibration dead-time value.
PROVISIONAL_PC_MAX_DEADTIME_OCCUPANCY = 0.10


def build_thermodynamic_profile(
    ds_l1: xr.Dataset,
    altitude_agl_m: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Read the complete canonical atmosphere materialized by Level 1.

    No radiosonde/ERA5 selection, interpolation or standard-atmosphere fallback
    is permitted here. Old Level 1 files without ``Atmospheric_*`` variables are
    intentionally rejected and must be reprocessed.
    """
    del config
    required = ("Atmospheric_Temperature_K", "Atmospheric_Pressure_hPa")
    missing = [name for name in required if name not in ds_l1]
    if missing:
        raise KeyError(
            "Level 1 product lacks canonical thermodynamic variable(s) "
            f"{missing}; reprocess Level 1 with the current LIPANCORA pipeline."
        )

    altitude = np.asarray(altitude_agl_m, dtype=np.float64)
    temperature_k = np.asarray(ds_l1["Atmospheric_Temperature_K"].values, dtype=np.float64)
    pressure_hpa = np.asarray(ds_l1["Atmospheric_Pressure_hPa"].values, dtype=np.float64)
    if ds_l1["Atmospheric_Temperature_K"].dims != ("altitude",):
        raise ValueError("Atmospheric_Temperature_K must have dimensions ('altitude',).")
    if ds_l1["Atmospheric_Pressure_hPa"].dims != ("altitude",):
        raise ValueError("Atmospheric_Pressure_hPa must have dimensions ('altitude',).")
    if temperature_k.shape != altitude.shape or pressure_hpa.shape != altitude.shape:
        raise ValueError("Stored Level 1 atmosphere must match the Level 2 lidar altitude grid exactly.")
    if not np.all(np.isfinite(temperature_k)) or np.any(temperature_k <= 0.0):
        raise ValueError("Atmospheric_Temperature_K must be finite and positive on every altitude bin.")
    if not np.all(np.isfinite(pressure_hpa)) or np.any(pressure_hpa <= 0.0):
        raise ValueError("Atmospheric_Pressure_hPa must be finite and positive on every altitude bin.")

    source = str(ds_l1.attrs.get("thermodynamic_profile_source_type", "")).strip()
    if not source:
        raise ValueError("Level 1 product lacks thermodynamic_profile_source_type provenance.")
    return pressure_hpa, temperature_k, source


def build_molecular_model(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
) -> MolecularModel:
    """Build the molecular atmosphere from the canonical Level 1 thermodynamics."""
    pressure_hpa, temperature_k, source = build_thermodynamic_profile(ds_l1, altitude_m, config)
    backscatter, extinction = calculate_molecular_profile(temperature_k, pressure_hpa, wavelength_nm)
    simulated_signal, transmission = calculate_simulated_molecular_signal(backscatter, extinction, altitude_m)
    positive_altitudes = altitude_m[altitude_m > 0.0]
    safe_altitude = np.where(
        altitude_m > 0.0,
        altitude_m,
        positive_altitudes[0] if positive_altitudes.size else 1.0,
    )
    lidar_ratio, lidar_ratio_std = get_lidar_ratio(config, wavelength_nm, ds_l1["time"].values[0])
    fit_config = get_molecular_fit_config(config)
    fit_config["ref_window_bins"] = rayleigh_window_bins(altitude_m, fit_config["ref_window_m"])
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
    """Prevent Level 2 from silently accepting uncharacterized PC saturation.

    This function remains strict on its own: an uncharacterized PC source is
    marked scientifically unavailable. ``process_wavelength`` may subsequently
    apply the explicit provisional dead-time occupancy guard below when a
    traceable positive dead-time calibration is available. Keeping these two
    steps separate prevents an operational guard from being mistaken for a
    physical detector characterization.
    """
    if inputs.photon_channel is None:
        return inputs
    characterized = False
    if "pc_saturation_characterized" in ds_l1:
        try:
            characterized = bool(
                int(ds_l1["pc_saturation_characterized"].sel(channel=inputs.photon_channel).item()) == 1
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
                profile for profile in profiles
                if isinstance(profile, Mapping) and str(profile.get("id", "")).strip() == profile_id
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
    """Allow guarded use of an uncharacterized PC channel when dead-time is traceable.

    The guard is an operational QA rule, not a physical saturation calibration.
    The Level 1 corrected PC block is mapped back through the non-paralyzable
    dead-time relation as a rate proxy,

        observed ~= corrected / (1 + corrected * tau),

    and bins whose inferred occupancy ``observed * tau`` reaches 10% are masked.
    Because Level 1 has already background-subtracted the signal, this proxy is
    intentionally temporary; a dedicated raw-rate characterization must replace
    it once the detector saturation study is completed.
    """
    channel = inputs.photon_channel
    if channel is None or inputs.photon_block is None:
        return inputs

    if "pc_saturation_characterized" not in ds_l1:
        return inputs
    try:
        characterized = bool(
            int(ds_l1["pc_saturation_characterized"].sel(channel=channel).item()) == 1
        )
    except Exception:
        return inputs
    if characterized:
        return inputs

    try:
        correction_valid = bool(
            "channel_correction_success" in ds_l1
            and int(ds_l1["channel_correction_success"].sel(channel=channel).item()) == 1
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
    positive_rate = np.where(np.isfinite(corrected_rate) & (corrected_rate > 0.0), corrected_rate, 0.0)
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


def _result_source_label(signal_source: np.ndarray, inputs: WavelengthBlockInputs) -> str:
    """Return the public source label after post-gluing fallback selection."""
    selected_sources = set(np.asarray(signal_source, dtype=np.int8).astype(int).tolist())
    if len(selected_sources) != 1:
        return "blockwise_selected_corrected_signal"
    selected = SignalSource(next(iter(selected_sources)))
    return {
        SignalSource.INVALID: "invalid_no_retrieval_input",
        SignalSource.GLUED: "block_mean_corrected_signal_analog_photon_glued",
        SignalSource.PHOTON_COUNTING: f"block_mean_corrected_signal_single_channel_{inputs.photon_channel}",
        SignalSource.ANALOG: f"block_mean_corrected_signal_single_channel_{inputs.analog_channel}",
    }[selected]


def _apply_single_channel_fallback_after_input_qa(
    inputs: WavelengthBlockInputs,
    result: BlockGluingResult,
    altitude_m: np.ndarray,
    logger: logging.Logger,
) -> BlockGluingResult:
    """Try configured single-channel candidates for blocks rejected after gluing.

    Numerical AN/PC gluing and scientific retrieval-input validity are different
    questions. A block may have an excellent regression window yet be unusable
    because one detector lacks saturation assurance. Such a block must still be
    allowed to fall back to an independently valid channel.
    """
    if not bool(inputs.gluing_config["allow_single_channel_fallback"]):
        return result

    invalid_blocks = np.where(np.asarray(result.retrieval_input_valid_flag) != 1)[0]
    if invalid_blocks.size == 0:
        return result

    corrected = np.asarray(result.corrected_signal, dtype=np.float64).copy()
    corrected_error = np.asarray(result.corrected_signal_error, dtype=np.float64).copy()
    rcs = np.asarray(result.range_corrected_signal, dtype=np.float64).copy()
    rcs_error = np.asarray(result.range_corrected_signal_error, dtype=np.float64).copy()
    merge_source = np.asarray(result.merge_source_flag, dtype=np.int8).copy()
    success = np.asarray(result.success_flag, dtype=np.int8).copy()
    fallback = np.asarray(result.single_channel_fallback_flag, dtype=np.int8).copy()
    signal_source = np.asarray(result.signal_source_flag, dtype=np.int8).copy()
    retrieval_valid = np.asarray(result.retrieval_input_valid_flag, dtype=np.int8).copy()
    invalid_reason = np.asarray(result.retrieval_input_invalid_reason, dtype=np.int8).copy()
    snr_median = np.asarray(result.retrieval_input_snr_median, dtype=np.float64).copy()
    split = np.asarray(result.split_altitude_m, dtype=np.float64).copy()
    start = np.asarray(result.start_altitude_m, dtype=np.float64).copy()
    stop = np.asarray(result.stop_altitude_m, dtype=np.float64).copy()
    slope = np.asarray(result.slope, dtype=np.float64).copy()
    intercept = np.asarray(result.intercept, dtype=np.float64).copy()
    correlation = np.asarray(result.correlation, dtype=np.float64).copy()
    relative_rmse = np.asarray(result.relative_rmse, dtype=np.float64).copy()
    relative_bias = np.asarray(result.relative_bias, dtype=np.float64).copy()

    selected_count = 0
    for block_index in invalid_blocks:
        first_reason = RetrievalInputInvalidReason.NO_VALID_CHANNEL
        first_snr = np.nan
        for (
            candidate_source,
            _channel,
            candidate_signal,
            candidate_error,
            candidate_saturation,
            correction_valid,
            require_saturation_diagnostic,
        ) in _impl._single_channel_candidates(inputs, int(block_index)):
            valid, reason, block_snr = _impl._evaluate_retrieval_input(
                candidate_signal,
                candidate_error,
                altitude_m,
                inputs.molecular_fit_config,
                correction_valid=correction_valid,
                saturation_fraction=candidate_saturation,
                require_saturation_diagnostic=require_saturation_diagnostic,
            )
            if first_reason == RetrievalInputInvalidReason.NO_VALID_CHANNEL:
                first_reason = reason
                first_snr = block_snr
            if not valid:
                continue

            corrected[block_index, :] = candidate_signal
            corrected_error[block_index, :] = candidate_error
            rcs[block_index, :], rcs_error[block_index, :] = _impl.to_rcs(
                candidate_signal,
                candidate_error,
                altitude_m,
            )
            merge_source[block_index, :] = (
                0 if candidate_source == SignalSource.PHOTON_COUNTING else 2
            )
            success[block_index] = 0
            fallback[block_index] = 1
            signal_source[block_index] = np.int8(candidate_source)
            retrieval_valid[block_index] = 1
            invalid_reason[block_index] = RetrievalInputInvalidReason.VALID
            snr_median[block_index] = block_snr
            split[block_index] = np.nan
            start[block_index] = np.nan
            stop[block_index] = np.nan
            slope[block_index] = np.nan
            intercept[block_index] = np.nan
            correlation[block_index] = np.nan
            relative_rmse[block_index] = np.nan
            relative_bias[block_index] = np.nan
            selected_count += 1
            break
        else:
            invalid_reason[block_index] = np.int8(first_reason)
            snr_median[block_index] = first_snr

    updated = replace(
        result,
        source=_result_source_label(signal_source, inputs),
        corrected_signal=corrected,
        corrected_signal_error=corrected_error,
        range_corrected_signal=rcs,
        range_corrected_signal_error=rcs_error,
        merge_source_flag=merge_source,
        success_flag=success,
        single_channel_fallback_flag=fallback,
        signal_source_flag=signal_source,
        retrieval_input_valid_flag=retrieval_valid,
        retrieval_input_invalid_reason=invalid_reason,
        retrieval_input_snr_median=snr_median,
        split_altitude_m=split,
        start_altitude_m=start,
        stop_altitude_m=stop,
        slope=slope,
        intercept=intercept,
        correlation=correlation,
        relative_rmse=relative_rmse,
        relative_bias=relative_bias,
    )
    _impl._validate_block_signal_state(inputs, updated)

    if selected_count:
        logger.warning(
            "  -> %d nm post-QA single-channel fallback selected for %d/%d rejected block(s).",
            inputs.wavelength_nm,
            selected_count,
            int(invalid_blocks.size),
        )
    remaining = np.where(np.asarray(updated.retrieval_input_valid_flag) != 1)[0]
    if remaining.size:
        counts: dict[str, int] = {}
        for block_index in remaining:
            try:
                name = RetrievalInputInvalidReason(
                    int(updated.retrieval_input_invalid_reason[block_index])
                ).name.lower()
            except ValueError:
                name = f"unknown_{int(updated.retrieval_input_invalid_reason[block_index])}"
            counts[name] = counts.get(name, 0) + 1
        logger.warning(
            "  -> %d nm retrieval-input rejection summary after fallback: %s",
            inputs.wavelength_nm,
            ", ".join(f"{name}={count}" for name, count in sorted(counts.items())),
        )
    return updated


def glue_signal_blocks(
    inputs: WavelengthBlockInputs,
    altitude_m: np.ndarray,
    logger: logging.Logger,
) -> BlockGluingResult:
    """Run numerical gluing, then recover scientifically valid single-channel inputs."""
    result = _impl.glue_signal_blocks(inputs, altitude_m, logger)
    return _apply_single_channel_fallback_after_input_qa(inputs, result, altitude_m, logger)


def process_wavelength(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> WavelengthRetrievalResult:
    """Process one wavelength using the canonical Level 1 atmosphere contract."""
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
    molecular_model = _run_retrieval_stage(
        "molecular_model",
        lambda: build_molecular_model(ds_l1, wavelength_nm, altitude_m, config),
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
        lambda: assemble_wavelength_result(inputs, glued, molecular, optical, rayleigh, kfs),
    )
