"""Canonical Level 2 signal selection and block gluing.

This module owns the productive decision that turns block-averaged Level 1
channels into one retrieval input per block.  Numerical gluing remains in
``milgrau.level2.gluing``; retrieval-input QA is applied directly through the
Rayleigh-search supported-domain evaluator.  No import-time patching is used.
"""

from __future__ import annotations

from dataclasses import replace
import logging

import numpy as np

from milgrau.level2._retrieval_impl import (
    BlockGluingResult,
    WavelengthBlockInputs,
    _single_channel_candidates,
    _validate_block_signal_state,
    to_rcs,
)
from milgrau.level2.contracts import RetrievalInputInvalidReason, SignalSource
from milgrau.level2.gluing import (
    merge_source_flags,
    propagate_glued_error,
    slide_glue_signals,
)
from milgrau.level2.retrieval_input_qa import evaluate_retrieval_input_supported_domain


def _result_source_label(signal_source: np.ndarray, inputs: WavelengthBlockInputs) -> str:
    """Return the public source label after block-wise source selection."""
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
    """Recover blocks rejected after numerical gluing with a valid single channel."""
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
        ) in _single_channel_candidates(inputs, int(block_index)):
            valid, reason, block_snr = evaluate_retrieval_input_supported_domain(
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
            rcs[block_index, :], rcs_error[block_index, :] = to_rcs(
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
    _validate_block_signal_state(inputs, updated)

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
    """Select one scientifically usable block signal after AN/PC gluing QA."""
    n_block = len(inputs.block_groups)
    n_altitude = inputs.n_altitude
    corrected = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    corrected_error = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    rcs = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    rcs_error = np.full((n_block, n_altitude), np.nan, dtype=np.float64)
    merge_source = np.full((n_block, n_altitude), 3, dtype=np.int8)
    attempted = np.zeros(n_block, dtype=np.int8)
    success = np.zeros(n_block, dtype=np.int8)
    single_channel_fallback = np.zeros(n_block, dtype=np.int8)
    signal_source = np.full(n_block, SignalSource.INVALID, dtype=np.int8)
    retrieval_input_valid = np.zeros(n_block, dtype=np.int8)
    invalid_reason = np.full(
        n_block,
        RetrievalInputInvalidReason.NO_VALID_CHANNEL,
        dtype=np.int8,
    )
    snr_median = np.full(n_block, np.nan, dtype=np.float64)
    split = np.full(n_block, np.nan, dtype=np.float64)
    start = np.full(n_block, np.nan, dtype=np.float64)
    stop = np.full(n_block, np.nan, dtype=np.float64)
    slope = np.full(n_block, np.nan, dtype=np.float64)
    intercept = np.full(n_block, np.nan, dtype=np.float64)
    correlation = np.full(n_block, np.nan, dtype=np.float64)
    relative_rmse = np.full(n_block, np.nan, dtype=np.float64)
    relative_bias = np.full(n_block, np.nan, dtype=np.float64)
    gluing_config = inputs.gluing_config

    if (
        inputs.analog_block is not None
        and inputs.photon_block is not None
        and inputs.analog_error_block is not None
        and inputs.photon_error_block is not None
    ):
        attempted[:] = 1
        for block_index in range(n_block):
            glued_profile, split_point, slope_i, intercept_i, diagnostics = slide_glue_signals(
                analog_sig=inputs.analog_block[block_index, :],
                pc_sig=inputs.photon_block[block_index, :],
                altitude=altitude_m,
                window_size=gluing_config["window_size"],
                min_corr=gluing_config["min_corr"],
                search_min_idx=gluing_config["search_min_idx"],
                search_max_idx=gluing_config["search_max_idx"],
                intercept_threshold=gluing_config["intercept_threshold"],
                gaussian_threshold=gluing_config["gaussian_threshold"],
                minmax_threshold=gluing_config["minmax_threshold"],
                max_relative_rmse=gluing_config["max_relative_rmse"],
                max_relative_bias=gluing_config["max_relative_bias"],
                min_valid_fraction=gluing_config["min_valid_fraction"],
                max_saturation_fraction=gluing_config["max_saturation_fraction"],
                invalid_saturation_fraction=gluing_config["invalid_saturation_fraction"],
                pc_saturation_mask=(
                    inputs.photon_mask_block[block_index, :]
                    if inputs.photon_mask_block is not None
                    else None
                ),
                return_diagnostics=True,
            )
            slope[block_index] = slope_i
            intercept[block_index] = intercept_i
            correlation[block_index] = float(diagnostics.get("best_corr", np.nan))
            relative_rmse[block_index] = float(diagnostics.get("relative_rmse", np.nan))
            relative_bias[block_index] = float(diagnostics.get("relative_bias", np.nan))
            if split_point >= 0:
                min_bin = int(
                    diagnostics.get(
                        "min_bin",
                        max(split_point - gluing_config["window_size"] // 2, 0),
                    )
                )
                max_bin = int(
                    diagnostics.get(
                        "max_bin",
                        min(split_point + gluing_config["window_size"] // 2, n_altitude),
                    )
                )
                corrected[block_index, :] = glued_profile
                corrected_error[block_index, :] = propagate_glued_error(
                    inputs.analog_error_block[block_index, :],
                    inputs.photon_error_block[block_index, :],
                    slope_i,
                    min_bin,
                    max_bin,
                )
                rcs[block_index, :], rcs_error[block_index, :] = to_rcs(
                    corrected[block_index, :],
                    corrected_error[block_index, :],
                    altitude_m,
                )
                merge_source[block_index, :] = merge_source_flags(
                    n_altitude, min_bin, max_bin
                )
                success[block_index] = 1
                signal_source[block_index] = SignalSource.GLUED
                split[block_index] = float(altitude_m[split_point])
                start[block_index] = float(altitude_m[min_bin])
                stop[block_index] = float(altitude_m[max_bin - 1])
                valid, reason, block_snr = evaluate_retrieval_input_supported_domain(
                    corrected[block_index, :],
                    corrected_error[block_index, :],
                    altitude_m,
                    inputs.molecular_fit_config,
                    correction_valid=(
                        inputs.analog_correction_valid and inputs.photon_correction_valid
                    ),
                    saturation_fraction=(
                        np.where(
                            merge_source[block_index, :] == 2,
                            0.0,
                            inputs.photon_mask_block[block_index, :],
                        )
                        if inputs.photon_mask_block is not None
                        else None
                    ),
                    require_saturation_diagnostic=True,
                )
                retrieval_input_valid[block_index] = np.int8(valid)
                invalid_reason[block_index] = np.int8(reason)
                snr_median[block_index] = block_snr
            elif gluing_config["allow_single_channel_fallback"]:
                first_rejection = RetrievalInputInvalidReason.NO_VALID_CHANNEL
                first_snr = np.nan
                for (
                    candidate_source,
                    _channel,
                    candidate_signal,
                    candidate_error,
                    candidate_saturation,
                    correction_valid,
                    require_saturation_diagnostic,
                ) in _single_channel_candidates(inputs, block_index):
                    valid, reason, block_snr = evaluate_retrieval_input_supported_domain(
                        candidate_signal,
                        candidate_error,
                        altitude_m,
                        inputs.molecular_fit_config,
                        correction_valid=correction_valid,
                        saturation_fraction=candidate_saturation,
                        require_saturation_diagnostic=require_saturation_diagnostic,
                    )
                    if first_rejection == RetrievalInputInvalidReason.NO_VALID_CHANNEL:
                        first_rejection = reason
                        first_snr = block_snr
                    if not valid:
                        continue
                    corrected[block_index, :] = candidate_signal
                    corrected_error[block_index, :] = candidate_error
                    rcs[block_index, :], rcs_error[block_index, :] = to_rcs(
                        candidate_signal,
                        candidate_error,
                        altitude_m,
                    )
                    merge_source[block_index, :] = (
                        0 if candidate_source == SignalSource.PHOTON_COUNTING else 2
                    )
                    single_channel_fallback[block_index] = 1
                    signal_source[block_index] = np.int8(candidate_source)
                    retrieval_input_valid[block_index] = 1
                    invalid_reason[block_index] = RetrievalInputInvalidReason.VALID
                    snr_median[block_index] = block_snr
                    break
                else:
                    invalid_reason[block_index] = np.int8(first_rejection)
                    snr_median[block_index] = first_snr
            else:
                invalid_reason[block_index] = (
                    RetrievalInputInvalidReason.SINGLE_CHANNEL_FALLBACK_DISABLED
                )
        logger.info(
            "  -> %d nm block gluing success: %.1f%% (%s + %s); "
            "valid single-channel fallback blocks: %d.",
            inputs.wavelength_nm,
            100.0 * success.sum() / max(n_block, 1),
            inputs.analog_channel,
            inputs.photon_channel,
            int(single_channel_fallback.sum()),
        )
    else:
        for block_index in range(n_block):
            if not gluing_config["allow_single_channel_fallback"]:
                invalid_reason[block_index] = (
                    RetrievalInputInvalidReason.SINGLE_CHANNEL_FALLBACK_DISABLED
                )
                continue
            candidates = _single_channel_candidates(inputs, block_index)
            if not candidates:
                continue
            (
                candidate_source,
                _channel,
                candidate_signal,
                candidate_error,
                candidate_saturation,
                correction_valid,
                require_saturation_diagnostic,
            ) = candidates[0]
            valid, reason, block_snr = evaluate_retrieval_input_supported_domain(
                candidate_signal,
                candidate_error,
                altitude_m,
                inputs.molecular_fit_config,
                correction_valid=correction_valid,
                saturation_fraction=candidate_saturation,
                require_saturation_diagnostic=require_saturation_diagnostic,
            )
            invalid_reason[block_index] = np.int8(reason)
            snr_median[block_index] = block_snr
            if not valid:
                continue
            corrected[block_index, :] = candidate_signal
            corrected_error[block_index, :] = candidate_error
            rcs[block_index, :], rcs_error[block_index, :] = to_rcs(
                candidate_signal,
                candidate_error,
                altitude_m,
            )
            merge_source[block_index, :] = (
                0 if candidate_source == SignalSource.PHOTON_COUNTING else 2
            )
            single_channel_fallback[block_index] = 1
            signal_source[block_index] = np.int8(candidate_source)
            retrieval_input_valid[block_index] = 1
        logger.warning(
            "  -> %d nm single-channel selection: %d/%d valid block(s).",
            inputs.wavelength_nm,
            int(retrieval_input_valid.sum()),
            n_block,
        )

    result = BlockGluingResult(
        source=_result_source_label(signal_source, inputs),
        corrected_signal=corrected,
        corrected_signal_error=corrected_error,
        range_corrected_signal=rcs,
        range_corrected_signal_error=rcs_error,
        merge_source_flag=merge_source,
        attempted_flag=attempted,
        success_flag=success,
        single_channel_fallback_flag=single_channel_fallback,
        signal_source_flag=signal_source,
        retrieval_input_valid_flag=retrieval_input_valid,
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
    _validate_block_signal_state(inputs, result)
    return _apply_single_channel_fallback_after_input_qa(
        inputs, result, altitude_m, logger
    )
