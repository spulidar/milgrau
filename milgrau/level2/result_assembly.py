"""Assemble block-level Level 2 retrieval state into the public result contract."""

from __future__ import annotations

import numpy as np

from milgrau.level2.block_average import (
    expand_block_vector_to_time,
    expand_blocks_to_time,
    valid_block_error,
    valid_block_mean,
)
from milgrau.level2.contracts import (
    GluedSignals,
    GluingDiagnostics,
    KfsDiagnostics,
    MolecularProfiles,
    OpticalProducts,
    RayleighDiagnostics,
    SignalSelectionDiagnostics,
    WavelengthRetrievalResult,
)
from milgrau.level2.signal_selection import BlockGluingResult, WavelengthBlockInputs


def assemble_wavelength_result(
    inputs: WavelengthBlockInputs,
    glued: BlockGluingResult,
    molecular: MolecularProfiles,
    optical: OpticalProducts,
    rayleigh: RayleighDiagnostics,
    kfs: KfsDiagnostics,
) -> WavelengthRetrievalResult:
    """Expand block diagnostics to time and assemble the validated public result."""
    valid_block = np.asarray(optical.retrieval_success_flag, dtype=bool)
    time_corrected = expand_blocks_to_time(
        glued.corrected_signal, inputs.block_groups, inputs.n_time
    )
    time_corrected_error = expand_blocks_to_time(
        glued.corrected_signal_error, inputs.block_groups, inputs.n_time
    )
    time_rcs = expand_blocks_to_time(
        glued.range_corrected_signal, inputs.block_groups, inputs.n_time
    )
    time_rcs_error = expand_blocks_to_time(
        glued.range_corrected_signal_error, inputs.block_groups, inputs.n_time
    )
    time_merge_source = np.full(
        (inputs.n_time, inputs.n_altitude), 3, dtype=np.int8
    )
    for block_index, group in enumerate(inputs.block_groups):
        time_merge_source[group, :] = glued.merge_source_flag[block_index, :]

    result = WavelengthRetrievalResult(
        wavelength_nm=inputs.wavelength_nm,
        block_time=inputs.block_time,
        molecular=molecular,
        glued=GluedSignals(
            source=glued.source,
            analog_channel=inputs.analog_channel,
            photon_channel=inputs.photon_channel,
            corrected_signal=time_corrected,
            corrected_signal_error=time_corrected_error,
            corrected_signal_block=glued.corrected_signal,
            corrected_signal_error_block=glued.corrected_signal_error,
            corrected_signal_mean=valid_block_mean(glued.corrected_signal, valid_block),
            corrected_signal_error_mean=valid_block_error(
                glued.corrected_signal_error, valid_block
            ),
            range_corrected_signal=time_rcs,
            range_corrected_signal_error=time_rcs_error,
            range_corrected_signal_block=glued.range_corrected_signal,
            range_corrected_signal_error_block=glued.range_corrected_signal_error,
            range_corrected_signal_mean=valid_block_mean(
                glued.range_corrected_signal, valid_block
            ),
            range_corrected_signal_error_mean=valid_block_error(
                glued.range_corrected_signal_error, valid_block
            ),
            merge_source_flag=time_merge_source,
            merge_source_flag_block=glued.merge_source_flag,
        ),
        optical=optical,
        rayleigh=rayleigh,
        kfs=kfs,
        gluing=GluingDiagnostics(
            attempted_flag=expand_block_vector_to_time(
                glued.attempted_flag,
                inputs.block_groups,
                inputs.n_time,
                dtype=np.int8,
            ),
            success_flag=expand_block_vector_to_time(
                glued.success_flag,
                inputs.block_groups,
                inputs.n_time,
                dtype=np.int8,
            ),
            single_channel_fallback_flag=expand_block_vector_to_time(
                glued.single_channel_fallback_flag,
                inputs.block_groups,
                inputs.n_time,
                dtype=np.int8,
            ),
            split_altitude_m=expand_block_vector_to_time(
                glued.split_altitude_m, inputs.block_groups, inputs.n_time
            ),
            start_altitude_m=expand_block_vector_to_time(
                glued.start_altitude_m, inputs.block_groups, inputs.n_time
            ),
            stop_altitude_m=expand_block_vector_to_time(
                glued.stop_altitude_m, inputs.block_groups, inputs.n_time
            ),
            slope=expand_block_vector_to_time(
                glued.slope, inputs.block_groups, inputs.n_time
            ),
            intercept=expand_block_vector_to_time(
                glued.intercept, inputs.block_groups, inputs.n_time
            ),
            correlation=expand_block_vector_to_time(
                glued.correlation, inputs.block_groups, inputs.n_time
            ),
            relative_rmse=expand_block_vector_to_time(
                glued.relative_rmse, inputs.block_groups, inputs.n_time
            ),
            relative_bias=expand_block_vector_to_time(
                glued.relative_bias, inputs.block_groups, inputs.n_time
            ),
            attempted_flag_block=glued.attempted_flag,
            success_flag_block=glued.success_flag,
            single_channel_fallback_flag_block=glued.single_channel_fallback_flag,
            split_altitude_m_block=glued.split_altitude_m,
            start_altitude_m_block=glued.start_altitude_m,
            stop_altitude_m_block=glued.stop_altitude_m,
            slope_block=glued.slope,
            intercept_block=glued.intercept,
            correlation_block=glued.correlation,
            relative_rmse_block=glued.relative_rmse,
            relative_bias_block=glued.relative_bias,
        ),
        signal_selection=SignalSelectionDiagnostics(
            source_flag=expand_block_vector_to_time(
                glued.signal_source_flag,
                inputs.block_groups,
                inputs.n_time,
                dtype=np.int8,
            ),
            retrieval_input_valid_flag=expand_block_vector_to_time(
                glued.retrieval_input_valid_flag,
                inputs.block_groups,
                inputs.n_time,
                dtype=np.int8,
            ),
            retrieval_input_invalid_reason=expand_block_vector_to_time(
                glued.retrieval_input_invalid_reason,
                inputs.block_groups,
                inputs.n_time,
                dtype=np.int8,
            ),
            retrieval_input_snr_median=expand_block_vector_to_time(
                glued.retrieval_input_snr_median,
                inputs.block_groups,
                inputs.n_time,
            ),
            source_flag_block=glued.signal_source_flag,
            retrieval_input_valid_flag_block=glued.retrieval_input_valid_flag,
            retrieval_input_invalid_reason_block=glued.retrieval_input_invalid_reason,
            retrieval_input_snr_median_block=glued.retrieval_input_snr_median,
        ),
    )
    result.validate(n_time=inputs.n_time, n_altitude=inputs.n_altitude)
    return result
