"""Regression tests for provisional PC guarding and post-gluing fallback."""

from __future__ import annotations

from dataclasses import replace
import logging

import numpy as np
import xarray as xr

from milgrau.level2.contracts import RetrievalInputInvalidReason, SignalSource
from milgrau.level2.retrieval import (
    BlockGluingResult,
    WavelengthBlockInputs,
    _apply_provisional_pc_deadtime_guard,
    _apply_single_channel_fallback_after_input_qa,
    _enforce_pc_saturation_characterization,
)


def _logger() -> logging.Logger:
    logger = logging.getLogger("test.level2.pc_guarded_fallback")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def _inputs() -> WavelengthBlockInputs:
    altitude = np.array([0.0, 100.0, 200.0, 300.0])
    analog = np.array([[5.0, 4.0, 3.0, 2.0]])
    photon = np.array([[10.0, 20.0, 40.0, 20.0]])
    return WavelengthBlockInputs(
        wavelength_nm=532,
        analog_channel="532.AN",
        photon_channel="532.PC",
        n_time=1,
        n_altitude=altitude.size,
        block_time=np.array([np.datetime64("2024-01-01")]),
        block_groups=[np.array([0])],
        gluing_config={
            "allow_single_channel_fallback": True,
            "single_channel_priority": "photon_counting",
        },
        molecular_fit_config={
            "ref_alt_min_m": 100.0,
            "ref_alt_max_m": 300.0,
            "ref_window_bins": 3,
            "min_valid_fraction": 1.0,
        },
        analog_block=analog,
        analog_error_block=np.full_like(analog, 0.1),
        analog_correction_valid=True,
        photon_block=photon,
        photon_error_block=np.full_like(photon, 0.2),
        photon_mask_block=np.zeros_like(photon),
        photon_correction_valid=True,
    )


def _uncharacterized_level1() -> xr.Dataset:
    return xr.Dataset(
        {
            "channel_correction_success": (
                ("channel",),
                np.array([1, 1], dtype=np.int8),
            ),
            "pc_saturation_characterized": (
                ("channel",),
                np.array([0, 0], dtype=np.int8),
            ),
        },
        coords={"channel": ["532.PC", "532.AN"]},
        attrs={"instrument_calibration_id": "test-calibration"},
    )


def _station_config() -> dict:
    return {
        "_station_catalog": {
            "calibrations": {
                "test-calibration": {
                    "channels": {
                        "532.PC": {
                            "detector_mode": "photon_counting",
                            "deadtime_us": 0.0035,
                            "bin_shift_bins": 0,
                            "background_offset": 0.0,
                            "saturation": {"status": "not_characterized"},
                        },
                        "532.AN": {
                            "detector_mode": "analog",
                            "deadtime_us": 0.0,
                            "bin_shift_bins": 0,
                            "background_offset": 0.0,
                        },
                    }
                }
            }
        }
    }


def test_provisional_guard_keeps_physical_saturation_uncharacterized_but_allows_guarded_pc() -> None:
    inputs = _inputs()
    strict = _enforce_pc_saturation_characterization(_uncharacterized_level1(), inputs)

    assert strict.photon_correction_valid is False

    guarded = _apply_provisional_pc_deadtime_guard(
        _uncharacterized_level1(),
        strict,
        _station_config(),
        _logger(),
    )

    assert guarded.photon_correction_valid is True
    # With tau=0.0035 us, corrected 40 MHz maps to an observed-rate proxy
    # above the 10% occupancy guard; 10 and 20 MHz remain below it.
    np.testing.assert_array_equal(
        guarded.photon_mask_block,
        np.array([[0.0, 0.0, 1.0, 0.0]]),
    )
    assert int(_uncharacterized_level1()["pc_saturation_characterized"].sel(channel="532.PC")) == 0


def test_invalid_successful_gluing_can_fall_back_to_valid_analog_channel() -> None:
    inputs = _inputs()
    inputs = replace(inputs, photon_correction_valid=False)
    n_altitude = inputs.n_altitude
    result = BlockGluingResult(
        source="block_mean_corrected_signal_analog_photon_glued",
        corrected_signal=np.array([[5.0, 4.0, 3.0, 2.0]]),
        corrected_signal_error=np.full((1, n_altitude), 0.1),
        range_corrected_signal=np.zeros((1, n_altitude)),
        range_corrected_signal_error=np.zeros((1, n_altitude)),
        merge_source_flag=np.ones((1, n_altitude), dtype=np.int8),
        attempted_flag=np.array([1], dtype=np.int8),
        success_flag=np.array([1], dtype=np.int8),
        single_channel_fallback_flag=np.array([0], dtype=np.int8),
        signal_source_flag=np.array([SignalSource.GLUED], dtype=np.int8),
        retrieval_input_valid_flag=np.array([0], dtype=np.int8),
        retrieval_input_invalid_reason=np.array(
            [RetrievalInputInvalidReason.LEVEL1_CORRECTION_FAILED_OR_UNCONFIRMED],
            dtype=np.int8,
        ),
        retrieval_input_snr_median=np.array([np.nan]),
        split_altitude_m=np.array([200.0]),
        start_altitude_m=np.array([100.0]),
        stop_altitude_m=np.array([300.0]),
        slope=np.array([2.0]),
        intercept=np.array([0.0]),
        correlation=np.array([0.99]),
        relative_rmse=np.array([0.01]),
        relative_bias=np.array([0.0]),
    )

    updated = _apply_single_channel_fallback_after_input_qa(
        inputs,
        result,
        np.array([0.0, 100.0, 200.0, 300.0]),
        _logger(),
    )

    assert int(updated.success_flag[0]) == 0
    assert int(updated.single_channel_fallback_flag[0]) == 1
    assert int(updated.signal_source_flag[0]) == int(SignalSource.ANALOG)
    assert int(updated.retrieval_input_valid_flag[0]) == 1
    assert int(updated.retrieval_input_invalid_reason[0]) == int(RetrievalInputInvalidReason.VALID)
    np.testing.assert_array_equal(updated.merge_source_flag[0], np.full(n_altitude, 2, dtype=np.int8))
    assert updated.source.endswith("532.AN")
