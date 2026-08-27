"""Tests for MILGRAU NetCDF contract validators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from milgrau.io.contracts import validate_level0_contract, validate_level1_contract, validate_level2_contract


def _minimal_level0() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "Raw_Data_Start_Time": (("time", "nb_of_time_scales"), np.array([[0], [60]], dtype=np.int32)),
            "Raw_Data_Stop_Time": (("time", "nb_of_time_scales"), np.array([[60], [120]], dtype=np.int32)),
            "Raw_Data_Range_Resolution": (("channels",), np.array([7.5, 7.5])),
            "Laser_Pointing_Angle": (("scan_angles",), np.array([0.0])),
            "Laser_Pointing_Angle_of_Profiles": (("time", "nb_of_time_scales"), np.zeros((2, 1), dtype=np.int32)),
            "Laser_Shots": (("time", "channels"), np.array([[1200, 2400], [1300, 2600]], dtype=np.int32)),
            "Molecular_Calc": ((), np.array(0, dtype=np.int32)),
            "id_timescale": (("channels",), np.zeros(2, dtype=np.int32)),
            "channel_string": (("channels",), np.array(["532.AN", "532.PC"], dtype=object)),
            "DAQ_Range": (("channels",), np.array([500.0, np.nan])),
            "Raw_Lidar_Data": (("time", "channels", "points"), np.ones((2, 2, 4))),
        }
    )


def test_validate_level0_contract_accepts_scc_acquisition_metadata() -> None:
    validate_level0_contract(_minimal_level0())


def test_validate_level0_contract_requires_daq_range_for_analog() -> None:
    with pytest.raises(KeyError, match="DAQ_Range"):
        validate_level0_contract(_minimal_level0().drop_vars("DAQ_Range"))


def test_validate_level0_contract_rejects_bad_laser_shots() -> None:
    ds = _minimal_level0()
    ds["Laser_Shots"][0, 0] = 0
    with pytest.raises(ValueError, match="Laser_Shots"):
        validate_level0_contract(ds)


def test_validate_level0_contract_requires_background_scc_metadata() -> None:
    ds = _minimal_level0()
    ds["Background_Profile"] = (("time_bck", "channels", "points"), np.ones((1, 2, 4)))
    with pytest.raises(KeyError, match="Raw_Bck"):
        validate_level0_contract(ds)


def test_validate_level1_contract_rejects_missing_rcs_error() -> None:
    time = pd.date_range("2024-01-01", periods=2)
    shape = (2, 1, 4)
    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), np.ones(shape)),
            "corrected_signal_error": (("time", "channel", "altitude"), np.ones(shape)),
            "range_corrected_signal": (("time", "channel", "altitude"), np.ones(shape)),
        },
        coords={"time": time, "channel": ["532.AN"], "altitude": np.arange(4.0)},
    )
    with pytest.raises(KeyError):
        validate_level1_contract(ds)


def test_validate_level1_contract_accepts_required_signal_tensors() -> None:
    time = pd.date_range("2024-01-01", periods=2)
    shape = (2, 1, 4)
    values = {name: (("time", "channel", "altitude"), np.ones(shape)) for name in (
        "corrected_signal", "corrected_signal_error", "range_corrected_signal", "range_corrected_signal_error"
    )}
    ds = xr.Dataset(values, coords={"time": time, "channel": ["532.AN"], "altitude": np.arange(4.0)})
    validate_level1_contract(ds)


def test_validate_level1_contract_accepts_noncanonical_dim_order() -> None:
    time = pd.date_range("2024-01-01", periods=2)
    shape = (1, 4, 2)
    values = {name: (("channel", "altitude", "time"), np.ones(shape)) for name in (
        "corrected_signal", "corrected_signal_error", "range_corrected_signal", "range_corrected_signal_error"
    )}
    ds = xr.Dataset(values, coords={"time": time, "channel": ["532.AN"], "altitude": np.arange(4.0)})
    validate_level1_contract(ds)


def _level2(glued_dims: tuple[str, ...]) -> xr.Dataset:
    sizes = {"time": 2, "wavelength": 1, "altitude": 4}
    glued_shape = tuple(sizes[dim] for dim in glued_dims)
    time_state = np.ones((2, 1), dtype=np.int8)
    return xr.Dataset(
        data_vars={
            "molecular_backscatter": (("wavelength", "altitude"), np.ones((1, 4))),
            "molecular_extinction": (("wavelength", "altitude"), np.ones((1, 4))),
            "glued_range_corrected_signal": (glued_dims, np.ones(glued_shape)),
            "aerosol_backscatter_mean": (("wavelength", "altitude"), np.ones((1, 4))),
            "aerosol_extinction_mean": (("wavelength", "altitude"), np.ones((1, 4))),
            "gluing_attempted_flag": (("time", "wavelength"), time_state),
            "gluing_success_flag": (("time", "wavelength"), time_state),
            "single_channel_fallback_flag": (("time", "wavelength"), np.zeros_like(time_state)),
            "signal_source_flag": (("time", "wavelength"), time_state),
            "retrieval_input_valid_flag": (("time", "wavelength"), time_state),
            "retrieval_input_invalid_reason": (("time", "wavelength"), np.zeros_like(time_state)),
            "retrieval_success_flag": (("block_time", "wavelength"), np.ones((1, 1), dtype=np.int8)),
            "retrieval_success_fraction": (("wavelength",), np.ones(1)),
            "requested_wavelengths": (("requested_wavelength",), np.array([532], dtype=np.int32)),
            "processed_wavelengths": (("processed_wavelength",), np.array([532], dtype=np.int32)),
            "failed_wavelengths": (("failed_wavelength",), np.array([], dtype=np.int32)),
            "failed_wavelength_stage": (("failed_wavelength",), np.array([], dtype=np.int8)),
            "failed_wavelength_code": (("failed_wavelength",), np.array([], dtype=np.int16)),
            "failed_wavelength_message": (("failed_wavelength",), np.array([], dtype=str)),
            "failed_wavelength_cause": (("failed_wavelength",), np.array([], dtype=str)),
        },
        coords={
            "time": pd.date_range("2024-01-01", periods=2),
            "block_time": pd.date_range("2024-01-01", periods=1),
            "wavelength": [532],
            "altitude": np.arange(4.0),
        },
        attrs={"product_completeness": "complete", "product_status": "success"},
    )


def test_validate_level2_contract_rejects_wrong_glued_signal_dims() -> None:
    with pytest.raises(ValueError):
        validate_level2_contract(_level2(("wavelength", "time", "altitude")))


def test_validate_level2_contract_accepts_minimal_optical_dataset() -> None:
    validate_level2_contract(_level2(("time", "wavelength", "altitude")))
