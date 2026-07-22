"""Tests for MILGRAU NetCDF contract validators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from milgrau.io.contracts import validate_level0_contract, validate_level1_contract, validate_level2_contract


def test_validate_level0_contract_accepts_minimal_dataset() -> None:
    """A minimal Level 0 dataset with the canonical raw tensor should validate."""
    ds = xr.Dataset(
        data_vars={
            "Raw_Data_Start_Time": (("time", "nb_of_time_scales"), np.array([[0], [60]], dtype=np.int32)),
            "Raw_Data_Stop_Time": (("time", "nb_of_time_scales"), np.array([[60], [120]], dtype=np.int32)),
            "Raw_Data_Range_Resolution": (("channels",), np.array([7.5, 7.5])),
            "Laser_Pointing_Angle": (("scan_angles",), np.array([0.0])),
            "Laser_Pointing_Angle_of_Profiles": (("time", "nb_of_time_scales"), np.zeros((2, 1), dtype=np.int32)),
            "Laser_Shots": (("time", "channels"), np.full((2, 2), 1200, dtype=np.int32)),
            "Molecular_Calc": ((), np.array(0, dtype=np.int32)),
            "id_timescale": (("channels",), np.zeros(2, dtype=np.int32)),
            "channel_string": (("channels",), np.array(["532.AN", "532.PC"], dtype=object)),
            "Raw_Lidar_Data": (("time", "channels", "points"), np.ones((2, 2, 4))),
        }
    )

    validate_level0_contract(ds)


def test_validate_level1_contract_rejects_missing_rcs_error() -> None:
    """A Level 1 file missing propagated RCS uncertainty should fail early."""
    time = pd.date_range("2024-01-01", periods=2)
    channel = np.array(["532.AN"], dtype=object)
    altitude = np.arange(4.0)
    shape = (2, 1, 4)
    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), np.ones(shape)),
            "corrected_signal_error": (("time", "channel", "altitude"), np.ones(shape)),
            "range_corrected_signal": (("time", "channel", "altitude"), np.ones(shape)),
        },
        coords={"time": time, "channel": channel, "altitude": altitude},
    )

    with pytest.raises(KeyError):
        validate_level1_contract(ds)


def test_validate_level1_contract_accepts_required_signal_tensors() -> None:
    """The Level 1 contract should accept all four core signal tensors."""
    time = pd.date_range("2024-01-01", periods=2)
    channel = np.array(["532.AN"], dtype=object)
    altitude = np.arange(4.0)
    shape = (2, 1, 4)
    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), np.ones(shape)),
            "corrected_signal_error": (("time", "channel", "altitude"), np.ones(shape)),
            "range_corrected_signal": (("time", "channel", "altitude"), np.ones(shape)),
            "range_corrected_signal_error": (("time", "channel", "altitude"), np.ones(shape)),
        },
        coords={"time": time, "channel": channel, "altitude": altitude},
    )

    validate_level1_contract(ds)


def test_validate_level1_contract_accepts_noncanonical_dim_order_when_named_dims_match() -> None:
    """Level 1 tensors may be transposed as long as they keep the required named dimensions."""
    time = pd.date_range("2024-01-01", periods=2)
    channel = np.array(["532.AN"], dtype=object)
    altitude = np.arange(4.0)
    shape = (1, 4, 2)
    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("channel", "altitude", "time"), np.ones(shape)),
            "corrected_signal_error": (("channel", "altitude", "time"), np.ones(shape)),
            "range_corrected_signal": (("channel", "altitude", "time"), np.ones(shape)),
            "range_corrected_signal_error": (("channel", "altitude", "time"), np.ones(shape)),
        },
        coords={"time": time, "channel": channel, "altitude": altitude},
    )

    validate_level1_contract(ds)


def test_validate_level2_contract_rejects_wrong_glued_signal_dims() -> None:
    """Level 2 glued signal should use the canonical exact dimension order."""
    time = pd.date_range("2024-01-01", periods=2)
    wavelength = np.array([532])
    altitude = np.arange(4.0)
    block_time = pd.date_range("2024-01-01", periods=1)
    time_state = np.ones((2, 1), dtype=np.int8)
    ds = xr.Dataset(
        data_vars={
            "molecular_backscatter": (("wavelength", "altitude"), np.ones((1, 4))),
            "molecular_extinction": (("wavelength", "altitude"), np.ones((1, 4))),
            "glued_range_corrected_signal": (("wavelength", "time", "altitude"), np.ones((1, 2, 4))),
            "aerosol_backscatter_mean": (("wavelength", "altitude"), np.ones((1, 4))),
            "aerosol_extinction_mean": (("wavelength", "altitude"), np.ones((1, 4))),
            "gluing_attempted_flag": (("time", "wavelength"), time_state),
            "gluing_success_flag": (("time", "wavelength"), time_state),
            "single_channel_fallback_flag": (("time", "wavelength"), np.zeros_like(time_state)),
            "signal_source_flag": (("time", "wavelength"), time_state),
            "retrieval_input_valid_flag": (("time", "wavelength"), time_state),
            "retrieval_input_invalid_reason": (("time", "wavelength"), np.zeros_like(time_state)),
            "retrieval_success_flag": (("block_time", "wavelength"), np.ones((1, 1), dtype=np.int8)),
        },
        coords={"time": time, "block_time": block_time, "wavelength": wavelength, "altitude": altitude},
    )

    with pytest.raises(ValueError):
        validate_level2_contract(ds)


def test_validate_level2_contract_accepts_minimal_optical_dataset() -> None:
    """A minimal Level 2 optical product should validate."""
    time = pd.date_range("2024-01-01", periods=2)
    wavelength = np.array([532])
    altitude = np.arange(4.0)
    block_time = pd.date_range("2024-01-01", periods=1)
    time_state = np.ones((2, 1), dtype=np.int8)
    ds = xr.Dataset(
        data_vars={
            "molecular_backscatter": (("wavelength", "altitude"), np.ones((1, 4))),
            "molecular_extinction": (("wavelength", "altitude"), np.ones((1, 4))),
            "glued_range_corrected_signal": (("time", "wavelength", "altitude"), np.ones((2, 1, 4))),
            "aerosol_backscatter_mean": (("wavelength", "altitude"), np.ones((1, 4))),
            "aerosol_extinction_mean": (("wavelength", "altitude"), np.ones((1, 4))),
            "gluing_attempted_flag": (("time", "wavelength"), time_state),
            "gluing_success_flag": (("time", "wavelength"), time_state),
            "single_channel_fallback_flag": (("time", "wavelength"), np.zeros_like(time_state)),
            "signal_source_flag": (("time", "wavelength"), time_state),
            "retrieval_input_valid_flag": (("time", "wavelength"), time_state),
            "retrieval_input_invalid_reason": (("time", "wavelength"), np.zeros_like(time_state)),
            "retrieval_success_flag": (("block_time", "wavelength"), np.ones((1, 1), dtype=np.int8)),
        },
        coords={"time": time, "block_time": block_time, "wavelength": wavelength, "altitude": altitude},
    )

    validate_level2_contract(ds)
