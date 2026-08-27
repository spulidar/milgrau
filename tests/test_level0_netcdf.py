"""Tests for Level 0 NetCDF writing and provenance metadata."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from milgrau.io.contracts import validate_level0_contract
from milgrau.level0.netcdf import build_level0_netcdf, validate_lidar_tensors


def _config() -> dict:
    return {
        "physics": {"latitude": -23.5615, "longitude": -46.7383, "vertical_resolution_m": 7.5, "background_start_m": 29000.0, "background_stop_m": 29999.0, "default_surface_temp_c": 25.0, "default_surface_pressure_hpa": 940.0},
        "hardware": {"name_to_id": {"day": {"532.AN": 1593, "532.PC": 716}, "night": {"532.AN": 722, "532.PC": 716}}},
    }


def _group_df(tmp_path: Path, include_dark_current: bool = True) -> pd.DataFrame:
    records = [
        {"filepath": str(tmp_path / "meas_0001"), "meas_type": "measurements", "start_time_utc": pd.Timestamp("2024-01-01T00:00:00Z"), "stop_time": pd.Timestamp("2024-01-01T00:05:00Z"), "original_meas_id": "20231231nt", "association_method": "measurement", "dark_current_association_delta_hours": np.nan},
        {"filepath": str(tmp_path / "meas_0002"), "meas_type": "measurements", "start_time_utc": pd.Timestamp("2024-01-01T00:05:00Z"), "stop_time": pd.Timestamp("2024-01-01T00:10:00Z"), "original_meas_id": "20231231nt", "association_method": "measurement", "dark_current_association_delta_hours": np.nan},
    ]
    if include_dark_current:
        records.append({"filepath": str(tmp_path / "dark_0001"), "meas_type": "dark_current", "start_time_utc": pd.Timestamp("2023-12-31T23:40:00Z"), "stop_time": pd.Timestamp("2023-12-31T23:45:00Z"), "original_meas_id": "20231231pm", "association_method": "nearest_measurement", "dark_current_association_delta_hours": 0.5})
    return pd.DataFrame.from_records(records)


def _lidar_data() -> dict:
    return {
        "channels": ["532.AN", "532.PC"], "shots": 1200,
        "laser_shots": np.array([[1200, 2400], [1300, 2600]], dtype=np.int32),
        "channel_metadata": {"532.AN": {"is_pc": False, "bin_width_m": 7.5, "daq_range_mV": 500.0}, "532.PC": {"is_pc": True, "bin_width_m": 15.0, "daq_range_mV": np.nan}},
        "tensors": {"532.AN": np.ones((2, 4), dtype=np.float64), "532.PC": np.ones((2, 4), dtype=np.float64) * 2.0},
    }


def test_validate_lidar_tensors_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        validate_lidar_tensors({"532.AN": np.ones((2, 4)), "532.PC": np.ones((3, 4))}, ["532.AN", "532.PC"])


def test_build_level0_netcdf_writes_scc_acquisition_metadata(tmp_path: Path) -> None:
    output_path = tmp_path / "level0_scc.nc"
    build_level0_netcdf(str(output_path), "20240101sant", "nt", _lidar_data(), _group_df(tmp_path, False), {"temperature_c": 23.0, "pressure_hpa": 935.0}, _config(), logging.getLogger("test"))
    with xr.open_dataset(output_path) as ds:
        validate_level0_contract(ds)
        np.testing.assert_array_equal(ds["Laser_Shots"].values, np.array([[1200, 2400], [1300, 2600]], dtype=np.int32))
        np.testing.assert_allclose(ds["Raw_Data_Range_Resolution"].values, np.array([7.5, 15.0]))
        assert "DAQ_Range" in ds
        assert float(ds["DAQ_Range"].isel(channels=0).values) == 500.0
        assert float(ds["DAQ_Range"].isel(channels=1).values) > 1e30
        assert ds["DAQ_Range"].attrs["units"] == "mV"


def test_build_level0_netcdf_truncates_time_axis_and_shots(tmp_path: Path) -> None:
    output_path = tmp_path / "level0_truncated.nc"
    lidar_data = _lidar_data()
    lidar_data["tensors"] = {"532.AN": np.ones((1, 4)), "532.PC": np.ones((1, 4)) * 2.0}
    build_level0_netcdf(str(output_path), "20240101sant", "nt", lidar_data, _group_df(tmp_path, False), {"temperature_c": 23.0, "pressure_hpa": 935.0}, _config(), logging.getLogger("test"))
    with xr.open_dataset(output_path) as ds:
        assert ds.sizes["time"] == 1
        np.testing.assert_array_equal(ds["Laser_Shots"].values, np.array([[1200, 2400]], dtype=np.int32))


def test_build_level0_netcdf_uses_fallback_channel_id_when_missing_from_config(tmp_path: Path) -> None:
    output_path = tmp_path / "level0_default_channel_id.nc"
    config = _config(); config["hardware"]["name_to_id"]["night"] = {"532.AN": 722}
    build_level0_netcdf(str(output_path), "20240101sant", "nt", _lidar_data(), _group_df(tmp_path, False), {"temperature_c": 23.0, "pressure_hpa": 935.0}, config, logging.getLogger("test"))
    with xr.open_dataset(output_path) as ds:
        assert np.array_equal(ds["channel_ID"].values, np.array([722, 9999]))


def test_build_level0_netcdf_accepts_flat_shared_channel_id_mapping(tmp_path: Path) -> None:
    output_path = tmp_path / "level0_flat_channel_id.nc"
    config = _config(); config["hardware"]["name_to_id"] = {"532.AN": 722, "532.PC": 716}
    build_level0_netcdf(str(output_path), "20240101sant", "pm", _lidar_data(), _group_df(tmp_path, False), {"temperature_c": 23.0, "pressure_hpa": 935.0}, config, logging.getLogger("test"))
    with xr.open_dataset(output_path) as ds:
        assert np.array_equal(ds["channel_ID"].values, np.array([722, 716]))


def test_build_level0_netcdf_writes_dark_current_scc_times_and_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import milgrau.level0.netcdf as netcdf_module
    monkeypatch.setattr(netcdf_module, "parse_licel_group", lambda files, logger: {"channels": ["532.AN", "532.PC"], "tensors": {"532.AN": np.ones((1, 4)) * 0.1, "532.PC": np.ones((1, 4)) * 0.2}})
    output_path = tmp_path / "level0.nc"
    build_level0_netcdf(str(output_path), "20240101sant", "nt", _lidar_data(), _group_df(tmp_path, True), {"temperature_c": 23.0, "pressure_hpa": 935.0}, _config(), logging.getLogger("test"))
    with xr.open_dataset(output_path) as ds:
        validate_level0_contract(ds)
        assert ds.attrs["RawBck_Start_Date"] == "20231231"
        assert ds.attrs["RawBck_Start_Time_UT"] == "234000"
        assert ds.attrs["RawBck_Stop_Time_UT"] == "234500"
        assert ds.attrs["Dark_Current_Source_File_Count"] == 1
        assert np.array_equal(ds["Background_Profile_Available"].values, np.array([1, 1], dtype=np.int8))


def test_build_level0_netcdf_flags_missing_dark_current_channel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import milgrau.level0.netcdf as netcdf_module
    monkeypatch.setattr(netcdf_module, "parse_licel_group", lambda files, logger: {"channels": ["532.AN"], "tensors": {"532.AN": np.ones((1, 4)) * 0.1}})
    output_path = tmp_path / "level0_missing_dc_channel.nc"
    build_level0_netcdf(str(output_path), "20240101sant", "nt", _lidar_data(), _group_df(tmp_path, True), {"temperature_c": 23.0, "pressure_hpa": 935.0}, _config(), logging.getLogger("test"))
    with xr.open_dataset(output_path) as ds:
        validate_level0_contract(ds)
        assert np.array_equal(ds["Background_Profile_Available"].values, np.array([1, 0], dtype=np.int8))
        assert np.all(np.isnan(ds["Background_Profile"].isel(channels=1).values))


def test_build_level0_netcdf_without_dark_current_writes_unavailable_flags(tmp_path: Path) -> None:
    output_path = tmp_path / "level0_no_dc.nc"
    build_level0_netcdf(str(output_path), "20240101sant", "nt", _lidar_data(), _group_df(tmp_path, False), {"temperature_c": 23.0, "pressure_hpa": 935.0}, _config(), logging.getLogger("test"))
    with xr.open_dataset(output_path) as ds:
        validate_level0_contract(ds)
        assert "Background_Profile" not in ds
        assert np.array_equal(ds["Background_Profile_Available"].values, np.array([0, 0], dtype=np.int8))
