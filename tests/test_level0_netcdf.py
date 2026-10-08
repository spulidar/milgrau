"""Tests for Level 0 NetCDF writing and station-owned metadata."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from milgrau.io.contracts import validate_level0_contract
from milgrau.level0.netcdf import build_level0_netcdf, validate_lidar_tensors

SESSION_ID = "spu_20240101-0000Z_20240101-0010Z"


def _config() -> dict:
    site = {"latitude": -23.5615, "longitude": -46.7383, "station_altitude_m": 740.0}
    return {
        "physics": {"vertical_resolution_m": 7.5},
        "level0": {"solar_regime": {"day_night_threshold_deg": -3.0}},
        "level1": {"background": {"start_altitude_m": 29000.0, "stop_altitude_m": 29999.0}},
        "_station_catalog": {
            "station": {
                "id": "spu",
                "name": "SPU-Lidar",
                "institution": "IPEN/USP",
                "timezone": "America/Sao_Paulo",
                "site": dict(site),
                "lidar_geometry": {"pointing_angle_deg_from_zenith": 0.0},
            }
        },
        "_resolved_station": {
            "station_id": "spu",
            "station_name": "SPU-Lidar",
            "profile_id": "spu-test",
            "calibration_id": "spu-channel-corrections-v1",
            "timezone": "America/Sao_Paulo",
            "site": dict(site),
            "scc_available": True,
            "scc_configuration_id": 484,
            "scc_configuration_name": "test SCC",
            "channel_ids": {"532.AN": 722, "532.PC": 716},
            "lr_input": {},
        },
    }


def _group_df(tmp_path: Path, include_dark_current: bool = True) -> pd.DataFrame:
    records = [
        {"filepath": str(tmp_path / "meas_0001"), "meas_type": "measurements", "start_time_utc": pd.Timestamp("2024-01-01T00:00:00Z"), "stop_time": pd.Timestamp("2024-01-01T00:05:00Z"), "original_session_id": SESSION_ID, "association_method": "measurement", "dark_current_association_delta_hours": np.nan, "solar_elevation_deg": -35.0, "solar_regime": "night", "segment_id": "seg00", "_profile_index": 0},
        {"filepath": str(tmp_path / "meas_0002"), "meas_type": "measurements", "start_time_utc": pd.Timestamp("2024-01-01T00:05:00Z"), "stop_time": pd.Timestamp("2024-01-01T00:10:00Z"), "original_session_id": SESSION_ID, "association_method": "measurement", "dark_current_association_delta_hours": np.nan, "solar_elevation_deg": -34.0, "solar_regime": "night", "segment_id": "seg00", "_profile_index": 1},
    ]
    if include_dark_current:
        records.append({"filepath": str(tmp_path / "dark_0001"), "meas_type": "dark_current", "start_time_utc": pd.Timestamp("2023-12-31T23:40:00Z"), "stop_time": pd.Timestamp("2023-12-31T23:45:00Z"), "original_session_id": SESSION_ID, "association_method": "nearest_session", "dark_current_association_delta_hours": 0.5})
    return pd.DataFrame.from_records(records)




def _weather_data(
    temperature_c: float = 23.0,
    pressure_hpa: float = 935.0,
) -> dict:
    return {
        "weather_time": np.array(
            ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
            dtype="datetime64[ns]",
        ),
        "temperature_c": np.array([temperature_c, temperature_c], dtype=np.float64),
        "pressure_hpa": np.array([pressure_hpa, pressure_hpa], dtype=np.float64),
        "relative_humidity_percent": np.array([60.0, 61.0], dtype=np.float64),
        "cloud_cover_percent": np.array([20.0, 25.0], dtype=np.float64),
        "wind_speed_kmh": np.array([5.0, 6.0], dtype=np.float64),
        "source": "synthetic",
        "cadence": "hourly",
    }

def _lidar_data() -> dict:
    return {
        "channels": ["532.AN", "532.PC"],
        "shots": 1200,
        "laser_shots": np.array([[1200, 2400], [1300, 2600]], dtype=np.int32),
        "channel_metadata": {
            "532.AN": {"is_pc": False, "bin_width_m": 7.5, "daq_range_mV": 500.0},
            "532.PC": {"is_pc": True, "bin_width_m": 15.0, "daq_range_mV": np.nan},
        },
        "tensors": {
            "532.AN": np.ones((2, 4), dtype=np.float64),
            "532.PC": np.ones((2, 4), dtype=np.float64) * 2.0,
        },
    }


def test_validate_lidar_tensors_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        validate_lidar_tensors({"532.AN": np.ones((2, 4)), "532.PC": np.ones((3, 4))}, ["532.AN", "532.PC"])


def test_build_level0_netcdf_writes_resolved_station_and_scc_metadata(tmp_path: Path) -> None:
    output_path = tmp_path / "level0_scc.nc"
    build_level0_netcdf(
        str(output_path), SESSION_ID, _lidar_data(), _group_df(tmp_path, False),
        _weather_data(), _config(), logging.getLogger("test")
    )
    with xr.open_dataset(output_path) as ds:
        validate_level0_contract(ds)
        np.testing.assert_array_equal(ds["Laser_Shots"].values, np.array([[1200, 2400], [1300, 2600]], dtype=np.int32))
        np.testing.assert_allclose(ds["Raw_Data_Range_Resolution"].values, np.array([7.5, 15.0]))
        np.testing.assert_allclose(ds["Background_Low"].values, np.array([29000.0, 29000.0]))
        np.testing.assert_allclose(ds["Background_High"].values, np.array([29999.0, 29999.0]))
        np.testing.assert_array_equal(ds["channel_ID"].values, np.array([722, 716]))
        np.testing.assert_allclose(ds["Laser_Pointing_Angle"].values, np.array([0.0]))
        assert ds.attrs["System"] == "SPU-Lidar"
        assert ds.attrs["Station_Profile"] == "spu-test"
        assert ds.attrs["Session_ID"] == SESSION_ID
        assert ds.attrs["measurement_start_time"] == "2024-01-01T00:00:00Z"
        assert ds.attrs["measurement_end_time"] == "2024-01-01T00:10:00Z"
        assert ds.attrs["session_duration_seconds"] == pytest.approx(600.0)
        assert ds.attrs["timezone"] == "America/Sao_Paulo"
        assert ds.attrs["Latitude_degrees_north"] == pytest.approx(-23.5615)
        assert ds.attrs["Longitude_degrees_east"] == pytest.approx(-46.7383)
        assert "DAQ_Range" in ds
        assert ds["Surface_Temperature_C"].dims == ("weather_time",)
        assert ds["Surface_Pressure_hPa"].dims == ("weather_time",)
        assert ds.sizes["weather_time"] == 2
        assert ds["solar_elevation_deg"].dims == ("time",)
        assert ds["solar_regime"].values.astype(str).tolist() == ["night", "night"]
        assert ds["segment_id"].values.astype(str).tolist() == ["seg00", "seg00"]
        assert ds["Segment_Label"].values.astype(str).tolist() == ["seg00"]
        assert ds["Segment_Regime"].values.astype(str).tolist() == ["night"]
        assert ds.attrs["Solar_Day_Night_Threshold_deg"] == pytest.approx(-3.0)
        assert float(ds["Temperature_at_Lidar_Station"].values) == pytest.approx(23.0)
        assert float(ds["Pressure_at_Lidar_Station"].values) == pytest.approx(935.0)
        assert float(ds["DAQ_Range"].isel(channels=0).values) == 500.0
        assert float(ds["DAQ_Range"].isel(channels=1).values) > 1e30


def test_pointing_angle_has_no_physics_fallback(tmp_path: Path) -> None:
    output_path = tmp_path / "level0_pointing.nc"
    config = _config()
    config["physics"]["laser_pointing_angle_deg"] = 17.0
    build_level0_netcdf(
        str(output_path), SESSION_ID, _lidar_data(), _group_df(tmp_path, False),
        _weather_data(), config, logging.getLogger("test")
    )
    with xr.open_dataset(output_path) as ds:
        np.testing.assert_allclose(ds["Laser_Pointing_Angle"].values, np.array([0.0]))


def test_build_level0_netcdf_requires_station_pointing_geometry(tmp_path: Path) -> None:
    config = _config()
    del config["_station_catalog"]["station"]["lidar_geometry"]
    with pytest.raises(RuntimeError, match="lidar_geometry"):
        build_level0_netcdf(
            str(tmp_path / "missing_geometry.nc"), SESSION_ID, _lidar_data(), _group_df(tmp_path, False),
            _weather_data(), config, logging.getLogger("test")
        )


def test_build_level0_netcdf_rejects_missing_native_bin_width_even_with_legacy_vertical_resolution(tmp_path: Path) -> None:
    output_path = tmp_path / "level0_missing_bin_width.nc"
    lidar_data = _lidar_data()
    lidar_data["channel_metadata"]["532.PC"].pop("bin_width_m")
    config = _config()
    config["physics"]["vertical_resolution_m"] = 7.5

    with pytest.raises(RuntimeError, match="range resolution cannot be invented"):
        build_level0_netcdf(
            str(output_path), SESSION_ID, lidar_data, _group_df(tmp_path, False),
            _weather_data(), config, logging.getLogger("test")
        )


def test_build_level0_netcdf_requires_explicit_background_window(tmp_path: Path) -> None:
    config = _config()
    del config["level1"]
    with pytest.raises(RuntimeError, match="level1.*background"):
        build_level0_netcdf(
            str(tmp_path / "missing_background.nc"), SESSION_ID, _lidar_data(), _group_df(tmp_path, False),
            _weather_data(), config, logging.getLogger("test")
        )


def test_build_level0_netcdf_truncates_time_axis_and_shots(tmp_path: Path) -> None:
    output_path = tmp_path / "level0_truncated.nc"
    lidar_data = _lidar_data()
    lidar_data["tensors"] = {"532.AN": np.ones((1, 4)), "532.PC": np.ones((1, 4)) * 2.0}
    build_level0_netcdf(
        str(output_path), SESSION_ID, lidar_data, _group_df(tmp_path, False),
        _weather_data(), _config(), logging.getLogger("test")
    )
    with xr.open_dataset(output_path) as ds:
        assert ds.sizes["time"] == 1
        np.testing.assert_array_equal(ds["Laser_Shots"].values, np.array([[1200, 2400]], dtype=np.int32))


def test_build_level0_netcdf_rejects_missing_resolved_scc_channel_id(tmp_path: Path) -> None:
    config = _config()
    config["_resolved_station"]["channel_ids"] = {"532.AN": 722}
    with pytest.raises(RuntimeError, match="no SCC channel ID"):
        build_level0_netcdf(
            str(tmp_path / "missing_channel_id.nc"), SESSION_ID, _lidar_data(), _group_df(tmp_path, False),
            _weather_data(), config, logging.getLogger("test")
        )


def test_legacy_hardware_map_cannot_override_resolved_station_mapping(tmp_path: Path) -> None:
    output_path = tmp_path / "level0_resolved_ids.nc"
    config = _config()
    config["hardware"] = {"name_to_id": {"532.AN": 1, "532.PC": 2}}
    build_level0_netcdf(
        str(output_path), SESSION_ID, _lidar_data(), _group_df(tmp_path, False),
        _weather_data(), config, logging.getLogger("test")
    )
    with xr.open_dataset(output_path) as ds:
        np.testing.assert_array_equal(ds["channel_ID"].values, np.array([722, 716]))


def test_missing_surface_weather_is_persisted_as_nan_without_25_940_fallback(tmp_path: Path) -> None:
    output_path = tmp_path / "level0_missing_weather.nc"
    build_level0_netcdf(
        str(output_path), SESSION_ID, _lidar_data(), _group_df(tmp_path, False),
        _weather_data(np.nan, np.nan), _config(), logging.getLogger("test")
    )
    with xr.open_dataset(output_path) as ds:
        assert np.isnan(ds["Surface_Temperature_C"].values).all()
        assert np.isnan(ds["Surface_Pressure_hPa"].values).all()
        assert np.isnan(float(ds["Temperature_at_Lidar_Station"].values))
        assert np.isnan(float(ds["Pressure_at_Lidar_Station"].values))


def test_build_level0_netcdf_writes_dark_current_scc_times_and_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import milgrau.level0.netcdf as netcdf_module
    monkeypatch.setattr(
        netcdf_module,
        "parse_licel_group",
        lambda files, logger: {
            "channels": ["532.AN", "532.PC"],
            "tensors": {"532.AN": np.ones((1, 4)) * 0.1, "532.PC": np.ones((1, 4)) * 0.2},
            "laser_shots": np.array([[111, 222]], dtype=np.int32),
        },
    )
    output_path = tmp_path / "level0.nc"
    build_level0_netcdf(
        str(output_path), SESSION_ID, _lidar_data(), _group_df(tmp_path, True),
        _weather_data(), _config(), logging.getLogger("test")
    )
    with xr.open_dataset(output_path) as ds:
        validate_level0_contract(ds)
        assert ds.attrs["RawBck_Start_Date"] == "20231231"
        assert ds.attrs["RawBck_Start_Time_UT"] == "234000"
        assert ds.attrs["RawBck_Stop_Time_UT"] == "234500"
        assert ds.attrs["Dark_Current_Source_File_Count"] == 1
        assert np.array_equal(ds["Background_Profile_Available"].values, np.array([1, 1], dtype=np.int8))
        np.testing.assert_allclose(ds["Background_Laser_Shots"].values, np.array([[111.0, 222.0]]))


def test_build_level0_netcdf_flags_missing_dark_current_channel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import milgrau.level0.netcdf as netcdf_module
    monkeypatch.setattr(
        netcdf_module,
        "parse_licel_group",
        lambda files, logger: {
            "channels": ["532.AN"],
            "tensors": {"532.AN": np.ones((1, 4)) * 0.1},
            "laser_shots": np.array([[111]], dtype=np.int32),
        },
    )
    output_path = tmp_path / "level0_missing_dc_channel.nc"
    build_level0_netcdf(
        str(output_path), SESSION_ID, _lidar_data(), _group_df(tmp_path, True),
        _weather_data(), _config(), logging.getLogger("test")
    )
    with xr.open_dataset(output_path) as ds:
        validate_level0_contract(ds)
        assert np.array_equal(ds["Background_Profile_Available"].values, np.array([1, 0], dtype=np.int8))
        assert np.all(np.isnan(ds["Background_Profile"].isel(channels=1).values))
        assert float(ds["Background_Laser_Shots"].isel(channels=0).values[0]) == 111.0
        assert np.isnan(float(ds["Background_Laser_Shots"].isel(channels=1).values[0]))


def test_build_level0_netcdf_without_dark_current_writes_unavailable_flags(tmp_path: Path) -> None:
    output_path = tmp_path / "level0_no_dc.nc"
    build_level0_netcdf(
        str(output_path), SESSION_ID, _lidar_data(), _group_df(tmp_path, False),
        _weather_data(), _config(), logging.getLogger("test")
    )
    with xr.open_dataset(output_path) as ds:
        validate_level0_contract(ds)
        assert "Background_Profile" not in ds
        assert "Background_Laser_Shots" not in ds
        assert np.array_equal(ds["Background_Profile_Available"].values, np.array([0, 0], dtype=np.int8))
