"""Tests for MILGRAU NetCDF contract validators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from milgrau.io.contracts import validate_level0_contract, validate_level1_contract


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
            "Surface_Temperature_C": (("weather_time",), np.array([20.0, 19.5])),
            "Surface_Pressure_hPa": (("weather_time",), np.array([930.0, 931.0])),
            "Surface_Relative_Humidity_percent": (("weather_time",), np.array([60.0, 62.0])),
            "Surface_Cloud_Cover_percent": (("weather_time",), np.array([20.0, 30.0])),
            "Surface_Wind_Speed_kmh": (("weather_time",), np.array([5.0, 6.0])),
            "solar_elevation_deg": (("time",), np.array([-20.0, -18.0])),
            "solar_regime": (("time",), np.array(["night", "night"], dtype=object)),
            "segment_id": (("time",), np.array(["seg00", "seg00"], dtype=object)),
            "Segment_Label": (("segments",), np.array(["seg00"], dtype=object)),
            "Segment_Regime": (("segments",), np.array(["night"], dtype=object)),
            "Segment_Start_Time_UTC": (("segments",), np.array([1704067200], dtype=np.int64)),
            "Segment_End_Time_UTC": (("segments",), np.array([1704067320], dtype=np.int64)),
        },
        coords={"weather_time": pd.date_range("2024-01-01T00:00:00", periods=2, freq="1h")},
        attrs={
            "Solar_Day_Night_Threshold_deg": -3.0,
            "Solar_Position_Algorithm": "test",
        },
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


def _add_level1_atmosphere(ds: xr.Dataset, *, source_type: str = "ussa76") -> xr.Dataset:
    n_altitude = ds.sizes["altitude"]
    atmosphere_time = pd.date_range("2024-01-01T00:00:00", periods=2, freq="1h")
    ds = ds.assign_coords(atmosphere_time=atmosphere_time)
    temperature = np.linspace(288.0, 270.0, n_altitude)
    pressure = np.linspace(1000.0, 900.0, n_altitude)
    ds["Atmospheric_Temperature_K"] = (
        ("atmosphere_time", "altitude"),
        np.vstack([temperature, temperature]),
    )
    ds["Atmospheric_Pressure_hPa"] = (
        ("atmosphere_time", "altitude"),
        np.vstack([pressure, pressure]),
    )
    ds["Atmospheric_Source_Type"] = (
        ("atmosphere_time",),
        np.array([source_type, source_type], dtype=object),
    )
    ds["Atmospheric_Source_Time_Delta_hours"] = (
        ("atmosphere_time",),
        np.array([0.0, 0.0]),
    )
    ds["Atmospheric_USSA76_Fallback_Fraction"] = (
        ("atmosphere_time",),
        np.array([1.0 if source_type == "ussa76" else 0.0] * 2),
    )
    ds["solar_elevation_deg"] = (("time",), np.array([20.0, 22.0]))
    ds["solar_regime"] = (("time",), np.array(["day", "day"], dtype=object))
    ds["segment_id"] = (("time",), np.array(["seg00", "seg00"], dtype=object))
    ds["Segment_Label"] = (("segments",), np.array(["seg00"], dtype=object))
    ds["Segment_Regime"] = (("segments",), np.array(["day"], dtype=object))
    ds["Segment_Start_Time_UTC"] = (("segments",), np.array([1704067200], dtype=np.int64))
    ds["Segment_End_Time_UTC"] = (("segments",), np.array([1704153600], dtype=np.int64))
    ds.attrs.update({
        "thermodynamic_profile_available": "true",
        "thermodynamic_profile_source_type": "time_resolved",
        "thermodynamic_profile_standard_fallback_fraction": 1.0 if source_type == "ussa76" else 0.0,
        "Solar_Day_Night_Threshold_deg": -3.0,
        "Solar_Position_Algorithm": "test",
        "Segment_Count": 1,
    })
    return ds


def _level1_signals(dim_order=("time", "channel", "altitude")) -> xr.Dataset:
    time = pd.date_range("2024-01-01", periods=2)
    coords = {"time": time, "channel": ["532.AN"], "altitude": np.arange(4.0)}
    sizes = {"time": 2, "channel": 1, "altitude": 4}
    shape = tuple(sizes[dim] for dim in dim_order)
    values = {name: (dim_order, np.ones(shape)) for name in (
        "corrected_signal", "corrected_signal_error", "range_corrected_signal", "range_corrected_signal_error"
    )}
    return xr.Dataset(values, coords=coords)


def test_validate_level1_contract_rejects_missing_materialized_atmosphere() -> None:
    with pytest.raises(KeyError, match="Atmospheric_Temperature_K"):
        validate_level1_contract(_level1_signals())


def test_validate_level1_contract_accepts_required_signals_and_atmosphere() -> None:
    ds = _add_level1_atmosphere(_level1_signals())
    validate_level1_contract(ds)


def test_validate_level1_contract_accepts_noncanonical_signal_dim_order() -> None:
    ds = _add_level1_atmosphere(_level1_signals(("channel", "altitude", "time")))
    validate_level1_contract(ds)


def test_validate_level1_contract_rejects_nonfinite_atmosphere() -> None:
    ds = _add_level1_atmosphere(_level1_signals())
    ds["Atmospheric_Pressure_hPa"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Atmospheric_Pressure_hPa"):
        validate_level1_contract(ds)
