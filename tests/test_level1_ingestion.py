"""Characterization tests for Level 0 ingestion and correction failures."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from milgrau.level1.corrections import apply_instrumental_corrections
from milgrau.level1.ingestion import load_and_prepare_level0


class _ListLogger(logging.Logger):
    """Capture stdlib-compatible logger messages without global configuration."""

    def __init__(self) -> None:
        super().__init__("test.level1.ingestion", level=logging.DEBUG)
        self.messages: list[str] = []
        self.propagate = False

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):  # noqa: D401
        rendered = str(msg) % args if args else str(msg)
        self.messages.append(f"{logging.getLevelName(level)}: {rendered}")


def _write_level0(path: Path, resolutions: np.ndarray) -> Path:
    ds = xr.Dataset(
        data_vars={
            "Raw_Data_Start_Time": (("time", "nb_of_time_scales"), np.array([[0], [60]], dtype=np.int32)),
            "Raw_Data_Stop_Time": (("time", "nb_of_time_scales"), np.array([[60], [120]], dtype=np.int32)),
            "Raw_Data_Range_Resolution": (("channels",), resolutions),
            "Laser_Pointing_Angle": (("scan_angles",), np.array([0.0])),
            "Laser_Pointing_Angle_of_Profiles": (("time", "nb_of_time_scales"), np.zeros((2, 1), dtype=np.int32)),
            "Laser_Shots": (("time", "channels"), np.array([[1200, 2400], [1300, 2600]], dtype=np.int32)),
            "Molecular_Calc": ((), np.array(0, dtype=np.int32)),
            "id_timescale": (("channels",), np.zeros(2, dtype=np.int32)),
            "channel_string": (("channels",), np.array(["532.AN", "532.PC"], dtype=object)),
            "DAQ_Range": (("channels",), np.array([500.0, np.nan])),
            "Raw_Lidar_Data": (("time", "channels", "points"), np.ones((2, 2, 4), dtype=np.float32)),
        },
        attrs={"RawData_Start_Date": "20240101", "RawData_Start_Time_UT": "000000"},
    )
    ds.to_netcdf(path)
    return path


def _station_config(
    day_channels: dict[str, int],
    night_channels: dict[str, int] | None = None,
    *,
    day_configuration_id: int = 10,
    night_configuration_id: int = 11,
) -> dict:
    return {
        "_station_catalog": {
            "station": {"timezone": "America/Sao_Paulo"},
            "profiles": [
                {
                    "id": "test-profile",
                    "valid_from": "2020-01-01",
                    "valid_to": None,
                    "scc": {
                        "day": {
                            "configuration_id": day_configuration_id,
                            "channels": day_channels,
                        },
                        "night": {
                            "configuration_id": night_configuration_id,
                            "channels": day_channels if night_channels is None else night_channels,
                        },
                    },
                }
            ]
        }
    }


def _write_scc_id_level0(
    source: Path,
    destination: Path,
    channel_ids: np.ndarray,
    *,
    keep_channel_string: bool = False,
    attrs: dict[str, object] | None = None,
) -> Path:
    with xr.open_dataset(source) as opened:
        ds = opened.load()
    if not keep_channel_string:
        ds = ds.drop_vars("channel_string")
    ds["channel_ID"] = xr.DataArray(np.asarray(channel_ids, dtype=np.int32), dims=("channels",))
    if attrs:
        ds.attrs.update(attrs)
    ds.to_netcdf(destination)
    return destination


def test_load_and_prepare_level0_decodes_time_and_center_bin_coordinates(tmp_path: Path) -> None:
    path = _write_level0(tmp_path / "level0.nc", np.array([7.5, 7.5]))
    logger = _ListLogger()
    ds, altitude = load_and_prepare_level0(path, logger)
    try:
        np.testing.assert_array_equal(altitude, np.array([3.75, 11.25, 18.75, 26.25]))
        np.testing.assert_array_equal(ds.altitude.values, altitude)
        np.testing.assert_array_equal(ds.channel.values.astype(str), np.array(["532.AN", "532.PC"]))
        np.testing.assert_array_equal(ds.time.values, pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T00:01:00"]).values)
        assert ds["Raw_Lidar_Data"].dims == ("time", "channel", "altitude")
        assert ds.altitude.attrs == {"units": "m", "long_name": "Altitude above station (range-bin centers)"}
        assert any("2 profiles | 2 channels | 4 bins" in message for message in logger.messages)
    finally:
        ds.close()


def test_load_and_prepare_level0_uses_finest_resolution_and_preserves_native_values(tmp_path: Path) -> None:
    path = _write_level0(tmp_path / "level0.nc", np.array([15.0, 7.5]))
    logger = _ListLogger()
    ds, altitude = load_and_prepare_level0(path, logger)
    try:
        np.testing.assert_array_equal(altitude, np.array([3.75, 11.25, 18.75, 26.25]))
        np.testing.assert_array_equal(ds["Raw_Data_Range_Resolution"].values, np.array([15.0, 7.5]))
        assert any("mixed native range resolution" in message for message in logger.messages)
    finally:
        ds.close()


def test_load_and_prepare_level0_rejects_nonfinite_resolution(tmp_path: Path) -> None:
    path = _write_level0(tmp_path / "level0.nc", np.array([np.nan, np.nan]))
    logger = _ListLogger()
    with pytest.raises(ValueError, match="Range_Resolution"):
        load_and_prepare_level0(path, logger)
    assert any(f"ERROR: failed | {path.name} |" in message for message in logger.messages)


def test_scc_raw_channel_ids_are_canonicalized_from_station_mapping(tmp_path: Path) -> None:
    source = _write_level0(tmp_path / "canonical.nc", np.array([7.5, 7.5]))
    path = _write_scc_id_level0(source, tmp_path / "external_scc.nc", np.array([4069, 4070]))
    logger = _ListLogger()
    config = _station_config({"532.AN": 4069, "532.PC": 4070})

    ds, _ = load_and_prepare_level0(path, logger, config=config)
    try:
        np.testing.assert_array_equal(ds.channel.values.astype(str), np.array(["532.AN", "532.PC"]))
        assert ds.attrs["milgrau_level0_input_schema"] == "scc_raw_channel_ID_canonicalized"
        assert ds.attrs["milgrau_channel_identity_source"] == "station.yaml_scc_channel_ID_mapping"
        assert ds.attrs["milgrau_scc_mapping_modes"] == "night"
    finally:
        ds.close()


def test_milgrau_scc_file_cross_checks_channel_string_against_channel_id(tmp_path: Path) -> None:
    source = _write_level0(tmp_path / "canonical.nc", np.array([7.5, 7.5]))
    path = _write_scc_id_level0(
        source,
        tmp_path / "20240101_spu_06_L0_scc.nc",
        np.array([4069, 4070]),
        keep_channel_string=True,
        attrs={
            "Measurement_ID": "20240101_spu_06",
            "RawData_Start_Time_UT": "120000",
            "SCC_Configuration_ID": 10,
        },
    )
    logger = _ListLogger()
    config = _station_config({"532.AN": 4069, "532.PC": 4070})

    ds, _ = load_and_prepare_level0(path, logger, config=config)
    try:
        np.testing.assert_array_equal(ds.channel.values.astype(str), np.array(["532.AN", "532.PC"]))
        assert ds.attrs["milgrau_channel_identity_source"] == "channel_string_verified_against_station_scc_channel_ID"
        assert ds.attrs["milgrau_scc_mapping_modes"] == "day"
    finally:
        ds.close()


def test_scc_channel_id_mapping_rejects_unknown_id(tmp_path: Path) -> None:
    source = _write_level0(tmp_path / "canonical.nc", np.array([7.5, 7.5]))
    path = _write_scc_id_level0(source, tmp_path / "external_scc.nc", np.array([4069, 9999]))
    logger = _ListLogger()
    config = _station_config({"532.AN": 4069, "532.PC": 4070})

    with pytest.raises(ValueError, match="do not resolve"):
        load_and_prepare_level0(path, logger, config=config)


def test_scc_channel_id_mapping_uses_station_local_time_for_day_night_identity(tmp_path: Path) -> None:
    source = _write_level0(tmp_path / "canonical.nc", np.array([7.5, 7.5]))
    path = _write_scc_id_level0(source, tmp_path / "external_scc.nc", np.array([20, 21]))
    logger = _ListLogger()
    config = _station_config(
        {"532.AN": 20, "532.PC": 21},
        {"355.AN": 20, "355.PC": 21},
    )

    ds, _ = load_and_prepare_level0(path, logger, config=config)
    try:
        np.testing.assert_array_equal(ds.channel.values.astype(str), np.array(["355.AN", "355.PC"]))
        assert ds.attrs["milgrau_scc_mapping_modes"] == "night"
    finally:
        ds.close()


def test_scc_configuration_id_disambiguates_channel_identity(tmp_path: Path) -> None:
    source = _write_level0(tmp_path / "canonical.nc", np.array([7.5, 7.5]))
    path = _write_scc_id_level0(
        source,
        tmp_path / "external_scc.nc",
        np.array([20, 21]),
        attrs={
            "RawData_Start_Time_UT": "120000",
            "SCC_Configuration_ID": 10,
        },
    )
    logger = _ListLogger()
    config = _station_config(
        {"532.AN": 20, "532.PC": 21},
        {"355.AN": 20, "355.PC": 21},
        day_configuration_id=10,
        night_configuration_id=11,
    )

    ds, _ = load_and_prepare_level0(path, logger, config=config)
    try:
        np.testing.assert_array_equal(ds.channel.values.astype(str), np.array(["532.AN", "532.PC"]))
        assert ds.attrs["milgrau_scc_mapping_configuration_ids"] == "10"
    finally:
        ds.close()


@pytest.mark.parametrize(("shots", "bin_time_us", "message"), [(0.0, 0.05, "Invalid laser shots value"), (1200.0, 0.0, "Invalid bin_time_us value")])
def test_apply_instrumental_corrections_rejects_invalid_acquisition_scale(shots: float, bin_time_us: float, message: str) -> None:
    sig = xr.DataArray(np.ones((1, 4)), dims=("time", "range"))
    altitude = xr.DataArray(np.array([3.75, 11.25, 18.75, 26.25]), dims=("range",))
    background_mask = altitude >= 18.0
    with pytest.raises(ValueError, match=message):
        apply_instrumental_corrections(
            sig=sig, z_da=altitude, shots=shots, bin_time_us=bin_time_us,
            deadtime=0.0, shift=0, bg_offset=0.0, is_photon=True, bg_mask=background_mask,
            deadtime_min_denominator=0.05, pc_saturation_max_rate_mhz=None,
        )


def test_apply_instrumental_corrections_uses_per_profile_laser_shots() -> None:
    sig = xr.DataArray(np.array([[100.0, 50.0, 0.0], [100.0, 50.0, 0.0]]), dims=("time", "range"), coords={"time": [0, 1]})
    altitude = xr.DataArray(np.array([3.75, 11.25, 18.75]), dims=("range",))
    background_mask = xr.DataArray(np.array([False, False, True]), dims=("range",))
    shots = xr.DataArray(np.array([100.0, 200.0]), dims=("time",), coords={"time": [0, 1]})
    corrected, _, _, _ = apply_instrumental_corrections(
        sig=sig, z_da=altitude, shots=shots, bin_time_us=0.05,
        deadtime=0.0, shift=0, bg_offset=0.0, is_photon=True, bg_mask=background_mask,
        deadtime_min_denominator=0.05, pc_saturation_max_rate_mhz=None,
    )
    np.testing.assert_allclose(corrected.isel(range=0).values, np.array([20.0, 10.0]))
