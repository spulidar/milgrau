"""Characterization tests for Level 0 ingestion and correction failures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from milgrau.level1.corrections import apply_instrumental_corrections
from milgrau.level1.ingestion import load_and_prepare_level0


class _ListLogger:
    """Capture ingestion diagnostics for assertions."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(f"INFO: {message}")

    def warning(self, message: str) -> None:
        self.messages.append(f"WARNING: {message}")

    def error(self, message: str) -> None:
        self.messages.append(f"ERROR: {message}")


def _write_level0(path: Path, resolutions: np.ndarray) -> Path:
    """Write a minimal SCC-style Level 0 file accepted by the ingestion contract."""
    ds = xr.Dataset(
        data_vars={
            "Raw_Data_Start_Time": (("time", "nb_of_time_scales"), np.array([[0], [60]], dtype=np.int32)),
            "Raw_Data_Stop_Time": (("time", "nb_of_time_scales"), np.array([[60], [120]], dtype=np.int32)),
            "Raw_Data_Range_Resolution": (("channels",), resolutions),
            "Laser_Pointing_Angle": (("scan_angles",), np.array([0.0])),
            "Laser_Pointing_Angle_of_Profiles": (
                ("time", "nb_of_time_scales"),
                np.zeros((2, 1), dtype=np.int32),
            ),
            "Laser_Shots": (("time", "channels"), np.full((2, 2), 1200, dtype=np.int32)),
            "Molecular_Calc": ((), np.array(0, dtype=np.int32)),
            "id_timescale": (("channels",), np.zeros(2, dtype=np.int32)),
            "channel_string": (("channels",), np.array(["532.AN", "532.PC"], dtype=object)),
            "Raw_Lidar_Data": (("time", "channels", "points"), np.ones((2, 2, 4), dtype=np.float32)),
        },
        attrs={"RawData_Start_Date": "20240101", "RawData_Start_Time_UT": "000000"},
    )
    ds.to_netcdf(path)
    return path


def test_load_and_prepare_level0_decodes_time_and_named_coordinates(tmp_path: Path) -> None:
    """Ingestion should decode relative seconds and expose canonical Level 1 coordinates."""
    path = _write_level0(tmp_path / "level0.nc", np.array([7.5, 7.5]))
    logger = _ListLogger()

    ds, altitude = load_and_prepare_level0(path, logger)

    try:
        np.testing.assert_array_equal(altitude, np.array([0.0, 7.5, 15.0, 22.5]))
        np.testing.assert_array_equal(ds.altitude.values, altitude)
        np.testing.assert_array_equal(ds.channel.values.astype(str), np.array(["532.AN", "532.PC"]))
        np.testing.assert_array_equal(
            ds.time.values,
            pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T00:01:00"]).values,
        )
        assert ds["Raw_Lidar_Data"].dims == ("time", "channel", "altitude")
        assert ds.altitude.attrs == {"units": "m", "long_name": "Altitude above station"}
        assert any("2 profiles, 2 channels, 4 bins" in message for message in logger.messages)
    finally:
        ds.close()


def test_load_and_prepare_level0_warns_and_uses_first_resolution(tmp_path: Path) -> None:
    """Differing channel resolutions should preserve the current first-channel policy."""
    path = _write_level0(tmp_path / "level0.nc", np.array([7.5, 15.0]))
    logger = _ListLogger()

    ds, altitude = load_and_prepare_level0(path, logger)

    try:
        np.testing.assert_array_equal(altitude, np.array([0.0, 7.5, 15.0, 22.5]))
        assert any("Using the first value: 7.500000 m" in message for message in logger.messages)
    finally:
        ds.close()


def test_load_and_prepare_level0_rejects_nonfinite_resolution(tmp_path: Path) -> None:
    """A Level 0 file without a finite range resolution should fail and log its path."""
    path = _write_level0(tmp_path / "level0.nc", np.array([np.nan, np.nan]))
    logger = _ListLogger()

    with pytest.raises(ValueError, match="contains no finite values"):
        load_and_prepare_level0(path, logger)

    assert any(message.startswith(f"ERROR:   -> Failed to ingest Level 0 file {path}") for message in logger.messages)


@pytest.mark.parametrize(
    ("shots", "bin_time_us", "message"),
    [
        (0.0, 0.05, "Invalid laser shots value"),
        (1200.0, 0.0, "Invalid bin_time_us value"),
    ],
)
def test_apply_instrumental_corrections_rejects_invalid_acquisition_scale(
    shots: float,
    bin_time_us: float,
    message: str,
) -> None:
    """Invalid acquisition scalars should fail before any scientific correction is applied."""
    sig = xr.DataArray(np.ones((1, 4)), dims=("time", "range"))
    altitude = xr.DataArray(np.arange(4, dtype=float) * 7.5, dims=("range",))
    background_mask = altitude >= 15.0

    with pytest.raises(ValueError, match=message):
        apply_instrumental_corrections(
            sig=sig,
            z_da=altitude,
            shots=shots,
            bin_time_us=bin_time_us,
            deadtime=0.0,
            shift=0,
            bg_offset=0.0,
            is_photon=True,
            bg_mask=background_mask,
        )
