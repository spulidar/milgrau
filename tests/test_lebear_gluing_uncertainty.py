"""Tests for LEBEAR gluing uncertainty propagation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.io.paths import level2_output_path
from milgrau.level2 import lebear
from milgrau.operations import ExecutionStatus


def _logger() -> logging.Logger:
    logger = logging.getLogger("test.lebear.gluing_uncertainty")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def test_propagate_glued_error_uses_fade_weights() -> None:
    analog_error = np.ones(10, dtype=np.float64) * 2.0
    photon_error = np.ones(10, dtype=np.float64) * 10.0

    result = lebear._propagate_glued_error(
        analog_error=analog_error,
        photon_error=photon_error,
        slope=3.0,
        min_bin=2,
        max_bin=6,
    )

    assert np.allclose(result[:2], 6.0)
    assert np.allclose(result[6:], 10.0)
    analog_weights = 1.0 - np.arange(4, dtype=np.float64) / 4.0
    photon_weights = 1.0 - analog_weights
    expected_window = np.sqrt((analog_weights * 6.0) ** 2 + (photon_weights * 10.0) ** 2)
    assert np.allclose(result[2:6], expected_window)
    assert not np.isclose(result[5], result[6])


def _write_level1(path: Path) -> Path:
    """Write a strict synthetic Level 1 product with canonical atmosphere."""
    time = pd.date_range("2024-01-01T00:00:00", periods=2, freq="5min")
    altitude = np.arange(240, dtype=np.float64) * 7.5
    channel = np.array(["532.AN", "532.PC"], dtype=object)
    shape = (time.size, channel.size, altitude.size)

    analog = np.tile(np.exp(-altitude / 900.0) + 0.1, (time.size, 1))
    photon = 2.0 * analog + 0.05
    corrected = np.stack([analog, photon], axis=1).astype(np.float32)
    corrected_error = (0.02 * corrected).astype(np.float32)
    range_factor = altitude.astype(np.float32) ** 2
    rcs = corrected * range_factor[None, None, :]
    rcs_error = corrected_error * range_factor[None, None, :]
    temperature_k = 288.15 - 0.0065 * altitude
    pressure_hpa = 1013.25 * np.exp(-altitude / 8434.0)

    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), corrected),
            "corrected_signal_error": (("time", "channel", "altitude"), corrected_error),
            "range_corrected_signal": (("time", "channel", "altitude"), rcs),
            "range_corrected_signal_error": (("time", "channel", "altitude"), rcs_error),
            "pc_saturation_mask": (("time", "channel", "altitude"), np.zeros(shape, dtype=np.int8)),
            "pc_saturation_characterized": (("channel",), np.array([0, 1], dtype=np.int8)),
            "channel_correction_success": (("channel",), np.ones(channel.size, dtype=np.int8)),
            "Atmospheric_Temperature_K": (("altitude",), temperature_k.astype(np.float64)),
            "Atmospheric_Pressure_hPa": (("altitude",), pressure_hpa.astype(np.float64)),
        },
        coords={"time": time, "channel": channel, "altitude": altitude},
        attrs={
            "Processing_level": "Level 1 synthetic gluing test product",
            "Altitude_units": "m",
            "thermodynamic_profile_source_type": "ussa76",
            "thermodynamic_profile_available": "true",
            "thermodynamic_profile_standard_fallback_fraction": 1.0,
        },
    )
    ds.to_netcdf(path)
    return path


def _config(tmp_path: Path) -> dict:
    months = {f"{month:02d}": 60.0 for month in range(1, 13)}
    return {
        "processing": {"incremental": False},
        "directories": {"processed_data": str(tmp_path)},
        "inversion": {
            "wavelengths_to_process": [532],
            "block_average_minutes": 15,
            "kfs_mode": "two_sided",
            "monte_carlo_iterations": 5,
            "random_seed": 123,
            "beta_ref_relative_std": 0.10,
            "aerosol_ref_fraction": 0.0,
            "min_lidar_ratio_sr": 10.0,
            "allow_negative_aerosol": False,
            "molecular_fit": {
                "ref_alt_min_m": 500.0,
                "ref_alt_max_m": 1500.0,
                "ref_window_bins": 20,
                "max_relative_slope": 10.0,
                "max_relative_variance": 10.0,
                "min_valid_fraction": 0.50,
            },
            "gluing": {
                "window_length_bins": 20,
                "correlation_threshold": 0.5,
                "search_min_idx": 20,
                "search_max_idx": 120,
                "intercept_threshold": 5.0,
                "gaussian_threshold": 1.0,
                "minmax_threshold": 1.0,
                "max_relative_rmse": 1.0,
                "max_relative_bias": 1.0,
                "min_valid_fraction": 0.50,
                "max_saturation_fraction": 0.20,
                "invalid_saturation_fraction": 1.0,
                "allow_single_channel_fallback": True,
                "single_channel_priority": "photon_counting",
            },
            "cloud_screening": {"enabled": False},
            "lidar_ratios_sr": {"532": months},
            "lidar_ratio_std_sr": {"532": 5.0},
        },
        "visualization": {"level2_qa": {"enabled": False}},
    }


def test_level2_saves_gluing_window_diagnostics(tmp_path: Path) -> None:
    level1 = _write_level1(tmp_path / "20240101sant_level1_rcs.nc")

    summary = lebear.process_single_level1_file(level1, _config(tmp_path), _logger())

    assert summary.results[0].status is ExecutionStatus.OK
    with xr.open_dataset(level2_output_path(level1)) as ds:
        assert "gluing_start_altitude_m" in ds
        assert "gluing_stop_altitude_m" in ds
        assert "Gluing_Error_Propagation" in ds.attrs
        assert np.isfinite(ds["gluing_start_altitude_m"].values).any()
        assert np.isfinite(ds["gluing_stop_altitude_m"].values).any()
        start = float(ds["gluing_start_altitude_m"].isel(time=0, wavelength=0))
        split = float(ds["gluing_split_altitude_m"].isel(time=0, wavelength=0))
        stop = float(ds["gluing_stop_altitude_m"].isel(time=0, wavelength=0))
        assert start < split < stop
        error_profile = ds["glued_range_corrected_signal_error"].isel(time=0, wavelength=0)
        window_error = error_profile.sel(altitude=slice(start, stop)).values
        assert np.isfinite(window_error).all()
        assert np.nanmin(window_error) > 0.0
