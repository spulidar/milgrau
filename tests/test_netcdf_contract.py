"""Synthetic NetCDF contract tests for Level 1 and LEBEAR inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.level2.lebear import process_single_level1_file
from milgrau.operations import ExecutionStatus
from milgrau.physics.atmosphere import get_standard_atmosphere


class _ListLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(f"INFO: {message}")

    def warning(self, message: str) -> None:
        self.messages.append(f"WARNING: {message}")

    def error(self, message: str) -> None:
        self.messages.append(f"ERROR: {message}")


def _write_synthetic_level1(path: Path) -> Path:
    time = pd.date_range("2024-01-01T00:00:00", periods=3, freq="5min")
    altitude = np.arange(0.0, 1500.0, 7.5)
    channel = np.array(["532.AN", "532.PC"], dtype=object)
    shape = (time.size, channel.size, altitude.size)
    base_profile = np.exp(-altitude / 1000.0)
    corrected_signal = np.empty(shape, dtype=np.float32)
    corrected_signal_error = np.empty(shape, dtype=np.float32)
    range_corrected_signal = np.empty(shape, dtype=np.float32)
    range_corrected_signal_error = np.empty(shape, dtype=np.float32)

    for t_idx in range(time.size):
        for c_idx in range(channel.size):
            scale = 1.0 + 0.1 * t_idx + 0.05 * c_idx
            corrected_signal[t_idx, c_idx, :] = scale * base_profile
            corrected_signal_error[t_idx, c_idx, :] = 0.05 * np.abs(corrected_signal[t_idx, c_idx, :])
            range_corrected_signal[t_idx, c_idx, :] = corrected_signal[t_idx, c_idx, :] * altitude**2
            range_corrected_signal_error[t_idx, c_idx, :] = corrected_signal_error[t_idx, c_idx, :] * altitude**2

    pressure_hpa, temperature_k = get_standard_atmosphere(altitude + 760.0)
    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), corrected_signal),
            "corrected_signal_error": (("time", "channel", "altitude"), corrected_signal_error),
            "range_corrected_signal": (("time", "channel", "altitude"), range_corrected_signal),
            "range_corrected_signal_error": (("time", "channel", "altitude"), range_corrected_signal_error),
            "pc_saturation_mask": (("time", "channel", "altitude"), np.zeros(shape, dtype=np.int8)),
            "channel_correction_success": (("channel",), np.ones(channel.size, dtype=np.int8)),
            "PBL_Height_km": (("time",), np.array([0.8, 0.9, 1.0], dtype=np.float32)),
            "Atmospheric_Temperature_K": (("altitude",), temperature_k),
            "Atmospheric_Pressure_hPa": (("altitude",), pressure_hpa),
        },
        coords={"time": time, "channel": channel, "altitude": altitude},
        attrs={
            "Processing_level": "Level 1 synthetic test product",
            "Altitude_units": "m",
            "tropopause_cpt_km": -999.0,
            "tropopause_lrt_km": -999.0,
            "thermodynamic_profile_available": "true",
            "thermodynamic_profile_source_type": "ussa76",
            "thermodynamic_profile_source": "US Standard Atmosphere 1976",
            "thermodynamic_profile_standard_fallback_fraction": 1.0,
        },
    )
    ds["corrected_signal"].attrs["units"] = "channel native corrected units"
    ds["corrected_signal_error"].attrs["units"] = "channel native corrected units"
    ds["range_corrected_signal"].attrs["units"] = "a.u. m^2"
    ds["range_corrected_signal_error"].attrs["units"] = "a.u. m^2"
    ds["Atmospheric_Temperature_K"].attrs["units"] = "K"
    ds["Atmospheric_Pressure_hPa"].attrs["units"] = "hPa"
    ds["altitude"].attrs["units"] = "m"
    ds.to_netcdf(path)
    return path


def test_synthetic_level1_contract_contains_canonical_atmosphere(tmp_path: Path) -> None:
    path = _write_synthetic_level1(tmp_path / "synthetic_level1_rcs.nc")
    with xr.open_dataset(path) as ds:
        assert ds["Atmospheric_Temperature_K"].dims == ("altitude",)
        assert ds["Atmospheric_Pressure_hPa"].dims == ("altitude",)
        assert "Radiosonde_Temperature_K" not in ds
        assert "Radiosonde_Pressure_hPa" not in ds
        assert ds.attrs["thermodynamic_profile_source_type"] == "ussa76"


def test_lebear_uses_level1_atmosphere_and_generates_level2(tmp_path: Path) -> None:
    path = _write_synthetic_level1(tmp_path / "synthetic_level1_rcs.nc")
    logger = _ListLogger()
    config = {
        "directories": {"processed_data": str(tmp_path)},
        "site": {"station_altitude_m": 760.0},
        "inversion": {
            "wavelengths_to_process": [532],
            "monte_carlo_iterations": 10,
            "random_seed": 123,
            "molecular_fit": {"ref_alt_min_m": 500.0, "ref_alt_max_m": 1400.0, "ref_window_bins": 20},
            "gluing": {
                "window_length_bins": 80,
                "correlation_threshold": 0.95,
                "search_min_idx": 20,
                "search_max_idx": 120,
                "allow_single_channel_fallback": True,
                "single_channel_priority": "photon_counting",
            },
        },
        "visualization": {"level2_qa": {"enabled": False}},
    }

    summary = process_single_level1_file(path, config, logger)  # type: ignore[arg-type]
    output_path = tmp_path / "synthetic_level2_optical.nc"
    assert summary.results[0].status is ExecutionStatus.SUCCESS
    assert output_path.exists()

    with xr.open_dataset(output_path) as ds_l2:
        assert ds_l2.attrs["Molecular_sources"] == "ussa76"
        assert ds_l2.attrs["molecular_atmosphere_implementation_version"] == "3"
        assert ds_l2.attrs["molecular_atmosphere_scientific_change"] == "level1_materialized_atmosphere_with_log_pressure_interpolation"
        assert ds_l2["molecular_backscatter"].dims == ("wavelength", "altitude")
        assert np.all(np.isfinite(ds_l2["molecular_backscatter"].values))
        assert set(np.unique(ds_l2["retrieval_success_flag"].values).tolist()) == {1}
