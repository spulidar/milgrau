"""Synthetic NetCDF contract tests for Level 1 and productive method-v5 Level 2."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.level2.lebear import process_single_level1_file
from milgrau.level2.schema_v5 import validate_method_v5_level2_contract
from milgrau.operations import ExecutionStatus
from milgrau.physics.atmosphere import get_standard_atmosphere
from milgrau.scientific import (
    LEVEL2_PRODUCT_SCHEMA_VERSION,
    LEVEL2_RETRIEVAL_METHOD_VERSION,
)


class _ListLogger(logging.Logger):
    """Capture stdlib-compatible messages without configuring global logging."""

    def __init__(self) -> None:
        super().__init__("test.netcdf.contract", level=logging.DEBUG)
        self.messages: list[str] = []
        self.propagate = False

    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
    ):  # noqa: D401
        rendered = str(msg) % args if args else str(msg)
        self.messages.append(f"{logging.getLevelName(level)}: {rendered}")


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
            corrected_signal_error[t_idx, c_idx, :] = 0.05 * np.abs(
                corrected_signal[t_idx, c_idx, :]
            )
            range_corrected_signal[t_idx, c_idx, :] = (
                corrected_signal[t_idx, c_idx, :] * altitude**2
            )
            range_corrected_signal_error[t_idx, c_idx, :] = (
                corrected_signal_error[t_idx, c_idx, :] * altitude**2
            )

    pressure_hpa, temperature_k = get_standard_atmosphere(altitude + 760.0)
    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), corrected_signal),
            "corrected_signal_error": (
                ("time", "channel", "altitude"),
                corrected_signal_error,
            ),
            "range_corrected_signal": (
                ("time", "channel", "altitude"),
                range_corrected_signal,
            ),
            "range_corrected_signal_error": (
                ("time", "channel", "altitude"),
                range_corrected_signal_error,
            ),
            "pc_saturation_mask": (
                ("time", "channel", "altitude"),
                np.zeros(shape, dtype=np.int8),
            ),
            "channel_correction_success": (
                ("channel",),
                np.ones(channel.size, dtype=np.int8),
            ),
            "PBL_Height_km": (
                ("time",),
                np.array([0.8, 0.9, 1.0], dtype=np.float32),
            ),
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


def _level2_config(tmp_path: Path) -> dict:
    months = {f"{month:02d}": 60.0 for month in range(1, 13)}
    return {
        "processing": {"incremental": False},
        "directories": {"processed_data": str(tmp_path)},
        "inversion": {
            "wavelengths_to_process": [532],
            "block_average_minutes": 15,
            "kfs_mode": "backward",
            "monte_carlo_iterations": 6,
            "random_seed": 123,
            "beta_ref_relative_std": 0.0,
            "aerosol_ref_fraction": 0.0,
            "min_lidar_ratio_sr": 10.0,
            "allow_negative_aerosol": False,
            "molecular_fit": {
                "ref_alt_min_m": 500.0,
                "ref_alt_max_m": 1400.0,
                "ref_window_m": 150.0,
                "max_relative_slope": 10.0,
                "max_relative_variance": 10.0,
                "min_valid_fraction": 0.50,
            },
            "method_v5": {
                "reference_tier_min_altitudes_m": [1200.0, 900.0, 600.0],
                "reference_search_max_m": 1400.0,
                "path_start_altitude_m": 7.5,
                "residual_aerosol_fractions": [0.0, 0.02],
                "uncertainty_mode": "independent",
                "progressive_grid_schedule": [[0.0, 7.5]],
            },
            "gluing": {
                "window_length_bins": 20,
                "correlation_threshold": 0.5,
                "search_min_idx": 20,
                "search_max_idx": 120,
                "intercept_threshold": 5.0,
                "minmax_threshold": 0.01,
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


def test_synthetic_level1_contract_contains_canonical_atmosphere(tmp_path: Path) -> None:
    path = _write_synthetic_level1(tmp_path / "synthetic_level1_rcs.nc")
    with xr.open_dataset(path) as ds:
        assert ds["Atmospheric_Temperature_K"].dims == ("altitude",)
        assert ds["Atmospheric_Pressure_hPa"].dims == ("altitude",)
        assert "Radiosonde_Temperature_K" not in ds
        assert "Radiosonde_Pressure_hPa" not in ds
        assert ds.attrs["thermodynamic_profile_source_type"] == "ussa76"


def test_lebear_generates_schema4_method5_from_level1_atmosphere(tmp_path: Path) -> None:
    path = _write_synthetic_level1(tmp_path / "synthetic_level1_rcs.nc")
    logger = _ListLogger()

    summary = process_single_level1_file(path, _level2_config(tmp_path), logger)
    output_path = tmp_path / "synthetic_level1_rcs_L2.nc"
    assert summary.results[0].status is ExecutionStatus.OK
    assert output_path.exists()

    with xr.open_dataset(output_path) as ds_l2:
        validate_method_v5_level2_contract(ds_l2)
        assert ds_l2.attrs["level2_product_schema_version"] == LEVEL2_PRODUCT_SCHEMA_VERSION == "4"
        assert ds_l2.attrs["level2_retrieval_method_version"] == LEVEL2_RETRIEVAL_METHOD_VERSION == "5"
        assert ds_l2.attrs["integration_mode"] == "backward"
        assert ds_l2.attrs["uncertainty_method"] == "selection-aware Monte Carlo"
        assert ds_l2.attrs["kfs_reference_boundary_model"] == "beta_total_ref=beta_mol_ref*(1+f)"
        assert ds_l2.attrs["kfs_nominal_aerosol_ref_fraction"] == 0.0
        assert ds_l2.attrs["kfs_beta_ref_relative_std"] == 0.0
        assert ds_l2.attrs["kfs_min_lidar_ratio_sr"] == 10.0
        assert ds_l2.attrs["kfs_allow_negative_aerosol"] == 0
        assert ds_l2.attrs["gluing_selection_score_version"] == "1"
        assert "relative_rmse" in ds_l2.attrs["gluing_selection_score_formula"]

        assert ds_l2["molecular_backscatter"].dims == ("wavelength", "altitude")
        assert np.all(np.isfinite(ds_l2["molecular_backscatter"].values))
        assert set(np.unique(ds_l2["retrieval_success_flag"].values).tolist()) == {1}
        assert ds_l2["aerosol_backscatter_mean"].dims == ("wavelength", "altitude")
        assert ds_l2["aerosol_extinction_mean"].dims == ("wavelength", "altitude")
        assert ds_l2["mc_valid_fraction"].dims == (
            "block_time",
            "wavelength",
            "residual_fraction",
            "altitude",
        )
        assert ds_l2["selected_reference_altitude_m_mc"].dims == (
            "block_time",
            "wavelength",
            "mc_iteration",
        )
        assert ds_l2["effective_vertical_resolution_m"].dims == (
            "wavelength",
            "altitude",
        )
        assert np.asarray(ds_l2["residual_fraction"].values).tolist() == [0.0, 0.02]
        assert np.all(ds_l2["period_support_count"].values >= 0)
        assert ds_l2["altitude"].attrs["units"] == "m"
        assert ds_l2["aerosol_backscatter_mean"].attrs["units"] == "m-1 sr-1"
        assert ds_l2["aerosol_extinction_mean"].attrs["units"] == "m-1"
