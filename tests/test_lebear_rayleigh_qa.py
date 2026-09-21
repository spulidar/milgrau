"""Tests for LEBEAR Rayleigh reference quality diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.io.paths import level2_output_path
from milgrau.level2 import lebear
from milgrau.level2.optical_retrieval import evaluate_rayleigh_reference
from milgrau.operations import ExecutionStatus


def _logger() -> logging.Logger:
    logger = logging.getLogger("test.lebear.rayleigh_qa")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def test_rayleigh_reference_qa_accepts_flat_ratio() -> None:
    altitude = np.arange(100, dtype=np.float64) * 7.5
    simulated = np.exp(-altitude / 9000.0) + 1.0
    measured = simulated * 42.0
    fit_config = {
        "max_relative_slope": 0.05,
        "max_relative_variance": 0.10,
        "min_valid_fraction": 0.50,
    }

    qa = evaluate_rayleigh_reference(
        measured_signal=measured,
        simulated_molecular_signal=simulated,
        altitude_m=altitude,
        reference_center_idx=50,
        reference_window_bins=20,
        fit_config=fit_config,
        calibration_factor=42.0,
    )

    assert qa["success_flag"] == 1
    assert float(qa["relative_slope"]) <= fit_config["max_relative_slope"]
    assert float(qa["relative_variance"]) <= fit_config["max_relative_variance"]
    assert float(qa["valid_fraction"]) == 1.0


def test_rayleigh_reference_qa_rejects_sloped_ratio() -> None:
    altitude = np.arange(100, dtype=np.float64) * 7.5
    simulated = np.ones_like(altitude)
    measured = 1.0 + altitude / np.nanmax(altitude)
    fit_config = {
        "max_relative_slope": 0.05,
        "max_relative_variance": 10.0,
        "min_valid_fraction": 0.50,
    }

    qa = evaluate_rayleigh_reference(
        measured_signal=measured,
        simulated_molecular_signal=simulated,
        altitude_m=altitude,
        reference_center_idx=50,
        reference_window_bins=40,
        fit_config=fit_config,
        calibration_factor=1.0,
    )

    assert qa["success_flag"] == 0
    assert float(qa["relative_slope"]) > fit_config["max_relative_slope"]


def _write_level1(path: Path) -> Path:
    """Write a strict synthetic Level 1 file for Rayleigh QA product tests."""
    time = pd.date_range("2024-01-01T00:00:00", periods=2, freq="5min")
    altitude = np.arange(240, dtype=np.float64) * 7.5
    channel = np.array(["532.AN", "532.PC"], dtype=object)
    base = np.exp(-altitude / 1200.0) + 0.2
    analog = np.tile(base, (time.size, 1))
    photon = analog * 1.02
    rcs = np.stack([analog, photon], axis=1).astype(np.float32)
    rcs_error = np.abs(rcs * 0.02).astype(np.float32)
    temperature_k = 288.15 - 0.0065 * altitude
    pressure_hpa = 1013.25 * np.exp(-altitude / 8434.0)

    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), rcs.copy()),
            "corrected_signal_error": (
                ("time", "channel", "altitude"),
                rcs_error.copy(),
            ),
            "range_corrected_signal": (("time", "channel", "altitude"), rcs),
            "range_corrected_signal_error": (
                ("time", "channel", "altitude"),
                rcs_error,
            ),
            "pc_saturation_mask": (
                ("time", "channel", "altitude"),
                np.zeros_like(rcs, dtype=np.int8),
            ),
            "pc_saturation_characterized": (
                ("channel",),
                np.array([0, 1], dtype=np.int8),
            ),
            "channel_correction_success": (
                ("channel",),
                np.ones(channel.size, dtype=np.int8),
            ),
            "Atmospheric_Temperature_K": (
                ("altitude",),
                temperature_k.astype(np.float64),
            ),
            "Atmospheric_Pressure_hPa": (
                ("altitude",),
                pressure_hpa.astype(np.float64),
            ),
        },
        coords={"time": time, "channel": channel, "altitude": altitude},
        attrs={
            "Processing_level": "Level 1 synthetic Rayleigh QA test product",
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
            "kfs_mode": "backward",
            "monte_carlo_iterations": 5,
            "random_seed": 123,
            "beta_ref_relative_std": 0.0,
            "aerosol_ref_fraction": 0.0,
            "min_lidar_ratio_sr": 10.0,
            "allow_negative_aerosol": False,
            "molecular_fit": {
                "ref_alt_min_m": 500.0,
                "ref_alt_max_m": 1500.0,
                "ref_window_m": 150.0,
                "max_relative_slope": 10.0,
                "max_relative_variance": 10.0,
                "min_valid_fraction": 0.10,
            },
            "method_v5": {
                "reference_tier_min_altitudes_m": [1200.0, 900.0, 600.0],
                "reference_search_max_m": 1500.0,
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
                "minmax_threshold": 0.05,
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


def test_level2_saves_method_v5_selected_reference_qa_variables(tmp_path: Path) -> None:
    level1 = _write_level1(tmp_path / "20240101_spu_00_L1.nc")

    summary = lebear.process_single_level1_file(level1, _config(tmp_path), _logger())

    assert summary.results[0].status is ExecutionStatus.OK
    with xr.open_dataset(level2_output_path(level1)) as ds:
        required = {
            "rayleigh_reference_altitude_m_block",
            "rayleigh_reference_relative_slope_block",
            "rayleigh_reference_relative_variance_block",
            "rayleigh_reference_valid_fraction_block",
            "rayleigh_reference_diagnostic_cost_block",
            "rayleigh_reference_snr_median_block",
            "rayleigh_reference_effective_resolution_m_block",
            "rayleigh_reference_source_bin_count_block",
            "rayleigh_reference_tier_min_altitude_m_block",
            "rayleigh_reference_fallback_used_block",
        }
        assert required <= set(ds.data_vars)
        assert ds.attrs["level2_retrieval_method_version"] == "5"
        assert ds.attrs["integration_mode"] == "backward"
        assert "highest_supported_declared_tier" in ds.attrs[
            "reference_selection_policy"
        ]
        success = ds["retrieval_success_flag"].isel(block_time=0, wavelength=0).item()
        assert int(success) == 1
        assert float(
            ds["rayleigh_reference_valid_fraction_block"].isel(
                block_time=0, wavelength=0
            )
        ) >= 0.0
        assert float(
            ds["rayleigh_reference_effective_resolution_m_block"].isel(
                block_time=0, wavelength=0
            )
        ) > 0.0
        assert int(
            ds["rayleigh_reference_source_bin_count_block"].isel(
                block_time=0, wavelength=0
            )
        ) > 0
        cost = float(
            ds["rayleigh_reference_diagnostic_cost_block"].isel(
                block_time=0, wavelength=0
            )
        )
        slope = float(
            ds["rayleigh_reference_relative_slope_block"].isel(
                block_time=0, wavelength=0
            )
        )
        variance = float(
            ds["rayleigh_reference_relative_variance_block"].isel(
                block_time=0, wavelength=0
            )
        )
        assert np.isclose(cost, slope + variance)
