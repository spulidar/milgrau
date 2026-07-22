"""Synthetic NetCDF contract tests for Level 1 and LEBEAR inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.level2.lebear import process_single_level1_file
from milgrau.operations import ExecutionStatus


EXPECTED_LEVEL2_DATA_VARS = frozenset(
    """
    aerosol_backscatter aerosol_backscatter_block aerosol_backscatter_error
    aerosol_backscatter_error_block aerosol_backscatter_mean aerosol_backscatter_mean_error
    aerosol_extinction aerosol_extinction_block aerosol_extinction_error
    aerosol_extinction_error_block aerosol_extinction_mean aerosol_extinction_mean_error
    glued_corrected_signal glued_corrected_signal_block glued_corrected_signal_error
    glued_corrected_signal_error_block glued_corrected_signal_error_mean glued_corrected_signal_mean
    glued_range_corrected_signal glued_range_corrected_signal_block glued_range_corrected_signal_error
    glued_range_corrected_signal_error_block glued_range_corrected_signal_error_mean
    glued_range_corrected_signal_mean gluing_attempted_flag gluing_attempted_flag_block
    gluing_correlation gluing_correlation_block gluing_intercept gluing_intercept_block
    gluing_merge_source_flag gluing_merge_source_flag_block gluing_relative_bias
    gluing_relative_bias_block gluing_relative_rmse gluing_relative_rmse_block gluing_slope
    gluing_slope_block gluing_split_altitude_m gluing_split_altitude_m_block
    gluing_start_altitude_m gluing_start_altitude_m_block gluing_stop_altitude_m
    gluing_stop_altitude_m_block gluing_success_flag gluing_success_flag_block
    kfs_backward_valid_flag kfs_backward_valid_flag_block kfs_branch kfs_branch_block
    kfs_forward_valid_flag kfs_forward_valid_flag_block lidar_ratio_assumed_sr lidar_ratio_std_sr molecular_backscatter
    molecular_extinction molecular_transmission rayleigh_calibration_factor
    rayleigh_calibration_factor_block rayleigh_calibration_intercept
    rayleigh_calibration_intercept_block rayleigh_reference_altitude_m
    rayleigh_reference_altitude_m_block rayleigh_reference_relative_slope
    rayleigh_reference_relative_slope_block rayleigh_reference_relative_variance
    rayleigh_reference_relative_variance_block rayleigh_reference_start_altitude_m
    rayleigh_reference_start_altitude_m_block rayleigh_reference_stop_altitude_m
    rayleigh_reference_stop_altitude_m_block rayleigh_reference_success_flag
    rayleigh_reference_success_flag_block rayleigh_reference_valid_bins
    rayleigh_reference_valid_bins_block rayleigh_reference_valid_fraction
    rayleigh_reference_valid_fraction_block scaled_molecular_range_corrected_signal
    scaled_molecular_range_corrected_signal_block scattering_ratio_block scattering_ratio_mean
    retrieval_input_invalid_reason retrieval_input_invalid_reason_block
    retrieval_input_snr_median retrieval_input_snr_median_block
    retrieval_input_valid_flag retrieval_input_valid_flag_block retrieval_success_flag retrieval_success_fraction
    requested_wavelengths processed_wavelengths failed_wavelengths
    failed_wavelength_stage failed_wavelength_code failed_wavelength_message failed_wavelength_cause
    signal_source_flag signal_source_flag_block simulated_molecular_range_corrected_signal
    simulated_molecular_signal single_channel_fallback_flag single_channel_fallback_flag_block
    """.split()
)


class _ListLogger:
    """Small logger stub used to capture pipeline messages in tests."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(f"INFO: {message}")

    def warning(self, message: str) -> None:
        self.messages.append(f"WARNING: {message}")

    def error(self, message: str) -> None:
        self.messages.append(f"ERROR: {message}")


def _write_synthetic_level1(path: Path) -> Path:
    """Write a tiny Level 1 NetCDF product for contract testing."""
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

    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), corrected_signal),
            "corrected_signal_error": (("time", "channel", "altitude"), corrected_signal_error),
            "range_corrected_signal": (("time", "channel", "altitude"), range_corrected_signal),
            "range_corrected_signal_error": (("time", "channel", "altitude"), range_corrected_signal_error),
            "pc_saturation_mask": (("time", "channel", "altitude"), np.zeros(shape, dtype=np.int8)),
            "channel_correction_success": (("channel",), np.ones(channel.size, dtype=np.int8)),
            "PBL_Height_km": (("time",), np.array([0.8, 0.9, 1.0], dtype=np.float32)),
        },
        coords={"time": time, "channel": channel, "altitude": altitude},
        attrs={
            "Processing_level": "Level 1 synthetic test product",
            "Altitude_units": "m",
            "tropopause_cpt_km": -999.0,
            "tropopause_lrt_km": -999.0,
        },
    )
    ds["corrected_signal"].attrs["units"] = "channel native corrected units"
    ds["corrected_signal_error"].attrs["units"] = "channel native corrected units"
    ds["range_corrected_signal"].attrs["units"] = "a.u. m^2"
    ds["range_corrected_signal_error"].attrs["units"] = "a.u. m^2"
    ds["altitude"].attrs["units"] = "m"
    ds.to_netcdf(path)
    return path


def test_synthetic_level1_contract(tmp_path: Path) -> None:
    """A synthetic Level 1 file should expose the variables expected by LEBEAR."""
    path = _write_synthetic_level1(tmp_path / "synthetic_level1_rcs.nc")

    with xr.open_dataset(path) as ds:
        assert "corrected_signal" in ds
        assert "corrected_signal_error" in ds
        assert "range_corrected_signal" in ds
        assert "range_corrected_signal_error" in ds
        assert "altitude" in ds.coords
        assert ds["range_corrected_signal"].dims == ("time", "channel", "altitude")
        assert ds["range_corrected_signal_error"].shape == ds["range_corrected_signal"].shape
        assert float(ds["altitude"].max()) > 100.0


def test_lebear_generates_synthetic_level2_product(tmp_path: Path) -> None:
    """Freeze the compact retrieval-to-NetCDF schema and deterministic reference values."""
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
        assert set(ds_l2.data_vars) == EXPECTED_LEVEL2_DATA_VARS
        assert dict(ds_l2.sizes) == {
            "wavelength": 1,
            "altitude": 200,
            "block_time": 1,
            "time": 3,
            "requested_wavelength": 1,
            "processed_wavelength": 1,
            "failed_wavelength": 0,
        }
        assert ds_l2["molecular_backscatter"].dims == ("wavelength", "altitude")
        assert ds_l2["glued_corrected_signal"].dims == ("time", "wavelength", "altitude")
        assert ds_l2["glued_corrected_signal_block"].dims == ("block_time", "wavelength", "altitude")
        assert ds_l2["retrieval_success_flag"].dims == ("block_time", "wavelength")
        assert ds_l2["molecular_backscatter"].dtype == np.dtype("float64")
        assert ds_l2["gluing_success_flag"].dtype == np.dtype("int8")
        assert ds_l2["kfs_branch"].dtype == np.dtype("int8")
        assert int(ds_l2["wavelength"].values[0]) == 532
        assert ds_l2.attrs["Processing_level"] == "Level 2: LEBEAR block-based optical inversion"
        assert ds_l2.attrs["Pipeline"] == "MILGRAU/LEBEAR"
        assert ds_l2.attrs["Input_Level1_File"] == path.name
        assert ds_l2.attrs["LEBEAR_Mode"] == "block_mean_signal_selection_rayleigh_kfs"
        assert ds_l2.attrs["KFS_Mode"] == "two_sided"
        assert ds_l2.attrs["elastic_backscatter_inversion_method"] == "Klett-Fernald-Sasano"
        assert ds_l2.attrs["integration_mode"] == "two_sided"
        assert ds_l2.attrs["uncertainty_method"] == "Monte Carlo"
        assert ds_l2.attrs["fernald_implementation_version"] == "2"
        assert ds_l2.attrs["scientific_change"] == "corrected_backward_molecular_factor_sign"
        assert ds_l2["gluing_success_flag"].attrs["flag_values"] == "0, 1"
        assert ds_l2["gluing_success_flag"].attrs["flag_meanings"] == "failed_or_not_attempted approved"
        assert ds_l2["kfs_branch"].attrs["flag_values"] == "0, 1, 2, 3"
        assert set(np.unique(ds_l2["kfs_branch"].values).tolist()) == {1, 2, 3}
        assert set(np.unique(ds_l2["retrieval_success_flag"].values).tolist()) == {1}

        # Tight relative tolerances allow minor platform-level floating-point variation
        # while detecting changes to the molecular model, gluing, or retrieval output.
        np.testing.assert_allclose(
            ds_l2["molecular_backscatter"].values[0, :5],
            np.array(
                [
                    1.438635095291557e-06,
                    1.4375814972649463e-06,
                    1.4365284896434596e-06,
                    1.4354760721978301e-06,
                    1.4344242446988414e-06,
                ]
            ),
            rtol=1e-10,
            atol=0.0,
        )
        np.testing.assert_allclose(
            ds_l2["glued_corrected_signal"].values[0, 0, :5],
            np.array([1.15000002712507, 1.1414073077465436, 1.1328787301905783, 1.1244139205734942, 1.1160123803837163]),
            rtol=1e-7,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.nansum(ds_l2["aerosol_backscatter"].values),
            3.835861304163948e-06,
            rtol=1e-5,
            atol=1e-12,
        )
