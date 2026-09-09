"""Tests for LIPANCORA Level 1 correction diagnostics."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.level1.corrections import apply_instrumental_corrections
from milgrau.level1.lipancora import apply_all_physical_corrections


def _kernel_kwargs() -> dict[str, float | None]:
    return {
        "deadtime_min_denominator": 0.05,
        "pc_saturation_max_rate_mhz": None,
    }


def test_apply_instrumental_corrections_marks_shifted_bins_as_nan() -> None:
    time = pd.date_range("2024-01-01", periods=2)
    raw = xr.DataArray(np.arange(10, dtype=np.float64).reshape(2, 5) + 100.0, dims=("time", "range"), coords={"time": time, "range": np.arange(5)})
    z_da = xr.DataArray(np.arange(5, dtype=np.float64) * 7.5, dims=["range"])
    bg_mask = xr.DataArray(np.array([False, False, False, True, True]), dims=["range"])
    corrected, corrected_error, rcs, rcs_error, diagnostics = apply_instrumental_corrections(
        sig=raw, z_da=z_da, shots=1000.0, bin_time_us=0.05, deadtime=0.0,
        shift=2, bg_offset=0.0, is_photon=False, bg_mask=bg_mask,
        return_diagnostics=True, **_kernel_kwargs(),
    )
    assert np.all(np.isnan(corrected.isel(range=slice(0, 2)).values))
    assert np.all(np.isnan(rcs.isel(range=slice(0, 2)).values))
    assert np.allclose(diagnostics["bin_shift_invalid_fraction"].values, np.array([0.4, 0.4]))
    assert corrected_error.shape == corrected.shape
    assert rcs_error.shape == rcs.shape


def test_deadtime_clipping_does_not_claim_physical_saturation_when_uncharacterized() -> None:
    time = pd.date_range("2024-01-01", periods=1)
    raw = xr.DataArray(np.array([[900.0, 900.0, 10.0, 10.0, 10.0]]), dims=("time", "range"), coords={"time": time, "range": np.arange(5)})
    z_da = xr.DataArray(np.arange(5, dtype=np.float64) * 7.5, dims=["range"])
    bg_mask = xr.DataArray(np.array([False, False, False, True, True]), dims=["range"])
    *_signals, diagnostics = apply_instrumental_corrections(
        sig=raw, z_da=z_da, shots=1.0, bin_time_us=1.0, deadtime=0.0035,
        shift=0, bg_offset=0.0, is_photon=True, bg_mask=bg_mask,
        deadtime_min_denominator=0.05, pc_saturation_max_rate_mhz=None,
        return_diagnostics=True,
    )
    assert diagnostics["deadtime_correction_applied"] is True
    assert np.isclose(float(diagnostics["deadtime_clipping_fraction"].values[0]), 0.4)
    assert np.isclose(float(diagnostics["pc_saturation_fraction"].values[0]), 0.0)
    assert not np.any(diagnostics["pc_saturation_mask"].values)
    assert diagnostics["pc_saturation_characterized"] is False
    assert np.isnan(diagnostics["pc_saturation_rate_limit_mhz"])
    assert np.isclose(diagnostics["deadtime_min_denominator_allowed"], 0.05)


def test_characterized_pc_saturation_uses_physical_rate_limit() -> None:
    time = pd.date_range("2024-01-01", periods=1)
    raw = xr.DataArray(np.array([[10.0, 20.0, 30.0, 40.0]]), dims=("time", "range"), coords={"time": time, "range": np.arange(4)})
    z_da = xr.DataArray(np.arange(4, dtype=np.float64) * 7.5, dims=["range"])
    bg_mask = xr.DataArray(np.array([False, False, True, True]), dims=["range"])
    *_signals, diagnostics = apply_instrumental_corrections(
        sig=raw, z_da=z_da, shots=10.0, bin_time_us=0.5, deadtime=0.0,
        shift=0, bg_offset=0.0, is_photon=True, bg_mask=bg_mask,
        deadtime_min_denominator=0.05, pc_saturation_max_rate_mhz=5.0,
        return_diagnostics=True,
    )
    assert diagnostics["pc_saturation_characterized"] is True
    assert np.isclose(diagnostics["pc_saturation_rate_limit_mhz"], 5.0)
    expected_rate = raw / 5.0
    np.testing.assert_array_equal(diagnostics["pc_saturation_mask"].values, (expected_rate >= 5.0).values)


def test_apply_instrumental_corrections_converts_pc_counts_to_mhz_deterministically() -> None:
    time = pd.date_range("2024-01-01", periods=1)
    raw = xr.DataArray(np.array([[10.0, 20.0, 30.0, 40.0]]), dims=("time", "range"), coords={"time": time, "range": np.arange(4)})
    z_da = xr.DataArray(np.arange(4, dtype=np.float64) * 7.5, dims=["range"])
    bg_mask = xr.DataArray(np.array([False, False, True, True]), dims=["range"])
    corrected, *_ = apply_instrumental_corrections(
        sig=raw, z_da=z_da, shots=10.0, bin_time_us=0.5, deadtime=0.0,
        shift=0, bg_offset=0.0, is_photon=True, bg_mask=bg_mask,
        return_diagnostics=True, **_kernel_kwargs(),
    )
    expected_mhz = raw / 5.0
    expected_corrected = expected_mhz - expected_mhz.where(bg_mask).mean(dim="range", skipna=True)
    assert np.allclose(corrected.values, expected_corrected.values)


def test_pc_poisson_uncertainty_uses_raw_counts_before_dark_subtraction() -> None:
    """Shot noise must be sqrt(N), not sqrt(N-D), for observed PC counts."""
    time = pd.date_range("2024-01-01", periods=1)
    raw = xr.DataArray(
        np.full((1, 4), 100.0, dtype=np.float64),
        dims=("time", "range"),
        coords={"time": time, "range": np.arange(4)},
    )
    dark = xr.DataArray(np.full(4, 40.0, dtype=np.float64), dims=["range"], coords={"range": np.arange(4)})
    z_da = xr.DataArray((np.arange(4, dtype=np.float64) + 1.0) * 7.5, dims=["range"])
    bg_mask = xr.DataArray(np.array([False, False, True, True]), dims=["range"])

    _, corrected_error, _, _ = apply_instrumental_corrections(
        sig=raw,
        z_da=z_da,
        shots=10.0,
        bin_time_us=0.5,
        deadtime=0.0,
        shift=0,
        bg_offset=0.0,
        is_photon=True,
        bg_mask=bg_mask,
        dc_prof=dark,
        **_kernel_kwargs(),
    )

    rate_scale = 10.0 * 0.5
    expected_poisson_mhz = np.sqrt(100.0) / rate_scale
    biased_dark_subtracted_value = np.sqrt(100.0 - 40.0) / rate_scale
    assert np.allclose(corrected_error.values, expected_poisson_mhz)
    assert not np.isclose(float(corrected_error.values[0, 0]), biased_dark_subtracted_value)


def test_pc_dark_current_uncertainty_is_independent_quadrature_term() -> None:
    """Dark-profile uncertainty is added independently to raw-count Poisson noise."""
    time = pd.date_range("2024-01-01", periods=1)
    raw = xr.DataArray(
        np.full((1, 4), 100.0, dtype=np.float64),
        dims=("time", "range"),
        coords={"time": time, "range": np.arange(4)},
    )
    dark = xr.DataArray(np.full(4, 40.0, dtype=np.float64), dims=["range"], coords={"range": np.arange(4)})
    dark_error = xr.DataArray(np.full(4, 3.0, dtype=np.float64), dims=["range"], coords={"range": np.arange(4)})
    z_da = xr.DataArray((np.arange(4, dtype=np.float64) + 1.0) * 7.5, dims=["range"])
    bg_mask = xr.DataArray(np.array([False, False, True, True]), dims=["range"])

    _, corrected_error, _, _ = apply_instrumental_corrections(
        sig=raw,
        z_da=z_da,
        shots=10.0,
        bin_time_us=0.5,
        deadtime=0.0,
        shift=0,
        bg_offset=0.0,
        is_photon=True,
        bg_mask=bg_mask,
        dc_prof=dark,
        dc_err=dark_error,
        **_kernel_kwargs(),
    )

    rate_scale = 10.0 * 0.5
    expected_mhz = np.sqrt(100.0 + 3.0**2) / rate_scale
    old_biased_mhz = np.sqrt((100.0 - 40.0) + 3.0**2) / rate_scale
    assert np.allclose(corrected_error.values, expected_mhz)
    assert not np.isclose(float(corrected_error.values[0, 0]), old_biased_mhz)


def test_apply_all_physical_corrections_persists_distinct_diagnostics() -> None:
    time = pd.date_range("2024-01-01", periods=2)
    altitude = (np.arange(5, dtype=np.float64) + 0.5) * 7.5
    channels = np.array(["532.PC", "532.AN"], dtype=object)
    raw_data = np.ones((2, 2, 5), dtype=np.float64) * 10.0
    raw_data[:, 0, :2] = 900.0
    ds = xr.Dataset(
        data_vars={
            "Raw_Lidar_Data": (("time", "channel", "altitude"), raw_data),
            "Background_Low": (("channel",), np.array([18.0, 18.0])),
            "Background_High": (("channel",), np.array([34.0, 34.0])),
            "Raw_Data_Range_Resolution": (("channel",), np.array([7.5, 7.5])),
            "Laser_Shots": (("time", "channel"), np.ones((2, 2), dtype=np.int32)),
        },
        coords={"time": time, "channel": channels, "altitude": altitude},
        attrs={"Accumulated_Shots": 1},
    )
    config = {
        "level1": {
            "background": {"start_altitude_m": 18.0, "stop_altitude_m": 34.0},
            "photon_counting": {"deadtime_min_denominator": 0.05},
            "pbl": {"reference_channel": "532.AN", "min_search_altitude_m": 5.0, "max_search_altitude_m": 30.0, "smooth_bins": 3},
        },
        "_resolved_station": {
            "calibration_id": "test-calibration",
            "channel_calibrations": {
                "532.PC": {
                    "detector_mode": "photon_counting",
                    "deadtime_us": 0.0035,
                    "bin_shift_bins": 0,
                    "background_offset": 0.0,
                    "saturation": {"status": "not_characterized"},
                },
                "532.AN": {
                    "detector_mode": "analog",
                    "deadtime_us": 0.0,
                    "bin_shift_bins": 2,
                    "background_offset": 0.0,
                },
            },
        },
    }
    result = apply_all_physical_corrections(ds, altitude, config, logging.getLogger("test"))
    assert "deadtime_clipping_fraction" in result
    assert "pc_saturation_mask" in result
    assert "pc_saturation_fraction" in result
    assert "pc_saturation_characterized" in result
    assert "pc_saturation_rate_limit_mhz" in result
    assert "deadtime_correction_applied" in result
    assert "bin_shift_invalid_fraction" in result
    assert "bin_shift_bins" in result
    assert int(result["deadtime_correction_applied"].sel(channel="532.PC")) == 1
    assert int(result["deadtime_correction_applied"].sel(channel="532.AN")) == 0
    assert float(result["deadtime_clipping_fraction"].sel(channel="532.PC").max()) > 0.0
    assert float(result["pc_saturation_fraction"].sel(channel="532.PC").max()) == 0.0
    assert int(result["pc_saturation_characterized"].sel(channel="532.PC")) == 0
    assert np.isnan(float(result["pc_saturation_rate_limit_mhz"].sel(channel="532.PC")))
    assert int(result["pc_saturation_mask"].sel(channel="532.PC").max()) == 0
    assert int(result["pc_saturation_mask"].sel(channel="532.AN").max()) == 0
    assert np.isclose(float(result["bin_shift_invalid_fraction"].sel(channel="532.AN").max()), 0.4)
    assert np.all(np.isnan(result["corrected_signal"].sel(channel="532.AN").isel(altitude=slice(0, 2)).values))
