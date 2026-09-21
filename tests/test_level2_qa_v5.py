"""Tests for schema-4/method-v5-native Level 2 QA plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from milgrau.viz.level2_qa_v5 import is_method_v5_dataset, plot_all_method_v5_qa


def _config() -> dict:
    return {
        "visualization": {
            "output_format": "png",
            "dpi": 50,
            "altitude_ranges_km": [15.0, 30.0],
            "channels_to_plot": ["355.AN", "532.AN"],
            "quicklook": {
                "show_pbl": True,
                "show_tropopause": True,
                "mean_profile_smooth_bins": 5,
                "max_time_gap_minutes": 60.0,
                "missing_data_color": "white",
                "colormap": "viridis",
            },
            "level2_qa": {
                "enabled": True,
                "generate_gluing_qa": True,
                "generate_molecular_fit_qa": True,
                "generate_scattering_ratio_qa": True,
                "generate_kfs_qa": True,
            },
        },
        "inversion": {
            "molecular_fit": {
                "max_relative_slope": 0.25,
                "max_relative_variance": 0.50,
            }
        },
    }


def _dataset() -> xr.Dataset:
    block_time = np.array(
        ["2020-01-26T21:00", "2020-01-26T21:20", "2020-01-26T21:40"],
        dtype="datetime64[m]",
    )
    wavelength = np.array([532], dtype=np.int32)
    altitude = np.array([500.0, 2000.0, 6000.0, 10000.0, 14000.0, 18000.0])
    n_block = block_time.size
    n_alt = altitude.size
    shape = (n_block, 1, n_alt)
    beta = np.full(shape, 1.5e-6)
    alpha = beta * 55.0
    rcs = np.linspace(10.0, 1.0, n_alt)[None, None, :] * np.ones(shape)
    ref = np.array([[10000.0], [10000.0], [9000.0]])
    path_top = np.array([[18000.0], [18000.0], [14000.0]])
    selected_mc = np.array(
        [
            [[10000.0, 10000.0, 9000.0, 10000.0]],
            [[10000.0, 9000.0, 9000.0, 10000.0]],
            [[9000.0, 9000.0, 8000.0, np.nan]],
        ]
    )
    return xr.Dataset(
        data_vars={
            "range_corrected_signal_block": (
                ("block_time", "wavelength", "altitude"),
                rcs,
            ),
            "aerosol_backscatter_nominal_block": (
                ("block_time", "wavelength", "altitude"),
                beta,
            ),
            "aerosol_extinction_nominal_block": (
                ("block_time", "wavelength", "altitude"),
                alpha,
            ),
            "aerosol_backscatter_mean": (
                ("wavelength", "altitude"),
                np.nanmean(beta, axis=0),
            ),
            "aerosol_extinction_mean": (
                ("wavelength", "altitude"),
                np.nanmean(alpha, axis=0),
            ),
            "period_support_fraction": (
                ("wavelength", "altitude"),
                np.array([[1.0, 1.0, 1.0, 1.0, 2.0 / 3.0, 1.0 / 3.0]]),
            ),
            "retrieval_top_altitude_m": (("wavelength",), np.array([18000.0])),
            "rayleigh_reference_altitude_m_block": (
                ("block_time", "wavelength"),
                ref,
            ),
            "rayleigh_reference_tier_min_altitude_m_block": (
                ("block_time", "wavelength"),
                np.array([[10000.0], [10000.0], [9000.0]]),
            ),
            "rayleigh_reference_fallback_used_block": (
                ("block_time", "wavelength"),
                np.array([[0], [0], [1]], dtype=np.int8),
            ),
            "rayleigh_reference_relative_slope_block": (
                ("block_time", "wavelength"),
                np.array([[0.03], [0.04], [0.08]]),
            ),
            "rayleigh_reference_relative_variance_block": (
                ("block_time", "wavelength"),
                np.array([[0.05], [0.06], [0.09]]),
            ),
            "rayleigh_reference_snr_median_block": (
                ("block_time", "wavelength"),
                np.array([[5.0], [4.0], [3.0]]),
            ),
            "contiguous_path_top_altitude_m_block": (
                ("block_time", "wavelength"),
                path_top,
            ),
            "selection_success_fraction_block": (
                ("block_time", "wavelength"),
                np.array([[0.95], [0.90], [0.75]]),
            ),
            "selected_reference_altitude_m_mc": (
                ("block_time", "wavelength", "mc_iteration"),
                selected_mc,
            ),
            "gluing_attempted_flag": (
                ("block_time", "wavelength"),
                np.ones((n_block, 1), dtype=np.int8),
            ),
            "gluing_success_flag": (
                ("block_time", "wavelength"),
                np.ones((n_block, 1), dtype=np.int8),
            ),
            "gluing_start_altitude_m": (
                ("block_time", "wavelength"),
                np.full((n_block, 1), 1800.0),
            ),
            "gluing_split_altitude_m": (
                ("block_time", "wavelength"),
                np.full((n_block, 1), 2200.0),
            ),
            "gluing_stop_altitude_m": (
                ("block_time", "wavelength"),
                np.full((n_block, 1), 2600.0),
            ),
            "gluing_correlation": (
                ("block_time", "wavelength"),
                np.full((n_block, 1), 0.99),
            ),
            "gluing_relative_rmse": (
                ("block_time", "wavelength"),
                np.full((n_block, 1), 0.02),
            ),
            "gluing_relative_bias": (
                ("block_time", "wavelength"),
                np.full((n_block, 1), 0.01),
            ),
        },
        coords={
            "block_time": block_time,
            "wavelength": wavelength,
            "altitude": altitude,
            "mc_iteration": np.arange(4),
        },
        attrs={
            "level2_product_schema_version": "4",
            "level2_retrieval_method_version": "5",
        },
    )


def test_method_v5_dataset_is_detected() -> None:
    assert is_method_v5_dataset(_dataset())


def test_method_v5_qa_generates_native_panels(tmp_path: Path) -> None:
    generated = plot_all_method_v5_qa(
        ds_l2=_dataset(),
        output_folder=tmp_path,
        file_name_prefix="case",
        config=_config(),
        root_dir=tmp_path,
    )

    assert len(generated) == 4
    assert all(path.is_file() for path in generated)
    names = {path.name for path in generated}
    assert any("Reference" in name for name in names)
    assert any("KFS" in name for name in names)
    assert any("MCReference" in name for name in names)
    assert any("Gluing" in name for name in names)
