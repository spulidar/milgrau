"""Tests for the current Level 2 figures and optical-profile plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from milgrau.viz import level2_figures
from milgrau.viz.level2 import (
    is_level2_dataset,
    plot_all_level2_figures,
    plot_optical_profiles,
)


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
            "level2_figures": {
                "enabled": True,
                "max_altitude_km": 30.0,
                "smooth_bins": 5,
                "generate_gluing": True,
                "generate_molecular_reference": True,
                "generate_mc_reference": True,
                "generate_optical_profiles": True,
            },
        },
        "inversion": {
            "molecular_fit": {
                "ref_alt_min_m": 8000.0,
                "ref_alt_max_m": 12000.0,
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
    beta_mol = 8.0e-7 * np.exp(-altitude / 8000.0)
    alpha_mol = beta_mol * (8.0 * np.pi / 3.0)
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
            "aerosol_backscatter_mc_std": (
                ("block_time", "wavelength", "residual_fraction", "altitude"),
                (0.15 * beta)[:, :, None, :],
            ),
            "aerosol_extinction_mc_std": (
                ("block_time", "wavelength", "residual_fraction", "altitude"),
                (0.15 * alpha)[:, :, None, :],
            ),
            "molecular_backscatter": (
                ("block_time", "wavelength", "altitude"),
                np.broadcast_to(beta_mol[None, None, :], shape),
            ),
            "molecular_extinction": (
                ("block_time", "wavelength", "altitude"),
                np.broadcast_to(alpha_mol[None, None, :], shape),
            ),
            "solar_elevation_deg": (
                ("block_time",),
                np.array([-15.0, -12.0, -9.0]),
            ),
            "Segment_Label": (
                ("segments",),
                np.array(["seg00"], dtype=object),
            ),
            "Segment_Regime": (
                ("segments",),
                np.array(["night"], dtype=object),
            ),
            "Segment_Start_Time_UTC": (
                ("segments",),
                np.array([1579986000], dtype=np.int64),
            ),
            "Segment_End_Time_UTC": (
                ("segments",),
                np.array([1579989600], dtype=np.int64),
            ),
            "period_support_fraction": (
                ("wavelength", "altitude"),
                np.array([[1.0, 1.0, 1.0, 1.0, 2.0 / 3.0, 1.0 / 3.0]]),
            ),
            "retrieval_top_altitude_m": (("wavelength",), np.array([18000.0])),
            "kfs_forward_endpoint_altitude_m": (
                ("block_time", "wavelength"),
                path_top,
            ),
            "rayleigh_reference_altitude_m_block": (
                ("block_time", "wavelength"),
                ref,
            ),
            "rayleigh_reference_search_min_altitude_m_block": (
                ("block_time", "wavelength"),
                np.array([[10000.0], [10000.0], [9000.0]]),
            ),
            "rayleigh_reference_search_max_altitude_m_block": (
                ("block_time", "wavelength"),
                np.array([[15000.0], [15000.0], [20000.0]]),
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
            "rayleigh_background_offset_block": (
                ("block_time", "wavelength"),
                np.array([[0.03], [0.02], [0.04]]),
            ),
            "rayleigh_calibration_background_correlation_block": (
                ("block_time", "wavelength"),
                np.array([[-0.30], [-0.25], [-0.35]]),
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
            "gluing_slope": (
                ("block_time", "wavelength"),
                np.full((n_block, 1), 1.2),
            ),
            "gluing_intercept": (
                ("block_time", "wavelength"),
                np.zeros((n_block, 1)),
            ),
        },
        coords={
            "block_time": block_time,
            "segment_id": (("block_time",), np.array(["seg00", "seg00", "seg00"], dtype=object)),
            "solar_regime": (("block_time",), np.array(["night", "night", "night"], dtype=object)),
            "wavelength": wavelength,
            "altitude": altitude,
            "mc_iteration": np.arange(4),
            "residual_fraction": np.array([0.0]),
        },
        attrs={
            "level2_product_schema_version": "8",
            "Solar_Day_Night_Threshold_deg": -3.0,
            "Solar_Position_Algorithm": "test",
            "Segment_Count": 1,
        },
    )


def _level1_dataset() -> xr.Dataset:
    altitude = np.array([500.0, 2000.0, 6000.0, 10000.0, 14000.0, 18000.0])
    time = np.array(
        ["2020-01-26T21:00", "2020-01-26T21:20", "2020-01-26T21:40"],
        dtype="datetime64[m]",
    )
    base = np.linspace(8.0, 0.8, altitude.size)
    analog = np.stack([base * 0.82, base * 0.80, base * 0.84])
    photon = np.stack([base * 1.01, base, base * 0.99])
    signal = np.stack([analog, photon], axis=1)
    return xr.Dataset(
        data_vars={
            "range_corrected_signal": (
                ("time", "channel", "altitude"),
                signal,
            )
        },
        coords={
            "time": time,
            "channel": np.array(["532.AN", "532.PC"]),
            "altitude": altitude,
        },
    )


def test_level2_dataset_is_detected() -> None:
    assert is_level2_dataset(_dataset())


def test_level2_figures_generates_native_panels(tmp_path: Path) -> None:
    generated = plot_all_level2_figures(
        ds_l2=_dataset(),
        output_folder=tmp_path,
        file_name_prefix="case",
        config=_config(),
        root_dir=tmp_path,
        ds_l1=_level1_dataset(),
    )

    assert len(generated) == 4
    assert all(path.is_file() for path in generated)
    names = {path.name for path in generated}
    assert any("MolecularReference" in name for name in names)
    assert any("OpticalProfiles" in name for name in names)
    assert any("MCReference" in name for name in names)
    assert any("SignalProfile" in name for name in names)


def test_optical_profiles_omit_support_panel_and_keep_retrieval_top_line(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def capture_figure(fig, output_folder, name, dpi):
        captured["titles"] = [axis.get_title() for axis in fig.axes]
        beta_axis = next(
            axis for axis in fig.axes if axis.get_title() == "Aerosol backscatter"
        )
        captured["black_dashed_altitudes"] = [
            np.asarray(line.get_ydata(), dtype=float)
            for line in beta_axis.lines
            if line.get_color() == "black" and line.get_linestyle() == "--"
        ]
        plt.close(fig)
        return Path(output_folder) / name

    monkeypatch.setattr(level2_figures, "_save", capture_figure)
    output = plot_optical_profiles(
        _dataset(), 532, tmp_path, "case", _config(), tmp_path
    )

    assert output is not None
    assert not any("Retrieval support" in title for title in captured["titles"])
    assert any(
        np.allclose(altitudes, [18.0, 18.0])
        for altitudes in captured["black_dashed_altitudes"]
    )
