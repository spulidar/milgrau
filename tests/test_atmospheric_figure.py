"""Tests for the Level 1 atmospheric comparison figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.viz.atmosphere import plot_atmospheric_evolution, plot_atmospheric_profile


def _config() -> dict:
    return {
        "visualization": {
            "output_format": "png",
            "dpi": 50,
            "altitude_ranges_km": [5.0, 15.0, 30.0],
            "channels_to_plot": ["532.AN"],
            "quicklook": {
                "show_pbl": True,
                "show_tropopause": True,
                "mean_profile_smooth_bins": 5,
                "max_time_gap_minutes": 60.0,
                "missing_data_color": "white",
                "colormap": "viridis",
            },
            "atmospheric_profile": {
                "max_altitude_km": 20.0,
                "evolution_max_altitude_bins": 20,
                "comparison_altitude_bands_km": [
                    [0.0, 3.0],
                    [3.0, 10.0],
                    [10.0, 20.0],
                ],
            },
        }
    }


def _dataset(*, radiosonde: bool = True) -> xr.Dataset:
    altitude = np.arange(0.0, 20_500.0, 500.0)
    atmosphere_time = pd.date_range("2024-06-10T12:00:00", periods=2, freq="1h")
    temperature = 292.0 - 0.0063 * altitude
    pressure = 930.0 * np.exp(-altitude / 8400.0)
    ds = xr.Dataset(
        data_vars={
            "Atmospheric_Temperature_K": (
                ("atmosphere_time", "altitude"),
                np.vstack([temperature, temperature - 0.5]),
            ),
            "Atmospheric_Pressure_hPa": (
                ("atmosphere_time", "altitude"),
                np.vstack([pressure, pressure * 0.995]),
            ),
            "Atmospheric_Source_Type": (
                ("atmosphere_time",),
                np.array(["era5", "era5"], dtype=object),
            ),
            "Atmospheric_USSA76_Fallback_Fraction": (
                ("atmosphere_time",),
                np.array([0.20, 0.20], dtype=np.float64),
            ),
            "Atmospheric_Source_Min_Altitude_ASL_m": (
                ("atmosphere_time",),
                np.array([800.0, 800.0]),
            ),
            "Atmospheric_Source_Max_Altitude_ASL_m": (
                ("atmosphere_time",),
                np.array([16_500.0, 16_500.0]),
            ),
        },
        coords={"atmosphere_time": atmosphere_time, "altitude": altitude},
        attrs={
            "Session_ID": "spu_20240610-1200Z_20240610-1300Z",
            "thermodynamic_station_altitude_m": 760.0,
            "radiosonde_available": "true" if radiosonde else "false",
            "radiosonde_qa_target_datetime_utc": "2024-06-10T12:00:00+00:00",
            "radiosonde_qa_source_profile_min_altitude_asl_m": 820.0,
            "radiosonde_qa_source_profile_max_altitude_asl_m": 16_000.0,
        },
    )
    if radiosonde:
        ds["Radiosonde_QA_Temperature_K"] = (
            ("altitude",),
            temperature + 0.8,
        )
        ds["Radiosonde_QA_Pressure_hPa"] = (
            ("altitude",),
            pressure * 1.005,
        )
    return ds


def test_atmospheric_profile_figure_compares_available_sources(tmp_path: Path) -> None:
    output = plot_atmospheric_profile(
        _dataset(),
        output_folder=tmp_path,
        file_name_prefix="spu_20240610-1200Z_20240610-1300Z",
        config=_config(),
        root_dir=tmp_path,
    )

    assert output.name == (
        "spu_20240610-1200Z_20240610-1300Z_L1_AtmosphericProfile.png"
    )
    assert output.is_file()


def test_atmospheric_profile_figure_still_renders_without_radiosonde(tmp_path: Path) -> None:
    output = plot_atmospheric_profile(
        _dataset(radiosonde=False),
        output_folder=tmp_path,
        file_name_prefix="spu_20240610-1200Z_20240610-1300Z",
        config=_config(),
        root_dir=tmp_path,
    )

    assert output.is_file()



def test_atmospheric_evolution_figure_is_generated(tmp_path: Path) -> None:
    output = plot_atmospheric_evolution(
        _dataset(),
        output_folder=tmp_path,
        file_name_prefix="spu_20240610-1200Z_20240610-1300Z",
        config=_config(),
        root_dir=tmp_path,
    )

    assert output.name == (
        "spu_20240610-1200Z_20240610-1300Z_L1_AtmosphericEvolution.png"
    )
    assert output.is_file()


def test_atmospheric_profile_visible_text_avoids_canonical_wording(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: list[str] = []

    def capture_savefig(self, *args, **kwargs):
        captured.extend(text.get_text() for text in self.texts)
        for axis in self.axes:
            captured.append(axis.get_title())
            captured.append(axis.get_xlabel())
            captured.append(axis.get_ylabel())
            captured.extend(text.get_text() for text in axis.texts)

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", capture_savefig)

    plot_atmospheric_profile(
        _dataset(),
        output_folder=tmp_path,
        file_name_prefix="spu_20240610-1200Z_20240610-1300Z",
        config=_config(),
        root_dir=tmp_path,
    )

    visible = " ".join(captured).lower()
    assert "canonical" not in visible
    assert "l1 source used" in visible
