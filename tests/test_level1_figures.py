"""Tests for Level 1 figure ownership and reusable quicklook behavior."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from milgrau.level1 import figures as level1_figures
from milgrau.operations import ExecutionStatus
from milgrau.viz import quicklooks
from milgrau.viz.quicklooks import (
    _decimate_for_display,
    _insert_time_gap_markers,
    _streaming_time_statistics,
)

SESSION_ID = "spu_20240101-0000Z_20240101-0010Z"


class _ListLogger(logging.Logger):
    def __init__(self) -> None:
        super().__init__("test.level1.figures", level=logging.DEBUG)
        self.messages: list[str] = []
        self.propagate = False

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        rendered = str(msg) % args if args else str(msg)
        self.messages.append(f"{logging.getLevelName(level)}: {rendered}")


def _write_level1(path: Path, channels: list[str]) -> Path:
    time = pd.date_range("2024-01-01T00:00:00", periods=3, freq="5min")
    altitude = np.arange(0.0, 1500.0, 7.5)
    channel = np.array(channels, dtype=object)
    shape = (time.size, channel.size, altitude.size)
    profile = np.exp(-altitude / 1000.0).astype(np.float32)
    corrected = np.zeros(shape, dtype=np.float32)
    corrected_error = np.zeros(shape, dtype=np.float32)
    rcs = np.zeros(shape, dtype=np.float32)
    rcs_error = np.zeros(shape, dtype=np.float32)
    for time_idx in range(time.size):
        for channel_idx in range(channel.size):
            scale = 1.0 + 0.1 * time_idx + 0.05 * channel_idx
            corrected[time_idx, channel_idx, :] = scale * profile
            corrected_error[time_idx, channel_idx, :] = 0.05 * corrected[time_idx, channel_idx, :]
            rcs[time_idx, channel_idx, :] = corrected[time_idx, channel_idx, :] * altitude.astype(np.float32) ** 2
            rcs_error[time_idx, channel_idx, :] = corrected_error[time_idx, channel_idx, :] * altitude.astype(np.float32) ** 2

    temperature_k = 288.15 - 0.0065 * altitude
    pressure_hpa = 1013.25 * np.exp(-altitude / 8434.0)
    ds = xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), corrected),
            "corrected_signal_error": (("time", "channel", "altitude"), corrected_error),
            "range_corrected_signal": (("time", "channel", "altitude"), rcs),
            "range_corrected_signal_error": (("time", "channel", "altitude"), rcs_error),
            "PBL_Height_km": (("time",), np.array([0.7, 0.8, 0.9], dtype=np.float32)),
            "Atmospheric_Temperature_K": (
                ("atmosphere_time", "altitude"),
                np.vstack([temperature_k, temperature_k]).astype(np.float64),
            ),
            "Atmospheric_Pressure_hPa": (
                ("atmosphere_time", "altitude"),
                np.vstack([pressure_hpa, pressure_hpa]).astype(np.float64),
            ),
            "Atmospheric_Source_Type": (
                ("atmosphere_time",),
                np.array(["ussa76", "ussa76"], dtype=object),
            ),
            "Atmospheric_Source_Time_Delta_hours": (
                ("atmosphere_time",),
                np.array([np.nan, np.nan]),
            ),
            "Atmospheric_USSA76_Fallback_Fraction": (
                ("atmosphere_time",),
                np.array([1.0, 1.0]),
            ),
            "solar_elevation_deg": (("time",), np.array([-30.0, -29.0, -28.0])),
            "solar_regime": (("time",), np.array(["night", "night", "night"], dtype=object)),
            "segment_id": (("time",), np.array(["seg00", "seg00", "seg00"], dtype=object)),
            "Segment_Label": (("segments",), np.array(["seg00"], dtype=object)),
            "Segment_Regime": (("segments",), np.array(["night"], dtype=object)),
            "Segment_Start_Time_UTC": (("segments",), np.array([1704067200], dtype=np.int64)),
            "Segment_End_Time_UTC": (("segments",), np.array([1704068100], dtype=np.int64)),
        },
        coords={
            "time": time,
            "channel": channel,
            "altitude": altitude,
            "atmosphere_time": pd.date_range("2024-01-01T00:00:00", periods=2, freq="1h"),
        },
        attrs={
            "Session_ID": SESSION_ID,
            "tropopause_cpt_km": np.nan,
            "tropopause_lrt_km": np.nan,
            "thermodynamic_profile_source_type": "time_resolved",
            "thermodynamic_profile_available": "true",
            "thermodynamic_profile_standard_fallback_fraction": 1.0,
            "Solar_Day_Night_Threshold_deg": -3.0,
            "Solar_Position_Algorithm": "test",
            "Segment_Count": 1,
        },
    )
    ds.to_netcdf(path)
    return path


def _config(channels: list[str], incremental: bool = True, config_file: Path | None = None) -> dict:
    config = {
        "processing": {"incremental": incremental},
        "_station_catalog": {"station": {"timezone": "America/Sao_Paulo"}},
        "visualization": {
            "output_format": "png",
            "dpi": 60,
            "level1_figures": {"enabled": True},
            "altitude_ranges_km": [1.0],
            "channels_to_plot": channels,
            "quicklook": {
                "show_pbl": True,
                "show_tropopause": True,
                "mean_profile_smooth_bins": 20,
                "mean_chunk_profiles": 2,
                "max_time_gap_minutes": 10,
                "max_time_samples": 2,
                "max_altitude_bins": 2,
                "missing_data_color": "lightgray",
                "colormap": "viridis",
            },
            "atmospheric_profile": {
                "max_altitude_km": 1.5,
                "evolution_max_altitude_bins": 20,
                "comparison_altitude_bands_km": [[0.0, 1.5]],
            },
        },
    }
    if config_file is not None:
        config["_config_file"] = str(config_file)
    return config


def test_time_gap_markers_insert_nan_profiles() -> None:
    times = pd.to_datetime(
        ["2024-01-01T00:00:00", "2024-01-01T00:05:00", "2024-01-01T00:40:00"]
    )
    data = xr.DataArray(
        np.ones((3, 3)),
        dims=("time", "altitude"),
        coords={"time": times, "altitude": np.array([0.0, 0.5, 1.0])},
    )

    result = _insert_time_gap_markers(data, _config(["532.AN"]))

    assert result.sizes["time"] == 5
    assert np.isnan(result.isel(time=2).values).all()
    assert np.isnan(result.isel(time=3).values).all()


def test_plot_quicklook_defaults_to_observed_session_extent(tmp_path: Path, monkeypatch) -> None:
    times = pd.to_datetime(
        ["2024-01-01T00:02:00", "2024-01-01T00:05:00", "2024-01-01T00:09:00"]
    )
    data = xr.DataArray(
        np.ones((3, 3)),
        dims=("time", "altitude"),
        coords={"time": times, "altitude": np.array([0.5, 1.0, 1.5])},
    )
    captured: dict[str, object] = {}

    def fake_save(fig, out_path, dpi):
        del dpi
        captured["xlim"] = fig.axes[0].get_xlim()
        captured["footer_texts"] = [item.get_text() for item in fig.texts]
        plt.close(fig)
        return Path(out_path)

    monkeypatch.setattr(quicklooks, "_save_figure", fake_save)

    output = quicklooks.plot_quicklook(
        data_slice=data,
        error_slice=data * 0.05,
        max_altitude=1.5,
        channel_name="532.PC",
        ds=xr.Dataset(coords={"time": times}),
        output_folder=tmp_path,
        file_name_prefix=SESSION_ID,
        config=_config(["532.PC"]),
        root_dir=tmp_path,
        session_id=SESSION_ID,
        timezone_name="America/Sao_Paulo",
    )

    assert output.name == f"{SESSION_ID}_L1_RCS_532PC_1.5km.png"
    assert not any("Local period" in text for text in captured["footer_texts"])


def test_level1_figure_orchestrator_passes_session_context(tmp_path: Path, monkeypatch) -> None:
    level1 = _write_level1(tmp_path / f"{SESSION_ID}_L1.nc", ["532.AN"])
    captured: dict[str, object] = {}

    def fake_quicklook(**kwargs):
        captured["session_id"] = kwargs["session_id"]
        captured["timezone_name"] = kwargs["timezone_name"]
        out = Path(kwargs["output_folder"]) / f"{SESSION_ID}_L1_RCS_532AN_1km.png"
        out.write_text("quicklook", encoding="utf-8")
        return out

    def fake_global(ds, output_folder, file_name_prefix, config, root_dir):
        out = Path(output_folder) / f"{SESSION_ID}_L1_MeanRCS.png"
        out.write_text("mean", encoding="utf-8")
        return out

    def fake_atmosphere(ds, *, output_folder, file_name_prefix, config, root_dir):
        out = Path(output_folder) / f"{SESSION_ID}_L1_AtmosphericProfile.png"
        out.write_text("atmosphere", encoding="utf-8")
        return out

    def fake_evolution(ds, *, output_folder, file_name_prefix, config, root_dir):
        out = Path(output_folder) / f"{SESSION_ID}_L1_AtmosphericEvolution.png"
        out.write_text("evolution", encoding="utf-8")
        return out

    monkeypatch.setattr(level1_figures, "plot_quicklook", fake_quicklook)
    monkeypatch.setattr(level1_figures, "plot_global_mean_rcs", fake_global)
    monkeypatch.setattr(level1_figures, "plot_atmospheric_profile", fake_atmosphere)
    monkeypatch.setattr(level1_figures, "plot_atmospheric_evolution", fake_evolution)

    result = level1_figures.generate_level1_figures(
        level1,
        _config(["532.AN"], incremental=False),
        _ListLogger(),
        root_dir=tmp_path,
    )

    assert result.status is ExecutionStatus.OK
    assert result.output_path == tmp_path / "figures"
    assert captured == {
        "session_id": SESSION_ID,
        "timezone_name": "America/Sao_Paulo",
    }
    assert result.metadata["generated"] == 4


def test_level1_figures_are_incrementally_skipped_when_current(tmp_path: Path, monkeypatch) -> None:
    level1 = _write_level1(tmp_path / f"{SESSION_ID}_L1.nc", ["532.AN"])
    calls = {"quicklook": 0, "mean": 0, "atmosphere": 0, "evolution": 0}

    def fake_quicklook(**kwargs):
        calls["quicklook"] += 1
        out = Path(kwargs["output_folder"]) / f"{SESSION_ID}_L1_RCS_532AN_1km.png"
        out.write_text("quicklook", encoding="utf-8")
        return out

    def fake_global(ds, output_folder, file_name_prefix, config, root_dir):
        calls["mean"] += 1
        out = Path(output_folder) / f"{SESSION_ID}_L1_MeanRCS.png"
        out.write_text("mean", encoding="utf-8")
        return out

    def fake_atmosphere(ds, *, output_folder, file_name_prefix, config, root_dir):
        calls["atmosphere"] += 1
        out = Path(output_folder) / f"{SESSION_ID}_L1_AtmosphericProfile.png"
        out.write_text("atmosphere", encoding="utf-8")
        return out

    def fake_evolution(ds, *, output_folder, file_name_prefix, config, root_dir):
        calls["evolution"] += 1
        out = Path(output_folder) / f"{SESSION_ID}_L1_AtmosphericEvolution.png"
        out.write_text("evolution", encoding="utf-8")
        return out

    monkeypatch.setattr(level1_figures, "plot_quicklook", fake_quicklook)
    monkeypatch.setattr(level1_figures, "plot_global_mean_rcs", fake_global)
    monkeypatch.setattr(level1_figures, "plot_atmospheric_profile", fake_atmosphere)
    monkeypatch.setattr(level1_figures, "plot_atmospheric_evolution", fake_evolution)

    config = _config(["532.AN"], incremental=True)
    first = level1_figures.generate_level1_figures(level1, config, _ListLogger(), root_dir=tmp_path)
    second = level1_figures.generate_level1_figures(level1, config, _ListLogger(), root_dir=tmp_path)

    assert first.status is ExecutionStatus.OK
    assert second.status is ExecutionStatus.SKIPPED
    assert calls == {"quicklook": 1, "mean": 1, "atmosphere": 1, "evolution": 1}


def test_level1_figures_regenerate_when_config_file_changes(tmp_path: Path, monkeypatch) -> None:
    level1 = _write_level1(tmp_path / f"{SESSION_ID}_L1.nc", ["532.AN"])
    config_file = tmp_path / "config.yaml"
    config_file.write_text("first", encoding="utf-8")
    calls = {"quicklook": 0, "mean": 0, "atmosphere": 0, "evolution": 0}

    def fake_quicklook(**kwargs):
        calls["quicklook"] += 1
        out = Path(kwargs["output_folder"]) / f"{SESSION_ID}_L1_RCS_532AN_1km.png"
        out.write_text("quicklook", encoding="utf-8")
        return out

    def fake_global(ds, output_folder, file_name_prefix, config, root_dir):
        calls["mean"] += 1
        out = Path(output_folder) / f"{SESSION_ID}_L1_MeanRCS.png"
        out.write_text("mean", encoding="utf-8")
        return out

    def fake_atmosphere(ds, *, output_folder, file_name_prefix, config, root_dir):
        calls["atmosphere"] += 1
        out = Path(output_folder) / f"{SESSION_ID}_L1_AtmosphericProfile.png"
        out.write_text("atmosphere", encoding="utf-8")
        return out

    def fake_evolution(ds, *, output_folder, file_name_prefix, config, root_dir):
        calls["evolution"] += 1
        out = Path(output_folder) / f"{SESSION_ID}_L1_AtmosphericEvolution.png"
        out.write_text("evolution", encoding="utf-8")
        return out

    monkeypatch.setattr(level1_figures, "plot_quicklook", fake_quicklook)
    monkeypatch.setattr(level1_figures, "plot_global_mean_rcs", fake_global)
    monkeypatch.setattr(level1_figures, "plot_atmospheric_profile", fake_atmosphere)
    monkeypatch.setattr(level1_figures, "plot_atmospheric_evolution", fake_evolution)

    first_config = _config(["532.AN"], incremental=True, config_file=config_file)
    level1_figures.generate_level1_figures(level1, first_config, _ListLogger(), root_dir=tmp_path)

    mean_path = tmp_path / "figures" / f"{SESSION_ID}_L1_MeanRCS.png"
    newer_ns = mean_path.stat().st_mtime_ns + 1_000_000_000
    config_file.write_text("changed", encoding="utf-8")
    os.utime(config_file, ns=(newer_ns, newer_ns))

    second_config = _config(["532.AN"], incremental=True, config_file=config_file)
    level1_figures.generate_level1_figures(level1, second_config, _ListLogger(), root_dir=tmp_path)

    assert calls == {"quicklook": 2, "mean": 2, "atmosphere": 2, "evolution": 2}



def test_quicklook_display_decimation_does_not_change_scientific_array() -> None:
    data = xr.DataArray(
        np.arange(6 * 8, dtype=np.float32).reshape(6, 8),
        dims=("time", "altitude"),
        coords={
            "time": pd.date_range("2024-01-01", periods=6, freq="1min"),
            "altitude": np.arange(8, dtype=np.float64),
        },
    )

    displayed = _decimate_for_display(data, _config(["532.AN"]))

    assert data.shape == (6, 8)
    assert displayed.sizes["time"] <= 2
    assert displayed.sizes["altitude"] <= 2


def test_streaming_mean_matches_full_resolution_reference() -> None:
    signal = xr.DataArray(
        np.arange(6 * 4, dtype=np.float32).reshape(6, 4),
        dims=("time", "altitude"),
        coords={"time": np.arange(6), "altitude": np.arange(4)},
    )
    error = xr.full_like(signal, 2.0)

    mean, err = _streaming_time_statistics(signal, error, chunk_profiles=2)

    np.testing.assert_allclose(mean.values, signal.mean("time").values)
    np.testing.assert_allclose(
        err.values,
        np.sqrt((error.values.astype(np.float64) ** 2).sum(axis=0)) / 6.0,
    )
