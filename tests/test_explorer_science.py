"""Regression coverage for lightweight Explorer science and catalogue helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from milgrau.explorer.catalog import discover_products, product_paths, selected_session_context
from milgrau.explorer.science import bounded_view, optical_field, scenario_options, support_mask


def test_discover_canonical_and_derived_products_without_loading(tmp_path: Path) -> None:
    sid = "spu_20240620-2103Z_20240621-0848Z"
    folder = tmp_path / "spu" / "2024" / "06" / sid
    folder.mkdir(parents=True)
    names = [
        f"{sid}_L0.nc",
        f"{sid}_day_L0_scc.nc",
        f"{sid}_L1.nc",
        f"{sid}_day_L1_scc.nc",
        f"{sid}_L2.nc",
        f"{sid}_night_L2.nc",
    ]
    for name in names:
        (folder / name).touch()
    paths = product_paths(folder, sid)
    assert len(paths["level0_products"]) == 2
    assert len(paths["level1_products"]) == 2
    assert len(paths["level2_products"]) == 2
    assert paths["level2_path"].endswith(f"{sid}_L2.nc")
    inv = discover_products(str(tmp_path))
    assert len(inv) == 1
    assert inv.iloc[0]["session_id"] == sid
    assert inv.iloc[0]["highest_available_level"] == "Level 2"


def test_selected_session_context_reads_small_segments_table(tmp_path: Path) -> None:
    ds = xr.Dataset({
        "Segment_Label": (("segments",), ["seg00", "seg01"]),
        "Segment_Regime": (("segments",), ["night", "day"]),
        "Raw_Lidar_Data": (("time", "channels", "points"), np.ones((1, 1, 3))),
    })
    path = tmp_path / "test.nc"
    ds.to_netcdf(path)
    summary = selected_session_context(str(path))
    assert summary["solar_regimes"] == "night, day"
    assert "seg00 night" in summary["segments"]


def test_science_accessor_preserves_missing_values_and_limits_display() -> None:
    times = np.arange(10, dtype="timedelta64[m]") + np.datetime64("2024-06-20T00:00")
    alt = np.arange(100, dtype=float) * 100.0
    nominal = np.ones((10, 2, 100), dtype=float)
    nominal[2, 0, 15] = np.nan
    mc = np.ones((10, 2, 1, 100), dtype=float) * 2
    valid = np.ones((10, 2, 1, 100), dtype=float)
    valid[2, 0, 0, 15] = 0.0
    ds = xr.Dataset(
        {
            "aerosol_backscatter_nominal_block": (
                ("block_time", "wavelength", "altitude"), nominal
            ),
            "aerosol_backscatter_mc_mean": (
                ("block_time", "wavelength", "residual_fraction", "altitude"), mc
            ),
            "mc_valid_fraction": (
                ("block_time", "wavelength", "residual_fraction", "altitude"), valid
            ),
        },
        coords={
            "block_time": times, "wavelength": [355, 532],
            "residual_fraction": [0.0], "altitude": alt,
        },
    )
    assert scenario_options(ds) == [0.0]
    field = optical_field(ds, "Retroespalhamento", "nominal", 355)
    view = bounded_view(field, start=2, stop=8, top_km=2.5, max_time=3, max_altitude=10)
    assert view.sizes["block_time"] <= 3
    assert view.sizes["altitude"] <= 10
    assert np.isnan(field.isel(block_time=2, altitude=15).item())
    mask = support_mask(ds, "Retroespalhamento", 355)
    assert not bool(mask.isel(block_time=2, altitude=15).item())
    assert bool(mask.isel(block_time=3, altitude=16).item())


def test_utc_calendar_keeps_cross_midnight_session_intact() -> None:
    from datetime import date

    import pandas as pd

    from milgrau.explorer.navigation import (
        available_utc_dates, session_utc_dates, sessions_on_utc_date,
    )

    a = "spu_20240620-2103Z_20240621-0848Z"
    b = "spu_20240621-1200Z_20240621-1530Z"
    inv = pd.DataFrame({"session_id": [a, b], "highest_available_level": ["Level 2", "Level 1"]})
    assert session_utc_dates(a) == (date(2024, 6, 20), date(2024, 6, 21))
    assert available_utc_dates(inv) == [date(2024, 6, 21), date(2024, 6, 20)]
    assert sessions_on_utc_date(inv, date(2024, 6, 20))["session_id"].tolist() == [a]
    assert sessions_on_utc_date(inv, date(2024, 6, 21))["session_id"].tolist() == [a, b]
