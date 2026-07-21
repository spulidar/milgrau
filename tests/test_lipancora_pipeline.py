"""Tests for LIPANCORA Level 1 pipeline helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from milgrau.level1.lipancora import _discover_level0_files, _make_level1_netcdf_safe


def _config(tmp_path: Path) -> dict:
    """Return a minimal Level 1 config rooted at one temp processed-data tree."""
    return {"directories": {"processed_data": str(tmp_path / "processed")}}


def test_discover_level0_files_ignores_visualization_artifacts(tmp_path: Path) -> None:
    """Only canonical product-root Level 0 files should be discovered."""
    root = tmp_path / "processed" / "2024" / "09" / "20240902sapm"
    root.mkdir(parents=True)
    level0 = root / "20240902sapm.nc"
    level0.write_text("raw", encoding="utf-8")
    (root / "20240902sapm_level1_rcs.nc").write_text("l1", encoding="utf-8")
    (root / "quicklooks").mkdir()
    (root / "quicklooks" / "20240902sapm.nc").write_text("artifact", encoding="utf-8")
    (root / "level2_qa").mkdir()
    (root / "level2_qa" / "20240902sapm.nc").write_text("artifact", encoding="utf-8")

    discovered = _discover_level0_files(_config(tmp_path))

    assert discovered == [level0]


def test_make_level1_netcdf_safe_removes_timezone_from_time_coord() -> None:
    """Timezone-aware UTC coordinates should be converted to NetCDF-safe naive UTC."""
    ds = xr.Dataset(coords={"time": pd.date_range("2024-01-01T00:00:00Z", periods=2, freq="5min", tz="UTC")})

    safe = _make_level1_netcdf_safe(ds)

    assert str(safe["time"].dtype).startswith("datetime64")
    assert pd.to_datetime(safe["time"].values).tz is None
