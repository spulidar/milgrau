"""Tests for LIPANCORA Level 1 pipeline helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from milgrau.level1.lipancora import (
    _discover_level0_files,
    _level1_target_path,
    _make_level1_netcdf_safe,
)


def _config(tmp_path: Path) -> dict:
    """Return a minimal Level 1 config rooted at one temp processed-data tree."""
    return {"directories": {"processed_data": str(tmp_path / "processed")}}


def test_discover_level0_files_ignores_figure_artifacts(tmp_path: Path) -> None:
    """Only session-root Level 0 products should be discovered."""
    session_id = "spu_20240902-2100Z_20240903-0400Z"
    root = tmp_path / "processed" / "spu" / "2024" / "09" / session_id
    root.mkdir(parents=True)
    level0 = root / f"{session_id}_L0.nc"
    level0.write_text("raw", encoding="utf-8")
    (root / f"{session_id}_L1.nc").write_text("l1", encoding="utf-8")
    figures = root / "figures"
    figures.mkdir()
    (figures / f"{session_id}_L0.nc").write_text("artifact", encoding="utf-8")

    discovered = _discover_level0_files(_config(tmp_path))

    assert discovered == [level0]


def test_make_level1_netcdf_safe_removes_timezone_from_time_coord() -> None:
    """Timezone-aware UTC coordinates should be converted to NetCDF-safe naive UTC."""
    ds = xr.Dataset(coords={"time": pd.date_range("2024-01-01T00:00:00Z", periods=2, freq="5min", tz="UTC")})

    safe = _make_level1_netcdf_safe(ds)

    assert str(safe["time"].dtype).startswith("datetime64")
    assert pd.to_datetime(safe["time"].values).tz is None


def test_radiosonde_only_output_is_noncanonical_and_repeatable(tmp_path: Path) -> None:
    config = _config(tmp_path)
    level0 = tmp_path / "external_L0.nc"
    canonical = _level1_target_path(level0, config)
    assert canonical.name == "external_L0_L1.nc"
    alternate = _level1_target_path(
        level0, {**config, "_radiosonde_only_mode": True},
    )
    assert alternate.name == "external_L0_radiosonde_L1.nc"
    assert alternate != canonical
    assert _level1_target_path(level0, config) == canonical


def test_radiosonde_only_canonical_session_has_separate_variant(tmp_path: Path) -> None:
    session = "spu_20240902-2100Z_20240903-0400Z"
    config = {
        "directories": {"processed_data": str(tmp_path)},
        "_station_catalog": {"station": {"id": "spu"}},
    }
    input_path = tmp_path / (session + "_L0.nc")
    output = _level1_target_path(
        input_path, {**config, "_radiosonde_only_mode": True},
    )
    assert output.name == session + "_radiosonde_L1.nc"
    assert output != _level1_target_path(input_path, config)
