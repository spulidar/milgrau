"""Contracts for the low-memory real-Level-1 method-v5 R&D runner."""

from __future__ import annotations

import numpy as np
import xarray as xr

from milgrau.level2.v5_validation_cli import _altitude_subset, _output_path


def test_altitude_subset_is_lazy_contiguous_and_converts_km_to_m() -> None:
    altitude_km = np.arange(0.0, 40.0, 0.0075)
    ds = xr.Dataset(
        data_vars={
            "dummy": (("time", "altitude"), np.zeros((2, altitude_km.size))),
        },
        coords={
            "time": np.array(["2025-01-01T00:00", "2025-01-01T00:01"], dtype="datetime64[m]"),
            "altitude": altitude_km,
        },
    )

    subset, altitude_m = _altitude_subset(ds, max_altitude_m=30_000.0)

    assert subset.sizes["altitude"] == altitude_m.size
    assert altitude_m[-1] <= 30_000.0
    assert altitude_m[-1] > 29_990.0
    assert np.isclose(altitude_m[1] - altitude_m[0], 7.5)
    assert subset.sizes["time"] == 2


def test_altitude_subset_preserves_meter_coordinate() -> None:
    altitude_m_input = np.arange(0.0, 40_000.0, 7.5)
    ds = xr.Dataset(coords={"altitude": altitude_m_input})

    subset, altitude_m = _altitude_subset(ds, max_altitude_m=25_500.0)

    assert subset.sizes["altitude"] == altitude_m.size
    assert np.array_equal(subset["altitude"].values, altitude_m)
    assert altitude_m[-1] <= 25_500.0


def test_output_path_is_small_json_evidence_file(tmp_path) -> None:
    path = _output_path(tmp_path, tmp_path / "20250629sa03z_level1_rcs.nc", 532)
    assert path.name == "20250629sa03z_level1_rcs_v5_validation_532nm.json"
