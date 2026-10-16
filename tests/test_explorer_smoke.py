"""Smoke-test the real optional Streamlit Explorer interface."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest

from milgrau.explorer import streamlit_app


def test_streamlit_explorer_imports_and_starts_without_products() -> None:
    app_path = Path(streamlit_app.__file__)
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception, [str(exc.message) for exc in app.exception]


def test_interactive_l2_plots_only_use_saved_products() -> None:
    import numpy as np
    import xarray as xr

    from milgrau.explorer.l2_products import _heatmap, _profile, _spectral, _time_series

    block_time = np.array(["2024-06-20T21:00", "2024-06-20T21:20", "2024-06-20T21:40"], dtype="datetime64[m]")
    altitude = np.arange(10, dtype=float) * 500.0
    nominal = np.ones((3, 2, 10), dtype=float)
    nominal[1, 0, 2] = np.nan
    mc = np.ones((3, 2, 1, 10), dtype=float)
    data_vars = {
        "mc_valid_fraction": (("block_time", "wavelength", "residual_fraction", "altitude"), mc),
        "solar_regime": (("block_time",), ["day", "day", "night"]),
        "segment_id": (("block_time",), ["seg00", "seg00", "seg01"]),
    }
    for quantity in ("aerosol_backscatter", "aerosol_extinction"):
        data_vars[f"{quantity}_nominal_block"] = (
            ("block_time", "wavelength", "altitude"), nominal
        )
        for suffix in ("mc_mean", "mc_q025", "mc_q975", "mc_std"):
            data_vars[f"{quantity}_{suffix}"] = (
                ("block_time", "wavelength", "residual_fraction", "altitude"), mc
            )
    ds = xr.Dataset(
        data_vars,
        coords={"block_time": block_time, "altitude": altitude, "wavelength": [355, 532], "residual_fraction": [0.0]},
    )
    heat = _heatmap(ds, "Retroespalhamento", 355, "nominal", 0.0, 0, 3, 5.0, ["day"], ["seg00"])
    assert np.isnan(np.asarray(heat.data[0].z, dtype=float)[:, 2]).all()
    assert np.isnan(np.asarray(heat.data[0].z, dtype=float)[2, 1])
    assert len(_profile(ds, "Retroespalhamento", 355, 0, 0.0, 5.0).data) >= 2
    assert len(_spectral(ds, "Extinção", [355, 532], 0, 3, 5.0).data) == 2
    assert len(_time_series(ds, "Extinção", 532, [1.0], 0, 3).data) == 1
