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


def test_streamlit_can_select_and_view_saved_l2_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new header selects dates and tabs without any directory picker."""
    import numpy as np
    import xarray as xr

    from milgrau.explorer import catalog

    session = "spu_20240620-2103Z_20240621-0848Z"
    folder = tmp_path / "spu" / "2024" / "06" / session
    folder.mkdir(parents=True)
    ds = xr.Dataset(
        {
            "aerosol_backscatter_nominal_block": (
                ("block_time", "wavelength", "altitude"),
                np.ones((2, 1, 4), dtype=float) * 1e-6,
            ),
            "aerosol_extinction_nominal_block": (
                ("block_time", "wavelength", "altitude"),
                np.ones((2, 1, 4), dtype=float) * 2e-6,
            ),
            "solar_regime": (("block_time",), ["night", "night"]),
            "segment_id": (("block_time",), ["seg00", "seg00"]),
            "Segment_Label": (("segments",), ["seg00"]),
            "Segment_Regime": (("segments",), ["night"]),
        },
        coords={
            "block_time": np.array(
                ["2024-06-20T21:00", "2024-06-20T21:20"], dtype="datetime64[m]"
            ),
            "wavelength": [532],
            "altitude": [0.0, 500.0, 1000.0, 1500.0],
        },
        attrs={"product_completeness": "complete"},
    )
    ds.to_netcdf(folder / f"{session}_L2.nc")
    monkeypatch.setattr(
        catalog, "source_context_from_config",
        lambda _path="config.yaml": {
            "root": tmp_path,
            "timezone": "UTC",
            "station": "SPU Lidar Station",
            "latitude": -23.5607,
            "longitude": -46.7398,
            "elevation_m": 766.0,
        },
    )
    app = AppTest.from_file(str(Path(streamlit_app.__file__)), default_timeout=30).run()
    assert not app.exception, [str(e.message) for e in app.exception]
    assert len(app.sidebar.radio) == 0
    assert len(app.sidebar.text_input) == 0
    assert not any(w.label == "Altitude máxima (km)" for w in app.slider)
    day_widget = next(w for w in app.selectbox if w.label == "Data UTC")
    assert len(day_widget.options) == 2  # continuous cross-midnight session
    nav = next(w for w in app.radio if w.label == "Seções")
    assert "L2 Products" in nav.options
    app = nav.set_value("L2 Products").run()
    assert not app.exception, [str(e.message) for e in app.exception]
    assert any(w.label == "Comprimento de onda (nm)" for w in app.selectbox)
    assert not any(w.label == "Altitude máxima (km)" for w in app.slider)


def test_hybrid_theme_is_local_to_explorer() -> None:
    from milgrau.explorer.theme import _BASE_CSS, _FOCUS_CSS

    assert '[data-testid="stSidebar"]' in _BASE_CSS
    assert '[data-testid="stMain"]' in _BASE_CSS
    assert "display: none" in _FOCUS_CSS


def test_site_style_hides_sidebar_and_shows_calendar() -> None:
    from milgrau.explorer.theme import _SITE_CSS

    assert "mg-site-brand" in _SITE_CSS
    assert "stSidebarCollapsedControl" in _SITE_CSS
