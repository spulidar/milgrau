"""Tests for explicit optical-support context in Level 2 QA plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from milgrau.level2.qa import _load_plotter
from milgrau.viz import level2_qa_support
from milgrau.viz.level2_qa_support import (
    _mark_algorithmic_support_top,
    plot_all_level2_qa_with_support,
    retrieval_top_altitude_km,
)


def _dataset(*, include_scalar_top: bool = True) -> xr.Dataset:
    altitude = np.array([3.75, 1003.75, 2003.75, 3003.75, 4003.75, 5003.75, 6003.75, 7003.75])
    support = np.array([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=np.int8)
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "retrieval_inversion_support_flag": (("wavelength", "altitude"), support),
    }
    if include_scalar_top:
        data_vars["retrieval_top_altitude_m"] = (("wavelength",), np.array([6003.75]))
    return xr.Dataset(
        data_vars=data_vars,
        coords={"wavelength": np.array([532]), "altitude": altitude},
    )


def test_retrieval_top_prefers_persisted_scalar() -> None:
    ds = _dataset(include_scalar_top=True)

    assert retrieval_top_altitude_km(ds, 532) == 6.00375


def test_retrieval_top_falls_back_to_altitude_resolved_support() -> None:
    ds = _dataset(include_scalar_top=False)

    assert retrieval_top_altitude_km(ds, 532) == 6.00375


def test_support_marker_labels_diagnostic_region_above_top() -> None:
    fig, ax = plt.subplots()
    try:
        assert _mark_algorithmic_support_top(ax, 6.0, 12.0)
        labels = [artist.get_label() for artist in [*ax.lines, *ax.patches]]
        assert any("Algorithmic optical top" in label for label in labels)
        assert "Outside supported aerosol retrieval" in labels
    finally:
        plt.close(fig)


def test_level2_qa_loader_uses_support_aware_wrapper() -> None:
    plotter = _load_plotter()

    assert plotter is plot_all_level2_qa_with_support


def test_wrapper_replaces_only_scattering_ratio_and_kfs_panels(monkeypatch) -> None:
    ds = _dataset()
    original_config = {
        "visualization": {
            "level2_qa": {
                "generate_gluing_qa": True,
                "generate_molecular_fit_qa": True,
                "generate_scattering_ratio_qa": True,
                "generate_kfs_qa": True,
            }
        }
    }
    captured: dict[str, object] = {}

    def fake_base(**kwargs):
        captured["base_config"] = kwargs["config"]
        return [Path("gluing.png"), Path("molecular.png")]

    monkeypatch.setattr(level2_qa_support, "plot_all_level2_qa", fake_base)
    monkeypatch.setattr(
        level2_qa_support,
        "plot_qa_scattering_ratio_with_support",
        lambda *args, **kwargs: Path("support-sr.png"),
    )
    monkeypatch.setattr(
        level2_qa_support,
        "plot_qa_l2_kfs_with_support",
        lambda *args, **kwargs: Path("support-kfs.png"),
    )

    generated = plot_all_level2_qa_with_support(
        ds_l2=ds,
        output_folder=Path("unused"),
        file_name_prefix="case",
        config=original_config,
        root_dir=Path("."),
    )

    base_qa = captured["base_config"]["visualization"]["level2_qa"]
    assert base_qa["generate_gluing_qa"] is True
    assert base_qa["generate_molecular_fit_qa"] is True
    assert base_qa["generate_scattering_ratio_qa"] is False
    assert base_qa["generate_kfs_qa"] is False
    assert original_config["visualization"]["level2_qa"]["generate_scattering_ratio_qa"] is True
    assert original_config["visualization"]["level2_qa"]["generate_kfs_qa"] is True
    assert generated == [
        Path("gluing.png"),
        Path("molecular.png"),
        Path("support-sr.png"),
        Path("support-kfs.png"),
    ]
