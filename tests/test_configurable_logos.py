"""Tests for configurable institution logo assets in L1/L2 figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from milgrau.viz.style import (
    add_footer_and_logos,
    configured_logos,
    logo_file_dependencies,
    resolve_logo_path,
)


def test_explicit_logo_selection_and_project_root(tmp_path: Path) -> None:
    custom = tmp_path / "assets" / "custom.png"
    custom.parent.mkdir()
    plt.imsave(custom, np.ones((3, 5, 3)))
    config = {"visualization": {"logos": [{"path": "assets/custom.png", "height": 0.055}]}}
    assert configured_logos(config) == (("assets/custom.png", 0.055),)
    assert logo_file_dependencies(config, tmp_path) == [custom]
    assert resolve_logo_path(tmp_path, "default.png") == tmp_path / "img" / "default.png"
    fig = plt.figure()
    try:
        add_footer_and_logos(fig, tmp_path, config=config)
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_empty_or_missing_logo_skips_artwork(tmp_path: Path) -> None:
    for paths in ([], [{"path": "other.png", "height": 0.05}]):
        config = {"visualization": {"logos": paths}}
        fig = plt.figure()
        try:
            add_footer_and_logos(fig, tmp_path, config=config)
            assert not fig.axes
            assert logo_file_dependencies(config, tmp_path) == []
        finally:
            plt.close(fig)


@pytest.mark.parametrize(
    "entries",
    [
        None,
        "logo.png",
        [{"path": "test.svg", "height": 0.05}],
        [{"path": "test.png", "height": 0}],
        [{"path": "test.png", "height": True}],
        [{"path": "test.png", "height": 0.05}, {"path": "test.png", "height": 0.04}],
        [{"path": "test.png"}],
    ],
)
def test_invalid_logo_config_is_rejected(entries: object) -> None:
    with pytest.raises(ValueError, match="logo|path|height"):
        configured_logos({"visualization": {"logos": entries}})
