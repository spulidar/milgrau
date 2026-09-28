"""Fail-fast tests for productive visualization configuration."""

from __future__ import annotations

import pytest

from milgrau.viz.config import VisualizationConfigurationError, resolve_visualization_config
from milgrau.viz.quicklooks import _get_gap_threshold_minutes
from milgrau.viz.style import get_output_settings


def _config() -> dict:
    return {
        "visualization": {
            "output_format": "webp",
            "dpi": 120,
            "altitude_ranges_km": [5, 15, 30],
            "channels_to_plot": ["532.AN", "355.AN"],
            "quicklook": {
                "show_pbl": True,
                "show_tropopause": True,
                "mean_profile_smooth_bins": 20,
                "max_time_gap_minutes": 10,
                "missing_data_color": "lightgray",
                "colormap": "jet",
            },
            "level2_qa": {"enabled": True},
        }
    }


def test_visualization_recipe_resolves_without_defaults() -> None:
    resolved = resolve_visualization_config(_config())
    assert resolved.output_format == "webp"
    assert resolved.dpi == 120
    assert resolved.altitude_ranges_km == (5.0, 15.0, 30.0)
    assert resolved.channels_to_plot == ("532.AN", "355.AN")
    assert resolved.quicklook.mean_profile_smooth_bins == 20
    assert resolved.quicklook.max_time_gap_minutes == 10.0
    assert get_output_settings(_config()) == ("webp", 120)


def test_visualization_output_format_and_dpi_are_required() -> None:
    config = _config()
    del config["visualization"]["output_format"]
    with pytest.raises(VisualizationConfigurationError, match="visualization.output_format"):
        resolve_visualization_config(config)

    config = _config()
    del config["visualization"]["dpi"]
    with pytest.raises(VisualizationConfigurationError, match="visualization.dpi"):
        resolve_visualization_config(config)


def test_visualization_altitude_ranges_do_not_fall_back() -> None:
    config = _config()
    config["visualization"]["altitude_ranges_km"] = []
    with pytest.raises(VisualizationConfigurationError, match="altitude_ranges_km"):
        resolve_visualization_config(config)


def test_quicklook_gap_threshold_is_required_and_not_derived_from_data() -> None:
    config = _config()
    del config["visualization"]["quicklook"]["max_time_gap_minutes"]
    with pytest.raises(VisualizationConfigurationError, match="max_time_gap_minutes"):
        _get_gap_threshold_minutes(config, data_slice=None)  # type: ignore[arg-type]


def test_mean_profile_smoothing_must_be_explicit_positive_integer() -> None:
    config = _config()
    config["visualization"]["quicklook"]["mean_profile_smooth_bins"] = 0
    with pytest.raises(VisualizationConfigurationError, match="mean_profile_smooth_bins"):
        resolve_visualization_config(config)
