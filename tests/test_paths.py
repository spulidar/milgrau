"""Tests for canonical MILGRAU path builders."""

from __future__ import annotations

from pathlib import Path

from milgrau.io.paths import (
    global_mean_rcs_output_path,
    level0_output_path,
    level1_output_path,
    level2_output_path,
    log_output_root,
    measurement_product_dir,
    measurement_save_id,
    quicklook_output_path,
    radiosonde_cache_dir,
    raw_data_root,
    surface_weather_cache_dir,
)


def _config() -> dict:
    """Return a minimal path configuration for tests."""
    return {"directories": {"processed_data": "processed"}}


def test_measurement_save_id_inserts_sa_marker() -> None:
    """Inventory measurement IDs should map to canonical product save IDs."""
    assert measurement_save_id("20240101nt") == "20240101sant"
    assert measurement_save_id("20240101am") == "20240101saam"


def test_level_product_paths_are_canonical(tmp_path: Path) -> None:
    """Level 0, Level 1 and Level 2 paths should share the same product folder."""
    config = _config()

    level0 = level0_output_path("20240101nt", config, root_dir=tmp_path)
    assert level0 == tmp_path / "processed" / "2024" / "01" / "20240101sant" / "20240101sant.nc"

    assert measurement_product_dir("20240101sant", config, root_dir=tmp_path) == level0.parent

    level1 = level1_output_path(level0, config, root_dir=tmp_path)
    assert level1 == level0.parent / "20240101sant_level1_rcs.nc"

    level2 = level2_output_path(level1)
    assert level2 == level0.parent / "20240101sant_level2_optical.nc"


def test_level2_output_path_supports_variant_tags(tmp_path: Path) -> None:
    """Level 2 paths should allow a tagged variant for filtered runs."""
    config = _config()
    level0 = level0_output_path("20240101nt", config, root_dir=tmp_path)
    level1 = level1_output_path(level0, config, root_dir=tmp_path)

    tagged = level2_output_path(level1, variant_tag="0400-0500")
    assert tagged == level0.parent / "20240101sant_0400-0500_level2_optical.nc"


def test_visual_product_paths() -> None:
    """Quicklook and global-mean plot paths should be deterministic."""
    assert quicklook_output_path("plots", "measure", "532nm AN", 15, "webp") == Path(
        "plots/Quicklook_measure_532nm_AN_15km.webp"
    )
    assert global_mean_rcs_output_path("plots", "measure", ".png") == Path("plots/GlobalMeanRCS_measure.png")


def test_configured_roots_are_resolved_from_directories_section(tmp_path: Path) -> None:
    """Raw, log and cache roots should honor configured relative directories."""
    config = {
        "directories": {"raw_data": "raw", "processed_data": "processed", "log_dir": "logs"},
        "surface_weather": {"cache_dir": "custom/weather-cache"},
        "radiosonde": {"cache_dir": "custom/radiosonde-cache"},
    }

    assert raw_data_root(config, root_dir=tmp_path) == tmp_path / "raw"
    assert log_output_root(config, root_dir=tmp_path) == tmp_path / "logs"
    assert surface_weather_cache_dir(config, root_dir=tmp_path) == tmp_path / "custom" / "weather-cache"
    assert radiosonde_cache_dir(config, root_dir=tmp_path) == tmp_path / "custom" / "radiosonde-cache"
