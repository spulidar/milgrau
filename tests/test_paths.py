"""Tests for canonical MILGRAU path builders."""

from __future__ import annotations

from pathlib import Path

import pytest

from milgrau.io.paths import (
    global_mean_rcs_output_path,
    level0_output_path,
    level1_output_path,
    level2_output_path,
    log_output_root,
    logging_save_id,
    measurement_product_dir,
    measurement_save_id,
    product_save_id,
    quicklook_output_path,
    radiosonde_cache_dir,
    raw_data_root,
    surface_weather_cache_dir,
)


def _config() -> dict:
    return {"directories": {"processed_data": "processed"}}


def test_measurement_save_id_inserts_sa_marker() -> None:
    assert measurement_save_id("2024010103z") == "20240101sa03z"
    assert measurement_save_id("2024010109z") == "20240101sa09z"
    assert measurement_save_id("202401010330z") == "20240101sa0330z"
    with pytest.raises(ValueError, match="Invalid measurement_id"):
        measurement_save_id("20240101nt")


def test_product_save_id_is_stable_across_processing_levels() -> None:
    assert product_save_id("20240101sa03z.nc") == "20240101sa03z"
    assert product_save_id("20240101sa03z_scc.nc") == "20240101sa03z"
    assert product_save_id("20240101sa03z_level1_rcs.nc") == "20240101sa03z"
    assert product_save_id("20240101sa03z_level2_optical.nc") == "20240101sa03z"
    assert product_save_id("20240101sa03z_0400-0500_level2_optical.nc") == "20240101sa03z"
    with pytest.raises(ValueError, match="Unrecognized"):
        product_save_id("arbitrary.nc.txt")


def test_logging_save_id_never_masks_an_unrelated_pipeline_error() -> None:
    assert logging_save_id("20240101sa03z_level1_rcs.nc") == "20240101sa03z"
    assert logging_save_id("noncanonical_level1_rcs.nc") == "-"
    assert logging_save_id("arbitrary.txt") == "-"


def test_level_product_paths_are_canonical(tmp_path: Path) -> None:
    config = _config()
    level0 = level0_output_path("2024010103z", config, root_dir=tmp_path)
    assert level0 == tmp_path / "processed" / "2024" / "01" / "20240101sa03z" / "20240101sa03z.nc"
    assert measurement_product_dir("20240101sa03z", config, root_dir=tmp_path) == level0.parent
    level1 = level1_output_path(level0, config, root_dir=tmp_path)
    assert level1 == level0.parent / "20240101sa03z_level1_rcs.nc"
    level2 = level2_output_path(level1)
    assert level2 == level0.parent / "20240101sa03z_level2_optical.nc"


def test_scc_level0_keeps_level1_in_canonical_measurement_directory(tmp_path: Path) -> None:
    config = _config()
    measurement_dir = measurement_product_dir("20240101sa03z", config, root_dir=tmp_path)
    scc = measurement_dir / "20240101sa03z_scc.nc"
    level1 = level1_output_path(scc, config, root_dir=tmp_path)
    assert level1 == measurement_dir / "20240101sa03z_scc_level1_rcs.nc"


def test_noncanonical_external_level0_writes_level1_beside_source(tmp_path: Path) -> None:
    config = _config()
    external = tmp_path / "imports" / "foreign_station_scc_raw.nc"
    level1 = level1_output_path(external, config, root_dir=tmp_path)
    assert level1 == external.parent / "foreign_station_scc_raw_level1_rcs.nc"


def test_level2_output_path_supports_variant_tags(tmp_path: Path) -> None:
    config = _config()
    level0 = level0_output_path("2024010103z", config, root_dir=tmp_path)
    level1 = level1_output_path(level0, config, root_dir=tmp_path)
    tagged = level2_output_path(level1, variant_tag="0400-0500")
    assert tagged == level0.parent / "20240101sa03z_0400-0500_level2_optical.nc"


def test_visual_product_paths() -> None:
    assert quicklook_output_path("plots", "measure", "532nm AN", 15, "webp") == Path(
        "plots/Quicklook_measure_532nm_AN_15km.webp"
    )
    assert global_mean_rcs_output_path("plots", "measure", ".png") == Path("plots/GlobalMeanRCS_measure.png")


def test_configured_roots_are_resolved_from_directories_section(tmp_path: Path) -> None:
    config = {
        "directories": {"raw_data": "raw", "processed_data": "processed", "log_dir": "logs"},
        "surface_weather": {"cache_dir": "custom/weather-cache"},
        "radiosonde": {"cache_dir": "custom/radiosonde-cache"},
    }
    assert raw_data_root(config, root_dir=tmp_path) == tmp_path / "raw"
    assert log_output_root(config, root_dir=tmp_path) == tmp_path / "logs"
    assert surface_weather_cache_dir(config, root_dir=tmp_path) == tmp_path / "custom" / "weather-cache"
    assert radiosonde_cache_dir(config, root_dir=tmp_path) == tmp_path / "custom" / "radiosonde-cache"
