"""Tests for canonical MILGRAU path and identity builders."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from milgrau.io.paths import (
    build_measurement_id,
    build_session_id,
    global_mean_rcs_output_path,
    level0_output_path,
    level0_scc_output_path,
    level1_output_path,
    level2_output_path,
    log_output_root,
    logging_measurement_id,
    measurement_day_dir,
    product_measurement_id,
    product_session_id,
    quicklook_output_path,
    session_dir,
    session_id_parts,
    session_level0_output_path,
    session_level0_scc_output_path,
    session_level1_output_path,
    session_level2_output_path,
    radiosonde_cache_dir,
    raw_data_root,
    surface_weather_cache_dir,
)


def _config() -> dict:
    return {
        "directories": {"processed_data": "processed"},
        "_station_catalog": {"station": {"id": "spu"}},
    }


def test_measurement_id_is_local_date_station_and_period_start() -> None:
    assert build_measurement_id("20240101", "spu", "00") == "20240101_spu_00"
    assert build_measurement_id("20240101", "SPU", 6) == "20240101_spu_06"
    with pytest.raises(ValueError, match="period start"):
        build_measurement_id("20240101", "spu", "03")
    with pytest.raises(ValueError, match="measurement date"):
        build_measurement_id("20240231", "spu", "06")


def test_product_measurement_id_is_stable_across_processing_levels() -> None:
    expected = "20240101_spu_06"
    assert product_measurement_id("20240101_spu_06_L0.nc") == expected
    assert product_measurement_id("20240101_spu_06_L0_scc.nc") == expected
    assert product_measurement_id("20240101_spu_06_L1.nc") == expected
    assert product_measurement_id("20240101_spu_06_L1_scc.nc") == expected
    assert product_measurement_id("20240101_spu_06_L2.nc") == expected
    assert product_measurement_id("20240101_spu_06_0400-0500Z_L2.nc") == expected
    with pytest.raises(ValueError, match="Unrecognized"):
        product_measurement_id("arbitrary.nc.txt")


def test_logging_measurement_id_never_masks_an_unrelated_pipeline_error() -> None:
    assert logging_measurement_id("20240101_spu_06_L1.nc") == "20240101_spu_06"
    assert logging_measurement_id("noncanonical_L1.nc") == "-"
    assert logging_measurement_id("arbitrary.txt") == "-"


def test_level_product_paths_are_grouped_by_station_and_local_day(tmp_path: Path) -> None:
    config = _config()
    measurement_id = "20240101_spu_06"
    day_dir = tmp_path / "processed" / "spu" / "2024" / "01" / "20240101"

    level0 = level0_output_path(measurement_id, config, root_dir=tmp_path)
    assert level0 == day_dir / "20240101_spu_06_L0.nc"
    assert measurement_day_dir(measurement_id, config, root_dir=tmp_path) == day_dir

    scc = level0_scc_output_path(measurement_id, config, root_dir=tmp_path)
    assert scc == day_dir / "20240101_spu_06_L0_scc.nc"

    level1 = level1_output_path(level0, config, root_dir=tmp_path)
    assert level1 == day_dir / "20240101_spu_06_L1.nc"

    level2 = level2_output_path(level1)
    assert level2 == day_dir / "20240101_spu_06_L2.nc"


def test_scc_level0_keeps_distinct_scc_lineage(tmp_path: Path) -> None:
    config = _config()
    scc = level0_scc_output_path("20240101_spu_06", config, root_dir=tmp_path)
    level1 = level1_output_path(scc, config, root_dir=tmp_path)
    assert level1 == scc.parent / "20240101_spu_06_L1_scc.nc"
    assert level2_output_path(level1) == scc.parent / "20240101_spu_06_L2_scc.nc"


def test_noncanonical_external_level0_writes_level1_beside_source(tmp_path: Path) -> None:
    config = _config()
    external = tmp_path / "imports" / "foreign_station_scc_raw.nc"
    level1 = level1_output_path(external, config, root_dir=tmp_path)
    assert level1 == external.parent / "foreign_station_scc_raw_L1.nc"


def test_level2_output_path_supports_explicit_utc_variant_tags(tmp_path: Path) -> None:
    config = _config()
    level0 = level0_output_path("20240101_spu_06", config, root_dir=tmp_path)
    level1 = level1_output_path(level0, config, root_dir=tmp_path)
    tagged = level2_output_path(level1, variant_tag="0400-0500Z")
    assert tagged == level0.parent / "20240101_spu_06_0400-0500Z_L2.nc"


def test_visual_product_paths() -> None:
    assert quicklook_output_path("plots", "20240101_spu_06", "532nm AN", 15, "webp") == Path(
        "plots/rcs_20240101_spu_06_532nm_AN_15km.webp"
    )
    assert global_mean_rcs_output_path("plots", "20240101_spu_06", ".png") == Path(
        "plots/rcs_20240101_spu_06_mean.png"
    )


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


def test_session_id_is_station_plus_complete_utc_interval() -> None:
    session_id = build_session_id(
        "SPU",
        datetime(2025, 5, 11, 0, 12, 17, tzinfo=timezone.utc),
        datetime(2025, 5, 11, 7, 37, 42, tzinfo=timezone.utc),
    )
    assert session_id == "spu_20250511-0012Z_20250511-0737Z"

    station, start, end = session_id_parts(session_id)
    assert station == "spu"
    assert start == datetime(2025, 5, 11, 0, 12, tzinfo=timezone.utc)
    assert end == datetime(2025, 5, 11, 7, 37, tzinfo=timezone.utc)


def test_session_id_requires_utc_aware_ordered_interval() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        build_session_id(
            "spu",
            datetime(2025, 5, 11, 0, 12),
            datetime(2025, 5, 11, 7, 37, tzinfo=timezone.utc),
        )
    with pytest.raises(ValueError, match="later than session start"):
        build_session_id(
            "spu",
            datetime(2025, 5, 11, 7, 37, tzinfo=timezone.utc),
            datetime(2025, 5, 11, 0, 12, tzinfo=timezone.utc),
        )


def test_session_products_live_inside_self_identifying_session_folder(tmp_path: Path) -> None:
    config = _config()
    session_id = "spu_20250511-0012Z_20250511-0737Z"
    expected_dir = tmp_path / "processed" / "spu" / "2025" / "05" / session_id

    assert session_dir(session_id, config, root_dir=tmp_path) == expected_dir

    level0 = session_level0_output_path(session_id, config, root_dir=tmp_path)
    assert level0 == expected_dir / f"{session_id}_L0.nc"
    assert product_session_id(level0) == session_id

    scc = session_level0_scc_output_path(session_id, config, root_dir=tmp_path)
    assert scc == expected_dir / f"{session_id}_L0_scc.nc"
    assert product_session_id(scc) == session_id

    level1 = session_level1_output_path(level0, config, root_dir=tmp_path)
    assert level1 == expected_dir / f"{session_id}_L1.nc"
    assert product_session_id(level1) == session_id

    level2 = session_level2_output_path(level1)
    assert level2 == expected_dir / f"{session_id}_L2.nc"
    assert product_session_id(level2) == session_id


def test_session_level2_variant_preserves_session_identity(tmp_path: Path) -> None:
    config = _config()
    session_id = "spu_20250511-0012Z_20250511-0737Z"
    level0 = session_level0_output_path(session_id, config, root_dir=tmp_path)
    level1 = session_level1_output_path(level0, config, root_dir=tmp_path)
    level2 = session_level2_output_path(level1, variant_tag="night")
    assert level2 == level0.parent / f"{session_id}_night_L2.nc"
    assert product_session_id(level2) == session_id
