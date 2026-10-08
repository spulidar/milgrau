"""Tests for canonical MILGRAU session paths and identity builders."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from milgrau.io.paths import (
    build_session_id,
    global_mean_rcs_output_path,
    level0_output_path,
    level0_scc_output_path,
    level1_output_path,
    level2_output_path,
    log_output_root,
    logging_session_id,
    product_session_id,
    quicklook_output_path,
    radiosonde_cache_dir,
    raw_data_root,
    session_dir,
    session_id_parts,
    surface_weather_cache_dir,
)


def _config() -> dict:
    return {
        "directories": {"processed_data": "processed"},
        "_station_catalog": {"station": {"id": "spu"}},
    }


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


def test_session_id_requires_timezone_aware_ordered_interval() -> None:
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


def test_product_session_id_is_stable_across_processing_levels() -> None:
    expected = "spu_20250511-0012Z_20250511-0737Z"
    assert product_session_id(f"{expected}_L0.nc") == expected
    assert product_session_id(f"{expected}_seg00_L0_scc.nc") == expected
    assert product_session_id(f"{expected}_L1.nc") == expected
    assert product_session_id(f"{expected}_L1_scc.nc") == expected
    assert product_session_id(f"{expected}_L2.nc") == expected
    assert product_session_id(f"{expected}_night_L2.nc") == expected
    with pytest.raises(ValueError, match="Unrecognized"):
        product_session_id("arbitrary.nc.txt")


def test_logging_session_id_never_masks_unrelated_pipeline_error() -> None:
    session_id = "spu_20250511-0012Z_20250511-0737Z"
    assert logging_session_id(f"{session_id}_L1.nc") == session_id
    assert logging_session_id("noncanonical_L1.nc") == "-"
    assert logging_session_id("arbitrary.txt") == "-"


def test_level_products_live_inside_self_identifying_session_folder(tmp_path: Path) -> None:
    config = _config()
    session_id = "spu_20250511-0012Z_20250511-0737Z"
    expected_dir = tmp_path / "processed" / "spu" / "2025" / "05" / session_id

    assert session_dir(session_id, config, root_dir=tmp_path) == expected_dir

    level0 = level0_output_path(session_id, config, root_dir=tmp_path)
    assert level0 == expected_dir / f"{session_id}_L0.nc"

    scc = level0_scc_output_path(
        session_id, config, segment_id="seg00", root_dir=tmp_path
    )
    assert scc == expected_dir / f"{session_id}_seg00_L0_scc.nc"

    level1 = level1_output_path(level0, config, root_dir=tmp_path)
    assert level1 == expected_dir / f"{session_id}_L1.nc"

    level2 = level2_output_path(level1)
    assert level2 == expected_dir / f"{session_id}_L2.nc"


def test_scc_level0_keeps_distinct_scc_lineage(tmp_path: Path) -> None:
    config = _config()
    session_id = "spu_20250511-0012Z_20250511-0737Z"
    scc = level0_scc_output_path(
        session_id, config, segment_id="seg01", root_dir=tmp_path
    )
    level1 = level1_output_path(scc, config, root_dir=tmp_path)
    assert level1 == scc.parent / f"{session_id}_seg01_L1_scc.nc"
    assert level2_output_path(level1) == scc.parent / f"{session_id}_seg01_L2_scc.nc"


def test_external_level0_writes_level1_beside_source(tmp_path: Path) -> None:
    config = _config()
    external = tmp_path / "imports" / "foreign_station_scc_raw.nc"
    level1 = level1_output_path(external, config, root_dir=tmp_path)
    assert level1 == external.parent / "foreign_station_scc_raw_L1.nc"


def test_level2_output_path_supports_named_variants(tmp_path: Path) -> None:
    config = _config()
    session_id = "spu_20250511-0012Z_20250511-0737Z"
    level0 = level0_output_path(session_id, config, root_dir=tmp_path)
    level1 = level1_output_path(level0, config, root_dir=tmp_path)
    tagged = level2_output_path(level1, variant_tag="night")
    assert tagged == level0.parent / f"{session_id}_night_L2.nc"


def test_visual_product_paths_use_session_level_figure_names() -> None:
    session_id = "spu_20250511-0012Z_20250511-0737Z"
    assert quicklook_output_path("figures", session_id, "532nm AN", 15, "webp") == Path(
        f"figures/{session_id}_L1_RCS_532nm_AN_15km.webp"
    )
    assert global_mean_rcs_output_path("figures", session_id, ".png") == Path(
        f"figures/{session_id}_L1_MeanRCS.png"
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
