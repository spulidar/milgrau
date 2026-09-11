"""Tests for current MILGRAU configuration ownership and loading."""

from __future__ import annotations

from pathlib import Path

import yaml

from milgrau.config.loader import load_config, normalize_config
from milgrau.level0.config import resolve_level0_config, station_coordinates, station_timezone
from milgrau.level1.config import resolve_level1_config
from milgrau.level2.config import get_wavelengths_to_process
from milgrau.viz.config import resolve_visualization_config


def test_repository_config_loads_without_legacy_station_aliases() -> None:
    config = load_config("config.yaml")

    assert "directories" in config
    assert "processing" in config
    assert "level0" in config
    assert "level1" in config
    assert "inversion" in config
    assert "visualization" in config
    assert "_station_catalog" in config

    # Station/instrument reality stays in station.yaml instead of being copied
    # into old top-level compatibility structures.
    assert "site" not in config
    assert "radiosonde" not in config
    assert "hardware" not in config
    assert "channels" not in config["physics"]
    assert "speed_of_light" not in config["physics"]
    assert "speed_of_light_m_s" not in config["physics"]
    assert "bg_start" not in config["physics"]
    assert "bg_stop" not in config["physics"]


def test_repository_config_passes_stage_specific_resolvers() -> None:
    config = load_config("config.yaml")

    level0 = resolve_level0_config(config)
    level1 = resolve_level1_config(config)
    visualization = resolve_visualization_config(config)
    wavelengths = get_wavelengths_to_process(config)

    assert level0.directories.raw_data == "01-data"
    assert level0.discovery.raw_scan_ignore_dirs == ("openmeteo_cache", "wyoming_cache")
    assert level1.background.start_altitude_m == 29000.0
    assert visualization.output_format == "webp"
    assert wavelengths == [355, 532]


def test_station_identity_is_resolved_from_catalog_not_aliases() -> None:
    config = load_config("config.yaml")

    assert station_timezone(config) == "America/Sao_Paulo"
    latitude, longitude = station_coordinates(config)
    assert latitude == -23.5607
    assert longitude == -46.7398


def test_station_lidar_ratio_climatology_materializes_only_level2_recipe_view() -> None:
    config = load_config("config.yaml")
    station_lr = config["_station_catalog"]["station"]["lidar_ratio_climatology"]

    assert config["inversion"]["lidar_ratios_sr"] == station_lr["monthly_sr"]
    assert config["inversion"]["lidar_ratio_std_sr"] == station_lr["std_sr"]
    assert "lidar_ratios" not in config["inversion"]


def test_normalize_config_is_defensive_copy_without_alias_creation() -> None:
    source = {
        "physics": {
            "vertical_resolution_m": 7.5,
            "speed_of_light_m_s": 299792458.0,
            "background_start_m": 29000.0,
        },
        "inversion": {"lidar_ratios_sr": {"532": {"01": 60.0}}},
    }

    normalized = normalize_config(source)

    assert normalized == source
    assert normalized is not source
    assert "speed_of_light" not in normalized["physics"]
    assert "bg_start" not in normalized["physics"]
    assert "lidar_ratios" not in normalized["inversion"]


def test_positional_channel_corrections_are_not_normalized_back_into_productive_config() -> None:
    source = {"physics": {"channels": {"532.PC": [0.0035, -3, 0.0]}}}

    normalized = normalize_config(source)

    assert normalized["physics"]["channels"]["532.PC"] == [0.0035, -3, 0.0]
    # Productive calibration resolution comes from station.yaml; the loader does
    # not reinterpret this legacy shape into named correction fields.


def test_load_minimal_config_without_station_keeps_only_declared_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "directories": {"raw_data": "raw", "processed_data": "processed", "log_dir": "logs"},
        "processing": {"incremental": False},
        "custom_section": {"kept": True},
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_config(config_path)

    assert config["directories"] == payload["directories"]
    assert config["processing"] == payload["processing"]
    assert config["custom_section"] == {"kept": True}
    assert "_station_catalog" not in config
    assert config["_config_file"] == str(config_path.resolve())


def test_loader_does_not_recreate_removed_hardware_or_site_views(tmp_path: Path) -> None:
    repository = load_config("config.yaml")
    config_path = tmp_path / "config.yaml"
    station_path = tmp_path / "station.yaml"

    processing_payload = {
        key: value
        for key, value in repository.items()
        if not key.startswith("_") and key not in {"inversion"}
    }
    processing_payload["station_config"] = "station.yaml"
    processing_payload["inversion"] = {
        key: value
        for key, value in repository["inversion"].items()
        if key not in {"lidar_ratios_sr", "lidar_ratio_std_sr"}
    }
    config_path.write_text(yaml.safe_dump(processing_payload, sort_keys=False), encoding="utf-8")
    station_path.write_text(Path(repository["_station_config_path"]).read_text(encoding="utf-8"), encoding="utf-8")

    loaded = load_config(config_path)

    assert "hardware" not in loaded
    assert "site" not in loaded
    assert "radiosonde" not in loaded
    assert "channels" not in loaded["physics"]
    assert loaded["_station_catalog"]["station"]["id"] == "spu"
