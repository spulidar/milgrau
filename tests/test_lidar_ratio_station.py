"""Ownership and fallback tests for station lidar-ratio climatology."""

from __future__ import annotations

from pathlib import Path

import yaml

from milgrau.config.loader import load_config
from milgrau.level2.config import get_lidar_ratio


def test_repository_lidar_ratio_climatology_is_owned_by_station_yaml() -> None:
    raw_config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    raw_station = yaml.safe_load(Path("station.yaml").read_text(encoding="utf-8"))

    assert "lidar_ratios_sr" not in raw_config["inversion"]
    assert "lidar_ratio_std_sr" not in raw_config["inversion"]
    climatology = raw_station["station"]["lidar_ratio_climatology"]
    assert climatology["monthly_sr"]["532"]["09"] == 55.78
    assert climatology["std_sr"]["532"] == 10.0
    assert climatology["provenance"]["source"]


def test_loader_materializes_station_climatology_for_transitional_level2_consumer() -> None:
    config = load_config("config.yaml")

    assert get_lidar_ratio(config, 355, "2026-02-15T00:00:00") == (116.08, 15.0)
    assert get_lidar_ratio(config, 532, "2026-09-15T00:00:00") == (55.78, 10.0)


def test_explicit_config_lidar_ratio_remains_compatibility_fallback_without_station_climatology(tmp_path: Path) -> None:
    base = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    station = yaml.safe_load(Path("station.yaml").read_text(encoding="utf-8"))
    station["station"].pop("lidar_ratio_climatology")
    base["inversion"]["wavelengths_to_process"] = [532]
    base["inversion"]["lidar_ratios_sr"] = {"532": {f"{month:02d}": 61.0 for month in range(1, 13)}}
    base["inversion"]["lidar_ratio_std_sr"] = {"532": 7.0}
    base["station_config"] = "station.yaml"

    config_path = tmp_path / "config.yaml"
    station_path = tmp_path / "station.yaml"
    config_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    station_path.write_text(yaml.safe_dump(station), encoding="utf-8")

    config = load_config(config_path)

    assert get_lidar_ratio(config, 532, "2026-09-15T00:00:00") == (61.0, 7.0)
