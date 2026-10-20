"""Phase 12.1: SCC is a separate opt-in catalog and not scientific core state."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from milgrau.config.loader import load_config
from milgrau.config.station import resolve_station_context, validate_station_config
from milgrau.level0 import libids
from milgrau.level0.processing import _write_scc_exports


def _files(tmp_path: Path):
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    station = Path("station.yaml").read_text(encoding="utf-8")
    scc = Path("scc.yaml").read_text(encoding="utf-8")
    config["station_config"] = "station.yaml"
    config["scc"]["config_file"] = "scc.yaml"
    (tmp_path / "station.yaml").write_text(station, encoding="utf-8")
    (tmp_path / "scc.yaml").write_text(scc, encoding="utf-8")
    return config


def _load_temp(tmp_path: Path, payload: dict):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8",
    )
    return load_config(tmp_path / "config.yaml")


def test_disabled_scc_does_not_read_missing_or_invalid_scc_file(tmp_path: Path) -> None:
    payload = _files(tmp_path)
    (tmp_path / "scc.yaml").write_text("this is not: [valid", encoding="utf-8")
    payload["scc"]["enabled"] = False
    config = _load_temp(tmp_path, payload)
    assert "_scc_catalog" not in config
    assert "scc_policy" not in config["_station_catalog"]
    assert all("scc" not in profile for profile in config["_station_catalog"]["profiles"])
    assert config["inversion"]["lidar_ratios_sr"]["532"]["09"] == 55.78
    context = resolve_station_context(
        config, __import__("datetime").datetime(2025, 9, 13, tzinfo=__import__("datetime").timezone.utc),
        ["532.AN"], mode="day",
    )
    assert not context["scc_available"]


def test_enabled_scc_migrates_unchanged_historical_id_mappings(tmp_path: Path) -> None:
    payload = _files(tmp_path)
    payload["scc"]["enabled"] = True
    config = _load_temp(tmp_path, payload)
    scc = config["_scc_catalog"]
    expected = {
        "spu-apel-2017": (248, 294),
        "spu-raman-2018": (565, 484),
        "spu-merionc-2024": (1047, 1046),
    }
    assert {p: (scc["profiles"][p]["day"]["configuration_id"],
                scc["profiles"][p]["night"]["configuration_id"])
            for p in expected} == expected
    assert scc["profiles"]["spu-merionc-2024"]["night"]["channels"]["387.PC"] == 4077
    assert scc["raman_channel_overrides"]["spu-merionc-2024"]["530"]["scc_night_channel_ids"]["PC"] == 4074
    assert config["_scc_config_path"] == str(tmp_path / "scc.yaml")
    assert config["inversion"]["lidar_ratios_sr"]["532"]["09"] == 55.78


def test_enabled_scc_requires_catalog(tmp_path: Path) -> None:
    payload = _files(tmp_path)
    payload["scc"]["enabled"] = True
    payload["scc"]["config_file"] = "missing-scc.yaml"
    with pytest.raises(FileNotFoundError, match="SCC configuration file not found"):
        _load_temp(tmp_path, payload)


def test_enabled_scc_rejects_mismatched_station_and_unknown_profile(tmp_path: Path) -> None:
    payload = _files(tmp_path)
    payload["scc"]["enabled"] = True
    scc = yaml.safe_load((tmp_path / "scc.yaml").read_text(encoding="utf-8"))
    scc["station_id"] = "another_station"
    (tmp_path / "scc.yaml").write_text(yaml.safe_dump(scc), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _load_temp(tmp_path, payload)

    scc["station_id"] = "spu"
    scc["profiles"]["unknown-instrument"] = deepcopy(scc["profiles"]["spu-merionc-2024"])
    (tmp_path / "scc.yaml").write_text(yaml.safe_dump(scc), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown station profile"):
        _load_temp(tmp_path, payload)


def test_station_catalog_rejects_legacy_scc_fields(tmp_path: Path) -> None:
    _files(tmp_path)
    catalog = yaml.safe_load((tmp_path / "station.yaml").read_text(encoding="utf-8"))
    catalog["scc_policy"] = {"lr_input": {}}
    with pytest.raises(ValueError, match="Unknown station catalog key"):
        validate_station_config(catalog)
    del catalog["scc_policy"]
    catalog["profiles"][1]["scc"] = {"day": {}, "night": {}}
    with pytest.raises(ValueError, match="SCC-only metadata"):
        validate_station_config(catalog)


def test_disabled_scientific_incremental_does_not_check_scc(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        libids, "output_is_current",
        lambda path, inputs, **kwargs: calls.append(str(path)) or True,
    )
    source = tmp_path / "raw"
    source.touch()
    config = {"_station_catalog": {"station": {"id": "spu"}}, "scc": {"enabled": False}}
    assert libids._level0_is_current("spu_20250913-1500Z_20250913-1600Z",
                                   __import__("pandas").DataFrame({"filepath": [str(source)]}),
                                   config, tmp_path / "canonical_L0.nc")
    assert len(calls) == 1


def test_disabled_scc_export_does_not_require_scc_dependencies() -> None:
    import logging

    assert _write_scc_exports(
        session_id="spu_20250913-1500Z_20250913-1600Z",
        lidar_data={},
        group_df=__import__("pandas").DataFrame(),
        weather_data={},
        config={"scc": {"enabled": False}},
        logger=logging.getLogger("test.scc.disabled"),
    ) == []


def test_scc_cleanup_policy_is_rejected(tmp_path: Path) -> None:
    payload = _files(tmp_path)
    payload["scc"]["cleanup"] = {"policy": "after_download"}
    with pytest.raises(ValueError, match="Unknown config.scc keys"):
        _load_temp(tmp_path, payload)
