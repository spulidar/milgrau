"""Tests for MILGRAU configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
import logging

import pytest
import yaml

from milgrau.config.loader import load_config, normalize_config
from milgrau.config.schema import find_unknown_config_keys, validate_config_minimum
from milgrau.level1.common import get_channel_constant


LEGACY_CHANNELS = {
    "355.PC": [0.0020, -2, 0.0005],
    "355.AN": [0.0000, -11, 0.0000],
    "387.PC": [0.0000, -2, 0.0000],
    "387.AN": [0.0000, 8, 0.0000],
    "408.PC": [0.0000, -2, 0.0000],
    "408.AN": [0.0000, 8, 0.0000],
    "530.PC": [0.0000, -2, 0.0000],
    "530.AN": [0.0000, 7, 0.0000],
    "532.PC": [0.0035, -3, 0.0000],
    "532.AN": [0.0000, -12, 0.0000],
    "1064.PC": [0.0000, 0, 0.0000],
    "1064.AN": [0.0000, 1, 0.0000],
}


def _named_channels(legacy: dict[str, list[float | int]]) -> dict[str, dict[str, float | int]]:
    return {
        channel: {
            "deadtime_us": values[0],
            "bin_shift_bins": values[1],
            "background_offset": values[2],
        }
        for channel, values in legacy.items()
    }


def _minimal_config() -> dict:
    return {
        "directories": {"raw_data": "raw", "processed_data": "processed", "log_dir": "logs"},
        "processing": {"incremental": False},
        "physics": {
            "vertical_resolution_m": 7.5,
            "channels": {
                "532.PC": {"deadtime_us": 0.0035, "bin_shift_bins": -3, "background_offset": 0.0015}
            },
        },
        "hardware": {"name_to_id": {"532.PC": 716}},
    }


def test_load_repository_config() -> None:
    """The repository config.yaml should load and expose required sections."""
    config = load_config("config.yaml")

    assert "directories" in config
    assert "processing" in config
    assert "physics" in config
    assert "hardware" in config
    assert config["physics"]["vertical_resolution_m"] > 0
    assert "532.PC" in config["physics"]["channels"]
    assert config["physics"]["channels"]["532.PC"]["bin_shift_bins"] == -3


def test_load_repository_config_injects_legacy_aliases() -> None:
    """Canonical YAML keys should be exposed with compatibility aliases in memory."""
    config = load_config("config.yaml")

    assert config["physics"]["speed_of_light"] == config["physics"]["speed_of_light_m_s"]
    assert config["physics"]["bg_start"] == config["physics"]["background_start_m"]
    assert config["physics"]["bg_stop"] == config["physics"]["background_stop_m"]
    assert config["radiosonde"]["fallback_to_standard"] is config["radiosonde"]["fallback_to_standard_atmosphere"]
    assert config["inversion"]["lidar_ratios"] == config["inversion"]["lidar_ratios_sr"]


def test_minimum_schema_rejects_missing_sections() -> None:
    """The lightweight schema should fail on incomplete configs."""
    with pytest.raises(KeyError):
        validate_config_minimum({"processing": {}, "physics": {}})


def test_schema_rejects_invalid_background_window() -> None:
    """Background stop altitude must be greater than the start altitude."""
    config = {
        "directories": {"raw_data": "raw", "processed_data": "processed", "log_dir": "logs"},
        "processing": {"incremental": True},
        "physics": {
            "vertical_resolution_m": 7.5,
            "background_start_m": 30000.0,
            "background_stop_m": 29000.0,
            "channels": {
                "532.PC": {"deadtime_us": 0.0035, "bin_shift_bins": -3, "background_offset": 0.0015}
            },
        },
        "hardware": {"name_to_id": {"day": {}, "night": {}}},
    }

    with pytest.raises(ValueError):
        validate_config_minimum(config)


def test_normalize_config_preserves_existing_aliases() -> None:
    """Explicit legacy aliases should not be overwritten during normalization."""
    config = {
        "directories": {"raw_data": "raw", "processed_data": "processed", "log_dir": "logs"},
        "processing": {"incremental": False},
        "physics": {
            "vertical_resolution_m": 7.5,
            "speed_of_light_m_s": 299792458.0,
            "speed_of_light": 1.0,
            "background_start_m": 29000.0,
            "background_stop_m": 30000.0,
            "channels": {
                "532.PC": {"deadtime_us": 0.0035, "bin_shift_bins": -3, "background_offset": 0.0015}
            },
        },
        "hardware": {"name_to_id": {"day": {}, "night": {}}},
    }

    normalized = normalize_config(config)

    assert normalized["physics"]["speed_of_light"] == 1.0
    assert normalized["physics"]["bg_start"] == 29000.0
    assert normalized["physics"]["bg_stop"] == 30000.0


def test_load_minimal_valid_config_from_tmp_path(tmp_path: Path) -> None:
    """A minimal valid YAML file should load through the public loader."""
    config_path = tmp_path / "config.yaml"
    config_payload = {
        "directories": {
            "raw_data": "raw",
            "processed_data": "processed",
            "log_dir": "logs",
        },
        "processing": {"incremental": False},
        "physics": {
            "vertical_resolution_m": 7.5,
            "channels": {
                "532.PC": {"deadtime_us": 0.0035, "bin_shift_bins": -3, "background_offset": 0.0015}
            },
        },
        "hardware": {"name_to_id": {"day": {}, "night": {}}},
    }
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

    config = load_config(config_path)

    assert config["directories"]["raw_data"] == "raw"
    assert config["physics"]["channels"]["532.PC"]["deadtime_us"] == 0.0035


def test_schema_rejects_invalid_io_related_option_types() -> None:
    """Optional IO-facing settings should be type-checked when present."""
    config = {
        "directories": {"raw_data": "raw", "processed_data": "processed", "log_dir": "logs"},
        "processing": {
            "incremental": False,
            "raw_scan_ignore_dirs": "bad",
        },
        "physics": {
            "vertical_resolution_m": 7.5,
            "channels": {"532.PC": [0.0035, -3, 0.0015]},
        },
        "hardware": {"name_to_id": {"day": {}, "night": {}}},
    }

    with pytest.raises(ValueError):
        validate_config_minimum(config)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf"), True])
def test_schema_rejects_nonfinite_and_boolean_numeric_values(invalid_value: object) -> None:
    config = _minimal_config()
    config["physics"]["vertical_resolution_m"] = invalid_value

    with pytest.raises(ValueError, match="finite|booleans"):
        validate_config_minimum(config)


@pytest.mark.parametrize(
    ("constants", "message"),
    [
        ([0.0035, -3], "named fields"),
        ([True, -3, 0.0], "booleans"),
        ([0.0035, 1.5, 0.0], "integer"),
        ([0.0035, -3, float("nan")], "finite"),
    ],
)
def test_schema_rejects_invalid_positional_channel_structures(constants: list[object], message: str) -> None:
    config = _minimal_config()
    config["physics"]["channels"]["532.PC"] = constants

    with pytest.raises(ValueError, match=message):
        validate_config_minimum(config)


def test_schema_reports_unknown_key_with_full_path() -> None:
    config = _minimal_config()
    config["processing"]["consol_level"] = "DEBUG"

    assert find_unknown_config_keys(config) == ("processing.consol_level",)
    with pytest.raises(ValueError, match="processing.consol_level"):
        validate_config_minimum(config)


def test_schema_rejects_incompatible_nested_structure_and_range_order() -> None:
    config = _minimal_config()
    config["inversion"] = {
        "molecular_fit": {"ref_alt_min_m": 5000.0, "ref_alt_max_m": 1000.0},
    }

    with pytest.raises(ValueError, match="ref_alt_max_m"):
        validate_config_minimum(config)


@pytest.mark.parametrize("removed_key", ["quarantine_spurious_files", "delete_spurious_files"])
def test_schema_rejects_removed_implicit_filesystem_actions(removed_key: str) -> None:
    config = _minimal_config()
    config["processing"][removed_key] = False

    with pytest.raises(ValueError, match="was removed"):
        validate_config_minimum(config)


def test_dormant_controls_are_validated_and_preserved_without_activation() -> None:
    config = _minimal_config()
    config["processing"].update({"interactive_qa": True, "max_workers_io": 2, "max_workers_cpu": 3})
    config["directories"]["site_output"] = "measurements"
    config["surface_weather"] = {"provider": "open-meteo", "fallback_to_config_defaults": True}
    config["inversion"] = {
        "enabled": False,
        "interactive_qa": True,
        "products": {"save_glued_signal": False},
        "cloud_screening": {"enabled": True, "min_altitude_m": 500.0, "max_altitude_m": 15000.0},
    }

    normalized = normalize_config(config)

    assert normalized["processing"]["max_workers_io"] == 2
    assert normalized["surface_weather"]["provider"] == "open-meteo"
    assert normalized["inversion"]["enabled"] is False
    assert normalized["inversion"]["products"]["save_glued_signal"] is False
    assert normalized["inversion"]["cloud_screening"]["enabled"] is True


def test_repository_named_channel_values_match_frozen_legacy_contract() -> None:
    config = load_config("config.yaml")

    assert config["physics"]["channels"] == _named_channels(LEGACY_CHANNELS)


def test_legacy_and_named_channels_normalize_to_identical_values() -> None:
    legacy = _minimal_config()
    legacy["physics"]["channels"] = {"532.PC": [0.0035, -3, 0.0015]}
    named = _minimal_config()
    named["physics"]["channels"] = {
        "532.PC": {"deadtime_us": 0.0035, "bin_shift_bins": -3, "background_offset": 0.0015}
    }

    with pytest.warns(DeprecationWarning, match="removed in 0.2.0"):
        normalized_legacy = normalize_config(legacy)
    normalized_named = normalize_config(named)

    assert normalized_legacy["physics"]["channels"] == normalized_named["physics"]["channels"]
    assert get_channel_constant(legacy["physics"]["channels"], "532.PC", logging.getLogger("test")) == (0.0035, -3, 0.0015)
    assert get_channel_constant(named["physics"]["channels"], "532.PC", logging.getLogger("test")) == (0.0035, -3, 0.0015)


@pytest.mark.parametrize(
    "constants",
    [
        {"deadtime_us": 0.0035, "bin_shift_bins": -3},
        {"deadtime_us": 0.0035, "bin_shift_bins": -3, "background_offset": 0.0, "extra": 1},
        {"deadtime_us": 0.0035, "bin_shift_bins": True, "background_offset": 0.0},
    ],
)
def test_schema_rejects_incomplete_unknown_or_boolean_named_channel_fields(constants: dict[str, object]) -> None:
    config = _minimal_config()
    config["physics"]["channels"]["532.PC"] = constants

    with pytest.raises(ValueError):
        validate_config_minimum(config)
