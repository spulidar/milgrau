"""Tests for station temporal, calibration, and SCC resolution."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pytest

from milgrau.config.loader import load_config
from milgrau.config.station import (
    resolve_station_context,
    select_lidar_channels,
    validate_station_config,
)


def _context(config: dict, when: str, period: str, channels: list[str]) -> dict:
    return resolve_station_context(
        config,
        datetime.fromisoformat(when).replace(tzinfo=timezone.utc),
        period,
        channels,
    )


def test_repository_station_catalog_covers_all_scc_eras() -> None:
    config = load_config("config.yaml")

    apel_day = _context(
        config,
        "2017-10-01T12:00:00",
        "pm",
        ["532.PC", "532.AN", "355.PC", "355.AN", "607.AN", "607.PC", "387.PC", "387.AN", "1064.AN", "1064.PC", "408.AN", "408.PC"],
    )
    raman_night = _context(
        config,
        "2019-06-01T02:00:00",
        "nt",
        ["532.PC", "532.AN", "355.PC", "355.AN", "387.PC", "387.AN", "1064.AN", "1064.PC", "408.AN", "408.PC", "530.PC", "530.AN"],
    )
    merion_day = _context(
        config,
        "2025-01-01T15:00:00",
        "pm",
        ["532.AN", "532.PC", "1064.AN", "355.PC", "355.AN"],
    )

    assert apel_day["profile_id"] == "spu-apel-2017"
    assert apel_day["scc_configuration_id"] == 248
    assert apel_day["lr_input"] == {"1064.AN": 1, "1064.PC": 1}
    assert raman_night["profile_id"] == "spu-raman-2018"
    assert raman_night["scc_configuration_id"] == 484
    assert raman_night["lr_input"] == {"1064.AN": 1, "1064.PC": 1}
    assert merion_day["profile_id"] == "spu-merionc-2024"
    assert merion_day["scc_configuration_id"] == 1047
    assert merion_day["channel_ids"]["532.AN"] == 4069
    assert merion_day["channel_ids"]["355.AN"] == 4073
    assert merion_day["scc_export_ready"] is True
    assert merion_day["lr_input"] == {
        "532.AN": 1,
        "532.PC": 1,
        "1064.AN": 1,
        "355.PC": 1,
        "355.AN": 1,
    }


def test_profiles_resolve_named_instrument_calibration() -> None:
    config = load_config("config.yaml")
    contexts = [
        _context(config, "2015-06-01T12:00:00", "pm", ["355.PC"]),
        _context(config, "2017-10-01T12:00:00", "pm", ["355.PC"]),
        _context(config, "2019-06-01T12:00:00", "pm", ["355.PC"]),
        _context(config, "2025-01-01T12:00:00", "pm", ["355.PC"]),
    ]

    assert {ctx["calibration_id"] for ctx in contexts} == {"spu-channel-corrections-v1"}
    for ctx in contexts:
        assert ctx["calibration_provenance"]["source"] == "migrated_from_legacy_global_channel_corrections"
        assert ctx["channel_calibrations"]["355.PC"]["deadtime_us"] == 0.002
        assert ctx["channel_calibrations"]["355.PC"]["saturation"] == {"status": "not_characterized"}


def test_pre_scc_measurement_uses_legacy_profile_without_scc_mapping() -> None:
    config = load_config("config.yaml")
    channels = ["355.AN", "355.PC", "532.AN", "532.PC", "1064.AN"]
    context = _context(config, "2015-06-01T12:00:00", "pm", channels)

    assert context["profile_id"] == "spu-legacy"
    assert context["calibration_id"] == "spu-channel-corrections-v1"
    assert context["scc_available"] is False
    assert context["scc_export_ready"] is False
    assert context["scc_configuration_id"] is None
    assert context["channel_ids"] == {}
    assert context["lr_input"] == {}
    assert context["selected_channels"] == channels


def test_merion_night_configuration_contains_raman_channels_and_only_1064_lr_input() -> None:
    config = load_config("config.yaml")
    channels = [
        "532.AN", "532.PC", "1064.AN", "355.PC", "355.AN",
        "530.PC", "530.AN", "387.AN", "387.PC",
    ]
    context = _context(config, "2025-01-01T23:00:00", "nt", channels)

    assert context["scc_configuration_id"] == 1046
    assert context["channel_ids"]["530.PC"] == 4074
    assert context["channel_ids"]["530.AN"] == 4075
    assert context["channel_ids"]["387.AN"] == 4076
    assert context["channel_ids"]["387.PC"] == 4077
    assert context["lr_input"] == {"1064.AN": 1}


def test_raman_2018_day_uses_raman_for_355_532_and_fixed_lr_only_for_1064() -> None:
    config = load_config("config.yaml")
    channels = ["1064.AN", "532.AN", "355.AN", "530.AN", "387.AN"]
    context = _context(config, "2019-06-01T15:00:00", "pm", channels)

    assert context["scc_configuration_id"] == 565
    assert context["lr_input"] == {"1064.AN": 1}


def test_station_resolver_preserves_all_raw_channels_and_separates_scc_subset() -> None:
    config = load_config("config.yaml")
    channels = [
        "532.AN", "532.PC", "1064.AN", "355.PC", "355.AN",
        "530.PC", "530.AN", "387.AN", "387.PC",
    ]
    context = _context(config, "2025-01-01T12:00:00", "am", channels)

    assert context["selected_channels"] == channels
    assert context["scc_channels"] == ["532.AN", "532.PC", "1064.AN", "355.PC", "355.AN"]
    assert context["extra_channels"] == ["530.PC", "530.AN", "387.AN", "387.PC"]
    assert context["missing_scc_channels"] == []
    assert context["scc_export_ready"] is True


def test_missing_scc_channel_disables_only_scc_export() -> None:
    config = load_config("config.yaml")
    channels = ["532.AN", "532.PC", "1064.AN", "355.PC"]
    context = _context(config, "2025-01-01T12:00:00", "am", channels)

    assert context["selected_channels"] == channels
    assert context["missing_scc_channels"] == ["355.AN"]
    assert context["scc_export_ready"] is False
    assert context["scc_available"] is True


def test_station_context_is_self_contained_for_productive_consumers() -> None:
    config = load_config("config.yaml")
    channels = ["532.AN", "532.PC", "1064.AN", "355.PC", "355.AN"]
    context = _context(config, "2025-01-01T12:00:00", "am", channels)

    assert context["site"]["station_altitude_m"] == 740.0
    assert context["channel_ids"] == {
        "532.AN": 4069,
        "532.PC": 4070,
        "1064.AN": 4071,
        "355.PC": 4072,
        "355.AN": 4073,
    }
    assert context["channel_calibrations"]["532.PC"] == {
        "detector_mode": "photon_counting",
        "deadtime_us": 0.0035,
        "bin_shift_bins": -3,
        "background_offset": 0.0,
        "saturation": {"status": "not_characterized"},
    }
    assert context["calibration_id"] == "spu-channel-corrections-v1"
    assert context["lr_input"]["532.AN"] == 1
    assert "hardware" not in config
    assert "site" not in config
    assert "channels" not in config["physics"]


def test_station_catalog_declares_vertical_pointing_geometry() -> None:
    config = load_config("config.yaml")
    geometry = config["_station_catalog"]["station"]["lidar_geometry"]
    assert geometry == {"pointing_angle_deg_from_zenith": 0.0}


def test_unknown_profile_calibration_is_rejected() -> None:
    config = load_config("config.yaml")
    catalog = deepcopy(config["_station_catalog"])
    catalog["profiles"][0]["calibration_id"] = "does-not-exist"

    with pytest.raises(ValueError, match="unknown calibration"):
        validate_station_config(catalog)


def test_photon_counting_saturation_status_must_be_explicit() -> None:
    config = load_config("config.yaml")
    catalog = deepcopy(config["_station_catalog"])
    del catalog["calibrations"]["spu-channel-corrections-v1"]["channels"]["532.PC"]["saturation"]

    with pytest.raises(ValueError, match="must contain exactly"):
        validate_station_config(catalog)


def test_characterized_saturation_requires_positive_rate() -> None:
    config = load_config("config.yaml")
    catalog = deepcopy(config["_station_catalog"])
    catalog["calibrations"]["spu-channel-corrections-v1"]["channels"]["532.PC"]["saturation"] = {
        "status": "characterized",
        "max_rate_mhz": 0.0,
    }

    with pytest.raises(ValueError, match="max_rate_mhz must be positive"):
        validate_station_config(catalog)


def test_select_lidar_channels_reindexes_laser_shots() -> None:
    lidar_data = {
        "channels": ["532.AN", "530.AN", "532.PC"],
        "tensors": {
            "532.AN": np.ones((2, 3)),
            "530.AN": np.ones((2, 3)) * 2,
            "532.PC": np.ones((2, 3)) * 3,
        },
        "laser_shots": np.array([[10, 20, 30], [11, 21, 31]], dtype=np.int32),
        "channel_metadata": {
            "532.AN": {"is_pc": False},
            "530.AN": {"is_pc": False},
            "532.PC": {"is_pc": True},
        },
    }

    selected = select_lidar_channels(lidar_data, ["532.AN", "532.PC"])

    assert selected["channels"] == ["532.AN", "532.PC"]
    np.testing.assert_array_equal(selected["laser_shots"], np.array([[10, 30], [11, 31]], dtype=np.int32))
    assert set(selected["tensors"]) == {"532.AN", "532.PC"}
