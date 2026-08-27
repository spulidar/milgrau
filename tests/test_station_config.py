"""Tests for compact station.yaml temporal and SCC resolution."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from milgrau.config.loader import load_config
from milgrau.config.station import apply_station_context, resolve_station_context, select_lidar_channels


def _context(config: dict, when: str, period: str, channels: list[str]) -> dict:
    return resolve_station_context(
        config,
        datetime.fromisoformat(when).replace(tzinfo=timezone.utc),
        period,
        channels,
    )


def test_repository_station_catalog_covers_all_scc_eras() -> None:
    config = load_config("config.yaml")

    legacy_day = _context(
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

    assert legacy_day["profile_id"] == "spu-apel-2017"
    assert legacy_day["scc_configuration_id"] == 248
    assert raman_night["profile_id"] == "spu-raman-2018"
    assert raman_night["scc_configuration_id"] == 484
    assert merion_day["profile_id"] == "spu-merionc-2024"
    assert merion_day["scc_configuration_id"] == 1047
    assert merion_day["channel_ids"]["532.AN"] == 4069
    assert merion_day["channel_ids"]["355.AN"] == 4073


def test_merion_night_configuration_contains_raman_channels() -> None:
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


def test_station_resolver_rejects_scc_configuration_with_missing_raw_channel() -> None:
    config = load_config("config.yaml")
    with pytest.raises(ValueError, match="requires channels missing"):
        _context(
            config,
            "2025-01-01T12:00:00",
            "am",
            ["532.AN", "532.PC", "1064.AN", "355.PC"],
        )


def test_station_context_applies_profile_altitude_and_flat_channel_map() -> None:
    config = load_config("config.yaml")
    channels = ["532.AN", "532.PC", "1064.AN", "355.PC", "355.AN"]
    context = _context(config, "2025-01-01T12:00:00", "am", channels)
    resolved = apply_station_context(config, context)

    assert resolved["site"]["station_altitude_m"] == 740.0
    assert resolved["physics"]["station_altitude_m"] == 740.0
    assert resolved["hardware"]["name_to_id"] == {
        "532.AN": 4069,
        "532.PC": 4070,
        "1064.AN": 4071,
        "355.PC": 4072,
        "355.AN": 4073,
    }


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
