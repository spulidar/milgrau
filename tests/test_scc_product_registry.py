"""SCC registry: verified products, shared identities and LR targeting by type."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest
import yaml

from milgrau.config.loader import load_config
from milgrau.config.scc import validate_scc_config
from milgrau.config.station import resolve_station_context
from milgrau.level0.scc_lidar_ratio import write_scc_lidar_ratio_ancillary

SYSTEMS = {
    ("spu-apel-2017", "day"): (248, {626, 634, 698, 701, 746, 754, 3683}),
    ("spu-apel-2017", "night"): (294, {626, 634, 698, 701, 702, 746, 754, 819, 820, 3683}),
    ("spu-raman-2018", "day"): (565, {1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717}),
    ("spu-raman-2018", "night"): (484, {626, 634, 698, 702, 746, 819, 1419, 1420, 1601}),
    ("spu-merionc-2024", "day"): (1047, {3683, 3684, 3685}),
    ("spu-merionc-2024", "night"): (1046, {3683, 3688, 3689, 3690, 3691, 3692, 3693}),
}


def _setup() -> dict:
    config = load_config("config.yaml")
    scc = yaml.safe_load(Path("scc.yaml").read_text(encoding="utf-8"))
    validate_scc_config(scc, config["_station_catalog"])
    config["_scc_catalog"] = scc
    return config


def test_all_30_verified_products_and_six_historical_systems() -> None:
    config = _setup()
    catalog = config["_scc_catalog"]
    products = catalog["products"]
    assert len(products) == 30
    assert {kind["type"] for kind in products.values()} == {
        "elastic_backscatter", "raman_backscatter",
        "lidar_ratio_extinction", "extinction_only",
    }
    assert products[3683] == {
        "type": "elastic_backscatter", "wavelength_nm": 1064,
        "usecase": 0, "channel_count": 1,
    }
    assert products[3692]["type"] == "lidar_ratio_extinction"
    assert products[3689]["parent_product_id"] == 3692
    assert products[3688]["parent_product_id"] == 3693
    for (profile, mode), (configuration_id, expected_ids) in SYSTEMS.items():
        system = catalog["profiles"][profile][mode]
        assert system["configuration_id"] == configuration_id
        assert set(system["product_ids"]) == expected_ids


@pytest.mark.parametrize(
    ("profile", "mode", "when", "expected"),
    [
        ("spu-apel-2017", "day", "2017-10-01T15:00:00Z", {"1064": [746, 3683]}),
        ("spu-apel-2017", "night", "2017-10-01T02:00:00Z", {"1064": [746, 3683]}),
        ("spu-raman-2018", "day", "2019-06-01T15:00:00Z", {"1064": [1710]}),
        ("spu-raman-2018", "night", "2019-06-01T02:00:00Z", {"1064": [746]}),
        ("spu-merionc-2024", "day", "2025-09-13T15:49:00Z",
         {"355": [3685], "532": [3684], "1064": [3683]}),
        ("spu-merionc-2024", "night", "2025-09-13T02:00:00Z", {"1064": [3683]}),
    ],
)
def test_external_lr_targets_only_elastic_products(
    profile: str, mode: str, when: str, expected: dict[str, list[int]],
) -> None:
    config = _setup()
    system = config["_scc_catalog"]["profiles"][profile][mode]
    context = resolve_station_context(
        config, datetime.fromisoformat(when.replace("Z", "+00:00")),
        list(system["channels"]), mode=mode,
    )
    assert context["profile_id"] == profile
    assert context["scc_product_ids"] == system["product_ids"]
    assert all(value == 0 for value in context["lr_input"].values())
    targets = {
        wavelength: ids for wavelength, ids in context["elastic_product_ids"].items()
        if any(str(channel).startswith(wavelength + ".") and value == 0
               for channel, value in context["lr_input"].items())
    }
    assert targets == expected
    for ids in targets.values():
        for pid in ids:
            assert context["scc_products"][pid]["type"] == "elastic_backscatter"


def test_shared_1064_variants_are_distinct_products_not_arbitrarily_selected(
    tmp_path: Path,
) -> None:
    config = _setup()
    system = config["_scc_catalog"]["profiles"]["spu-apel-2017"]["day"]
    start = pd.Timestamp("2017-10-01T15:00:00Z")
    context = resolve_station_context(
        config, start.to_pydatetime(), list(system["channels"]), mode="day",
    )
    group = pd.DataFrame({
        "start_time_utc": [start], "stop_time": [start + pd.Timedelta(minutes=1)],
    })
    channels = context["scc_channels"]
    lidar = {
        "channels": channels,
        "tensors": {c: np.ones((1, 8)) for c in channels},
        "channel_metadata": {c: {"bin_width_m": 15.0} for c in channels},
    }
    path = write_scc_lidar_ratio_ancillary(
        tmp_path / "historical_day_L0_scc.nc", group, lidar, config, context,
    )
    assert path is not None
    with nc.Dataset(path) as ds:
        np.testing.assert_array_equal(ds["product_ID"][:], [746, 3683])
        np.testing.assert_allclose(
            ds["Lidar_Ratio"][:, 0], [config["inversion"]["lidar_ratios_sr"]["1064"]["10"]] * 2,
        )


@pytest.mark.parametrize("bad", ["unknown", "duplicate", "kind", "missing_parent", "mixed_wavelength"])
def test_product_registry_rejects_unverified_or_corrupt_mappings(bad: str) -> None:
    config = _setup()
    catalog = deepcopy(config["_scc_catalog"])
    system = catalog["profiles"]["spu-merionc-2024"]["day"]
    if bad == "unknown":
        system["product_ids"].append(999999)
    elif bad == "duplicate":
        system["product_ids"].append(3683)
    elif bad == "kind":
        catalog["products"][3683]["type"] = "unknown_type"
    elif bad == "missing_parent":
        catalog["products"][3689]["parent_product_id"] = 999999
    elif bad == "mixed_wavelength":
        catalog["products"][3683]["wavelength_nm"] = 1234
    with pytest.raises(ValueError):
        validate_scc_config(catalog, config["_station_catalog"])


def test_product_parent_may_be_shared_outside_its_system_membership() -> None:
    config = _setup()
    system = config["_scc_catalog"]["profiles"]["spu-apel-2017"]["day"]
    assert 626 in system["product_ids"]
    assert 819 not in system["product_ids"]
    assert config["_scc_catalog"]["products"][626]["parent_product_id"] == 819


def test_lidar_ratio_numeric_climatology_is_not_duplicated_in_scc_yaml() -> None:
    config = _setup()
    assert "monthly_sr" not in config["_scc_catalog"]
    assert "std_sr" not in config["_scc_catalog"]
    assert config["_station_catalog"]["station"]["lidar_ratio_climatology"]["monthly_sr"]["1064"]["09"] == 29.74
    assert config["_scc_catalog"]["scc_policy"]["lr_input"]["source"] == "climatology_ancillary"
