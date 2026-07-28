"""SCI-004B strict, credential-free meteorology configuration."""

from __future__ import annotations

from copy import deepcopy

import pytest

from milgrau.config.loader import load_config
from milgrau.config.schema import validate_config_minimum


def test_repository_meteorology_config_is_valid_and_contains_no_credentials() -> None:
    config = load_config()
    meteorology = config["meteorology"]
    assert meteorology["acquisition_mode"] == "auto"
    assert meteorology["radiosonde"]["station_id"] == "83779"
    assert meteorology["era5"]["levels"] == "1-137"
    assert not any(
        term in str(meteorology).lower()
        for term in ("token", "password", "api_key", "authorization")
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("acquisition_mode",), "sometimes", "auto"),
        (("timeout_seconds",), 0.0, "positive"),
        (("max_retries",), 0, "at least"),
        (("radiosonde", "station_id"), "SBMT", "digits"),
        (("era5", "levels"), "1-136", "1-137"),
        (("era5", "raw_format"), "netcdf", "grib"),
        (("era5", "variables"), ["temperature"], "fixed"),
    ],
)
def test_meteorology_config_rejects_contract_drift(path, value, message) -> None:
    config = load_config()
    working = deepcopy(config)
    target = working["meteorology"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_config_minimum(working)


def test_meteorology_config_rejects_unknown_keys() -> None:
    config = load_config()
    config["meteorology"]["era5"]["credential_file"] = "secret"
    with pytest.raises(ValueError, match="credential_file"):
        validate_config_minimum(config)
