"""SCC-only monthly aerosol lidar-ratio ancillary export regressions."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from milgrau.config.loader import load_config
from milgrau.config.scc import validate_scc_config
from milgrau.config.station import resolve_station_context
from milgrau.level0.scc_lidar_ratio import (
    climatology_ancillary_enabled,
    scc_lr_ancillary_matches,
    write_scc_lidar_ratio_ancillary,
)


def _setup(tmp_path: Path, *, regime: str = "day", month: int = 9):
    config = deepcopy(load_config("config.yaml"))
    catalog = yaml.safe_load(Path("scc.yaml").read_text(encoding="utf-8"))
    config["_scc_catalog"] = catalog
    catalog["scc_policy"]["lr_input"]["source"] = "climatology_ancillary"
    profile = catalog["profiles"]["spu-merionc-2024"]
    profile["day"]["elastic_product_ids"] = {
        "355": [10011], "532": [10012, 10013], "1064": [10014],
    }
    profile["night"]["elastic_product_ids"] = {"1064": [10015]}
    validate_scc_config(catalog, config["_station_catalog"])
    channels = (
        ["532.AN", "532.PC", "1064.AN", "355.PC", "355.AN"]
        if regime == "day"
        else [
            "532.AN", "532.PC", "1064.AN", "355.PC", "355.AN",
            "530.PC", "530.AN", "387.AN", "387.PC",
        ]
    )
    first = pd.Timestamp(f"2025-{month:02d}-13T15:49:00Z")
    context = resolve_station_context(
        config, first.to_pydatetime(), channels, mode=regime,
    )
    group = pd.DataFrame({
        "start_time_utc": [first, first + pd.Timedelta(minutes=5)],
        "stop_time": [
            first + pd.Timedelta(minutes=5),
            first + pd.Timedelta(minutes=10),
        ],
    })
    scc_channels = context["scc_channels"]
    lidar = {
        "channels": scc_channels,
        "tensors": {name: np.ones((2, 8)) for name in scc_channels},
        "channel_metadata": {
            name: {"bin_width_m": 7.5 if name.endswith(".AN") else 15.0}
            for name in scc_channels
        },
    }
    output = tmp_path / f"session_{regime}_L0_scc.nc"
    return config, context, group, lidar, output


def test_scc_monthly_climatology_writes_exact_ancillary_schema(tmp_path: Path) -> None:
    config, context, group, lidar, output = _setup(tmp_path, regime="day")
    assert climatology_ancillary_enabled(config)
    assert set(context["lr_input"].values()) == {0}

    path = write_scc_lidar_ratio_ancillary(output, group, lidar, config, context)
    assert path is not None
    assert path.name == "lr_20250913spu1549.nc"

    with nc.Dataset(path) as ds:
        assert ds.file_format == "NETCDF4_CLASSIC"
        assert ds.getncattr("Lidar_Station_Name") == "spu"
        assert ds.getncattr("Climatology_Month_UTC") == "09"
        assert ds.variables["product_ID"].dimensions == ("products",)
        assert ds.variables["Altitude"].dimensions == ("points",)
        assert ds.variables["Lidar_Ratio"].dimensions == ("products", "points")
        assert ds.variables["Lidar_Ratio_Error"].dimensions == ("products", "points")
        np.testing.assert_array_equal(
            ds.variables["product_ID"][:], [10011, 10012, 10013, 10014],
        )
        np.testing.assert_allclose(
            ds.variables["Lidar_Ratio"][:, 0],
            [84.69, 55.78, 55.78, 29.74],
        )
        np.testing.assert_allclose(
            ds.variables["Lidar_Ratio_Error"][:, 0], [15, 10, 10, 10],
        )
        assert np.all(np.diff(ds.variables["Altitude"][:]) > 0)
        assert ds.variables["Altitude"][-1] > 8 * 15

    attrs = {
        "Measurement_ID": "20250913spu1549",
        "RawData_Start_Date": "20250913",
        "LR_File_Name": path.name,
    }
    assert scc_lr_ancillary_matches(output, attrs, context, config["_station_catalog"])
    attrs["LR_File_Name"] = "lr_wrong.nc"
    assert not scc_lr_ancillary_matches(output, attrs, context, config["_station_catalog"])


def test_night_raman_wavelengths_are_not_overwritten(tmp_path: Path) -> None:
    config, context, group, lidar, output = _setup(tmp_path, regime="night", month=10)
    assert context["lr_input"] == {"1064.AN": 0}
    path = write_scc_lidar_ratio_ancillary(output, group, lidar, config, context)
    assert path is not None
    with nc.Dataset(path) as ds:
        np.testing.assert_array_equal(ds.variables["product_ID"][:], [10015])
        np.testing.assert_allclose(ds.variables["Lidar_Ratio"][:, :], 33.44)
        assert ds.dimensions["products"].size == 1


def test_missing_real_product_id_fails_before_creating_file(tmp_path: Path) -> None:
    config, context, group, lidar, output = _setup(tmp_path)
    context["elastic_product_ids"] = {}
    with pytest.raises(ValueError, match="product_ID"):
        write_scc_lidar_ratio_ancillary(output, group, lidar, config, context)
    assert not list(tmp_path.glob("lr_*.nc"))


def test_month_boundary_is_rejected_not_silently_assigned(tmp_path: Path) -> None:
    config, context, group, lidar, output = _setup(tmp_path)
    group.loc[1, "start_time_utc"] = pd.Timestamp("2025-10-01T00:00:00Z")
    group.loc[1, "stop_time"] = pd.Timestamp("2025-10-01T00:05:00Z")
    with pytest.raises(ValueError, match="two UTC calendar months"):
        write_scc_lidar_ratio_ancillary(output, group, lidar, config, context)


def test_scc_db_mode_does_not_write_ancillary(tmp_path: Path) -> None:
    config, context, group, lidar, output = _setup(tmp_path)
    config["_scc_catalog"]["scc_policy"]["lr_input"]["source"] = "scc_db"
    assert not climatology_ancillary_enabled(config)
    assert write_scc_lidar_ratio_ancillary(output, group, lidar, config, context) is None
    assert not list(tmp_path.glob("lr_*.nc"))
