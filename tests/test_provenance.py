"""Tests for reusable MILGRAU FAIR provenance helpers."""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import xarray as xr

from milgrau.provenance import configuration_provenance, write_netcdf_provenance
from milgrau.version import __version__


def test_configuration_provenance_is_human_readable(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    station_path = tmp_path / "station.yaml"
    config_path.write_text("processing: {}\n", encoding="utf-8")
    station_path.write_text("station: {}\n", encoding="utf-8")
    config = {
        "_config_file": str(config_path),
        "_station_config_path": str(station_path),
        "_resolved_station": {
            "profile_id": "spu-raman-2018",
            "calibration_id": "spu-channel-corrections-v1",
        },
    }

    attrs = configuration_provenance(config)

    assert attrs["software_name"] == "MILGRAU"
    assert attrs["software_version"] == __version__
    assert attrs["processing_configuration_file"] == "config.yaml"
    assert attrs["station_configuration_file"] == "station.yaml"
    assert attrs["station_profile_id"] == "spu-raman-2018"
    assert attrs["instrument_calibration_id"] == "spu-channel-corrections-v1"
    assert not any("sha" in key.lower() for key in attrs)


def test_netcdf_provenance_embeds_exact_yaml_and_inherits_station_identity(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    station_path = tmp_path / "station.yaml"
    config_text = "processing:\n  incremental: true\n"
    station_text = "station:\n  id: spu\n"
    config_path.write_text(config_text, encoding="utf-8")
    station_path.write_text(station_text, encoding="utf-8")
    output = tmp_path / "derived.nc"
    xr.Dataset({"value": (("x",), [1.0])}).to_netcdf(output)

    config = {"_config_file": str(config_path), "_station_config_path": str(station_path)}
    source_attrs = {
        "station_profile_id": "spu-merionc-2024",
        "instrument_calibration_id": "spu-channel-corrections-v1",
    }

    written = write_netcdf_provenance(output, config, source_attrs=source_attrs)

    assert written["software_version"] == __version__
    assert written["station_profile_id"] == "spu-merionc-2024"
    assert written["instrument_calibration_id"] == "spu-channel-corrections-v1"
    with nc.Dataset(str(output)) as dataset:
        assert dataset.getncattr("software_name") == "MILGRAU"
        assert dataset.getncattr("software_version") == __version__
        assert dataset.variables["processing_configuration_yaml"].getValue() == config_text
        assert dataset.variables["station_configuration_yaml"].getValue() == station_text
        assert dataset.variables["processing_configuration_yaml"].getncattr("media_type") == "application/yaml"
