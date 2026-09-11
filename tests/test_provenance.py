"""Tests for reusable MILGRAU FAIR provenance helpers."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import netCDF4 as nc
import xarray as xr

from milgrau.provenance import configuration_provenance, file_sha256, write_netcdf_provenance


def test_file_sha256_hashes_exact_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    payload = b"level0:\n  value: 1\n"
    path.write_bytes(payload)
    assert file_sha256(path) == sha256(payload).hexdigest()


def test_configuration_provenance_uses_source_files_and_resolved_station(tmp_path: Path) -> None:
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

    assert attrs["processing_config_sha256"] == file_sha256(config_path)
    assert attrs["station_config_sha256"] == file_sha256(station_path)
    assert attrs["station_profile_id"] == "spu-raman-2018"
    assert attrs["instrument_calibration_id"] == "spu-channel-corrections-v1"


def test_netcdf_provenance_rehashes_current_recipe_and_inherits_station_identity(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    station_path = tmp_path / "station.yaml"
    config_path.write_text("processing:\n  incremental: true\n", encoding="utf-8")
    station_path.write_text("station:\n  id: spu\n", encoding="utf-8")
    output = tmp_path / "derived.nc"
    xr.Dataset({"value": (("x",), [1.0])}).to_netcdf(output)

    config = {
        "_config_file": str(config_path),
        "_station_config_path": str(station_path),
    }
    source_attrs = {
        "processing_config_sha256": "old-config-hash",
        "station_config_sha256": "old-station-hash",
        "station_profile_id": "spu-merionc-2024",
        "instrument_calibration_id": "spu-channel-corrections-v1",
    }

    written = write_netcdf_provenance(output, config, source_attrs=source_attrs)

    assert written["processing_config_sha256"] == file_sha256(config_path)
    assert written["station_config_sha256"] == file_sha256(station_path)
    assert written["station_profile_id"] == "spu-merionc-2024"
    assert written["instrument_calibration_id"] == "spu-channel-corrections-v1"
    with nc.Dataset(str(output)) as dataset:
        assert dataset.getncattr("processing_config_sha256") == file_sha256(config_path)
        assert dataset.getncattr("station_config_sha256") == file_sha256(station_path)
        assert dataset.getncattr("station_profile_id") == "spu-merionc-2024"
        assert dataset.getncattr("instrument_calibration_id") == "spu-channel-corrections-v1"
