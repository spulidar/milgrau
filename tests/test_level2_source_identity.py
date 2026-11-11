"""Regression tests for exact Level 1 content identity in provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import xarray as xr

import milgrau.level2.lebear as lebear
from milgrau.level2.gluing import gluing_selection_score_metadata
from milgrau.provenance import file_sha256
from milgrau.scientific import (
    LEVEL2_PRODUCT_SCHEMA_VERSION,
    elastic_inversion_algorithm_metadata,
)


def test_file_sha256_identifies_exact_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    payload = b"MILGRAU\x00Level1\n"
    source.write_bytes(payload)

    assert file_sha256(source) == hashlib.sha256(payload).hexdigest()


def _write_currentness_fixture(source: Path, output: Path) -> None:
    attrs = {
        "level2_product_schema_version": LEVEL2_PRODUCT_SCHEMA_VERSION,
        "source_level1_sha256": file_sha256(source),
        "product_completeness": "complete",
        "product_status": "success",
        **elastic_inversion_algorithm_metadata(),
        **gluing_selection_score_metadata(),
    }
    xr.Dataset(
        data_vars={
            "requested_wavelengths": (
                ("requested_wavelength",),
                np.array([532], dtype=np.int32),
            ),
            "processed_wavelengths": (
                ("processed_wavelength",),
                np.array([532], dtype=np.int32),
            ),
            "failed_wavelengths": (
                ("failed_wavelength",),
                np.array([], dtype=np.int32),
            ),
        },
        attrs=attrs,
    ).to_netcdf(output)


def test_level2_currentness_rejects_changed_bytes_even_when_other_checks_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "input_level1.nc"
    output = tmp_path / "output_level2.nc"
    xr.Dataset(attrs={"thermodynamic_source_priority": "era5,ussa76"}).to_netcdf(source)
    _write_currentness_fixture(source, output)

    monkeypatch.setattr(
        lebear, "validate_level2_contract", lambda _ds: None
    )
    monkeypatch.setattr(lebear, "get_wavelengths_to_process", lambda _config: [532])
    monkeypatch.setattr(lebear, "get_kfs_mode", lambda _config: "two_sided")
    monkeypatch.setattr(lebear, "output_is_current", lambda *args, **kwargs: True)

    assert lebear.level2_output_is_current(source, output, {})

    original_stat = source.stat()
    xr.Dataset(attrs={
        "thermodynamic_source_priority": "era5,ussa76",
        "updated": "new-scientific-bytes",
    }).to_netcdf(source)
    source.touch()
    source.chmod(original_stat.st_mode)
    # Preserve an old-enough mtime so timestamp-only reuse would not detect the change.
    import os

    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert not lebear.level2_output_is_current(source, output, {})


def test_l2_radiosonde_input_reaches_normal_contract_validation(
    tmp_path: Path, monkeypatch,
) -> None:
    import logging
    from milgrau.operations import ExecutionStatus

    source = tmp_path / "spu_20240902-2100Z_20240903-0400Z_L1.nc"
    xr.Dataset(attrs={
        "level1_atmosphere_mode": "radiosonde_only",
        "thermodynamic_source_priority": "radiosonde",
        "thermodynamic_sources_present": "radiosonde",
    }).to_netcdf(source)

    reached_validation = []

    def check_contract(_ds: xr.Dataset) -> None:
        reached_validation.append(True)
        raise ValueError("contract validation reached")

    monkeypatch.setattr(lebear, "validate_level1_contract", check_contract)
    result = lebear.process_single_level1_file(
        source, {}, logging.getLogger("test.l2.radio.allowed"),
    )
    failures = [entry for entry in result.results if entry.status is ExecutionStatus.ERROR]
    assert reached_validation == [True]
    assert len(failures) == 1
    assert failures[0].stage == "level2.validation.input"
    assert "contract validation reached" in str(failures[0].cause)
    assert not source.with_name(source.name.replace("_L1.nc", "_L2.nc")).exists()


def test_l2_can_reuse_current_radio_only_product_with_matching_l1_hash(
    tmp_path: Path, monkeypatch,
) -> None:
    source = tmp_path / "source_L1.nc"
    output = tmp_path / "current_L2.nc"
    xr.Dataset(attrs={
        "level1_atmosphere_mode": "radiosonde_only",
        "thermodynamic_source_priority": "radiosonde",
        "thermodynamic_sources_present": "radiosonde",
    }).to_netcdf(source)
    _write_currentness_fixture(source, output)
    monkeypatch.setattr(lebear, "validate_level2_contract", lambda _ds: None)
    monkeypatch.setattr(lebear, "get_wavelengths_to_process", lambda _config: [532])
    monkeypatch.setattr(lebear, "get_kfs_mode", lambda _config: "two_sided")
    monkeypatch.setattr(lebear, "output_is_current", lambda *args, **kwargs: True)
    assert lebear.level2_output_is_current(source, output, {})


def test_l2_records_actual_radiosonde_atmosphere_and_mode() -> None:
    ds = xr.Dataset(
        data_vars={"Atmospheric_Source_Type": (
            ("atmosphere_time",), np.array(["radiosonde", "radiosonde"])
        )},
        attrs={
            "level1_atmosphere_mode": "radiosonde_only",
            "thermodynamic_source_priority": "radiosonde",
            "thermodynamic_sources_present": "radiosonde",
        },
    )
    attrs = lebear._level2_atmosphere_provenance(ds)
    assert attrs["level2_atmospheric_profile_source"] == "radiosonde"
    assert attrs["source_level1_atmosphere_mode"] == "radiosonde_only"
    assert attrs["source_thermodynamic_source_priority"] == "radiosonde"
    assert attrs["source_thermodynamic_sources_present"] == "radiosonde"


def test_l2_records_standard_era5_source_without_changing_preference() -> None:
    ds = xr.Dataset(
        data_vars={"Atmospheric_Source_Type": (
            ("atmosphere_time",), np.array(["era5", "ussa76"])
        )},
        attrs={
            "level1_atmosphere_mode": "standard",
            "thermodynamic_source_priority": "era5,ussa76",
            "thermodynamic_sources_present": "era5,ussa76",
        },
    )
    attrs = lebear._level2_atmosphere_provenance(ds)
    assert attrs["level2_atmospheric_profile_source"] == "era5,ussa76"
    assert attrs["source_level1_atmosphere_mode"] == "standard"
    assert attrs["source_thermodynamic_source_priority"] == "era5,ussa76"


def test_l2_supports_older_radiosonde_provenance_without_mode_marker() -> None:
    ds = xr.Dataset(attrs={
        "thermodynamic_source_priority": "radiosonde",
        "thermodynamic_sources_present": "radiosonde",
    })
    attrs = lebear._level2_atmosphere_provenance(ds)
    assert attrs["level2_atmospheric_profile_source"] == "radiosonde"
    assert attrs["source_level1_atmosphere_mode"] == "unspecified"
