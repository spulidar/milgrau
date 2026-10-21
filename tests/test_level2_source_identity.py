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


def test_l2_rejects_radiosonde_l1_before_contract_or_retrieval(
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
    monkeypatch.setattr(
        lebear, "validate_level1_contract",
        lambda _ds: __import__("pytest").fail("L2 must reject before signal validation."),
    )
    result = lebear.process_single_level1_file(
        source, {}, logging.getLogger("test.l2.reject.radio"),
    )
    failures = [entry for entry in result.results if entry.status is ExecutionStatus.ERROR]
    assert len(failures) == 1
    assert failures[0].stage == "level2.validation.input"
    assert failures[0].cause is not None
    assert "milgrau-lipancora -i SESSION_ID" in str(failures[0].cause)
    assert not source.with_name(source.name.replace("_L1.nc", "_L2.nc")).exists()


def test_l2_cannot_skip_existing_current_product_if_l1_became_radio_only(
    tmp_path: Path, monkeypatch,
) -> None:
    source = tmp_path / "source_L1.nc"
    output = tmp_path / "old_L2.nc"
    xr.Dataset(attrs={
        "level1_atmosphere_mode": "radiosonde_only",
        "thermodynamic_source_priority": "radiosonde",
    }).to_netcdf(source)
    _write_currentness_fixture(source, output)
    monkeypatch.setattr(lebear, "output_is_current", lambda *args, **kwargs: True)
    assert not lebear.level2_output_is_current(source, output, {})


def test_l2_rejects_older_radiosonde_only_file_without_new_mode_marker() -> None:
    import pytest

    with pytest.raises(ValueError, match="Rerun 'milgrau-lipancora"):
        lebear._reject_radiosonde_only_l1(
            xr.Dataset(attrs={
                "thermodynamic_source_priority": "radiosonde",
                "thermodynamic_sources_present": "radiosonde",
            })
        )
