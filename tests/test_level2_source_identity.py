"""Regression tests for exact Level 1 content identity in method-v5 provenance."""

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
    LEVEL2_RETRIEVAL_METHOD_VERSION,
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
        "level2_retrieval_method_version": LEVEL2_RETRIEVAL_METHOD_VERSION,
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
    source.write_bytes(b"original-level1-content")
    _write_currentness_fixture(source, output)

    monkeypatch.setattr(
        lebear, "validate_method_v5_level2_contract", lambda _ds: None
    )
    monkeypatch.setattr(lebear, "get_wavelengths_to_process", lambda _config: [532])
    monkeypatch.setattr(lebear, "output_is_current", lambda *args, **kwargs: True)

    assert lebear.level2_output_is_current(source, output, {})

    original_stat = source.stat()
    source.write_bytes(b"changed-level1-content!")
    source.touch()
    source.chmod(original_stat.st_mode)
    # Preserve an old-enough mtime so timestamp-only reuse would not detect the change.
    import os

    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert not lebear.level2_output_is_current(source, output, {})
