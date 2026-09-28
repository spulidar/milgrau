"""Regression guards for the current Level 2 identity and QA outputs."""

from __future__ import annotations

import numpy as np
import xarray as xr

from milgrau.level2 import lebear
from milgrau.level2.config import get_kfs_mode, kfs_mode_description
from milgrau.level2.gluing import gluing_selection_score_metadata
from milgrau.provenance import file_sha256
from milgrau.scientific import (
    LEVEL2_PRODUCT_SCHEMA_VERSION,
    elastic_inversion_algorithm_metadata,
)
from milgrau.viz.level2_qa import plot_all_level2_qa


def test_productive_kfs_identity_is_two_sided() -> None:
    config = {"inversion": {"kfs_mode": "two_sided"}}

    assert get_kfs_mode(config) == "two_sided"
    assert "Two-sided Klett--Fernald" in kfs_mode_description("two_sided")
    metadata = elastic_inversion_algorithm_metadata()
    assert LEVEL2_PRODUCT_SCHEMA_VERSION == "6"
    assert metadata["integration_mode"] == "two_sided"
    assert metadata["uncertainty_method"] == "selection-aware Monte Carlo"


def test_level2_incremental_rejects_stale_schema_or_gluing_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "level1.nc"
    product_path = tmp_path / "level2.nc"
    input_path.write_text("synthetic source", encoding="utf-8")

    metadata = elastic_inversion_algorithm_metadata()
    gluing_metadata = gluing_selection_score_metadata()
    ds = xr.Dataset(
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
        attrs={
            "level2_product_schema_version": LEVEL2_PRODUCT_SCHEMA_VERSION,
            "source_level1_sha256": file_sha256(input_path),
            "product_completeness": "complete",
            "product_status": "success",
            **metadata,
            **gluing_metadata,
        },
    )
    ds.to_netcdf(product_path)

    monkeypatch.setattr(lebear, "get_wavelengths_to_process", lambda _config: [532])
    monkeypatch.setattr(lebear, "get_kfs_mode", lambda _config: "two_sided")
    monkeypatch.setattr(
        lebear, "validate_level2_contract", lambda _dataset: None
    )
    monkeypatch.setattr(lebear, "output_is_current", lambda *args, **kwargs: True)

    assert lebear.level2_output_is_current(input_path, product_path, {}) is True

    del ds.attrs["level2_product_schema_version"]
    ds.to_netcdf(product_path)
    assert lebear.level2_output_is_current(input_path, product_path, {}) is False

    ds.attrs["level2_product_schema_version"] = LEVEL2_PRODUCT_SCHEMA_VERSION
    ds.attrs["gluing_selection_score_version"] = "stale"
    ds.to_netcdf(product_path)
    assert lebear.level2_output_is_current(input_path, product_path, {}) is False

    ds.attrs.update(gluing_metadata)
    ds.attrs["integration_mode"] = "backward"
    ds.to_netcdf(product_path)
    assert lebear.level2_output_is_current(input_path, product_path, {}) is False


def test_level2_qa_does_not_write_redundant_status_txt(tmp_path) -> None:
    ds_l2 = xr.Dataset(coords={"wavelength": np.array([532], dtype=np.int32)})
    config = {
        "visualization": {
            "level2_qa": {
                "generate_gluing_qa": False,
                "generate_molecular_fit_qa": False,
                "generate_scattering_ratio_qa": False,
                "generate_kfs_qa": False,
            }
        }
    }

    generated = plot_all_level2_qa(
        ds_l2=ds_l2,
        output_folder=tmp_path,
        file_name_prefix="synthetic",
        config=config,
        root_dir=tmp_path,
    )

    assert generated == []
    assert list(tmp_path.glob("QA_L2_Product_Status_*.txt")) == []
