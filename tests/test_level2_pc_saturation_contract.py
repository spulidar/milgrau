"""Tests for the Level 1 -> Level 2 PC saturation-characterization contract."""

from __future__ import annotations

import numpy as np
import xarray as xr

from milgrau.level2.retrieval import WavelengthBlockInputs, _enforce_pc_saturation_characterization


def _inputs() -> WavelengthBlockInputs:
    return WavelengthBlockInputs(
        wavelength_nm=532,
        analog_channel="532.AN",
        photon_channel="532.PC",
        n_time=1,
        n_altitude=3,
        block_time=np.array([np.datetime64("2024-01-01")]),
        block_groups=[np.array([0])],
        gluing_config={},
        molecular_fit_config={},
        analog_block=np.ones((1, 3)),
        analog_error_block=np.ones((1, 3)),
        analog_correction_valid=True,
        photon_block=np.ones((1, 3)),
        photon_error_block=np.ones((1, 3)),
        photon_mask_block=np.zeros((1, 3)),
        photon_correction_valid=True,
    )


def test_uncharacterized_pc_cannot_be_scientifically_accepted_by_level2() -> None:
    ds = xr.Dataset(
        {"pc_saturation_characterized": (("channel",), np.array([0, 0], dtype=np.int8))},
        coords={"channel": ["532.PC", "532.AN"]},
    )

    resolved = _enforce_pc_saturation_characterization(ds, _inputs())

    assert resolved.photon_correction_valid is False
    assert resolved.analog_correction_valid is True


def test_characterized_pc_retains_level1_correction_validity() -> None:
    ds = xr.Dataset(
        {"pc_saturation_characterized": (("channel",), np.array([1, 0], dtype=np.int8))},
        coords={"channel": ["532.PC", "532.AN"]},
    )

    resolved = _enforce_pc_saturation_characterization(ds, _inputs())

    assert resolved.photon_correction_valid is True


def test_old_level1_without_characterization_metadata_is_not_assumed_clear() -> None:
    ds = xr.Dataset(coords={"channel": ["532.PC", "532.AN"]})

    resolved = _enforce_pc_saturation_characterization(ds, _inputs())

    assert resolved.photon_correction_valid is False
