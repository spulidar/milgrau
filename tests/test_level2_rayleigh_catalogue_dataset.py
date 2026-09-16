"""Tests for the persisted Level 2 Rayleigh candidate catalogue."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import xarray as xr

from milgrau.level2.rayleigh_candidates import (
    catalogue_rayleigh_candidates,
    select_minimum_cost_accepted_candidate,
)
from milgrau.level2.rayleigh_catalogue_dataset import (
    RAYLEIGH_CANDIDATE_STATE_DIMS,
    attach_rayleigh_candidate_catalogue,
    validate_rayleigh_candidate_catalogue,
)
from milgrau.level2.rayleigh_window import rayleigh_window_bins


def _config() -> dict:
    return {
        "inversion": {
            "molecular_fit": {
                "ref_alt_min_m": 300.0,
                "ref_alt_max_m": 1500.0,
                "ref_window_m": 300.0,
                "max_relative_slope": 0.20,
                "max_relative_variance": 0.20,
                "min_valid_fraction": 0.67,
            }
        }
    }


def _result(altitude: np.ndarray, measured: np.ndarray, error: np.ndarray) -> SimpleNamespace:
    simulated = np.exp(-altitude / 7000.0) + 0.25
    fit = _config()["inversion"]["molecular_fit"]
    catalogue = catalogue_rayleigh_candidates(
        measured,
        simulated,
        altitude,
        min_altitude_m=fit["ref_alt_min_m"],
        max_altitude_m=fit["ref_alt_max_m"],
        window_bins=rayleigh_window_bins(altitude, fit["ref_window_m"]),
        max_relative_slope=fit["max_relative_slope"],
        max_relative_variance=fit["max_relative_variance"],
        min_valid_fraction=fit["min_valid_fraction"],
        measured_signal_error=error,
    )
    selected = select_minimum_cost_accepted_candidate(catalogue)
    return SimpleNamespace(
        block_time=np.asarray([np.datetime64("2025-11-07T16:40")]),
        signal_selection=SimpleNamespace(
            retrieval_input_valid_flag_block=np.asarray([1], dtype=np.int8)
        ),
        glued=SimpleNamespace(
            range_corrected_signal_block=measured[np.newaxis, :],
            range_corrected_signal_error_block=error[np.newaxis, :],
        ),
        molecular=SimpleNamespace(simulated_range_corrected_signal=simulated),
        rayleigh=SimpleNamespace(
            reference_success_flag_block=np.asarray([1], dtype=np.int8),
            reference_altitude_m_block=np.asarray([selected.center_altitude_m], dtype=np.float64),
        ),
    )


def test_attach_catalogue_persists_qa_and_selection_states() -> None:
    altitude = np.arange(0.0, 2000.0, 100.0)
    simulated = np.exp(-altitude / 7000.0) + 0.25
    measured = 8.0 * simulated
    error = np.full_like(measured, 0.2)
    result = _result(altitude, measured, error)
    ds = xr.Dataset(
        data_vars={
            "rayleigh_reference_success_flag_block": (
                ("block_time", "wavelength"),
                np.asarray([[1]], dtype=np.int8),
            ),
            "rayleigh_reference_altitude_m_block": (
                ("block_time", "wavelength"),
                np.asarray([[result.rayleigh.reference_altitude_m_block[0]]]),
            ),
        },
        coords={
            "block_time": result.block_time,
            "wavelength": np.asarray([532], dtype=np.int32),
        },
    )

    output = attach_rayleigh_candidate_catalogue(ds, [result], altitude, _config())

    validate_rayleigh_candidate_catalogue(output)
    assert output["rayleigh_candidate_selected_flag"].dims == RAYLEIGH_CANDIDATE_STATE_DIMS
    assert int(output["rayleigh_candidate_selected_flag"].sum()) == 1
    assert int(output["rayleigh_candidate_unfiltered_min_cost_flag"].sum()) == 1
    assert np.all(output["rayleigh_candidate_accepted_flag"].values == 1)
    selected = output["rayleigh_candidate_selected_flag"].values[0, 0].astype(bool)
    selected_altitude = output["rayleigh_candidate_center_altitude_m"].values[selected]
    np.testing.assert_allclose(selected_altitude, result.rayleigh.reference_altitude_m_block)
    assert output.attrs["Rayleigh_Candidate_Catalogue_Policy"].startswith("persist every complete")


def test_catalogue_keeps_rejected_candidates_and_reason_mask() -> None:
    altitude = np.arange(0.0, 2000.0, 100.0)
    simulated = np.exp(-altitude / 7000.0) + 0.25
    measured = 8.0 * simulated
    measured[(altitude >= 1100.0) & (altitude <= 1500.0)] *= np.linspace(1.0, 3.0, 5)
    error = np.full_like(measured, 0.2)
    result = _result(altitude, measured, error)
    ds = xr.Dataset(
        data_vars={
            "rayleigh_reference_success_flag_block": (
                ("block_time", "wavelength"),
                np.asarray([[1]], dtype=np.int8),
            ),
            "rayleigh_reference_altitude_m_block": (
                ("block_time", "wavelength"),
                np.asarray([[result.rayleigh.reference_altitude_m_block[0]]]),
            ),
        },
        coords={
            "block_time": result.block_time,
            "wavelength": np.asarray([355], dtype=np.int32),
        },
    )

    output = attach_rayleigh_candidate_catalogue(ds, [result], altitude, _config())

    rejected = output["rayleigh_candidate_accepted_flag"].values[0, 0] == 0
    assert rejected.any()
    assert np.any(output["rayleigh_candidate_rejection_mask"].values[0, 0, rejected] != 0)
    assert int(output["rayleigh_candidate_selected_flag"].sum()) == 1
