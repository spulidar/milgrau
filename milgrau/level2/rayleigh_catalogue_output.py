"""Assemble the productive Rayleigh candidate catalogue for NetCDF output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from milgrau.level2.config import get_molecular_fit_config
from milgrau.level2.rayleigh_candidates import (
    RayleighReferenceCandidate,
    catalogue_rayleigh_candidates,
    minimum_cost_rayleigh_candidate,
)
from milgrau.level2.rayleigh_window import rayleigh_window_bins


@dataclass(frozen=True, slots=True)
class RayleighCandidateOutput:
    """Rectangular block/wavelength catalogue plus common window geometry."""

    center_altitude_m: np.ndarray
    start_altitude_m: np.ndarray
    stop_altitude_m: np.ndarray
    center_index: np.ndarray
    evaluated_flag: np.ndarray
    valid_bins: np.ndarray
    valid_fraction: np.ndarray
    relative_slope: np.ndarray
    relative_variance: np.ndarray
    calibration_factor: np.ndarray
    free_intercept: np.ndarray
    uncertainty_snr_median: np.ndarray
    uncertainty_snr_valid_bins: np.ndarray
    diagnostic_cost: np.ndarray
    rejection_mask: np.ndarray
    accepted_flag: np.ndarray
    unfiltered_min_cost_flag: np.ndarray
    selected_flag: np.ndarray


def _candidate_geometry(catalogue: Sequence[RayleighReferenceCandidate]) -> tuple[np.ndarray, ...]:
    return (
        np.asarray([candidate.center_altitude_m for candidate in catalogue], dtype=np.float64),
        np.asarray([candidate.start_altitude_m for candidate in catalogue], dtype=np.float64),
        np.asarray([candidate.stop_altitude_m for candidate in catalogue], dtype=np.float64),
        np.asarray([candidate.center_index for candidate in catalogue], dtype=np.int32),
    )


def _assert_same_geometry(
    catalogue: Sequence[RayleighReferenceCandidate],
    geometry: tuple[np.ndarray, ...],
) -> None:
    observed = _candidate_geometry(catalogue)
    for actual, expected in zip(observed, geometry, strict=True):
        if not np.array_equal(actual, expected):
            raise ValueError(
                "Rayleigh candidate geometry changed between processed blocks/wavelengths; "
                "the current rectangular NetCDF catalogue requires one common altitude/configuration grid."
            )


def assemble_rayleigh_candidate_output(
    results: Sequence[Any],
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
) -> RayleighCandidateOutput:
    """Reconstruct and expose every evaluated productive Rayleigh candidate.

    The catalogue is deterministically rebuilt from the public block selected
    RCS/error arrays, molecular RCS, altitude grid and current fit configuration.
    This avoids hiding selection decisions in logs while keeping candidate
    evaluation as a pure function rather than mutable retrieval state.

    Blocks whose retrieval input was invalid are represented with
    ``evaluated_flag=0``.  ``selected_flag`` identifies only the final accepted
    productive candidate.  ``unfiltered_min_cost_flag`` identifies the minimum
    historical diagnostic cost irrespective of QA, so method-v4 decisions are
    auditable when the unfiltered minimum failed QA.
    """
    if not results:
        raise ValueError("At least one processed wavelength is required.")
    altitude = np.asarray(altitude_m, dtype=np.float64)
    if altitude.ndim != 1 or altitude.size < 3:
        raise ValueError("altitude_m must be a one-dimensional grid with at least three bins.")

    fit = get_molecular_fit_config(config)
    window_bins = rayleigh_window_bins(altitude, float(fit["ref_window_m"]))
    n_block = int(np.asarray(results[0].block_time).size)
    n_wavelength = len(results)

    catalogues: dict[tuple[int, int], tuple[RayleighReferenceCandidate, ...]] = {}
    geometry: tuple[np.ndarray, ...] | None = None
    for wavelength_index, result in enumerate(results):
        if int(np.asarray(result.block_time).size) != n_block:
            raise ValueError("All wavelength results must share the same block axis.")
        input_valid = np.asarray(
            result.signal_selection.retrieval_input_valid_flag_block, dtype=np.int8
        )
        if input_valid.shape != (n_block,):
            raise ValueError("retrieval_input_valid_flag_block does not match block_time.")
        for block_index in range(n_block):
            if int(input_valid[block_index]) != 1:
                continue
            catalogue = catalogue_rayleigh_candidates(
                measured_signal=np.asarray(
                    result.glued.range_corrected_signal_block[block_index], dtype=np.float64
                ),
                simulated_molecular_signal=np.asarray(
                    result.molecular.simulated_range_corrected_signal, dtype=np.float64
                ),
                altitude_m=altitude,
                min_altitude_m=float(fit["ref_alt_min_m"]),
                max_altitude_m=float(fit["ref_alt_max_m"]),
                window_bins=window_bins,
                max_relative_slope=float(fit["max_relative_slope"]),
                max_relative_variance=float(fit["max_relative_variance"]),
                min_valid_fraction=float(fit["min_valid_fraction"]),
                measured_signal_error=np.asarray(
                    result.glued.range_corrected_signal_error_block[block_index], dtype=np.float64
                ),
            )
            if geometry is None:
                geometry = _candidate_geometry(catalogue)
            else:
                _assert_same_geometry(catalogue, geometry)
            catalogues[(block_index, wavelength_index)] = catalogue

    if geometry is None:
        raise ValueError("No retrieval-input-valid block exists to assemble the Rayleigh catalogue.")

    center_altitude, start_altitude, stop_altitude, center_index = geometry
    n_candidate = center_index.size
    shape = (n_block, n_wavelength, n_candidate)
    evaluated = np.zeros(shape, dtype=np.int8)
    valid_bins = np.zeros(shape, dtype=np.int32)
    valid_fraction = np.full(shape, np.nan, dtype=np.float64)
    relative_slope = np.full(shape, np.nan, dtype=np.float64)
    relative_variance = np.full(shape, np.nan, dtype=np.float64)
    calibration_factor = np.full(shape, np.nan, dtype=np.float64)
    free_intercept = np.full(shape, np.nan, dtype=np.float64)
    snr_median = np.full(shape, np.nan, dtype=np.float64)
    snr_valid_bins = np.zeros(shape, dtype=np.int32)
    diagnostic_cost = np.full(shape, np.nan, dtype=np.float64)
    rejection_mask = np.zeros(shape, dtype=np.int16)
    accepted = np.zeros(shape, dtype=np.int8)
    unfiltered_min = np.zeros(shape, dtype=np.int8)
    selected = np.zeros(shape, dtype=np.int8)

    for (block_index, wavelength_index), catalogue in catalogues.items():
        evaluated[block_index, wavelength_index, :] = 1
        valid_bins[block_index, wavelength_index, :] = [
            candidate.valid_bins for candidate in catalogue
        ]
        valid_fraction[block_index, wavelength_index, :] = [
            candidate.valid_fraction for candidate in catalogue
        ]
        relative_slope[block_index, wavelength_index, :] = [
            candidate.relative_slope for candidate in catalogue
        ]
        relative_variance[block_index, wavelength_index, :] = [
            candidate.relative_variance for candidate in catalogue
        ]
        calibration_factor[block_index, wavelength_index, :] = [
            candidate.calibration_factor for candidate in catalogue
        ]
        free_intercept[block_index, wavelength_index, :] = [
            candidate.free_intercept for candidate in catalogue
        ]
        snr_median[block_index, wavelength_index, :] = [
            candidate.uncertainty_snr_median for candidate in catalogue
        ]
        snr_valid_bins[block_index, wavelength_index, :] = [
            candidate.uncertainty_snr_valid_bins for candidate in catalogue
        ]
        diagnostic_cost[block_index, wavelength_index, :] = [
            candidate.diagnostic_cost for candidate in catalogue
        ]
        rejection_mask[block_index, wavelength_index, :] = [
            candidate.rejection_mask for candidate in catalogue
        ]
        accepted[block_index, wavelength_index, :] = [
            int(candidate.accepted) for candidate in catalogue
        ]

        raw_best = minimum_cost_rayleigh_candidate(catalogue)
        raw_match = np.flatnonzero(center_index == int(raw_best.center_index))
        if raw_match.size != 1:
            raise ValueError("Unfiltered minimum-cost candidate is absent from common candidate geometry.")
        unfiltered_min[block_index, wavelength_index, int(raw_match[0])] = 1

        result = results[wavelength_index]
        if int(result.rayleigh.reference_success_flag_block[block_index]) == 1:
            selected_altitude = float(result.rayleigh.reference_altitude_m_block[block_index])
            match = np.flatnonzero(
                np.isclose(center_altitude, selected_altitude, rtol=0.0, atol=1.0e-9)
            )
            if match.size != 1:
                raise ValueError(
                    "Selected productive Rayleigh reference does not identify exactly one persisted candidate."
                )
            selected[block_index, wavelength_index, int(match[0])] = 1
            if accepted[block_index, wavelength_index, int(match[0])] != 1:
                raise ValueError("Productive Rayleigh selection points to a candidate that failed QA.")

    return RayleighCandidateOutput(
        center_altitude_m=center_altitude,
        start_altitude_m=start_altitude,
        stop_altitude_m=stop_altitude,
        center_index=center_index,
        evaluated_flag=evaluated,
        valid_bins=valid_bins,
        valid_fraction=valid_fraction,
        relative_slope=relative_slope,
        relative_variance=relative_variance,
        calibration_factor=calibration_factor,
        free_intercept=free_intercept,
        uncertainty_snr_median=snr_median,
        uncertainty_snr_valid_bins=snr_valid_bins,
        diagnostic_cost=diagnostic_cost,
        rejection_mask=rejection_mask,
        accepted_flag=accepted,
        unfiltered_min_cost_flag=unfiltered_min,
        selected_flag=selected,
    )


def attach_rayleigh_candidate_catalogue(
    ds: xr.Dataset,
    results: Sequence[Any],
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
) -> None:
    """Attach the full auditable candidate catalogue to a Level 2 dataset in place."""
    catalogue = assemble_rayleigh_candidate_output(results, altitude_m, config)
    candidate_dim = "rayleigh_candidate"
    block_dims = ("block_time", "wavelength", candidate_dim)
    ds.coords[candidate_dim] = np.arange(catalogue.center_index.size, dtype=np.int32)
    ds["rayleigh_candidate_center_index"] = ((candidate_dim,), catalogue.center_index)
    ds["rayleigh_candidate_center_altitude_m"] = ((candidate_dim,), catalogue.center_altitude_m)
    ds["rayleigh_candidate_start_altitude_m"] = ((candidate_dim,), catalogue.start_altitude_m)
    ds["rayleigh_candidate_stop_altitude_m"] = ((candidate_dim,), catalogue.stop_altitude_m)
    ds["rayleigh_candidate_evaluated_flag"] = (block_dims, catalogue.evaluated_flag)
    ds["rayleigh_candidate_valid_bins"] = (block_dims, catalogue.valid_bins)
    ds["rayleigh_candidate_valid_fraction"] = (block_dims, catalogue.valid_fraction)
    ds["rayleigh_candidate_relative_slope"] = (block_dims, catalogue.relative_slope)
    ds["rayleigh_candidate_relative_variance"] = (block_dims, catalogue.relative_variance)
    ds["rayleigh_candidate_calibration_factor"] = (block_dims, catalogue.calibration_factor)
    ds["rayleigh_candidate_free_intercept"] = (block_dims, catalogue.free_intercept)
    ds["rayleigh_candidate_uncertainty_snr_median"] = (
        block_dims, catalogue.uncertainty_snr_median
    )
    ds["rayleigh_candidate_uncertainty_snr_valid_bins"] = (
        block_dims, catalogue.uncertainty_snr_valid_bins
    )
    ds["rayleigh_candidate_diagnostic_cost"] = (block_dims, catalogue.diagnostic_cost)
    ds["rayleigh_candidate_rejection_mask"] = (block_dims, catalogue.rejection_mask)
    ds["rayleigh_candidate_accepted_flag"] = (block_dims, catalogue.accepted_flag)
    ds["rayleigh_candidate_unfiltered_min_cost_flag"] = (
        block_dims, catalogue.unfiltered_min_cost_flag
    )
    ds["rayleigh_candidate_selected_flag"] = (block_dims, catalogue.selected_flag)
