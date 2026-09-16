"""Persist and validate the auditable Rayleigh candidate catalogue in Level 2."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from milgrau.level2.config import get_molecular_fit_config
from milgrau.level2.rayleigh_catalogue_output import assemble_rayleigh_candidate_output

RAYLEIGH_CANDIDATE_DIM = "rayleigh_candidate"
RAYLEIGH_CANDIDATE_STATE_DIMS = ("block_time", "wavelength", RAYLEIGH_CANDIDATE_DIM)
RAYLEIGH_CANDIDATE_VARIABLES: tuple[str, ...] = (
    "rayleigh_candidate_center_altitude_m",
    "rayleigh_candidate_start_altitude_m",
    "rayleigh_candidate_stop_altitude_m",
    "rayleigh_candidate_center_index",
    "rayleigh_candidate_evaluated_flag",
    "rayleigh_candidate_valid_bins",
    "rayleigh_candidate_valid_fraction",
    "rayleigh_candidate_relative_slope",
    "rayleigh_candidate_relative_variance",
    "rayleigh_candidate_calibration_factor",
    "rayleigh_candidate_free_intercept",
    "rayleigh_candidate_uncertainty_snr_median",
    "rayleigh_candidate_uncertainty_snr_valid_bins",
    "rayleigh_candidate_diagnostic_cost",
    "rayleigh_candidate_rejection_mask",
    "rayleigh_candidate_accepted_flag",
    "rayleigh_candidate_unfiltered_min_cost_flag",
    "rayleigh_candidate_selected_flag",
)


def _set_flag_metadata(variable: xr.DataArray, *, false_meaning: str, true_meaning: str) -> None:
    variable.attrs.update(
        {
            "units": "1",
            "flag_values": np.asarray([0, 1], dtype=np.int8),
            "flag_meanings": f"{false_meaning} {true_meaning}",
        }
    )


def _apply_candidate_metadata(ds: xr.Dataset) -> None:
    geometry = {
        "rayleigh_candidate_center_altitude_m": "Rayleigh candidate center altitude",
        "rayleigh_candidate_start_altitude_m": "Rayleigh candidate start altitude",
        "rayleigh_candidate_stop_altitude_m": "Rayleigh candidate stop altitude",
    }
    for name, long_name in geometry.items():
        ds[name].attrs.update({"long_name": long_name, "units": "m"})

    ds["rayleigh_candidate_center_index"].attrs.update(
        {
            "long_name": "Rayleigh candidate center grid index",
            "units": "1",
            "description": "Zero-based altitude-grid index of the candidate center.",
        }
    )
    ds["rayleigh_candidate_valid_bins"].attrs.update(
        {"long_name": "Rayleigh candidate valid sample count", "units": "1"}
    )
    ds["rayleigh_candidate_valid_fraction"].attrs.update(
        {"long_name": "Rayleigh candidate valid sample fraction", "units": "1"}
    )
    ds["rayleigh_candidate_relative_slope"].attrs.update(
        {
            "long_name": "Rayleigh candidate relative slope diagnostic",
            "units": "1",
        }
    )
    ds["rayleigh_candidate_relative_variance"].attrs.update(
        {
            "long_name": "Rayleigh candidate relative variance diagnostic",
            "units": "1",
        }
    )
    ds["rayleigh_candidate_calibration_factor"].attrs.update(
        {
            "long_name": "Rayleigh candidate origin-constrained calibration factor",
            "unit_status": "instrument_native_scaling_factor",
            "description": (
                "Multiplicative scaling between simulated molecular RCS and measured selected-signal RCS. "
                "Its numerical unit follows the instrument-native RCS representation."
            ),
        }
    )
    ds["rayleigh_candidate_free_intercept"].attrs.update(
        {
            "long_name": "Rayleigh candidate free-intercept diagnostic",
            "unit_status": "instrument_native_rcs",
            "description": "Free-intercept fit diagnostic in the measured selected-signal RCS representation.",
        }
    )
    ds["rayleigh_candidate_uncertainty_snr_median"].attrs.update(
        {
            "long_name": "Rayleigh candidate median propagated-uncertainty signal-to-noise ratio",
            "units": "1",
            "description": "Diagnostic only; no productive hard SNR threshold is currently enabled.",
        }
    )
    ds["rayleigh_candidate_uncertainty_snr_valid_bins"].attrs.update(
        {
            "long_name": "Rayleigh candidate bins contributing to propagated-uncertainty SNR diagnostic",
            "units": "1",
        }
    )
    ds["rayleigh_candidate_diagnostic_cost"].attrs.update(
        {
            "long_name": "Rayleigh candidate historical diagnostic ranking cost",
            "units": "1",
            "description": "relative_slope + relative_variance; applied only after configured minimum QA in method v4.",
        }
    )
    ds["rayleigh_candidate_rejection_mask"].attrs.update(
        {
            "long_name": "Rayleigh candidate minimum-QA rejection bit mask",
            "units": "1",
            "flag_masks": np.asarray([1, 2, 4, 8], dtype=np.int16),
            "flag_meanings": (
                "insufficient_valid_fraction invalid_calibration "
                "excess_relative_slope excess_relative_variance"
            ),
            "description": "Zero means the candidate passed every currently enabled minimum-QA gate.",
        }
    )
    _set_flag_metadata(
        ds["rayleigh_candidate_evaluated_flag"],
        false_meaning="not_evaluated",
        true_meaning="evaluated",
    )
    ds["rayleigh_candidate_evaluated_flag"].attrs["long_name"] = "Rayleigh candidate evaluated flag"
    _set_flag_metadata(
        ds["rayleigh_candidate_accepted_flag"],
        false_meaning="rejected_or_not_evaluated",
        true_meaning="accepted",
    )
    ds["rayleigh_candidate_accepted_flag"].attrs["long_name"] = "Rayleigh candidate minimum-QA accepted flag"
    _set_flag_metadata(
        ds["rayleigh_candidate_unfiltered_min_cost_flag"],
        false_meaning="not_unfiltered_minimum",
        true_meaning="unfiltered_minimum_cost",
    )
    ds["rayleigh_candidate_unfiltered_min_cost_flag"].attrs.update(
        {
            "long_name": "Rayleigh candidate unfiltered minimum-cost flag",
            "description": (
                "Marks the minimum historical diagnostic cost before minimum-QA filtering; retained only to audit "
                "how method v4 differs from the previous select-then-QA ordering."
            ),
        }
    )
    _set_flag_metadata(
        ds["rayleigh_candidate_selected_flag"],
        false_meaning="not_selected",
        true_meaning="selected_productively",
    )
    ds["rayleigh_candidate_selected_flag"].attrs.update(
        {
            "long_name": "Rayleigh candidate productive selection flag",
            "description": "Exactly one accepted candidate is selected for each successful retrieval block/wavelength.",
        }
    )


def attach_rayleigh_candidate_catalogue(
    ds: xr.Dataset,
    results: Sequence[Any],
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
) -> xr.Dataset:
    """Attach the deterministic block/wavelength candidate catalogue to a Level 2 dataset."""
    catalogue = assemble_rayleigh_candidate_output(results, altitude_m, config)
    n_candidate = int(catalogue.center_index.size)
    ds = ds.assign_coords(
        {
            RAYLEIGH_CANDIDATE_DIM: np.arange(n_candidate, dtype=np.int32),
        }
    )
    ds["rayleigh_candidate_center_altitude_m"] = (
        (RAYLEIGH_CANDIDATE_DIM,),
        catalogue.center_altitude_m,
    )
    ds["rayleigh_candidate_start_altitude_m"] = (
        (RAYLEIGH_CANDIDATE_DIM,),
        catalogue.start_altitude_m,
    )
    ds["rayleigh_candidate_stop_altitude_m"] = (
        (RAYLEIGH_CANDIDATE_DIM,),
        catalogue.stop_altitude_m,
    )
    ds["rayleigh_candidate_center_index"] = (
        (RAYLEIGH_CANDIDATE_DIM,),
        catalogue.center_index.astype(np.int32),
    )
    state = RAYLEIGH_CANDIDATE_STATE_DIMS
    ds["rayleigh_candidate_evaluated_flag"] = (state, catalogue.evaluated_flag.astype(np.int8))
    ds["rayleigh_candidate_valid_bins"] = (state, catalogue.valid_bins.astype(np.int32))
    ds["rayleigh_candidate_valid_fraction"] = (state, catalogue.valid_fraction)
    ds["rayleigh_candidate_relative_slope"] = (state, catalogue.relative_slope)
    ds["rayleigh_candidate_relative_variance"] = (state, catalogue.relative_variance)
    ds["rayleigh_candidate_calibration_factor"] = (state, catalogue.calibration_factor)
    ds["rayleigh_candidate_free_intercept"] = (state, catalogue.free_intercept)
    ds["rayleigh_candidate_uncertainty_snr_median"] = (state, catalogue.uncertainty_snr_median)
    ds["rayleigh_candidate_uncertainty_snr_valid_bins"] = (
        state,
        catalogue.uncertainty_snr_valid_bins.astype(np.int32),
    )
    ds["rayleigh_candidate_diagnostic_cost"] = (state, catalogue.diagnostic_cost)
    ds["rayleigh_candidate_rejection_mask"] = (state, catalogue.rejection_mask.astype(np.int16))
    ds["rayleigh_candidate_accepted_flag"] = (state, catalogue.accepted_flag.astype(np.int8))
    ds["rayleigh_candidate_unfiltered_min_cost_flag"] = (
        state,
        catalogue.unfiltered_min_cost_flag.astype(np.int8),
    )
    ds["rayleigh_candidate_selected_flag"] = (state, catalogue.selected_flag.astype(np.int8))

    fit = get_molecular_fit_config(config)
    ds.attrs.update(
        {
            "Rayleigh_Candidate_Search_Min_Altitude_m": float(fit["ref_alt_min_m"]),
            "Rayleigh_Candidate_Search_Max_Altitude_m": float(fit["ref_alt_max_m"]),
            "Rayleigh_Candidate_Window_Width_m": float(fit["ref_window_m"]),
            "Rayleigh_Candidate_Catalogue_Policy": (
                "persist every complete search-window candidate; minimum-QA is evaluated before productive ranking; "
                "propagated-uncertainty SNR remains diagnostic-only"
            ),
        }
    )
    _apply_candidate_metadata(ds)
    validate_rayleigh_candidate_catalogue(ds)
    return ds


def validate_rayleigh_candidate_catalogue(ds: xr.Dataset) -> None:
    """Validate dimensions and selection invariants of the persisted catalogue."""
    missing = [name for name in RAYLEIGH_CANDIDATE_VARIABLES if name not in ds]
    if missing:
        raise KeyError(f"Level 2 file lacks Rayleigh candidate catalogue variable(s): {missing}")
    if RAYLEIGH_CANDIDATE_DIM not in ds.dims:
        raise KeyError(f"Level 2 file lacks {RAYLEIGH_CANDIDATE_DIM!r} dimension.")
    if ds.sizes[RAYLEIGH_CANDIDATE_DIM] <= 0:
        raise ValueError("Rayleigh candidate catalogue must contain at least one candidate.")

    geometry_dims = (RAYLEIGH_CANDIDATE_DIM,)
    for name in (
        "rayleigh_candidate_center_altitude_m",
        "rayleigh_candidate_start_altitude_m",
        "rayleigh_candidate_stop_altitude_m",
        "rayleigh_candidate_center_index",
    ):
        if ds[name].dims != geometry_dims:
            raise ValueError(f"{name} must have dimensions {geometry_dims}; got {ds[name].dims}.")
    for name in RAYLEIGH_CANDIDATE_VARIABLES[4:]:
        if ds[name].dims != RAYLEIGH_CANDIDATE_STATE_DIMS:
            raise ValueError(
                f"{name} must have dimensions {RAYLEIGH_CANDIDATE_STATE_DIMS}; got {ds[name].dims}."
            )

    center = np.asarray(ds["rayleigh_candidate_center_altitude_m"].values, dtype=np.float64)
    start = np.asarray(ds["rayleigh_candidate_start_altitude_m"].values, dtype=np.float64)
    stop = np.asarray(ds["rayleigh_candidate_stop_altitude_m"].values, dtype=np.float64)
    center_index = np.asarray(ds["rayleigh_candidate_center_index"].values, dtype=np.int64)
    if not np.all(np.isfinite(center)) or not np.all(np.diff(center) > 0.0):
        raise ValueError("Rayleigh candidate centers must be finite and strictly increasing.")
    if np.any(start >= center) or np.any(center >= stop):
        raise ValueError("Every Rayleigh candidate must satisfy start < center < stop.")
    if np.any(np.diff(center_index) <= 0):
        raise ValueError("Rayleigh candidate center indices must be strictly increasing.")

    evaluated = np.asarray(ds["rayleigh_candidate_evaluated_flag"].values, dtype=np.int8)
    accepted = np.asarray(ds["rayleigh_candidate_accepted_flag"].values, dtype=np.int8)
    selected = np.asarray(ds["rayleigh_candidate_selected_flag"].values, dtype=np.int8)
    raw_min = np.asarray(ds["rayleigh_candidate_unfiltered_min_cost_flag"].values, dtype=np.int8)
    rejection = np.asarray(ds["rayleigh_candidate_rejection_mask"].values, dtype=np.int16)
    for name, values in (
        ("evaluated", evaluated),
        ("accepted", accepted),
        ("selected", selected),
        ("unfiltered minimum", raw_min),
    ):
        if not np.isin(values, (0, 1)).all():
            raise ValueError(f"Rayleigh candidate {name} flags must contain only 0 or 1.")
    expected_accepted = ((evaluated == 1) & (rejection == 0)).astype(np.int8)
    if not np.array_equal(accepted, expected_accepted):
        raise ValueError("Rayleigh candidate accepted flags must equal evaluated candidates with rejection_mask == 0.")
    if np.any(selected > accepted):
        raise ValueError("A productively selected Rayleigh candidate must pass minimum QA.")

    evaluated_block = np.any(evaluated == 1, axis=2)
    if not np.array_equal(raw_min.sum(axis=2), evaluated_block.astype(np.int64)):
        raise ValueError("Each evaluated block/wavelength must expose exactly one unfiltered minimum-cost candidate.")

    success = np.asarray(ds["rayleigh_reference_success_flag_block"].values, dtype=np.int8)
    if not np.array_equal(selected.sum(axis=2), (success == 1).astype(np.int64)):
        raise ValueError("Each successful Rayleigh block/wavelength must expose exactly one productive selected candidate.")
    selected_indices = np.argmax(selected, axis=2)
    selected_center = center[selected_indices]
    reference_altitude = np.asarray(ds["rayleigh_reference_altitude_m_block"].values, dtype=np.float64)
    success_mask = success == 1
    if np.any(~np.isclose(selected_center[success_mask], reference_altitude[success_mask], rtol=0.0, atol=1.0e-9)):
        raise ValueError("Persisted selected candidate altitude does not match the productive Rayleigh reference altitude.")
