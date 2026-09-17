"""Offline schema-3 evidence export; never imported by productive retrieval.

This is a diagnostic join over the persisted catalogue, not a new selection
policy. Missing evidence remains missing. No fitted boundary is used for KFS.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr

from milgrau.level2.high_column_evidence import build_high_column_evidence
from milgrau.level2.rayleigh_catalogue_dataset import validate_rayleigh_candidate_catalogue
from milgrau.level2.rayleigh_uncertainty import origin_calibration_uncertainty
from milgrau.level2.temporal_support import temporal_support_diagnostics


def _array(ds: xr.Dataset, name: str, dims: tuple[str, ...]) -> np.ndarray:
    if ds[name].dims != dims:
        raise ValueError(f"{name} must have dimensions {dims}.")
    return np.asarray(ds[name].values)


def _finite_or_none(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def candidate_evidence_rows(
    ds: xr.Dataset,
    *,
    block_weights: np.ndarray,
    effective_vertical_resolution_m: float,
) -> list[dict[str, object]]:
    """Return every catalogue slot, including rejected/unevaluated candidates.

    Weights and effective resolution are explicit experiment inputs, never
    inferred from missing metadata. Temporal persistence concerns this exact
    candidate window across ALL blocks (not any candidate above its altitude).
    It is unknown if any of those blocks has not evaluated that window.

    The two subwindows are disjoint contiguous positional halves, with the odd
    bin assigned to the upper half. Disagreement is abs(C_low-C_high)/C_full,
    using the existing positive-signal/error calibration diagnostic throughout.
    It does not measure contamination probability or certify molecular purity.
    """
    if str(ds.attrs.get("level2_product_schema_version")) != "3":
        raise ValueError("Offline export requires Level 2 product schema 3.")
    if str(ds.attrs.get("level2_retrieval_method_version")) != "4":
        raise ValueError("Offline export requires retrieval method 4.")
    validate_rayleigh_candidate_catalogue(ds)
    altitude = _array(ds, "altitude", ("altitude",)).astype(float)
    if (altitude.size < 3 or not np.all(np.isfinite(altitude))
            or not np.all(np.diff(altitude) > 0)):
        raise ValueError("altitude must be finite and strictly increasing.")
    resolution = float(effective_vertical_resolution_m)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("effective_vertical_resolution_m must be finite and positive.")
    weights = np.asarray(block_weights, dtype=float)
    if (weights.shape != (ds.sizes["block_time"],) or not np.all(np.isfinite(weights))
            or np.any(weights <= 0)):
        raise ValueError("block_weights must contain one positive finite weight per block.")

    dims = ("block_time", "wavelength", "altitude")
    signal = _array(ds, "glued_range_corrected_signal_block", dims)
    error = _array(ds, "glued_range_corrected_signal_error_block", dims)
    molecular = _array(ds, "simulated_molecular_range_corrected_signal", ("wavelength", "altitude"))
    centers = ds.rayleigh_candidate_center_altitude_m.values
    center_indices = ds.rayleigh_candidate_center_index.values
    starts = ds.rayleigh_candidate_start_altitude_m.values
    stops = ds.rayleigh_candidate_stop_altitude_m.values
    geometry = []
    for center, index, start, stop in zip(centers, center_indices, starts, stops, strict=True):
        if not np.all(np.isfinite([center, index, start, stop])) or index != int(index):
            raise ValueError("Candidate geometry must be finite with integer center indices.")
        index = int(index)
        lo, hi = np.searchsorted(altitude, [start, stop])
        if (index < 0 or index >= altitude.size or lo >= altitude.size or hi >= altitude.size
                or altitude[index] != center or altitude[lo] != start or altitude[hi] != stop
                or not lo < index < hi):
            raise ValueError("Candidate geometry must match exact native altitude bins.")
        geometry.append((index, int(lo), int(hi) + 1))  # stored stop is inclusive

    evaluated = ds.rayleigh_candidate_evaluated_flag.values.astype(bool)
    accepted = ds.rayleigh_candidate_accepted_flag.values.astype(bool)
    selected = ds.rayleigh_candidate_selected_flag.values.astype(bool)
    rows = []
    for wave_index, wavelength in enumerate(ds.wavelength.values):
        temporal = temporal_support_diagnostics(signal[:, wave_index], error[:, wave_index], weights)
        persistence = np.sum(accepted[:, wave_index] * weights[:, None], axis=0) / weights.sum()
        persistence[~np.all(evaluated[:, wave_index], axis=0)] = np.nan
        for block in range(ds.sizes["block_time"]):
            for candidate, (center, lo, hi) in enumerate(geometry):
                key = (block, wave_index, candidate)
                full_factor = low_factor = high_factor = disagreement = np.nan
                independent = correlated = np.nan
                valid_bins = low_bins = high_bins = 0
                if evaluated[key]:
                    y = signal[block, wave_index, lo:hi]
                    x = molecular[wave_index, lo:hi]
                    sigma = error[block, wave_index, lo:hi]
                    fit = origin_calibration_uncertainty(y, x, sigma)
                    midpoint = y.size // 2
                    low = origin_calibration_uncertainty(y[:midpoint], x[:midpoint], sigma[:midpoint])
                    high = origin_calibration_uncertainty(y[midpoint:], x[midpoint:], sigma[midpoint:])
                    full_factor, low_factor, high_factor = (
                        fit.calibration_factor, low.calibration_factor, high.calibration_factor
                    )
                    valid_bins, low_bins, high_bins = fit.valid_bins, low.valid_bins, high.valid_bins
                    independent, correlated = fit.snr_independent, fit.snr_fully_correlated
                    if np.isfinite(full_factor) and full_factor > 0:
                        disagreement = abs(low_factor - high_factor) / full_factor
                record = build_high_column_evidence(
                    wavelength_nm=int(wavelength), block_index=block,
                    candidate_altitude_m=float(centers[candidate]),
                    candidate_shape_qa_accepted=bool(accepted[key]),
                    candidate_binwise_snr=(
                        float(ds.rayleigh_candidate_uncertainty_snr_median.values[key])
                        if evaluated[key] else np.nan
                    ),
                    window_calibration_snr_independent=independent,
                    window_calibration_snr_fully_correlated=correlated,
                    temporal_candidate_persistence_fraction=float(persistence[candidate]),
                    dominant_signal_contribution_fraction=float(temporal.dominant_contribution_fraction[center]),
                    subwindow_relative_disagreement=disagreement,
                    effective_vertical_resolution_m=resolution,
                    boundary_estimator="exact_measured_rcs_bin_productive_method_v4",
                    noise_dependence_model="independent_and_fully_correlated_limits_only",
                ).to_dict()
                record.update(
                    candidate_index=candidate, candidate_center_index=center,
                    candidate_start_altitude_m=float(starts[candidate]),
                    candidate_stop_altitude_m=float(stops[candidate]),
                    candidate_evaluated=bool(evaluated[key]),
                    candidate_productively_selected=bool(selected[key]),
                    candidate_rejection_mask=int(ds.rayleigh_candidate_rejection_mask.values[key]),
                    block_time=str(ds.block_time.values[block]),
                    window_calibration_factor=full_factor,
                    window_calibration_valid_bins=valid_bins,
                    subwindow_lower_calibration_factor=low_factor,
                    subwindow_upper_calibration_factor=high_factor,
                    subwindow_lower_valid_bins=low_bins, subwindow_upper_valid_bins=high_bins,
                    temporal_evaluated_weight_fraction=float(
                        np.sum(weights * evaluated[:, wave_index, candidate]) / weights.sum()
                    ),
                    temporal_center_support_weight_fraction=float(temporal.supporting_weight_fraction[center]),
                    contamination_status="not_evaluated_no_validated_detector_supplied",
                    dependence_model_status="not_evaluated_no_covariance_supplied",
                )
                rows.append({name: _finite_or_none(value) for name, value in record.items()})
    return rows


def evidence_summary(rows: list[dict[str, object]], altitude_edges_m: np.ndarray) -> dict:
    """Descriptive strata and missing counts, never candidate scores or gates."""
    edges = np.asarray(altitude_edges_m, dtype=float)
    if (edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges))
            or not np.all(np.diff(edges) > 0)):
        raise ValueError("altitude_edges_m must be finite and strictly increasing.")
    metrics = (
        "candidate_binwise_snr", "window_calibration_snr_independent",
        "window_calibration_snr_fully_correlated", "window_calibration_snr_dependence_model",
        "temporal_candidate_persistence_fraction", "dominant_signal_contribution_fraction",
        "subwindow_relative_disagreement", "window_contamination_fraction",
    )
    strata = []
    for wavelength in sorted({row["wavelength_nm"] for row in rows}):
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            for state in ("accepted", "rejected", "unevaluated"):
                group = [row for row in rows if row["wavelength_nm"] == wavelength
                         and lower <= row["candidate_altitude_m"] < upper
                         and ("unevaluated" if not row["candidate_evaluated"] else
                              "accepted" if row["candidate_shape_qa_accepted"] else "rejected") == state]
                statistics = {}
                for metric in metrics:
                    values = [row[metric] for row in group if row[metric] is not None]
                    statistics[metric] = {
                        "finite_count": len(values), "missing_count": len(group) - len(values),
                        "median": float(np.median(values)) if values else None,
                        "p25": float(np.percentile(values, 25)) if values else None,
                        "p75": float(np.percentile(values, 75)) if values else None,
                    }
                strata.append(dict(wavelength_nm=wavelength, lower_altitude_m=float(lower),
                                   upper_altitude_m_exclusive=float(upper), state=state,
                                   count=len(group), diagnostics=statistics))
    return {
        "scope": "offline_R&D_diagnostic_only", "candidate_slot_count": len(rows),
        "evaluated_count": sum(row["candidate_evaluated"] for row in rows),
        "accepted_count": sum(row["candidate_shape_qa_accepted"] for row in rows),
        "selected_count": sum(row["candidate_productively_selected"] for row in rows),
        "outside_altitude_bands_count": sum(
            not edges[0] <= row["candidate_altitude_m"] < edges[-1] for row in rows
        ),
        "altitude_edges_m": edges.tolist(), "strata": strata,
        "caveat": "Overlapping windows and repeated temporal diagnostics are not independent samples.",
    }


def write_evidence_export(
    output_dir: str | Path, rows: list[dict[str, object]], summary: dict, provenance: dict,
) -> None:
    """Write strict JSON and CSV to a new directory; never overwrite evidence."""
    if not rows:
        raise ValueError("Cannot export an empty candidate catalogue.")
    # Validate JSON before creating a partially written export.
    report = json.dumps({"provenance": provenance, "summary": summary}, indent=2, allow_nan=False)
    records = json.dumps(rows, indent=2, allow_nan=False)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    (target / "summary.json").write_text(report + "\n", encoding="utf-8")
    (target / "candidates.json").write_text(records + "\n", encoding="utf-8")
    with (target / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
