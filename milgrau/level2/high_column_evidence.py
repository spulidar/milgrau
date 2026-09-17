"""Diagnostic-only evidence vector for high-column Level 2 R&D.

The purpose of this module is to keep distinct scientific observables distinct.
It deliberately provides no combined score, threshold, ranking, pass/fail state
or target-altitude preference. Productive method v4 is unchanged.

A future high-column decision rule may consume these fields only after each
observable and any threshold applied to it have independent validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class HighColumnEvidence:
    """One candidate's diagnostic evidence without an overall decision."""

    wavelength_nm: int
    block_index: int
    candidate_altitude_m: float
    candidate_shape_qa_accepted: bool
    candidate_binwise_snr: float
    window_calibration_snr_independent: float
    window_calibration_snr_fully_correlated: float
    window_calibration_snr_dependence_model: float
    temporal_candidate_persistence_fraction: float
    dominant_signal_contribution_fraction: float
    subwindow_relative_disagreement: float
    window_contamination_fraction: float
    effective_vertical_resolution_m: float
    boundary_estimator: str
    noise_dependence_model: str

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly mapping without deriving a score."""
        return asdict(self)


def _finite_positive(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _nonnegative_or_nan(value: float, name: str) -> float:
    number = float(value)
    if np.isnan(number):
        return number
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be non-negative or NaN when not evaluated.")
    return number


def _fraction_or_nan(value: float, name: str) -> float:
    number = float(value)
    if np.isnan(number):
        return number
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must lie within [0, 1] or be NaN when not evaluated.")
    return number


def build_high_column_evidence(
    *,
    wavelength_nm: int,
    block_index: int,
    candidate_altitude_m: float,
    candidate_shape_qa_accepted: bool,
    candidate_binwise_snr: float,
    window_calibration_snr_independent: float,
    window_calibration_snr_fully_correlated: float,
    window_calibration_snr_dependence_model: float = np.nan,
    temporal_candidate_persistence_fraction: float = np.nan,
    dominant_signal_contribution_fraction: float = np.nan,
    subwindow_relative_disagreement: float = np.nan,
    window_contamination_fraction: float = np.nan,
    effective_vertical_resolution_m: float,
    boundary_estimator: str,
    noise_dependence_model: str,
) -> HighColumnEvidence:
    """Validate and assemble one high-column diagnostic evidence record.

    NaN is allowed only for diagnostics that genuinely were not evaluated. The
    function does not replace missing evidence with a favorable value and does
    not compute a composite quality indicator.
    """
    wavelength = int(wavelength_nm)
    block = int(block_index)
    if wavelength <= 0:
        raise ValueError("wavelength_nm must be positive.")
    if block < 0:
        raise ValueError("block_index must be non-negative.")

    altitude = _finite_positive(candidate_altitude_m, "candidate_altitude_m")
    resolution = _finite_positive(
        effective_vertical_resolution_m,
        "effective_vertical_resolution_m",
    )
    estimator = str(boundary_estimator).strip()
    dependence = str(noise_dependence_model).strip()
    if not estimator:
        raise ValueError("boundary_estimator must be a non-empty explicit identifier.")
    if not dependence:
        raise ValueError("noise_dependence_model must be a non-empty explicit identifier.")

    return HighColumnEvidence(
        wavelength_nm=wavelength,
        block_index=block,
        candidate_altitude_m=altitude,
        candidate_shape_qa_accepted=bool(candidate_shape_qa_accepted),
        candidate_binwise_snr=_nonnegative_or_nan(
            candidate_binwise_snr,
            "candidate_binwise_snr",
        ),
        window_calibration_snr_independent=_nonnegative_or_nan(
            window_calibration_snr_independent,
            "window_calibration_snr_independent",
        ),
        window_calibration_snr_fully_correlated=_nonnegative_or_nan(
            window_calibration_snr_fully_correlated,
            "window_calibration_snr_fully_correlated",
        ),
        window_calibration_snr_dependence_model=_nonnegative_or_nan(
            window_calibration_snr_dependence_model,
            "window_calibration_snr_dependence_model",
        ),
        temporal_candidate_persistence_fraction=_fraction_or_nan(
            temporal_candidate_persistence_fraction,
            "temporal_candidate_persistence_fraction",
        ),
        dominant_signal_contribution_fraction=_fraction_or_nan(
            dominant_signal_contribution_fraction,
            "dominant_signal_contribution_fraction",
        ),
        subwindow_relative_disagreement=_nonnegative_or_nan(
            subwindow_relative_disagreement,
            "subwindow_relative_disagreement",
        ),
        window_contamination_fraction=_fraction_or_nan(
            window_contamination_fraction,
            "window_contamination_fraction",
        ),
        effective_vertical_resolution_m=resolution,
        boundary_estimator=estimator,
        noise_dependence_model=dependence,
    )
