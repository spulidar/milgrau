"""Tests for the diagnostic-only high-column evidence record."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from milgrau.level2.high_column_evidence import (
    HighColumnEvidence,
    build_high_column_evidence,
)


def _record(**overrides: object) -> HighColumnEvidence:
    values: dict[str, object] = {
        "wavelength_nm": 532,
        "block_index": 2,
        "candidate_altitude_m": 15000.0,
        "candidate_shape_qa_accepted": True,
        "candidate_binwise_snr": 1.2,
        "window_calibration_snr_independent": 18.0,
        "window_calibration_snr_fully_correlated": 1.1,
        "window_calibration_snr_dependence_model": 8.4,
        "temporal_candidate_persistence_fraction": 0.8,
        "dominant_signal_contribution_fraction": 0.42,
        "subwindow_relative_disagreement": 0.35,
        "window_contamination_fraction": 0.0,
        "effective_vertical_resolution_m": 7.5,
        "boundary_estimator": "window_origin_fit_same_altitude",
        "noise_dependence_model": "empirical_lag_autocorrelation_diagnostic",
    }
    values.update(overrides)
    return build_high_column_evidence(**values)  # type: ignore[arg-type]


def test_evidence_vector_preserves_distinct_diagnostics_without_score() -> None:
    evidence = _record()
    names = {field.name for field in fields(HighColumnEvidence)}

    assert evidence.candidate_binwise_snr == 1.2
    assert evidence.window_calibration_snr_dependence_model == 8.4
    assert evidence.temporal_candidate_persistence_fraction == 0.8
    assert evidence.window_contamination_fraction == 0.0
    assert "score" not in names
    assert "overall_score" not in names
    assert "scientifically_valid" not in names
    assert "high_column_accepted" not in names


def test_missing_unevaluated_diagnostic_remains_nan_not_favorable_default() -> None:
    evidence = _record(
        window_calibration_snr_dependence_model=np.nan,
        window_contamination_fraction=np.nan,
        temporal_candidate_persistence_fraction=np.nan,
    )

    assert np.isnan(evidence.window_calibration_snr_dependence_model)
    assert np.isnan(evidence.window_contamination_fraction)
    assert np.isnan(evidence.temporal_candidate_persistence_fraction)


def test_to_dict_is_serialization_only_and_preserves_method_labels() -> None:
    evidence = _record()
    serialized = evidence.to_dict()

    assert serialized["boundary_estimator"] == "window_origin_fit_same_altitude"
    assert serialized["noise_dependence_model"] == (
        "empirical_lag_autocorrelation_diagnostic"
    )
    assert "score" not in serialized


@pytest.mark.parametrize(
    "field_name, invalid_value",
    [
        ("temporal_candidate_persistence_fraction", 1.1),
        ("dominant_signal_contribution_fraction", -0.1),
        ("window_contamination_fraction", 2.0),
        ("candidate_binwise_snr", -1.0),
        ("subwindow_relative_disagreement", -0.01),
        ("effective_vertical_resolution_m", 0.0),
    ],
)
def test_evidence_contract_rejects_physically_invalid_values(
    field_name: str,
    invalid_value: float,
) -> None:
    with pytest.raises(ValueError):
        _record(**{field_name: invalid_value})


def test_shape_qa_flag_does_not_imply_other_evidence_is_complete() -> None:
    evidence = _record(
        candidate_shape_qa_accepted=True,
        window_contamination_fraction=np.nan,
        temporal_candidate_persistence_fraction=np.nan,
    )

    assert evidence.candidate_shape_qa_accepted is True
    assert np.isnan(evidence.window_contamination_fraction)
    assert np.isnan(evidence.temporal_candidate_persistence_fraction)
