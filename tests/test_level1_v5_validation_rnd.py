"""Lightweight contracts for real-Level-1 method-v5 validation evidence."""

from __future__ import annotations

import json

import numpy as np

from milgrau.level2.level1_v5_validation_rnd import (
    Level1V5ValidationSummary,
    V5BlockValidationSummary,
    _lower_common_samples,
    _relative_l2,
    validation_summary_dict,
    write_validation_summary_json,
)


def test_lower_common_samples_preserve_native_low_column_without_interpolation() -> None:
    native_altitude = np.arange(0.0, 6015.0, 7.5)
    native_values = native_altitude * 2.0
    progressive_altitude = native_altitude.copy()
    progressive_values = native_values + 1.0

    reference, comparison = _lower_common_samples(
        native_altitude,
        native_values,
        progressive_altitude,
        progressive_values,
    )

    assert reference.size > 0
    assert reference.shape == comparison.shape
    assert np.allclose(comparison - reference, 1.0)
    assert reference[0] == 1200.0
    assert reference[-1] == 12000.0


def test_lower_common_samples_do_not_bridge_shifted_grid() -> None:
    native_altitude = np.arange(0.0, 6015.0, 7.5)
    native_values = np.ones(native_altitude.size)
    shifted_altitude = native_altitude + 3.75
    shifted_values = np.ones(shifted_altitude.size)

    reference, comparison = _lower_common_samples(
        native_altitude,
        native_values,
        shifted_altitude,
        shifted_values,
    )

    assert reference.size == 0
    assert comparison.size == 0


def test_relative_l2_is_zero_for_identical_profiles() -> None:
    profile = np.array([1.0, 2.0, np.nan, 4.0])
    assert _relative_l2(profile, profile) == 0.0


def test_validation_summary_json_replaces_nonfinite_values(tmp_path) -> None:
    block = V5BlockValidationSummary(
        block_index=0,
        block_time_utc="2025-01-01T00:00:00",
        retrieval_input_valid=True,
        signal_source_flag=3,
        v4_retrieval_success=True,
        v4_reference_altitude_m=6000.0,
        v5_success=False,
        v5_failure="no admissible reference",
        v5_reference_altitude_m=float("nan"),
        v5_reference_effective_resolution_m=float("nan"),
        v5_reference_diagnostic_cost=float("nan"),
        v5_accepted_admissible_candidates=0,
        v5_contiguous_top_altitude_m=15000.0,
        mc_selection_success_fraction=float("nan"),
        mc_reference_altitude_median=float("nan"),
        mc_reference_altitude_std=float("nan"),
        mc_lower_valid_fraction_min=float("nan"),
        v4_v5_lower_relative_l2=float("nan"),
        residual_fraction_lower_relative_l2={"0.02": float("nan")},
    )
    summary = Level1V5ValidationSummary(
        wavelength_nm=532,
        analog_channel="532.AN",
        photon_channel="532.PC",
        n_blocks=1,
        residual_fractions=(0.0, 0.02),
        n_iterations=50,
        beta_ref_relative_std=0.0,
        uncertainty_mode="independent",
        blocks=(block,),
    )

    payload = validation_summary_dict(summary)
    assert payload["blocks"][0]["v5_reference_altitude_m"] is None
    assert payload["blocks"][0]["residual_fraction_lower_relative_l2"]["0.02"] is None

    path = write_validation_summary_json(summary, tmp_path / "evidence.json")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["wavelength_nm"] == 532
    assert parsed["blocks"][0]["v5_reference_altitude_m"] is None
