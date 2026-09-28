"""Pytest collection policy for reproducible scientific validation sweeps."""

from __future__ import annotations

import os


# These reproducible synthetic matrices are executable scientific validation,
# but their parameter sweeps are deliberately excluded from routine CI. Run
# them explicitly after a change in retrieval physics or uncertainty semantics.
_EXPENSIVE_VALIDATION_SWEEPS = [
    "test_high_column_selector_synthetic_validation.py",
    "test_high_column_selector_cross_wavelength_validation.py",
    "test_high_column_mc_coverage_validation.py",
    "test_high_column_mc_interval_validation.py",
    "test_high_column_mc_random_coverage_validation.py",
    "test_selection_aware_mc_coverage_validation.py",
    "test_selection_aware_mc_measurement_validation.py",
]

collect_ignore = (
    []
    if os.environ.get("MILGRAU_RUN_VALIDATION_SWEEPS", "").strip() == "1"
    else _EXPENSIVE_VALIDATION_SWEEPS
)
