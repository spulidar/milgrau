"""Pytest collection policy for reproducible but expensive R&D evidence sweeps."""

from __future__ import annotations

import os


# These files generate frozen method-v5 evidence matrices and remain executable,
# but they are deliberately excluded from routine CI after their results are
# captured under docs/regression_baselines/. Run them explicitly with
# MILGRAU_RUN_RND_SWEEPS=1 when the scientific method or synthetic design changes.
_EXPENSIVE_RND_SWEEPS = [
    "test_high_column_selector_synthetic_rnd.py",
    "test_high_column_selector_cross_wavelength_rnd.py",
    "test_high_column_mc_coverage_rnd.py",
    "test_high_column_mc_interval_diagnostic_rnd.py",
    "test_high_column_mc_random_coverage_rnd.py",
    "test_selection_aware_mc_coverage_rnd.py",
    "test_selection_aware_mc_measurement_only_rnd.py",
]

collect_ignore = (
    []
    if os.environ.get("MILGRAU_RUN_RND_SWEEPS", "").strip() == "1"
    else _EXPENSIVE_RND_SWEEPS
)
