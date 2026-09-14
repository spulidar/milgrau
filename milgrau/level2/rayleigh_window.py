"""Rayleigh reference-window geometry on the Level 1 altitude grid."""

from __future__ import annotations

import numpy as np


def rayleigh_window_bins(altitude_m: np.ndarray, width_m: float) -> int:
    """Convert a physical Rayleigh-window width to bins on a uniform grid."""
    altitude = np.asarray(altitude_m, dtype=np.float64)
    width = float(width_m)
    if altitude.ndim != 1 or altitude.size < 2:
        raise ValueError("Rayleigh altitude grid must be one-dimensional with at least two bins.")
    if not np.all(np.isfinite(altitude)) or not np.all(np.diff(altitude) > 0.0):
        raise ValueError("Rayleigh altitude grid must be finite and strictly increasing.")
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("Rayleigh reference-window width must be positive and finite.")

    spacing = np.diff(altitude)
    step_m = float(np.median(spacing))
    if not np.allclose(spacing, step_m, rtol=1e-6, atol=1e-9):
        raise ValueError("Rayleigh reference-window conversion requires a uniform altitude grid.")
    bins = int(round(width / step_m))
    if bins < 3:
        raise ValueError(
            f"Rayleigh reference-window width {width:g} m resolves to fewer than three bins on a {step_m:g} m grid."
        )
    return bins
