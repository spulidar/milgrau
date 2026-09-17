"""P5.4 R&D leave-part-out diagnostics for Rayleigh fitting windows.

The tests ask whether calibration-factor sensitivity to removing contiguous
parts of a window can expose localized contamination. They also preserve the
counterexample that broad symmetric contamination can bias every retained
subset similarly. No productive veto or numerical threshold is defined here.
"""

from __future__ import annotations

import numpy as np

from milgrau.level2.rayleigh_uncertainty import origin_calibration_uncertainty


def _case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    altitude_m = np.arange(8000.0, 10000.0 + 7.5, 7.5, dtype=np.float64)
    molecular_signal = np.exp(-(altitude_m - 8000.0) / 7000.0)
    measured_error = 0.02 * molecular_signal
    reference = (altitude_m >= 8500.0) & (altitude_m <= 9500.0)
    return altitude_m, molecular_signal, measured_error, reference


def _factor(
    measured_signal: np.ndarray,
    molecular_signal: np.ndarray,
    measured_error: np.ndarray,
    mask: np.ndarray,
) -> float:
    return float(
        origin_calibration_uncertainty(
            measured_signal[mask],
            molecular_signal[mask],
            measured_error[mask],
        ).calibration_factor
    )


def _leave_quarter_out_relative_range(
    measured_signal: np.ndarray,
    molecular_signal: np.ndarray,
    measured_error: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    indices = np.flatnonzero(reference)
    full = _factor(measured_signal, molecular_signal, measured_error, reference)
    factors = []
    for removed in np.array_split(indices, 4):
        retained = reference.copy()
        retained[removed] = False
        factors.append(
            _factor(measured_signal, molecular_signal, measured_error, retained)
        )
    factors_array = np.asarray(factors, dtype=np.float64)
    relative_range = float((np.max(factors_array) - np.min(factors_array)) / full)
    return full, factors_array, relative_range


def test_clean_window_is_stable_under_leave_quarter_out() -> None:
    altitude_m, molecular_signal, measured_error, reference = _case()
    del altitude_m
    full, factors, relative_range = _leave_quarter_out_relative_range(
        molecular_signal,
        molecular_signal,
        measured_error,
        reference,
    )

    assert full == 1.0
    assert np.allclose(factors, 1.0)
    assert relative_range == 0.0


def test_localized_asymmetric_contamination_changes_leave_out_factor() -> None:
    altitude_m, molecular_signal, measured_error, reference = _case()
    ratio = 1.0 + 0.25 * np.exp(-0.5 * ((altitude_m - 8600.0) / 220.0) ** 2)
    measured_signal = molecular_signal * ratio

    full, _factors, relative_range = _leave_quarter_out_relative_range(
        measured_signal,
        molecular_signal,
        measured_error,
        reference,
    )

    assert full > 1.08
    assert relative_range > 0.05


def test_broad_symmetric_bias_can_remain_stable_under_leave_quarter_out() -> None:
    """Leave-out stability is sensitivity evidence, not a molecular-purity certificate."""
    altitude_m, molecular_signal, measured_error, reference = _case()
    ratio = 1.0 + 0.25 * np.exp(-0.5 * ((altitude_m - 9000.0) / 650.0) ** 2)
    measured_signal = molecular_signal * ratio

    full, _factors, relative_range = _leave_quarter_out_relative_range(
        measured_signal,
        molecular_signal,
        measured_error,
        reference,
    )

    assert full > 1.20
    assert relative_range < 0.02
