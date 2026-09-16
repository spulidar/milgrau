"""R&D tests for legacy-inspired Rayleigh-window contamination screening.

The historical detector is useful for sharp positive anomalies, but these tests
also preserve an explicit counterexample: absence of a sharp-layer flag does not
certify a molecular window.  Productive Rayleigh selection remains unchanged.
"""

from __future__ import annotations

import numpy as np

from milgrau.level2.cloud_screening import (
    detect_anomalous_layer_mask,
    detect_reference_contamination,
)
from milgrau.level2.rayleigh_uncertainty import origin_calibration_uncertainty


_DETECTOR_OPTIONS = {
    "min_altitude_m": 8000.0,
    "max_altitude_m": 10000.0,
    "smooth_bins": 41,
    "baseline_percentile": 20.0,
    "robust_z_threshold": 5.0,
    "min_cloud_bins": 3,
    "vertical_dilation_bins": 2,
}


def _window_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    altitude_m = np.arange(8000.0, 10000.0 + 7.5, 7.5, dtype=np.float64)
    molecular_signal = np.exp(-(altitude_m - 8000.0) / 7000.0)
    measured_error = 0.02 * molecular_signal
    reference = (altitude_m >= 8500.0) & (altitude_m <= 9500.0)
    return altitude_m, molecular_signal, measured_error, reference


def _fit_factor(
    measured_signal: np.ndarray,
    molecular_signal: np.ndarray,
    measured_error: np.ndarray,
    reference: np.ndarray,
) -> float:
    result = origin_calibration_uncertainty(
        measured_signal[reference],
        molecular_signal[reference],
        measured_error[reference],
    )
    return float(result.calibration_factor)


def test_sharp_layer_that_biases_window_fit_is_flagged_by_legacy_inspired_detector() -> None:
    """A sharp positive anomaly is a useful failure mode for the old detector."""
    altitude_m, molecular_signal, measured_error, reference = _window_case()
    ratio = 1.0 + 0.8 * np.exp(-0.5 * ((altitude_m - 8650.0) / 60.0) ** 2)
    measured_signal = molecular_signal * ratio

    mask = detect_anomalous_layer_mask(ratio, altitude_m, **_DETECTOR_OPTIONS)
    contamination_fraction = detect_reference_contamination(
        mask,
        altitude_m,
        8500.0,
        9500.0,
    )
    fitted_factor = _fit_factor(
        measured_signal,
        molecular_signal,
        measured_error,
        reference,
    )

    assert contamination_fraction > 0.0
    assert fitted_factor > 1.10


def test_clean_molecular_ratio_has_no_sharp_layer_flag_and_unbiased_fit() -> None:
    """A clean synthetic molecular window should stay unflagged and fit unity."""
    altitude_m, molecular_signal, measured_error, reference = _window_case()
    measured_signal = molecular_signal.copy()
    ratio = measured_signal / molecular_signal

    mask = detect_anomalous_layer_mask(ratio, altitude_m, **_DETECTOR_OPTIONS)
    contamination_fraction = detect_reference_contamination(
        mask,
        altitude_m,
        8500.0,
        9500.0,
    )
    fitted_factor = _fit_factor(
        measured_signal,
        molecular_signal,
        measured_error,
        reference,
    )

    assert contamination_fraction == 0.0
    assert fitted_factor == 1.0


def test_broad_smooth_contamination_can_bias_fit_without_triggering_sharp_layer_detector() -> None:
    """Legacy sharp-layer screening alone must never certify a molecular window."""
    altitude_m, molecular_signal, measured_error, reference = _window_case()
    ratio = 1.0 + 0.25 * np.exp(-0.5 * ((altitude_m - 8800.0) / 600.0) ** 2)
    measured_signal = molecular_signal * ratio

    mask = detect_anomalous_layer_mask(ratio, altitude_m, **_DETECTOR_OPTIONS)
    contamination_fraction = detect_reference_contamination(
        mask,
        altitude_m,
        8500.0,
        9500.0,
    )
    fitted_factor = _fit_factor(
        measured_signal,
        molecular_signal,
        measured_error,
        reference,
    )

    # This is the important counterexample: a broad, smooth excess can look
    # locally non-anomalous to a sharp-layer detector while materially shifting
    # the molecular calibration scale.  Therefore zero flagged fraction is not
    # evidence of molecular purity.
    assert contamination_fraction == 0.0
    assert fitted_factor > 1.20
