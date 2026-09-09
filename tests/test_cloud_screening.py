"""Tests for preliminary cloud/anomalous-layer screening utilities."""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.level2.cloud_screening import (
    cloud_screening_config,
    detect_anomalous_layer_mask,
    detect_reference_contamination,
)
from milgrau.level2.config import Level2ConfigurationError


DETECTOR_OPTIONS = {
    "min_altitude_m": 500.0,
    "max_altitude_m": 2500.0,
    "smooth_bins": 11,
    "baseline_percentile": 20.0,
    "robust_z_threshold": 5.0,
    "min_cloud_bins": 3,
    "vertical_dilation_bins": 2,
}


def test_detect_anomalous_layer_mask_flags_positive_spike_layer() -> None:
    """A strong multi-bin positive anomaly should be flagged and dilated."""
    altitude = np.arange(0.0, 3000.0, 7.5)
    signal = np.exp(-altitude / 1200.0) + 0.1
    layer = (altitude >= 1200.0) & (altitude <= 1260.0)
    signal[layer] += 20.0

    mask = detect_anomalous_layer_mask(signal, altitude, **DETECTOR_OPTIONS)

    assert mask.dtype == bool
    assert mask[layer].any()
    assert mask.sum() >= layer.sum()


def test_detect_anomalous_layer_mask_ignores_below_min_altitude() -> None:
    """A strong anomaly below the configured search range should not be flagged."""
    altitude = np.arange(0.0, 3000.0, 7.5)
    signal = np.exp(-altitude / 1200.0) + 0.1
    low_layer = (altitude >= 150.0) & (altitude <= 220.0)
    signal[low_layer] += 50.0

    mask = detect_anomalous_layer_mask(signal, altitude, **DETECTOR_OPTIONS)

    assert not mask[low_layer].any()


def test_detect_reference_contamination_returns_fraction() -> None:
    """Reference contamination should be measured as flagged-bin fraction."""
    altitude = np.arange(0.0, 1000.0, 100.0)
    mask = np.zeros_like(altitude, dtype=bool)
    mask[(altitude >= 300.0) & (altitude <= 500.0)] = True

    fraction = detect_reference_contamination(mask, altitude, 300.0, 700.0)

    assert np.isclose(fraction, 3.0 / 5.0)


def test_cloud_screening_config_requires_explicit_policy() -> None:
    """Cloud screening cannot silently default to disabled."""
    with pytest.raises(Level2ConfigurationError, match="cloud_screening"):
        cloud_screening_config({"inversion": {}})


def test_cloud_screening_config_accepts_explicit_disabled_state() -> None:
    config = {"inversion": {"cloud_screening": {"enabled": False}}}

    assert cloud_screening_config(config) == {"enabled": False}


def test_cloud_screening_config_requires_complete_enabled_detector() -> None:
    config = {
        "inversion": {
            "cloud_screening": {
                "enabled": True,
                "min_altitude_m": 500.0,
                "max_altitude_m": 15000.0,
                "smooth_bins": 9,
                "baseline_percentile": 20.0,
                "robust_z_threshold": 4.5,
                "min_cloud_bins": 3,
                "vertical_dilation_bins": 4,
                "exclude_clouds_from_reference_fit": True,
            }
        }
    }

    extracted = cloud_screening_config(config)

    assert extracted["enabled"] is True
    assert extracted["baseline_percentile"] == 20.0
    assert extracted["robust_z_threshold"] == 4.5
    assert extracted["vertical_dilation_bins"] == 4
