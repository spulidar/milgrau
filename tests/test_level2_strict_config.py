"""Fail-fast tests for productive Level 2 scientific configuration."""

from __future__ import annotations

import pytest

from milgrau.level2.config import (
    Level2ConfigurationError,
    get_block_average_minutes,
    get_cloud_screening_config,
    get_gluing_config,
    get_kfs_config,
    get_lidar_ratio,
    get_molecular_fit_config,
    get_wavelengths_to_process,
    validate_level2_config,
)


def _complete_level2_config() -> dict:
    return {
        "inversion": {
            "wavelengths_to_process": [532],
            "block_average_minutes": 20,
            "kfs_mode": "two_sided",
            "monte_carlo_iterations": 300,
            "random_seed": 143,
            "beta_ref_relative_std": 0.10,
            "aerosol_ref_fraction": 0.0,
            "min_lidar_ratio_sr": 10.0,
            "allow_negative_aerosol": False,
            "molecular_fit": {
                "ref_alt_min_m": 5000.0,
                "ref_alt_max_m": 25000.0,
                "ref_window_bins": 667,
                "max_relative_slope": 0.25,
                "max_relative_variance": 0.50,
                "min_valid_fraction": 0.50,
            },
            "gluing": {
                "window_length_bins": 120,
                "correlation_threshold": 0.95,
                "intercept_threshold": 5.0,
                "gaussian_threshold": 0.10,
                "minmax_threshold": 0.05,
                "max_relative_rmse": 0.08,
                "max_relative_bias": 0.05,
                "min_valid_fraction": 0.80,
                "max_saturation_fraction": 0.20,
                "invalid_saturation_fraction": 1.0,
                "search_min_idx": 150,
                "search_max_idx": 1200,
                "allow_single_channel_fallback": True,
                "single_channel_priority": "photon_counting",
            },
            "cloud_screening": {"enabled": False},
            "lidar_ratio_std_sr": {"532": 10.0},
            "lidar_ratios_sr": {
                "532": {f"{month:02d}": 60.0 + month for month in range(1, 13)}
            },
        }
    }


def test_complete_level2_config_validates_and_extracts_values() -> None:
    config = _complete_level2_config()

    validate_level2_config(config)

    assert get_wavelengths_to_process(config) == [532]
    assert get_block_average_minutes(config) == 20
    assert get_kfs_config(config)["random_seed"] == 143
    assert get_gluing_config(config)["gaussian_threshold"] == 0.10
    assert get_molecular_fit_config(config)["ref_window_bins"] == 667
    assert get_cloud_screening_config(config) == {"enabled": False}
    assert get_lidar_ratio(config, 532, "2026-09-09T00:00:00") == (69.0, 10.0)


def test_wavelength_request_triggers_complete_recipe_validation() -> None:
    config = _complete_level2_config()
    del config["inversion"]["random_seed"]

    with pytest.raises(Level2ConfigurationError, match="random_seed"):
        get_wavelengths_to_process(config)


def test_level2_does_not_fall_back_to_532_when_wavelengths_are_missing() -> None:
    config = _complete_level2_config()
    del config["inversion"]["wavelengths_to_process"]

    with pytest.raises(Level2ConfigurationError, match="wavelengths_to_process"):
        validate_level2_config(config)


def test_level2_does_not_accept_legacy_temporal_average_as_block_default() -> None:
    config = _complete_level2_config()
    del config["inversion"]["block_average_minutes"]
    config["inversion"]["temporal_average_minutes"] = 15

    with pytest.raises(Level2ConfigurationError, match="block_average_minutes"):
        get_block_average_minutes(config)


def test_missing_monthly_lidar_ratio_fails_before_processing() -> None:
    config = _complete_level2_config()
    del config["inversion"]["lidar_ratios_sr"]["532"]["09"]

    with pytest.raises(Level2ConfigurationError, match=r"lidar_ratios_sr\.532\.09"):
        validate_level2_config(config)


def test_missing_lidar_ratio_uncertainty_fails_before_processing() -> None:
    config = _complete_level2_config()
    del config["inversion"]["lidar_ratio_std_sr"]["532"]

    with pytest.raises(Level2ConfigurationError, match=r"lidar_ratio_std_sr\.532"):
        validate_level2_config(config)


def test_gluing_requires_every_scientific_threshold() -> None:
    config = _complete_level2_config()
    del config["inversion"]["gluing"]["gaussian_threshold"]

    with pytest.raises(Level2ConfigurationError, match="gaussian_threshold"):
        get_gluing_config(config)


def test_gluing_rejects_search_interval_smaller_than_window() -> None:
    config = _complete_level2_config()
    config["inversion"]["gluing"]["search_min_idx"] = 150
    config["inversion"]["gluing"]["search_max_idx"] = 200

    with pytest.raises(Level2ConfigurationError, match="search interval"):
        get_gluing_config(config)


def test_cloud_screening_policy_itself_is_required() -> None:
    config = _complete_level2_config()
    del config["inversion"]["cloud_screening"]

    with pytest.raises(Level2ConfigurationError, match="cloud_screening"):
        validate_level2_config(config)


def test_enabled_cloud_screening_requires_complete_detector_configuration() -> None:
    config = _complete_level2_config()
    config["inversion"]["cloud_screening"] = {
        "enabled": True,
        "min_altitude_m": 500.0,
        "max_altitude_m": 15000.0,
        "smooth_bins": 9,
        "robust_z_threshold": 6.0,
        "min_cloud_bins": 3,
        "vertical_dilation_bins": 2,
        "exclude_clouds_from_reference_fit": True,
    }

    with pytest.raises(Level2ConfigurationError, match="baseline_percentile"):
        get_cloud_screening_config(config)


def test_molecular_rayleigh_lidar_ratio_is_not_a_required_yaml_setting() -> None:
    config = _complete_level2_config()

    molecular = get_molecular_fit_config(config)

    assert "lidar_ratio_molecular_sr" not in molecular
    assert "lidar_ratio_molecular" not in molecular


def test_level2_rejects_boolean_numeric_values() -> None:
    config = _complete_level2_config()
    config["inversion"]["monte_carlo_iterations"] = True

    with pytest.raises(Level2ConfigurationError, match="monte_carlo_iterations"):
        get_kfs_config(config)


def test_level2_rejects_duplicate_wavelengths() -> None:
    config = _complete_level2_config()
    config["inversion"]["wavelengths_to_process"] = [532, 532]

    with pytest.raises(Level2ConfigurationError, match="duplicate wavelength"):
        validate_level2_config(config)
