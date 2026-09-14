"""Regression guards for the Level 2 P1 monolith/default cleanup."""

from __future__ import annotations

import numpy as np
import pytest

from milgrau.level2.config import Level2ConfigurationError
import milgrau.level2.optical_retrieval as optical_retrieval
from milgrau.level2.optical_retrieval import (
    evaluate_rayleigh_reference,
    run_kfs_profile,
)


def test_rayleigh_qa_has_no_local_threshold_defaults() -> None:
    altitude = np.arange(5, dtype=np.float64) * 100.0
    simulated = np.ones(5, dtype=np.float64)
    measured = simulated * 2.0
    incomplete_fit_config = {
        "max_relative_slope": 0.25,
        "max_relative_variance": 0.50,
    }

    with pytest.raises(KeyError, match="min_valid_fraction"):
        evaluate_rayleigh_reference(
            measured,
            simulated,
            altitude,
            reference_center_idx=2,
            reference_window_bins=5,
            fit_config=incomplete_fit_config,
            calibration_factor=2.0,
        )


def test_kfs_profile_has_no_local_scientific_defaults() -> None:
    altitude = np.arange(5, dtype=np.float64) * 100.0
    rcs = np.linspace(5.0, 1.0, 5)
    beta_mol = np.full(5, 1.0e-6)

    with pytest.raises(Level2ConfigurationError, match="monte_carlo_iterations"):
        run_kfs_profile(
            rcs=rcs,
            rcs_error=np.full(5, 0.1),
            altitude_m=altitude,
            beta_mol=beta_mol,
            ref_idx=4,
            lr_base=50.0,
            lr_std=5.0,
            config={"inversion": {}},
        )


def test_productive_kfs_wrapper_passes_explicit_backward_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_kfs(**kwargs):
        captured.update(kwargs)
        profile = np.zeros(5, dtype=np.float64)
        diagnostics = {"backward_valid": True, "forward_valid": False}
        return profile, profile.copy(), profile.copy(), profile.copy(), diagnostics

    monkeypatch.setattr(
        optical_retrieval,
        "kfs_inversion_monte_carlo",
        fake_kfs,
    )
    config = {
        "inversion": {
            "kfs_mode": "backward",
            "monte_carlo_iterations": 20,
            "random_seed": 17,
            "beta_ref_relative_std": 0.05,
            "aerosol_ref_fraction": 0.0,
            "min_lidar_ratio_sr": 10.0,
            "allow_negative_aerosol": False,
        }
    }

    run_kfs_profile(
        rcs=np.linspace(5.0, 1.0, 5),
        rcs_error=np.full(5, 0.1),
        altitude_m=np.arange(5, dtype=np.float64) * 100.0,
        beta_mol=np.full(5, 1.0e-6),
        ref_idx=4,
        lr_base=50.0,
        lr_std=5.0,
        config=config,
    )

    assert captured["mode"] == "backward"
    assert captured["n_iterations"] == 20
    assert captured["seed"] == 17
