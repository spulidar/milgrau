"""Public Level 2 retrieval API with a strict Level 1 atmosphere contract.

The numerical retrieval implementation remains in ``_retrieval_impl`` while
this module owns the Level 1 -> Level 2 boundary. Thermodynamic source
selection, interpolation and fallback are completed by Level 1; Level 2 only
consumes the canonical atmosphere stored in the Level 1 NetCDF.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import xarray as xr

# Re-export the existing numerical retrieval API while the monolithic module is
# progressively split into smaller scientific components.
from milgrau.level2._retrieval_impl import *  # noqa: F401,F403
from milgrau.level2._retrieval_impl import _run_retrieval_stage
from milgrau.level2.config import get_kfs_mode, get_lidar_ratio, get_molecular_fit_config
from milgrau.level2.molecular import calculate_molecular_profile, calculate_simulated_molecular_signal


def build_thermodynamic_profile(
    ds_l1: xr.Dataset,
    altitude_agl_m: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Read the complete canonical atmosphere materialized by Level 1.

    No radiosonde/ERA5 selection, interpolation or standard-atmosphere fallback
    is permitted here. Old Level 1 files without ``Atmospheric_*`` variables are
    intentionally rejected and must be reprocessed.
    """
    del config
    required = ("Atmospheric_Temperature_K", "Atmospheric_Pressure_hPa")
    missing = [name for name in required if name not in ds_l1]
    if missing:
        raise KeyError(
            "Level 1 product lacks canonical thermodynamic variable(s) "
            f"{missing}; reprocess Level 1 with the current LIPANCORA pipeline."
        )

    altitude = np.asarray(altitude_agl_m, dtype=np.float64)
    temperature_k = np.asarray(ds_l1["Atmospheric_Temperature_K"].values, dtype=np.float64)
    pressure_hpa = np.asarray(ds_l1["Atmospheric_Pressure_hPa"].values, dtype=np.float64)
    if ds_l1["Atmospheric_Temperature_K"].dims != ("altitude",):
        raise ValueError("Atmospheric_Temperature_K must have dimensions ('altitude',).")
    if ds_l1["Atmospheric_Pressure_hPa"].dims != ("altitude",):
        raise ValueError("Atmospheric_Pressure_hPa must have dimensions ('altitude',).")
    if temperature_k.shape != altitude.shape or pressure_hpa.shape != altitude.shape:
        raise ValueError("Stored Level 1 atmosphere must match the Level 2 lidar altitude grid exactly.")
    if not np.all(np.isfinite(temperature_k)) or np.any(temperature_k <= 0.0):
        raise ValueError("Atmospheric_Temperature_K must be finite and positive on every altitude bin.")
    if not np.all(np.isfinite(pressure_hpa)) or np.any(pressure_hpa <= 0.0):
        raise ValueError("Atmospheric_Pressure_hPa must be finite and positive on every altitude bin.")

    source = str(ds_l1.attrs.get("thermodynamic_profile_source_type", "")).strip()
    if not source:
        raise ValueError("Level 1 product lacks thermodynamic_profile_source_type provenance.")
    return pressure_hpa, temperature_k, source


def build_molecular_model(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
) -> MolecularModel:
    """Build the molecular atmosphere from the canonical Level 1 thermodynamics."""
    pressure_hpa, temperature_k, source = build_thermodynamic_profile(ds_l1, altitude_m, config)
    backscatter, extinction = calculate_molecular_profile(temperature_k, pressure_hpa, wavelength_nm)
    simulated_signal, transmission = calculate_simulated_molecular_signal(backscatter, extinction, altitude_m)
    positive_altitudes = altitude_m[altitude_m > 0.0]
    safe_altitude = np.where(
        altitude_m > 0.0,
        altitude_m,
        positive_altitudes[0] if positive_altitudes.size else 1.0,
    )
    lidar_ratio, lidar_ratio_std = get_lidar_ratio(config, wavelength_nm, ds_l1["time"].values[0])
    return MolecularModel(
        source=source,
        backscatter=backscatter,
        extinction=extinction,
        transmission=transmission,
        simulated_signal=simulated_signal,
        simulated_range_corrected_signal=simulated_signal * safe_altitude**2,
        fit_config=get_molecular_fit_config(config),
        lidar_ratio_assumed_sr=lidar_ratio,
        lidar_ratio_std_sr=lidar_ratio_std,
        kfs_mode=get_kfs_mode(config),
    )


def process_wavelength(
    ds_l1: xr.Dataset,
    wavelength_nm: int,
    altitude_m: np.ndarray,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> WavelengthRetrievalResult:
    """Process one wavelength using the canonical Level 1 atmosphere contract."""
    inputs = _run_retrieval_stage(
        "selection_and_blocking",
        lambda: prepare_wavelength_blocks(ds_l1, wavelength_nm, altitude_m, config),
    )
    glued = _run_retrieval_stage(
        "gluing",
        lambda: glue_signal_blocks(inputs, altitude_m, logger),
    )
    molecular_model = _run_retrieval_stage(
        "molecular_model",
        lambda: build_molecular_model(ds_l1, wavelength_nm, altitude_m, config),
    )
    molecular, optical, rayleigh, kfs = _run_retrieval_stage(
        "rayleigh_kfs",
        lambda: retrieve_optical_blocks(
            inputs,
            glued,
            molecular_model,
            altitude_m,
            config,
            logger,
        ),
    )
    return _run_retrieval_stage(
        "result_assembly",
        lambda: assemble_wavelength_result(inputs, glued, molecular, optical, rayleigh, kfs),
    )
