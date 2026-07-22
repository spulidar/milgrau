"""Independent elastic-lidar forward model for scientific KFS tests.

This module deliberately does not import :mod:`milgrau.level2.kfs`.  It starts
from the range-corrected elastic lidar equation, in SI units,

    X(z) = C [beta_m(z) + beta_a(z)]
           exp(-2 integral[z0, z] (S_m beta_m(s) + S_a(s) beta_a(s)) ds),

where ``X`` is the range-corrected signal, beta is in m-1 sr-1, lidar ratios
are in sr, altitude is in m, and ``C`` is an arbitrary instrumental constant.
The trapezoidal optical-depth integration below is therefore independent of
the algebra used by the inverse Fernald implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from milgrau.level2.atmosphere import get_standard_atmosphere
from milgrau.level2.constants import RAYLEIGH_LIDAR_RATIO_SR
from milgrau.level2.molecular import calculate_molecular_profile


@dataclass(frozen=True, slots=True)
class ElasticSyntheticCase:
    """Known atmosphere, signal and exact boundary condition on an SI grid."""

    wavelength_nm: int
    altitude_m: np.ndarray
    molecular_backscatter_m_inv_sr_inv: np.ndarray
    aerosol_backscatter_m_inv_sr_inv: np.ndarray
    aerosol_lidar_ratio_sr: np.ndarray
    molecular_lidar_ratio_sr: float
    range_corrected_signal: np.ndarray
    reference_index: int
    beta_total_reference_m_inv_sr_inv: float


def _cumulative_trapezoid(values: np.ndarray, altitude_m: np.ndarray) -> np.ndarray:
    """Integrate a 1D profile from the first altitude using explicit trapezoids."""
    integral = np.zeros(values.size, dtype=np.float64)
    for index in range(1, values.size):
        dz_m = altitude_m[index] - altitude_m[index - 1]
        integral[index] = integral[index - 1] + 0.5 * (values[index - 1] + values[index]) * dz_m
    return integral


def elastic_lidar_forward_model(
    altitude_m: np.ndarray,
    molecular_backscatter_m_inv_sr_inv: np.ndarray,
    aerosol_backscatter_m_inv_sr_inv: np.ndarray,
    molecular_lidar_ratio_sr: float,
    aerosol_lidar_ratio_sr: float | np.ndarray,
    *,
    instrumental_constant: float = 3.0e12,
) -> np.ndarray:
    """Generate a noiseless elastic range-corrected signal from known profiles."""
    altitude = np.asarray(altitude_m, dtype=np.float64)
    beta_molecular = np.asarray(molecular_backscatter_m_inv_sr_inv, dtype=np.float64)
    beta_aerosol = np.asarray(aerosol_backscatter_m_inv_sr_inv, dtype=np.float64)
    lidar_ratio_aerosol = np.broadcast_to(
        np.asarray(aerosol_lidar_ratio_sr, dtype=np.float64),
        altitude.shape,
    )
    if not (altitude.ndim == beta_molecular.ndim == beta_aerosol.ndim == 1):
        raise ValueError("Forward-model profiles must be one-dimensional.")
    if not (altitude.shape == beta_molecular.shape == beta_aerosol.shape == lidar_ratio_aerosol.shape):
        raise ValueError("Forward-model profiles must share one shape.")
    if altitude.size < 3 or not np.all(np.isfinite(altitude)) or not np.all(np.diff(altitude) > 0.0):
        raise ValueError("Forward-model altitude must be finite and strictly increasing.")
    if np.any(beta_molecular <= 0.0) or np.any(beta_aerosol < 0.0):
        raise ValueError("Forward-model backscatter must be physical.")
    if not np.isfinite(molecular_lidar_ratio_sr) or molecular_lidar_ratio_sr <= 0.0:
        raise ValueError("Molecular lidar ratio must be finite and positive.")
    if np.any(~np.isfinite(lidar_ratio_aerosol)) or np.any(lidar_ratio_aerosol <= 0.0):
        raise ValueError("Aerosol lidar ratio must be finite and positive.")
    if not np.isfinite(instrumental_constant) or instrumental_constant <= 0.0:
        raise ValueError("Instrumental constant must be finite and positive.")

    total_backscatter = beta_molecular + beta_aerosol
    total_extinction_m_inv = (
        molecular_lidar_ratio_sr * beta_molecular
        + lidar_ratio_aerosol * beta_aerosol
    )
    optical_depth = _cumulative_trapezoid(total_extinction_m_inv, altitude)
    signal = instrumental_constant * total_backscatter * np.exp(-2.0 * optical_depth)
    if np.any(~np.isfinite(signal)) or np.any(signal <= 0.0):
        raise AssertionError("Synthetic elastic signal must remain positive and finite.")
    return signal.astype(np.float64)


def make_elastic_case(
    wavelength_nm: int,
    *,
    vertical_step_m: float = 30.0,
    aerosol: bool = True,
    variable_lidar_ratio: bool = False,
    reference_altitude_m: float = 7200.0,
) -> ElasticSyntheticCase:
    """Build equivalent 355/532-nm cases with an exact internal boundary."""
    altitude_m = np.arange(300.0, 12000.0 + 0.5 * vertical_step_m, vertical_step_m, dtype=np.float64)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_m)
    beta_molecular, _ = calculate_molecular_profile(temperature_k, pressure_hpa, float(wavelength_nm))
    if aerosol:
        beta_aerosol = 2.2e-6 * np.exp(-(altitude_m - altitude_m[0]) / 2100.0)
    else:
        beta_aerosol = np.zeros_like(altitude_m)
    if variable_lidar_ratio:
        # Smooth 48--62 sr profile: this exercises the generalized integral
        # without claiming an operationally validated altitude-dependent prior.
        lidar_ratio_aerosol = 55.0 + 7.0 * np.tanh((altitude_m - 4500.0) / 1800.0)
    else:
        lidar_ratio_aerosol = np.full_like(altitude_m, 55.0)
    signal = elastic_lidar_forward_model(
        altitude_m,
        beta_molecular,
        beta_aerosol,
        RAYLEIGH_LIDAR_RATIO_SR,
        lidar_ratio_aerosol,
    )
    reference_index = int(np.argmin(np.abs(altitude_m - reference_altitude_m)))
    return ElasticSyntheticCase(
        wavelength_nm=int(wavelength_nm),
        altitude_m=altitude_m,
        molecular_backscatter_m_inv_sr_inv=beta_molecular,
        aerosol_backscatter_m_inv_sr_inv=beta_aerosol,
        aerosol_lidar_ratio_sr=lidar_ratio_aerosol,
        molecular_lidar_ratio_sr=float(RAYLEIGH_LIDAR_RATIO_SR),
        range_corrected_signal=signal,
        reference_index=reference_index,
        beta_total_reference_m_inv_sr_inv=float(
            beta_molecular[reference_index] + beta_aerosol[reference_index]
        ),
    )
