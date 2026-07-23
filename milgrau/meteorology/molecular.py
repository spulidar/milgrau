"""Molecular optical coefficients derived from a normalized atmospheric profile."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from milgrau.level2.molecular import (
    rayleigh_phase_function,
    scattering_cross_section_bucholtz,
)
from milgrau.meteorology.contracts import AtmosphericProfile


def _immutable(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class MolecularOpticalProfile:
    wavelength_nm: float
    number_density_m3: np.ndarray
    extinction_m_inv: np.ndarray
    backscatter_m_inv_sr_inv: np.ndarray
    lidar_ratio_sr: np.ndarray
    two_way_transmission: np.ndarray
    formulation: str = "Bucholtz-1995 dry-air optical composition with moist thermodynamic number density"

    def __post_init__(self) -> None:
        if not np.isfinite(self.wavelength_nm) or self.wavelength_nm <= 0.0:
            raise ValueError("wavelength_nm must be finite and positive.")
        for name in (
            "number_density_m3",
            "extinction_m_inv",
            "backscatter_m_inv_sr_inv",
            "lidar_ratio_sr",
            "two_way_transmission",
        ):
            object.__setattr__(self, name, _immutable(getattr(self, name)))
        shapes = {getattr(self, name).shape for name in (
            "number_density_m3",
            "extinction_m_inv",
            "backscatter_m_inv_sr_inv",
            "lidar_ratio_sr",
            "two_way_transmission",
        )}
        if (
            len(shapes) != 1
            or len(next(iter(shapes))) != 1
            or next(iter(shapes))[0] < 2
        ):
            raise ValueError("All molecular fields must be conformable 1D profiles.")


def molecular_optical_profile(
    profile: AtmosphericProfile,
    wavelength_nm: float,
) -> MolecularOpticalProfile:
    """Compute alpha, beta(pi), lidar ratio and transmission in SI units."""
    if not isinstance(profile, AtmosphericProfile):
        raise TypeError("profile must be AtmosphericProfile.")
    wavelength = float(wavelength_nm)
    sigma_m2 = float(scattering_cross_section_bucholtz(wavelength)) * 1e-4
    phase_pi = float(rayleigh_phase_function(np.pi, wavelength))
    number_density = np.asarray(profile.molecular_number_density_m3, dtype=np.float64)
    extinction = number_density * sigma_m2
    backscatter = extinction * phase_pi / (4.0 * np.pi)
    lidar_ratio = np.divide(
        extinction,
        backscatter,
        out=np.full(extinction.shape, np.nan, dtype=np.float64),
        where=np.isfinite(backscatter) & (backscatter > 0.0),
    )
    if np.isfinite(extinction).all() and np.all(extinction >= 0.0):
        optical_depth = cumulative_trapezoid(
            extinction,
            profile.geometric_altitude_m,
            initial=0.0,
        )
        transmission = np.exp(-2.0 * optical_depth)
    else:
        transmission = np.full(extinction.shape, np.nan, dtype=np.float64)
    return MolecularOpticalProfile(
        wavelength_nm=wavelength,
        number_density_m3=number_density,
        extinction_m_inv=extinction,
        backscatter_m_inv_sr_inv=backscatter,
        lidar_ratio_sr=lidar_ratio,
        two_way_transmission=transmission,
    )
