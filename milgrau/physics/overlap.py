"""First-order geometrical lidar overlap diagnostics.

The current kernel intentionally models only a coaxial, parallel transmitter and
receiver with circular apertures and a uniform circular laser footprint.  It is
an instrument-diagnostic model: MILGRAU must not use this curve as a productive
overlap correction until the station geometry has been experimentally
characterized.
"""

from __future__ import annotations

from math import acos, pi, sqrt
from numbers import Integral, Real

import numpy as np

OVERLAP_MODEL_ID = "coaxial_uniform_disk_geometric_v1"


def _finite_number(value: Real, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _positive(value: Real, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return result


def _non_negative(value: Real, label: str) -> float:
    result = _finite_number(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative.")
    return result


def receiver_field_stop_diameter_m(*, focal_length_m: Real, fov_full_angle_mrad: Real) -> float:
    """Return the paraxial focal-plane field-stop diameter implied by a full FOV.

    This is a consistency diagnostic only.  It assumes the configured FOV is the
    full angular field and a field stop in the telescope focal plane.
    """
    focal_length = _positive(focal_length_m, "focal_length_m")
    fov_full = _positive(fov_full_angle_mrad, "fov_full_angle_mrad") * 1.0e-3
    return focal_length * fov_full


def coaxial_full_overlap_range_m(
    *,
    telescope_diameter_m: Real,
    laser_beam_diameter_m: Real,
    receiver_fov_full_angle_mrad: Real,
    laser_divergence_full_angle_mrad: Real,
) -> float | None:
    """Return the first-order exact-full-overlap range for coaxial parallel axes.

    For full angular receiver FOV ``Psi_T`` and full laser divergence ``Psi_L``,
    the paraxial circular-aperture geometry gives

        R_full = (D_T + D_L) / (Psi_T - Psi_L)

    for a coaxial system.  No finite exact-full-overlap range exists in this
    idealized model when ``Psi_T <= Psi_L``.
    """
    telescope = _positive(telescope_diameter_m, "telescope_diameter_m")
    beam = _positive(laser_beam_diameter_m, "laser_beam_diameter_m")
    receiver_fov = _positive(receiver_fov_full_angle_mrad, "receiver_fov_full_angle_mrad") * 1.0e-3
    divergence = _non_negative(
        laser_divergence_full_angle_mrad,
        "laser_divergence_full_angle_mrad",
    ) * 1.0e-3
    angular_margin = receiver_fov - divergence
    if angular_margin <= 0.0:
        return None
    return (telescope + beam) / angular_margin


def _circle_intersection_area(radius_a: float, radius_b: float, separation: float) -> float:
    if radius_a <= 0.0 or radius_b <= 0.0:
        return 0.0
    if separation >= radius_a + radius_b:
        return 0.0
    if separation <= abs(radius_a - radius_b):
        return pi * min(radius_a, radius_b) ** 2
    if separation == 0.0:
        return pi * min(radius_a, radius_b) ** 2

    cos_a = np.clip(
        (separation**2 + radius_a**2 - radius_b**2) / (2.0 * separation * radius_a),
        -1.0,
        1.0,
    )
    cos_b = np.clip(
        (separation**2 + radius_b**2 - radius_a**2) / (2.0 * separation * radius_b),
        -1.0,
        1.0,
    )
    radicand = (
        (-separation + radius_a + radius_b)
        * (separation + radius_a - radius_b)
        * (separation - radius_a + radius_b)
        * (separation + radius_a + radius_b)
    )
    return (
        radius_a**2 * acos(float(cos_a))
        + radius_b**2 * acos(float(cos_b))
        - 0.5 * sqrt(max(0.0, radicand))
    )


def coaxial_geometric_overlap(
    altitude_m: np.ndarray | Real,
    *,
    telescope_diameter_m: Real,
    laser_beam_diameter_m: Real,
    receiver_fov_full_angle_mrad: Real,
    laser_divergence_full_angle_mrad: Real,
    quadrature_order: int = 64,
) -> np.ndarray:
    """Estimate the coaxial geometrical overlap fraction on an AGL range grid.

    The laser is represented as a uniform circular disk whose radius grows
    linearly with the configured full-angle divergence.  For each scattering
    point, the accepted telescope-aperture fraction is computed from the
    intersection of the physical telescope aperture and the receiver angular
    acceptance disk.  The result is then area-averaged over the laser disk.

    This deliberately excludes real-beam intensity structure, central
    obstruction, defocus, filter-angle effects, optical vignetting and
    misalignment.  It therefore remains a diagnostic estimate until validated
    against an experimental overlap characterization.
    """
    telescope = _positive(telescope_diameter_m, "telescope_diameter_m")
    beam = _positive(laser_beam_diameter_m, "laser_beam_diameter_m")
    receiver_fov = _positive(receiver_fov_full_angle_mrad, "receiver_fov_full_angle_mrad") * 1.0e-3
    divergence = _non_negative(
        laser_divergence_full_angle_mrad,
        "laser_divergence_full_angle_mrad",
    ) * 1.0e-3
    if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, Integral) or quadrature_order < 8:
        raise ValueError("quadrature_order must be an integer >= 8.")

    altitude = np.asarray(altitude_m, dtype=float)
    if np.any(~np.isfinite(altitude)):
        raise ValueError("altitude_m must contain only finite values.")
    if np.any(altitude < 0.0):
        raise ValueError("altitude_m must be non-negative.")

    telescope_radius = 0.5 * telescope
    receiver_half_angle = 0.5 * receiver_fov
    laser_half_angle = 0.5 * divergence
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    output = np.zeros_like(altitude, dtype=float)

    for index, range_m in np.ndenumerate(altitude):
        if range_m <= 0.0:
            output[index] = 0.0
            continue
        laser_radius = 0.5 * beam + laser_half_angle * range_m
        acceptance_radius = receiver_half_angle * range_m
        if acceptance_radius <= 0.0:
            output[index] = 0.0
            continue

        radial = 0.5 * (nodes + 1.0) * laser_radius
        radial_weights = 0.5 * weights * laser_radius
        aperture_fractions = np.fromiter(
            (
                _circle_intersection_area(telescope_radius, acceptance_radius, float(rho))
                / (pi * telescope_radius**2)
                for rho in radial
            ),
            dtype=float,
            count=radial.size,
        )
        laser_area_weights = 2.0 * radial / laser_radius**2
        value = float(np.sum(radial_weights * laser_area_weights * aperture_fractions))
        output[index] = np.clip(value, 0.0, 1.0)

    return output
