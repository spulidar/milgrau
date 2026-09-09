"""Pure scientific-physics kernels shared across MILGRAU processing levels."""

from milgrau.physics.atmosphere import geometric_to_geopotential_altitude, get_standard_atmosphere

__all__ = ["geometric_to_geopotential_altitude", "get_standard_atmosphere"]
