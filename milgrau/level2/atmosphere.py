"""Compatibility import path for the shared atmosphere physics kernel.

New code should import from :mod:`milgrau.physics.atmosphere`. This module keeps
only the Python import path stable; no legacy Level 1 data compatibility is
implemented here.
"""

from milgrau.physics.atmosphere import geometric_to_geopotential_altitude, get_standard_atmosphere

__all__ = ["geometric_to_geopotential_altitude", "get_standard_atmosphere"]
