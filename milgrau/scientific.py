"""Versioned scientific identities shared by products and provenance."""

from __future__ import annotations

from typing import Final

ELASTIC_BACKSCATTER_INVERSION_METHOD: Final[str] = "Klett-Fernald-Sasano"
ELASTIC_BACKSCATTER_INTEGRATION_MODE: Final[str] = "two_sided"
ELASTIC_BACKSCATTER_UNCERTAINTY_METHOD: Final[str] = "Monte Carlo"
FERNALD_IMPLEMENTATION_VERSION: Final[str] = "2"
FERNALD_SCIENTIFIC_CHANGE: Final[str] = "corrected_backward_molecular_factor_sign"

MOLECULAR_ATMOSPHERE_FALLBACK: Final[str] = "US Standard Atmosphere 1976"
MOLECULAR_ATMOSPHERE_IMPLEMENTATION_VERSION: Final[str] = "3"
MOLECULAR_ATMOSPHERE_SCIENTIFIC_CHANGE: Final[str] = (
    "level1_materialized_atmosphere_with_log_pressure_interpolation"
)


def elastic_inversion_algorithm_metadata() -> dict[str, str]:
    """Return the immutable scientific identity used by metadata/provenance."""
    return {
        "elastic_backscatter_inversion_method": ELASTIC_BACKSCATTER_INVERSION_METHOD,
        "integration_mode": ELASTIC_BACKSCATTER_INTEGRATION_MODE,
        "uncertainty_method": ELASTIC_BACKSCATTER_UNCERTAINTY_METHOD,
        "fernald_implementation_version": FERNALD_IMPLEMENTATION_VERSION,
        "scientific_change": FERNALD_SCIENTIFIC_CHANGE,
        "fernald_scientific_change": FERNALD_SCIENTIFIC_CHANGE,
        "molecular_atmosphere_fallback": MOLECULAR_ATMOSPHERE_FALLBACK,
        "molecular_atmosphere_implementation_version": MOLECULAR_ATMOSPHERE_IMPLEMENTATION_VERSION,
        "molecular_atmosphere_scientific_change": MOLECULAR_ATMOSPHERE_SCIENTIFIC_CHANGE,
    }
