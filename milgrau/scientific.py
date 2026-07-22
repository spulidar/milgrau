"""Versioned scientific identities shared by products and provenance."""

from __future__ import annotations

from typing import Final

ELASTIC_BACKSCATTER_INVERSION_METHOD: Final[str] = "Klett-Fernald-Sasano"
ELASTIC_BACKSCATTER_INTEGRATION_MODE: Final[str] = "two_sided"
ELASTIC_BACKSCATTER_UNCERTAINTY_METHOD: Final[str] = "Monte Carlo"
FERNALD_IMPLEMENTATION_VERSION: Final[str] = "2"
FERNALD_SCIENTIFIC_CHANGE: Final[str] = "corrected_backward_molecular_factor_sign"


def elastic_inversion_algorithm_metadata() -> dict[str, str]:
    """Return the immutable scientific identity used by metadata/provenance."""
    return {
        "elastic_backscatter_inversion_method": ELASTIC_BACKSCATTER_INVERSION_METHOD,
        "integration_mode": ELASTIC_BACKSCATTER_INTEGRATION_MODE,
        "uncertainty_method": ELASTIC_BACKSCATTER_UNCERTAINTY_METHOD,
        "fernald_implementation_version": FERNALD_IMPLEMENTATION_VERSION,
        "scientific_change": FERNALD_SCIENTIFIC_CHANGE,
    }
