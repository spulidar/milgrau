"""Versioned scientific and product identities shared by outputs/provenance."""

from __future__ import annotations

from typing import Final

LEVEL2_PRODUCT_SCHEMA_VERSION: Final[str] = "4"
LEVEL2_PRODUCT_SCHEMA_CHANGE: Final[str] = (
    "method_v5_progressive_grid_selection_aware_uncertainty_and_support"
)
LEVEL2_RETRIEVAL_METHOD_VERSION: Final[str] = "5"
LEVEL2_RETRIEVAL_METHOD_CHANGE: Final[str] = (
    "progressive_grid_tiered_high_reference_selection_with_selection_aware_monte_carlo"
)

ELASTIC_BACKSCATTER_INVERSION_METHOD: Final[str] = "Klett-Fernald-Sasano"
ELASTIC_BACKSCATTER_INTEGRATION_MODE: Final[str] = "backward"
ELASTIC_BACKSCATTER_UNCERTAINTY_METHOD: Final[str] = "selection-aware Monte Carlo"
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
        "level2_retrieval_method_change": LEVEL2_RETRIEVAL_METHOD_CHANGE,
        "rayleigh_reference_selection_policy": (
            "native-grid Rayleigh minimum QA plus progressive-path admissibility; "
            "attempt declared altitude tiers in order and, within the highest supported tier, "
            "minimize relative_slope+relative_variance with lower-altitude deterministic tie break"
        ),
        "rayleigh_reference_altitude_policy": (
            "declared tiers are search domains only; altitude is not a molecular-purity claim"
        ),
        "rayleigh_reference_snr_policy": "diagnostic_only_no_hard_threshold",
        "vertical_representation": (
            "strict progressive native-bin aggregation with no interpolation, padding, or gap bridging"
        ),
        "reference_selection_uncertainty_policy": (
            "native signal noise is propagated through progressive aggregation, Rayleigh QA, "
            "tier fallback, reference selection, and KFS"
        ),
        "boundary_systematic_policy": (
            "residual aerosol fraction f is an outer deterministic sensitivity scenario, not a random prior"
        ),
        "mc_support_policy": (
            "altitude-resolved finite-realization fraction is diagnostic and is not converted to a pass/fail cutoff"
        ),
        "gluing_uncertainty_scope": (
            "partial measurement-noise propagation; fitted gluing slope/intercept uncertainty excluded"
        ),
        "fernald_implementation_version": FERNALD_IMPLEMENTATION_VERSION,
        "scientific_change": FERNALD_SCIENTIFIC_CHANGE,
        "fernald_scientific_change": FERNALD_SCIENTIFIC_CHANGE,
        "molecular_atmosphere_fallback": MOLECULAR_ATMOSPHERE_FALLBACK,
        "molecular_atmosphere_implementation_version": MOLECULAR_ATMOSPHERE_IMPLEMENTATION_VERSION,
        "molecular_atmosphere_scientific_change": MOLECULAR_ATMOSPHERE_SCIENTIFIC_CHANGE,
    }
