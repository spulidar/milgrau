"""Versioned scientific and product identities shared by outputs/provenance."""

from __future__ import annotations

from typing import Final

LEVEL2_PRODUCT_SCHEMA_VERSION: Final[str] = "1"
LEVEL2_RETRIEVAL_METHOD_VERSION: Final[str] = "3"
LEVEL2_RETRIEVAL_METHOD_CHANGE: Final[str] = (
    "joint_signal_uncertainty_support_missing_uncertainty_rejection_and_"
    "conservative_correlated_block_uncertainty"
)

ELASTIC_BACKSCATTER_INVERSION_METHOD: Final[str] = "Klett-Fernald-Sasano"
ELASTIC_BACKSCATTER_INTEGRATION_MODE: Final[str] = "backward"
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
        "level2_retrieval_method_change": LEVEL2_RETRIEVAL_METHOD_CHANGE,
        "optical_block_uncertainty_correlation_policy": "fully_correlated_upper_bound",
        "optical_block_uncertainty_aggregation_formula": (
            "sigma_mean=sum(sigma_block)/n_effective on common value/error support"
        ),
        "uncertainty_component_dependence": (
            "profile measurement noise treated independent within temporal block means; "
            "aerosol lidar-ratio nuisance shared across blocks; reference-boundary dependence "
            "not decomposed; aggregate optical uncertainty therefore uses the fully correlated upper bound"
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
