"""Scientific and product identities shared by outputs and provenance."""

from __future__ import annotations

from typing import Final

LEVEL2_PRODUCT_SCHEMA_VERSION: Final[str] = "8"
LEVEL2_PRODUCT_SCHEMA_CHANGE: Final[str] = (
    "solar_segment_homogeneous_blocks_with_block_resolved_atmosphere"
)
ELASTIC_BACKSCATTER_INVERSION_METHOD: Final[str] = "Klett-Fernald-Sasano"
ELASTIC_BACKSCATTER_INTEGRATION_MODE: Final[str] = "two_sided"
ELASTIC_BACKSCATTER_UNCERTAINTY_METHOD: Final[str] = "selection-aware Monte Carlo"
FERNALD_SCIENTIFIC_CHANGE: Final[str] = "corrected_backward_molecular_factor_sign"

MOLECULAR_ATMOSPHERE_FALLBACK: Final[str] = "US Standard Atmosphere 1976"
MOLECULAR_ATMOSPHERE_SCIENTIFIC_CHANGE: Final[str] = (
    "hourly_level1_atmosphere_with_block_time_temperature_and_log_pressure_interpolation"
)


def elastic_inversion_algorithm_metadata(
    integration_mode: str = ELASTIC_BACKSCATTER_INTEGRATION_MODE,
) -> dict[str, str]:
    """Return the immutable scientific identity used by metadata/provenance."""
    normalized_mode = str(integration_mode).strip().lower()
    if normalized_mode != "two_sided":
        raise ValueError("The productive integration_mode is 'two_sided'.")
    return {
        "elastic_backscatter_inversion_method": ELASTIC_BACKSCATTER_INVERSION_METHOD,
        "integration_mode": normalized_mode,
        "uncertainty_method": ELASTIC_BACKSCATTER_UNCERTAINTY_METHOD,
        "rayleigh_reference_selection_policy": (
            "native-grid Rayleigh minimum QA plus progressive-path admissibility; "
            "attempt declared altitude ranges in order and, within the first supported range, "
            "minimize relative_slope+relative_variance with lower-altitude deterministic tie break"
        ),
        "rayleigh_reference_altitude_policy": (
            "declared ranges are search domains only; altitude is not a molecular-purity claim"
        ),
        "rayleigh_reference_snr_policy": "diagnostic_only_no_hard_threshold",
        "vertical_representation": (
            "strict progressive native-bin aggregation with no interpolation, padding, or gap bridging"
        ),
        "reference_selection_uncertainty_policy": (
            "native signal noise is propagated through progressive aggregation, Rayleigh QA, "
            "range fallback, reference selection, and KFS"
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
        "scientific_change": FERNALD_SCIENTIFIC_CHANGE,
        "fernald_scientific_change": FERNALD_SCIENTIFIC_CHANGE,
        "molecular_atmosphere_fallback": MOLECULAR_ATMOSPHERE_FALLBACK,
        "molecular_atmosphere_scientific_change": MOLECULAR_ATMOSPHERE_SCIENTIFIC_CHANGE,
    }
