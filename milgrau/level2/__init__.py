"""Level 2 processing modules."""

from milgrau.level2.discovery import discover_level1_files
from milgrau.io.paths import LEVEL2_SUFFIX

# The public Level 2 boundary owns retrieval-input semantics while the numerical
# implementation is still split across the legacy monolith. Install the
# supported-domain evaluator before importing LEBEAR/retrieval so productive
# paths accept only contiguous Rayleigh-anchored support and do not reject
# intentional Level 1 bin-shift edge NaNs.
from milgrau.level2 import _retrieval_impl as _retrieval_impl
from milgrau.level2.retrieval_input_qa import evaluate_retrieval_input_supported_domain as _supported_domain_qa

_retrieval_impl._evaluate_retrieval_input = _supported_domain_qa

from milgrau.level2.lebear import process_level_2, process_single_level1_file
from milgrau.level2.cloud_screening import cloud_screening_config, detect_anomalous_layer_mask, detect_reference_contamination
from milgrau.level2.contracts import RetrievalInputInvalidReason, SignalSource, WavelengthRetrievalResult
from milgrau.level2.completeness import (
    Level2ProductContract,
    ProductCompleteness,
    ProductStatus,
    WavelengthFailureCode,
    WavelengthFailureStage,
)
from milgrau.level2.gluing import propagate_glued_error, slide_glue_signals
from milgrau.level2.kfs import fernald_inversion, kfs_inversion_monte_carlo
from milgrau.level2.molecular import calculate_molecular_profile, find_optimal_reference_altitude

__all__ = [
    "LEVEL2_SUFFIX",
    "Level2ProductContract",
    "ProductCompleteness",
    "ProductStatus",
    "RetrievalInputInvalidReason",
    "SignalSource",
    "WavelengthFailureCode",
    "WavelengthFailureStage",
    "WavelengthRetrievalResult",
    "calculate_molecular_profile",
    "cloud_screening_config",
    "detect_anomalous_layer_mask",
    "detect_reference_contamination",
    "discover_level1_files",
    "find_optimal_reference_altitude",
    "fernald_inversion",
    "kfs_inversion_monte_carlo",
    "process_level_2",
    "process_single_level1_file",
    "propagate_glued_error",
    "slide_glue_signals",
]
