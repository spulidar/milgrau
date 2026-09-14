"""Level 2 processing and scientific APIs.

Package import is intentionally declarative: scientific behavior is defined by
canonical modules and is never installed through import-time monkey patching.
The package surface contains productive processing/contracts plus a small set of
explicitly retained numerical research kernels. Productive elastic retrieval is
backward Klett--Fernald; availability of forward/two-sided research kernels does
not change that processing contract.
"""

from milgrau.io.paths import LEVEL2_SUFFIX
from milgrau.level2.cloud_screening import (
    cloud_screening_config,
    detect_anomalous_layer_mask,
    detect_reference_contamination,
)
from milgrau.level2.completeness import (
    Level2ProductContract,
    ProductCompleteness,
    ProductStatus,
    WavelengthFailureCode,
    WavelengthFailureStage,
)
from milgrau.level2.contracts import (
    RetrievalInputInvalidReason,
    SignalSource,
    WavelengthRetrievalResult,
)
from milgrau.level2.discovery import discover_level1_files
from milgrau.level2.gluing import propagate_glued_error, slide_glue_signals
from milgrau.level2.kfs import fernald_inversion, kfs_inversion_monte_carlo
from milgrau.level2.lebear import process_level_2, process_single_level1_file
from milgrau.level2.molecular import (
    calculate_molecular_profile,
    find_optimal_reference_altitude,
)

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
