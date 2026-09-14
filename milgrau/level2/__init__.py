"""Level 2 processing modules.

Package import is intentionally declarative: scientific behavior is defined by
canonical modules and is never installed through import-time monkey patching.
"""

from milgrau.io.paths import LEVEL2_SUFFIX
from milgrau.level2.cloud_screening import detect_cloud_layers, mask_cloud_layers
from milgrau.level2.completeness import (
    Level2ProductContract,
    ProductCompleteness,
    ProductStatus,
    WavelengthAttempt,
    WavelengthAttemptStatus,
    WavelengthFailureCode,
    WavelengthFailureDiagnostic,
    WavelengthFailureStage,
)
from milgrau.level2.contracts import (
    GluedSignals,
    GluingDiagnostics,
    KfsDiagnostics,
    MolecularProfiles,
    OpticalProducts,
    RayleighDiagnostics,
    RetrievalInputInvalidReason,
    SignalSelectionDiagnostics,
    SignalSource,
    WavelengthRetrievalResult,
)
from milgrau.level2.discovery import discover_level1_files
from milgrau.level2.gluing import (
    check_gluing_region,
    find_gluing_region,
    propagate_glued_error,
    slide_glue_signals,
)
from milgrau.level2.kfs import fernald_inversion, kfs_inversion_monte_carlo
from milgrau.level2.lebear import process_level_2, process_single_level1_file
from milgrau.level2.molecular import (
    calculate_molecular_profile,
    calculate_simulated_molecular_signal,
    find_optimal_reference_altitude,
    linear_rayleigh_calibration_factor,
)

__all__ = [
    "LEVEL2_SUFFIX",
    "GluedSignals",
    "GluingDiagnostics",
    "KfsDiagnostics",
    "Level2ProductContract",
    "MolecularProfiles",
    "OpticalProducts",
    "ProductCompleteness",
    "ProductStatus",
    "RayleighDiagnostics",
    "RetrievalInputInvalidReason",
    "SignalSelectionDiagnostics",
    "SignalSource",
    "WavelengthAttempt",
    "WavelengthAttemptStatus",
    "WavelengthFailureCode",
    "WavelengthFailureDiagnostic",
    "WavelengthFailureStage",
    "WavelengthRetrievalResult",
    "calculate_molecular_profile",
    "calculate_simulated_molecular_signal",
    "check_gluing_region",
    "detect_cloud_layers",
    "discover_level1_files",
    "fernald_inversion",
    "find_gluing_region",
    "find_optimal_reference_altitude",
    "kfs_inversion_monte_carlo",
    "linear_rayleigh_calibration_factor",
    "mask_cloud_layers",
    "process_level_2",
    "process_single_level1_file",
    "propagate_glued_error",
    "slide_glue_signals",
]
