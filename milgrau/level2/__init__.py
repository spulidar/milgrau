"""Level 2 processing modules."""

from milgrau.level2.discovery import discover_level1_files
from milgrau.io.paths import LEVEL2_SUFFIX
from milgrau.level2.lebear import process_level_2, process_single_level1_file
from milgrau.level2.cloud_screening import cloud_screening_config, detect_anomalous_layer_mask, detect_reference_contamination
from milgrau.level2.contracts import WavelengthRetrievalResult
from milgrau.level2.gluing import propagate_glued_error, slide_glue_signals
from milgrau.level2.kfs import kfs_inversion_monte_carlo
from milgrau.level2.molecular import calculate_molecular_profile, find_optimal_reference_altitude

__all__ = [
    "LEVEL2_SUFFIX",
    "WavelengthRetrievalResult",
    "calculate_molecular_profile",
    "cloud_screening_config",
    "detect_anomalous_layer_mask",
    "detect_reference_contamination",
    "discover_level1_files",
    "find_optimal_reference_altitude",
    "kfs_inversion_monte_carlo",
    "process_level_2",
    "process_single_level1_file",
    "propagate_glued_error",
    "slide_glue_signals",
]
