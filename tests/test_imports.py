"""Smoke tests for the refactored MILGRAU package imports."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys


def test_core_package_imports() -> None:
    """The root package stays light while explicit subpackages import cleanly."""
    import milgrau

    for module_name in (
        "milgrau.cli",
        "milgrau.config",
        "milgrau.io",
        "milgrau.level0",
        "milgrau.level1",
        "milgrau.level2",
        "milgrau.physics",
        "milgrau.viz",
    ):
        assert importlib.import_module(module_name) is not None

    assert milgrau.__all__ == ["__version__"]
    assert isinstance(milgrau.__version__, str)


def test_subpackage_public_surfaces_are_explicit_and_stable() -> None:
    """Supported convenience/research APIs must change only by deliberate review."""
    expected = {
        "milgrau.io": {
            "ensure_directories",
            "fetch_era5_pressure_level_profile",
            "fetch_surface_weather",
            "fetch_wyoming_radiosonde",
            "level0_output_path",
            "level1_output_path",
            "level2_output_path",
            "log_output_root",
            "measurement_save_id",
            "parse_licel_group",
            "processed_data_root",
            "radiosonde_cache_dir",
            "read_licel_header",
            "raw_data_root",
            "scan_raw_files",
            "setup_logger",
            "surface_weather_cache_dir",
            "validate_level0_contract",
            "validate_level1_contract",
            "validate_level2_contract",
        },
        "milgrau.level0": {
            "build_level0_netcdf",
            "build_measurement_inventory",
            "classify_period",
            "filter_laser_shots",
            "measurement_id_for_local_time",
            "period_start_local",
            "period_utc_label",
            "process_level_0",
            "validate_lidar_tensors",
        },
        "milgrau.level1": {
            "apply_all_physical_corrections",
            "apply_instrumental_corrections",
            "calculate_pbl_height_gradient",
            "calculate_tropopause_heights",
            "estimate_pbl_timeseries",
            "integrate_thermodynamics",
            "load_and_prepare_level0",
            "process_level_1",
            "process_single_file",
        },
        "milgrau.level2": {
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
        },
        "milgrau.physics": {
            "geometric_to_geopotential_altitude",
            "get_standard_atmosphere",
        },
        "milgrau.viz": {
            "plot_all_level2_qa",
            "plot_global_mean_rcs",
            "plot_quicklook",
        },
    }

    for module_name, expected_names in expected.items():
        module = importlib.import_module(module_name)
        assert set(module.__all__) == expected_names
        assert all(hasattr(module, name) for name in expected_names)


def test_level1_public_exports_use_canonical_owners() -> None:
    """Package-level Level 1 helpers must come directly from their owner modules."""
    import milgrau.level1 as level1
    import milgrau.level1.ingestion as ingestion
    import milgrau.level1.pbl as pbl
    import milgrau.level1.thermodynamics as thermodynamics

    assert level1.load_and_prepare_level0 is ingestion.load_and_prepare_level0
    assert level1.estimate_pbl_timeseries is pbl.estimate_pbl_timeseries
    assert level1.calculate_pbl_height_gradient is pbl.calculate_pbl_height_gradient
    assert level1.integrate_thermodynamics is thermodynamics.integrate_thermodynamics


def test_level1_common_has_no_obsolete_compatibility_helpers() -> None:
    """Removed compatibility wrappers must not silently return."""
    import milgrau.level1.common as common

    assert not hasattr(common, "get_channel_constant")
    assert not hasattr(common, "level1_output_path")


def test_level2_productive_orchestration_has_no_legacy_module_dependency() -> None:
    """Canonical retrieval functions must come from their cohesive owner modules."""
    import milgrau.level2.optical_retrieval as optical_retrieval
    import milgrau.level2.retrieval as retrieval
    import milgrau.level2.signal_selection as signal_selection

    assert importlib.util.find_spec("milgrau.level2._retrieval_impl") is None
    assert importlib.util.find_spec("milgrau.level2.atmosphere") is None
    assert retrieval.prepare_wavelength_blocks is signal_selection.prepare_wavelength_blocks
    assert retrieval.glue_signal_blocks is signal_selection.glue_signal_blocks
    assert retrieval.evaluate_rayleigh_reference is optical_retrieval.evaluate_rayleigh_reference
    assert retrieval.retrieve_optical_blocks is optical_retrieval.retrieve_optical_blocks


def test_level2_public_cloud_screening_api_uses_existing_symbols() -> None:
    """Package re-exports must refer to the real cloud-screening API."""
    import milgrau.level2 as level2

    expected = {
        "cloud_screening_config",
        "detect_anomalous_layer_mask",
        "detect_reference_contamination",
    }
    assert expected <= set(level2.__all__)
    for name in expected:
        assert callable(getattr(level2, name))

    assert "detect_cloud_layers" not in level2.__all__
    assert "mask_cloud_layers" not in level2.__all__


def test_pipeline_entrypoints_import() -> None:
    """Command-line entrypoint modules should import cleanly."""
    modules = [
        importlib.import_module("milgrau.cli.explorer"),
        importlib.import_module("milgrau.cli.lebear"),
        importlib.import_module("milgrau.cli.libids"),
        importlib.import_module("milgrau.cli.lipancora"),
        importlib.import_module("milgrau.cli.liracos"),
    ]

    assert all(callable(module.main) for module in modules)


def test_level2_core_import_does_not_require_matplotlib() -> None:
    """The Level 2 retrieval/orchestration core should import with plotting blocked."""
    script = """
import importlib.abc
import sys

class BlockMatplotlib(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'matplotlib' or fullname.startswith('matplotlib.'):
            raise ModuleNotFoundError('matplotlib intentionally blocked')
        return None

sys.meta_path.insert(0, BlockMatplotlib())
import milgrau.level2.lebear
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
