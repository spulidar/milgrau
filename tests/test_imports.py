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
