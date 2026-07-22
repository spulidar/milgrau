"""Smoke tests for the refactored MILGRAU package imports."""

from __future__ import annotations

import subprocess
import sys


def test_core_package_imports() -> None:
    """The main MILGRAU subpackages should import without side effects."""
    import milgrau
    import milgrau.cli
    import milgrau.config
    import milgrau.io
    import milgrau.level0
    import milgrau.level1
    import milgrau.level2
    import milgrau.viz

    assert "cli" in milgrau.__all__
    assert "config" in milgrau.__all__
    assert "io" in milgrau.__all__
    assert "level0" in milgrau.__all__
    assert "level1" in milgrau.__all__
    assert "level2" in milgrau.__all__
    assert "viz" in milgrau.__all__


def test_pipeline_entrypoints_import() -> None:
    """Command-line entrypoint modules should import cleanly."""
    import milgrau.cli.explorer
    import milgrau.cli.lebear
    import milgrau.cli.libids
    import milgrau.cli.lipancora
    import milgrau.cli.liracos

    assert callable(milgrau.cli.explorer.main)
    assert callable(milgrau.cli.libids.main)
    assert callable(milgrau.cli.lipancora.main)
    assert callable(milgrau.cli.liracos.main)
    assert callable(milgrau.cli.lebear.main)


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
