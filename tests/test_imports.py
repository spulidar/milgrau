"""Smoke tests for the refactored MILGRAU package imports."""

from __future__ import annotations


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
