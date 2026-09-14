"""Architecture guardrails that should stay scientifically neutral."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "milgrau"
PUBLIC_PACKAGES = (
    "milgrau",
    "milgrau.config",
    "milgrau.io",
    "milgrau.level0",
    "milgrau.level1",
    "milgrau.level2",
    "milgrau.physics",
    "milgrau.viz",
)
LEVEL2_PRODUCTIVE_SCIENCE = (
    PACKAGE_ROOT / "level2" / "retrieval.py",
    PACKAGE_ROOT / "level2" / "signal_selection.py",
    PACKAGE_ROOT / "level2" / "retrieval_input_qa.py",
    PACKAGE_ROOT / "level2" / "optical_retrieval.py",
)
SEMANTIC_MAPPING_NAMES = {
    "config",
    "inv_cfg",
    "fit_config",
    "gluing_config",
    "molecular_fit_config",
    "cloud_cfg",
}


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_public_all_entries_are_real_bound_symbols() -> None:
    """Every advertised package export must resolve after importing that package."""
    for module_name in PUBLIC_PACKAGES:
        module = importlib.import_module(module_name)
        for name in getattr(module, "__all__", ()):  # pragma: no branch - tiny public lists
            assert hasattr(module, name), f"{module_name}.__all__ advertises missing symbol {name!r}"


def test_productive_package_has_no_wildcard_imports() -> None:
    """Wildcard imports obscure ownership and can reintroduce import-order behavior."""
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
                offenders.append(f"{path.relative_to(PACKAGE_ROOT.parent)}:{node.lineno}")
    assert not offenders, "Wildcard imports are not allowed in milgrau: " + ", ".join(offenders)


def test_level2_productive_science_has_no_mapping_get_fallbacks() -> None:
    """Scientific settings in productive L2 paths must come from strict resolvers.

    This intentionally targets mappings whose names denote scientific/configuration
    state. Diagnostic dictionaries and non-scientific metadata may still use
    ``dict.get`` where absence is genuinely optional.
    """
    offenders: list[str] = []
    for path in LEVEL2_PRODUCTIVE_SCIENCE:
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            function = node.func
            if not isinstance(function, ast.Attribute) or function.attr != "get":
                continue
            owner = function.value
            if isinstance(owner, ast.Name) and owner.id in SEMANTIC_MAPPING_NAMES:
                offenders.append(
                    f"{path.relative_to(PACKAGE_ROOT.parent)}:{node.lineno} uses {owner.id}.get(..., fallback)"
                )
    assert not offenders, (
        "Productive Level 2 science must not hide missing configuration behind local defaults: "
        + ", ".join(offenders)
    )
