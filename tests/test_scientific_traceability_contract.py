"""Keep human scientific traceability aligned with canonical code identities."""

from __future__ import annotations

from pathlib import Path

from milgrau.scientific import (
    LEVEL2_PRODUCT_SCHEMA_VERSION,
    LEVEL2_RETRIEVAL_METHOD_VERSION,
)


def test_scientific_traceability_names_current_schema_and_method() -> None:
    text = Path("docs/scientific_traceability.md").read_text(encoding="utf-8")

    assert f"product schema **v{LEVEL2_PRODUCT_SCHEMA_VERSION}**" in text
    assert f"retrieval method **v{LEVEL2_RETRIEVAL_METHOD_VERSION}**" in text


def test_scientific_traceability_does_not_restore_known_method_v3_drift() -> None:
    text = Path("docs/scientific_traceability.md").read_text(encoding="utf-8")

    assert "Level 2 retrieval method version 3" not in text
    assert "candidate-catalogue redesign remains pending" not in text
