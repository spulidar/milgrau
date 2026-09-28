"""Keep human scientific traceability aligned with the current retrieval."""

from pathlib import Path


def test_scientific_traceability_names_current_decisions() -> None:
    text = Path("docs/scientific_traceability.md").read_text(encoding="utf-8")

    assert "Two-sided Klett–Fernald–Sasano" in text
    assert "Primary reference range 10–15 km; fallback 5–20 km" in text
    assert "150 configurable selection-aware Monte Carlo" in text
    assert "Routine nominal boundary `f=0`" in text


def test_scientific_traceability_exposes_one_product() -> None:
    text = Path("docs/scientific_traceability.md").read_text(encoding="utf-8")

    assert "one current\nLevel 2 method" in text
    assert "internal retrieval revision" not in text
    assert "method-v5" not in text
    assert "method v4" not in text
