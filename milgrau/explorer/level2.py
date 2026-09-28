"""Dependency-light Level 2 status helpers used by the optional Explorer UI."""

from __future__ import annotations

from typing import Any

from milgrau.level2.completeness import dataset_product_summary


def available_level2_wavelengths(dataset: Any) -> list[int]:
    """Expose only scientific wavelengths actually present in the product."""
    if "wavelength" not in dataset.coords:
        return []
    values: list[int] = []
    for wavelength in dataset["wavelength"].values:
        try:
            values.append(int(wavelength))
        except (TypeError, ValueError):
            continue
    return values


def level2_status_summary(dataset: Any) -> dict[str, object]:
    """Return completeness plus processed/failed diagnostics for UI rendering."""
    return dataset_product_summary(dataset)
