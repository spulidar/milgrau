"""Compact helpers for Level 2 wavelength completeness."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def canonical_wavelengths(values: Iterable[int]) -> tuple[int, ...]:
    """Return positive wavelengths in deterministic order without duplicates."""
    normalized: set[int] = set()
    for value in values:
        if isinstance(value, bool):
            raise TypeError("Wavelengths must be positive integers, not booleans.")
        wavelength = int(value)
        if wavelength <= 0:
            raise ValueError("Wavelengths must be positive integers.")
        normalized.add(wavelength)
    if not normalized:
        raise ValueError("At least one requested wavelength is required.")
    return tuple(sorted(normalized))


def dataset_product_summary(dataset: Any) -> dict[str, object]:
    """Read compact completeness information from a Level 2 dataset."""

    def integer_list(variable_name: str, fallback: tuple[int, ...] = ()) -> list[int]:
        if variable_name not in dataset:
            return list(fallback)
        return [int(value) for value in dataset[variable_name].values.tolist()]

    scientific = (
        tuple(int(value) for value in dataset["wavelength"].values.tolist())
        if "wavelength" in dataset.coords
        else ()
    )
    return {
        "product_completeness": str(
            dataset.attrs.get("product_completeness", "unknown")
        ),
        "product_status": str(dataset.attrs.get("product_status", "unknown")),
        "requested_wavelengths": integer_list("requested_wavelengths", scientific),
        "processed_wavelengths": integer_list("processed_wavelengths", scientific),
        "failed_wavelengths": integer_list("failed_wavelengths"),
    }


def format_dataset_product_summary(dataset: Any) -> tuple[str, ...]:
    """Return short completeness lines suitable for QA and Explorer views."""
    summary = dataset_product_summary(dataset)

    def joined(name: str) -> str:
        values = summary[name]
        return ", ".join(str(value) for value in values) or "none"  # type: ignore[union-attr]

    return (
        f"product_completeness: {summary['product_completeness']}",
        f"product_status: {summary['product_status']}",
        f"requested_wavelengths_nm: {joined('requested_wavelengths')}",
        f"processed_wavelengths_nm: {joined('processed_wavelengths')}",
        f"failed_wavelengths_nm: {joined('failed_wavelengths')}",
    )


__all__ = [
    "canonical_wavelengths",
    "dataset_product_summary",
    "format_dataset_product_summary",
]
