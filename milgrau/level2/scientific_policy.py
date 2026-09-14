"""Approved productive scientific policy at the Level 2 boundary.

This module keeps the evidence-backed SPU retrieval choices explicit while the
legacy numerical implementation is progressively decomposed. The primary
elastic aerosol product uses a high-reference, backward Klett--Fernald
integration, consistent with operational network practice and with the
far-range molecular boundary used by the LPP SPU configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from milgrau.level2.config import Level2ConfigurationError


def get_kfs_mode_backward(config: Mapping[str, Any]) -> str:
    """Require backward Klett--Fernald for the productive elastic retrieval."""
    if not isinstance(config, Mapping):
        raise Level2ConfigurationError("MILGRAU configuration must be a mapping.")
    inversion = config.get("inversion")
    if not isinstance(inversion, Mapping):
        raise Level2ConfigurationError(
            "Configuration config.inversion must be a mapping."
        )
    if "kfs_mode" not in inversion:
        raise Level2ConfigurationError(
            "Missing required configuration: inversion.kfs_mode"
        )
    mode = str(inversion["kfs_mode"]).strip().lower()
    if mode != "backward":
        raise Level2ConfigurationError(
            "Level 2 primary elastic retrieval requires inversion.kfs_mode = "
            "'backward'. Forward/two-sided solutions may be diagnostic research "
            "products, but are not the productive aerosol contract."
        )
    return "backward"
