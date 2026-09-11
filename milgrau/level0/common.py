"""Shared helpers for Level 0 processing."""

from __future__ import annotations

from statistics import StatisticsError, mode
from typing import Any, Mapping

import numpy as np


def safe_mode(values: Any) -> float:
    """Return the statistical mode with a median fallback for data statistics."""
    try:
        return float(mode(values))
    except StatisticsError:
        return float(np.nanmedian(values))


def incremental_enabled(config: Mapping[str, Any]) -> bool:
    """Return the explicitly configured incremental-processing policy."""
    processing = config.get("processing")
    if not isinstance(processing, Mapping):
        raise KeyError("Configuration processing section is required.")
    if "incremental" not in processing:
        raise KeyError("Missing required configuration: processing.incremental")
    value = processing["incremental"]
    if not isinstance(value, bool):
        raise ValueError("Configuration processing.incremental must be a boolean.")
    return value
