"""Shared helpers for Level 0 processing."""

from __future__ import annotations

from statistics import StatisticsError, mode
from typing import Any, Mapping

import numpy as np


def safe_mode(values: Any) -> float:
    """Return the statistical mode with a median fallback."""
    try:
        return float(mode(values))
    except StatisticsError:
        return float(np.nanmedian(values))


def incremental_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether incremental processing is enabled."""
    return bool(config.get("processing", {}).get("incremental", False))
