"""Shared helpers and constants for Level 0 processing."""

from __future__ import annotations

from statistics import StatisticsError, mode
from typing import Any, Final, Mapping

import numpy as np

DEFAULT_LASER_SHOT_TOLERANCE_FRACTION: Final[float] = 2e-3
LICEL_HEADER_TIME_JITTER_S: Final[float] = 1.0


def safe_mode(values: Any) -> float:
    """Return the statistical mode with a median fallback."""
    try:
        return float(mode(values))
    except StatisticsError:
        return float(np.nanmedian(values))


def incremental_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether incremental processing is enabled."""
    return bool(config.get("processing", {}).get("incremental", False))
