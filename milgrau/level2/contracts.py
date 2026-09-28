"""Small stable enums shared by Level 2 signal selection."""

from __future__ import annotations

from enum import IntEnum


class SignalSource(IntEnum):
    """Per-block source selected as the Level 2 retrieval input."""

    INVALID = 0
    GLUED = 1
    PHOTON_COUNTING = 2
    ANALOG = 3


class RetrievalInputInvalidReason(IntEnum):
    """Stable summary code for a rejected retrieval-input block."""

    VALID = 0
    NO_VALID_CHANNEL = 1
    NONFINITE_SIGNAL = 2
    INVALID_UNCERTAINTY = 3
    PHOTON_COUNTING_SATURATED = 4
    INSUFFICIENT_VERTICAL_COVERAGE = 5
    NONPOSITIVE_SIGNAL = 6
    LEVEL1_CORRECTION_FAILED_OR_UNCONFIRMED = 7
    SATURATION_DIAGNOSTIC_MISSING = 8
    SNR_UNAVAILABLE = 9
    SINGLE_CHANNEL_FALLBACK_DISABLED = 10


__all__ = ["RetrievalInputInvalidReason", "SignalSource"]
