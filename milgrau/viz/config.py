"""Strict configuration accessors for MILGRAU visualization products."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping

import math


class VisualizationConfigurationError(ValueError):
    """Raised when a productive visualization setting is missing or invalid."""


@dataclass(frozen=True, slots=True)
class QuicklookConfig:
    show_pbl: bool
    show_tropopause: bool
    mean_profile_smooth_bins: int
    max_time_gap_minutes: float
    missing_data_color: str
    colormap: str


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    output_format: str
    dpi: int
    altitude_ranges_km: tuple[float, ...]
    channels_to_plot: tuple[str, ...]
    quicklook: QuicklookConfig


def _mapping(parent: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    if key not in parent:
        raise VisualizationConfigurationError(f"Missing required configuration: {path}.{key}")
    value = parent[key]
    if not isinstance(value, Mapping):
        raise VisualizationConfigurationError(f"Configuration {path}.{key} must be a mapping.")
    return value


def _required(parent: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in parent:
        raise VisualizationConfigurationError(f"Missing required configuration: {path}.{key}")
    return parent[key]


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VisualizationConfigurationError(f"Configuration {path} must be a non-empty string.")
    return value.strip()


def _positive_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise VisualizationConfigurationError(f"Configuration {path} must be a positive finite number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise VisualizationConfigurationError(f"Configuration {path} must be a positive finite number.")
    return result


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise VisualizationConfigurationError(f"Configuration {path} must be a positive integer.")
    return int(value)


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise VisualizationConfigurationError(f"Configuration {path} must be a boolean.")
    return value


def resolve_visualization_config(config: Mapping[str, Any]) -> VisualizationConfig:
    """Resolve operator-facing visualization settings without semantic defaults."""
    if not isinstance(config, Mapping):
        raise VisualizationConfigurationError("MILGRAU configuration must be a mapping.")
    viz = _mapping(config, "visualization", "config")

    output_format = _text(_required(viz, "output_format", "visualization"), "visualization.output_format")
    output_format = output_format.lstrip(".").lower()
    if not output_format:
        raise VisualizationConfigurationError("Configuration visualization.output_format must contain a file extension.")
    dpi = _positive_int(_required(viz, "dpi", "visualization"), "visualization.dpi")

    raw_ranges = _required(viz, "altitude_ranges_km", "visualization")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise VisualizationConfigurationError("Configuration visualization.altitude_ranges_km must be a non-empty list.")
    altitude_ranges: list[float] = []
    for index, value in enumerate(raw_ranges):
        altitude = _positive_float(value, f"visualization.altitude_ranges_km[{index}]")
        if altitude in altitude_ranges:
            raise VisualizationConfigurationError(
                f"Configuration visualization.altitude_ranges_km contains duplicate value {altitude}."
            )
        altitude_ranges.append(altitude)

    raw_channels = _required(viz, "channels_to_plot", "visualization")
    if not isinstance(raw_channels, list) or not raw_channels:
        raise VisualizationConfigurationError("Configuration visualization.channels_to_plot must be a non-empty list.")
    channels: list[str] = []
    for index, value in enumerate(raw_channels):
        channel = _text(value, f"visualization.channels_to_plot[{index}]")
        if channel in channels:
            raise VisualizationConfigurationError(
                f"Configuration visualization.channels_to_plot contains duplicate channel {channel!r}."
            )
        channels.append(channel)

    quicklook = _mapping(viz, "quicklook", "visualization")
    quicklook_config = QuicklookConfig(
        show_pbl=_boolean(_required(quicklook, "show_pbl", "visualization.quicklook"), "visualization.quicklook.show_pbl"),
        show_tropopause=_boolean(
            _required(quicklook, "show_tropopause", "visualization.quicklook"),
            "visualization.quicklook.show_tropopause",
        ),
        mean_profile_smooth_bins=_positive_int(
            _required(quicklook, "mean_profile_smooth_bins", "visualization.quicklook"),
            "visualization.quicklook.mean_profile_smooth_bins",
        ),
        max_time_gap_minutes=_positive_float(
            _required(quicklook, "max_time_gap_minutes", "visualization.quicklook"),
            "visualization.quicklook.max_time_gap_minutes",
        ),
        missing_data_color=_text(
            _required(quicklook, "missing_data_color", "visualization.quicklook"),
            "visualization.quicklook.missing_data_color",
        ),
        colormap=_text(
            _required(quicklook, "colormap", "visualization.quicklook"),
            "visualization.quicklook.colormap",
        ),
    )
    return VisualizationConfig(
        output_format=output_format,
        dpi=dpi,
        altitude_ranges_km=tuple(altitude_ranges),
        channels_to_plot=tuple(channels),
        quicklook=quicklook_config,
    )
