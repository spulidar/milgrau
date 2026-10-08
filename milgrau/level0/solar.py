"""Solar-regime annotation and contiguous scientific-segment helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from milgrau.level0.config import resolve_level0_config, station_coordinates
from milgrau.physics.solar import (
    SOLAR_POSITION_ALGORITHM,
    solar_elevation_deg,
    solar_regime,
)


def annotate_solar_context(group_df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Attach per-profile solar state and contiguous day/night segment IDs."""
    rows = group_df.copy()
    mask = rows["meas_type"] == "measurements"
    measurements = rows.loc[mask].sort_values("start_time_utc")
    if measurements.empty:
        return rows

    latitude, longitude = station_coordinates(config)
    threshold = resolve_level0_config(config).solar_regime.day_night_threshold_deg
    times = pd.to_datetime(measurements["start_time_utc"], utc=True)
    elevation = solar_elevation_deg(times, latitude, longitude)
    regimes = solar_regime(
        elevation,
        day_night_threshold_deg=threshold,
    )

    segment_numbers = np.zeros(len(measurements), dtype=np.int32)
    current = 0
    for index in range(1, len(regimes)):
        if str(regimes[index]) != str(regimes[index - 1]):
            current += 1
        segment_numbers[index] = current
    segment_ids = np.asarray(
        [f"seg{int(value):02d}" for value in segment_numbers],
        dtype=object,
    )

    rows.loc[measurements.index, "solar_elevation_deg"] = elevation
    rows.loc[measurements.index, "solar_regime"] = regimes
    rows.loc[measurements.index, "segment_id"] = segment_ids
    return rows


def solar_segment_table(measurement_rows: pd.DataFrame) -> pd.DataFrame:
    """Return one row per contiguous solar segment in chronological order."""
    if measurement_rows.empty:
        return pd.DataFrame(
            columns=[
                "segment_id",
                "solar_regime",
                "segment_start_utc",
                "segment_end_utc",
            ]
        )
    required = {"segment_id", "solar_regime", "start_time_utc", "stop_time"}
    missing = sorted(required - set(measurement_rows))
    if missing:
        raise KeyError(f"Solar segment table lacks required column(s): {missing}")

    rows = measurement_rows.sort_values("start_time_utc")
    records: list[dict[str, Any]] = []
    for segment_id, group in rows.groupby("segment_id", sort=False):
        regimes = [str(value) for value in group["solar_regime"].dropna().unique()]
        if len(regimes) != 1:
            raise ValueError(
                f"Solar segment {segment_id!r} must contain exactly one regime; got {regimes}."
            )
        records.append(
            {
                "segment_id": str(segment_id),
                "solar_regime": regimes[0],
                "segment_start_utc": pd.to_datetime(
                    group["start_time_utc"], utc=True
                ).min(),
                "segment_end_utc": pd.to_datetime(
                    group["stop_time"], utc=True
                ).max(),
            }
        )
    return pd.DataFrame.from_records(records)


def solar_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return immutable solar-classification provenance for product attrs."""
    threshold = resolve_level0_config(config).solar_regime.day_night_threshold_deg
    return {
        "solar_day_night_threshold_deg": float(threshold),
        "solar_position_algorithm": SOLAR_POSITION_ALGORITHM,
        "solar_elevation_reference": "geometric solar center; no atmospheric refraction",
    }


__all__ = [
    "annotate_solar_context",
    "solar_metadata",
    "solar_segment_table",
]
