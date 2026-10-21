"""Global, opt-in-by-invocation ERA5 planning for selected L0 sessions.

Reads the canonical Level 0 Raw_Data_Start_Time acquisition axis and resolved
station positions. Raw Level 0 has a `time` *dimension*, not necessarily a
datetime `time` coordinate; LIPANCORA constructs that coordinate at ingestion.
The cache is auxiliary: L1 records the actual hourly atmospheric sources.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import xarray as xr

from milgrau.io.era5 import prefetch_era5_pressure_level_profiles
from milgrau.io.logging_utils import bind_log_context
from milgrau.level1.config import resolve_level1_config, resolve_station_site
from milgrau.level1.ingestion import _decode_level0_time_axis


def prefetch_selected_level0_atmosphere(
    paths: Sequence[Path],
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> bool:
    """Collect deduplicated ERA5 hours across all selected, pending L0 files.

    True signals that global network planning was attempted; downstream Level 1
    must not independently resubmit missed hours during the same run.
    """
    # Explicit-file/embedding callers may provide a minimal mocked config;
    # the real Level 1 pipeline performs strict validation independently.
    if not isinstance(config.get("level1"), Mapping):
        return False
    atmosphere = resolve_level1_config(config).atmosphere
    if "era5" not in atmosphere.source_priority or atmosphere.era5 is None:
        return False
    required: dict[tuple[float, float], set[pd.Timestamp]] = defaultdict(set)
    failures = 0
    for path in paths:
        try:
            with xr.open_dataset(path) as ds:
                if ds.sizes.get("time", 0) <= 0:
                    raise ValueError("empty Level 0 acquisition time dimension")
                # Reuse precisely the time decoding of the scientific L1
                # ingestion step; the native L0 has Raw_Data_Start_Time offsets.
                decoded_times = _decode_level0_time_axis(ds)
                source_times = pd.to_datetime(decoded_times, utc=True)
                if source_times.empty or source_times.isna().any():
                    raise ValueError("invalid Level 0 Raw_Data_Start_Time axis")
                first = pd.Timestamp(source_times.min()).floor("h")
                last = pd.Timestamp(source_times.max()).ceil("h")
                # Profiles without stored Station_Profile require the real
                # acquisition datetimes when resolving historical site coordinates.
                site = resolve_station_site(config, ds.assign_coords(time=decoded_times))
                key = (float(site["latitude"]), float(site["longitude"]))
                required[key].update(pd.date_range(first, last, freq="1h", tz="UTC"))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            bind_log_context(logger, stage="era5").warning(
                "ERA5 prefetch cannot inspect L0 %s | %s", path.name, exc,
            )
            failures += 1

    if not required:
        return False

    total_hours = sum(len(hours) for hours in required.values())
    bind_log_context(logger, stage="era5").info(
        "global ERA5 plan | %d L0 files | %d unique site-hours | %d site(s)",
        len(paths), total_hours, len(required),
    )
    for (lat, lon), hours in sorted(required.items()):
        prefetch_era5_pressure_level_profiles(
            [ts.to_pydatetime() for ts in sorted(hours)],
            lat, lon,
            bind_log_context(logger, stage="era5"),
            settings=atmosphere.era5.as_io_mapping(),
        )
    if failures:
        bind_log_context(logger, stage="era5").warning(
            "ERA5 planning skipped %d unreadable L0 file(s); "
            "their scientific processing will report errors independently",
            failures,
        )
    # Even with some malformed L0 inputs, successfully inspected sessions have
    # already been prefetched. Prevent per-session CDS resubmission for them.
    return True
