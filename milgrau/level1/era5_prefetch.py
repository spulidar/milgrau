"""Global, opt-in-by-invocation ERA5 planning for selected L0 sessions.

Reads only Level 0 time coordinates and resolved station positions. The cache
is auxiliary: Level 1 thermodynamics still records actual hourly source types.
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
                if "time" not in ds.coords or ds.sizes.get("time", 0) <= 0:
                    raise ValueError("missing or empty Level 0 time coordinate")
                source_times = pd.to_datetime(ds["time"].values, utc=True)
                first = pd.Timestamp(source_times.min()).floor("h")
                last = pd.Timestamp(source_times.max()).ceil("h")
                site = resolve_station_site(config, ds)
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
    return failures == 0
