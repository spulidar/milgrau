"""Offline retrieval of one sounding from a cached NOAA/NCEI IGRA2 text archive.

IGRA2 source files use fixed-column ASCII records (not CSV). Only the
requested UTC sounding is parsed; an in-process offset map avoids rescanning
multi-decade archives for every Level 1 atmosphere hour.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd


@lru_cache(maxsize=3)
def _archive_index(
    filename: str, size: int, mtime_ns: int,
) -> dict[datetime, tuple[int, int]]:
    """Map UTC nominal sounding times to (byte offset, level count)."""
    del size, mtime_ns  # Fingerprint used as part of the cache key.
    index: dict[datetime, tuple[int, int]] = {}
    expected_station = Path(filename).name.removesuffix("-data.txt")
    with open(filename, "rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.startswith(b"#") or len(line) < 36:
                continue
            if line[1:12].decode("ascii", errors="replace") != expected_station:
                continue
            try:
                date = datetime(
                    int(line[13:17]), int(line[18:20]),
                    int(line[21:23]), int(line[24:26]),
                    tzinfo=timezone.utc,
                )
                nlevels = int(line[32:36])
            except ValueError:
                continue
            if not 0 < nlevels <= 10000:
                continue
            # If duplicate nominal times exist, prefer the most recent block.
            index[date] = (offset, nlevels)
    return index


def _parse_sounding(
    archive: Path, offset: int, nlevels: int, expected_datetime: datetime,
) -> pd.DataFrame:
    """Decode IGRA2 PRESS (Pa), GPH (m ASL) and TEMP (0.1 deg C).

    -8888 is data removed by IGRA quality assurance, -9999 is missing.
    Both are excluded, as are rows without coincident height/temperature/
    pressure. Other meteorological variables are not needed for L1.
    """
    rows: list[tuple[float, float, float]] = []
    with archive.open("rb") as handle:
        handle.seek(offset)
        header = handle.readline()
        actual_datetime = datetime(
            int(header[13:17]), int(header[18:20]),
            int(header[21:23]), int(header[24:26]),
            tzinfo=timezone.utc,
        )
        if actual_datetime != expected_datetime:
            raise ValueError("IGRA2 archive was modified since the index was built.")
        for _ in range(nlevels):
            line = handle.readline()
            if len(line) < 28 or line.startswith(b"#"):
                raise ValueError("Incomplete IGRA2 sounding level records.")
            try:
                pressure_pa = int(line[9:15])
                gph_m = int(line[16:21])
                temp_tenths_c = int(line[22:27])
            except ValueError:
                continue
            if any(value in {-8888, -9999} for value in (
                pressure_pa, gph_m, temp_tenths_c,
            )):
                continue
            pressure_hpa = pressure_pa / 100.0
            temperature_c = temp_tenths_c / 10.0
            if not (
                5.0 <= pressure_hpa <= 1100.0
                and -1000.0 <= gph_m <= 60000.0
                and -110.0 <= temperature_c <= 60.0
            ):
                continue
            rows.append((float(gph_m), temperature_c, pressure_hpa))

    df = (
        pd.DataFrame(rows, columns=["height", "temperature", "pressure"])
        .drop_duplicates(subset=["height"], keep="first")
        .sort_values("height")
        .reset_index(drop=True)
    )
    if len(df) < 2 or df["height"].iloc[-1] <= df["height"].iloc[0]:
        raise ValueError("IGRA2 sounding lacks a usable vertical temperature/pressure profile.")
    return df


def load_igra2_cached_sounding(
    cache_root: Path,
    station_id: str,
    target_datetime_utc: datetime,
    logger: logging.Logger,
) -> pd.DataFrame | None:
    """Return local IGRA2 profile, or None for absent/unusable data.

    The SPU Wyoming WMO ID 83779 maps to IGRA2 station BRM00083779.
    Never use a sounding from a different station or nominal UTC time.
    """
    station = str(station_id).strip()
    if not station.isdigit() or len(station) != 5:
        return None
    igra_station = f"BRM000{station}"
    archive = cache_root / f"{igra_station}-data.txt"
    if not archive.is_file():
        return None
    try:
        stat = archive.stat()
        index = _archive_index(
            str(archive.resolve()), stat.st_size, stat.st_mtime_ns,
        )
        entry = index.get(target_datetime_utc)
        if entry is None:
            logger.debug(
                "IGRA2 historical cache has no sounding | station=%s | %s",
                igra_station, target_datetime_utc.isoformat(),
            )
            return None
        df = _parse_sounding(archive, *entry, target_datetime_utc)
    except (OSError, ValueError, UnicodeError) as exc:
        logger.warning(
            "IGRA2 cached profile unusable | %s | %s",
            target_datetime_utc.isoformat(), exc,
        )
        return None

    logger.info(
        "radiosonde IGRA2 cache hit | %s | station=%s | %d valid levels",
        target_datetime_utc.strftime("%Y-%m-%d %H:%MZ"),
        igra_station, len(df),
    )
    df.attrs.update({
        "source": "NOAA NCEI IGRA2 historical sounding archive",
        "source_type": "radiosonde",
        "historical_archive": archive.name,
        "historical_station_id": igra_station,
        "target_datetime_utc": target_datetime_utc.isoformat(),
    })
    return df
