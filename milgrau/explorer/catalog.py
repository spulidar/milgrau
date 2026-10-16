"""Fast, read-only MILGRAU session and variant discovery."""
from __future__ import annotations
import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any
import numpy as np
import pandas as pd
import xarray as xr
from milgrau.config.loader import load_config
from milgrau.io.paths import session_id_parts, processed_data_root, product_session_id

LEVEL_TO_PATH = {"Level 0": "level0_path", "Level 1": "level1_path", "Level 2": "level2_path"}

def safe_key(*parts: Any) -> str:
    text = "_".join(str(part) for part in parts if part is not None)
    return re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_") or "widget"


def processed_root_from_config(config_path: str) -> Path:
    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    return processed_data_root(config, root_dir=config_file.parent)


def station_timezone_from_config(config_path: str) -> str:
    """Return station IANA timezone for human session presentation."""
    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    catalog = config.get("_station_catalog", {})
    station = catalog.get("station", {}) if isinstance(catalog, dict) else {}
    value = str(station.get("timezone", "UTC")).strip()
    ZoneInfo(value)
    return value


def session_display(
    session_id: str,
    timezone_name: str,
) -> dict[str, Any]:
    """Return UTC/local interval, duration and a compact human session label."""
    try:
        station, start_utc, end_utc = session_id_parts(session_id)
        timezone = ZoneInfo(str(timezone_name))
        start_local = start_utc.astimezone(timezone)
        end_local = end_utc.astimezone(timezone)
        duration_seconds = float((end_utc - start_utc).total_seconds())
        total_minutes = int(round(duration_seconds / 60.0))
        hours, minutes = divmod(total_minutes, 60)
        duration = f"{hours}h{minutes:02d}" if hours else f"{minutes}min"
        interval_utc = (
            f"{start_utc.strftime('%Y-%m-%d %H:%M')} → "
            f"{end_utc.strftime('%Y-%m-%d %H:%M')} UTC"
        )
        interval_local = (
            f"{start_local.strftime('%d/%m/%Y %H:%M')} → "
            f"{end_local.strftime('%d/%m/%Y %H:%M')}"
        )
        human_label = (
            f"{station.upper()} · {start_local.strftime('%d/%m %H:%M')} → "
            f"{end_local.strftime('%d/%m %H:%M')} · {duration}"
        )
        return {
            "date": start_local.date(),
            "interval_utc": interval_utc,
            "interval_local": interval_local,
            "duration": duration,
            "duration_seconds": duration_seconds,
            "human_label": human_label,
        }
    except (TypeError, ValueError, KeyError):
        return {
            "date": None,
            "interval_utc": "--",
            "interval_local": "--",
            "duration": "--",
            "duration_seconds": np.nan,
            "human_label": str(session_id),
        }


def parse_session_id(session_id: str) -> tuple[date | None, str]:
    """Compatibility helper for small internal callers; display uses station local time."""
    info = session_display(session_id, "UTC")
    return info["date"], info["interval_utc"]


def product_paths(session_dir: Path, session_id: str) -> dict[str, Any]:
    """Discover canonical and derived NetCDF products without opening arrays."""
    result: dict[str, Any] = {}
    groups: dict[str, list[str]] = {"level0": [], "level1": [], "level2": []}
    for path in sorted(session_dir.glob("*.nc")):
        if not path.is_file():
            continue
        try:
            if product_session_id(path) != session_id:
                continue
        except ValueError:
            continue
        match = re.search(r"_L([012])(?:_scc)?\.nc$", path.name, re.IGNORECASE)
        if match:
            groups[f"level{match.group(1)}"].append(str(path))
    for number in range(3):
        name = f"level{number}"
        available = groups[name]
        canonical = str(session_dir / f"{session_id}_L{number}.nc")
        preferred = canonical if canonical in available else (available[0] if available else "")
        result[f"{name}_path"] = preferred
        result[f"{name}_products"] = available
    return result


def _highest_available_level(paths: dict[str, str]) -> str:
    for level in ("Level 2", "Level 1", "Level 0"):
        if paths[LEVEL_TO_PATH[level]]:
            return level
    return "--"


def _session_context_from_products(
    session_dir: Path,
    paths: dict[str, str],
) -> dict[str, Any]:
    """Read only compact session metadata from the highest available product."""
    level = _highest_available_level(paths)
    # The catalogue must never read the large matrices of every archived session.
    # Solar/segment metadata is inspected only for the selected session.
    regimes = []
    for level_number in range(3):
        for product in paths.get(f"level{level_number}_products", []):
            stem = Path(product).stem.lower()
            for regime in ("day", "night"):
                if re.search(rf"(?:^|_){regime}(?:_|$)", stem) and regime not in regimes:
                    regimes.append(regime)
    segments: list[str] = []
    figures_dir = session_dir / "figures"
    figure_paths = (
        sorted(path for path in figures_dir.iterdir() if path.is_file())
        if figures_dir.is_dir()
        else []
    )
    return {
        "highest_available_level": level,
        "solar_regimes": ", ".join(regimes) if regimes else "--",
        "segments": ", ".join(segments) if segments else "--",
        "figure_count": len(figure_paths),
        "figures": [path.name for path in figure_paths],
    }


def discover_products(processed_root: str, timezone_name: str = "UTC") -> pd.DataFrame:
    root = Path(processed_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame()

    session_dirs: list[Path] = []
    for path in sorted(root.glob("*/*/*/*")):
        if not path.is_dir():
            continue
        try:
            session_id_parts(path.name)
        except ValueError:
            continue
        session_dirs.append(path)

    for session_dir in session_dirs:
        session_id = session_dir.name
        paths = product_paths(session_dir, session_id)
        if not any(paths.values()):
            continue
        display = session_display(session_id, timezone_name)
        context = _session_context_from_products(session_dir, paths)
        day = display["date"]
        mtimes = [Path(paths[key]).stat().st_mtime for key in LEVEL_TO_PATH.values() if paths[key]]
        rows.append(
            {
                "date": day,
                "year": day.year if day else None,
                "month": day.month if day else None,
                "day": day.day if day else None,
                "session_id": session_id,
                "human_label": display["human_label"],
                "interval_local": display["interval_local"],
                "interval_utc": display["interval_utc"],
                "duration": display["duration"],
                "product_dir": str(session_dir),
                **paths,
                **context,
                "available_levels": ", ".join(
                    label for label, key in LEVEL_TO_PATH.items() if paths[key]
                ),
                "modified": (
                    datetime.fromtimestamp(max(mtimes)).isoformat(timespec="seconds")
                    if mtimes
                    else ""
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["date", "session_id"], na_position="last")
        .reset_index(drop=True)
    )

def selected_session_context(path: str) -> dict[str, str]:
    """Read only tiny solar/segment table for a single selected product."""
    result = {"solar_regimes": "--", "segments": "--"}
    if not path:
        return result
    try:
        with xr.open_dataset(path, decode_times=False, mask_and_scale=False, cache=False) as ds:
            if "Segment_Label" in ds and "Segment_Regime" in ds:
                labels = np.asarray(ds["Segment_Label"].values).astype(str).reshape(-1)
                regimes = np.asarray(ds["Segment_Regime"].values).astype(str).reshape(-1)
                result["segments"] = ", ".join(
                    f"{label} {regime}" for label, regime in zip(labels, regimes, strict=False)
                ) or "--"
                result["solar_regimes"] = ", ".join(dict.fromkeys(regimes)) or "--"
    except (OSError, ValueError, KeyError):
        pass
    return result
