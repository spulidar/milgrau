"""Read-only discovery of continuous MILGRAU scientific sessions."""
from __future__ import annotations

import re
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from milgrau.config.loader import load_config
from milgrau.io.paths import (
    processed_data_root, product_session_id, session_dir, session_id_parts,
    station_id, validate_session_id_for_config,
)

_PRODUCT = re.compile(r"_L([012])(?:_scc)?\.nc$", re.IGNORECASE)


def _utc_days(start: Any, end: Any) -> list[str]:
    """Calendar intersection with half-open [start, end) scientific sessions."""
    last = (end - timedelta(microseconds=1)).date()
    current = start.date()
    days = []
    while current <= last:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


class Catalog:
    """Metadata-only index. Never scan the contents of large NetCDF variables."""

    def __init__(self, config_path: str | Path = "config.yaml", ttl: float = 120.0):
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = load_config(self.config_path)
        self.root = processed_data_root(
            self.config, root_dir=self.config_path.parent
        ).resolve()
        self.station_id = station_id(self.config)
        station = self.config.get("_station_catalog", {}).get("station", {})
        site = station.get("site", {})
        self.site = {
            "name": station.get("name") or self.station_id,
            "id": self.station_id,
            "latitude": site.get("latitude"),
            "longitude": site.get("longitude"),
            "elevation_m": site.get("station_altitude_m"),
            "timezone": station.get("timezone", "UTC"),
        }
        self.ttl = ttl
        self._expires = 0.0
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def _refresh(self) -> None:
        with self._lock:
            if time.monotonic() < self._expires:
                return
            sessions: dict[str, dict[str, Any]] = {}
            for folder in sorted(self.root.glob(f"{self.station_id}/*/*/*")):
                if not folder.is_dir() or folder.is_symlink():
                    continue
                sid = folder.name
                try:
                    validate_session_id_for_config(sid, self.config)
                    expected = session_dir(
                        sid, self.config, root_dir=self.config_path.parent
                    ).resolve()
                    if folder.resolve() != expected:
                        continue
                    _, start, end = session_id_parts(sid)
                except ValueError:
                    continue
                products = []
                for path in sorted(folder.glob("*.nc")):
                    if not path.is_file() or path.is_symlink():
                        continue
                    if not path.resolve().is_relative_to(self.root):
                        continue
                    match = _PRODUCT.search(path.name)
                    if not match:
                        continue
                    try:
                        if product_session_id(path) != sid:
                            continue
                    except ValueError:
                        continue
                    level = int(match.group(1))
                    variant = path.name[len(sid):-len(match.group(0))].strip("_")
                    scc = path.stem.lower().endswith("_scc")
                    label = ("SCC" if scc else "MILGRAU") + f" L{level}"
                    if variant:
                        label += f" · {variant}"
                    products.append({
                        "id": path.name, "level": level,
                        "variant": variant or "canonical",
                        "scc": scc, "label": label,
                        "_path": path,
                    })
                if not products:
                    continue
                sessions[sid] = {
                    "id": sid,
                    "start_utc": start.isoformat(timespec="minutes"),
                    "end_utc": end.isoformat(timespec="minutes"),
                    "dates_utc": _utc_days(start, end),
                    "_products": products,
                }
            self._sessions = sessions
            self._expires = time.monotonic() + self.ttl

    def dates(self) -> list[str]:
        self._refresh()
        return sorted(
            {day for s in self._sessions.values() for day in s["dates_utc"]},
            reverse=True,
        )

    def sessions(self, utc_date: date) -> list[dict[str, Any]]:
        self._refresh()
        return sorted(
            [
                {key: value for key, value in entry.items() if not key.startswith("_")}
                for entry in self._sessions.values()
                if utc_date.isoformat() in entry["dates_utc"]
            ],
            key=lambda row: row["start_utc"],
            reverse=True,
        )

    def all_sessions(self) -> list[dict[str, Any]]:
        """Newest-first inventory without imposing calendar/publication windows."""
        self._refresh()
        return sorted(
            [
                {k: v for k, v in entry.items() if not k.startswith("_")}
                for entry in self._sessions.values()
            ],
            key=lambda entry: entry["start_utc"],
            reverse=True,
        )

    def products(self, session_id: str) -> list[dict[str, Any]]:
        self._refresh()
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Sessão não encontrada")
        return [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in session["_products"]
        ]

    def resolve(self, session_id: str, product_id: str, *, level: int = 1) -> Path:
        self._refresh()
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Sessão não encontrada")
        for entry in session["_products"]:
            if entry["id"] == product_id and entry["level"] == level:
                path = entry["_path"]
                if path.is_file() and not path.is_symlink():
                    resolved = path.resolve()
                    if resolved.is_relative_to(self.root):
                        return resolved
        raise KeyError("Produto L1 não encontrado nesta sessão")
