"""Evidence for the operator's documented command surface and solar regime.

All tests are offline and use clean-install console scripts or mathematical
reference calculations. They are not substitutes for real Licel validation.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import distribution

import numpy as np
import pandas as pd
import pytest

from milgrau.physics.solar import build_solar_segments, solar_elevation_deg, solar_regime


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("milgrau-libids", "--scc-reexport"),
        ("milgrau-lipancora", "--radiosonde-only"),
        ("milgrau-lebear", "--time-window-utc"),
        ("milgrau-inspect", "--input"),
        ("milgrau-explorer", "--help"),
        ("milgrau-scc", "upload"),
    ],
)
def test_documented_console_commands_have_installed_help(command: str, expected: str) -> None:
    scripts = {entry.name for entry in distribution("milgrau").entry_points if entry.group == "console_scripts"}
    assert command in scripts, f"Missing console entry point: {command}"
    executable = shutil.which(command)
    assert executable is not None, f"Installed console entry point is absent from PATH: {command}"
    result = subprocess.run(
        [executable, "--help"],
        capture_output=True, text=True, timeout=35, check=False,
    )
    assert result.returncode == 0, f"{command} --help failed: {result.stderr}"
    assert expected in result.stdout


def _independent_noaa_approximation(
    time_utc: datetime, latitude_deg: float, longitude_deg: float,
) -> float:
    """Published NOAA fractional-year solar-position approximation.

    Deliberately different from MILGRAU's Julian-century astronomy equations;
    compare at the precision justified by this simpler approximation, not at
    sub-arcminute precision.
    """
    assert time_utc.tzinfo is not None
    utc = time_utc.astimezone(timezone.utc)
    day_of_year = utc.timetuple().tm_yday
    days_in_year = 366 if (
        utc.year % 4 == 0 and (utc.year % 100 != 0 or utc.year % 400 == 0)
    ) else 365
    h = utc.hour + utc.minute / 60 + utc.second / 3600
    gamma = 2 * math.pi / days_in_year * (day_of_year - 1 + (h - 12) / 24)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    true_solar_minutes = (h * 60 + eqtime + 4 * longitude_deg) % 1440
    hour_angle_rad = math.radians(true_solar_minutes / 4 - 180)
    latitude_rad = math.radians(latitude_deg)
    cosine_zenith = (
        math.sin(latitude_rad) * math.sin(decl)
        + math.cos(latitude_rad) * math.cos(decl) * math.cos(hour_angle_rad)
    )
    return 90 - math.degrees(math.acos(max(-1, min(1, cosine_zenith))))


@pytest.mark.parametrize(
    ("timestamp", "latitude", "longitude"),
    [
        ("2025-03-20T12:00:00Z", 0.0, 0.0),
        ("2025-06-21T15:00:00Z", -23.5615, -46.7383),
        ("2025-09-22T07:00:00Z", 40.7128, -74.0060),
        ("2025-12-21T16:00:00Z", -23.5615, -46.7383),
        ("2024-06-21T12:00:00Z", 51.5074, -0.1278),
    ],
)
def test_geometric_solar_elevation_agrees_with_independent_noaa_formula(
    timestamp: str, latitude: float, longitude: float,
) -> None:
    when = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    predicted = float(solar_elevation_deg(pd.DatetimeIndex([when]), latitude, longitude)[0])
    approximation = _independent_noaa_approximation(when, latitude, longitude)
    assert np.isfinite(predicted)
    # NOAA's simplified fractional-year formula is approximate; no claim of
    # sub-degree astronomical reference accuracy is made here.
    assert abs(predicted - approximation) < 2.0


def test_solar_threshold_and_return_to_day_are_contiguous_not_one_global_regime() -> None:
    regimes = solar_regime(np.array([-3.01, -3.0, 7.0, -3.01, 6.0]), day_night_threshold_deg=-3.0)
    assert regimes.tolist() == ["night", "day", "day", "night", "day"]
    starts = pd.to_datetime(
        ["2025-01-01T20:00Z", "2025-01-01T20:30Z", "2025-01-01T21:00Z",
         "2025-01-02T07:00Z", "2025-01-02T12:00Z"], utc=True,
    )
    labels, segments = build_solar_segments(starts, starts + pd.Timedelta(minutes=20), regimes)
    assert labels.tolist() == ["seg00", "seg01", "seg01", "seg02", "seg03"]
    assert [part.regime for part in segments] == ["night", "day", "night", "day"]
    assert segments[-1].start_time_utc == starts[-1]
