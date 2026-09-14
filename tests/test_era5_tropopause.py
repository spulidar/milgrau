"""ERA5 tropopause integration regression."""

from __future__ import annotations

from copy import deepcopy
import logging

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.config.loader import load_config
from milgrau.level1.thermodynamics import integrate_thermodynamics
from milgrau.level1.tropopause import calculate_tropopause_heights


def test_era5_reuses_thermal_tropopause_kernel(monkeypatch) -> None:
    profile = pd.DataFrame(
        {
            "height": [800, 5000, 10000, 14000, 16000, 17000, 18000, 19000, 20000],
            "temperature": [20, -10, -45, -55, -60, -62, -62.5, -62, -60],
            "pressure": [930, 540, 265, 145, 105, 88, 73, 60, 50],
        }
    )
    profile.attrs.update(
        {
            "source_type": "era5",
            "source": "synthetic ERA5 profile",
            "analysis_datetime_utc": "2024-06-10T12:00:00+00:00",
            "time_delta_hours": 0.0,
            "doi": "test-doi",
        }
    )
    expected_cpt, expected_lrt = calculate_tropopause_heights(profile)

    monkeypatch.setattr("milgrau.level1.thermodynamics.fetch_wyoming_radiosonde", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "milgrau.level1.thermodynamics.fetch_era5_pressure_level_profile",
        lambda *args, **kwargs: profile,
    )

    config = deepcopy(load_config("config.yaml"))
    config["level1"]["atmosphere"]["source_priority"] = ["radiosonde", "era5", "ussa76"]
    shell = xr.Dataset(
        coords={
            "time": pd.date_range("2024-06-10T12:00:00", periods=3, freq="10min"),
            "altitude": np.arange(0.0, 20_000.0, 500.0),
        }
    )

    result = integrate_thermodynamics(shell, config, logging.getLogger("test-era5-tropopause"))

    assert result.attrs["tropopause_source_type"] == "era5"
    assert np.isclose(float(result.attrs["tropopause_cpt_km"]), expected_cpt)
    assert np.isclose(float(result.attrs["tropopause_lrt_km"]), expected_lrt)
