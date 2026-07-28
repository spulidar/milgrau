"""Synthetic offline helpers shared by SCI-004B tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from milgrau.meteorology.cache import Era5Release
from milgrau.meteorology.era5_acquisition import Era5DecodedData
from milgrau.meteorology.radiosonde_acquisition import (
    RAW_CANONICAL_DATAFRAME,
    RadiosondeRawPayload,
)
from milgrau.meteorology.request import (
    AcquisitionMode,
    MeteorologyProvider,
    MeteorologyRequest,
)

FIXTURE_DIR = Path(__file__).with_name("fixtures")
ANALYSIS_TIME = datetime(2026, 7, 5, 12, tzinfo=UTC)


def meteorology_request(
    cache_directory: Path,
    *,
    mode: AcquisitionMode = AcquisitionMode.AUTO,
    provider: MeteorologyProvider = MeteorologyProvider.BOTH,
    measurement_timestamps: tuple[datetime, ...] = (
        datetime(2026, 7, 5, 12, 30, tzinfo=UTC),
    ),
    radiosonde_nominal_times: tuple[datetime, ...] = (ANALYSIS_TIME,),
    allow_era5t: bool = True,
) -> MeteorologyRequest:
    return MeteorologyRequest(
        site_id="spu",
        latitude_deg_north=-23.5615,
        longitude_deg_east=-46.7383,
        station_altitude_m=760.0,
        measurement_timestamps=measurement_timestamps,
        provider=provider,
        mode=mode,
        cache_directory=cache_directory,
        radiosonde_station_id="83779",
        radiosonde_nominal_times=radiosonde_nominal_times,
        allow_era5t=allow_era5t,
        fallback_altitudes_m=(760.0, 1760.0, 2760.0),
    )


def radiosonde_table() -> pd.DataFrame:
    payload = json.loads(
        (FIXTURE_DIR / "radiosonde_campo_de_marte.json").read_text(encoding="utf-8")
    )
    table = pd.DataFrame(payload["records"])
    table["station"] = "SBMT"
    table["station_number"] = 83779
    table["time"] = pd.Timestamp(payload["observation_time_utc"])
    table["latitude"] = payload["latitude_deg_north"]
    table["longitude"] = payload["longitude_deg_east"]
    table["elevation"] = 722.0
    table["pw"] = 25.0
    object.__setattr__(table, "units", {
        "pressure": "hPa",
        "height": "meter",
        "temperature": "degC",
        "dewpoint": "degC",
        "direction": "degrees",
        "speed": "knot",
        "station": None,
        "station_number": None,
        "time": None,
        "latitude": "degrees",
        "longitude": "degrees",
        "elevation": "meter",
        "pw": "millimeter",
    })
    return table


def radiosonde_transport(_time: datetime, station_id: str) -> RadiosondeRawPayload:
    assert station_id == "83779"
    return RadiosondeRawPayload(
        payload=b"replaced by deterministic canonical snapshot",
        payload_kind=RAW_CANONICAL_DATAFRAME,
        table=radiosonde_table(),
    )


def era5_decoded(
    *,
    times: tuple[datetime, ...] = (
        datetime(2026, 7, 5, 12, tzinfo=UTC),
        datetime(2026, 7, 5, 13, tzinfo=UTC),
    ),
    release: Era5Release = Era5Release.FINAL,
) -> Era5DecodedData:
    payload = json.loads(
        (FIXTURE_DIR / "era5_l137_four_points.json").read_text(encoding="utf-8")
    )
    fraction = np.linspace(0.0, 1.0, 137, dtype=np.float64)
    definition = payload["temperature_profile"]
    one_temperature = float(definition["top_k"]) + (
        float(definition["surface_k"]) - float(definition["top_k"])
    ) * fraction ** float(definition["exponent"])
    one_temperature = one_temperature[:, None] + np.asarray(
        definition["corner_offsets_k"], dtype=np.float64
    )[None, :]
    humidity_definition = payload["specific_humidity_profile"]
    one_humidity = float(humidity_definition["top_kg_kg"]) + (
        float(humidity_definition["surface_kg_kg"])
        - float(humidity_definition["top_kg_kg"])
    ) * fraction ** float(humidity_definition["exponent"])
    one_humidity = one_humidity[:, None] * np.asarray(
        humidity_definition["corner_scale"], dtype=np.float64
    )[None, :]
    return Era5DecodedData(
        analysis_times=times,
        coordinates_lat_lon=np.asarray(
            payload["corner_coordinates_lat_lon"], dtype=np.float64
        ),
        hybrid_a_pa=np.asarray(payload["hybrid_a_pa"], dtype=np.float64),
        hybrid_b=np.asarray(payload["hybrid_b"], dtype=np.float64),
        temperature_k=np.stack(
            [one_temperature + 0.1 * index for index in range(len(times))]
        ),
        specific_humidity_kg_kg=np.stack(
            [one_humidity for _ in times]
        ),
        logarithm_surface_pressure=np.stack(
            [
                np.log(
                    np.asarray(
                        payload["surface_pressure_pa_by_corner"], dtype=np.float64
                    )
                )
                for _ in times
            ]
        ),
        surface_geopotential_m2_s2=np.stack(
            [
                np.asarray(
                    payload["surface_geopotential_m2_s2_by_corner"],
                    dtype=np.float64,
                )
                for _ in times
            ]
        ),
        release=release,
    )
