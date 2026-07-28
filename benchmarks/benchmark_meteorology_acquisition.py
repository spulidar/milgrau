#!/usr/bin/env python3
"""Deterministic local-only SCI-004B cache and normalization benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milgrau.meteorology.cache import Era5Release
from milgrau.meteorology.era5_acquisition import Era5DecodedData, acquire_era5
from milgrau.meteorology.request import (
    AcquisitionMode,
    MeteorologyProvider,
    MeteorologyRequest,
    plan_era5_hours,
)
from milgrau.meteorology.snapshots import profiles_to_netcdf_bytes

_FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "era5_l137_four_points.json"


def _decoded(times: tuple[datetime, ...]) -> Era5DecodedData:
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    fraction = np.linspace(0.0, 1.0, 137)
    temperature_definition = payload["temperature_profile"]
    temperature = float(temperature_definition["top_k"]) + (
        float(temperature_definition["surface_k"])
        - float(temperature_definition["top_k"])
    ) * fraction ** float(temperature_definition["exponent"])
    temperature = temperature[:, None] + np.asarray(
        temperature_definition["corner_offsets_k"]
    )[None, :]
    humidity_definition = payload["specific_humidity_profile"]
    humidity = float(humidity_definition["top_kg_kg"]) + (
        float(humidity_definition["surface_kg_kg"])
        - float(humidity_definition["top_kg_kg"])
    ) * fraction ** float(humidity_definition["exponent"])
    humidity = humidity[:, None] * np.asarray(
        humidity_definition["corner_scale"]
    )[None, :]
    return Era5DecodedData(
        analysis_times=times,
        coordinates_lat_lon=np.asarray(payload["corner_coordinates_lat_lon"]),
        hybrid_a_pa=np.asarray(payload["hybrid_a_pa"]),
        hybrid_b=np.asarray(payload["hybrid_b"]),
        temperature_k=np.stack(
            [temperature + 0.1 * index for index in range(len(times))]
        ),
        specific_humidity_kg_kg=np.stack([humidity for _ in times]),
        logarithm_surface_pressure=np.stack(
            [
                np.log(np.asarray(payload["surface_pressure_pa_by_corner"]))
                for _ in times
            ]
        ),
        surface_geopotential_m2_s2=np.stack(
            [
                np.asarray(payload["surface_geopotential_m2_s2_by_corner"])
                for _ in times
            ]
        ),
        release=Era5Release.FINAL,
    )


def _measure(operation: Callable[[], Any], repetitions: int) -> float:
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return float(statistics.median(samples))


def run_benchmark(repetitions: int = 20) -> dict[str, object]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive.")
    measurements = tuple(
        datetime(2026, 7, 5, 12, 15, tzinfo=UTC) + timedelta(minutes=10 * index)
        for index in range(24)
    )
    planned = plan_era5_hours(measurements)
    decoded = _decoded(planned)
    raw_payload = b"GRIB-MOCK\0" + _FIXTURE.read_bytes()
    network_calls = 0
    with tempfile.TemporaryDirectory(prefix="milgrau-meteorology-benchmark-") as directory:
        request = MeteorologyRequest(
            site_id="spu",
            latitude_deg_north=-23.5615,
            longitude_deg_east=-46.7383,
            station_altitude_m=760.0,
            measurement_timestamps=measurements,
            provider=MeteorologyProvider.ERA5,
            mode=AcquisitionMode.AUTO,
            cache_directory=Path(directory),
            radiosonde_nominal_times=(),
        )

        def local_transport(*_args) -> bytes:
            nonlocal network_calls
            return raw_payload

        first = acquire_era5(
            request,
            transport=local_transport,
            decoder=lambda _payload: decoded,
        )
        if not first.available:
            raise RuntimeError(first.error_message)

        planning_seconds = _measure(
            lambda: plan_era5_hours(measurements),
            repetitions,
        )
        cache_hit_seconds = _measure(
            lambda: acquire_era5(
                request,
                transport=lambda *_args: (_ for _ in ()).throw(
                    AssertionError("benchmark cache hit accessed transport")
                ),
                decoder=lambda _payload: (_ for _ in ()).throw(
                    AssertionError("benchmark cache hit decoded raw GRIB")
                ),
            ),
            repetitions,
        )
        normalization_seconds = _measure(
            lambda: profiles_to_netcdf_bytes(
                decoded.profiles(request, raw_payload),
                cache_metadata={
                    "snapshot_schema": "milgrau-normalized-era5-l137-v1",
                    "era5_release": "final",
                    "meteorology_provisional": 0,
                },
            ),
            repetitions,
        )
        raw_size = first.raw_files[0].stat().st_size
        normalized_size = first.normalized_files[0].stat().st_size
        manifest_size = sum(path.stat().st_size for path in first.manifest_files)
    return {
        "repetitions": repetitions,
        "network_calls": network_calls,
        "measurement_count": len(measurements),
        "planned_hour_count": len(planned),
        "planning_median_seconds": planning_seconds,
        "cache_hit_median_seconds": cache_hit_seconds,
        "local_raw_normalization_median_seconds": normalization_seconds,
        "raw_mock_grib_bytes": raw_size,
        "normalized_netcdf_bytes": normalized_size,
        "manifest_bytes": manifest_size,
        "manifest_overhead_fraction": manifest_size / (raw_size + normalized_size),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args(argv)
    print(json.dumps(run_benchmark(args.repetitions), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
