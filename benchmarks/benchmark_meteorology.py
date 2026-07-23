#!/usr/bin/env python3
"""Local-only SCI-004A microbenchmark over redistributable profile fixtures."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milgrau.meteorology.blending import blend_radiosonde_and_era5
from milgrau.meteorology.contracts import AtmosphericProfile
from milgrau.meteorology.era5_model_levels import normalize_era5_model_levels
from milgrau.meteorology.radiosonde import normalize_wyoming_radiosonde

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "tests" / "fixtures"


def _load_fixture(name: str) -> tuple[dict[str, Any], bytes]:
    raw = (_FIXTURES / name).read_bytes()
    return json.loads(raw), raw


def _era5_fields(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    level_fraction = np.linspace(0.0, 1.0, 137, dtype=np.float64)
    temperature_definition = payload["temperature_profile"]
    temperature = float(temperature_definition["top_k"]) + (
        float(temperature_definition["surface_k"])
        - float(temperature_definition["top_k"])
    ) * level_fraction ** float(temperature_definition["exponent"])
    temperature = temperature[:, None] + np.asarray(
        temperature_definition["corner_offsets_k"], dtype=np.float64
    )[None, :]
    humidity_definition = payload["specific_humidity_profile"]
    humidity = float(humidity_definition["top_kg_kg"]) + (
        float(humidity_definition["surface_kg_kg"])
        - float(humidity_definition["top_kg_kg"])
    ) * level_fraction ** float(humidity_definition["exponent"])
    humidity = humidity[:, None] * np.asarray(
        humidity_definition["corner_scale"], dtype=np.float64
    )[None, :]
    return temperature, humidity


def _radio_operation(payload: dict[str, Any], raw: bytes, *, complete: bool = False):
    records = payload["records"]
    if complete:
        records = [
            record
            for record in records
            if record["height"] <= 6500.0 and record["pressure"] is not None
        ]
    return lambda: normalize_wyoming_radiosonde(
        pd.DataFrame(records),
        nominal_time=datetime.fromisoformat(payload["nominal_time_utc"]),
        observation_time=datetime.fromisoformat(payload["observation_time_utc"]),
        station_id=str(payload["station_id"]),
        latitude_deg_north=float(payload["latitude_deg_north"]),
        longitude_deg_east=float(payload["longitude_deg_east"]),
        raw_snapshot=raw,
    )


def _era5_operation(payload: dict[str, Any], raw: bytes):
    temperature, humidity = _era5_fields(payload)
    return lambda: normalize_era5_model_levels(
        hybrid_a_pa=np.asarray(payload["hybrid_a_pa"], dtype=np.float64),
        hybrid_b=np.asarray(payload["hybrid_b"], dtype=np.float64),
        temperature_k_by_level_corner=temperature,
        specific_humidity_by_level_corner=humidity,
        logarithm_surface_pressure_by_corner=np.log(
            np.asarray(payload["surface_pressure_pa_by_corner"], dtype=np.float64)
        ),
        surface_geopotential_m2_s2_by_corner=np.asarray(
            payload["surface_geopotential_m2_s2_by_corner"], dtype=np.float64
        ),
        corner_coordinates_lat_lon=np.asarray(
            payload["corner_coordinates_lat_lon"], dtype=np.float64
        ),
        target_latitude_deg_north=float(payload["target_latitude_deg_north"]),
        target_longitude_deg_east=float(payload["target_longitude_deg_east"]),
        analysis_time=datetime.fromisoformat(payload["analysis_time_utc"]),
        dataset_id=str(payload["dataset_id"]),
        raw_snapshot=raw,
    )


def _measure(operation: Callable[[], Any], repetitions: int) -> tuple[dict[str, float], Any]:
    operation()
    timings: list[float] = []
    peaks: list[int] = []
    result: Any = None
    for _ in range(repetitions):
        tracemalloc.start()
        started = time.perf_counter()
        result = operation()
        timings.append(time.perf_counter() - started)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
    return (
        {
            "median_seconds": statistics.median(timings),
            "minimum_seconds": min(timings),
            "maximum_seconds": max(timings),
            "median_peak_allocated_bytes": float(statistics.median(peaks)),
        },
        result,
    )


def profile_array_bytes(profile: AtmosphericProfile) -> int:
    return int(
        sum(
            value.nbytes
            for field in fields(profile)
            if isinstance((value := getattr(profile, field.name)), np.ndarray)
        )
    )


def run_benchmark(repetitions: int = 30) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least one.")
    radiosonde_payload, radiosonde_raw = _load_fixture("radiosonde_campo_de_marte.json")
    era5_payload, era5_raw = _load_fixture("era5_l137_four_points.json")
    radio_operation = _radio_operation(radiosonde_payload, radiosonde_raw)
    complete_radio_operation = _radio_operation(
        radiosonde_payload, radiosonde_raw, complete=True
    )
    era5_operation = _era5_operation(era5_payload, era5_raw)
    radio_metrics, radio_result = _measure(radio_operation, repetitions)
    era5_metrics, era5_result = _measure(era5_operation, repetitions)
    complete_radio = complete_radio_operation().profile
    target = np.arange(800.0, 20001.0, 50.0)
    blend_operation = lambda: blend_radiosonde_and_era5(
        complete_radio,
        era5_result.profile,
        target,
        blend_width_m=1200.0,
    )
    blend_metrics, blend_result = _measure(blend_operation, repetitions)
    return {
        "repetitions": repetitions,
        "network_calls": 0,
        "radiosonde_normalization": radio_metrics,
        "era5_l137_reconstruction": era5_metrics,
        "hybrid_blend_385_bins": blend_metrics,
        "profile_sizes": {
            "radiosonde_levels": radio_result.profile.geometric_altitude_m.size,
            "radiosonde_array_bytes": profile_array_bytes(radio_result.profile),
            "era5_levels": era5_result.profile.geometric_altitude_m.size,
            "era5_array_bytes": profile_array_bytes(era5_result.profile),
            "hybrid_levels": blend_result.profile.geometric_altitude_m.size,
            "hybrid_array_bytes": profile_array_bytes(blend_result.profile),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run_benchmark(args.repetitions)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for name in (
            "radiosonde_normalization",
            "era5_l137_reconstruction",
            "hybrid_blend_385_bins",
        ):
            print(f"{name}: {report[name]['median_seconds']:.6f} s median")
        print(json.dumps(report["profile_sizes"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
