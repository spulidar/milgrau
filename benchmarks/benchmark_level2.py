"""Reproducible synthetic benchmarks for the LEBEAR Level 2 pipeline."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import logging
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, TypeVar

import numpy as np
import xarray as xr

from milgrau.io.contracts import validate_level1_contract, validate_level2_contract
from milgrau.level2.dataset import build_level2_dataset
from milgrau.level2.completeness import (
    Level2ProductContract,
    ProductCompleteness,
    WavelengthAttempt,
    WavelengthFailureCode,
    WavelengthFailureDiagnostic,
    WavelengthFailureStage,
    diagnostic_from_exception,
)
from milgrau.level2.lebear import _write_level2_atomically
from milgrau.level2.retrieval import (
    assemble_wavelength_result,
    build_molecular_model,
    glue_signal_blocks,
    prepare_wavelength_blocks,
    retrieve_optical_blocks,
)

_Result = TypeVar("_Result")
_PACKAGE_NAMES: Final = ("milgrau", "numpy", "pandas", "xarray", "netCDF4", "scipy", "numba")
_THREAD_ENVIRONMENT: Final = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True, slots=True)
class Scenario:
    """One deterministic benchmark dataset and retrieval workload."""

    name: str
    description: str
    n_profiles: int
    n_altitude: int
    wavelengths_nm: tuple[int, ...]
    monte_carlo_iterations: int
    local_only: bool = False
    profile_interval_minutes: int = 5


SCENARIOS: Final[dict[str, Scenario]] = {
    "ci": Scenario(
        name="ci",
        description="Small two-wavelength smoke benchmark suitable for CI.",
        n_profiles=3,
        n_altitude=240,
        wavelengths_nm=(355, 532),
        monte_carlo_iterations=5,
    ),
    "typical": Scenario(
        name="typical",
        description="Reduced redistributable two-hour measurement with two wavelengths.",
        n_profiles=24,
        n_altitude=800,
        wavelengths_nm=(355, 532),
        monte_carlo_iterations=30,
    ),
    "large": Scenario(
        name="large",
        description="Local-only synthetic 24-hour day with full-height profiles.",
        n_profiles=288,
        n_altitude=4000,
        wavelengths_nm=(355, 532),
        monte_carlo_iterations=300,
        local_only=True,
    ),
}


class _MaterializationCounter:
    """Count unique NumPy arrays observed at explicit pipeline boundaries."""

    def __init__(self) -> None:
        self._seen: set[int] = set()
        self.array_count = 0
        self.array_bytes = 0

    def observe(self, value: Any) -> None:
        if isinstance(value, np.ndarray):
            identity = id(value)
            if identity not in self._seen:
                self._seen.add(identity)
                self.array_count += 1
                self.array_bytes += int(value.nbytes)
            return
        if isinstance(value, xr.Dataset):
            for variable in value.variables.values():
                self.observe(variable.data)
            return
        if isinstance(value, xr.DataArray):
            self.observe(value.data)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                self.observe(getattr(value, field.name))
            return
        if isinstance(value, Mapping):
            for item in value.values():
                self.observe(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self.observe(item)


def build_synthetic_level1(scenario: Scenario) -> xr.Dataset:
    """Build a deterministic, generated-in-memory Level 1 fixture."""
    altitude_m = np.arange(scenario.n_altitude, dtype=np.float64) * 7.5
    time_values = np.datetime64("2024-01-01T00:00:00", "ns") + np.arange(
        scenario.n_profiles,
        dtype=np.int64,
    ) * np.timedelta64(scenario.profile_interval_minutes, "m")
    channel_names = tuple(
        channel
        for wavelength_nm in scenario.wavelengths_nm
        for channel in (f"{wavelength_nm}.AN", f"{wavelength_nm}.PC")
    )
    shape = (scenario.n_profiles, len(channel_names), scenario.n_altitude)
    corrected = np.empty(shape, dtype=np.float32)

    altitude_structure = np.exp(-altitude_m / 7000.0) * (
        1.0 + 0.025 * np.sin(altitude_m / 350.0)
    ) + 0.05
    for profile_index in range(scenario.n_profiles):
        time_scale = 1.0 + 0.02 * np.sin(2.0 * np.pi * profile_index / max(scenario.n_profiles, 1))
        for channel_index, channel_name in enumerate(channel_names):
            wavelength_index = channel_index // 2
            detector_scale = 1.04 if channel_name.endswith(".PC") else 1.0
            corrected[profile_index, channel_index, :] = (
                altitude_structure * time_scale * detector_scale * (1.0 + 0.03 * wavelength_index)
            )

    corrected_error = (0.02 * np.abs(corrected)).astype(np.float32)
    range_factor = altitude_m.astype(np.float32) ** 2
    range_corrected = corrected * range_factor[None, None, :]
    range_corrected_error = corrected_error * range_factor[None, None, :]
    saturation_mask = np.zeros(shape, dtype=bool)
    return xr.Dataset(
        data_vars={
            "corrected_signal": (("time", "channel", "altitude"), corrected),
            "corrected_signal_error": (("time", "channel", "altitude"), corrected_error),
            "range_corrected_signal": (("time", "channel", "altitude"), range_corrected),
            "range_corrected_signal_error": (
                ("time", "channel", "altitude"),
                range_corrected_error,
            ),
            "pc_saturation_mask": (("time", "channel", "altitude"), saturation_mask),
            "channel_correction_success": (
                ("channel",),
                np.ones(len(channel_names), dtype=np.int8),
            ),
        },
        coords={
            "time": time_values,
            "channel": np.asarray(channel_names, dtype=object),
            "altitude": altitude_m,
        },
        attrs={
            "Processing_level": "Level 1 deterministic synthetic benchmark product",
            "Altitude_units": "m",
            "Benchmark_scenario": scenario.name,
        },
    )


def benchmark_config(scenario: Scenario) -> dict[str, Any]:
    """Return fixed retrieval settings appropriate to one synthetic fixture."""
    highest_altitude_m = (scenario.n_altitude - 1) * 7.5
    reference_min_m = max(500.0, highest_altitude_m * 0.25)
    reference_max_m = min(25000.0, highest_altitude_m * 0.80)
    reference_window_bins = min(80, max(20, scenario.n_altitude // 12))
    gluing_window_bins = min(80, max(20, scenario.n_altitude // 20))
    return {
        "processing": {"incremental": False},
        "site": {"station_altitude_m": 760.0},
        "inversion": {
            "wavelengths_to_process": list(scenario.wavelengths_nm),
            "kfs_mode": "two_sided",
            "temporal_average_minutes": 15,
            "monte_carlo_iterations": scenario.monte_carlo_iterations,
            "random_seed": 20260722,
            "molecular_fit": {
                "ref_alt_min_m": reference_min_m,
                "ref_alt_max_m": reference_max_m,
                "ref_window_bins": reference_window_bins,
                "max_relative_slope": 10.0,
                "max_relative_variance": 10.0,
                "min_valid_fraction": 0.50,
            },
            "gluing": {
                "window_length_bins": gluing_window_bins,
                "correlation_threshold": 0.10,
                "search_min_idx": max(10, scenario.n_altitude // 10),
                "search_max_idx": max(40, scenario.n_altitude * 3 // 4),
                "intercept_threshold": 100.0,
                "gaussian_threshold": 0.001,
                "minmax_threshold": 0.001,
                "max_relative_rmse": 1.0,
                "max_relative_bias": 1.0,
                "min_valid_fraction": 0.80,
                "allow_single_channel_fallback": False,
                "single_channel_priority": "photon_counting",
            },
            "lidar_ratios_sr": {
                str(wavelength_nm): {"01": 60.0}
                for wavelength_nm in scenario.wavelengths_nm
            },
            "lidar_ratio_std_sr": {
                str(wavelength_nm): 5.0
                for wavelength_nm in scenario.wavelengths_nm
            },
        },
        "visualization": {"level2_qa": {"enabled": False}},
    }


def _timed(
    timings: dict[str, float],
    stage: str,
    operation: Callable[[], _Result],
) -> _Result:
    started_at = time.perf_counter()
    try:
        return operation()
    finally:
        timings[stage] = timings.get(stage, 0.0) + time.perf_counter() - started_at


def _load_level1(input_path: Path) -> xr.Dataset:
    with xr.open_dataset(input_path) as opened:
        loaded = opened.load()
        validate_level1_contract(loaded)
    return loaded


def execute_pipeline(
    input_path: Path,
    output_path: Path,
    scenario: Scenario,
    product_mode: str = "complete",
) -> dict[str, Any]:
    """Execute and time the production Level 2 stages once."""
    timings: dict[str, float] = {}
    materializations = _MaterializationCounter()
    logger = logging.getLogger("milgrau.benchmark.level2")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    config = benchmark_config(scenario)

    total_started_at = time.perf_counter()
    ds_l1 = _timed(timings, "input_open_load_validation", lambda: _load_level1(input_path))
    materializations.observe(ds_l1)
    altitude_m = np.asarray(ds_l1["altitude"].values, dtype=np.float64)
    if np.nanmax(altitude_m) <= 100.0:
        altitude_m = altitude_m * 1000.0

    attempts: list[WavelengthAttempt] = []
    valid_retrieval_blocks = 0
    for wavelength_nm in scenario.wavelengths_nm:
        retrieval_stage = "selection_and_blocking"
        try:
            inputs = _timed(
                timings,
                retrieval_stage,
                lambda wavelength_nm=wavelength_nm: prepare_wavelength_blocks(
                    ds_l1,
                    wavelength_nm,
                    altitude_m,
                    config,
                ),
            )
            materializations.observe(inputs)
            retrieval_stage = "gluing"
            glued = _timed(
                timings,
                retrieval_stage,
                lambda: glue_signal_blocks(inputs, altitude_m, logger),
            )
            materializations.observe(glued)
            retrieval_stage = "molecular_model"
            molecular_model = _timed(
                timings,
                retrieval_stage,
                lambda wavelength_nm=wavelength_nm: build_molecular_model(
                    ds_l1,
                    wavelength_nm,
                    altitude_m,
                    config,
                ),
            )
            materializations.observe(molecular_model)
            retrieval_stage = "rayleigh_kfs"
            molecular, optical, rayleigh, kfs = _timed(
                timings,
                retrieval_stage,
                lambda: retrieve_optical_blocks(
                    inputs,
                    glued,
                    molecular_model,
                    altitude_m,
                    config,
                    logger,
                ),
            )
            materializations.observe((molecular, optical, rayleigh, kfs))
            retrieval_stage = "result_assembly"
            result = _timed(
                timings,
                retrieval_stage,
                lambda: assemble_wavelength_result(inputs, glued, molecular, optical, rayleigh, kfs),
            )
        except Exception as exc:
            attempts.append(
                WavelengthAttempt.recoverable_failure(
                    diagnostic_from_exception(
                        wavelength_nm,
                        exc,
                        retrieval_stage=retrieval_stage,
                    )
                )
            )
            continue
        materializations.observe(result)
        block_successes = int(result.optical.retrieval_success_flag.sum())
        valid_retrieval_blocks += block_successes
        if block_successes:
            attempts.append(WavelengthAttempt.success(result))
        else:
            attempts.append(
                WavelengthAttempt.recoverable_failure(
                    WavelengthFailureDiagnostic(
                        wavelength_nm=wavelength_nm,
                        stage=WavelengthFailureStage.RETRIEVAL_VALIDATION,
                        code=WavelengthFailureCode.NO_VALID_RETRIEVAL_BLOCK,
                        message="Synthetic benchmark produced no valid optical block.",
                    )
                )
            )

    if valid_retrieval_blocks == 0:
        raise RuntimeError("Synthetic benchmark produced no valid KFS retrieval blocks.")
    product_contract = Level2ProductContract.from_attempts(
        scenario.wavelengths_nm,
        attempts,
    )
    if product_mode == "complete" and product_contract.completeness is not ProductCompleteness.COMPLETE:
        raise RuntimeError("Complete benchmark fixture did not process every requested wavelength.")
    if product_mode == "partial" and product_contract.completeness is not ProductCompleteness.PARTIAL:
        raise RuntimeError("Partial benchmark fixture did not produce a partial product.")
    results = [
        attempt.result
        for attempt in sorted(attempts, key=lambda item: item.wavelength_nm)
        if attempt.result is not None
    ]

    ds_l2 = _timed(
        timings,
        "dataset_assembly",
        lambda: build_level2_dataset(
            ds_l1,
            results,
            altitude_m,
            input_path,
            config,
            product_contract,
        ),
    )
    materializations.observe(ds_l2)
    _timed(timings, "output_validation", lambda: validate_level2_contract(ds_l2))
    encoding = {
        variable: {"zlib": True, "complevel": 4}
        for variable in ds_l2.data_vars
        if ds_l2[variable].ndim > 0 and ds_l2[variable].dtype.kind not in {"O", "S", "U"}
    }
    _timed(
        timings,
        "netcdf_write",
        lambda: _write_level2_atomically(ds_l2, output_path, encoding),
    )
    total_seconds = time.perf_counter() - total_started_at
    result = {
        "total_seconds": total_seconds,
        "stages_seconds": timings,
        "output_size_bytes": output_path.stat().st_size,
        "input_dataset_bytes": int(ds_l1.nbytes),
        "output_dataset_bytes": int(ds_l2.nbytes),
        "observed_materialized_arrays": materializations.array_count,
        "observed_materialized_bytes": materializations.array_bytes,
        "valid_retrieval_blocks": valid_retrieval_blocks,
        "product_completeness": product_contract.completeness.value,
        "processed_wavelengths": list(product_contract.processed_wavelengths),
        "failed_wavelengths": list(product_contract.failed_wavelengths),
    }
    ds_l1.close()
    ds_l2.close()
    return result


def _current_rss_bytes() -> int | None:
    statm_path = Path("/proc/self/statm")
    if not statm_path.exists():
        return None
    resident_pages = int(statm_path.read_text(encoding="utf-8").split()[1])
    return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _git_value(arguments: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def environment_snapshot() -> dict[str, Any]:
    """Collect the runtime and hardware fields needed to compare baselines."""
    package_versions: dict[str, str] = {}
    for package_name in _PACKAGE_NAMES:
        try:
            package_versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package_name] = "not-installed"

    cpu_model = None
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break

    memory_total_bytes = None
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_total_bytes = int(line.split()[1]) * 1024
                break

    git_status = _git_value(("status", "--porcelain"))
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_model": cpu_model,
        "memory_total_bytes": memory_total_bytes,
        "package_versions": package_versions,
        "thread_environment": {
            name: os.environ.get(name)
            for name in _THREAD_ENVIRONMENT
        },
        "git_revision": _git_value(("rev-parse", "HEAD")),
        "git_dirty": bool(git_status),
    }


def _worker(
    input_path: Path,
    scenario: Scenario,
    warmup_runs: int,
    product_mode: str,
) -> dict[str, Any]:
    rss_before_warmup = _current_rss_bytes()
    with tempfile.TemporaryDirectory(prefix="milgrau-level2-benchmark-worker-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        for warmup_index in range(warmup_runs):
            execute_pipeline(
                input_path,
                temporary_path / f"warmup-{warmup_index}.nc",
                scenario,
                product_mode,
            )
            gc.collect()
        measured = execute_pipeline(input_path, temporary_path / "measured.nc", scenario, product_mode)
    peak_rss_bytes = _peak_rss_bytes()
    measured.update(
        {
            "rss_before_warmup_bytes": rss_before_warmup,
            "peak_rss_bytes": peak_rss_bytes,
            "peak_rss_increment_bytes": (
                max(0, peak_rss_bytes - rss_before_warmup)
                if rss_before_warmup is not None
                else None
            ),
            "environment": environment_snapshot(),
        }
    )
    return measured


def _statistics(values: Sequence[int | float]) -> dict[str, float | int]:
    numeric = [float(value) for value in values]
    mean = statistics.fmean(numeric)
    return {
        "count": len(numeric),
        "median": statistics.median(numeric),
        "mean": mean,
        "min": min(numeric),
        "max": max(numeric),
        "coefficient_of_variation_percent": (
            100.0 * statistics.pstdev(numeric) / mean
            if len(numeric) > 1 and mean != 0.0
            else 0.0
        ),
    }


def summarize_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate measured repetitions without discarding raw results."""
    stage_names = sorted(
        {
            stage_name
            for run in runs
            for stage_name in run["stages_seconds"]
        }
    )
    scalar_metrics = (
        "total_seconds",
        "peak_rss_bytes",
        "peak_rss_increment_bytes",
        "output_size_bytes",
        "input_dataset_bytes",
        "output_dataset_bytes",
        "observed_materialized_arrays",
        "observed_materialized_bytes",
        "valid_retrieval_blocks",
    )
    summary = {
        metric: _statistics([run[metric] for run in runs if run[metric] is not None])
        for metric in scalar_metrics
    }
    summary["stages_seconds"] = {
        stage_name: _statistics([run["stages_seconds"][stage_name] for run in runs])
        for stage_name in stage_names
    }
    return summary


def _coordinator(
    scenario: Scenario,
    repetitions: int,
    warmup_runs: int,
    threads: int,
    product_mode: str,
) -> dict[str, Any]:
    worker_environment = os.environ.copy()
    for name in _THREAD_ENVIRONMENT:
        worker_environment[name] = str(threads)
    worker_environment["PYTHONHASHSEED"] = "0"

    runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="milgrau-level2-benchmark-input-") as temporary_directory:
        input_path = Path(temporary_directory) / f"{scenario.name}-level1.nc"
        fixture_scenario = (
            scenario
            if product_mode == "complete"
            else replace(scenario, wavelengths_nm=(scenario.wavelengths_nm[-1],))
        )
        fixture = build_synthetic_level1(fixture_scenario)
        fixture.to_netcdf(input_path)
        fixture.close()
        for _ in range(repetitions):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--scenario",
                scenario.name,
                "--warmup",
                str(warmup_runs),
                "--product-mode",
                product_mode,
                "--input",
                str(input_path),
            ]
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=worker_environment,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Benchmark worker failed with exit code "
                    f"{completed.returncode}: {completed.stderr.strip()}"
                )
            runs.append(json.loads(completed.stdout))

    return {
        "schema_version": 1,
        "scenario": {
            "name": scenario.name,
            "description": scenario.description,
            "n_profiles": scenario.n_profiles,
            "n_altitude": scenario.n_altitude,
            "wavelengths_nm": list(scenario.wavelengths_nm),
            "available_wavelengths_nm": list(fixture_scenario.wavelengths_nm),
            "product_mode": product_mode,
            "monte_carlo_iterations": scenario.monte_carlo_iterations,
            "profile_interval_minutes": scenario.profile_interval_minutes,
            "local_only": scenario.local_only,
        },
        "protocol": {
            "repetitions": repetitions,
            "warmup_runs_per_repetition": warmup_runs,
            "isolated_worker_per_repetition": True,
            "threads_per_runtime": threads,
            "peak_memory_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss in each isolated Linux worker",
            "materialization_method": "unique NumPy arrays observed at explicit stage boundaries; this is not an allocation event count",
        },
        "environment": runs[0]["environment"],
        "runs": [{key: value for key, value in run.items() if key != "environment"} for run in runs],
        "summary": summarize_runs(runs),
    }


def _human_summary(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    scenario = report["scenario"]
    total = summary["total_seconds"]
    peak = summary["peak_rss_bytes"]
    output = summary["output_size_bytes"]
    return "\n".join(
        (
            f"Scenario: {scenario['name']} / {scenario['product_mode']} "
            f"({scenario['n_profiles']} profiles, {scenario['n_altitude']} bins, "
            f"requested {scenario['wavelengths_nm']} nm, available {scenario['available_wavelengths_nm']} nm)",
            f"Total: median {total['median']:.3f} s; min {total['min']:.3f} s; "
            f"max {total['max']:.3f} s; CV {total['coefficient_of_variation_percent']:.2f}%",
            f"Peak RSS: median {peak['median'] / (1024 ** 2):.1f} MiB",
            f"Output: median {output['median'] / (1024 ** 2):.2f} MiB",
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="ci")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1, help="Full unmeasured pipeline runs in each worker.")
    parser.add_argument("--threads", type=int, default=1, help="Threads for Numba, OpenMP and BLAS workers.")
    parser.add_argument("--product-mode", choices=("complete", "partial"), default="complete")
    parser.add_argument("--output", type=Path, help="Optional path for the complete JSON report.")
    parser.add_argument("--json", action="store_true", help="Print the complete JSON report instead of a summary.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.warmup < 0:
        raise SystemExit("--warmup must be zero or greater")
    scenario = SCENARIOS[args.scenario]
    if args.worker:
        if args.input is None:
            raise SystemExit("--input is required in worker mode")
        print(json.dumps(_worker(args.input, scenario, args.warmup, args.product_mode), sort_keys=True))
        return 0
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be one or greater")
    if args.threads < 1:
        raise SystemExit("--threads must be one or greater")

    report = _coordinator(
        scenario,
        args.repetitions,
        args.warmup,
        args.threads,
        args.product_mode,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized if args.json else _human_summary(report), end="" if args.json else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
