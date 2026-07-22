"""Tests for the reproducible Level 2 benchmark protocol."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from benchmarks.benchmark_level2 import (
    SCENARIOS,
    Scenario,
    build_synthetic_level1,
    execute_pipeline,
    summarize_runs,
)


def test_benchmark_scenarios_cover_ci_typical_and_local_large_day() -> None:
    assert SCENARIOS["ci"].local_only is False
    assert len(SCENARIOS["typical"].wavelengths_nm) > 1
    assert SCENARIOS["typical"].monte_carlo_iterations > 1
    assert SCENARIOS["large"].local_only is True
    assert SCENARIOS["large"].n_profiles == 24 * 60 // 5


def test_synthetic_level1_fixture_is_deterministic() -> None:
    first = build_synthetic_level1(SCENARIOS["ci"])
    second = build_synthetic_level1(SCENARIOS["ci"])

    xr.testing.assert_identical(first, second)
    assert first.sizes == {"time": 3, "channel": 4, "altitude": 240}
    assert first.attrs["Benchmark_scenario"] == "ci"


def test_pipeline_benchmark_measures_all_explicit_stages(tmp_path: Path) -> None:
    scenario = Scenario(
        name="test",
        description="Minimal test workload.",
        n_profiles=3,
        n_altitude=240,
        wavelengths_nm=(532,),
        monte_carlo_iterations=2,
    )
    input_path = tmp_path / "synthetic-level1.nc"
    output_path = tmp_path / "synthetic-level2.nc"
    build_synthetic_level1(scenario).to_netcdf(input_path)

    result = execute_pipeline(input_path, output_path, scenario)

    assert set(result["stages_seconds"]) == {
        "input_open_load_validation",
        "selection_and_blocking",
        "gluing",
        "molecular_model",
        "rayleigh_kfs",
        "result_assembly",
        "dataset_assembly",
        "output_validation",
        "netcdf_write",
    }
    assert result["total_seconds"] >= sum(result["stages_seconds"].values())
    assert result["valid_retrieval_blocks"] == 1
    assert result["observed_materialized_arrays"] > 0
    assert result["observed_materialized_bytes"] > result["output_dataset_bytes"]
    assert result["output_size_bytes"] == output_path.stat().st_size


def test_pipeline_benchmark_measures_partial_product_without_nan_wavelength(tmp_path: Path) -> None:
    scenario = Scenario(
        name="test-partial",
        description="Minimal partial test workload.",
        n_profiles=3,
        n_altitude=240,
        wavelengths_nm=(355, 532),
        monte_carlo_iterations=2,
    )
    available = Scenario(
        name="test-partial",
        description=scenario.description,
        n_profiles=scenario.n_profiles,
        n_altitude=scenario.n_altitude,
        wavelengths_nm=(532,),
        monte_carlo_iterations=scenario.monte_carlo_iterations,
    )
    input_path = tmp_path / "synthetic-level1.nc"
    output_path = tmp_path / "synthetic-level2.nc"
    build_synthetic_level1(available).to_netcdf(input_path)

    result = execute_pipeline(input_path, output_path, scenario, "partial")

    assert result["product_completeness"] == "partial"
    assert result["processed_wavelengths"] == [532]
    assert result["failed_wavelengths"] == [355]
    with xr.open_dataset(output_path) as dataset:
        assert dataset["wavelength"].values.tolist() == [532]


def test_benchmark_summary_records_variability() -> None:
    runs = [
        {
            "total_seconds": total,
            "stages_seconds": {"gluing": total / 2},
            "peak_rss_bytes": 100 + index,
            "peak_rss_increment_bytes": 20 + index,
            "output_size_bytes": 50,
            "input_dataset_bytes": 40,
            "output_dataset_bytes": 80,
            "observed_materialized_arrays": 10,
            "observed_materialized_bytes": 120,
            "valid_retrieval_blocks": 1,
        }
        for index, total in enumerate((1.0, 1.2, 0.8))
    ]

    summary = summarize_runs(runs)

    assert summary["total_seconds"]["median"] == 1.0
    assert summary["total_seconds"]["coefficient_of_variation_percent"] > 0.0
    assert summary["stages_seconds"]["gluing"]["count"] == 3
