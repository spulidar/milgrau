"""The SCI-004B benchmark is deterministic and never accesses public network."""

from benchmarks.benchmark_meteorology_acquisition import run_benchmark


def test_meteorology_acquisition_benchmark_is_local_and_reports_storage() -> None:
    result = run_benchmark(repetitions=1)
    assert result["network_calls"] == 0
    assert result["planned_hour_count"] < result["measurement_count"]
    assert result["planning_median_seconds"] >= 0.0
    assert result["cache_hit_median_seconds"] >= 0.0
    assert result["local_raw_normalization_median_seconds"] >= 0.0
    assert result["raw_mock_grib_bytes"] > 0
    assert result["normalized_netcdf_bytes"] > 0
    assert result["manifest_bytes"] > 0
