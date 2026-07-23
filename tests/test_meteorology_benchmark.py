"""The SCI-004A benchmark is deterministic and strictly local."""

from benchmarks.benchmark_meteorology import run_benchmark


def test_meteorology_benchmark_covers_all_small_profile_stages() -> None:
    report = run_benchmark(repetitions=1)

    assert report["network_calls"] == 0
    assert report["radiosonde_normalization"]["median_seconds"] > 0.0
    assert report["era5_l137_reconstruction"]["median_seconds"] > 0.0
    assert report["hybrid_blend_385_bins"]["median_seconds"] > 0.0
    assert report["profile_sizes"]["era5_levels"] == 137
    assert report["profile_sizes"]["hybrid_levels"] == 385
