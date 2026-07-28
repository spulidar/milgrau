"""SCI-004B offline characterization and cache-first Wyoming acquisition."""

from __future__ import annotations

import json

import pandas as pd

from milgrau.meteorology.cache import (
    CachePaths,
    publish_artifact,
    radiosonde_cache_paths,
    sha256_bytes,
)
from milgrau.meteorology.radiosonde_acquisition import (
    RAW_CANONICAL_DATAFRAME,
    acquire_radiosondes,
    canonical_dataframe_snapshot,
    dataframe_from_canonical_snapshot,
)
from milgrau.meteorology.request import AcquisitionMode, MeteorologyProvider
from milgrau.meteorology.results import ProviderStatus
from tests.meteorology_acquisition_helpers import (
    ANALYSIS_TIME,
    meteorology_request,
    radiosonde_table,
    radiosonde_transport,
)


def test_canonical_snapshot_preserves_all_columns_units_dtypes_and_values() -> None:
    table = radiosonde_table()
    payload = canonical_dataframe_snapshot(table)
    decoded = json.loads(payload)
    restored = dataframe_from_canonical_snapshot(payload)
    assert decoded["columns"] == list(table.columns)
    assert decoded["units"] == table.units
    assert list(restored.columns) == list(table.columns)
    pd.testing.assert_series_equal(
        restored["pressure"], table["pressure"], check_names=False
    )
    assert pd.to_datetime(restored["time"]).tolist() == pd.to_datetime(table["time"]).tolist()


def test_auto_downloads_once_then_uses_exact_normalized_cache(tmp_path) -> None:
    calls = []

    def transport(time, station):
        calls.append((time, station))
        return radiosonde_transport(time, station)

    request = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.RADIOSONDE,
    )
    first = acquire_radiosondes(request, transport=transport)
    second = acquire_radiosondes(
        request,
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("network used")),
    )
    assert first.status is ProviderStatus.DOWNLOADED
    assert second.status is ProviderStatus.CACHE_HIT
    assert len(calls) == 1
    assert first.profiles[0].station_or_dataset_id == "83779"
    assert len(first.raw_files) == len(first.normalized_files) == 1
    manifest = json.loads(first.manifest_files[0].read_text(encoding="utf-8"))
    assert manifest["raw_payload_kind"] == RAW_CANONICAL_DATAFRAME


def test_cache_only_miss_never_calls_transport(tmp_path) -> None:
    request = meteorology_request(
        tmp_path,
        mode=AcquisitionMode.CACHE_ONLY,
        provider=MeteorologyProvider.RADIOSONDE,
    )
    result = acquire_radiosondes(
        request,
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("network used")),
    )
    assert result.status is ProviderStatus.RECOVERABLE_FAILURE
    assert result.error_code == "cache_miss"


def test_cache_only_reuses_artifact_created_in_auto_without_network(tmp_path) -> None:
    auto = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.RADIOSONDE,
    )
    assert acquire_radiosondes(auto, transport=radiosonde_transport).available
    cache_only = meteorology_request(
        tmp_path,
        mode=AcquisitionMode.CACHE_ONLY,
        provider=MeteorologyProvider.RADIOSONDE,
    )
    result = acquire_radiosondes(
        cache_only,
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("network used")),
    )
    assert result.status is ProviderStatus.CACHE_HIT


def test_corrupt_normalized_cache_is_rebuilt_from_valid_raw_in_auto(tmp_path) -> None:
    request = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.RADIOSONDE,
    )
    first = acquire_radiosondes(request, transport=radiosonde_transport)
    normalized = first.normalized_files[0]
    normalized.write_bytes(b"corrupt")
    rebuilt = acquire_radiosondes(
        request,
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("network used")),
    )
    assert rebuilt.status is ProviderStatus.NORMALIZED
    assert rebuilt.profiles
    assert normalized.read_bytes() != b"corrupt"
    assert any("Corrupt radiosonde normalized" in message for message in rebuilt.warnings)


def test_hash_valid_but_structurally_invalid_normalized_cache_is_rebuilt(tmp_path) -> None:
    request = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.RADIOSONDE,
    )
    first = acquire_radiosondes(request, transport=radiosonde_transport)
    manifest_path = first.manifest_files[1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    invalid_payload = b"not-a-netcdf"
    manifest["size_bytes"] = len(invalid_payload)
    manifest["sha256"] = sha256_bytes(invalid_payload)
    publish_artifact(
        CachePaths(
            artifact=first.normalized_files[0],
            manifest=manifest_path,
            identity=manifest["identity"],
        ),
        invalid_payload,
        manifest,
    )

    rebuilt = acquire_radiosondes(
        request,
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("network used")),
    )
    assert rebuilt.status is ProviderStatus.NORMALIZED
    assert rebuilt.profiles
    assert any("normalized_contract_invalid" in warning for warning in rebuilt.warnings)


def test_corrupt_normalized_cache_fails_in_cache_only_even_with_valid_raw(tmp_path) -> None:
    auto_request = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.RADIOSONDE,
    )
    first = acquire_radiosondes(auto_request, transport=radiosonde_transport)
    first.normalized_files[0].write_bytes(b"corrupt")
    cache_request = meteorology_request(
        tmp_path,
        mode=AcquisitionMode.CACHE_ONLY,
        provider=MeteorologyProvider.RADIOSONDE,
    )
    result = acquire_radiosondes(
        cache_request,
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("network used")),
    )
    assert result.status is ProviderStatus.RECOVERABLE_FAILURE
    assert result.error_code == "cache_miss"


def test_corrupt_raw_cache_is_downloaded_again_in_auto(tmp_path) -> None:
    request = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.RADIOSONDE,
    )
    first = acquire_radiosondes(request, transport=radiosonde_transport)
    first.normalized_files[0].unlink()
    first.raw_files[0].write_bytes(b"truncated")
    calls = []

    def transport(time, station):
        calls.append(time)
        return radiosonde_transport(time, station)

    result = acquire_radiosondes(request, transport=transport)
    assert result.available
    assert calls == [ANALYSIS_TIME]
    assert any("Corrupt radiosonde raw" in message for message in result.warnings)


def test_transport_failure_is_recoverable_and_does_not_create_partial_cache(tmp_path) -> None:
    request = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.RADIOSONDE,
    )
    result = acquire_radiosondes(
        request,
        transport=lambda *_args: (_ for _ in ()).throw(OSError("offline")),
    )
    paths = radiosonde_cache_paths(request, ANALYSIS_TIME, normalized=False)
    assert result.status is ProviderStatus.RECOVERABLE_FAILURE
    assert result.error_code == "siphon_unavailable"
    assert not paths.artifact.exists()
    assert not paths.manifest.exists()
