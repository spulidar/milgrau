"""SCI-004B deterministic cache identity, manifests and integrity."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from milgrau.meteorology.cache import (
    Era5Release,
    build_manifest,
    era5_cache_paths,
    publish_artifact,
    radiosonde_cache_paths,
    validate_cached_artifact,
)
from milgrau.meteorology.request import MeteorologyProvider
from tests.meteorology_acquisition_helpers import ANALYSIS_TIME, meteorology_request


def test_release_and_artifact_kind_create_distinct_immutable_identities(tmp_path) -> None:
    request = meteorology_request(tmp_path)
    hours = request.era5_hours
    final_raw = era5_cache_paths(request, hours, Era5Release.FINAL, normalized=False)
    provisional_raw = era5_cache_paths(
        request, hours, Era5Release.ERA5T_PROVISIONAL, normalized=False
    )
    final_normalized = era5_cache_paths(
        request, hours, Era5Release.FINAL, normalized=True
    )
    assert len({final_raw.identity, provisional_raw.identity, final_normalized.identity}) == 3
    assert final_raw.artifact != provisional_raw.artifact


def test_manifest_requires_exact_hash_size_identity_and_request(tmp_path) -> None:
    request = meteorology_request(tmp_path)
    paths = radiosonde_cache_paths(request, ANALYSIS_TIME, normalized=False)
    request_payload = request.artifact_request_payload(
        provider=MeteorologyProvider.RADIOSONDE,
        timestamps=(ANALYSIS_TIME,),
    )
    payload = b"complete raw response"
    manifest = build_manifest(
        paths=paths,
        artifact_bytes=payload,
        provider="wyoming_siphon",
        dataset="83779",
        request_payload=request_payload,
        timestamps=(ANALYSIS_TIME,),
        area=None,
        variables=("pressure", "height"),
        levels=(),
        release=None,
        normalizer=None,
        normalizer_version=None,
        raw_payload_kind="http_response",
    )
    publish_artifact(paths, payload, manifest)
    assert validate_cached_artifact(
        paths, expected_request=request_payload, expected_release=None
    ).valid

    paths.artifact.write_bytes(b"truncated")
    validation = validate_cached_artifact(
        paths, expected_request=request_payload, expected_release=None
    )
    assert not validation.valid
    assert validation.reason in {"size_mismatch", "sha256_mismatch"}


def test_file_without_manifest_is_invalid(tmp_path) -> None:
    request = meteorology_request(tmp_path)
    paths = radiosonde_cache_paths(request, ANALYSIS_TIME, normalized=False)
    paths.artifact.parent.mkdir(parents=True)
    paths.artifact.write_bytes(b"orphan")
    validation = validate_cached_artifact(
        paths,
        expected_request=request.artifact_request_payload(
            provider=MeteorologyProvider.RADIOSONDE,
            timestamps=(ANALYSIS_TIME,),
        ),
        expected_release=None,
    )
    assert not validation.valid
    assert validation.reason == "manifest_missing"


def test_manifest_builder_rejects_credential_fields(tmp_path) -> None:
    request = meteorology_request(tmp_path)
    paths = radiosonde_cache_paths(request, ANALYSIS_TIME, normalized=False)
    with pytest.raises(ValueError, match="credential"):
        build_manifest(
            paths=paths,
            artifact_bytes=b"payload",
            provider="wyoming_siphon",
            dataset="83779",
            request_payload={"api_token": "must-not-persist"},
            timestamps=(ANALYSIS_TIME,),
            area=None,
            variables=(),
            levels=(),
            release=None,
            normalizer=None,
            normalizer_version=None,
        )


def test_failed_atomic_replace_preserves_previous_valid_artifact(tmp_path, monkeypatch) -> None:
    request = meteorology_request(tmp_path)
    paths = radiosonde_cache_paths(request, ANALYSIS_TIME, normalized=False)
    request_payload = request.artifact_request_payload(
        provider=MeteorologyProvider.RADIOSONDE,
        timestamps=(ANALYSIS_TIME,),
    )

    def manifest_for(payload: bytes):
        return build_manifest(
            paths=paths,
            artifact_bytes=payload,
            provider="wyoming_siphon",
            dataset="83779",
            request_payload=request_payload,
            timestamps=(ANALYSIS_TIME,),
            area=None,
            variables=(),
            levels=(),
            release=None,
            normalizer=None,
            normalizer_version=None,
        )

    publish_artifact(paths, b"old-valid", manifest_for(b"old-valid"))
    import milgrau.meteorology.cache as cache_module

    original_replace = cache_module.os.replace

    def fail_artifact_replace(source, target):
        if target == paths.artifact:
            raise OSError("interrupted")
        return original_replace(source, target)

    monkeypatch.setattr(cache_module.os, "replace", fail_artifact_replace)
    with pytest.raises(OSError, match="interrupted"):
        publish_artifact(paths, b"new-content", manifest_for(b"new-content"))
    assert paths.artifact.read_bytes() == b"old-valid"
    temporary_files = list(paths.artifact.parent.glob("*.tmp"))
    assert temporary_files == []


def test_failed_manifest_replace_restores_previous_valid_pair(tmp_path, monkeypatch) -> None:
    request = meteorology_request(tmp_path)
    paths = radiosonde_cache_paths(request, ANALYSIS_TIME, normalized=False)
    request_payload = request.artifact_request_payload(
        provider=MeteorologyProvider.RADIOSONDE,
        timestamps=(ANALYSIS_TIME,),
    )

    def manifest_for(payload: bytes):
        return build_manifest(
            paths=paths,
            artifact_bytes=payload,
            provider="wyoming_siphon",
            dataset="83779",
            request_payload=request_payload,
            timestamps=(ANALYSIS_TIME,),
            area=None,
            variables=(),
            levels=(),
            release=None,
            normalizer=None,
            normalizer_version=None,
        )

    old_manifest = manifest_for(b"old-valid")
    publish_artifact(paths, b"old-valid", old_manifest)
    import milgrau.meteorology.cache as cache_module

    original_replace = cache_module.os.replace

    def fail_manifest_replace(source, target):
        if target == paths.manifest and str(source).endswith(".tmp"):
            raise OSError("manifest interrupted")
        return original_replace(source, target)

    monkeypatch.setattr(cache_module.os, "replace", fail_manifest_replace)
    with pytest.raises(OSError, match="manifest interrupted"):
        publish_artifact(paths, b"new-content", manifest_for(b"new-content"))
    assert paths.artifact.read_bytes() == b"old-valid"
    assert validate_cached_artifact(
        paths,
        expected_request=request_payload,
        expected_release=None,
    ).valid
    assert list(paths.artifact.parent.glob("*.tmp")) == []
    assert list(paths.manifest.parent.glob("*.tmp")) == []
    assert list(paths.artifact.parent.glob("*.bak")) == []
    assert list(paths.manifest.parent.glob("*.bak")) == []


def test_manifest_is_canonical_json_without_authenticated_urls(tmp_path) -> None:
    request = meteorology_request(tmp_path)
    paths = radiosonde_cache_paths(request, ANALYSIS_TIME, normalized=False)
    request_payload = request.artifact_request_payload(
        provider=MeteorologyProvider.RADIOSONDE,
        timestamps=(ANALYSIS_TIME,),
    )
    manifest = build_manifest(
        paths=paths,
        artifact_bytes=b"payload",
        provider="wyoming_siphon",
        dataset="83779",
        request_payload=request_payload,
        timestamps=(ANALYSIS_TIME,),
        area=None,
        variables=(),
        levels=(),
        release=None,
        normalizer=None,
        normalizer_version=None,
    )
    assert json.loads(json.dumps(manifest))["request"] == request_payload
    assert "http" not in json.dumps(manifest).lower()
