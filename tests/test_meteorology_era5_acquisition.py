"""SCI-004B offline ERA5/ERA5T request, cache and normalization behavior."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import pytest

from milgrau.meteorology.cache import (
    CachePaths,
    Era5Release,
    publish_artifact,
    sha256_bytes,
)
from milgrau.meteorology.era5_acquisition import (
    Era5Download,
    Era5TransportError,
    _redact_error,
    acquire_era5,
    build_era5_cds_requests,
    decode_era5_grib,
    validate_cds_credentials,
)
from milgrau.meteorology.request import AcquisitionMode, MeteorologyProvider
from milgrau.meteorology.results import ProviderStatus
from tests.meteorology_acquisition_helpers import era5_decoded, meteorology_request


def test_cds_request_contains_only_l137_contract_and_four_point_area(tmp_path) -> None:
    request = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.ERA5,
        radiosonde_nominal_times=(),
    )
    payloads = build_era5_cds_requests(request, request.era5_hours)
    assert {(payload["levelist"], payload["param"]) for payload in payloads} == {
        ("1/to/137", "130/133"),
        ("1", "129/152"),
    }
    for payload in payloads:
        assert payload["levtype"] == "ml"
        assert payload["grid"] == "0.25/0.25"
        assert payload["area"] == "-23.5/-46.75/-23.75/-46.5"
        assert payload["time"] == "12:00:00/13:00:00"
        assert payload["date"] == "2026-07-05"


def test_sparse_days_are_split_into_exact_mars_rectangles_without_extra_hours(tmp_path) -> None:
    timestamps = (
        datetime(2026, 7, 5, 12, tzinfo=UTC),
        datetime(2026, 7, 6, 13, tzinfo=UTC),
    )
    request = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.ERA5,
        radiosonde_nominal_times=(),
        measurement_timestamps=timestamps,
    )
    payloads = build_era5_cds_requests(request, request.era5_hours)
    assert len(payloads) == 4
    selected = {
        (payload["date"], payload["time"])
        for payload in payloads
    }
    assert selected == {
        ("2026-07-05", "12:00:00"),
        ("2026-07-06", "13:00:00"),
    }


def test_auto_downloads_normalizes_then_reuses_cache_without_network(tmp_path) -> None:
    request = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.ERA5,
        radiosonde_nominal_times=(),
    )
    calls = []

    def transport(payloads, _request):
        calls.append(payloads)
        return b"GRIB-mock-final"

    first = acquire_era5(
        request,
        transport=transport,
        decoder=lambda _payload: era5_decoded(),
    )
    second = acquire_era5(
        request,
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("network used")),
        decoder=lambda _payload: (_ for _ in ()).throw(AssertionError("decoder used")),
    )
    assert first.status is ProviderStatus.DOWNLOADED
    assert first.release is Era5Release.FINAL
    assert len(first.profiles) == 2
    assert second.status is ProviderStatus.CACHE_HIT
    assert len(calls) == 1


def test_download_retry_metric_is_propagated(tmp_path) -> None:
    request = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.ERA5,
        radiosonde_nominal_times=(),
    )
    result = acquire_era5(
        request,
        transport=lambda *_args: Era5Download(b"GRIB-after-retries", retries=2),
        decoder=lambda _payload: era5_decoded(),
    )
    assert result.metrics.retries == 2
    assert result.metrics.bytes_downloaded == len(b"GRIB-after-retries")


def test_failed_download_retry_metric_is_preserved(tmp_path) -> None:
    request = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.ERA5,
        radiosonde_nominal_times=(),
    )
    result = acquire_era5(
        request,
        transport=lambda *_args: (_ for _ in ()).throw(
            Era5TransportError("offline", retries=2)
        ),
    )
    assert result.status is ProviderStatus.RECOVERABLE_FAILURE
    assert result.metrics.retries == 2


def test_cache_only_miss_never_accesses_network(tmp_path) -> None:
    request = meteorology_request(
        tmp_path,
        mode=AcquisitionMode.CACHE_ONLY,
        provider=MeteorologyProvider.ERA5,
        radiosonde_nominal_times=(),
    )
    result = acquire_era5(
        request,
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("network used")),
    )
    assert result.status is ProviderStatus.RECOVERABLE_FAILURE
    assert result.error_code == "cache_miss"


def test_cache_only_reuses_auto_artifact_without_network(tmp_path) -> None:
    auto = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.ERA5,
        radiosonde_nominal_times=(),
    )
    assert acquire_era5(
        auto,
        transport=lambda *_args: b"GRIB-mock-final",
        decoder=lambda _payload: era5_decoded(),
    ).available
    cache_only = meteorology_request(
        tmp_path,
        mode=AcquisitionMode.CACHE_ONLY,
        provider=MeteorologyProvider.ERA5,
        radiosonde_nominal_times=(),
    )
    result = acquire_era5(
        cache_only,
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("network used")),
    )
    assert result.status is ProviderStatus.CACHE_HIT


def test_corrupt_normalized_cache_rebuilds_from_raw_but_cache_only_fails(tmp_path) -> None:
    auto_request = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.ERA5,
        radiosonde_nominal_times=(),
    )
    first = acquire_era5(
        auto_request,
        transport=lambda *_args: b"GRIB-mock-final",
        decoder=lambda _payload: era5_decoded(),
    )
    first.normalized_files[0].write_bytes(b"corrupt")
    rebuilt = acquire_era5(
        auto_request,
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("network used")),
        decoder=lambda _payload: era5_decoded(),
    )
    assert rebuilt.status is ProviderStatus.NORMALIZED
    assert any("Corrupt ERA5 normalized" in message for message in rebuilt.warnings)

    rebuilt.normalized_files[0].write_bytes(b"corrupt-again")
    cache_request = meteorology_request(
        tmp_path,
        mode=AcquisitionMode.CACHE_ONLY,
        provider=MeteorologyProvider.ERA5,
        radiosonde_nominal_times=(),
    )
    failed = acquire_era5(cache_request)
    assert failed.status is ProviderStatus.RECOVERABLE_FAILURE
    assert failed.error_code == "cache_miss"


def test_hash_valid_but_structurally_invalid_era5_cache_is_rebuilt(tmp_path) -> None:
    request = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.ERA5,
        radiosonde_nominal_times=(),
    )
    first = acquire_era5(
        request,
        transport=lambda *_args: b"GRIB-mock-final",
        decoder=lambda _payload: era5_decoded(),
    )
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

    rebuilt = acquire_era5(
        request,
        transport=lambda *_args: (_ for _ in ()).throw(AssertionError("network used")),
        decoder=lambda _payload: era5_decoded(),
    )
    assert rebuilt.status is ProviderStatus.NORMALIZED
    assert rebuilt.profiles
    assert any("normalized_contract_invalid" in warning for warning in rebuilt.warnings)


def test_corrupt_raw_cache_downloads_again(tmp_path) -> None:
    request = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.ERA5,
        radiosonde_nominal_times=(),
    )
    first = acquire_era5(
        request,
        transport=lambda *_args: b"GRIB-mock-final",
        decoder=lambda _payload: era5_decoded(),
    )
    first.normalized_files[0].unlink()
    first.raw_files[0].write_bytes(b"corrupt")
    calls = []
    result = acquire_era5(
        request,
        transport=lambda *_args: calls.append(True) or b"GRIB-new-final",
        decoder=lambda _payload: era5_decoded(),
    )
    assert result.available
    assert calls == [True]
    assert any("Corrupt ERA5 raw" in message for message in result.warnings)


def test_provisional_and_later_final_have_distinct_cache_identity(tmp_path) -> None:
    request = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.ERA5,
        radiosonde_nominal_times=(),
    )
    provisional = acquire_era5(
        request,
        transport=lambda *_args: b"GRIB-mock-era5t",
        decoder=lambda _payload: era5_decoded(
            release=Era5Release.ERA5T_PROVISIONAL
        ),
    )
    final = acquire_era5(
        request,
        refresh_provisional=True,
        transport=lambda *_args: b"GRIB-mock-final",
        decoder=lambda _payload: era5_decoded(release=Era5Release.FINAL),
    )
    assert provisional.release is Era5Release.ERA5T_PROVISIONAL
    assert final.release is Era5Release.FINAL
    assert provisional.raw_files[0] != final.raw_files[0]
    assert provisional.raw_files[0].exists()
    assert final.raw_files[0].exists()
    provisional_manifest = json.loads(
        provisional.manifest_files[0].read_text(encoding="utf-8")
    )
    final_manifest = json.loads(final.manifest_files[0].read_text(encoding="utf-8"))
    assert provisional_manifest["meteorology_provisional"] is True
    assert final_manifest["meteorology_provisional"] is False
    assert any("supersede" in message for message in final.warnings)


def test_era5t_is_rejected_when_request_disallows_it(tmp_path) -> None:
    request = meteorology_request(
        tmp_path,
        provider=MeteorologyProvider.ERA5,
        radiosonde_nominal_times=(),
        allow_era5t=False,
    )
    result = acquire_era5(
        request,
        transport=lambda *_args: b"GRIB-mock-era5t",
        decoder=lambda _payload: era5_decoded(
            release=Era5Release.ERA5T_PROVISIONAL
        ),
    )
    assert result.status is ProviderStatus.RECOVERABLE_FAILURE
    assert result.error_code == "era5t_not_allowed"


def test_credentials_are_never_required_for_mock_or_cache_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CDSAPI_KEY", "top-secret")
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.delenv("CDSAPI_TOKEN", raising=False)
    monkeypatch.setattr("pathlib.Path.is_file", lambda self: False)
    with pytest.raises(RuntimeError) as error:
        validate_cds_credentials()
    assert "top-secret" not in str(error.value)


def test_error_redaction_masks_common_credential_shapes(monkeypatch) -> None:
    monkeypatch.setenv("CDSAPI_KEY", "exact-secret")
    monkeypatch.setenv("CDSAPI_URL", "https://user:secret@example.invalid/api")
    redacted = _redact_error(
        "key=exact-secret token:another-secret Authorization=Bearer-value "
        "Bearer third-secret api_key=fourth-secret "
        "https://user:secret@example.invalid/api"
    )
    assert "exact-secret" not in redacted
    assert "another-secret" not in redacted
    assert "Bearer-value" not in redacted
    assert "third-secret" not in redacted
    assert "fourth-secret" not in redacted
    assert "user:secret" not in redacted


class _FakeEccodes:
    def __init__(self, messages):
        self.messages = list(messages)
        self.index = 0

    def codes_grib_new_from_file(self, _stream):
        if self.index == len(self.messages):
            return None
        message = self.messages[self.index]
        self.index += 1
        return message

    @staticmethod
    def codes_get(message, key):
        return message[key]

    @staticmethod
    def codes_get_array(message, key):
        return message[key]

    @staticmethod
    def codes_release(_message):
        return None


def _fake_grib_messages(*, omit_q_level: int | None = None):
    decoded = era5_decoded(
        times=(datetime(2026, 7, 5, 12, tzinfo=UTC),),
    )
    common = {
        "dataDate": 20260705,
        "dataTime": 1200,
        "experimentVersionNumber": 1,
        "latitudes": decoded.coordinates_lat_lon[:, 0],
        "longitudes": decoded.coordinates_lat_lon[:, 1],
    }
    messages = []
    pv = list(decoded.hybrid_a_pa) + list(decoded.hybrid_b)
    for level in range(1, 138):
        messages.append(
            {
                **common,
                "shortName": "t",
                "level": level,
                "values": decoded.temperature_k[0, level - 1],
                "pv": pv if level == 1 else [],
            }
        )
        if level != omit_q_level:
            messages.append(
                {
                    **common,
                    "shortName": "q",
                    "level": level,
                    "values": decoded.specific_humidity_kg_kg[0, level - 1],
                    "pv": [],
                }
            )
    messages.extend(
        [
            {
                **common,
                "shortName": "lnsp",
                "values": decoded.logarithm_surface_pressure[0],
            },
            {
                **common,
                "shortName": "z",
                "values": decoded.surface_geopotential_m2_s2[0],
            },
        ]
    )
    return messages


def test_eccodes_reader_validates_and_decodes_complete_l137_contract(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "eccodes", _FakeEccodes(_fake_grib_messages()))
    decoded = decode_era5_grib(b"fake-grib-stream")
    assert decoded.release is Era5Release.FINAL
    assert decoded.temperature_k.shape == (1, 137, 4)
    assert decoded.specific_humidity_kg_kg.shape == (1, 137, 4)
    assert decoded.logarithm_surface_pressure.shape == (1, 4)


def test_eccodes_reader_rejects_missing_model_level_variable(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "eccodes",
        _FakeEccodes(_fake_grib_messages(omit_q_level=57)),
    )
    with pytest.raises(ValueError, match="level 57"):
        decode_era5_grib(b"fake-grib-stream")
