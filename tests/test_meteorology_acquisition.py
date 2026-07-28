"""SCI-004B provider failures, fallback semantics and operational result."""

from __future__ import annotations

from milgrau.meteorology import acquisition as acquisition_module
from milgrau.meteorology.acquisition import get_or_acquire_meteorology
from milgrau.meteorology.cache import Era5Release
from milgrau.meteorology.contracts import (
    FallbackFlag,
    PrimarySource,
    ProfileQuality,
)
from milgrau.meteorology.request import MeteorologyProvider
from milgrau.meteorology.results import (
    ProviderAcquisitionResult,
    ProviderStatus,
)
from tests.meteorology_acquisition_helpers import era5_decoded, meteorology_request


def _success(provider: str) -> ProviderAcquisitionResult:
    request = meteorology_request(
        __import__("pathlib").Path("/tmp/not-used"),
        provider=MeteorologyProvider.ERA5,
        radiosonde_nominal_times=(),
    )
    profile = era5_decoded().profiles(request, b"raw")[0]
    return ProviderAcquisitionResult(
        provider=provider,
        status=ProviderStatus.CACHE_HIT,
        profiles=(profile,),
        release=Era5Release.FINAL if provider == "era5" else None,
    )


def _failure(provider: str) -> ProviderAcquisitionResult:
    return ProviderAcquisitionResult(
        provider=provider,
        status=ProviderStatus.RECOVERABLE_FAILURE,
        error_code="offline",
        error_message=f"{provider} offline",
    )


def test_radiosonde_failure_preserves_era5(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(acquisition_module, "acquire_radiosondes", lambda *_a, **_k: _failure("radiosonde"))
    monkeypatch.setattr(acquisition_module, "acquire_era5", lambda *_a, **_k: _success("era5"))
    result = get_or_acquire_meteorology(meteorology_request(tmp_path))
    assert not result.radiosonde.available
    assert result.era5.available
    assert result.fallback_profile is None
    assert result.overall_status == "partial_observational_availability"


def test_era5_failure_preserves_radiosonde(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(acquisition_module, "acquire_radiosondes", lambda *_a, **_k: _success("radiosonde"))
    monkeypatch.setattr(acquisition_module, "acquire_era5", lambda *_a, **_k: _failure("era5"))
    result = get_or_acquire_meteorology(meteorology_request(tmp_path))
    assert result.radiosonde.available
    assert not result.era5.available
    assert result.fallback_profile is None


def test_both_fail_return_explicit_nonquantitative_standard_atmosphere(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(acquisition_module, "acquire_radiosondes", lambda *_a, **_k: _failure("radiosonde"))
    monkeypatch.setattr(acquisition_module, "acquire_era5", lambda *_a, **_k: _failure("era5"))
    result = get_or_acquire_meteorology(meteorology_request(tmp_path))
    fallback = result.fallback_profile
    assert fallback is not None
    assert result.overall_status == "fallback_standard_atmosphere"
    assert not result.usable_observational
    assert not result.quantitative_retrieval_allowed
    assert fallback.profile_quality is ProfileQuality.FALLBACK_DIAGNOSTIC
    assert not fallback.quantitative_retrieval_allowed
    assert set(fallback.primary_source_flag) == {int(PrimarySource.STANDARD_ATMOSPHERE)}
    assert set(fallback.fallback_flag) == {int(FallbackFlag.STANDARD_ATMOSPHERE)}
    assert any("USSA-1976" in warning for warning in result.warnings)


def test_unexpected_cache_write_failure_is_fatal(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        acquisition_module,
        "acquire_radiosondes",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(acquisition_module, "acquire_era5", lambda *_a, **_k: _failure("era5"))
    result = get_or_acquire_meteorology(meteorology_request(tmp_path))
    assert result.radiosonde.status is ProviderStatus.FATAL_FAILURE
    assert result.fatal_error == "disk full"
