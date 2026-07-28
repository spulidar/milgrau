"""Structured operational results for meteorology acquisition."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from milgrau.meteorology.cache import Era5Release
from milgrau.meteorology.contracts import AtmosphericProfile


class ProviderStatus(StrEnum):
    CACHE_HIT = "cache_hit"
    DOWNLOADED = "downloaded"
    NORMALIZED = "normalized"
    UNAVAILABLE = "unavailable"
    RECOVERABLE_FAILURE = "recoverable_failure"
    FATAL_FAILURE = "fatal_failure"
    FALLBACK_STANDARD_ATMOSPHERE = "fallback_standard_atmosphere"


@dataclass(frozen=True, slots=True)
class AcquisitionMetrics:
    cache_hits: int = 0
    cache_misses: int = 0
    bytes_downloaded: int = 0
    duration_seconds: float = 0.0
    retries: int = 0


@dataclass(frozen=True, slots=True)
class ProviderAcquisitionResult:
    provider: str
    status: ProviderStatus
    profiles: tuple[AtmosphericProfile, ...] = ()
    raw_files: tuple[Path, ...] = ()
    normalized_files: tuple[Path, ...] = ()
    manifest_files: tuple[Path, ...] = ()
    release: Era5Release | None = None
    metrics: AcquisitionMetrics = AcquisitionMetrics()
    warnings: tuple[str, ...] = ()
    error_code: str | None = None
    error_message: str | None = None
    inventory: tuple[Mapping[str, object], ...] = ()

    @property
    def available(self) -> bool:
        return bool(self.profiles) and self.status not in {
            ProviderStatus.UNAVAILABLE,
            ProviderStatus.RECOVERABLE_FAILURE,
            ProviderStatus.FATAL_FAILURE,
        }


@dataclass(frozen=True, slots=True)
class MeteorologyAcquisitionResult:
    radiosonde: ProviderAcquisitionResult
    era5: ProviderAcquisitionResult
    fallback_profile: AtmosphericProfile | None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    fatal_error: str | None = None

    @property
    def observational_provider_count(self) -> int:
        return int(self.radiosonde.available) + int(self.era5.available)

    @property
    def usable_observational(self) -> bool:
        return self.observational_provider_count > 0

    @property
    def quantitative_retrieval_allowed(self) -> bool:
        profiles = (*self.radiosonde.profiles, *self.era5.profiles)
        return (
            self.fallback_profile is None
            and any(profile.quantitative_retrieval_allowed for profile in profiles)
        )

    @property
    def overall_status(self) -> str:
        if self.fatal_error is not None:
            return "fatal_failure"
        if self.observational_provider_count == 2:
            return "both_observational_providers_available"
        if self.observational_provider_count == 1:
            return "partial_observational_availability"
        if self.fallback_profile is not None:
            return ProviderStatus.FALLBACK_STANDARD_ATMOSPHERE.value
        return "unavailable"

    @property
    def files(self) -> tuple[Path, ...]:
        return (
            *self.radiosonde.raw_files,
            *self.radiosonde.normalized_files,
            *self.radiosonde.manifest_files,
            *self.era5.raw_files,
            *self.era5.normalized_files,
            *self.era5.manifest_files,
        )
