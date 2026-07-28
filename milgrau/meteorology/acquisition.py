"""Provider-independent cache-first meteorology orchestration."""

from __future__ import annotations

import logging

import numpy as np

from milgrau.meteorology.era5_acquisition import (
    Era5Decoder,
    Era5Transport,
    acquire_era5,
)
from milgrau.meteorology.radiosonde_acquisition import (
    RadiosondeTransport,
    acquire_radiosondes,
)
from milgrau.meteorology.request import MeteorologyProvider, MeteorologyRequest
from milgrau.meteorology.results import (
    ProviderAcquisitionResult,
    ProviderStatus,
    MeteorologyAcquisitionResult,
)
from milgrau.meteorology.standard_atmosphere import (
    build_standard_atmosphere_profile,
)


def _not_requested(provider: str) -> ProviderAcquisitionResult:
    return ProviderAcquisitionResult(
        provider=provider,
        status=ProviderStatus.UNAVAILABLE,
        error_code="not_requested",
        error_message=f"{provider} was not requested.",
    )


def _fatal(provider: str, exc: Exception) -> ProviderAcquisitionResult:
    return ProviderAcquisitionResult(
        provider=provider,
        status=ProviderStatus.FATAL_FAILURE,
        error_code="cache_or_contract_failure",
        error_message=str(exc),
    )


def get_or_acquire_meteorology(
    request: MeteorologyRequest,
    *,
    radiosonde_transport: RadiosondeTransport | None = None,
    era5_transport: Era5Transport | None = None,
    era5_decoder: Era5Decoder | None = None,
    logger: logging.Logger | None = None,
    refresh_provisional: bool = False,
) -> MeteorologyAcquisitionResult:
    """Acquire providers independently and return an explicit diagnostic fallback."""
    if not isinstance(request, MeteorologyRequest):
        raise TypeError("request must be MeteorologyRequest.")
    wants_radiosonde = request.provider in {
        MeteorologyProvider.RADIOSONDE,
        MeteorologyProvider.BOTH,
    }
    wants_era5 = request.provider in {
        MeteorologyProvider.ERA5,
        MeteorologyProvider.BOTH,
    }

    try:
        radiosonde = (
            acquire_radiosondes(
                request,
                transport=radiosonde_transport,
                logger=logger,
            )
            if wants_radiosonde
            else _not_requested("radiosonde")
        )
    except Exception as exc:
        radiosonde = _fatal("radiosonde", exc)
    try:
        era5 = (
            acquire_era5(
                request,
                transport=era5_transport,
                decoder=era5_decoder,
                logger=logger,
                refresh_provisional=refresh_provisional,
            )
            if wants_era5
            else _not_requested("era5")
        )
    except Exception as exc:
        era5 = _fatal("era5", exc)

    warnings = [*radiosonde.warnings, *era5.warnings]
    available_count = int(radiosonde.available) + int(era5.available)
    requested_count = int(wants_radiosonde) + int(wants_era5)
    fallback = None
    if available_count == 0:
        fallback = build_standard_atmosphere_profile(
            np.asarray(request.fallback_altitudes_m, dtype=np.float64),
            nominal_time=request.measurement_timestamps[0],
            latitude_deg_north=request.latitude_deg_north,
            longitude_deg_east=request.longitude_deg_east,
        )
        if requested_count == 2:
            unavailable_scope = "Both requested meteorology providers are"
        else:
            unavailable_scope = "The requested meteorology provider is"
        message = (
            f"{unavailable_scope} unavailable; using explicit USSA-1976 "
            "diagnostic fallback with quantitative retrieval disabled."
        )
        warnings.append(message)
        if logger is not None:
            logger.warning(message)
    elif available_count < requested_count:
        missing = "radiosonde" if not radiosonde.available and wants_radiosonde else "ERA5"
        message = f"Only one meteorology provider is available; {missing} acquisition failed."
        warnings.append(message)
        if logger is not None:
            logger.warning(message)

    fatal_errors = [
        result.error_message
        for result in (radiosonde, era5)
        if result.status is ProviderStatus.FATAL_FAILURE and result.error_message
    ]
    return MeteorologyAcquisitionResult(
        radiosonde=radiosonde,
        era5=era5,
        fallback_profile=fallback,
        warnings=tuple(warnings),
        fatal_error="; ".join(fatal_errors) if fatal_errors else None,
    )
