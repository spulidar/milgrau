"""Minimal-hour ERA5 model-level acquisition, GRIB validation and normalization."""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from milgrau.meteorology.cache import (
    Era5Release,
    build_manifest,
    era5_cache_paths,
    publish_artifact,
    source_reference,
    validate_cached_artifact,
)
from milgrau.meteorology.era5_model_levels import normalize_era5_model_levels
from milgrau.meteorology.request import (
    AcquisitionMode,
    MeteorologyProvider,
    MeteorologyRequest,
)
from milgrau.meteorology.results import (
    AcquisitionMetrics,
    ProviderAcquisitionResult,
    ProviderStatus,
)
from milgrau.meteorology.snapshots import (
    profiles_from_netcdf_bytes,
    profiles_to_netcdf_bytes,
)

ERA5_DATASET = "reanalysis-era5-complete"
ERA5_MODEL_LEVEL_PARAMETER_IDS = "130/133"
ERA5_LEVEL_ONE_PARAMETER_IDS = "129/152"


@dataclass(frozen=True, slots=True)
class Era5DecodedData:
    """Validated four-corner model-level fields decoded from one GRIB artifact."""

    analysis_times: tuple[datetime, ...]
    coordinates_lat_lon: np.ndarray
    hybrid_a_pa: np.ndarray
    hybrid_b: np.ndarray
    temperature_k: np.ndarray
    specific_humidity_kg_kg: np.ndarray
    logarithm_surface_pressure: np.ndarray
    surface_geopotential_m2_s2: np.ndarray
    release: Era5Release

    def __post_init__(self) -> None:
        times = tuple(sorted(value.astimezone(UTC) for value in self.analysis_times))
        if not times or len(set(times)) != len(times):
            raise ValueError("ERA5 decoded analysis times must be unique and non-empty.")
        object.__setattr__(self, "analysis_times", times)
        arrays = {
            "coordinates_lat_lon": (self.coordinates_lat_lon, (4, 2)),
            "hybrid_a_pa": (self.hybrid_a_pa, (138,)),
            "hybrid_b": (self.hybrid_b, (138,)),
            "temperature_k": (self.temperature_k, (len(times), 137, 4)),
            "specific_humidity_kg_kg": (
                self.specific_humidity_kg_kg,
                (len(times), 137, 4),
            ),
            "logarithm_surface_pressure": (
                self.logarithm_surface_pressure,
                (len(times), 4),
            ),
            "surface_geopotential_m2_s2": (
                self.surface_geopotential_m2_s2,
                (len(times), 4),
            ),
        }
        for name, (value, shape) in arrays.items():
            array = np.array(value, dtype=np.float64, copy=True)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape {shape}.")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if not isinstance(self.release, Era5Release):
            raise TypeError("release must be Era5Release.")
        if np.any(self.temperature_k <= 0.0):
            raise ValueError("ERA5 temperature must be positive.")
        if np.any(
            (self.specific_humidity_kg_kg < 0.0)
            | (self.specific_humidity_kg_kg > 0.1)
        ):
            raise ValueError("ERA5 specific humidity must be within [0, 0.1].")
        if not np.all(np.diff(self.hybrid_a_pa + self.hybrid_b * 100_000.0) > 0.0):
            raise ValueError("ERA5 hybrid interfaces must increase top-down.")
        coordinates = {
            (float(latitude), float(longitude))
            for latitude, longitude in self.coordinates_lat_lon
        }
        if len(coordinates) != 4:
            raise ValueError("ERA5 decoded data must contain four distinct grid points.")

    def validate_request(
        self,
        request: MeteorologyRequest,
        expected_hours: tuple[datetime, ...],
    ) -> None:
        if self.analysis_times != tuple(expected_hours):
            raise ValueError(
                "ERA5 GRIB hours do not exactly match the requested minimal-hour set."
            )
        expected_points = np.asarray(request.era5_grid_points, dtype=np.float64)
        actual = np.asarray(self.coordinates_lat_lon, dtype=np.float64)
        expected_order = np.lexsort((expected_points[:, 1], expected_points[:, 0]))
        actual_order = np.lexsort((actual[:, 1], actual[:, 0]))
        if not np.allclose(
            actual[actual_order], expected_points[expected_order], rtol=0.0, atol=1e-7
        ):
            raise ValueError("ERA5 GRIB does not contain the requested four surrounding points.")

    def profiles(self, request: MeteorologyRequest, raw_payload: bytes):
        results = []
        for index, analysis_time in enumerate(self.analysis_times):
            reconstruction = normalize_era5_model_levels(
                hybrid_a_pa=self.hybrid_a_pa,
                hybrid_b=self.hybrid_b,
                temperature_k_by_level_corner=self.temperature_k[index],
                specific_humidity_by_level_corner=self.specific_humidity_kg_kg[index],
                logarithm_surface_pressure_by_corner=self.logarithm_surface_pressure[index],
                surface_geopotential_m2_s2_by_corner=self.surface_geopotential_m2_s2[index],
                corner_coordinates_lat_lon=self.coordinates_lat_lon,
                target_latitude_deg_north=request.latitude_deg_north,
                target_longitude_deg_east=request.longitude_deg_east,
                analysis_time=analysis_time,
                dataset_id=f"ERA5-L137-{self.release.value}",
                raw_snapshot=raw_payload,
                require_137_levels=True,
            )
            results.append(reconstruction.profile)
        return tuple(results)


@dataclass(frozen=True, slots=True)
class Era5Download:
    payload: bytes
    retries: int = 0

    def __post_init__(self) -> None:
        if not self.payload:
            raise ValueError("ERA5 download payload cannot be empty.")
        if (
            isinstance(self.retries, bool)
            or not isinstance(self.retries, int)
            or self.retries < 0
        ):
            raise ValueError("ERA5 download retries must be a non-negative integer.")


class Era5TransportError(RuntimeError):
    """Redacted transport failure that preserves retry telemetry."""

    def __init__(self, message: str, *, retries: int):
        super().__init__(message)
        self.retries = retries


Era5Decoder = Callable[[bytes], Era5DecodedData]
Era5Transport = Callable[
    [tuple[Mapping[str, object], ...], MeteorologyRequest], bytes | Era5Download
]


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _request_for_dates_and_times(
    request: MeteorologyRequest,
    dates: tuple[str, ...],
    times: tuple[str, ...],
    *,
    levelist: str,
    parameters: str,
) -> dict[str, object]:
    return {
        "class": "ea",
        "date": "/".join(dates),
        "levelist": levelist,
        "levtype": "ml",
        "param": parameters,
        "step": "0",
        "stream": "oper",
        "time": "/".join(times),
        "type": "an",
        "grid": f"{request.era5_grid_degrees:g}/{request.era5_grid_degrees:g}",
        "area": "/".join(
            f"{value:g}" for value in request.era5_area_north_west_south_east
        ),
    }


def build_era5_cds_requests(
    request: MeteorologyRequest,
    hours: tuple[datetime, ...],
) -> tuple[dict[str, object], ...]:
    """Build exact MARS rectangles without requesting unused date/time products.

    MARS treats ``date`` and ``time`` as a Cartesian product. Days with different
    needed hour sets are therefore split into the minimum number of exact
    rectangles. Each rectangle has an L137 T/q request and a level-1 z/lnsp
    request, concatenated into one monthly raw GRIB artifact.
    """
    if not hours or len({(value.year, value.month) for value in hours}) != 1:
        raise ValueError("ERA5 CDS requests must contain one non-empty calendar month.")
    by_date: dict[str, set[str]] = {}
    for value in hours:
        timestamp = value.astimezone(UTC)
        if timestamp.minute or timestamp.second or timestamp.microsecond:
            raise ValueError("ERA5 request hours must be exact UTC hours.")
        by_date.setdefault(timestamp.strftime("%Y-%m-%d"), set()).add(
            timestamp.strftime("%H:%M:%S")
        )
    dates_by_times: dict[tuple[str, ...], list[str]] = {}
    for date, time_values in by_date.items():
        dates_by_times.setdefault(tuple(sorted(time_values)), []).append(date)
    temporal_rectangles = tuple(
        (tuple(sorted(dates)), time_values)
        for time_values, dates in sorted(dates_by_times.items())
    )
    field_groups = (
        ("1/to/137", ERA5_MODEL_LEVEL_PARAMETER_IDS),
        ("1", ERA5_LEVEL_ONE_PARAMETER_IDS),
    )
    return tuple(
        _request_for_dates_and_times(
            request,
            dates,
            times,
            levelist=levelist,
            parameters=parameters,
        )
        for dates, times in temporal_rectangles
        for levelist, parameters in field_groups
    )


def _redact_error(message: str) -> str:
    redacted = str(message)
    for name in ("CDSAPI_KEY", "CDSAPI_TOKEN", "CDSAPI_URL"):
        value = os.environ.get(name)
        if value:
            replacement = "[REDACTED_URL]" if name == "CDSAPI_URL" else "[REDACTED]"
            redacted = redacted.replace(value, replacement)
    redacted = re.sub(
        r"(?i)\b(bearer)\s+\S+",
        r"\1 [REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)\b(api[_-]?key|key|token|password|authorization)(\s*[:=]\s*)\S+",
        r"\1\2[REDACTED]",
        redacted,
    )
    return redacted


def validate_cds_credentials() -> str:
    """Confirm an official home config or explicit environment adaptation exists."""
    configured_file = Path.home() / ".cdsapirc"
    environment_key = os.environ.get("CDSAPI_KEY") or os.environ.get("CDSAPI_TOKEN")
    environment_url = os.environ.get("CDSAPI_URL")
    if configured_file.is_file() and configured_file.stat().st_size > 0:
        return "home_cdsapirc"
    if environment_key and environment_url:
        return "environment"
    raise RuntimeError(
        "CDS credentials are unavailable. Configure the official ~/.cdsapirc file "
        "or both CDSAPI_URL and CDSAPI_KEY/CDSAPI_TOKEN."
    )


def default_cds_transport(
    requests: tuple[Mapping[str, object], ...],
    request: MeteorologyRequest,
) -> Era5Download:
    """Retrieve exact MARS subrequests and concatenate their GRIB messages."""
    credential_source = validate_cds_credentials()
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError(
            "ERA5 acquisition requires the optional meteorology dependency cdsapi."
        ) from exc

    client_kwargs: dict[str, object] = {
        "timeout": request.timeout_seconds,
        "retry_max": 0,
        "quiet": True,
    }
    if credential_source == "environment":
        client_kwargs["url"] = os.environ["CDSAPI_URL"]
        client_kwargs["key"] = os.environ.get("CDSAPI_KEY") or os.environ["CDSAPI_TOKEN"]
    try:
        client = cdsapi.Client(**client_kwargs)
    except Exception as exc:
        raise RuntimeError(_redact_error(str(exc))) from exc

    combined = bytearray()
    retry_count = 0
    for request_payload in requests:
        last_error: Exception | None = None
        for attempt in range(1, request.max_retries + 1):
            descriptor, temporary_name = tempfile.mkstemp(suffix=".grib")
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            try:
                client.retrieve(ERA5_DATASET, dict(request_payload), str(temporary_path))
                chunk = temporary_path.read_bytes()
                if not chunk:
                    raise OSError("CDS returned an empty GRIB artifact.")
                combined.extend(chunk)
                retry_count += attempt - 1
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < request.max_retries:
                    logging.getLogger(__name__).warning(
                        "ERA5 CDS request retry %d/%d after %s.",
                        attempt,
                        request.max_retries,
                        type(exc).__name__,
                    )
                if attempt == request.max_retries:
                    raise Era5TransportError(
                        _redact_error(str(exc)),
                        retries=retry_count + attempt - 1,
                    ) from exc
            finally:
                temporary_path.unlink(missing_ok=True)
        if last_error is not None:
            raise RuntimeError(_redact_error(str(last_error)))
    return Era5Download(payload=bytes(combined), retries=retry_count)


def _grib_datetime(eccodes, handle) -> datetime:
    date_value = int(eccodes.codes_get(handle, "dataDate"))
    time_value = int(eccodes.codes_get(handle, "dataTime"))
    return datetime.strptime(
        f"{date_value:08d}{time_value:04d}", "%Y%m%d%H%M"
    ).replace(tzinfo=UTC)


def _grib_release(eccodes, handle) -> Era5Release:
    for key in ("experimentVersionNumber", "expver"):
        try:
            value = str(eccodes.codes_get(handle, key)).lstrip("0") or "0"
            if value == "5":
                return Era5Release.ERA5T_PROVISIONAL
            if value == "1":
                return Era5Release.FINAL
        except Exception:
            continue
    raise ValueError("ERA5 GRIB does not expose expver release metadata.")


def decode_era5_grib(payload: bytes) -> Era5DecodedData:
    """Decode and validate the four-point L137 contract using optional ecCodes."""
    try:
        import eccodes
    except ImportError as exc:
        raise RuntimeError(
            "ERA5 GRIB normalization requires the optional meteorology dependency eccodes."
        ) from exc
    if not payload:
        raise ValueError("ERA5 GRIB payload is empty.")
    fields: dict[tuple[datetime, str, int], np.ndarray] = {}
    coordinates: np.ndarray | None = None
    hybrid_a: np.ndarray | None = None
    hybrid_b: np.ndarray | None = None
    releases: set[Era5Release] = set()
    stream = io.BytesIO(payload)
    while True:
        handle = eccodes.codes_grib_new_from_file(stream)
        if handle is None:
            break
        try:
            short_name = str(eccodes.codes_get(handle, "shortName"))
            if short_name not in {"t", "q", "lnsp", "z"}:
                continue
            analysis_time = _grib_datetime(eccodes, handle)
            releases.add(_grib_release(eccodes, handle))
            latitudes = np.asarray(
                eccodes.codes_get_array(handle, "latitudes"), dtype=np.float64
            )
            longitudes = np.asarray(
                eccodes.codes_get_array(handle, "longitudes"), dtype=np.float64
            )
            longitudes = np.where(longitudes > 180.0, longitudes - 360.0, longitudes)
            values = np.asarray(
                eccodes.codes_get_array(handle, "values"), dtype=np.float64
            )
            current_coordinates = np.column_stack((latitudes, longitudes))
            if current_coordinates.shape != (4, 2) or values.shape != (4,):
                raise ValueError("ERA5 GRIB messages must contain exactly four grid points.")
            order = np.lexsort((current_coordinates[:, 1], current_coordinates[:, 0]))
            current_coordinates = current_coordinates[order]
            values = values[order]
            if coordinates is None:
                coordinates = current_coordinates
            elif not np.allclose(coordinates, current_coordinates, rtol=0.0, atol=1e-7):
                raise ValueError("ERA5 GRIB messages use inconsistent grid points.")
            level = (
                int(eccodes.codes_get(handle, "level"))
                if short_name in {"t", "q"}
                else 0
            )
            key = (analysis_time, short_name, level)
            if key in fields:
                raise ValueError(f"Duplicate ERA5 GRIB field: {key}.")
            fields[key] = values
            if hybrid_a is None and short_name in {"t", "q"}:
                try:
                    pv = np.asarray(
                        eccodes.codes_get_array(handle, "pv"), dtype=np.float64
                    )
                except Exception:
                    pv = np.empty(0)
                if pv.shape == (276,):
                    hybrid_a = pv[:138]
                    hybrid_b = pv[138:]
        finally:
            eccodes.codes_release(handle)
    if coordinates is None or hybrid_a is None or hybrid_b is None:
        raise ValueError("ERA5 GRIB lacks four-point coordinates or L137 hybrid coefficients.")
    if len(releases) != 1:
        raise ValueError(
            "One raw ERA5 artifact cannot mix final and ERA5T fields without per-message splitting."
        )
    times = tuple(sorted({key[0] for key in fields}))
    temperature = np.empty((len(times), 137, 4), dtype=np.float64)
    humidity = np.empty_like(temperature)
    lnsp = np.empty((len(times), 4), dtype=np.float64)
    surface_geopotential = np.empty_like(lnsp)
    for time_index, analysis_time in enumerate(times):
        for level in range(1, 138):
            try:
                temperature[time_index, level - 1] = fields[
                    (analysis_time, "t", level)
                ]
                humidity[time_index, level - 1] = fields[
                    (analysis_time, "q", level)
                ]
            except KeyError as exc:
                raise ValueError(
                    f"ERA5 GRIB is missing t/q at {analysis_time.isoformat()} level {level}."
                ) from exc
        try:
            lnsp[time_index] = fields[(analysis_time, "lnsp", 0)]
            surface_geopotential[time_index] = fields[(analysis_time, "z", 0)]
        except KeyError as exc:
            raise ValueError(
                f"ERA5 GRIB is missing lnsp/surface geopotential at {analysis_time.isoformat()}."
            ) from exc
    return Era5DecodedData(
        analysis_times=times,
        coordinates_lat_lon=coordinates,
        hybrid_a_pa=hybrid_a,
        hybrid_b=hybrid_b,
        temperature_k=temperature,
        specific_humidity_kg_kg=humidity,
        logarithm_surface_pressure=lnsp,
        surface_geopotential_m2_s2=surface_geopotential,
        release=next(iter(releases)),
    )


def _warning(logger: logging.Logger | None, message: str) -> None:
    if logger is not None:
        logger.warning(message)


def _info(logger: logging.Logger | None, message: str) -> None:
    if logger is not None:
        logger.info(message)


def _cache_candidate(
    request: MeteorologyRequest,
    hours: tuple[datetime, ...],
    release: Era5Release,
    *,
    normalized: bool,
):
    paths = era5_cache_paths(request, hours, release, normalized=normalized)
    request_payload = request.artifact_request_payload(
        provider=MeteorologyProvider.ERA5,
        timestamps=hours,
    )
    validation = validate_cached_artifact(
        paths,
        expected_request=request_payload,
        expected_release=release,
    )
    return paths, request_payload, validation


def acquire_era5(
    request: MeteorologyRequest,
    *,
    transport: Era5Transport | None = None,
    decoder: Era5Decoder | None = None,
    logger: logging.Logger | None = None,
    refresh_provisional: bool = False,
) -> ProviderAcquisitionResult:
    """Acquire exact monthly hour groups, preferring final over provisional cache."""
    started = time.perf_counter()
    transport = default_cds_transport if transport is None else transport
    decoder = decode_era5_grib if decoder is None else decoder
    profiles = []
    raw_files = []
    normalized_files = []
    manifests = []
    inventory: list[dict[str, object]] = []
    warnings: list[str] = []
    cache_hits = 0
    cache_misses = 0
    bytes_downloaded = 0
    final_status = ProviderStatus.CACHE_HIT
    result_releases: set[Era5Release] = set()
    retry_count = 0
    provisional_cache_seen = False

    _info(
        logger,
        "ERA5 acquisition started: mode=%s hours=%s cache=%s"
        % (
            request.mode.value,
            ",".join(value.isoformat() for value in request.era5_hours),
            request.cache_directory,
        ),
    )

    for _, hours in request.era5_month_groups:
        releases = (Era5Release.FINAL,) + (
            (Era5Release.ERA5T_PROVISIONAL,) if request.allow_era5t else ()
        )
        selected_release: Era5Release | None = None
        selected_normalized = None
        selected_request_payload = None
        selected_validation = None
        for release in releases:
            paths, request_payload, validation = _cache_candidate(
                request, hours, release, normalized=True
            )
            if (
                validation.valid
                and not (
                    release is Era5Release.ERA5T_PROVISIONAL
                    and refresh_provisional
                    and request.mode is not AcquisitionMode.CACHE_ONLY
                )
            ):
                selected_release = release
                selected_normalized = paths
                selected_request_payload = request_payload
                selected_validation = validation
                break
            if (
                validation.valid
                and release is Era5Release.ERA5T_PROVISIONAL
                and refresh_provisional
            ):
                provisional_cache_seen = True
            if validation.reason not in {"artifact_missing", "manifest_missing"}:
                message = (
                    "Corrupt ERA5 normalized cache rejected: "
                    f"{paths.artifact} ({validation.reason})"
                )
                warnings.append(message)
                _warning(logger, message)
        if selected_validation is not None and selected_validation.valid:
            try:
                cached_profiles = profiles_from_netcdf_bytes(
                    selected_normalized.artifact.read_bytes()
                )
                if tuple(profile.nominal_time for profile in cached_profiles) != hours:
                    raise ValueError(
                        "Normalized ERA5 cache does not match its analysis hours."
                    )
            except Exception:
                message = (
                    "Corrupt ERA5 normalized cache rejected: "
                    f"{selected_normalized.artifact} (normalized_contract_invalid)"
                )
                warnings.append(message)
                _warning(logger, message)
            else:
                profiles.extend(cached_profiles)
                normalized_files.append(selected_normalized.artifact)
                manifests.append(selected_normalized.manifest)
                cache_hits += 1
                result_releases.add(selected_release)
                _info(
                    logger,
                    f"ERA5 normalized cache hit ({selected_release.value}): "
                    f"{selected_normalized.artifact}",
                )
                if selected_release.provisional:
                    message = "ERA5T provisional meteorology loaded from cache."
                    warnings.append(message)
                    _warning(logger, message)
                inventory.append(
                    {
                        "provider": "era5",
                        "release": selected_release.value,
                        "cache_status": "normalized_hit",
                        "hours_utc": [value.isoformat() for value in hours],
                        "path": selected_normalized.artifact.as_posix(),
                    }
                )
                continue

        cache_misses += 1
        _info(
            logger,
            "ERA5 normalized cache miss for hours "
            + ",".join(value.isoformat() for value in hours),
        )
        if request.mode is AcquisitionMode.CACHE_ONLY:
            return ProviderAcquisitionResult(
                provider="era5",
                status=ProviderStatus.RECOVERABLE_FAILURE,
                profiles=tuple(profiles),
                raw_files=tuple(raw_files),
                normalized_files=tuple(normalized_files),
                manifest_files=tuple(manifests),
                metrics=AcquisitionMetrics(
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                    duration_seconds=time.perf_counter() - started,
                ),
                warnings=tuple(warnings),
                error_code="cache_miss",
                error_message="Exact ERA5 normalized cache is unavailable.",
                inventory=tuple(inventory),
            )

        raw_payload = None
        raw_paths = None
        raw_validation = None
        decoded = None
        downloaded_this_artifact = False
        for release in releases:
            candidate_paths, request_payload, candidate_validation = _cache_candidate(
                request, hours, release, normalized=False
            )
            if (
                candidate_validation.valid
                and not (
                    release is Era5Release.ERA5T_PROVISIONAL
                    and refresh_provisional
                )
            ):
                raw_payload = candidate_paths.artifact.read_bytes()
                raw_paths = candidate_paths
                raw_validation = candidate_validation
                selected_request_payload = request_payload
                selected_release = release
                cache_hits += 1
                _info(
                    logger,
                    f"ERA5 raw cache hit ({release.value}): {candidate_paths.artifact}",
                )
                break
            if (
                candidate_validation.valid
                and release is Era5Release.ERA5T_PROVISIONAL
                and refresh_provisional
            ):
                provisional_cache_seen = True
            if candidate_validation.reason not in {
                "artifact_missing",
                "manifest_missing",
            }:
                message = (
                    "Corrupt ERA5 raw cache rejected; downloading exact request again: "
                    f"{candidate_paths.artifact} ({candidate_validation.reason})"
                )
                warnings.append(message)
                _warning(logger, message)

        if raw_payload is None:
            cds_requests = build_era5_cds_requests(request, hours)
            _info(
                logger,
                f"ERA5 download started: {len(cds_requests)} exact request(s), "
                f"max_retries={request.max_retries}, timeout={request.timeout_seconds:g}s",
            )
            try:
                downloaded = transport(cds_requests, request)
                if isinstance(downloaded, Era5Download):
                    raw_payload = downloaded.payload
                    retry_count += downloaded.retries
                else:
                    raw_payload = downloaded
                bytes_downloaded += len(raw_payload)
                downloaded_this_artifact = True
                decoded = decoder(raw_payload)
                decoded.validate_request(request, hours)
            except Exception as exc:
                if isinstance(exc, Era5TransportError):
                    retry_count += exc.retries
                return ProviderAcquisitionResult(
                    provider="era5",
                    status=ProviderStatus.RECOVERABLE_FAILURE,
                    profiles=tuple(profiles),
                    raw_files=tuple(raw_files),
                    normalized_files=tuple(normalized_files),
                    manifest_files=tuple(manifests),
                    metrics=AcquisitionMetrics(
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        bytes_downloaded=bytes_downloaded,
                        duration_seconds=time.perf_counter() - started,
                        retries=retry_count,
                    ),
                    warnings=tuple(warnings),
                    error_code="era5_download_or_decode_failed",
                    error_message=_redact_error(str(exc)),
                    inventory=tuple(inventory),
                )
            selected_release = decoded.release
            if selected_release.provisional and not request.allow_era5t:
                return ProviderAcquisitionResult(
                    provider="era5",
                    status=ProviderStatus.RECOVERABLE_FAILURE,
                    error_code="era5t_not_allowed",
                    error_message="CDS returned ERA5T but the request disallows provisional data.",
                    metrics=AcquisitionMetrics(
                        cache_misses=cache_misses,
                        bytes_downloaded=bytes_downloaded,
                        duration_seconds=time.perf_counter() - started,
                        retries=retry_count,
                    ),
                )
            raw_paths = era5_cache_paths(
                request, hours, selected_release, normalized=False
            )
            selected_request_payload = request.artifact_request_payload(
                provider=MeteorologyProvider.ERA5,
                timestamps=hours,
            )
            raw_manifest = build_manifest(
                paths=raw_paths,
                artifact_bytes=raw_payload,
                provider="ecmwf_cds",
                dataset=ERA5_DATASET,
                request_payload=selected_request_payload,
                timestamps=hours,
                area=request.era5_area_north_west_south_east,
                variables=request.era5_variables,
                levels=request.era5_model_levels,
                release=selected_release,
                normalizer=None,
                normalizer_version=None,
                raw_payload_kind="grib",
                dependency_versions={"cdsapi": _package_version("cdsapi")},
            )
            publish_artifact(raw_paths, raw_payload, raw_manifest)
            raw_validation = validate_cached_artifact(
                raw_paths,
                expected_request=selected_request_payload,
                expected_release=selected_release,
            )
            if not raw_validation.valid:
                raise OSError("Published ERA5 raw cache failed integrity validation.")
            final_status = ProviderStatus.DOWNLOADED
            _info(
                logger,
                f"ERA5 GRIB published ({selected_release.value}, {len(raw_payload)} bytes): "
                f"{raw_paths.artifact}",
            )

        try:
            if decoded is None:
                decoded = decoder(raw_payload)
                decoded.validate_request(request, hours)
            if decoded.release is not selected_release:
                raise ValueError("Decoded ERA5 release does not match raw cache identity.")
            month_profiles = decoded.profiles(request, raw_payload)
            normalized_payload = profiles_to_netcdf_bytes(
                month_profiles,
                cache_metadata={
                    "snapshot_schema": "milgrau-normalized-era5-l137-v1",
                    "era5_release": selected_release.value,
                    "meteorology_provisional": int(selected_release.provisional),
                },
            )
            normalized_paths = era5_cache_paths(
                request, hours, selected_release, normalized=True
            )
            normalized_manifest = build_manifest(
                paths=normalized_paths,
                artifact_bytes=normalized_payload,
                provider="ecmwf_cds",
                dataset=ERA5_DATASET,
                request_payload=selected_request_payload,
                timestamps=hours,
                area=request.era5_area_north_west_south_east,
                variables=request.era5_variables,
                levels=request.era5_model_levels,
                release=selected_release,
                normalizer="normalize_era5_model_levels",
                normalizer_version=month_profiles[0].normalizer_version,
                source_files=(source_reference(raw_paths, raw_validation.manifest),),
                raw_payload_kind="grib",
                dependency_versions={
                    "eccodes": _package_version("eccodes"),
                    "xarray": _package_version("xarray"),
                },
            )
            publish_artifact(normalized_paths, normalized_payload, normalized_manifest)
        except Exception as exc:
            return ProviderAcquisitionResult(
                provider="era5",
                status=ProviderStatus.RECOVERABLE_FAILURE,
                profiles=tuple(profiles),
                raw_files=tuple((*raw_files, raw_paths.artifact)),
                normalized_files=tuple(normalized_files),
                manifest_files=tuple((*manifests, raw_paths.manifest)),
                metrics=AcquisitionMetrics(
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                    bytes_downloaded=bytes_downloaded,
                    duration_seconds=time.perf_counter() - started,
                    retries=retry_count,
                ),
                warnings=tuple(warnings),
                error_code="era5_normalization_failed",
                error_message=_redact_error(str(exc)),
                inventory=tuple(inventory),
            )

        profiles.extend(month_profiles)
        raw_files.append(raw_paths.artifact)
        normalized_files.append(normalized_paths.artifact)
        manifests.extend((raw_paths.manifest, normalized_paths.manifest))
        result_releases.add(selected_release)
        inventory.append(
            {
                "provider": "era5",
                "release": selected_release.value,
                "meteorology_provisional": selected_release.provisional,
                "cache_status": "downloaded_and_normalized"
                if downloaded_this_artifact
                else "normalized_from_raw_cache",
                "hours_utc": [value.isoformat() for value in hours],
                "raw_path": raw_paths.artifact.as_posix(),
                "normalized_path": normalized_paths.artifact.as_posix(),
            }
        )
        if selected_release.provisional:
            message = "ERA5T provisional meteorology acquired; reprocessing will be required after final replacement."
            warnings.append(message)
            _warning(logger, message)
        elif provisional_cache_seen and refresh_provisional:
            message = (
                "ERA5 final data supersede a retained ERA5T provisional cache entry; "
                "the meteorology context changed and later L2 reprocessing is required."
            )
            warnings.append(message)
            _warning(logger, message)
        _info(
            logger,
            f"ERA5 normalization complete ({len(month_profiles)} profiles): "
            f"{normalized_paths.artifact}",
        )
        if final_status is ProviderStatus.CACHE_HIT:
            final_status = ProviderStatus.NORMALIZED

    release = next(iter(result_releases)) if len(result_releases) == 1 else None
    return ProviderAcquisitionResult(
        provider="era5",
        status=final_status,
        profiles=tuple(profiles),
        raw_files=tuple(raw_files),
        normalized_files=tuple(normalized_files),
        manifest_files=tuple(manifests),
        release=release,
        metrics=AcquisitionMetrics(
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            bytes_downloaded=bytes_downloaded,
            duration_seconds=time.perf_counter() - started,
            retries=retry_count,
        ),
        warnings=tuple(warnings),
        inventory=tuple(inventory),
    )
