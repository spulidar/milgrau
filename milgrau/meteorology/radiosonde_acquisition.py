"""Cache-first Siphon/Wyoming acquisition preserving the complete response."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Callable

import numpy as np
import pandas as pd

from milgrau.meteorology.cache import (
    build_manifest,
    publish_artifact,
    radiosonde_cache_paths,
    source_reference,
    validate_cached_artifact,
)
from milgrau.meteorology.radiosonde import normalize_wyoming_radiosonde
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

RAW_HTTP_RESPONSE = "http_response"
RAW_CANONICAL_DATAFRAME = "canonical_dataframe_snapshot"


@dataclass(frozen=True, slots=True)
class RadiosondeRawPayload:
    payload: bytes
    payload_kind: str
    table: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.payload_kind not in {RAW_HTTP_RESPONSE, RAW_CANONICAL_DATAFRAME}:
            raise ValueError("Unknown radiosonde raw payload kind.")
        if not self.payload:
            raise ValueError("Radiosonde raw payload cannot be empty.")


RadiosondeTransport = Callable[[datetime, str], RadiosondeRawPayload]


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def canonical_dataframe_snapshot(table: pd.DataFrame) -> bytes:
    """Preserve every column, dtype, unit and value in deterministic JSON."""
    units = getattr(table, "units", {})
    records = json.loads(
        table.to_json(
            orient="records",
            date_format="iso",
            date_unit="us",
            double_precision=15,
        )
    )
    payload = {
        "snapshot_schema": "wyoming-canonical-dataframe-v1",
        "columns": [str(column) for column in table.columns],
        "dtypes": {str(column): str(table[column].dtype) for column in table.columns},
        "units": {str(key): value for key, value in sorted(dict(units).items())},
        "records": records,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def dataframe_from_canonical_snapshot(payload: bytes) -> pd.DataFrame:
    decoded = json.loads(payload.decode("utf-8"))
    if decoded.get("snapshot_schema") != "wyoming-canonical-dataframe-v1":
        raise ValueError("Unsupported Wyoming canonical dataframe snapshot.")
    table = pd.DataFrame(decoded["records"], columns=decoded["columns"])
    for column, dtype in decoded["dtypes"].items():
        if dtype.startswith("datetime64"):
            table[column] = pd.to_datetime(table[column])
        elif dtype not in {"object", "string"}:
            try:
                table[column] = table[column].astype(dtype)
            except (TypeError, ValueError):
                pass
    object.__setattr__(table, "units", decoded.get("units", {}))
    return table


class _ReplayWyomingUpperAir:
    """Use Siphon's parser once over already-downloaded response text."""

    @staticmethod
    def parse(raw_text: str, nominal_time: datetime, station_id: str) -> pd.DataFrame:
        from siphon.simplewebservice.wyoming import WyomingUpperAir

        class ReplayEndpoint(WyomingUpperAir):
            def __init__(self, response_text: str):
                self.response_text = response_text

            def _get_data_raw(self, time, site_id, recalc=False):
                return self.response_text

        return ReplayEndpoint(raw_text)._get_data(nominal_time, station_id)


def default_siphon_transport(
    nominal_time: datetime,
    station_id: str,
) -> RadiosondeRawPayload:
    """Download original server text once and parse it with the installed Siphon."""
    from siphon.simplewebservice.wyoming import WyomingUpperAir

    endpoint = WyomingUpperAir()
    raw_text = endpoint._get_data_raw(nominal_time, station_id)
    table = _ReplayWyomingUpperAir.parse(raw_text, nominal_time, station_id)
    return RadiosondeRawPayload(
        payload=raw_text.encode("utf-8"),
        payload_kind=RAW_HTTP_RESPONSE,
        table=table,
    )


def _table_from_raw(
    payload: bytes,
    payload_kind: str,
    nominal_time: datetime,
    station_id: str,
) -> pd.DataFrame:
    if payload_kind == RAW_HTTP_RESPONSE:
        return _ReplayWyomingUpperAir.parse(payload.decode("utf-8"), nominal_time, station_id)
    if payload_kind == RAW_CANONICAL_DATAFRAME:
        return dataframe_from_canonical_snapshot(payload)
    raise ValueError(f"Unsupported raw_payload_kind: {payload_kind!r}")


def _first_finite(table: pd.DataFrame, column: str, default: float) -> float:
    if column not in table:
        return default
    values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite[0]) if finite.size else default


def _observation_time(table: pd.DataFrame, nominal_time: datetime) -> datetime:
    if "time" not in table or table.empty:
        return nominal_time
    value = pd.to_datetime(table["time"].iloc[0]).to_pydatetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _normalize(
    raw: bytes,
    payload_kind: str,
    request: MeteorologyRequest,
    nominal_time: datetime,
    *,
    table: pd.DataFrame | None = None,
):
    parsed = (
        table.copy()
        if table is not None
        else _table_from_raw(raw, payload_kind, nominal_time, request.radiosonde_station_id)
    )
    return normalize_wyoming_radiosonde(
        parsed,
        nominal_time=nominal_time,
        observation_time=_observation_time(parsed, nominal_time),
        station_id=request.radiosonde_station_id,
        latitude_deg_north=_first_finite(
            parsed, "latitude", request.latitude_deg_north
        ),
        longitude_deg_east=_first_finite(
            parsed, "longitude", request.longitude_deg_east
        ),
        raw_snapshot=raw,
    ).profile


def _warning(logger: logging.Logger | None, message: str) -> None:
    if logger is not None:
        logger.warning(message)


def _info(logger: logging.Logger | None, message: str) -> None:
    if logger is not None:
        logger.info(message)


def acquire_radiosondes(
    request: MeteorologyRequest,
    *,
    transport: RadiosondeTransport | None = None,
    logger: logging.Logger | None = None,
) -> ProviderAcquisitionResult:
    """Return exact cached soundings or acquire only explicitly requested nominals."""
    started = time.perf_counter()
    transport = default_siphon_transport if transport is None else transport
    profiles = []
    raw_files = []
    normalized_files = []
    manifests = []
    inventory: list[dict[str, object]] = []
    cache_hits = 0
    cache_misses = 0
    bytes_downloaded = 0
    retries = 0
    final_status = ProviderStatus.CACHE_HIT
    warnings: list[str] = []

    _info(
        logger,
        "Radiosonde acquisition started: mode=%s station=%s times=%s cache=%s"
        % (
            request.mode.value,
            request.radiosonde_station_id,
            ",".join(value.isoformat() for value in request.radiosonde_nominal_times),
            request.cache_directory,
        ),
    )

    for nominal_time in request.radiosonde_nominal_times:
        request_payload = request.artifact_request_payload(
            provider=MeteorologyProvider.RADIOSONDE,
            timestamps=(nominal_time,),
        )
        normalized_paths = radiosonde_cache_paths(
            request, nominal_time, normalized=True
        )
        normalized_validation = validate_cached_artifact(
            normalized_paths,
            expected_request=request_payload,
            expected_release=None,
        )
        cached_profiles = None
        if normalized_validation.valid:
            try:
                candidate_profiles = profiles_from_netcdf_bytes(
                    normalized_paths.artifact.read_bytes()
                )
                if (
                    len(candidate_profiles) != 1
                    or candidate_profiles[0].nominal_time != nominal_time
                ):
                    raise ValueError(
                        "Normalized radiosonde cache does not match its nominal time."
                    )
                cached_profiles = candidate_profiles
            except Exception:
                normalized_reason = "normalized_contract_invalid"
            else:
                normalized_reason = "valid"
        else:
            normalized_reason = normalized_validation.reason
        if cached_profiles is not None:
            profiles.extend(cached_profiles)
            normalized_files.append(normalized_paths.artifact)
            manifests.append(normalized_paths.manifest)
            cache_hits += 1
            _info(
                logger,
                f"Radiosonde normalized cache hit: {normalized_paths.artifact}",
            )
            inventory.append(
                {
                    "provider": "radiosonde",
                    "timestamp_utc": nominal_time.isoformat(),
                    "cache_status": "normalized_hit",
                    "path": normalized_paths.artifact.as_posix(),
                }
            )
            continue

        cache_misses += 1
        _info(
            logger,
            f"Radiosonde normalized cache miss for {nominal_time.isoformat()}",
        )
        if normalized_reason not in {"artifact_missing", "manifest_missing"}:
            message = (
                "Corrupt radiosonde normalized cache rejected: "
                f"{normalized_paths.artifact} ({normalized_reason})"
            )
            warnings.append(message)
            _warning(logger, message)
        if request.mode is AcquisitionMode.CACHE_ONLY:
            return ProviderAcquisitionResult(
                provider="radiosonde",
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
                error_message=f"Exact radiosonde cache is unavailable for {nominal_time.isoformat()}.",
                inventory=tuple(inventory),
            )

        raw_paths = radiosonde_cache_paths(request, nominal_time, normalized=False)
        raw_validation = validate_cached_artifact(
            raw_paths,
            expected_request=request_payload,
            expected_release=None,
        )
        raw_payload: bytes
        payload_kind: str
        table: pd.DataFrame | None = None
        downloaded_this_artifact = False
        if raw_validation.valid:
            raw_payload = raw_paths.artifact.read_bytes()
            payload_kind = str(raw_validation.manifest["raw_payload_kind"])
            cache_hits += 1
            _info(logger, f"Radiosonde raw cache hit: {raw_paths.artifact}")
        else:
            if raw_validation.reason not in {"artifact_missing", "manifest_missing"}:
                message = (
                    "Corrupt radiosonde raw cache rejected; downloading exact request again: "
                    f"{raw_paths.artifact} ({raw_validation.reason})"
                )
                warnings.append(message)
                _warning(logger, message)
            response = None
            last_error: Exception | None = None
            _info(
                logger,
                f"Radiosonde download started for {nominal_time.isoformat()} "
                f"(station {request.radiosonde_station_id}, max_retries={request.max_retries})",
            )
            for attempt in range(1, request.max_retries + 1):
                try:
                    response = transport(
                        nominal_time, request.radiosonde_station_id
                    )
                    retries += attempt - 1
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < request.max_retries:
                        _warning(
                            logger,
                            f"Radiosonde retry {attempt}/{request.max_retries} "
                            f"after {type(exc).__name__}.",
                        )
            if response is None:
                retries += request.max_retries - 1
                return ProviderAcquisitionResult(
                    provider="radiosonde",
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
                        retries=retries,
                    ),
                    warnings=tuple(warnings),
                    error_code="siphon_unavailable",
                    error_message=str(last_error),
                    inventory=tuple(inventory),
                )
            raw_payload = response.payload
            payload_kind = response.payload_kind
            table = response.table
            if payload_kind == RAW_CANONICAL_DATAFRAME and table is not None:
                raw_payload = canonical_dataframe_snapshot(table)
            raw_manifest = build_manifest(
                paths=raw_paths,
                artifact_bytes=raw_payload,
                provider="wyoming_siphon",
                dataset=request.radiosonde_station_id,
                request_payload=request_payload,
                timestamps=(nominal_time,),
                area=None,
                variables=tuple(str(column) for column in (table.columns if table is not None else ())),
                levels=(),
                release=None,
                normalizer=None,
                normalizer_version=None,
                raw_payload_kind=payload_kind,
                dependency_versions={"siphon": _package_version("siphon")},
            )
            publish_artifact(raw_paths, raw_payload, raw_manifest)
            raw_validation = validate_cached_artifact(
                raw_paths,
                expected_request=request_payload,
                expected_release=None,
            )
            if not raw_validation.valid:
                raise OSError("Published radiosonde raw cache failed integrity validation.")
            bytes_downloaded += len(raw_payload)
            downloaded_this_artifact = True
            final_status = ProviderStatus.DOWNLOADED
            _info(
                logger,
                f"Radiosonde raw payload published ({len(raw_payload)} bytes): "
                f"{raw_paths.artifact}",
            )

        try:
            profile = _normalize(
                raw_payload,
                payload_kind,
                request,
                nominal_time,
                table=table,
            )
            normalized_payload = profiles_to_netcdf_bytes(
                (profile,),
                cache_metadata={
                    "snapshot_schema": "milgrau-normalized-radiosonde-v1",
                    "raw_payload_kind": payload_kind,
                },
            )
            normalized_manifest = build_manifest(
                paths=normalized_paths,
                artifact_bytes=normalized_payload,
                provider="wyoming_siphon",
                dataset=request.radiosonde_station_id,
                request_payload=request_payload,
                timestamps=(nominal_time,),
                area=None,
                variables=tuple(str(column) for column in _table_from_raw(
                    raw_payload,
                    payload_kind,
                    nominal_time,
                    request.radiosonde_station_id,
                ).columns),
                levels=(),
                release=None,
                normalizer="normalize_wyoming_radiosonde",
                normalizer_version=profile.normalizer_version,
                source_files=(source_reference(raw_paths, raw_validation.manifest),),
                raw_payload_kind=payload_kind,
                dependency_versions={
                    "siphon": _package_version("siphon"),
                    "pandas": _package_version("pandas"),
                    "xarray": _package_version("xarray"),
                },
            )
            publish_artifact(normalized_paths, normalized_payload, normalized_manifest)
        except Exception as exc:
            return ProviderAcquisitionResult(
                provider="radiosonde",
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
                    retries=retries,
                ),
                warnings=tuple(warnings),
                error_code="radiosonde_normalization_failed",
                error_message=str(exc),
                inventory=tuple(inventory),
            )
        profiles.append(profile)
        raw_files.append(raw_paths.artifact)
        normalized_files.append(normalized_paths.artifact)
        manifests.extend((raw_paths.manifest, normalized_paths.manifest))
        inventory.append(
            {
                "provider": "radiosonde",
                "timestamp_utc": nominal_time.isoformat(),
                "cache_status": "downloaded_and_normalized"
                if downloaded_this_artifact
                else "normalized_from_raw_cache",
                "raw_path": raw_paths.artifact.as_posix(),
                "normalized_path": normalized_paths.artifact.as_posix(),
            }
        )
        if final_status is ProviderStatus.CACHE_HIT:
            final_status = ProviderStatus.NORMALIZED
        _info(
            logger,
            f"Radiosonde normalization complete: {normalized_paths.artifact}",
        )

    return ProviderAcquisitionResult(
        provider="radiosonde",
        status=final_status,
        profiles=tuple(profiles),
        raw_files=tuple(raw_files),
        normalized_files=tuple(normalized_files),
        manifest_files=tuple(manifests),
        metrics=AcquisitionMetrics(
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            bytes_downloaded=bytes_downloaded,
            duration_seconds=time.perf_counter() - started,
            retries=retries,
        ),
        warnings=tuple(warnings),
        inventory=tuple(inventory),
    )
