"""Explicit meteorology prefetch CLI; it never runs LEBEAR."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from milgrau.cli.common import finish_cli, run_guarded
from milgrau.config.loader import load_config
from milgrau.io.logging_utils import setup_logger
from milgrau.meteorology.acquisition import get_or_acquire_meteorology
from milgrau.meteorology.era5_acquisition import build_era5_cds_requests
from milgrau.meteorology.request import (
    AcquisitionMode,
    MeteorologyProvider,
    MeteorologyRequest,
    hourly_timestamps_for_interval,
)
from milgrau.operations import ExecutionResult, ExecutionSummary


def _utc_datetime(raw: str) -> datetime:
    value = str(raw).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("Timestamps must include a UTC offset or Z.")
    return parsed.astimezone(UTC)


def _radiosonde_times_for_interval(
    start: datetime,
    end: datetime,
) -> tuple[datetime, ...]:
    """Return bounding and internal 00/12 soundings for explicit prefetch only."""
    start_utc = start.astimezone(UTC)
    end_utc = end.astimezone(UTC)
    previous_hour = 12 if start_utc.hour >= 12 else 0
    current = start_utc.replace(
        hour=previous_hour, minute=0, second=0, microsecond=0
    )
    values = []
    while current <= end_utc:
        values.append(current)
        current += timedelta(hours=12)
    if not values or values[-1] < end_utc:
        values.append(current)
    return tuple(values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="milgrau-meteorology",
        description="Plan and prefetch frozen meteorology cache artifacts without running retrieval.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prefetch = subparsers.add_parser("prefetch")
    prefetch.add_argument("--start", required=True, type=_utc_datetime)
    prefetch.add_argument("--end", required=True, type=_utc_datetime)
    prefetch.add_argument("--site", choices=("spu",), default="spu")
    prefetch.add_argument(
        "--provider",
        choices=[member.value for member in MeteorologyProvider],
        default=MeteorologyProvider.BOTH.value,
    )
    prefetch.add_argument(
        "--radiosonde-time",
        action="append",
        type=_utc_datetime,
        default=[],
        help="Explicit 00/12 UTC nominal sounding; repeatable.",
    )
    prefetch.add_argument("--cache-only", action="store_true")
    prefetch.add_argument("--dry-run", action="store_true")
    prefetch.add_argument("--refresh-provisional", action="store_true")
    prefetch.add_argument("--config", default="config.yaml")
    return parser


def _request_from_args(args: argparse.Namespace, config: dict) -> MeteorologyRequest:
    if args.end < args.start:
        raise ValueError("--end must not precede --start.")
    meteorology = config.get("meteorology", {})
    radio_config = meteorology.get("radiosonde", {})
    site = config.get("site", {})
    provider = MeteorologyProvider(args.provider)
    radiosonde_times = tuple(args.radiosonde_time)
    if provider in {MeteorologyProvider.RADIOSONDE, MeteorologyProvider.BOTH}:
        if not radiosonde_times:
            radiosonde_times = _radiosonde_times_for_interval(args.start, args.end)
    mode = (
        AcquisitionMode.CACHE_ONLY
        if args.cache_only
        else AcquisitionMode.PREFETCH
    )
    return MeteorologyRequest(
        site_id=str(args.site),
        latitude_deg_north=float(site["latitude"]),
        longitude_deg_east=float(site["longitude"]),
        station_altitude_m=float(site["station_altitude_m"]),
        measurement_timestamps=hourly_timestamps_for_interval(args.start, args.end),
        provider=provider,
        mode=mode,
        cache_directory=Path(
            meteorology.get("cache_directory", "01-data/meteorology_cache")
        ),
        radiosonde_station_id=str(
            radio_config.get(
                "station_id",
                config.get("radiosonde", {}).get("station_id", "83779"),
            )
        ),
        radiosonde_nominal_times=radiosonde_times,
        era5_grid_degrees=float(meteorology.get("era5", {}).get("grid_degrees", 0.25)),
        allow_era5t=bool(meteorology.get("allow_era5t", True)),
        timeout_seconds=float(meteorology.get("timeout_seconds", 300.0)),
        max_retries=int(meteorology.get("max_retries", 3)),
    )


def _prefetch(
    args: argparse.Namespace,
    config: dict,
    logger,
) -> ExecutionSummary:
    request = _request_from_args(args, config)
    logger.info(
        "Meteorology request: "
        + json.dumps(
            request.canonical_payload(include_cache_directory=True),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    for _, hours in request.era5_month_groups:
        logger.info(
            "ERA5 exact subrequests: "
            + json.dumps(
                build_era5_cds_requests(request, hours),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    if args.dry_run:
        return ExecutionSummary.from_results(
            [
                ExecutionResult.success(
                    "meteorology.plan",
                    "Dry-run completed; no network or cache writes were performed.",
                    metadata={
                        "era5_hours": len(request.era5_hours),
                        "radiosonde_times": len(request.radiosonde_nominal_times),
                    },
                )
            ]
        )

    acquisition = get_or_acquire_meteorology(
        request,
        logger=logger,
        refresh_provisional=bool(args.refresh_provisional),
    )
    metadata = {
        "overall_status": acquisition.overall_status,
        "observational_providers": acquisition.observational_provider_count,
        "files": len(acquisition.files),
        "quantitative_retrieval_allowed": acquisition.quantitative_retrieval_allowed,
    }
    if acquisition.fatal_error is not None:
        result = ExecutionResult.failure(
            "meteorology.prefetch",
            acquisition.fatal_error,
            fatal=True,
            metadata=metadata,
        )
    elif acquisition.usable_observational:
        result = ExecutionResult.success(
            "meteorology.prefetch",
            "Meteorology cache prepared with observational/reanalysis data.",
            metadata=metadata,
        )
    else:
        result = ExecutionResult.failure(
            "meteorology.prefetch",
            "No observational provider is available; diagnostic USSA-1976 fallback returned.",
            metadata=metadata,
        )
    result.log(logger)
    return ExecutionSummary.from_results([result])


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logger("METEOROLOGY", config=config)
    summary = run_guarded(
        "cli.meteorology",
        logger,
        lambda: _prefetch(args, config, logger),
    )
    return finish_cli("METEOROLOGY", summary, logger)


if __name__ == "__main__":
    raise SystemExit(main())
