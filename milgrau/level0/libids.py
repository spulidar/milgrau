"""LIBIDS Level 0 orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.config.station import resolve_station_context
from milgrau.incremental import output_is_current
from milgrau.io.contracts import netcdf_satisfies_contract, validate_level0_contract
from milgrau.io.logging_utils import bind_log_context
from milgrau.io.paths import level0_output_path, raw_data_root
from milgrau.io.selection import parse_input_selection, select_available_session_ids
from milgrau.level0.common import incremental_enabled
from milgrau.level0.config import resolve_level0_config, validate_level0_config
from milgrau.level0.header_cache import source_paths_from_level0
from milgrau.level0.inventory import build_session_inventory
from milgrau.level0.netcdf import scc_measurement_id
from milgrau.level0.processing import process_session_group
from milgrau.level0.quality import filter_laser_shots
from milgrau.operations import ExecutionResult, ExecutionStatus, ExecutionSummary
from milgrau.provenance import netcdf_provenance_is_complete


def _raw_input_paths(group_df) -> list[Path]:
    return [Path(path) for path in group_df["filepath"].tolist()]


def _resolve_expected_scc_contexts(
    session_id: str,
    group_df,
    config: Mapping,
    output_path: Path,
) -> list[tuple[str, dict]]:
    """Resolve expected day/night SCC derivatives from the canonical segment table."""
    del session_id, group_df
    if not isinstance(config.get("_scc_catalog"), Mapping):
        return []
    try:
        with xr.open_dataset(output_path) as ds:
            ds.load()
            channels = [str(value) for value in ds["channel_string"].values]
            labels = np.asarray(ds["Segment_Label"].values).astype(str)
            regimes = np.asarray(ds["Segment_Regime"].values).astype(str)
            starts = pd.to_datetime(ds["Segment_Start_Time_UTC"].values, utc=True)
        if not (labels.size == regimes.size == len(starts)):
            return []
    except Exception:
        return []

    contexts: list[tuple[str, dict]] = []
    for regime in ("day", "night"):
        mask = np.asarray(
            [str(value).lower() == regime for value in regimes],
            dtype=bool,
        )
        if not np.any(mask):
            continue
        source_segments = tuple(labels[mask].tolist())
        first_start = pd.Timestamp(starts[np.flatnonzero(mask)[0]])
        try:
            context = resolve_station_context(
                config,
                measurement_time=first_start.to_pydatetime(),
                available_channels=channels,
                mode=regime,
            )
        except Exception:
            return []
        if context.get("scc_available", False) and context.get(
            "scc_export_ready", False
        ):
            resolved = dict(context)
            resolved["solar_regime"] = regime
            resolved["source_segments"] = ",".join(source_segments)
            resolved["source_segment_count"] = int(len(source_segments))
            resolved["contains_time_gaps"] = bool(len(source_segments) > 1)
            contexts.append((regime, resolved))
    return contexts


def _primary_output_satisfies_contract(path: Path) -> bool:
    return netcdf_satisfies_contract(
        path, validate_level0_contract
    ) and netcdf_provenance_is_complete(path)


def _scc_output_satisfies_context(
    path: Path,
    context: Mapping,
    catalog: Mapping | None = None,
) -> bool:
    """Validate one solar-regime SCC Level 0 derivative and its ancillary."""
    if not _primary_output_satisfies_contract(path):
        return False
    try:
        with xr.open_dataset(path, mask_and_scale=False) as ds:
            if "channel_ID" not in ds:
                return False
            # Older SCC derivatives without Measurement_ID must be regenerated
            # even when the canonical scientific Level 0 is still current.
            start_utc = pd.to_datetime(
                f"{ds.attrs['RawData_Start_Date']}{ds.attrs['RawData_Start_Time_UT']}",
                format="%Y%m%d%H%M%S",
                utc=True,
            )
            expected_id = scc_measurement_id(start_utc, str(context["station_id"]))
            if ds.attrs.get("Measurement_ID") != expected_id:
                return False
            lr_flags = context.get("lr_input", {})
            if any(int(value) == 0 for value in lr_flags.values()):
                if catalog is None:
                    return False
                from milgrau.level0.scc_lidar_ratio import scc_lr_ancillary_matches

                if not scc_lr_ancillary_matches(path, ds.attrs, context, catalog):
                    return False
            elif "LR_File_Name" in ds.attrs:
                return False
            if str(ds.attrs.get("Solar_Regime", "")).lower() != str(
                context.get("solar_regime", "")
            ).lower():
                return False
            if str(ds.attrs.get("Source_Segments", "")) != str(
                context.get("source_segments", "")
            ):
                return False
            if int(ds.attrs.get("Source_Segment_Count", -1)) != int(
                context.get("source_segment_count", -2)
            ):
                return False
            expected_gaps = 1 if bool(
                context.get("contains_time_gaps", False)
            ) else 0
            if int(ds.attrs.get("Contains_Time_Gaps", -1)) != expected_gaps:
                return False
            expected_channels = [
                str(value) for value in context.get("scc_channels", [])
            ]
            actual_channels = [
                str(value) for value in ds["channel_string"].values
            ]
            if actual_channels != expected_channels:
                return False
            expected_ids = np.asarray(
                [
                    int(context["channel_ids"][channel])
                    for channel in expected_channels
                ],
                dtype=np.int64,
            )
            actual_ids = np.asarray(ds["channel_ID"].values, dtype=np.int64)
            if (
                actual_ids.shape != expected_ids.shape
                or not np.array_equal(actual_ids, expected_ids)
            ):
                return False
            lr_input = context.get("lr_input", {})
            if isinstance(lr_input, Mapping) and lr_input:
                if "LR_Input" not in ds:
                    return False
                values = np.ma.asarray(ds["LR_Input"].values)
                for index, channel in enumerate(actual_channels):
                    if channel in lr_input and (
                        np.ma.is_masked(values[index])
                        or int(values[index]) != int(lr_input[channel])
                    ):
                        return False
        return True
    except Exception:
        return False


def _level0_is_current(
    session_id: str,
    group_df,
    config: dict,
    output_path,
) -> bool:
    output = Path(output_path)
    inputs = _raw_input_paths(group_df)
    primary_current = output_is_current(
        output,
        inputs,
        config=config,
        integrity_check=_primary_output_satisfies_contract,
    )
    # A published/cached SCC derivative has its own independent lifetime.
    # It must never make a canonical Level 0 session stale.
    return primary_current


def _requested_session_ids(values, config: dict, df_raw) -> set[str] | None:
    """Resolve CLI selectors against the current raw inventory."""
    selection = parse_input_selection(values, config)
    if selection.is_empty:
        return None

    available_ids = [str(value) for value in df_raw["session_id"].dropna().unique()]
    selected = select_available_session_ids(selection, available_ids, config)

    if selection.paths:
        resolved_inventory_paths = df_raw["filepath"].map(lambda value: Path(str(value)).expanduser().resolve())
        for selected_path in selection.paths:
            if selected_path.is_dir():
                mask = resolved_inventory_paths.map(
                    lambda value: value == selected_path or selected_path in value.parents
                )
            else:
                mask = resolved_inventory_paths == selected_path
            matches = set(df_raw.loc[mask, "session_id"].astype(str))
            if not matches:
                raise FileNotFoundError(
                    f"Explicit raw input {selected_path} was not found in the LIBIDS inventory."
                )
            selected.update(matches)
    return selected


def _reexport_indexed_sources(
    selection, config: dict, logger: logging.Logger,
) -> list[str] | None:
    """Use canonical L0 source names only when all resolve uniquely and unchanged."""
    if selection.paths or selection.dates or len(selection.session_ids) != 1:
        return None
    config_file = config.get("_config_file")
    if not config_file:
        return None
    session = next(iter(selection.session_ids))
    canonical = level0_output_path(session, config)
    if not _primary_output_satisfies_contract(canonical):
        return None
    try:
        with xr.open_dataset(canonical, decode_times=False) as dataset:
            raw_sources = str(dataset.attrs.get("Source_Files", ""))
    except (OSError, ValueError, KeyError):
        return None
    names = [part for part in raw_sources.split(";") if part]
    if not names:
        return None
    cache_dir = Path(str(config_file)).resolve().parent / ".cache" / "libids"
    paths = source_paths_from_level0(cache_dir, names)
    if paths is not None:
        bind_log_context(logger, stage="scc_reexport").info(
            "resolved %d original Licel sources from L0 + header index",
            len(paths),
        )
    return paths


def process_level_0(
    config: dict,
    logger: logging.Logger,
    *,
    inputs: Sequence[str] | None = None,
    force: bool = False,
    scc_reexport: bool = False,
) -> ExecutionSummary:
    """Run LIBIDS; optionally stage only SCC products for existing scientific L0."""
    if force and scc_reexport:
        raise ValueError("--force and --scc-reexport are mutually exclusive.")
    if scc_reexport and not isinstance(config.get("_scc_catalog"), Mapping):
        raise ValueError("--scc-reexport requires config.scc.enabled: true.")
    validate_level0_config(config)
    level0_config = resolve_level0_config(config)
    pipeline_logger = bind_log_context(logger, pipeline="L0")
    raw_dir = raw_data_root(config)
    selection = parse_input_selection(inputs, config)
    if selection.is_empty:
        df_raw = build_session_inventory(str(raw_dir), config, pipeline_logger)
    elif scc_reexport:
        sources = _reexport_indexed_sources(selection, config, pipeline_logger)
        df_raw = build_session_inventory(
            str(raw_dir), config, pipeline_logger, selection=selection,
            source_paths=sources,
        )
    else:
        df_raw = build_session_inventory(
            str(raw_dir), config, pipeline_logger, selection=selection,
        )
    if not df_raw.empty:
        requested = _requested_session_ids(inputs, config, df_raw)
        if requested is not None:
            df_raw = df_raw[df_raw["session_id"].astype(str).isin(requested)].copy()
    if df_raw.empty:
        bind_log_context(pipeline_logger, stage="discovery").info("no raw measurements found")
        return ExecutionSummary.from_results(
            [ExecutionResult.skipped("level0.discovery", "No new data to process", input_path=raw_dir, metadata={"pipeline": "L0"})]
        )

    # Header discovery and chronological sessionization must precede QA:
    # sessions span directories/days, and dark currents associate to nearby
    # complete sessions. Avoid materializing QA results for the entire archive;
    # screen and process each session immediately after the lightweight index.
    session_count = int(df_raw["session_id"].nunique(dropna=True))
    measurements_count = int((df_raw["meas_type"] == "measurements").sum())
    dark_count = int((df_raw["meas_type"] == "dark_current").sum())
    bind_log_context(pipeline_logger, stage="discovery").info(
        "inventory complete | %d candidate sessions | %d measurements | "
        "%d dark currents | QA and L0 will run session by session",
        session_count,
        measurements_count,
        dark_count,
    )

    incremental = incremental_enabled(config)
    results: list[ExecutionResult] = []
    session_groups = df_raw.dropna(subset=["session_id"]).groupby("session_id", sort=False)
    for session_number, (session_id, raw_group) in enumerate(session_groups, start=1):
        group_logger = bind_log_context(pipeline_logger, session_id=session_id)
        bind_log_context(group_logger, stage="qa").info(
            "session %d/%d | screening %d measurement + %d dark-current files",
            session_number,
            session_count,
            int((raw_group["meas_type"] == "measurements").sum()),
            int((raw_group["meas_type"] == "dark_current").sum()),
        )
        group_df = filter_laser_shots(
            raw_group,
            group_logger,
            tolerance_fraction=level0_config.acquisition_qa.laser_shot_tolerance_fraction,
            header_time_jitter_s=level0_config.acquisition_qa.licel_header_time_jitter_s,
        )
        if group_df.empty:
            bind_log_context(group_logger, stage="qa").warning("no data survived acquisition QA")
            results.append(ExecutionResult.skipped(
                "level0.quality",
                "No data survived quality control",
                input_path=raw_dir,
                metadata={"pipeline": "L0", "session_id": session_id},
            ))
            continue

        netcdf_path = level0_output_path(session_id, config)
        if not scc_reexport and not force and incremental and _level0_is_current(session_id, group_df, config, netcdf_path):
            bind_log_context(group_logger, stage="skip").info("up to date | %s", netcdf_path.name)
            results.append(ExecutionResult.skipped("level0.incremental", "Level 0 is up to date", output_path=netcdf_path, metadata={"pipeline": "L0", "session_id": session_id}))
            continue

        measurement_count = int((group_df["meas_type"] == "measurements").sum())
        bind_log_context(group_logger, stage="start").info("%d raw files", measurement_count)
        try:
            if scc_reexport:
                result = process_session_group(
                    session_id, group_df, config, group_logger, scc_only=True,
                )
            else:
                result = process_session_group(session_id, group_df, config, group_logger)
            if not isinstance(result, ExecutionResult):
                raise TypeError(f"process_session_group returned {type(result).__name__}; expected ExecutionResult.")
        except Exception as exc:
            result = ExecutionResult.failure("level0.group", "unexpected group conversion error", output_path=netcdf_path, cause=exc, include_traceback=True, metadata={"pipeline": "L0", "session_id": session_id})
        if result.status is ExecutionStatus.OK:
            duration = 0.0 if result.duration_seconds is None else result.duration_seconds
            bind_log_context(group_logger, stage="done").info("%s | %.1f s", netcdf_path.name, duration)
        elif result.status is ExecutionStatus.ERROR:
            bind_log_context(group_logger, stage=result.stage.removeprefix("level0.")).error("%s | %s", result.message, result.cause or "unknown failure")
            if result.traceback:
                group_logger.debug("failure traceback\n%s", result.traceback)
        results.append(result)

    if not results:
        bind_log_context(pipeline_logger, stage="qa").warning(
            "no associated measurement sessions were available for QA"
        )
        return ExecutionSummary.from_results(
            [ExecutionResult.skipped(
                "level0.quality",
                "No associated measurements survived discovery",
                input_path=raw_dir,
                metadata={"pipeline": "L0"},
            )]
        )

    return ExecutionSummary.from_results(results)
