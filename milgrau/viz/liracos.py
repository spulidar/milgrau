"""LIRACOS Level 1 visualization pipeline orchestration."""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from milgrau.incremental import output_is_current
from milgrau.io.contracts import validate_level1_contract
from milgrau.io.filesystem import ensure_directories
from milgrau.io.logging_utils import bind_log_context
from milgrau.io.paths import (
    global_mean_rcs_output_path,
    processed_data_root,
    product_save_id,
    quicklook_output_path,
)
from milgrau.operations import ExecutionResult, ExecutionStatus, ExecutionSummary
from milgrau.viz.config import resolve_visualization_config
from milgrau.viz.quicklooks import format_channel_name, plot_global_mean_rcs, plot_quicklook
from milgrau.viz.style import DEFAULT_LOGO_SPECS

RCS_VARIABLE = "range_corrected_signal"
RCS_ERROR_VARIABLE = "range_corrected_signal_error"


def _incremental_enabled(config: Mapping[str, Any]) -> bool:
    processing = config.get("processing")
    if not isinstance(processing, Mapping):
        raise KeyError("Configuration processing section is required.")
    if "incremental" not in processing:
        raise KeyError("Missing required configuration: processing.incremental")
    value = processing["incremental"]
    if not isinstance(value, bool):
        raise ValueError("Configuration processing.incremental must be a boolean.")
    return value


def _get_output_format(config: dict[str, Any]) -> str:
    """Return explicitly configured plot output format."""
    return resolve_visualization_config(config).output_format


def _quicklook_output_path(
    output_folder: Path,
    file_name_prefix: str,
    channel_name: str,
    max_altitude: float,
    config: dict[str, Any],
) -> Path:
    return quicklook_output_path(
        output_folder=output_folder,
        file_name_prefix=file_name_prefix,
        formatted_channel_name=format_channel_name(channel_name),
        max_altitude_km=max_altitude,
        output_format=_get_output_format(config),
    )


def _global_mean_output_path(output_folder: Path, file_name_prefix: str, config: dict[str, Any]) -> Path:
    return global_mean_rcs_output_path(output_folder, file_name_prefix, _get_output_format(config))


def _get_visualization_channels(config: dict[str, Any]) -> list[str]:
    """Return validated channels in configured order."""
    return list(resolve_visualization_config(config).channels_to_plot)


def _get_altitude_ranges_km(config: dict[str, Any]) -> list[float]:
    """Return validated altitude limits in kilometers."""
    return list(resolve_visualization_config(config).altitude_ranges_km)


def _visual_dependencies(root_path: Path) -> list[Path]:
    return [
        logo_path
        for logo_name, _height in DEFAULT_LOGO_SPECS
        if (logo_path := root_path / "img" / logo_name).is_file()
    ]


def _prepare_level1_for_visualization(ds: xr.Dataset) -> xr.Dataset:
    """Return a plotting-ready Level 1 dataset with altitude in kilometers."""
    if "altitude" not in ds.coords:
        raise ValueError("Level 1 dataset does not contain an 'altitude' coordinate.")
    max_altitude = float(ds["altitude"].max().values)
    if max_altitude > 100.0:
        ds = ds.assign_coords(altitude=ds["altitude"] / 1000.0)
    ds["altitude"].attrs["units"] = "km"
    ds["altitude"].attrs["long_name"] = "Altitude above ground level"
    return ds


def _extract_level1_boundaries(ds: xr.Dataset) -> tuple[xr.DataArray | None, float, float]:
    """Extract PBL and tropopause metadata used as optional plot overlays."""
    pbl_da = ds["PBL_Height_km"] if "PBL_Height_km" in ds else None
    cpt_km = float(ds.attrs.get("tropopause_cpt_km", np.nan))
    lrt_km = float(ds.attrs.get("tropopause_lrt_km", np.nan))
    return pbl_da, cpt_km, lrt_km


def _validate_l1_visualization_contract(ds: xr.Dataset) -> None:
    validate_level1_contract(ds)


def process_single_nc(args: tuple[str | Path, dict[str, Any], str | Path, logging.Logger]) -> ExecutionResult:
    """Render all Level 1 quicklooks for one NetCDF file."""
    nc_file_path, config, root_dir, logger = args
    nc_file = Path(nc_file_path)
    save_id = product_save_id(nc_file)
    file_logger = bind_log_context(logger, pipeline="VIZ", save_id=save_id)
    root_path = Path(root_dir)
    started_at = time.perf_counter()
    output_folder: Path | None = None
    stage = "visualization.initialize"
    try:
        resolved = resolve_visualization_config(config)
        file_name_prefix = save_id
        output_folder = nc_file.parent / "quicklooks"
        ensure_directories(output_folder)
        incremental = _incremental_enabled(config)
        dependencies = _visual_dependencies(root_path)
        generated_count = 0
        skipped_count = 0

        stage = "visualization.ingestion"
        with xr.open_dataset(nc_file) as ds:
            ds.load()
            stage = "visualization.validation"
            _validate_l1_visualization_contract(ds)
            ds = _prepare_level1_for_visualization(ds)
            pbl_da, cpt_km, lrt_km = _extract_level1_boundaries(ds)
            channels_to_plot = list(resolved.channels_to_plot)
            altitude_ranges = list(resolved.altitude_ranges_km)
            available_channels = {str(channel) for channel in ds.channel.values}
            bind_log_context(file_logger, stage="start").info(
                "channels=%d | altitude ranges=%s km",
                len(channels_to_plot),
                ",".join(f"{value:g}" for value in altitude_ranges),
            )

            stage = "visualization.quicklooks"
            for channel_name in channels_to_plot:
                channel_logger = bind_log_context(file_logger, stage=channel_name)
                if channel_name not in available_channels:
                    channel_logger.warning("channel unavailable; skipped")
                    continue
                rc_signal = ds[RCS_VARIABLE].sel(channel=channel_name)
                rc_error = ds[RCS_ERROR_VARIABLE].sel(channel=channel_name)
                for max_altitude in altitude_ranges:
                    expected_path = _quicklook_output_path(output_folder, file_name_prefix, channel_name, max_altitude, config)
                    if incremental and output_is_current(
                        expected_path,
                        [nc_file],
                        config=config,
                        extra_dependencies=dependencies,
                    ):
                        channel_logger.debug("up to date: %s", expected_path.name)
                        skipped_count += 1
                        continue

                    sig_slice = rc_signal.sel(altitude=slice(0, max_altitude))
                    err_slice = rc_error.sel(altitude=slice(0, max_altitude))
                    if sig_slice.size == 0:
                        channel_logger.warning("empty altitude slice up to %.1f km", max_altitude)
                        continue
                    plot_quicklook(
                        data_slice=sig_slice,
                        error_slice=err_slice,
                        max_altitude=max_altitude,
                        channel_name=channel_name,
                        ds=ds,
                        output_folder=str(output_folder),
                        file_name_prefix=file_name_prefix,
                        config=config,
                        root_dir=str(root_path),
                        pbl_da=pbl_da,
                        cpt_km=cpt_km,
                        lrt_km=lrt_km,
                    )
                    generated_count += 1
                    channel_logger.debug("generated: %s", expected_path.name)
                    del sig_slice, err_slice
                    plt.close("all")
                    gc.collect()

            stage = "visualization.global_mean"
            global_mean_path = _global_mean_output_path(output_folder, file_name_prefix, config)
            if incremental and output_is_current(
                global_mean_path,
                [nc_file],
                config=config,
                extra_dependencies=dependencies,
            ):
                skipped_count += 1
                bind_log_context(file_logger, stage="mean").debug("up to date: %s", global_mean_path.name)
            else:
                plot_path = plot_global_mean_rcs(ds, str(output_folder), file_name_prefix, config, str(root_path))
                if plot_path is not None:
                    generated_count += 1

        plt.close("all")
        gc.collect()
        duration = time.perf_counter() - started_at
        bind_log_context(file_logger, stage="done").info(
            "generated=%d | skipped=%d | %.1f s",
            generated_count,
            skipped_count,
            duration,
        )
        return ExecutionResult.success(
            "visualization.complete",
            "Visualization products generated",
            input_path=nc_file,
            output_path=output_folder,
            duration_seconds=duration,
            metadata={"pipeline": "VIZ", "save_id": save_id, "generated": generated_count, "skipped": skipped_count},
        )
    except KeyError as exc:
        return ExecutionResult.skipped(
            stage,
            f"{nc_file.name} incompatible with current Level 1 contract: {exc}",
            input_path=nc_file,
            output_path=output_folder,
            metadata={"pipeline": "VIZ", "save_id": save_id, "cause_type": type(exc).__name__},
        )
    except Exception as exc:
        bind_log_context(file_logger, stage=stage.removeprefix("visualization.")).error("plotting failed: %s", exc)
        file_logger.debug("visualization failure traceback", exc_info=True)
        return ExecutionResult.failure(
            stage,
            "Visualization failed",
            input_path=nc_file,
            output_path=output_folder,
            cause=exc,
            include_traceback=True,
            duration_seconds=time.perf_counter() - started_at,
            metadata={"pipeline": "VIZ", "save_id": save_id},
        )


def process_all_level1_files(
    config: dict[str, Any],
    logger: logging.Logger,
    root_dir: str | Path | None = None,
) -> ExecutionSummary:
    """Discover and render all Level 1 NetCDF files under processed_data."""
    resolve_visualization_config(config)
    _incremental_enabled(config)
    pipeline_logger = bind_log_context(logger, pipeline="VIZ")
    root_path = Path.cwd() if root_dir is None else Path(root_dir)
    base_data_folder = processed_data_root(config, root_dir=root_path)
    nc_files = sorted(base_data_folder.rglob("*_level1_rcs.nc"))
    if not nc_files:
        bind_log_context(pipeline_logger, stage="discovery").warning("no Level 1 files found | %s", base_data_folder)
        return ExecutionSummary.from_results(
            [ExecutionResult.skipped("visualization.discovery", "No Level 1 NetCDF data found", input_path=base_data_folder, metadata={"pipeline": "VIZ"})]
        )
    bind_log_context(pipeline_logger, stage="queue").info("%d Level 1 files", len(nc_files))
    results: list[ExecutionResult] = []
    for nc_file in nc_files:
        result = process_single_nc((nc_file, config, root_path, pipeline_logger))
        if result.status.is_failure:
            bind_log_context(pipeline_logger, save_id=result.metadata.get("save_id"), stage="failed").warning("%s", result.message)
        results.append(result)
    return ExecutionSummary.from_results(results)
