"""Level 1 figure orchestration owned by LIPANCORA."""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from milgrau.incremental import output_is_current
from milgrau.io.contracts import validate_level1_contract
from milgrau.io.logging_utils import bind_log_context
from milgrau.io.paths import (
    global_mean_rcs_output_path,
    logging_session_id,
    product_session_id,
    quicklook_output_path,
)
from milgrau.operations import ExecutionResult
from milgrau.viz.atmosphere import plot_atmospheric_profile
from milgrau.viz.config import resolve_visualization_config
from milgrau.viz.quicklooks import (
    RCS_ERROR_VARIABLE,
    RCS_VARIABLE,
    channel_file_token,
    plot_global_mean_rcs,
    plot_quicklook,
)
from milgrau.viz.style import DEFAULT_LOGO_SPECS, get_output_settings


def level1_figures_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether LIPANCORA should maintain Level 1 figures."""
    visualization = config.get("visualization")
    if not isinstance(visualization, Mapping):
        raise KeyError("Configuration visualization section is required.")
    section = visualization.get("level1_figures")
    if not isinstance(section, Mapping):
        raise KeyError("Configuration visualization.level1_figures is required.")
    enabled = section.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("Configuration visualization.level1_figures.enabled must be boolean.")
    return enabled


def _figure_prefix(level1_path: Path) -> str:
    """Return a portable Level 1 figure prefix preserving explicit variants."""
    stem = level1_path.stem
    if stem.endswith("_L1_scc"):
        return stem[: -len("_L1_scc")] + "_scc"
    if stem.endswith("_L1"):
        return stem[: -len("_L1")]
    try:
        return product_session_id(level1_path)
    except ValueError:
        return stem


def _station_timezone_name(config: Mapping[str, Any]) -> str | None:
    catalog = config.get("_station_catalog")
    if not isinstance(catalog, Mapping):
        return None
    station = catalog.get("station")
    if not isinstance(station, Mapping):
        return None
    value = str(station.get("timezone", "")).strip()
    return value or None


def _plot_ready_level1(ds: xr.Dataset) -> xr.Dataset:
    """Return a plotting copy with altitude expressed in kilometers."""
    altitude = np.asarray(ds["altitude"].values, dtype=np.float64)
    result = ds
    if altitude.size and float(np.nanmax(altitude)) > 100.0:
        result = ds.assign_coords(altitude=ds["altitude"] / 1000.0)
    result["altitude"].attrs["units"] = "km"
    result["altitude"].attrs["long_name"] = "Altitude above ground level"
    return result


def _figure_dependencies(root_path: Path) -> list[Path]:
    return [
        logo_path
        for logo_name, _height in DEFAULT_LOGO_SPECS
        if (logo_path := root_path / "img" / logo_name).is_file()
    ]


def _atmospheric_output_path(
    output_dir: Path,
    prefix: str,
    config: Mapping[str, Any],
) -> Path:
    output_format, _dpi = get_output_settings(dict(config))
    return output_dir / f"{prefix}_L1_AtmosphericProfile.{output_format}"


def generate_level1_figures(
    level1_path: str | Path,
    config: Mapping[str, Any],
    logger: logging.Logger,
    *,
    root_dir: str | Path | None = None,
) -> ExecutionResult:
    """Generate and incrementally maintain all canonical Level 1 figures.

    Figure failures are explicitly non-fatal to the already validated Level 1
    scientific product.
    """
    started_at = time.perf_counter()
    path = Path(level1_path)
    output_dir = path.parent / "figures"
    root = Path.cwd() if root_dir is None else Path(root_dir)
    session_id = logging_session_id(path)
    figure_logger = bind_log_context(logger, stage="figures", session_id=session_id)

    if not level1_figures_enabled(config):
        return ExecutionResult.skipped(
            "level1.figures",
            "Level 1 figures disabled by configuration",
            input_path=path,
            output_path=output_dir,
            metadata={"pipeline": "L1", "session_id": session_id},
        )

    generated = 0
    skipped = 0
    failures: list[str] = []
    prefix = _figure_prefix(path)
    dependencies = _figure_dependencies(root)
    incremental = bool(config.get("processing", {}).get("incremental", False))

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved = resolve_visualization_config(config)
        output_format, _dpi = get_output_settings(dict(config))
        with xr.open_dataset(path, cache=False) as source:
            validate_level1_contract(source)
            ds_plot = _plot_ready_level1(source)
            available_channels = {str(value) for value in ds_plot["channel"].values}
            pbl = ds_plot["PBL_Height_km"] if "PBL_Height_km" in ds_plot else None
            cpt_km = float(ds_plot.attrs.get("tropopause_cpt_km", np.nan))
            lrt_km = float(ds_plot.attrs.get("tropopause_lrt_km", np.nan))
            timezone_name = _station_timezone_name(config)

            for channel_name in resolved.channels_to_plot:
                if channel_name not in available_channels:
                    figure_logger.debug("channel unavailable for figure: %s", channel_name)
                    continue
                signal = ds_plot[RCS_VARIABLE].sel(channel=channel_name)
                error = ds_plot[RCS_ERROR_VARIABLE].sel(channel=channel_name)
                token = channel_file_token(channel_name)
                for max_altitude in resolved.altitude_ranges_km:
                    expected = quicklook_output_path(
                        output_dir,
                        prefix,
                        token,
                        max_altitude,
                        output_format,
                    )
                    if incremental and output_is_current(
                        expected,
                        [path],
                        config=config,
                        extra_dependencies=dependencies,
                    ):
                        skipped += 1
                        continue
                    try:
                        signal_slice = signal.sel(altitude=slice(0, max_altitude))
                        error_slice = error.sel(altitude=slice(0, max_altitude))
                        if signal_slice.size == 0:
                            continue
                        plot_quicklook(
                            data_slice=signal_slice,
                            error_slice=error_slice,
                            max_altitude=max_altitude,
                            channel_name=channel_name,
                            ds=ds_plot,
                            output_folder=output_dir,
                            file_name_prefix=prefix,
                            config=dict(config),
                            root_dir=root,
                            session_id=session_id if session_id != "-" else None,
                            timezone_name=timezone_name,
                            pbl_da=pbl,
                            cpt_km=cpt_km,
                            lrt_km=lrt_km,
                            time_range_utc=None,
                        )
                        generated += 1
                    except Exception as exc:
                        failures.append(f"{expected.name}: {exc}")
                    finally:
                        plt.close("all")
                        gc.collect()

            mean_output = global_mean_rcs_output_path(output_dir, prefix, output_format)
            if incremental and output_is_current(
                mean_output,
                [path],
                config=config,
                extra_dependencies=dependencies,
            ):
                skipped += 1
            else:
                try:
                    if plot_global_mean_rcs(
                        ds_plot,
                        output_dir,
                        prefix,
                        dict(config),
                        root,
                    ) is not None:
                        generated += 1
                except Exception as exc:
                    failures.append(f"{mean_output.name}: {exc}")
                finally:
                    plt.close("all")
                    gc.collect()

            atmosphere_output = _atmospheric_output_path(output_dir, prefix, config)
            if incremental and output_is_current(
                atmosphere_output,
                [path],
                config=config,
                extra_dependencies=dependencies,
            ):
                skipped += 1
            else:
                try:
                    plot_atmospheric_profile(
                        source,
                        output_folder=output_dir,
                        file_name_prefix=prefix,
                        config=config,
                        root_dir=root,
                    )
                    generated += 1
                except Exception as exc:
                    failures.append(f"{atmosphere_output.name}: {exc}")
                finally:
                    plt.close("all")
                    gc.collect()

        gc.collect()
        duration = time.perf_counter() - started_at
        if failures:
            message = (
                f"Level 1 figures partially generated: {generated} generated, "
                f"{skipped} current, {len(failures)} failed"
            )
            figure_logger.warning("%s | %s", message, " | ".join(failures))
            return ExecutionResult.failure(
                "level1.figures",
                message,
                fatal=False,
                input_path=path,
                output_path=output_dir,
                cause=RuntimeError("; ".join(failures)),
                duration_seconds=duration,
                metadata={
                    "pipeline": "L1",
                    "session_id": session_id,
                    "generated": generated,
                    "skipped": skipped,
                    "failed": len(failures),
                },
            )
        if generated == 0 and skipped > 0:
            return ExecutionResult.skipped(
                "level1.figures",
                f"All {skipped} Level 1 figures are current",
                input_path=path,
                output_path=output_dir,
                duration_seconds=duration,
                metadata={
                    "pipeline": "L1",
                    "session_id": session_id,
                    "generated": 0,
                    "skipped": skipped,
                },
            )
        return ExecutionResult.success(
            "level1.figures",
            f"Generated {generated} Level 1 figure(s)",
            input_path=path,
            output_path=output_dir,
            duration_seconds=duration,
            metadata={
                "pipeline": "L1",
                "session_id": session_id,
                "generated": generated,
                "skipped": skipped,
            },
        )
    except Exception as exc:
        return ExecutionResult.failure(
            "level1.figures",
            "Level 1 figure generation failed",
            fatal=False,
            input_path=path,
            output_path=output_dir,
            cause=exc,
            include_traceback=True,
            duration_seconds=time.perf_counter() - started_at,
            metadata={"pipeline": "L1", "session_id": session_id},
        )


__all__ = ["generate_level1_figures", "level1_figures_enabled"]
