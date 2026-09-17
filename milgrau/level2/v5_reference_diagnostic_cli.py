"""Low-memory selector-domain diagnostic for real Level-1 method-v5 R&D.

This diagnostic intentionally does not run the method-v5 Monte Carlo.  It uses
productive Level-2 signal preparation/gluing one configured temporal block at a
time, builds the v5 progressive representation, and reports why candidate
reference cells do or do not survive Rayleigh QA and continuous-path
admissibility for caller-declared lower search bounds.

The purpose is to distinguish a search-domain failure from a true loss of
continuous high-column support without changing productive method v4.
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr

from milgrau.config.loader import load_config
from milgrau.level2.block_average import block_groups
from milgrau.level2.config import get_block_average_minutes, get_molecular_fit_config
from milgrau.level2.high_column_rnd import (
    catalogue_high_column_reference_cells,
    contiguous_usable_top_index,
    prepare_high_column_profile,
)
from milgrau.level2.high_column_selector import select_minimum_cost_high_column_reference
from milgrau.level2.retrieval import process_wavelength
from milgrau.level2.v5_validation_cli import _altitude_subset, _strict_json_write


def _finite_or_none(value: float) -> float | None:
    resolved = float(value)
    return resolved if np.isfinite(resolved) else None


def _domain_summary(catalogue: Any) -> dict[str, Any]:
    cells = tuple(catalogue.cells)
    accepted = tuple(cell for cell in cells if cell.accepted)
    admissible = tuple(cell for cell in cells if cell.nominal_path_admissible)
    both = tuple(
        cell for cell in cells if cell.accepted and cell.nominal_path_admissible
    )
    payload: dict[str, Any] = {
        "catalogued_cells": len(cells),
        "rayleigh_accepted_cells": len(accepted),
        "path_admissible_cells": len(admissible),
        "accepted_and_admissible_cells": len(both),
        "selected_reference_altitude_m": None,
        "selected_reference_diagnostic_cost": None,
        "selected_reference_effective_resolution_m": None,
    }
    if both:
        selected = select_minimum_cost_high_column_reference(
            catalogue,
            min_altitude_m=float(catalogue.search_min_altitude_m),
            max_altitude_m=float(catalogue.search_max_altitude_m),
        )
        payload.update(
            {
                "selected_reference_altitude_m": float(selected.altitude_m),
                "selected_reference_diagnostic_cost": float(
                    selected.native_rayleigh_candidate.diagnostic_cost
                ),
                "selected_reference_effective_resolution_m": float(
                    selected.effective_resolution_m
                ),
            }
        )
    return payload


def run_reference_domain_diagnostic(
    source: str | Path,
    *,
    wavelengths: Sequence[int] = (355, 532),
    search_min_altitudes_m: Sequence[float] = (6_000.0, 8_000.0, 10_000.0),
    search_max_altitude_m: float = 25_000.0,
    output_dir: str | Path = "v5_reference_diagnostics",
    config_path: str | Path = "config.yaml",
    station_config_path: str | Path | None = None,
    max_altitude_m: float = 30_000.0,
    block_start: int = 0,
    block_stop: int | None = None,
) -> list[Path]:
    """Report v5 candidate survival across lower search-domain bounds."""
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_root = Path(output_dir).expanduser().resolve()
    config = load_config(config_path, station_config_path)
    minutes = int(get_block_average_minutes(config))
    fit_cfg = get_molecular_fit_config(config)
    floors = tuple(float(value) for value in search_min_altitudes_m)
    if not floors or any(not np.isfinite(value) for value in floors):
        raise ValueError("search_min_altitudes_m must contain finite values.")
    if any(value >= float(search_max_altitude_m) for value in floors):
        raise ValueError("every search minimum must be below search_max_altitude_m.")

    logger = logging.getLogger("milgrau.v5_reference_diagnostic")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    written: list[Path] = []
    with xr.open_dataset(source_path, cache=False) as ds_source:
        ds, altitude_m = _altitude_subset(ds_source, max_altitude_m=float(max_altitude_m))
        if "time" not in ds:
            raise KeyError("Level 1 dataset lacks required time coordinate.")
        block_labels, groups = block_groups(ds["time"].values, minutes)
        first = max(int(block_start), 0)
        final = len(groups) if block_stop is None else min(int(block_stop), len(groups))
        if final <= first:
            raise ValueError(
                f"Empty block interval [{first}, {final}) for {len(groups)} available blocks."
            )

        for wavelength_nm in [int(value) for value in wavelengths]:
            blocks: list[dict[str, Any]] = []
            for global_index in range(first, final):
                group = groups[global_index]
                logger.info(
                    "%dnm | block %d/%d | %s | profiles=%d",
                    wavelength_nm,
                    global_index + 1,
                    len(groups),
                    str(block_labels[global_index]),
                    int(group.size),
                )
                ds_block = ds.isel(time=group)
                productive = process_wavelength(
                    ds_block,
                    wavelength_nm,
                    altitude_m,
                    config,
                    logger,
                )
                if productive.block_time.size != 1:
                    raise RuntimeError(
                        "Reference diagnostic expected exactly one configured temporal block."
                    )
                input_valid = bool(
                    int(productive.signal_selection.retrieval_input_valid_flag_block[0]) == 1
                )
                block_payload: dict[str, Any] = {
                    "block_index": global_index,
                    "block_time_utc": str(np.datetime_as_string(productive.block_time[0], unit="s")),
                    "retrieval_input_valid": input_valid,
                    "signal_source_flag": int(productive.signal_selection.source_flag_block[0]),
                    "v4_retrieval_success": bool(
                        int(productive.optical.retrieval_success_flag[0]) == 1
                    ),
                    "v4_reference_altitude_m": _finite_or_none(
                        productive.rayleigh.reference_altitude_m_block[0]
                    ),
                    "contiguous_top_altitude_m": None,
                    "domains": {},
                }
                if input_valid:
                    signal = np.asarray(
                        productive.glued.range_corrected_signal_block[0], dtype=np.float64
                    )
                    signal_error = np.asarray(
                        productive.glued.range_corrected_signal_error_block[0],
                        dtype=np.float64,
                    )
                    prepared = prepare_high_column_profile(
                        range_corrected_signal=signal,
                        range_corrected_signal_error=signal_error,
                        molecular_backscatter=np.asarray(
                            productive.molecular.backscatter, dtype=np.float64
                        ),
                        altitude_m=altitude_m,
                        uncertainty_mode="independent",
                    )
                    top_index = contiguous_usable_top_index(
                        prepared, path_start_altitude_m=600.0
                    )
                    if top_index is not None:
                        block_payload["contiguous_top_altitude_m"] = float(
                            prepared.grid.altitude_m[top_index]
                        )
                    for floor in floors:
                        catalogue = catalogue_high_column_reference_cells(
                            prepared=prepared,
                            native_range_corrected_signal=signal,
                            native_range_corrected_signal_error=signal_error,
                            native_simulated_molecular_signal=np.asarray(
                                productive.molecular.simulated_range_corrected_signal,
                                dtype=np.float64,
                            ),
                            native_altitude_m=altitude_m,
                            search_min_altitude_m=floor,
                            search_max_altitude_m=float(search_max_altitude_m),
                            rayleigh_window_m=float(fit_cfg["ref_window_m"]),
                            max_relative_slope=float(fit_cfg["max_relative_slope"]),
                            max_relative_variance=float(fit_cfg["max_relative_variance"]),
                            min_valid_fraction=float(fit_cfg["min_valid_fraction"]),
                            path_start_altitude_m=600.0,
                        )
                        block_payload["domains"][f"{floor:.1f}"] = _domain_summary(catalogue)
                blocks.append(block_payload)
                del productive, ds_block
                gc.collect()

            payload = {
                "source_level1": str(source_path),
                "wavelength_nm": wavelength_nm,
                "configured_block_minutes": minutes,
                "available_blocks": len(groups),
                "selected_block_start": first,
                "selected_block_stop": final,
                "max_altitude_m_loaded": float(altitude_m[-1]),
                "search_min_altitudes_m": list(floors),
                "search_max_altitude_m": float(search_max_altitude_m),
                "blocks": blocks,
            }
            output_path = output_root / (
                f"{source_path.stem}_v5_reference_diagnostic_{wavelength_nm}nm.json"
            )
            _strict_json_write(payload, output_path)
            written.append(output_path)
            logger.info("%dnm | done | %s", wavelength_nm, output_path)
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose real-Level-1 v5 reference survival without Monte Carlo."
    )
    parser.add_argument("source", help="Level-1 NetCDF path")
    parser.add_argument(
        "--wavelength",
        dest="wavelengths",
        nargs="+",
        type=int,
        default=[355, 532],
    )
    parser.add_argument(
        "--search-min-altitude-m",
        dest="search_min_altitudes_m",
        nargs="+",
        type=float,
        default=[6000.0, 8000.0, 10000.0],
    )
    parser.add_argument("--search-max-altitude-m", type=float, default=25000.0)
    parser.add_argument("--output-dir", default="v5_reference_diagnostics")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--station-config", default=None)
    parser.add_argument("--max-altitude-m", type=float, default=30000.0)
    parser.add_argument("--block-start", type=int, default=0)
    parser.add_argument("--block-stop", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = run_reference_domain_diagnostic(
        args.source,
        wavelengths=args.wavelengths,
        search_min_altitudes_m=args.search_min_altitudes_m,
        search_max_altitude_m=args.search_max_altitude_m,
        output_dir=args.output_dir,
        config_path=args.config,
        station_config_path=args.station_config,
        max_altitude_m=args.max_altitude_m,
        block_start=args.block_start,
        block_stop=args.block_stop,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
