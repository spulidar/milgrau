"""Low-memory command-line runner for real Level-1 method-v5 R&D validation.

The productive Level-2 pipeline loads a complete Level-1 dataset before
retrieval. That is convenient operationally but unnecessarily expensive for
method-v5 R&D on large files. This runner keeps the NetCDF lazy, restricts the
altitude domain, and processes one configured temporal block and one wavelength
at a time. It writes a small JSON checkpoint after every completed block.

This module is R&D-only. It does not alter productive method-v4 behavior.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr

from milgrau.config.loader import load_config
from milgrau.level2.block_average import block_groups
from milgrau.level2.config import get_block_average_minutes
from milgrau.level2.level1_v5_validation_rnd import (
    validate_level1_wavelength_v5_rnd,
    validation_summary_dict,
)


def _altitude_subset(
    ds: xr.Dataset,
    *,
    max_altitude_m: float,
) -> tuple[xr.Dataset, np.ndarray]:
    """Return a lazy contiguous altitude subset and its altitude coordinate in m."""
    if "altitude" not in ds:
        raise KeyError("Level 1 dataset lacks required altitude coordinate.")
    if ds["altitude"].dims != ("altitude",):
        raise ValueError("Level 1 altitude coordinate must have dimensions ('altitude',).")

    raw = np.asarray(ds["altitude"].values, dtype=np.float64)
    if raw.ndim != 1 or raw.size < 2 or not np.all(np.isfinite(raw)):
        raise ValueError("Level 1 altitude must be a finite one-dimensional coordinate.")
    altitude_m = raw * 1000.0 if float(np.nanmax(raw)) <= 100.0 else raw
    if not np.all(np.diff(altitude_m) > 0.0):
        raise ValueError("Level 1 altitude must be strictly increasing.")

    inside = np.flatnonzero(altitude_m <= float(max_altitude_m))
    if inside.size < 2:
        raise ValueError("Requested maximum altitude leaves fewer than two altitude bins.")
    stop = int(inside[-1]) + 1
    return ds.isel(altitude=slice(0, stop)), altitude_m[:stop]


def _strict_json_write(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid checkpoint root in {path}")
    return raw


def _output_path(output_dir: Path, source: Path, wavelength_nm: int) -> Path:
    return output_dir / f"{source.stem}_v5_validation_{int(wavelength_nm)}nm.json"


def run_low_memory_validation(
    source: str | Path,
    *,
    wavelengths: Sequence[int] = (355, 532),
    output_dir: str | Path = "v5_validation_results",
    config_path: str | Path = "config.yaml",
    station_config_path: str | Path | None = None,
    n_iterations: int = 60,
    residual_fractions: Sequence[float] = (0.0, 0.02, 0.05),
    reference_tier_min_altitudes_m: Sequence[float] = (
        10_000.0,
        9_000.0,
        8_000.0,
        6_000.0,
    ),
    beta_ref_relative_std: float = 0.0,
    max_altitude_m: float = 30_000.0,
    block_start: int = 0,
    block_stop: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Validate one Level-1 file block-by-block with bounded working memory."""
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_root = Path(output_dir).expanduser().resolve()
    config = load_config(config_path, station_config_path)
    minutes = int(get_block_average_minutes(config))
    iterations = int(n_iterations)
    tiers = tuple(float(value) for value in reference_tier_min_altitudes_m)
    if iterations <= 0:
        raise ValueError("n_iterations must be positive.")
    if not tiers:
        raise ValueError("reference_tier_min_altitudes_m must not be empty.")

    logger = logging.getLogger("milgrau.v5_validation")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    written: list[Path] = []
    with xr.open_dataset(source_path, cache=False) as ds_source:
        ds, altitude_m = _altitude_subset(
            ds_source,
            max_altitude_m=float(max_altitude_m),
        )
        if "time" not in ds:
            raise KeyError("Level 1 dataset lacks required time coordinate.")
        block_labels, groups = block_groups(ds["time"].values, minutes)
        first = max(int(block_start), 0)
        final = len(groups) if block_stop is None else min(int(block_stop), len(groups))
        if final <= first:
            raise ValueError(
                f"Empty block interval [{first}, {final}) for {len(groups)} available blocks."
            )

        logger.info(
            "source=%s | blocks=%d | selected=[%d,%d) | altitude_bins=%d | top=%.1f m",
            source_path.name,
            len(groups),
            first,
            final,
            altitude_m.size,
            float(altitude_m[-1]),
        )
        logger.info("reference tiers (m): %s", ", ".join(f"{v:g}" for v in tiers))

        for wavelength_nm in [int(value) for value in wavelengths]:
            output_path = _output_path(output_root, source_path, wavelength_nm)
            checkpoint = None if overwrite else _load_checkpoint(output_path)
            completed: dict[int, dict[str, Any]] = {}
            if checkpoint is not None:
                for block in checkpoint.get("blocks", []):
                    if isinstance(block, dict) and "block_index" in block:
                        completed[int(block["block_index"])] = block

            payload: dict[str, Any] = {
                "source_level1": str(source_path),
                "wavelength_nm": wavelength_nm,
                "configured_block_minutes": minutes,
                "available_blocks": len(groups),
                "selected_block_start": first,
                "selected_block_stop": final,
                "max_altitude_m_loaded": float(altitude_m[-1]),
                "n_iterations": iterations,
                "residual_fractions": [float(value) for value in residual_fractions],
                "reference_tier_min_altitudes_m": list(tiers),
                "beta_ref_relative_std": float(beta_ref_relative_std),
                "uncertainty_mode": "independent",
                "blocks": [completed[index] for index in sorted(completed)],
            }

            logger.info(
                "%dnm | checkpoint=%s | already_complete=%d",
                wavelength_nm,
                output_path.name,
                len(completed),
            )

            for global_index in range(first, final):
                if global_index in completed:
                    continue
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
                summary = validate_level1_wavelength_v5_rnd(
                    ds_block,
                    wavelength_nm,
                    altitude_m,
                    config,
                    logger,
                    residual_fractions=residual_fractions,
                    n_iterations=iterations,
                    beta_ref_relative_std=float(beta_ref_relative_std),
                    uncertainty_mode="independent",
                    reference_tier_min_altitudes_m=tiers,
                )
                if len(summary.blocks) != 1:
                    raise RuntimeError(
                        "Low-memory runner expected exactly one configured temporal block; "
                        f"received {len(summary.blocks)}."
                    )
                block_payload = validation_summary_dict(summary)["blocks"][0]
                block_payload["block_index"] = global_index
                completed[global_index] = block_payload
                payload.update(
                    {
                        "analog_channel": summary.analog_channel,
                        "photon_channel": summary.photon_channel,
                    }
                )
                payload["blocks"] = [completed[index] for index in sorted(completed)]
                _strict_json_write(payload, output_path)
                del summary, ds_block
                gc.collect()

            _strict_json_write(payload, output_path)
            written.append(output_path)
            logger.info("%dnm | done | %s", wavelength_nm, output_path)

    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run method-v5 real-Level-1 R&D validation one temporal block and "
            "one wavelength at a time, with JSON checkpoints."
        )
    )
    parser.add_argument("source", help="Level-1 NetCDF path")
    parser.add_argument(
        "--wavelength",
        dest="wavelengths",
        nargs="+",
        type=int,
        default=[355, 532],
        help="Wavelength(s) to validate; default: 355 532",
    )
    parser.add_argument("--output-dir", default="v5_validation_results")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--station-config", default=None)
    parser.add_argument("--n-iterations", type=int, default=60)
    parser.add_argument(
        "--residual-fractions",
        nargs="+",
        type=float,
        default=[0.0, 0.02, 0.05],
    )
    parser.add_argument(
        "--reference-tier-min-altitudes-m",
        nargs="+",
        type=float,
        default=[10_000.0, 9_000.0, 8_000.0, 6_000.0],
        help="Preferred-to-fallback reference-tier minima; default: 10000 9000 8000 6000",
    )
    parser.add_argument("--beta-ref-relative-std", type=float, default=0.0)
    parser.add_argument("--max-altitude-m", type=float, default=30_000.0)
    parser.add_argument("--block-start", type=int, default=0)
    parser.add_argument("--block-stop", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore an existing checkpoint and start this wavelength again.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = run_low_memory_validation(
        args.source,
        wavelengths=args.wavelengths,
        output_dir=args.output_dir,
        config_path=args.config,
        station_config_path=args.station_config,
        n_iterations=args.n_iterations,
        residual_fractions=args.residual_fractions,
        reference_tier_min_altitudes_m=args.reference_tier_min_altitudes_m,
        beta_ref_relative_std=args.beta_ref_relative_std,
        max_altitude_m=args.max_altitude_m,
        block_start=args.block_start,
        block_stop=args.block_stop,
        overwrite=args.overwrite,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
