"""Diagnose Analog/Photon-counting gluing window selection on real Level 1 data.

This script intentionally mirrors the LEBEAR gluing pre-processing but prints why
candidate windows are rejected.  It is meant for field-data tuning, not for
routine processing.

Example
-------
python scripts/diagnose_gluing.py \
    02-processed_data/2024/06/20240606sant/20240606sant_level1_rcs.nc \
    --config config.yaml \
    --wavelength 532
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from milgrau.physics.gluing import slide_glue_signals


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _infer_channel_pair(ds_l1: xr.Dataset, wavelength_nm: int) -> tuple[str | None, str | None]:
    channels = [str(channel) for channel in ds_l1["channel"].values]
    prefix = f"{int(wavelength_nm)}."
    analog = next((channel for channel in channels if channel.startswith(prefix) and channel.upper().endswith(".AN")), None)
    photon = next(
        (
            channel
            for channel in channels
            if channel.startswith(prefix) and (channel.upper().endswith(".PC") or channel.upper().endswith(".PH"))
        ),
        None,
    )
    return analog, photon


def _get_gluing_config(config: dict[str, Any]) -> dict[str, Any]:
    gluing_cfg = config.get("inversion", {}).get("gluing", {}) or {}
    return {
        "window_size": int(gluing_cfg.get("window_length_bins", 150)),
        "min_corr": float(gluing_cfg.get("correlation_threshold", 0.95)),
        "search_min_idx": int(gluing_cfg.get("search_min_idx", 200)),
        "search_max_idx": int(gluing_cfg.get("search_max_idx", 2000)),
        "intercept_threshold": float(gluing_cfg.get("intercept_threshold", 5.0)),
        "gaussian_threshold": float(gluing_cfg.get("gaussian_threshold", 0.1)),
        "minmax_threshold": float(gluing_cfg.get("minmax_threshold", 0.05)),
        "max_relative_rmse": float(gluing_cfg.get("max_relative_rmse", 0.08)),
        "max_relative_bias": float(gluing_cfg.get("max_relative_bias", 0.05)),
        "min_valid_fraction": float(gluing_cfg.get("min_valid_fraction", 0.80)),
        "max_saturation_fraction": float(gluing_cfg.get("max_saturation_fraction", 0.20)),
    }


def _block_groups(time_values: np.ndarray, minutes: int) -> tuple[np.ndarray, list[np.ndarray]]:
    times = pd.to_datetime(time_values)
    labels = times.floor(f"{int(minutes)}min")
    unique_labels = pd.Index(labels).unique().sort_values()
    groups = [np.where(labels == label)[0] for label in unique_labels]
    return unique_labels.to_numpy(dtype="datetime64[ns]"), groups


def _nanmean_or_nan(matrix: np.ndarray, axis: int = 0) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    valid = np.isfinite(arr)
    count = valid.sum(axis=axis)
    total = np.nansum(arr, axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float64), where=count > 0)


def _mean_by_groups(matrix: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    return np.stack([_nanmean_or_nan(matrix[group, :], axis=0) for group in groups], axis=0)


def _mask_by_groups(mask_matrix: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    mask = np.asarray(mask_matrix, dtype=bool)
    return np.stack([np.any(mask[group, :], axis=0) for group in groups], axis=0)


def _modified_regression(analog: np.ndarray, photon: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(analog) & np.isfinite(photon)
    x = photon[valid]
    y = analog[valid]
    if x.size < 4 or float(np.nanstd(x)) <= 0.0 or float(np.nanstd(y)) <= 0.0:
        return np.nan, np.nan
    a_prime, b_prime = np.polyfit(x, y, 1)
    if not np.isfinite(a_prime) or abs(float(a_prime)) <= 1.0e-30:
        return np.nan, np.nan
    return float(1.0 / a_prime), float(-b_prime / a_prime)


def _window_metrics(analog: np.ndarray, photon: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(analog) & np.isfinite(photon) & ~mask
    valid_count = int(valid.sum())
    saturation_fraction = float(np.mean(mask)) if mask.size else 1.0
    if valid_count < 4:
        return {
            "valid_count": valid_count,
            "saturation_fraction": saturation_fraction,
            "correlation": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "relative_rmse": np.inf,
            "relative_bias": np.inf,
            "intercept_percent": np.inf,
            "dynamic_range_ratio": np.nan,
        }

    a = analog[valid]
    p = photon[valid]
    if float(np.nanstd(a)) <= 0.0 or float(np.nanstd(p)) <= 0.0:
        return {
            "valid_count": valid_count,
            "saturation_fraction": saturation_fraction,
            "correlation": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "relative_rmse": np.inf,
            "relative_bias": np.inf,
            "intercept_percent": np.inf,
            "dynamic_range_ratio": np.nan,
        }

    slope, intercept = _modified_regression(a, p)
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0.0:
        return {
            "valid_count": valid_count,
            "saturation_fraction": saturation_fraction,
            "correlation": np.nan,
            "slope": slope,
            "intercept": intercept,
            "relative_rmse": np.inf,
            "relative_bias": np.inf,
            "intercept_percent": np.inf,
            "dynamic_range_ratio": np.nan,
        }

    virtual_pc = slope * a + intercept
    residual = virtual_pc - p
    scale = float(np.nanmean(np.abs(p)))
    if not np.isfinite(scale) or scale <= 1.0e-30:
        scale = float(np.nanmax(np.abs(p)))
    if not np.isfinite(scale) or scale <= 1.0e-30:
        scale = 1.0

    return {
        "valid_count": valid_count,
        "saturation_fraction": saturation_fraction,
        "correlation": float(np.corrcoef(a, p)[0, 1]),
        "slope": float(slope),
        "intercept": float(intercept),
        "relative_rmse": float(np.sqrt(np.nanmean(residual**2)) / scale),
        "relative_bias": float(np.nanmean(residual) / scale),
        "intercept_percent": float(abs(intercept) / scale * 100.0),
        "dynamic_range_ratio": float((np.nanmax(p) - np.nanmin(p)) / scale),
    }


def _failure_reasons(metrics: dict[str, float | int], gluing_cfg: dict[str, Any], window: int) -> list[str]:
    reasons: list[str] = []
    min_valid_count = max(int(np.ceil(float(gluing_cfg["min_valid_fraction"]) * window)), 4)
    if float(metrics["saturation_fraction"]) > float(gluing_cfg["max_saturation_fraction"]):
        reasons.append("saturation")
    if int(metrics["valid_count"]) < min_valid_count:
        reasons.append("valid_count")
    if not np.isfinite(float(metrics["correlation"])) or float(metrics["correlation"]) < float(gluing_cfg["min_corr"]):
        reasons.append("correlation")
    if not np.isfinite(float(metrics["slope"])) or float(metrics["slope"]) <= 0.0:
        reasons.append("slope")
    if not np.isfinite(float(metrics["intercept_percent"])) or float(metrics["intercept_percent"]) > float(gluing_cfg["intercept_threshold"]):
        reasons.append("intercept")
    if not np.isfinite(float(metrics["dynamic_range_ratio"])) or float(metrics["dynamic_range_ratio"]) < float(gluing_cfg["minmax_threshold"]):
        reasons.append("dynamic_range")
    if not np.isfinite(float(metrics["relative_rmse"])) or float(metrics["relative_rmse"]) > float(gluing_cfg["max_relative_rmse"]):
        reasons.append("rmse")
    if not np.isfinite(float(metrics["relative_bias"])) or abs(float(metrics["relative_bias"])) > float(gluing_cfg["max_relative_bias"]):
        reasons.append("bias")
    return reasons


def _scan_windows(
    analog: np.ndarray,
    photon: np.ndarray,
    mask: np.ndarray,
    altitude_m: np.ndarray,
    gluing_cfg: dict[str, Any],
    top_n: int = 5,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    window = max(int(gluing_cfg["window_size"]), 4)
    if window % 2:
        window += 1
    start = max(int(gluing_cfg["search_min_idx"]), 0)
    stop = min(int(gluing_cfg["search_max_idx"]), analog.size)
    counts = {
        "candidate": 0,
        "pass": 0,
        "saturation": 0,
        "valid_count": 0,
        "correlation": 0,
        "slope": 0,
        "intercept": 0,
        "dynamic_range": 0,
        "rmse": 0,
        "bias": 0,
    }
    ranked: list[dict[str, Any]] = []
    for idx in range(start, max(start, stop - window + 1)):
        counts["candidate"] += 1
        metrics = _window_metrics(analog[idx : idx + window], photon[idx : idx + window], mask[idx : idx + window])
        reasons = _failure_reasons(metrics, gluing_cfg, window)
        if not reasons:
            counts["pass"] += 1
        else:
            for reason in set(reasons):
                counts[reason] += 1
        score = (
            float(metrics["relative_rmse"])
            + abs(float(metrics["relative_bias"]))
            + 0.001 * float(metrics["intercept_percent"])
            + 0.01 * float(metrics["saturation_fraction"])
        )
        if not np.isfinite(score):
            score = 1.0e99
        ranked.append(
            {
                "idx": idx,
                "center_alt_km": float(altitude_m[idx + window // 2] / 1000.0),
                "start_alt_km": float(altitude_m[idx] / 1000.0),
                "stop_alt_km": float(altitude_m[min(idx + window - 1, altitude_m.size - 1)] / 1000.0),
                "score": score,
                "reasons": ",".join(reasons) if reasons else "PASS",
                **metrics,
            }
        )
    ranked.sort(key=lambda item: item["score"])
    return counts, ranked[:top_n]


def _fmt(value: Any, precision: int = 4) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        value_f = float(value)
    except Exception:
        return str(value)
    if np.isfinite(value_f):
        return f"{value_f:.{precision}g}"
    return str(value_f)


def diagnose(path: Path, config: dict[str, Any], wavelength: int, top_n: int) -> None:
    with xr.open_dataset(path) as ds:
        ds.load()

    altitude_m = np.asarray(ds["altitude"].values, dtype=np.float64)
    if np.nanmax(altitude_m) <= 100.0:
        altitude_m = altitude_m * 1000.0

    analog_ch, photon_ch = _infer_channel_pair(ds, wavelength)
    if analog_ch is None or photon_ch is None:
        raise RuntimeError(f"Could not infer AN/PC pair for {wavelength} nm. Found: AN={analog_ch}, PC={photon_ch}")

    block_minutes = int(config.get("inversion", {}).get("block_average_minutes", config.get("inversion", {}).get("temporal_average_minutes", 15)))
    block_time, groups = _block_groups(ds["time"].values, block_minutes)
    gluing_cfg = _get_gluing_config(config)

    analog = ds["corrected_signal"].sel(channel=analog_ch).values.astype(np.float64)
    photon = ds["corrected_signal"].sel(channel=photon_ch).values.astype(np.float64)
    analog_block = _mean_by_groups(analog, groups)
    photon_block = _mean_by_groups(photon, groups)
    if "pc_saturation_mask" in ds:
        mask = ds["pc_saturation_mask"].sel(channel=photon_ch).values.astype(bool)
        mask_block = _mask_by_groups(mask, groups)
    else:
        mask_block = np.zeros_like(photon_block, dtype=bool)

    print(f"File: {path}")
    print(f"Wavelength: {wavelength} nm | AN={analog_ch} | PC={photon_ch}")
    print(f"Blocks: {len(groups)} | block_average_minutes={block_minutes}")
    print("Gluing config:")
    for key, value in gluing_cfg.items():
        print(f"  {key}: {value}")
    print()

    success_count = 0
    aggregate_counts: dict[str, int] = {}
    for block_idx, time_value in enumerate(block_time):
        _, split_point, slope, intercept, diagnostics = slide_glue_signals(
            analog_sig=analog_block[block_idx, :],
            pc_sig=photon_block[block_idx, :],
            altitude=altitude_m,
            window_size=gluing_cfg["window_size"],
            min_corr=gluing_cfg["min_corr"],
            search_min_idx=gluing_cfg["search_min_idx"],
            search_max_idx=gluing_cfg["search_max_idx"],
            intercept_threshold=gluing_cfg["intercept_threshold"],
            gaussian_threshold=gluing_cfg["gaussian_threshold"],
            minmax_threshold=gluing_cfg["minmax_threshold"],
            max_relative_rmse=gluing_cfg["max_relative_rmse"],
            max_relative_bias=gluing_cfg["max_relative_bias"],
            min_valid_fraction=gluing_cfg["min_valid_fraction"],
            max_saturation_fraction=gluing_cfg["max_saturation_fraction"],
            pc_saturation_mask=mask_block[block_idx, :],
            return_diagnostics=True,
        )
        if split_point >= 0:
            success_count += 1
        counts, top = _scan_windows(
            analog_block[block_idx, :],
            photon_block[block_idx, :],
            mask_block[block_idx, :],
            altitude_m,
            gluing_cfg,
            top_n=top_n,
        )
        for key, value in counts.items():
            aggregate_counts[key] = aggregate_counts.get(key, 0) + int(value)

        print("=" * 100)
        print(f"Block {block_idx:03d} | {pd.to_datetime(time_value)} | operational_split={split_point} | slope={slope:.4g} | intercept={intercept:.4g}")
        print(
            "slide diagnostics: "
            f"candidates={diagnostics.get('candidate_count')} evaluated={diagnostics.get('evaluated_count')} "
            f"sat_rejected={diagnostics.get('saturation_rejected_count')} "
            f"corr={_fmt(diagnostics.get('best_corr'))} rmse={_fmt(diagnostics.get('relative_rmse'))} "
            f"bias={_fmt(diagnostics.get('relative_bias'))} intercept%={_fmt(diagnostics.get('intercept_percent'))} "
            f"dyn={_fmt(diagnostics.get('dynamic_range_ratio'))}"
        )
        print("manual rejection counts:", ", ".join(f"{key}={value}" for key, value in counts.items()))
        print("top candidate windows:")
        for item in top:
            print(
                "  "
                f"idx={item['idx']:5d} alt={item['start_alt_km']:.2f}-{item['stop_alt_km']:.2f} km "
                f"score={_fmt(item['score'])} reasons={item['reasons']} "
                f"sat={_fmt(item['saturation_fraction'])} valid={item['valid_count']} "
                f"corr={_fmt(item['correlation'])} rmse={_fmt(item['relative_rmse'])} "
                f"bias={_fmt(item['relative_bias'])} int%={_fmt(item['intercept_percent'])} "
                f"dyn={_fmt(item['dynamic_range_ratio'])} slope={_fmt(item['slope'])}"
            )

    print("=" * 100)
    print(f"Operational success blocks: {success_count}/{len(groups)} ({100.0 * success_count / max(len(groups), 1):.1f}%)")
    print("Aggregate rejection counts:", ", ".join(f"{key}={value}" for key, value in aggregate_counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose LEBEAR gluing window selection for one Level 1 file.")
    parser.add_argument("level1_file", type=Path, help="Path to *_level1_rcs.nc")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to MILGRAU config.yaml")
    parser.add_argument("--wavelength", type=int, required=True, help="Wavelength in nm, e.g. 355 or 532")
    parser.add_argument("--top", type=int, default=5, help="Number of best candidate windows to print per block")
    args = parser.parse_args()
    diagnose(args.level1_file, _load_config(args.config), args.wavelength, args.top)


if __name__ == "__main__":
    main()
