"""QA plot for the block-resolved method-v5 R&D NetCDF product."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _qa_output_path(source: Path, output: str | Path | None = None) -> Path:
    if output is not None:
        return Path(output).expanduser().resolve()
    return source.with_name(source.stem + "_qa.png")


def _time_label(value: np.datetime64) -> str:
    if np.isnat(value):
        return "NaT"
    return str(np.datetime_as_string(value, unit="m")).replace("T", " ")


def plot_v5_rnd_qa(
    source: str | Path,
    *,
    output: str | Path | None = None,
    max_altitude_km: float = 12.0,
) -> Path:
    """Plot period-mean optical profiles together with altitude support."""
    source_path = Path(source).expanduser().resolve()
    output_path = _qa_output_path(source_path, output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(source_path) as ds:
        altitude_km = np.asarray(ds["altitude"].values, dtype=np.float64) / 1000.0
        beta = np.asarray(
            ds["period_mean_aerosol_backscatter_nominal"].values,
            dtype=np.float64,
        ) * 1.0e6
        alpha = np.asarray(
            ds["period_mean_aerosol_extinction_nominal"].values,
            dtype=np.float64,
        ) * 1.0e6
        support = np.asarray(ds["period_support_fraction"].values, dtype=np.float64)
        support_count = np.asarray(ds["period_support_count"].values, dtype=np.int32)
        block_start = np.asarray(ds["block_start_utc"].values).astype("datetime64[ns]")
        block_end = np.asarray(ds["block_end_utc"].values).astype("datetime64[ns]")
        reference = np.asarray(ds["reference_altitude_m"].values, dtype=np.float64)
        tier = np.asarray(ds["reference_tier_min_altitude_m"].values, dtype=np.float64)
        success = np.asarray(ds["v5_success"].values, dtype=np.int8)
        wavelength = int(ds.attrs.get("wavelength_nm", 0))
        n_blocks = int(ds.sizes["block"])

    if block_start.size:
        period = f"{_time_label(block_start[0])} to {_time_label(block_end[-1])} UTC"
    else:
        period = "empty period"

    fig, axes = plt.subplots(1, 3, figsize=(15, 9), sharey=True)
    ax_beta, ax_alpha, ax_support = axes
    ax_beta.plot(beta, altitude_km, linewidth=2.0)
    ax_alpha.plot(alpha, altitude_km, linewidth=2.0)
    ax_support.plot(100.0 * support, altitude_km, linewidth=2.0)

    ax_beta.axvline(0.0, color="black", linewidth=0.8)
    ax_alpha.axvline(0.0, color="black", linewidth=0.8)
    ax_beta.set_xlabel(r"$\beta_{aer}$ [Mm$^{-1}$ sr$^{-1}$]")
    ax_alpha.set_xlabel(r"$\alpha_{aer}$ [Mm$^{-1}$]")
    ax_support.set_xlabel("Temporal-block support [%]")
    ax_beta.set_ylabel("Altitude [km a.g.l.]")
    ax_beta.set_title("Period mean backscatter")
    ax_alpha.set_title("Period mean extinction")
    ax_support.set_title("Altitude-resolved support")
    ax_support.set_xlim(-2.0, 102.0)
    ax_beta.set_ylim(0.0, float(max_altitude_km))
    for axis in axes:
        axis.grid(True, alpha=0.3)

    finite_support = np.isfinite(support)
    for count_value in np.unique(support_count[finite_support]):
        if count_value <= 0:
            continue
        indices = np.flatnonzero(support_count == count_value)
        if indices.size:
            altitude = altitude_km[indices[-1]]
            if altitude <= float(max_altitude_km):
                ax_support.text(
                    min(100.0 * count_value / max(n_blocks, 1) + 2.0, 96.0),
                    altitude,
                    f"{count_value}/{n_blocks}",
                    fontsize=8,
                    va="center",
                )

    rows: list[str] = []
    for index in range(n_blocks):
        start = _time_label(block_start[index]).split(" ")[-1]
        end = _time_label(block_end[index]).split(" ")[-1]
        if success[index] and np.isfinite(reference[index]):
            rows.append(
                f"B{index}: {start}-{end}  ref={reference[index] / 1000.0:.2f} km  "
                f"tier>={tier[index] / 1000.0:.1f} km"
            )
        else:
            rows.append(f"B{index}: {start}-{end}  unsupported")

    fig.suptitle(
        f"MILGRAU method-v5 R&D QA - {wavelength} nm\n{period}",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        " | ".join(rows),
        ha="center",
        va="bottom",
        fontsize=8,
        wrap=True,
    )
    fig.text(
        0.5,
        0.055,
        (
            "Period means are finite-only altitude-by-altitude. Read the support panel "
            "jointly: a high-altitude feature may represent only a subset of temporal blocks."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.10, 0.99, 0.92))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one method-v5 R&D profile NetCDF.")
    parser.add_argument("source", help="method-v5 R&D NetCDF")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-altitude-km", type=float, default=12.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    path = plot_v5_rnd_qa(
        args.source,
        output=args.output,
        max_altitude_km=args.max_altitude_km,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
