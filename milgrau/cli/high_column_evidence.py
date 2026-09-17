"""Explicit offline R&D command, separate from LEBEAR processing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from milgrau.level2.high_column_export import (
    candidate_evidence_rows,
    evidence_summary,
    write_evidence_export,
)
from milgrau.provenance import file_sha256, source_code_provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export schema-3 candidate evidence (offline R&D only).")
    parser.add_argument("level2", type=Path)
    parser.add_argument("output_dir", type=Path, help="New directory; existing outputs are never replaced.")
    parser.add_argument("--block-weights", type=float, nargs="+", required=True)
    parser.add_argument("--weight-basis", required=True)
    parser.add_argument("--effective-vertical-resolution-m", type=float, required=True)
    parser.add_argument("--resolution-basis", required=True)
    parser.add_argument("--altitude-edges-m", type=float, nargs="+", required=True)
    parser.add_argument("--expected-sha256", help="Fail if the input differs from the frozen product.")
    args = parser.parse_args(argv)
    if not args.weight_basis.strip() or not args.resolution_basis.strip():
        parser.error("Weight and resolution provenance must be non-empty.")
    digest = file_sha256(args.level2)
    if args.expected_sha256 is not None and digest != args.expected_sha256.lower():
        parser.error("Input SHA-256 does not match the expected frozen product.")
    with xr.open_dataset(args.level2) as ds:
        rows = candidate_evidence_rows(
            ds, block_weights=np.asarray(args.block_weights),
            effective_vertical_resolution_m=args.effective_vertical_resolution_m,
        )
        provenance = {
            "source_level2_filename": args.level2.name, "source_level2_sha256": digest,
            "source_product_identity": {
                key: str(ds.attrs[key]) for key in (
                    "level2_product_schema_version", "level2_retrieval_method_version",
                    "source_repository_revision", "source_code_sha256",
                ) if key in ds.attrs
            },
            "exporter_source": source_code_provenance(),
            "block_times": [str(value) for value in ds.block_time.values],
            "block_weights": args.block_weights, "weight_basis": args.weight_basis,
            "effective_vertical_resolution_m": args.effective_vertical_resolution_m,
            "resolution_basis": args.resolution_basis,
            "definitions": {
                "persistence": "weight fraction accepting this exact window; null if any block unevaluated",
                "dominance": "max weight*abs(RCS) fraction at exact center on signal/error common support",
                "subwindow_disagreement": "abs(C_lower-C_upper)/C_full; contiguous disjoint positional halves",
                "calibration_support": "finite positive measured RCS, molecular RCS and propagated error; at least 2 bins",
                "boundary": "productive exact measured bin unchanged; fitted factors diagnostic only",
                "missing": "JSON null / empty CSV cell; never zero uncertainty or favorable evidence",
                "contamination": "not evaluated; subwindow agreement cannot establish molecular purity",
                "covariance": "independent and fully correlated limits; no empirical model supplied",
                "resolution": "caller-declared; native spacing alone does not validate instrument resolution",
            },
        }
    if file_sha256(args.level2) != digest:
        raise ValueError("Input changed during export; rerun against an immutable product.")
    summary = evidence_summary(rows, np.asarray(args.altitude_edges_m))
    write_evidence_export(args.output_dir, rows, summary, provenance)
    print(f"Exported {len(rows)} candidate slots to {args.output_dir} (diagnostic-only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
