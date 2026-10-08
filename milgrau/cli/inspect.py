"""Human-readable inspection of MILGRAU NetCDF products.

Designed for interactive use, presentations and quick product audits.  The
command reads NetCDF metadata lazily: large signal arrays are not loaded just
to display their structure.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import xarray as xr

from milgrau.cli.common import add_input_argument
from milgrau.config.loader import load_config
from milgrau.io.contracts import (
    validate_level0_contract,
    validate_level1_contract,
    validate_level2_contract,
)
from milgrau.io.paths import (
    LEVEL0_SUFFIX,
    LEVEL1_SUFFIX,
    LEVEL2_SUFFIX,
    product_session_id,
    session_id_parts,
)
from milgrau.io.selection import InputSelection, parse_input_selection, resolve_product_selection


_LEVEL_VALIDATORS = {
    "L0": validate_level0_contract,
    "L1": validate_level1_contract,
    "L2": validate_level2_contract,
}

_IMPORTANT_GLOBAL_ATTRS = (
    "Session_ID",
    "measurement_start_time",
    "measurement_end_time",
    "session_duration_seconds",
    "Processing_level",
    "Pipeline",
    "System",
    "Station_Profile",
    "station_profile_id",
    "instrument_calibration_id",
    "RawData_Start_Date",
    "RawData_Start_Time_UT",
    "RawData_Stop_Time_UT",
    "Input_Level0_File",
    "Input_Level1_File",
    "thermodynamic_profile_source_type",
    "thermodynamic_profile_available",
    "thermodynamic_profile_standard_fallback_fraction",
    "level2_product_schema_version",
    "level2_retrieval_method_version",
    "product_status",
    "product_completeness",
    "software_name",
    "software_version",
    "source_repository_revision",
    "source_code_identity",
)

_LEVEL_SIGNATURES = {
    "L0": {"Raw_Lidar_Data", "channel_string", "Laser_Shots"},
    "L1": {"corrected_signal", "range_corrected_signal", "Atmospheric_Temperature_K"},
    "L2": {"molecular_backscatter", "aerosol_backscatter_mean", "retrieval_success_flag"},
}


def _short(value: Any, *, width: int = 92) -> str:
    """Return one compact, single-line representation."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    text = str(value).replace("\n", " ").strip()
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def _dims_text(dims: Sequence[str], shape: Sequence[int]) -> str:
    if not dims:
        return "scalar"
    return " × ".join(f"{name}={size}" for name, size in zip(dims, shape, strict=True))


def _detect_level(path: Path, ds: xr.Dataset) -> str:
    name = path.name.lower()
    if "_l0" in name:
        return "L0"
    if "_l1" in name:
        return "L1"
    if "_l2" in name:
        return "L2"

    processing_level = str(ds.attrs.get("Processing_level", "")).lower()
    for level in ("L0", "L1", "L2"):
        if f"level {level[-1]}" in processing_level:
            return level

    names = set(ds.variables)
    scored = [
        (len(signature & names), level)
        for level, signature in _LEVEL_SIGNATURES.items()
    ]
    score, level = max(scored)
    return level if score >= 2 else "UNKNOWN"


def _preview_values(values: np.ndarray) -> list[Any]:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.datetime64):
        rendered = np.datetime_as_string(
            array.astype("datetime64[s]"),
            unit="s",
        )
        return np.asarray(rendered).reshape(-1).tolist()
    return array.reshape(-1).tolist()


def _coordinate_preview(da: xr.DataArray) -> str:
    size = int(da.size)
    if size == 0:
        return "empty"
    if da.ndim != 1:
        return f"{size} values"

    if size <= 6:
        return _short(_preview_values(np.asarray(da.values)))

    first = _preview_values(
        np.asarray(da.isel({da.dims[0]: slice(0, 3)}).values)
    )
    last = _preview_values(
        np.asarray(da.isel({da.dims[0]: slice(-3, None)}).values)
    )
    return _short(f"{first} … {last}")


def _session_summary(
    path: Path,
    ds: xr.Dataset,
    *,
    timezone_name: str | None = None,
) -> dict[str, Any] | None:
    """Return a compact session-level summary using product metadata and siblings."""
    session_id = str(ds.attrs.get("Session_ID", "")).strip()
    if not session_id:
        try:
            session_id = product_session_id(path)
        except ValueError:
            return None
    try:
        station, start_utc, end_utc = session_id_parts(session_id)
    except ValueError:
        return None

    resolved_timezone = str(timezone_name or ds.attrs.get("timezone", "")).strip()
    if not resolved_timezone:
        for suffix in (LEVEL1_SUFFIX, LEVEL0_SUFFIX):
            sibling = path.parent / f"{session_id}{suffix}"
            if not sibling.is_file() or sibling.resolve() == path.resolve():
                continue
            try:
                with xr.open_dataset(sibling, decode_times=False, cache=False) as sibling_ds:
                    resolved_timezone = str(sibling_ds.attrs.get("timezone", "")).strip()
            except Exception:
                resolved_timezone = ""
            if resolved_timezone:
                break
    if not resolved_timezone:
        resolved_timezone = "UTC"
    try:
        timezone = ZoneInfo(resolved_timezone)
    except Exception:
        timezone = ZoneInfo("UTC")
        resolved_timezone = "UTC"
    start_local = start_utc.astimezone(timezone)
    end_local = end_utc.astimezone(timezone)
    duration_seconds = float((end_utc - start_utc).total_seconds())
    total_minutes = int(round(duration_seconds / 60.0))
    hours, minutes = divmod(total_minutes, 60)
    duration = f"{hours}h{minutes:02d}" if hours else f"{minutes}min"

    regimes: list[str] = []
    segments: list[str] = []
    if "Segment_Label" in ds and "Segment_Regime" in ds:
        labels = np.asarray(ds["Segment_Label"].values).astype(str).reshape(-1)
        regime_values = np.asarray(ds["Segment_Regime"].values).astype(str).reshape(-1)
        segments = [
            f"{label} {regime}"
            for label, regime in zip(labels, regime_values, strict=False)
        ]
        regimes = list(dict.fromkeys(regime_values.tolist()))
    elif "solar_regime" in ds:
        values = np.asarray(ds["solar_regime"].values).astype(str).reshape(-1)
        regimes = list(dict.fromkeys(value for value in values if value))

    available_levels: list[str] = []
    for suffix, label in (
        (LEVEL0_SUFFIX, "L0"),
        (LEVEL1_SUFFIX, "L1"),
        (LEVEL2_SUFFIX, "L2"),
    ):
        if (path.parent / f"{session_id}{suffix}").is_file():
            available_levels.append(label)
    highest = available_levels[-1] if available_levels else level_from_path(path)

    figures_dir = path.parent / "figures"
    figures = (
        sorted(item.name for item in figures_dir.iterdir() if item.is_file())
        if figures_dir.is_dir()
        else []
    )
    return {
        "session_id": session_id,
        "human_label": (
            f"{station.upper()} · {start_local.strftime('%d/%m/%Y %H:%M')} → "
            f"{end_local.strftime('%d/%m/%Y %H:%M')} · {duration}"
        ),
        "utc_interval": (
            f"{start_utc.strftime('%Y-%m-%d %H:%M')} → "
            f"{end_utc.strftime('%Y-%m-%d %H:%M')} UTC"
        ),
        "timezone": resolved_timezone,
        "duration": duration,
        "regimes": ", ".join(regimes) if regimes else "--",
        "regime_sequence": (
            " → ".join(item.split(" ", 1)[1] for item in segments)
            if segments
            else ", ".join(regimes) if regimes else "--"
        ),
        "segments": " | ".join(segments) if segments else "--",
        "available_levels": ", ".join(available_levels) if available_levels else "--",
        "highest_level": highest,
        "figures": figures,
    }


def level_from_path(path: Path) -> str:
    name = path.name.lower()
    if "_l2" in name:
        return "L2"
    if "_l1" in name:
        return "L1"
    if "_l0" in name:
        return "L0"
    return "--"


def _print_session_summary(
    path: Path,
    ds: xr.Dataset,
    *,
    timezone_name: str | None = None,
    list_figures: bool = False,
) -> None:
    summary = _session_summary(path, ds, timezone_name=timezone_name)
    if summary is None:
        return
    product_marks = []
    available = set(summary["available_levels"].split(", "))
    for level in ("L0", "L1", "L2"):
        product_marks.append(f"{level} {'✓' if level in available else '–'}")

    print("\nSESSION")
    print("-" * 100)
    print(f"  ID             : {summary['session_id']}")
    print(f"  Local time     : {summary['human_label']}")
    print(f"  UTC            : {summary['utc_interval']}")
    print(f"  Time zone      : {summary['timezone']}")
    print(
        f"  Products       : {'  '.join(product_marks)}"
        f"   | highest {summary['highest_level']}"
    )
    print(f"  Solar sequence : {summary['regime_sequence']}")
    print(f"  Segments       : {summary['segments']}")
    print(f"  Figures        : {len(summary['figures'])} files")
    if list_figures:
        for figure in summary["figures"]:
            print(f"    - {figure}")


def _print_header(path: Path, ds: xr.Dataset, level: str) -> None:
    print()
    print("=" * 100)
    print("MILGRAU PRODUCT INSPECTOR")
    print("=" * 100)
    print(f"File           : {path}")
    print(f"Detected level : {level}")
    processing = ds.attrs.get("Processing_level")
    if processing:
        print(f"Processing     : {_short(processing, width=78)}")
    print(
        f"Inventory      : {len(ds.dims)} dimensions | "
        f"{len(ds.coords)} coordinates | {len(ds.data_vars)} data variables | "
        f"{len(ds.attrs)} global attributes"
    )


def _print_dimensions(ds: xr.Dataset) -> None:
    print("\nDIMENSIONS")
    print("-" * 100)
    for name, size in ds.sizes.items():
        print(f"  {name:<28} {size:>10}")


def _print_coordinates(ds: xr.Dataset, *, show_values: bool) -> None:
    print("\nCOORDINATES")
    print("-" * 100)
    if not ds.coords:
        print("  (none)")
        return

    for name, da in ds.coords.items():
        units = da.attrs.get("units", "")
        detail = f"dtype={da.dtype}; {_dims_text(da.dims, da.shape)}"
        if units:
            detail += f"; units={units}"
        print(f"  {name:<28} {detail}")
        if show_values:
            print(f"    preview: {_coordinate_preview(da)}")


def _variable_metadata(da: xr.DataArray) -> str:
    parts: list[str] = []
    units = da.attrs.get("units")
    long_name = da.attrs.get("long_name")
    if units not in (None, ""):
        parts.append(f"units={_short(units, width=24)}")
    if long_name not in (None, ""):
        parts.append(f"long_name={_short(long_name, width=48)}")
    return "; ".join(parts)


def _print_variables(ds: xr.Dataset, *, full: bool, max_vars: int) -> None:
    print("\nDATA VARIABLES")
    print("-" * 100)

    variables = list(ds.data_vars.items())
    shown = variables if full else variables[:max_vars]

    for name, da in shown:
        print(
            f"  {name:<42} dtype={str(da.dtype):<10} "
            f"{_dims_text(da.dims, da.shape)}"
        )
        metadata = _variable_metadata(da)
        if metadata:
            print(f"    {metadata}")
        if full:
            for key, value in da.attrs.items():
                if key in {"units", "long_name"}:
                    continue
                print(f"    @{key}: {_short(value)}")

    hidden = len(variables) - len(shown)
    if hidden > 0:
        print(f"\n  … {hidden} more variables hidden. Use --full to show all.")


def _print_global_attrs(ds: xr.Dataset, *, full: bool) -> None:
    print("\nGLOBAL METADATA")
    print("-" * 100)

    if full:
        items = list(ds.attrs.items())
    else:
        important = [
            (name, ds.attrs[name])
            for name in _IMPORTANT_GLOBAL_ATTRS
            if name in ds.attrs
        ]
        remaining_count = len(ds.attrs) - len(important)
        items = important

    if not items:
        print("  (none)")
    else:
        for key, value in items:
            print(f"  {key:<46} {_short(value)}")

    if not full and remaining_count > 0:
        print(f"\n  … {remaining_count} additional global attributes hidden. Use --full to show all.")


def _print_validation(ds: xr.Dataset, level: str) -> None:
    print("\nCONTRACT VALIDATION")
    print("-" * 100)

    validator = _LEVEL_VALIDATORS.get(level)
    if validator is None:
        print("  ? level could not be detected; validation skipped")
        return

    try:
        validator(ds)
    except Exception as exc:
        print(f"  FAIL — {type(exc).__name__}: {_short(exc, width=84)}")
    else:
        print(f"  PASS — dataset satisfies the current MILGRAU {level} contract")


def _is_canonical_product(path: Path) -> bool:
    """Return whether a path has a recognized MILGRAU L0/L1/L2 product name."""
    if not path.is_file() or path.suffix.lower() != ".nc":
        return False
    try:
        product_session_id(path)
    except ValueError:
        return False
    return True


def _expand_inputs(values: Sequence[Any], config: dict) -> list[Path]:
    """Resolve dates/session IDs/paths to inspectable NetCDF products."""
    for raw_group in values:
        tokens = raw_group if isinstance(raw_group, (list, tuple)) else [raw_group]
        for raw in tokens:
            text = str(raw).strip()
            if text.lower().endswith(".nc"):
                candidate = Path(text).expanduser()
                if not candidate.exists():
                    raise FileNotFoundError(f"NetCDF product not found: {candidate}")

    selection = parse_input_selection(values, config)
    product_selection = InputSelection(
        dates=selection.dates,
        session_ids=selection.session_ids,
        paths=(),
    )
    resolved: list[Path] = []
    for suffix in (LEVEL0_SUFFIX, LEVEL1_SUFFIX, LEVEL2_SUFFIX):
        try:
            resolved.extend(
                resolve_product_selection(product_selection, config, suffix=suffix)
            )
        except FileNotFoundError:
            continue

    for path in selection.paths:
        if path.is_dir():
            resolved.extend(sorted(item for item in path.rglob("*.nc") if item.is_file()))
        else:
            resolved.append(path)

    unique = sorted(dict.fromkeys(path.resolve() for path in resolved))
    if not unique:
        raise FileNotFoundError("No NetCDF products matched the requested selection.")
    return unique


def _print_selection(paths: Sequence[Path]) -> None:
    print()
    print("=" * 100)
    print(f"SELECTED PRODUCTS: {len(paths)}")
    print("=" * 100)
    for path in paths:
        print(f"  {path.name}")


def inspect_product(
    path: str | Path,
    *,
    full: bool = False,
    show_values: bool = True,
    validate: bool = False,
    max_vars: int = 30,
    timezone_name: str | None = None,
    show_session_summary: bool = True,
    list_figures: bool = False,
) -> None:
    """Print a compact structural summary of one MILGRAU NetCDF product."""
    product_path = Path(path).expanduser()
    if not product_path.is_file():
        raise FileNotFoundError(f"NetCDF product not found: {product_path}")

    with xr.open_dataset(product_path) as ds:
        level = _detect_level(product_path, ds)
        _print_header(product_path, ds, level)
        if show_session_summary:
            _print_session_summary(
                product_path,
                ds,
                timezone_name=timezone_name,
                list_figures=list_figures,
            )
        _print_dimensions(ds)
        _print_coordinates(ds, show_values=show_values)
        _print_variables(ds, full=full, max_vars=max_vars)
        _print_global_attrs(ds, full=full)
        if validate:
            _print_validation(ds, level)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="milgrau-inspect",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Inspect MILGRAU sessions and Level 0/1/2 products without loading "
            "the full signal arrays. The simplest selector is a local date: YYYYMMDD."
        ),
        epilog=(
            "Examples:\n"
            "  milgrau-inspect 20240620\n"
            "  milgrau-inspect spu_20240620-2200Z_20240621-0700Z\n"
            "  milgrau-inspect path/to/product_L2.nc --validate\n"
            "  milgrau-inspect 20240620 --full\n"
        ),
    )
    parser.add_argument(
        "selectors",
        nargs="*",
        help=(
            "Station-local date (YYYYMMDD), session ID, or an existing NetCDF "
            "file/directory. A date selects every session intersecting that local day."
        ),
    )
    add_input_argument(parser, source="Product selection")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show every data variable and all global/per-variable attributes.",
    )
    parser.add_argument(
        "--no-values",
        action="store_true",
        help="Do not show short coordinate-value previews.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the detected product against the current MILGRAU level contract.",
    )
    parser.add_argument(
        "--max-vars",
        type=int,
        default=30,
        help="Maximum data variables shown in compact mode (default: 30).",
    )
    return parser


def _station_timezone_from_config(config: dict) -> str | None:
    catalog = config.get("_station_catalog")
    if not isinstance(catalog, dict):
        return None
    station = catalog.get("station")
    if not isinstance(station, dict):
        return None
    value = str(station.get("timezone", "")).strip()
    return value or None


def _group_paths_by_session(paths: Sequence[Path]) -> list[list[Path]]:
    groups: dict[str, list[Path]] = {}
    loose: list[list[Path]] = []
    for path in paths:
        try:
            session_id = product_session_id(path)
        except ValueError:
            loose.append([path])
            continue
        groups.setdefault(session_id, []).append(path)
    ordered = [
        sorted(group, key=lambda item: level_from_path(item))
        for _session_id, group in sorted(groups.items())
    ]
    return ordered + loose


def _summary_source_path(group: Sequence[Path]) -> Path:
    rank = {"L0": 0, "L1": 1, "L2": 2}
    return max(group, key=lambda path: rank.get(level_from_path(path), -1))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.max_vars <= 0:
        parser.error("--max-vars must be positive")

    selector_groups: list[Any] = list(args.inputs)
    if args.selectors:
        selector_groups.append(args.selectors)
    if not selector_groups:
        parser.error("provide a date, session ID, file, or directory to inspect")

    try:
        config = load_config()
        paths = _expand_inputs(selector_groups, config)
    except Exception as exc:
        print()
        print("=" * 100)
        print("ERROR: product selection")
        print("=" * 100)
        print(f"{type(exc).__name__}: {exc}")
        return 1

    _print_selection(paths)
    timezone_name = _station_timezone_from_config(config)

    failures = 0
    for group in _group_paths_by_session(paths):
        summary_path = _summary_source_path(group)
        try:
            with xr.open_dataset(summary_path, cache=False) as summary_ds:
                _print_session_summary(
                    summary_path,
                    summary_ds,
                    timezone_name=timezone_name,
                    list_figures=args.full,
                )
        except Exception as exc:
            failures += 1
            print(f"\nWARNING: session summary unavailable: {type(exc).__name__}: {exc}")

        for path in group:
            try:
                inspect_product(
                    path,
                    full=args.full,
                    show_values=not args.no_values,
                    validate=args.validate,
                    max_vars=args.max_vars,
                    timezone_name=timezone_name,
                    show_session_summary=False,
                    list_figures=args.full,
                )
            except Exception as exc:
                failures += 1
                print()
                print("=" * 100)
                print(f"ERROR: {path}")
                print("=" * 100)
                print(f"{type(exc).__name__}: {exc}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
