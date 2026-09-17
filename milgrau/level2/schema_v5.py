"""Schema-4 contract validator for productive MILGRAU method-v5 Level 2 files."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import xarray as xr

from milgrau.scientific import LEVEL2_PRODUCT_SCHEMA_VERSION, LEVEL2_RETRIEVAL_METHOD_VERSION


_REQUIRED_VARIABLES: tuple[str, ...] = (
    "effective_vertical_resolution_m",
    "source_bin_count",
    "molecular_backscatter",
    "molecular_extinction",
    "lidar_ratio_assumed_sr",
    "lidar_ratio_std_sr",
    "range_corrected_signal_block",
    "range_corrected_signal_error_block",
    "aerosol_backscatter_nominal_block",
    "aerosol_extinction_nominal_block",
    "aerosol_backscatter_mc_mean",
    "aerosol_backscatter_mc_std",
    "aerosol_backscatter_mc_q025",
    "aerosol_backscatter_mc_q975",
    "aerosol_extinction_mc_mean",
    "aerosol_extinction_mc_std",
    "aerosol_extinction_mc_q025",
    "aerosol_extinction_mc_q975",
    "mc_valid_fraction",
    "aerosol_backscatter_mean",
    "aerosol_extinction_mean",
    "period_support_count",
    "period_support_fraction",
    "retrieval_top_altitude_m",
    "rayleigh_reference_altitude_m_block",
    "rayleigh_reference_tier_min_altitude_m_block",
    "rayleigh_reference_tier_index_block",
    "rayleigh_reference_fallback_used_block",
    "contiguous_path_top_altitude_m_block",
    "selection_success_fraction_block",
    "selected_reference_altitude_m_mc",
    "selected_reference_tier_min_altitude_m_mc",
    "selected_reference_tier_index_mc",
    "retrieval_input_valid_flag",
    "retrieval_input_invalid_reason",
    "retrieval_success_flag",
    "retrieval_success_fraction",
    "signal_source_flag",
    "gluing_attempted_flag",
    "gluing_success_flag",
    "single_channel_fallback_flag",
    "requested_wavelengths",
    "processed_wavelengths",
    "failed_wavelengths",
)


def _require_names(ds: xr.Dataset, names: Iterable[str], *, coords: bool = False) -> None:
    source = ds.coords if coords else ds
    missing = [name for name in names if name not in source]
    if missing:
        kind = "coordinate" if coords else "variable"
        raise KeyError(f"Method-v5 Level 2 lacks required {kind}(s): {missing}")


def _require_dims(ds: xr.Dataset, name: str, expected: tuple[str, ...]) -> None:
    observed = ds[name].dims
    if observed != expected:
        raise ValueError(f"Method-v5 {name} must have dimensions {expected}; got {observed}.")


def _finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    count = np.count_nonzero(finite, axis=axis)
    total = np.sum(np.where(finite, array, 0.0), axis=axis)
    out = np.full(np.asarray(total).shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _same_with_nan(left: np.ndarray, right: np.ndarray, *, atol: float = 0.0) -> bool:
    return bool(
        np.allclose(
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
            rtol=0.0,
            atol=atol,
            equal_nan=True,
        )
    )


def validate_method_v5_level2_contract(ds: xr.Dataset) -> None:
    """Validate method-v5 dimensions, support semantics and version identity."""
    if str(ds.attrs.get("level2_product_schema_version", "")) != LEVEL2_PRODUCT_SCHEMA_VERSION:
        raise ValueError(
            f"Method-v5 Level 2 requires schema {LEVEL2_PRODUCT_SCHEMA_VERSION}."
        )
    if str(ds.attrs.get("level2_retrieval_method_version", "")) != LEVEL2_RETRIEVAL_METHOD_VERSION:
        raise ValueError(
            f"Method-v5 Level 2 requires retrieval method {LEVEL2_RETRIEVAL_METHOD_VERSION}."
        )

    _require_names(ds, _REQUIRED_VARIABLES)
    _require_names(
        ds,
        ("block_time", "block_start_utc", "block_end_utc", "wavelength", "altitude", "residual_fraction", "mc_iteration"),
        coords=True,
    )

    wavelength_altitude = ("wavelength", "altitude")
    block_wavelength = ("block_time", "wavelength")
    block_wavelength_altitude = ("block_time", "wavelength", "altitude")
    block_wavelength_fraction_altitude = (
        "block_time",
        "wavelength",
        "residual_fraction",
        "altitude",
    )
    block_wavelength_iteration = ("block_time", "wavelength", "mc_iteration")

    for name in (
        "effective_vertical_resolution_m",
        "source_bin_count",
        "molecular_backscatter",
        "molecular_extinction",
        "aerosol_backscatter_mean",
        "aerosol_extinction_mean",
        "period_support_count",
        "period_support_fraction",
    ):
        _require_dims(ds, name, wavelength_altitude)
    for name in (
        "range_corrected_signal_block",
        "range_corrected_signal_error_block",
        "aerosol_backscatter_nominal_block",
        "aerosol_extinction_nominal_block",
    ):
        _require_dims(ds, name, block_wavelength_altitude)
    for name in (
        "aerosol_backscatter_mc_mean",
        "aerosol_backscatter_mc_std",
        "aerosol_backscatter_mc_q025",
        "aerosol_backscatter_mc_q975",
        "aerosol_extinction_mc_mean",
        "aerosol_extinction_mc_std",
        "aerosol_extinction_mc_q025",
        "aerosol_extinction_mc_q975",
        "mc_valid_fraction",
    ):
        _require_dims(ds, name, block_wavelength_fraction_altitude)
    for name in (
        "rayleigh_reference_altitude_m_block",
        "rayleigh_reference_tier_min_altitude_m_block",
        "rayleigh_reference_tier_index_block",
        "rayleigh_reference_fallback_used_block",
        "contiguous_path_top_altitude_m_block",
        "selection_success_fraction_block",
        "retrieval_input_valid_flag",
        "retrieval_input_invalid_reason",
        "retrieval_success_flag",
        "signal_source_flag",
        "gluing_attempted_flag",
        "gluing_success_flag",
        "single_channel_fallback_flag",
    ):
        _require_dims(ds, name, block_wavelength)
    for name in (
        "selected_reference_altitude_m_mc",
        "selected_reference_tier_min_altitude_m_mc",
        "selected_reference_tier_index_mc",
    ):
        _require_dims(ds, name, block_wavelength_iteration)
    for name in ("lidar_ratio_assumed_sr", "lidar_ratio_std_sr", "retrieval_top_altitude_m", "retrieval_success_fraction"):
        _require_dims(ds, name, ("wavelength",))

    altitude = np.asarray(ds["altitude"].values, dtype=np.float64)
    if altitude.ndim != 1 or altitude.size < 2 or np.any(~np.isfinite(altitude)) or np.any(np.diff(altitude) <= 0.0):
        raise ValueError("Method-v5 altitude must be finite, 1D, and strictly increasing.")
    resolution = np.asarray(ds["effective_vertical_resolution_m"].values, dtype=np.float64)
    source_count = np.asarray(ds["source_bin_count"].values, dtype=np.int64)
    if np.any(~np.isfinite(resolution)) or np.any(resolution <= 0.0):
        raise ValueError("effective_vertical_resolution_m must be finite and positive.")
    if np.any(source_count <= 0):
        raise ValueError("source_bin_count must be positive for every progressive cell.")

    residual = np.asarray(ds["residual_fraction"].values, dtype=np.float64)
    if residual.ndim != 1 or residual.size == 0 or np.any(~np.isfinite(residual)) or np.any(residual < 0.0):
        raise ValueError("residual_fraction must contain finite non-negative scenarios.")
    if not np.any(residual == 0.0) or np.unique(residual).size != residual.size:
        raise ValueError("residual_fraction must include unique nominal f=0.")

    binary_names = (
        "retrieval_input_valid_flag",
        "retrieval_success_flag",
        "rayleigh_reference_fallback_used_block",
        "gluing_attempted_flag",
        "gluing_success_flag",
        "single_channel_fallback_flag",
    )
    for name in binary_names:
        values = np.asarray(ds[name].values)
        if not np.isin(values, (0, 1)).all():
            raise ValueError(f"{name} must contain only 0 or 1.")

    retrieval_success = np.asarray(ds["retrieval_success_flag"].values, dtype=np.int8)
    expected_success_fraction = np.mean(retrieval_success == 1, axis=0)
    if not _same_with_nan(ds["retrieval_success_fraction"].values, expected_success_fraction):
        raise ValueError("retrieval_success_fraction must equal the successful block fraction.")

    nominal_beta = np.asarray(ds["aerosol_backscatter_nominal_block"].values, dtype=np.float64)
    expected_support_count = np.count_nonzero(np.isfinite(nominal_beta), axis=0).astype(np.int32)
    observed_support_count = np.asarray(ds["period_support_count"].values, dtype=np.int32)
    if not np.array_equal(observed_support_count, expected_support_count):
        raise ValueError("period_support_count must equal finite nominal block support at each altitude.")
    expected_support_fraction = expected_support_count.astype(np.float64) / float(ds.sizes["block_time"])
    if not _same_with_nan(ds["period_support_fraction"].values, expected_support_fraction):
        raise ValueError("period_support_fraction must equal period_support_count / block count.")

    expected_beta_mean = _finite_mean(nominal_beta, axis=0)
    if not _same_with_nan(ds["aerosol_backscatter_mean"].values, expected_beta_mean):
        raise ValueError("aerosol_backscatter_mean must be the finite-only nominal block mean.")
    nominal_alpha = np.asarray(ds["aerosol_extinction_nominal_block"].values, dtype=np.float64)
    expected_alpha_mean = _finite_mean(nominal_alpha, axis=0)
    if not _same_with_nan(ds["aerosol_extinction_mean"].values, expected_alpha_mean):
        raise ValueError("aerosol_extinction_mean must be the finite-only nominal block mean.")

    top = np.asarray(ds["retrieval_top_altitude_m"].values, dtype=np.float64)
    for wavelength_index in range(ds.sizes["wavelength"]):
        support = np.flatnonzero(observed_support_count[wavelength_index] > 0)
        expected_top = float(altitude[support[-1]]) if support.size else np.nan
        if not np.isclose(top[wavelength_index], expected_top, rtol=0.0, atol=1.0e-9, equal_nan=True):
            raise ValueError("retrieval_top_altitude_m must match the highest altitude with period support.")

    mc_valid = np.asarray(ds["mc_valid_fraction"].values, dtype=np.float64)
    finite_mc = np.isfinite(mc_valid)
    if np.any((mc_valid[finite_mc] < 0.0) | (mc_valid[finite_mc] > 1.0)):
        raise ValueError("mc_valid_fraction must lie between 0 and 1 where finite.")
    selection_fraction = np.asarray(ds["selection_success_fraction_block"].values, dtype=np.float64)
    finite_selection = np.isfinite(selection_fraction)
    if np.any((selection_fraction[finite_selection] < 0.0) | (selection_fraction[finite_selection] > 1.0)):
        raise ValueError("selection_success_fraction_block must lie between 0 and 1 where finite.")

    references = np.asarray(ds["rayleigh_reference_altitude_m_block"].values, dtype=np.float64)
    if np.any((retrieval_success == 1) & ~np.isfinite(references)):
        raise ValueError("Every successful method-v5 block requires a finite selected reference altitude.")

    requested = tuple(int(value) for value in np.asarray(ds["requested_wavelengths"].values).tolist())
    wavelength = tuple(int(value) for value in np.asarray(ds["wavelength"].values).tolist())
    processed = tuple(int(value) for value in np.asarray(ds["processed_wavelengths"].values).tolist())
    failed = tuple(int(value) for value in np.asarray(ds["failed_wavelengths"].values).tolist())
    if requested != wavelength:
        raise ValueError("Schema-4 wavelength coordinate must equal requested_wavelengths exactly.")
    expected_processed = tuple(
        wavelength[index]
        for index in range(len(wavelength))
        if np.any(retrieval_success[:, index] == 1)
    )
    expected_failed = tuple(value for value in wavelength if value not in expected_processed)
    if processed != expected_processed or failed != expected_failed:
        raise ValueError("processed_wavelengths/failed_wavelengths must match method-v5 block success.")
    expected_completeness = "complete" if not failed else "partial"
    expected_status = "success" if not failed else "partial"
    if str(ds.attrs.get("product_completeness", "")) != expected_completeness:
        raise ValueError("product_completeness is inconsistent with wavelength support.")
    if str(ds.attrs.get("product_status", "")) != expected_status:
        raise ValueError("product_status is inconsistent with wavelength support.")


__all__ = ["validate_method_v5_level2_contract"]
