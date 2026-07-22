"""NetCDF product contract validators for MILGRAU.

The validators in this module check structural requirements shared by multiple
pipeline stages.  They intentionally avoid enforcing every possible metadata
attribute, but they do fail early on missing variables, dimensions and core
coordinates that would make downstream processing scientifically ambiguous.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Final

import numpy as np
import xarray as xr


LEVEL0_REQUIRED_VARIABLES: Final[tuple[str, ...]] = (
    "Raw_Data_Start_Time",
    "Raw_Data_Stop_Time",
    "Raw_Data_Range_Resolution",
    "Laser_Pointing_Angle",
    "Laser_Pointing_Angle_of_Profiles",
    "Laser_Shots",
    "Molecular_Calc",
    "id_timescale",
    "channel_string",
    "Raw_Lidar_Data",
)
LEVEL1_REQUIRED_VARIABLES: Final[tuple[str, ...]] = (
    "corrected_signal",
    "corrected_signal_error",
    "range_corrected_signal",
    "range_corrected_signal_error",
)
LEVEL2_REQUIRED_VARIABLES: Final[tuple[str, ...]] = (
    "molecular_backscatter",
    "molecular_extinction",
    "glued_range_corrected_signal",
    "aerosol_backscatter_mean",
    "aerosol_extinction_mean",
    "gluing_attempted_flag",
    "gluing_success_flag",
    "single_channel_fallback_flag",
    "signal_source_flag",
    "retrieval_input_valid_flag",
    "retrieval_input_invalid_reason",
    "retrieval_success_flag",
    "retrieval_success_fraction",
    "requested_wavelengths",
    "processed_wavelengths",
    "failed_wavelengths",
    "failed_wavelength_stage",
    "failed_wavelength_code",
    "failed_wavelength_message",
    "failed_wavelength_cause",
)
LEVEL0_RAW_DATA_DIMS: Final[tuple[str, ...]] = ("time", "channels", "points")
LEVEL0_TIME_SCALE_DIMS: Final[tuple[str, ...]] = ("time", "nb_of_time_scales")
LEVEL0_BACKGROUND_DIMS: Final[tuple[str, ...]] = ("time_bck", "channels", "points")
LEVEL1_CORE_DIMS: Final[tuple[str, ...]] = ("time", "channel", "altitude")
LEVEL2_GLUED_SIGNAL_DIMS: Final[tuple[str, ...]] = ("time", "wavelength", "altitude")
LEVEL2_TIME_STATE_DIMS: Final[tuple[str, ...]] = ("time", "wavelength")
LEVEL2_BLOCK_STATE_DIMS: Final[tuple[str, ...]] = ("block_time", "wavelength")


def _missing_names(ds: xr.Dataset, names: Iterable[str]) -> list[str]:
    """Return names that are absent from an xarray Dataset."""
    return [name for name in names if name not in ds]


def _require_variables(ds: xr.Dataset, names: Iterable[str], product_name: str) -> None:
    """Raise a KeyError if one or more required variables are absent."""
    missing = _missing_names(ds, names)
    if missing:
        raise KeyError(f"{product_name} lacks required variable(s): {missing}")


def _require_coords(ds: xr.Dataset, names: Iterable[str], product_name: str) -> None:
    """Raise a KeyError if one or more required coordinates are absent."""
    missing = [name for name in names if name not in ds.coords]
    if missing:
        raise KeyError(f"{product_name} lacks required coordinate(s): {missing}")


def _require_dims(ds: xr.Dataset, names: Iterable[str], product_name: str) -> None:
    """Raise a KeyError if one or more required dimensions are absent."""
    missing = [name for name in names if name not in ds.dims]
    if missing:
        raise KeyError(f"{product_name} lacks required dimension(s): {missing}")


def _require_exact_dims(data_array: xr.DataArray, expected_dims: tuple[str, ...], label: str) -> None:
    """Raise a ValueError when one variable does not follow the exact canonical dimension order."""
    if data_array.dims != expected_dims:
        raise ValueError(f"{label} must have dimensions {expected_dims}; got {data_array.dims}.")


def _require_named_dim_set(data_array: xr.DataArray, expected_dims: tuple[str, ...], label: str) -> None:
    """Raise a ValueError when one variable lacks required named dimensions."""
    expected_dim_set = set(expected_dims)
    if set(data_array.dims) != expected_dim_set:
        raise ValueError(f"{label} must contain dimensions {expected_dims}; got {data_array.dims}.")


def validate_level0_contract(ds: xr.Dataset) -> None:
    """Validate the minimum Level 0 structure required by LIPANCORA."""
    _require_variables(ds, LEVEL0_REQUIRED_VARIABLES, "Level 0 file")
    _require_dims(ds, LEVEL0_RAW_DATA_DIMS + ("nb_of_time_scales", "scan_angles"), "Level 0 file")
    _require_exact_dims(ds["Raw_Lidar_Data"], LEVEL0_RAW_DATA_DIMS, "Level 0 Raw_Lidar_Data")
    _require_exact_dims(ds["Raw_Data_Start_Time"], LEVEL0_TIME_SCALE_DIMS, "Level 0 Raw_Data_Start_Time")
    _require_exact_dims(ds["Raw_Data_Stop_Time"], LEVEL0_TIME_SCALE_DIMS, "Level 0 Raw_Data_Stop_Time")
    _require_exact_dims(ds["Laser_Pointing_Angle_of_Profiles"], LEVEL0_TIME_SCALE_DIMS, "Level 0 Laser_Pointing_Angle_of_Profiles")
    if "Background_Profile" in ds:
        _require_exact_dims(ds["Background_Profile"], LEVEL0_BACKGROUND_DIMS, "Level 0 Background_Profile")


def validate_level1_contract(ds: xr.Dataset) -> None:
    """Validate the minimum Level 1 structure required by LIRACOS and LEBEAR.

    The canonical MILGRAU order for signal tensors is
    ``(time, channel, altitude)``.  This validator accepts any xarray dimension
    order as long as the required named dimensions and coordinates are present
    and all four core signal tensors are mutually conformable after transposing
    by name.  This keeps older intermediate files readable while downstream code
    can still transpose explicitly to canonical order before saving.
    """
    _require_variables(ds, LEVEL1_REQUIRED_VARIABLES, "Level 1 file")
    _require_coords(ds, LEVEL1_CORE_DIMS, "Level 1 file")
    reference = ds["range_corrected_signal"].transpose(*LEVEL1_CORE_DIMS)
    reference_shape = reference.shape
    for name in LEVEL1_REQUIRED_VARIABLES:
        _require_named_dim_set(ds[name], LEVEL1_CORE_DIMS, f"Level 1 {name}")
        if ds[name].transpose(*LEVEL1_CORE_DIMS).shape != reference_shape:
            raise ValueError(f"Level 1 {name} shape does not match range_corrected_signal shape by named dimensions.")


def validate_level2_contract(ds: xr.Dataset) -> None:
    """Validate the minimum Level 2 optical-product structure."""
    _require_variables(ds, LEVEL2_REQUIRED_VARIABLES, "Level 2 file")
    _require_coords(ds, ("wavelength", "altitude"), "Level 2 file")
    if "glued_range_corrected_signal" in ds:
        _require_exact_dims(ds["glued_range_corrected_signal"], LEVEL2_GLUED_SIGNAL_DIMS, "Level 2 glued_range_corrected_signal")
    for name in (
        "gluing_attempted_flag",
        "gluing_success_flag",
        "single_channel_fallback_flag",
        "signal_source_flag",
        "retrieval_input_valid_flag",
        "retrieval_input_invalid_reason",
    ):
        _require_exact_dims(ds[name], LEVEL2_TIME_STATE_DIMS, f"Level 2 {name}")
    _require_exact_dims(
        ds["retrieval_success_flag"],
        LEVEL2_BLOCK_STATE_DIMS,
        "Level 2 retrieval_success_flag",
    )
    _require_exact_dims(
        ds["retrieval_success_fraction"],
        ("wavelength",),
        "Level 2 retrieval_success_fraction",
    )
    _validate_level2_completeness(ds)


def _validate_level2_completeness(ds: xr.Dataset) -> None:
    """Validate SCI-003 lists, diagnostics and scientific wavelength membership."""
    # Imported lazily to avoid the level2 package's public orchestration imports
    # forming a cycle while this shared I/O contract module is initialized.
    from milgrau.level2.completeness import (
        Level2ProductContract,
        ProductCompleteness,
        ProductStatus,
        WavelengthFailureCode,
        WavelengthFailureDiagnostic,
        WavelengthFailureStage,
    )

    expected_dims = {
        "requested_wavelengths": ("requested_wavelength",),
        "processed_wavelengths": ("processed_wavelength",),
        "failed_wavelengths": ("failed_wavelength",),
        "failed_wavelength_stage": ("failed_wavelength",),
        "failed_wavelength_code": ("failed_wavelength",),
        "failed_wavelength_message": ("failed_wavelength",),
        "failed_wavelength_cause": ("failed_wavelength",),
    }
    for name, dims in expected_dims.items():
        _require_exact_dims(ds[name], dims, f"Level 2 {name}")

    try:
        completeness = ProductCompleteness(str(ds.attrs["product_completeness"]))
        product_status = ProductStatus(str(ds.attrs["product_status"]))
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "Level 2 product_completeness/product_status attributes are missing or invalid."
        ) from exc
    requested = tuple(int(value) for value in np.asarray(ds["requested_wavelengths"].values).tolist())
    processed = tuple(int(value) for value in np.asarray(ds["processed_wavelengths"].values).tolist())
    failed = tuple(int(value) for value in np.asarray(ds["failed_wavelengths"].values).tolist())
    stages = np.asarray(ds["failed_wavelength_stage"].values).tolist()
    codes = np.asarray(ds["failed_wavelength_code"].values).tolist()
    messages = np.asarray(ds["failed_wavelength_message"].values).astype(str).tolist()
    causes = np.asarray(ds["failed_wavelength_cause"].values).astype(str).tolist()
    diagnostics: list[WavelengthFailureDiagnostic] = []
    try:
        diagnostics = [
            WavelengthFailureDiagnostic(
                wavelength_nm=wavelength,
                stage=WavelengthFailureStage(int(stage)),
                code=WavelengthFailureCode(int(code)),
                message=message,
                cause_summary=cause,
            )
            for wavelength, stage, code, message, cause in zip(
                failed,
                stages,
                codes,
                messages,
                causes,
                strict=True,
            )
        ]
        product_contract = Level2ProductContract(
            requested_wavelengths=requested,
            processed_wavelengths=processed,
            failed_wavelengths=failed,
            completeness=completeness,
            product_status=product_status,
            failure_diagnostics=tuple(diagnostics),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Level 2 multispectral completeness contract: {exc}") from exc
    if product_contract.completeness is ProductCompleteness.FAILED:
        raise ValueError("A published Level 2 scientific file requires at least one processed wavelength.")
    scientific_wavelengths = tuple(int(value) for value in np.asarray(ds["wavelength"].values).tolist())
    if scientific_wavelengths != product_contract.processed_wavelengths:
        raise ValueError("Scientific wavelength coordinate must equal processed_wavelengths exactly.")
    retrieval_success = np.asarray(ds["retrieval_success_flag"].values, dtype=np.int8)
    if retrieval_success.ndim != 2 or retrieval_success.shape[1] != len(scientific_wavelengths):
        raise ValueError("retrieval_success_flag is not conformable with processed wavelengths.")
    if np.any(np.sum(retrieval_success == 1, axis=0) == 0):
        raise ValueError("Every processed wavelength requires at least one successful optical block.")
    expected_fraction = np.mean(retrieval_success == 1, axis=0)
    observed_fraction = np.asarray(ds["retrieval_success_fraction"].values, dtype=np.float64)
    if not np.allclose(observed_fraction, expected_fraction, rtol=0.0, atol=0.0):
        raise ValueError("retrieval_success_fraction must equal the successful block fraction.")


def netcdf_satisfies_contract(path: str | Path, validator: Callable[[xr.Dataset], None]) -> bool:
    """Return whether a NetCDF file opens fully and satisfies one product contract."""
    try:
        with xr.open_dataset(path) as dataset:
            dataset.load()
            validator(dataset)
    except Exception:
        return False
    return True
