"""NetCDF product contract validators for MILGRAU."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Final

import numpy as np
import xarray as xr

LEVEL0_REQUIRED_VARIABLES: Final[tuple[str, ...]] = (
    "Raw_Data_Start_Time", "Raw_Data_Stop_Time", "Raw_Data_Range_Resolution",
    "Laser_Pointing_Angle", "Laser_Pointing_Angle_of_Profiles", "Laser_Shots",
    "Molecular_Calc", "id_timescale", "channel_string", "Raw_Lidar_Data",
)
LEVEL1_REQUIRED_VARIABLES: Final[tuple[str, ...]] = (
    "corrected_signal", "corrected_signal_error", "range_corrected_signal", "range_corrected_signal_error",
)
LEVEL2_REQUIRED_VARIABLES: Final[tuple[str, ...]] = (
    "molecular_backscatter", "molecular_extinction", "glued_range_corrected_signal",
    "aerosol_backscatter_mean", "aerosol_extinction_mean", "gluing_attempted_flag",
    "gluing_success_flag", "single_channel_fallback_flag", "signal_source_flag",
    "retrieval_input_valid_flag", "retrieval_input_invalid_reason", "retrieval_success_flag",
    "retrieval_success_fraction", "requested_wavelengths", "processed_wavelengths",
    "failed_wavelengths", "failed_wavelength_stage", "failed_wavelength_code",
    "failed_wavelength_message", "failed_wavelength_cause",
)
LEVEL0_RAW_DATA_DIMS: Final[tuple[str, ...]] = ("time", "channels", "points")
LEVEL0_TIME_SCALE_DIMS: Final[tuple[str, ...]] = ("time", "nb_of_time_scales")
LEVEL0_BACKGROUND_DIMS: Final[tuple[str, ...]] = ("time_bck", "channels", "points")
LEVEL0_BACKGROUND_TIME_DIMS: Final[tuple[str, ...]] = ("time_bck", "nb_of_time_scales")
LEVEL0_CHANNEL_DIMS: Final[tuple[str, ...]] = ("channels",)
LEVEL0_LASER_SHOTS_DIMS: Final[tuple[str, ...]] = ("time", "channels")
LEVEL1_CORE_DIMS: Final[tuple[str, ...]] = ("time", "channel", "altitude")
LEVEL2_GLUED_SIGNAL_DIMS: Final[tuple[str, ...]] = ("time", "wavelength", "altitude")
LEVEL2_TIME_STATE_DIMS: Final[tuple[str, ...]] = ("time", "wavelength")
LEVEL2_BLOCK_STATE_DIMS: Final[tuple[str, ...]] = ("block_time", "wavelength")


def _missing_names(ds: xr.Dataset, names: Iterable[str]) -> list[str]:
    return [name for name in names if name not in ds]


def _require_variables(ds: xr.Dataset, names: Iterable[str], product_name: str) -> None:
    missing = _missing_names(ds, names)
    if missing:
        raise KeyError(f"{product_name} lacks required variable(s): {missing}")


def _require_coords(ds: xr.Dataset, names: Iterable[str], product_name: str) -> None:
    missing = [name for name in names if name not in ds.coords]
    if missing:
        raise KeyError(f"{product_name} lacks required coordinate(s): {missing}")


def _require_dims(ds: xr.Dataset, names: Iterable[str], product_name: str) -> None:
    missing = [name for name in names if name not in ds.dims]
    if missing:
        raise KeyError(f"{product_name} lacks required dimension(s): {missing}")


def _require_exact_dims(data_array: xr.DataArray, expected_dims: tuple[str, ...], label: str) -> None:
    if data_array.dims != expected_dims:
        raise ValueError(f"{label} must have dimensions {expected_dims}; got {data_array.dims}.")


def _require_named_dim_set(data_array: xr.DataArray, expected_dims: tuple[str, ...], label: str) -> None:
    if set(data_array.dims) != set(expected_dims):
        raise ValueError(f"{label} must contain dimensions {expected_dims}; got {data_array.dims}.")


def _level0_channel_names(ds: xr.Dataset) -> np.ndarray:
    values = np.asarray(ds["channel_string"].values).astype(str)
    if values.ndim != 1 or values.size != ds.sizes.get("channels", 0):
        raise ValueError("Level 0 channel_string must contain exactly one value per channels entry.")
    return values


def _validate_level0_scc_acquisition_metadata(ds: xr.Dataset) -> None:
    for name in ("Raw_Data_Range_Resolution", "id_timescale", "channel_string"):
        _require_exact_dims(ds[name], LEVEL0_CHANNEL_DIMS, f"Level 0 {name}")
    _require_exact_dims(ds["Laser_Shots"], LEVEL0_LASER_SHOTS_DIMS, "Level 0 Laser_Shots")
    resolutions = np.asarray(ds["Raw_Data_Range_Resolution"].values, dtype=np.float64)
    if resolutions.size != ds.sizes.get("channels", 0) or not np.all(np.isfinite(resolutions)) or np.any(resolutions <= 0.0):
        raise ValueError("Level 0 Raw_Data_Range_Resolution must contain one positive finite value per channel.")
    laser_shots = np.asarray(ds["Laser_Shots"].values, dtype=np.float64)
    if not np.all(np.isfinite(laser_shots)) or np.any(laser_shots <= 0.0):
        raise ValueError("Level 0 Laser_Shots must contain positive finite shot counts for every stored profile/channel.")
    channel_names = _level0_channel_names(ds)
    analog_indices = [index for index, name in enumerate(channel_names) if name.upper().endswith(".AN")]
    if analog_indices:
        if "DAQ_Range" not in ds:
            raise KeyError("Level 0 file has analog channel(s) but lacks SCC-required DAQ_Range.")
        _require_exact_dims(ds["DAQ_Range"], LEVEL0_CHANNEL_DIMS, "Level 0 DAQ_Range")
        daq_range = np.asarray(ds["DAQ_Range"].values, dtype=np.float64)
        analog_values = daq_range[np.asarray(analog_indices, dtype=np.int64)]
        if not np.all(np.isfinite(analog_values)) or np.any(analog_values <= 0.0):
            raise ValueError("Level 0 DAQ_Range must contain a positive finite mV scale for every analog channel.")


def _validate_level0_background_contract(ds: xr.Dataset) -> None:
    if "Background_Profile" not in ds:
        return
    _require_exact_dims(ds["Background_Profile"], LEVEL0_BACKGROUND_DIMS, "Level 0 Background_Profile")
    _require_variables(ds, ("Raw_Bck_Start_Time", "Raw_Bck_Stop_Time"), "Level 0 file with Background_Profile")
    _require_exact_dims(ds["Raw_Bck_Start_Time"], LEVEL0_BACKGROUND_TIME_DIMS, "Level 0 Raw_Bck_Start_Time")
    _require_exact_dims(ds["Raw_Bck_Stop_Time"], LEVEL0_BACKGROUND_TIME_DIMS, "Level 0 Raw_Bck_Stop_Time")
    missing_attrs = [name for name in ("RawBck_Start_Date", "RawBck_Start_Time_UT", "RawBck_Stop_Time_UT") if not str(ds.attrs.get(name, "")).strip()]
    if missing_attrs:
        raise KeyError(f"Level 0 file with Background_Profile lacks SCC background attribute(s): {missing_attrs}")


def validate_level0_contract(ds: xr.Dataset) -> None:
    """Validate the Level 0 structure required by LIPANCORA and SCC handoff."""
    _require_variables(ds, LEVEL0_REQUIRED_VARIABLES, "Level 0 file")
    _require_dims(ds, LEVEL0_RAW_DATA_DIMS + ("nb_of_time_scales", "scan_angles"), "Level 0 file")
    _require_exact_dims(ds["Raw_Lidar_Data"], LEVEL0_RAW_DATA_DIMS, "Level 0 Raw_Lidar_Data")
    _require_exact_dims(ds["Raw_Data_Start_Time"], LEVEL0_TIME_SCALE_DIMS, "Level 0 Raw_Data_Start_Time")
    _require_exact_dims(ds["Raw_Data_Stop_Time"], LEVEL0_TIME_SCALE_DIMS, "Level 0 Raw_Data_Stop_Time")
    _require_exact_dims(ds["Laser_Pointing_Angle_of_Profiles"], LEVEL0_TIME_SCALE_DIMS, "Level 0 Laser_Pointing_Angle_of_Profiles")
    _validate_level0_scc_acquisition_metadata(ds)
    _validate_level0_background_contract(ds)


def validate_level1_contract(ds: xr.Dataset) -> None:
    _require_variables(ds, LEVEL1_REQUIRED_VARIABLES, "Level 1 file")
    _require_coords(ds, LEVEL1_CORE_DIMS, "Level 1 file")
    reference = ds["range_corrected_signal"].transpose(*LEVEL1_CORE_DIMS)
    reference_shape = reference.shape
    for name in LEVEL1_REQUIRED_VARIABLES:
        _require_named_dim_set(ds[name], LEVEL1_CORE_DIMS, f"Level 1 {name}")
        if ds[name].transpose(*LEVEL1_CORE_DIMS).shape != reference_shape:
            raise ValueError(f"Level 1 {name} shape does not match range_corrected_signal shape by named dimensions.")


def validate_level2_contract(ds: xr.Dataset) -> None:
    _require_variables(ds, LEVEL2_REQUIRED_VARIABLES, "Level 2 file")
    _require_coords(ds, ("wavelength", "altitude"), "Level 2 file")
    _require_exact_dims(ds["glued_range_corrected_signal"], LEVEL2_GLUED_SIGNAL_DIMS, "Level 2 glued_range_corrected_signal")
    for name in ("gluing_attempted_flag", "gluing_success_flag", "single_channel_fallback_flag", "signal_source_flag", "retrieval_input_valid_flag", "retrieval_input_invalid_reason"):
        _require_exact_dims(ds[name], LEVEL2_TIME_STATE_DIMS, f"Level 2 {name}")
    _require_exact_dims(ds["retrieval_success_flag"], LEVEL2_BLOCK_STATE_DIMS, "Level 2 retrieval_success_flag")
    _require_exact_dims(ds["retrieval_success_fraction"], ("wavelength",), "Level 2 retrieval_success_fraction")
    _validate_level2_completeness(ds)


def _validate_level2_completeness(ds: xr.Dataset) -> None:
    from milgrau.level2.completeness import (
        Level2ProductContract, ProductCompleteness, ProductStatus,
        WavelengthFailureCode, WavelengthFailureDiagnostic, WavelengthFailureStage,
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
        raise ValueError("Level 2 product_completeness/product_status attributes are missing or invalid.") from exc
    requested = tuple(int(value) for value in np.asarray(ds["requested_wavelengths"].values).tolist())
    processed = tuple(int(value) for value in np.asarray(ds["processed_wavelengths"].values).tolist())
    failed = tuple(int(value) for value in np.asarray(ds["failed_wavelengths"].values).tolist())
    stages = np.asarray(ds["failed_wavelength_stage"].values).tolist()
    codes = np.asarray(ds["failed_wavelength_code"].values).tolist()
    messages = np.asarray(ds["failed_wavelength_message"].values).astype(str).tolist()
    causes = np.asarray(ds["failed_wavelength_cause"].values).astype(str).tolist()
    try:
        diagnostics = [
            WavelengthFailureDiagnostic(
                wavelength_nm=wavelength,
                stage=WavelengthFailureStage(int(stage)),
                code=WavelengthFailureCode(int(code)),
                message=message,
                cause_summary=cause,
            )
            for wavelength, stage, code, message, cause in zip(failed, stages, codes, messages, causes, strict=True)
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
    try:
        with xr.open_dataset(path) as dataset:
            dataset.load()
            validator(dataset)
    except Exception:
        return False
    return True
