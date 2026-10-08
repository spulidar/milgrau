"""NetCDF product contract validators for MILGRAU."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Final

import numpy as np
import xarray as xr

LEVEL0_SURFACE_WEATHER_VARIABLES: Final[tuple[str, ...]] = (
    "Surface_Temperature_C",
    "Surface_Pressure_hPa",
    "Surface_Relative_Humidity_percent",
    "Surface_Cloud_Cover_percent",
    "Surface_Wind_Speed_kmh",
)
LEVEL0_REQUIRED_VARIABLES: Final[tuple[str, ...]] = (
    "Raw_Data_Start_Time", "Raw_Data_Stop_Time", "Raw_Data_Range_Resolution",
    "Laser_Pointing_Angle", "Laser_Pointing_Angle_of_Profiles", "Laser_Shots",
    "Molecular_Calc", "id_timescale", "channel_string", "Raw_Lidar_Data",
) + LEVEL0_SURFACE_WEATHER_VARIABLES
LEVEL1_SIGNAL_VARIABLES: Final[tuple[str, ...]] = (
    "corrected_signal", "corrected_signal_error", "range_corrected_signal", "range_corrected_signal_error",
)
LEVEL1_BACKGROUND_TRACE_VARIABLES: Final[tuple[str, ...]] = (
    "signal_pre_background", "signal_pre_background_error",
)
LEVEL1_BACKGROUND_DIAGNOSTIC_VARIABLES: Final[tuple[str, ...]] = (
    "background_estimate", "background_standard_error", "background_robust_scale",
    "background_valid_bins", "background_outlier_fraction",
)
LEVEL1_ATMOSPHERIC_VARIABLES: Final[tuple[str, ...]] = (
    "Atmospheric_Temperature_K",
    "Atmospheric_Pressure_hPa",
    "Atmospheric_Source_Type",
    "Atmospheric_Source_Time_Delta_hours",
    "Atmospheric_USSA76_Fallback_Fraction",
)
LEVEL1_REQUIRED_VARIABLES: Final[tuple[str, ...]] = (
    LEVEL1_SIGNAL_VARIABLES + LEVEL1_ATMOSPHERIC_VARIABLES
)
LEVEL0_RAW_DATA_DIMS: Final[tuple[str, ...]] = ("time", "channels", "points")
LEVEL0_TIME_SCALE_DIMS: Final[tuple[str, ...]] = ("time", "nb_of_time_scales")
LEVEL0_BACKGROUND_DIMS: Final[tuple[str, ...]] = ("time_bck", "channels", "points")
LEVEL0_BACKGROUND_TIME_DIMS: Final[tuple[str, ...]] = ("time_bck", "nb_of_time_scales")
LEVEL0_BACKGROUND_LASER_SHOTS_DIMS: Final[tuple[str, ...]] = ("time_bck", "channels")
LEVEL0_CHANNEL_DIMS: Final[tuple[str, ...]] = ("channels",)
LEVEL0_LASER_SHOTS_DIMS: Final[tuple[str, ...]] = ("time", "channels")
LEVEL1_CORE_DIMS: Final[tuple[str, ...]] = ("time", "channel", "altitude")


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

    if "Background_Laser_Shots" in ds:
        _require_exact_dims(
            ds["Background_Laser_Shots"],
            LEVEL0_BACKGROUND_LASER_SHOTS_DIMS,
            "Level 0 Background_Laser_Shots",
        )
        shots = np.asarray(ds["Background_Laser_Shots"].values, dtype=np.float64)
        if shots.shape != (ds.sizes.get("time_bck", 0), ds.sizes.get("channels", 0)):
            raise ValueError("Level 0 Background_Laser_Shots shape must match time_bck x channels.")
        if "Background_Profile_Available" in ds:
            available = np.asarray(ds["Background_Profile_Available"].values, dtype=np.int8) == 1
            if np.any(available):
                available_shots = shots[:, available]
                if not np.all(np.isfinite(available_shots)) or np.any(available_shots <= 0.0):
                    raise ValueError(
                        "Level 0 Background_Laser_Shots must contain positive finite shot counts for every available dark channel."
                    )


def validate_level0_contract(ds: xr.Dataset) -> None:
    """Validate the Level 0 structure required by LIPANCORA and SCC handoff."""
    _require_variables(ds, LEVEL0_REQUIRED_VARIABLES, "Level 0 file")
    _require_dims(ds, LEVEL0_RAW_DATA_DIMS + ("nb_of_time_scales", "scan_angles"), "Level 0 file")
    _require_exact_dims(ds["Raw_Lidar_Data"], LEVEL0_RAW_DATA_DIMS, "Level 0 Raw_Lidar_Data")
    _require_exact_dims(ds["Raw_Data_Start_Time"], LEVEL0_TIME_SCALE_DIMS, "Level 0 Raw_Data_Start_Time")
    _require_exact_dims(ds["Raw_Data_Stop_Time"], LEVEL0_TIME_SCALE_DIMS, "Level 0 Raw_Data_Stop_Time")
    _require_exact_dims(ds["Laser_Pointing_Angle_of_Profiles"], LEVEL0_TIME_SCALE_DIMS, "Level 0 Laser_Pointing_Angle_of_Profiles")
    _validate_level0_scc_acquisition_metadata(ds)
    _require_coords(ds, ("weather_time",), "Level 0 file")
    for name in LEVEL0_SURFACE_WEATHER_VARIABLES:
        _require_exact_dims(ds[name], ("weather_time",), f"Level 0 {name}")
        values = np.asarray(ds[name].values, dtype=np.float64)
        if values.shape != (ds.sizes.get("weather_time", 0),):
            raise ValueError(f"Level 0 {name} must contain one value per weather_time entry.")
    _validate_level0_background_contract(ds)


def _validate_level1_atmosphere(ds: xr.Dataset) -> None:
    _require_coords(ds, ("atmosphere_time",), "Level 1 atmosphere")
    expected = ("atmosphere_time", "altitude")
    n_time = ds.sizes.get("atmosphere_time", 0)
    n_altitude = ds.sizes.get("altitude", 0)
    if n_time <= 0:
        raise ValueError("Level 1 atmosphere_time must contain at least one source time.")

    for name in ("Atmospheric_Temperature_K", "Atmospheric_Pressure_hPa"):
        _require_exact_dims(ds[name], expected, f"Level 1 {name}")
        values = np.asarray(ds[name].values, dtype=np.float64)
        if values.shape != (n_time, n_altitude):
            raise ValueError(
                f"Level 1 {name} must contain one complete profile per atmosphere_time."
            )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"Level 1 {name} must be finite and positive everywhere.")

    for name in (
        "Atmospheric_Source_Type",
        "Atmospheric_Source_Time_Delta_hours",
        "Atmospheric_USSA76_Fallback_Fraction",
    ):
        _require_exact_dims(ds[name], ("atmosphere_time",), f"Level 1 {name}")

    source_values = {str(value) for value in np.asarray(ds["Atmospheric_Source_Type"].values).reshape(-1)}
    if not source_values or not source_values.issubset({"era5", "ussa76"}):
        raise ValueError(
            "Level 1 Atmospheric_Source_Type may contain only productive sources era5 and ussa76."
        )

    fallback = np.asarray(ds["Atmospheric_USSA76_Fallback_Fraction"].values, dtype=np.float64)
    if not np.all(np.isfinite(fallback)) or np.any((fallback < 0.0) | (fallback > 1.0)):
        raise ValueError(
            "Level 1 Atmospheric_USSA76_Fallback_Fraction must be finite and between 0 and 1."
        )

    source_type = str(ds.attrs.get("thermodynamic_profile_source_type", "")).strip()
    if source_type != "time_resolved":
        raise ValueError(
            "Level 1 thermodynamic_profile_source_type must be 'time_resolved'."
        )
    if str(ds.attrs.get("thermodynamic_profile_available", "")).lower() != "true":
        raise ValueError("Level 1 canonical atmospheric profile must be materialized and marked available.")
    try:
        fallback_fraction = float(ds.attrs["thermodynamic_profile_standard_fallback_fraction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Level 1 lacks a valid thermodynamic_profile_standard_fallback_fraction attribute.") from exc
    if not np.isfinite(fallback_fraction) or not 0.0 <= fallback_fraction <= 1.0:
        raise ValueError("Level 1 thermodynamic_profile_standard_fallback_fraction must be between 0 and 1.")


def validate_level1_contract(ds: xr.Dataset) -> None:
    """Validate Level 1 signals plus the mandatory materialized atmosphere."""
    _require_variables(ds, LEVEL1_REQUIRED_VARIABLES, "Level 1 file")
    _require_coords(ds, LEVEL1_CORE_DIMS, "Level 1 file")
    reference = ds["range_corrected_signal"].transpose(*LEVEL1_CORE_DIMS)
    reference_shape = reference.shape
    for name in LEVEL1_SIGNAL_VARIABLES:
        _require_named_dim_set(ds[name], LEVEL1_CORE_DIMS, f"Level 1 {name}")
        if ds[name].transpose(*LEVEL1_CORE_DIMS).shape != reference_shape:
            raise ValueError(f"Level 1 {name} shape does not match range_corrected_signal shape by named dimensions.")
    # Files produced before the robust-background implementation remain valid.
    # New files must, however, carry the complete provenance group: accepting a
    # partial group would make the subtraction impossible to audit reliably.
    background_group = LEVEL1_BACKGROUND_TRACE_VARIABLES + LEVEL1_BACKGROUND_DIAGNOSTIC_VARIABLES
    background_presence = tuple(name in ds for name in background_group)
    if any(background_presence) and not all(background_presence):
        missing = [name for name, present in zip(background_group, background_presence, strict=True) if not present]
        raise KeyError(f"Level 1 file has an incomplete background provenance group; missing {missing}.")
    if all(background_presence):
        for name in LEVEL1_BACKGROUND_TRACE_VARIABLES:
            _require_named_dim_set(ds[name], LEVEL1_CORE_DIMS, f"Level 1 {name}")
            if ds[name].transpose(*LEVEL1_CORE_DIMS).shape != reference_shape:
                raise ValueError(f"Level 1 {name} shape does not match range_corrected_signal shape by named dimensions.")
        for name in LEVEL1_BACKGROUND_DIAGNOSTIC_VARIABLES:
            _require_exact_dims(ds[name], ("time", "channel"), f"Level 1 {name}")
    _validate_level1_atmosphere(ds)


def validate_level2_contract(ds: xr.Dataset) -> None:
    """Validate the sole current Level 2 contract."""
    from milgrau.level2.level2_schema import validate_level2_contract as validate_current

    validate_current(ds)


def netcdf_satisfies_contract(path: str | Path, validator: Callable[[xr.Dataset], None]) -> bool:
    try:
        with xr.open_dataset(path) as dataset:
            dataset.load()
            validator(dataset)
    except Exception:
        return False
    return True
