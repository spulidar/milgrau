"""Level 0 NetCDF writing and provenance helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final, Mapping

import netCDF4 as nc
import numpy as np
import pandas as pd

from milgrau.io.licel import parse_licel_group

DEFAULT_CHANNEL_ID: Final[int] = 9999
DEFAULT_VERTICAL_RESOLUTION_M: Final[float] = 7.5
DEFAULT_BACKGROUND_START_M: Final[float] = 29000.0
DEFAULT_BACKGROUND_STOP_M: Final[float] = 29999.0
DEFAULT_LATITUDE_DEGREES: Final[float] = -23.561
DEFAULT_LONGITUDE_DEGREES: Final[float] = -46.735
RAW_SIGNAL_UNITS: Final[str] = "counts for PC, mV per shot for analog"
BINARY_DIMENSIONS: Final[tuple[str, str, str]] = ("time", "channels", "points")
TIME_SCALE_DIMENSIONS: Final[tuple[str, str]] = ("time", "nb_of_time_scales")
BCK_TIME_SCALE_DIMENSIONS: Final[tuple[str, str]] = ("time_bck", "nb_of_time_scales")


def validate_lidar_tensors(tensors: dict, channels: list[str]) -> tuple[int, int]:
    """Validate Level-0 tensor consistency before NetCDF export."""
    if not tensors:
        raise ValueError("No lidar tensors available for NetCDF export.")
    if not channels:
        raise ValueError("No channel list available for NetCDF export.")

    missing_channels = [ch for ch in channels if ch not in tensors]
    if missing_channels:
        raise ValueError(f"Channels missing from tensor dictionary: {missing_channels}")

    reference_shape = None
    for ch_name in channels:
        tensor = np.asarray(tensors[ch_name])
        if tensor.ndim != 2:
            raise ValueError(f"Tensor for channel {ch_name} must be 2D; got shape {tensor.shape}.")
        if reference_shape is None:
            reference_shape = tensor.shape
        elif tensor.shape != reference_shape:
            raise ValueError(f"Inconsistent tensor shape for channel {ch_name}: expected {reference_shape}, got {tensor.shape}.")
    num_times, num_points = reference_shape
    return int(num_times), int(num_points)


def _source_file_names(group_df: pd.DataFrame) -> list[str]:
    """Return sorted source-file names from an inventory group."""
    if "filepath" not in group_df:
        return []
    return sorted(Path(path).name for path in group_df["filepath"].tolist())


def _physics_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the physics configuration section."""
    physics = config.get("physics", {})
    return physics if isinstance(physics, Mapping) else {}


def _hardware_map(config: Mapping[str, Any], period: str) -> Mapping[str, Any]:
    """Return the configured hardware ID map for one acquisition mode."""
    system_mode = "night" if period == "nt" else "day"
    return config.get("hardware", {}).get("name_to_id", {}).get(system_mode, {})


def _background_window_m(config: Mapping[str, Any]) -> tuple[float, float]:
    """Return the configured background altitude window."""
    physics = _physics_config(config)
    start = float(physics.get("background_start_m", physics.get("bg_start", DEFAULT_BACKGROUND_START_M)))
    stop = float(physics.get("background_stop_m", physics.get("bg_stop", DEFAULT_BACKGROUND_STOP_M)))
    return start, stop


def _vertical_resolution_m(config: Mapping[str, Any]) -> float:
    """Return the configured vertical resolution."""
    return float(_physics_config(config).get("vertical_resolution_m", DEFAULT_VERTICAL_RESOLUTION_M))


def _surface_value(weather_data: Mapping[str, Any], config: Mapping[str, Any], weather_key: str, physics_key: str, default: float) -> float:
    """Return one scalar surface diagnostic with weather-data priority over config fallback."""
    physics = _physics_config(config)
    return float(weather_data.get(weather_key, physics.get(physics_key, default)))


def _measurement_rows(group_df: pd.DataFrame) -> pd.DataFrame:
    """Return measurement-only rows sorted by acquisition start time."""
    df_meas = group_df[group_df["meas_type"] == "measurements"].copy()
    if df_meas.empty:
        return df_meas
    return df_meas.sort_values("start_time_utc").reset_index(drop=True)


def _truncate_time_axis(measurement_rows: pd.DataFrame, num_times_tensor: int, save_id: str, logger: logging.Logger) -> tuple[pd.DataFrame, int]:
    """Align metadata time count with tensor time count, truncating when needed."""
    if len(measurement_rows) != num_times_tensor:
        n_copy = min(len(measurement_rows), num_times_tensor)
        logger.warning(
            f"  -> Time axis mismatch for {save_id}: metadata has {len(measurement_rows)} profiles but tensor has {num_times_tensor}. "
            f"Truncating to {n_copy}."
        )
        measurement_rows = measurement_rows.iloc[:n_copy].reset_index(drop=True)
        return measurement_rows, n_copy
    return measurement_rows, num_times_tensor


def _seconds_since(reference_time: pd.Timestamp, values: pd.Series) -> np.ndarray:
    """Return whole-second offsets relative to one UTC reference timestamp."""
    timestamps = pd.to_datetime(values, utc=True)
    offsets = (timestamps - reference_time) // pd.Timedelta("1s")
    return offsets.fillna(-1).astype(np.int32).to_numpy()


def _scalar_config_value(
    config: Mapping[str, Any],
    weather_data: Mapping[str, Any],
    *,
    weather_key: str,
    physics_key: str,
    default: float,
) -> float:
    """Return one scalar weather/config value for SCC scalar variables."""
    return _surface_value(weather_data, config, weather_key, physics_key, default)


def _stack_raw_lidar_data(
    tensors: Mapping[str, np.ndarray],
    channels: list[str],
    num_times: int,
    num_points: int,
) -> np.ndarray:
    """Return the canonical time-channel-range tensor for NetCDF export."""
    stacked_tensor = np.zeros((num_times, len(channels), num_points), dtype=np.float64)
    for index, channel_name in enumerate(channels):
        stacked_tensor[:, index, :] = np.asarray(tensors[channel_name], dtype=np.float64)[:num_times, :]
    return stacked_tensor


def _channel_id(channel_name: str, hardware_map: Mapping[str, Any], period: str, logger: logging.Logger) -> int:
    """Return one configured channel ID or the stable fallback ID."""
    system_mode = "night" if period == "nt" else "day"
    if channel_name not in hardware_map:
        logger.warning(f"  -> Channel {channel_name} missing in config for {system_mode} mode. Using default {DEFAULT_CHANNEL_ID}.")
    return int(hardware_map.get(channel_name, DEFAULT_CHANNEL_ID))


def _create_level0_dimensions(ds: nc.Dataset, *, num_times: int, num_channels: int, num_points: int) -> None:
    """Create the canonical Level 0 dimensions."""
    ds.createDimension("time", num_times)
    ds.createDimension("channels", num_channels)
    ds.createDimension("points", num_points)
    ds.createDimension("nb_of_time_scales", 1)
    ds.createDimension("scan_angles", 1)


def _create_level0_core_variables(ds: nc.Dataset) -> dict[str, nc.Variable]:
    """Create the core Level 0 variables and return them by logical name."""
    raw_data_start = ds.createVariable("Raw_Data_Start_Time", "i4", TIME_SCALE_DIMENSIONS)
    raw_data_start.units = "s"
    raw_data_start.long_name = "Raw data start time offset from RawData_Start_Time_UT"

    raw_data_stop = ds.createVariable("Raw_Data_Stop_Time", "i4", TIME_SCALE_DIMENSIONS)
    raw_data_stop.units = "s"
    raw_data_stop.long_name = "Raw data stop time offset from RawData_Start_Time_UT"

    raw_lidar_data = ds.createVariable("Raw_Lidar_Data", "f8", BINARY_DIMENSIONS, zlib=True)
    raw_lidar_data.long_name = "Raw lidar signal"
    raw_lidar_data.units = RAW_SIGNAL_UNITS

    laser_pointing_angle = ds.createVariable("Laser_Pointing_Angle", "f8", ("scan_angles",))
    laser_pointing_angle.units = "degree"

    laser_pointing_angle_of_profiles = ds.createVariable("Laser_Pointing_Angle_of_Profiles", "i4", TIME_SCALE_DIMENSIONS)
    laser_pointing_angle_of_profiles.long_name = "Scan-angle index used by each profile"

    laser_shots = ds.createVariable("Laser_Shots", "i4", ("time", "channels"))
    laser_shots.units = "shots"

    molecular_calc = ds.createVariable("Molecular_Calc", "i4")
    molecular_calc.long_name = "Molecular density calculation mode"

    pressure_at_station = ds.createVariable("Pressure_at_Lidar_Station", "f8")
    pressure_at_station.units = "hPa"

    temperature_at_station = ds.createVariable("Temperature_at_Lidar_Station", "f8")
    temperature_at_station.units = "C"

    range_res = ds.createVariable("Raw_Data_Range_Resolution", "f8", ("channels",))
    range_res.units = "m"

    bg_low = ds.createVariable("Background_Low", "f8", ("channels",))
    bg_low.units = "m"

    bg_high = ds.createVariable("Background_High", "f8", ("channels",))
    bg_high.units = "m"

    return {
        "raw_data_start": raw_data_start,
        "raw_data_stop": raw_data_stop,
        "raw_lidar_data": raw_lidar_data,
        "laser_pointing_angle": laser_pointing_angle,
        "laser_pointing_angle_of_profiles": laser_pointing_angle_of_profiles,
        "laser_shots": laser_shots,
        "molecular_calc": molecular_calc,
        "pressure_at_station": pressure_at_station,
        "temperature_at_station": temperature_at_station,
        "channel_ids": ds.createVariable("channel_ID", "i4", ("channels",)),
        "id_timescale": ds.createVariable("id_timescale", "i4", ("channels",)),
        "range_resolution": range_res,
        "background_low": bg_low,
        "background_high": bg_high,
        "channel_names": ds.createVariable("channel_string", str, ("channels",)),
    }


def _write_channel_metadata(
    variables: Mapping[str, nc.Variable],
    channels: list[str],
    config: Mapping[str, Any],
    period: str,
    logger: logging.Logger,
) -> None:
    """Write channel-scoped metadata variables."""
    hardware_map = _hardware_map(config, period)
    vertical_resolution_m = _vertical_resolution_m(config)
    background_start_m, background_stop_m = _background_window_m(config)

    for index, channel_name in enumerate(channels):
        variables["channel_names"][index] = channel_name
        variables["channel_ids"][index] = _channel_id(channel_name, hardware_map, period, logger)
        variables["id_timescale"][index] = 0
        variables["range_resolution"][index] = vertical_resolution_m
        variables["background_low"][index] = background_start_m
        variables["background_high"][index] = background_stop_m


def _dark_current_attributes(group_df: pd.DataFrame) -> dict:
    """Build global attributes describing dark-current provenance."""
    df_dc = group_df[group_df["meas_type"] == "dark_current"].copy()
    if df_dc.empty:
        return {
            "Dark_Current_Source_File_Count": 0,
            "Dark_Current_Source_Files": "",
            "Dark_Current_Association_Methods": "none",
            "Dark_Current_Max_Association_Delta_hours": np.nan,
        }

    methods = "unknown"
    if "association_method" in df_dc:
        methods = ";".join(sorted(str(value) for value in df_dc["association_method"].dropna().unique())) or "unknown"

    max_delta = np.nan
    if "dark_current_association_delta_hours" in df_dc:
        delta_values = pd.to_numeric(df_dc["dark_current_association_delta_hours"], errors="coerce")
        if delta_values.notna().any():
            max_delta = float(delta_values.max())

    return {
        "Dark_Current_Source_File_Count": int(len(df_dc)),
        "Dark_Current_Source_Files": ";".join(_source_file_names(df_dc)),
        "Dark_Current_Association_Methods": methods,
        "Dark_Current_Max_Association_Delta_hours": max_delta,
    }


def build_level0_global_attributes(
    save_id: str,
    lidar_data: dict,
    group_df: pd.DataFrame,
    weather_data: dict,
    config: dict,
) -> dict:
    """Build global NetCDF attributes for the Level-0 product."""
    min_start_utc = pd.to_datetime(group_df["start_time_utc"]).min()
    max_stop_utc = pd.to_datetime(group_df["stop_time"]).max()
    source_files = _source_file_names(group_df)
    physics = _physics_config(config)

    attrs = {
        "Measurement_ID": save_id,
        "System": "SPU-Lidar",
        "Processing_level": "Level 0: Raw Licel to SCC-compliant NetCDF",
        "Pipeline": "MILGRAU",
        "Latitude_degrees_north": float(physics.get("latitude", DEFAULT_LATITUDE_DEGREES)),
        "Longitude_degrees_east": float(physics.get("longitude", DEFAULT_LONGITUDE_DEGREES)),
        "Accumulated_Shots": int(lidar_data.get("shots", 0)),
        "RawData_Start_Date": min_start_utc.strftime("%Y%m%d"),
        "RawData_Start_Time_UT": min_start_utc.strftime("%H%M%S"),
        "RawData_Stop_Time_UT": max_stop_utc.strftime("%H%M%S"),
        "Temperature_C": _surface_value(weather_data, config, "temperature_c", "default_surface_temp_c", 25.0),
        "Pressure_hPa": _surface_value(weather_data, config, "pressure_hpa", "default_surface_pressure_hpa", 940.0),
        "CloudCover_percent": float(weather_data.get("cloud_cover_percent", np.nan)),
        "RelativeHumidity_percent": float(weather_data.get("relative_humidity_percent", np.nan)),
        "WindSpeed_kmh": float(weather_data.get("wind_speed_kmh", np.nan)),
        "Source_File_Count": int(len(source_files)),
        "Source_Files": ";".join(source_files),
    }
    attrs.update(_dark_current_attributes(group_df))
    return attrs


def _write_dark_current_availability(ds: nc.Dataset, availability: np.ndarray) -> None:
    """Write channel-wise dark-current availability flags."""
    var = ds.createVariable("Background_Profile_Available", "i1", ("channels",))
    var.long_name = "Dark-current profile availability by channel"
    var.flag_values = "0, 1"
    var.flag_meanings = "not_available available"
    var[:] = availability.astype(np.int8)


def _write_dark_current_time_axes(ds: nc.Dataset, dark_current_rows: pd.DataFrame, num_time_bck: int) -> None:
    """Write optional SCC background time-offset matrices for one time scale."""
    if dark_current_rows.empty:
        return
    dark_current_rows = dark_current_rows.sort_values("start_time_utc").reset_index(drop=True).iloc[:num_time_bck]
    reference_time = pd.to_datetime(dark_current_rows["start_time_utc"], utc=True).iloc[0]
    start_offsets = _seconds_since(reference_time, dark_current_rows["start_time_utc"])
    stop_offsets = _seconds_since(reference_time, dark_current_rows["stop_time"])
    raw_bck_start = ds.createVariable("Raw_Bck_Start_Time", "i4", BCK_TIME_SCALE_DIMENSIONS)
    raw_bck_stop = ds.createVariable("Raw_Bck_Stop_Time", "i4", BCK_TIME_SCALE_DIMENSIONS)
    raw_bck_start.units = "s"
    raw_bck_stop.units = "s"
    raw_bck_start[:, 0] = start_offsets
    raw_bck_stop[:, 0] = stop_offsets


def write_dark_current_profile(
    ds: nc.Dataset,
    group_df: pd.DataFrame,
    channels: list[str],
    num_channels: int,
    num_points: int,
    logger: logging.Logger,
) -> None:
    """Parse and write the optional dark-current matrix and availability flags."""
    availability = np.zeros(num_channels, dtype=np.int8)
    df_dc = group_df[group_df["meas_type"] == "dark_current"]
    if df_dc.empty:
        _write_dark_current_availability(ds, availability)
        return

    dc_files = df_dc["filepath"].tolist()
    dc_data = parse_licel_group(dc_files, logger)
    if not dc_data.get("tensors"):
        logger.warning("  -> Dark current files found but parsing failed. NetCDF will lack Background_Profile.")
        _write_dark_current_availability(ds, availability)
        return

    first_tensor = next(iter(dc_data["tensors"].values()))
    num_time_bck = first_tensor.shape[0]
    ds.createDimension("time_bck", num_time_bck)
    bck_prof = ds.createVariable("Background_Profile", "f8", ("time_bck", "channels", "points"), zlib=True)
    bck_prof.long_name = "Dark-current background profile"
    bck_prof.units = "channel native units"

    stacked_dc = np.full((num_time_bck, num_channels, num_points), np.nan, dtype=np.float64)
    for i, ch_name in enumerate(channels):
        if ch_name not in dc_data["tensors"]:
            logger.warning(f"  -> Dark-current data missing for channel {ch_name}. Filling with NaN and flagging unavailable.")
            continue
        dc_tensor = np.asarray(dc_data["tensors"][ch_name], dtype=np.float64)
        if dc_tensor.ndim != 2 or dc_tensor.shape[1] != num_points:
            logger.warning(
                f"  -> Dark-current channel {ch_name} has shape {dc_tensor.shape}; expected (*, {num_points}). "
                "Filling with NaN and flagging unavailable."
            )
            continue
        n_copy = min(num_time_bck, dc_tensor.shape[0])
        stacked_dc[:n_copy, i, :] = dc_tensor[:n_copy, :]
        availability[i] = 1

    bck_prof[:] = stacked_dc
    _write_dark_current_time_axes(ds, df_dc, num_time_bck)
    _write_dark_current_availability(ds, availability)
    logger.info(f"  -> Successfully injected Dark Current matrix ({num_time_bck} profiles).")


def build_level0_netcdf(
    netcdf_path: str,
    save_id: str,
    period: str,
    lidar_data: dict,
    group_df: pd.DataFrame,
    weather_data: dict,
    config: dict,
    logger: logging.Logger,
) -> None:
    """Generate an SCC-compliant Level-0 NetCDF from parsed Licel tensors."""
    try:
        tensors = lidar_data["tensors"]
        channels = lidar_data["channels"]
        num_times_tensor, num_points = validate_lidar_tensors(tensors, channels)
        num_channels = len(channels)
        measurement_rows = _measurement_rows(group_df)
        measurement_rows, num_times = _truncate_time_axis(measurement_rows, num_times_tensor, save_id, logger)
        if num_times <= 0:
            raise ValueError("No valid time profiles available after tensor/time-axis validation.")
        measurement_start_times = pd.to_datetime(measurement_rows["start_time_utc"], utc=True)
        reference_time = measurement_start_times.iloc[0]
        start_offsets = _seconds_since(reference_time, measurement_rows["start_time_utc"])
        stop_offsets = _seconds_since(reference_time, measurement_rows["stop_time"])
        laser_pointing_angle_deg = float(_physics_config(config).get("laser_pointing_angle_deg", 0.0))
        pressure_hpa = _scalar_config_value(
            config,
            weather_data,
            weather_key="pressure_hpa",
            physics_key="default_surface_pressure_hpa",
            default=940.0,
        )
        temperature_c = _scalar_config_value(
            config,
            weather_data,
            weather_key="temperature_c",
            physics_key="default_surface_temp_c",
            default=25.0,
        )

        with nc.Dataset(netcdf_path, "w", format="NETCDF4") as ds:
            ds.setncatts(build_level0_global_attributes(save_id, lidar_data, group_df, weather_data, config))
            _create_level0_dimensions(ds, num_times=num_times, num_channels=num_channels, num_points=num_points)
            variables = _create_level0_core_variables(ds)
            variables["raw_data_start"][:, 0] = start_offsets
            variables["raw_data_stop"][:, 0] = stop_offsets
            variables["raw_lidar_data"][:] = _stack_raw_lidar_data(tensors, channels, num_times, num_points)
            variables["laser_pointing_angle"][:] = np.array([laser_pointing_angle_deg], dtype=np.float64)
            variables["laser_pointing_angle_of_profiles"][:, 0] = np.zeros(num_times, dtype=np.int32)
            variables["laser_shots"][:] = np.full((num_times, num_channels), int(lidar_data.get("shots", 0)), dtype=np.int32)
            variables["molecular_calc"].assignValue(np.int32(0))
            variables["pressure_at_station"].assignValue(np.float64(pressure_hpa))
            variables["temperature_at_station"].assignValue(np.float64(temperature_c))
            _write_channel_metadata(variables, channels, config, period, logger)
            write_dark_current_profile(ds, group_df, channels, num_channels, num_points, logger)
    except Exception as exc:
        raise RuntimeError(f"Failed to build NetCDF: {exc}") from exc
