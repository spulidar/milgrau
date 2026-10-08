"""Level 0 NetCDF writing and provenance helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final, Mapping

import netCDF4 as nc
import numpy as np
import pandas as pd

from milgrau.io.licel import parse_licel_group
from milgrau.level0.config import resolve_level0_config, station_pointing_angle_deg_from_zenith
from milgrau.physics.solar import SOLAR_POSITION_ALGORITHM

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
            raise ValueError(
                f"Inconsistent tensor shape for channel {ch_name}: expected {reference_shape}, got {tensor.shape}."
            )
    num_times, num_points = reference_shape
    return int(num_times), int(num_points)


def _source_file_names(group_df: pd.DataFrame) -> list[str]:
    if "filepath" not in group_df:
        return []
    return sorted(Path(path).name for path in group_df["filepath"].tolist())


def _resolved_station(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("_resolved_station")
    if not isinstance(value, Mapping):
        raise ValueError("Level 0 NetCDF writing requires a resolved station context.")
    return value


def _scc_ready(config: Mapping[str, Any]) -> bool:
    return bool(_resolved_station(config).get("scc_available", False))


def _hardware_map(config: Mapping[str, Any]) -> Mapping[str, Any]:
    channel_ids = _resolved_station(config).get("channel_ids")
    if not isinstance(channel_ids, Mapping):
        raise ValueError("Resolved station context must provide channel_ids mapping.")
    return channel_ids


def _background_window_m(config: Mapping[str, Any]) -> tuple[float, float]:
    """Return the explicit processing background window used for SCC metadata."""
    level1 = config.get("level1")
    if not isinstance(level1, Mapping):
        raise ValueError("Configuration level1 is required to write SCC background metadata.")
    background = level1.get("background")
    if not isinstance(background, Mapping):
        raise ValueError("Configuration level1.background is required to write SCC background metadata.")
    if "start_altitude_m" not in background or "stop_altitude_m" not in background:
        raise ValueError(
            "Configuration level1.background.start_altitude_m and stop_altitude_m are required for Level 0 SCC metadata."
        )
    start = float(background["start_altitude_m"])
    stop = float(background["stop_altitude_m"])
    if not np.isfinite(start) or not np.isfinite(stop) or start < 0.0 or stop <= start:
        raise ValueError("Configuration level1.background must define a finite ordered altitude interval.")
    return start, stop


def _surface_values(weather_data: Mapping[str, Any], weather_key: str) -> np.ndarray:
    """Return one surface-weather series as float64, preserving missingness."""
    raw = weather_data.get(weather_key, [])
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    return values


def _surface_representative_value(weather_data: Mapping[str, Any], weather_key: str) -> float:
    """Return the finite session median used only for scalar interoperability fields."""
    values = _surface_values(weather_data, weather_key)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _write_surface_weather_series(ds: nc.Dataset, weather_data: Mapping[str, Any]) -> None:
    """Persist native-cadence surface weather independently from lidar profile time."""
    times = np.asarray(weather_data.get("weather_time", []), dtype="datetime64[ns]").reshape(-1)
    if times.size == 0:
        return
    ds.createDimension("weather_time", int(times.size))
    time_var = ds.createVariable("weather_time", "i8", ("weather_time",))
    time_var.units = "seconds since 1970-01-01 00:00:00 UTC"
    time_var.calendar = "standard"
    time_var.long_name = "Surface-weather source time in UTC"
    time_var[:] = times.astype("datetime64[s]").astype(np.int64)

    definitions = {
        "temperature_c": ("Surface_Temperature_C", "degree_Celsius", "Surface air temperature"),
        "pressure_hpa": ("Surface_Pressure_hPa", "hPa", "Surface pressure"),
        "relative_humidity_percent": (
            "Surface_Relative_Humidity_percent",
            "%",
            "Surface relative humidity",
        ),
        "cloud_cover_percent": ("Surface_Cloud_Cover_percent", "%", "Total cloud cover"),
        "wind_speed_kmh": ("Surface_Wind_Speed_kmh", "km h-1", "Surface wind speed"),
    }
    for source_key, (variable_name, units, long_name) in definitions.items():
        values = _surface_values(weather_data, source_key)
        if values.size != times.size:
            raise ValueError(
                f"Surface weather field {source_key!r} has {values.size} values for "
                f"{times.size} weather_time entries."
            )
        variable = ds.createVariable(variable_name, "f8", ("weather_time",), zlib=True)
        variable.units = units
        variable.long_name = long_name
        variable[:] = values

    ds.setncattr("Surface_Weather_Source", str(weather_data.get("source", "")))
    ds.setncattr("Surface_Weather_Cadence", str(weather_data.get("cadence", "hourly")))
    ds.setncattr("Surface_Weather_Scalar_Method", "finite session median for SCC interoperability")


def _write_solar_context(
    ds: nc.Dataset,
    measurement_rows: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    """Persist per-profile solar state plus a compact segment table."""
    required = {"solar_elevation_deg", "solar_regime", "segment_id", "start_time_utc", "stop_time"}
    missing = sorted(required - set(measurement_rows.columns))
    if missing:
        raise KeyError(f"Level 0 solar context lacks required column(s): {missing}")

    elevation = pd.to_numeric(
        measurement_rows["solar_elevation_deg"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    regimes = measurement_rows["solar_regime"].astype(str).to_numpy(dtype=object)
    segment_ids = measurement_rows["segment_id"].astype(str).to_numpy(dtype=object)
    if elevation.size != ds.dimensions["time"].size:
        raise ValueError("Solar context must contain one value per Level 0 time profile.")
    if not np.all(np.isfinite(elevation)):
        raise ValueError("Solar elevation must be finite for every Level 0 profile.")
    if not set(regimes).issubset({"day", "night"}):
        raise ValueError("Solar regime must contain only day/night.")
    if any(not value.startswith("seg") for value in segment_ids):
        raise ValueError("Segment IDs must use the segXX convention.")

    solar_var = ds.createVariable("solar_elevation_deg", "f8", ("time",), zlib=True)
    solar_var.units = "degree"
    solar_var.long_name = "Geometric solar-center elevation at lidar profile midpoint"
    solar_var[:] = elevation

    regime_var = ds.createVariable("solar_regime", str, ("time",))
    regime_var.long_name = "Solar day/night regime"
    regime_var[:] = regimes

    segment_var = ds.createVariable("segment_id", str, ("time",))
    segment_var.long_name = "Contiguous scientific segment identifier"
    segment_var[:] = segment_ids

    ordered_segments = list(dict.fromkeys(segment_ids.tolist()))
    ds.createDimension("segments", len(ordered_segments))
    segment_label = ds.createVariable("Segment_Label", str, ("segments",))
    segment_regime = ds.createVariable("Segment_Regime", str, ("segments",))
    segment_start = ds.createVariable("Segment_Start_Time_UTC", "i8", ("segments",))
    segment_end = ds.createVariable("Segment_End_Time_UTC", "i8", ("segments",))
    segment_start.units = "seconds since 1970-01-01 00:00:00 UTC"
    segment_end.units = "seconds since 1970-01-01 00:00:00 UTC"

    starts: list[int] = []
    ends: list[int] = []
    labels: list[str] = []
    regime_labels: list[str] = []
    for label in ordered_segments:
        rows = measurement_rows[measurement_rows["segment_id"].astype(str) == label]
        unique_regimes = rows["solar_regime"].astype(str).unique()
        if len(unique_regimes) != 1:
            raise ValueError(
                f"Segment {label!r} must contain exactly one solar regime; got {unique_regimes.tolist()}."
            )
        start = pd.to_datetime(rows["start_time_utc"], utc=True).min()
        end = pd.to_datetime(rows["stop_time"], utc=True).max()
        labels.append(str(label))
        regime_labels.append(str(unique_regimes[0]))
        starts.append(int(start.timestamp()))
        ends.append(int(end.timestamp()))

    segment_label[:] = np.asarray(labels, dtype=object)
    segment_regime[:] = np.asarray(regime_labels, dtype=object)
    segment_start[:] = np.asarray(starts, dtype=np.int64)
    segment_end[:] = np.asarray(ends, dtype=np.int64)

    level0 = resolve_level0_config(config)
    ds.setncattr(
        "Solar_Day_Night_Threshold_deg",
        float(level0.solar_regime.day_night_threshold_deg),
    )
    ds.setncattr("Solar_Position_Algorithm", SOLAR_POSITION_ALGORITHM)
    ds.setncattr("Segment_Count", np.int32(len(ordered_segments)))


def _measurement_rows(group_df: pd.DataFrame) -> pd.DataFrame:
    df_meas = group_df[group_df["meas_type"] == "measurements"].copy()
    if df_meas.empty:
        return df_meas
    return df_meas.sort_values("start_time_utc").reset_index(drop=True)


def _truncate_time_axis(
    measurement_rows: pd.DataFrame,
    num_times_tensor: int,
    session_id: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, int]:
    if len(measurement_rows) != num_times_tensor:
        n_copy = min(len(measurement_rows), num_times_tensor)
        logger.warning(
            f"  -> Time axis mismatch for {session_id}: metadata has {len(measurement_rows)} profiles "
            f"but tensor has {num_times_tensor}. Truncating to {n_copy}."
        )
        measurement_rows = measurement_rows.iloc[:n_copy].reset_index(drop=True)
        return measurement_rows, n_copy
    return measurement_rows, num_times_tensor


def _seconds_since(reference_time: pd.Timestamp, values: pd.Series) -> np.ndarray:
    timestamps = pd.to_datetime(values, utc=True)
    offsets = (timestamps - reference_time) // pd.Timedelta("1s")
    return offsets.fillna(-1).astype(np.int32).to_numpy()


def _scc_time_axis(
    rows: pd.DataFrame,
    reference_time: pd.Timestamp,
    config: Mapping[str, Any],
    logger: logging.Logger,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return raw or QA-normalized SCC time offsets without re-validating QA."""
    start_offsets = _seconds_since(reference_time, rows["start_time_utc"])
    stop_offsets = _seconds_since(reference_time, rows["stop_time"])
    if not _scc_ready(config) or len(rows) <= 1 or "qa_nominal_duration_s" not in rows:
        return start_offsets, stop_offsets, {}

    nominal_duration_s = int(round(float(pd.to_numeric(rows["qa_nominal_duration_s"], errors="coerce").iloc[0])))
    corrected_stop_offsets = (start_offsets.astype(np.int64) + nominal_duration_s).astype(np.int32)
    adjusted_count = int(np.count_nonzero(corrected_stop_offsets != stop_offsets))
    max_adjustment_s = int(
        np.max(np.abs(corrected_stop_offsets.astype(np.int64) - stop_offsets.astype(np.int64)))
    )
    prefix = "SCC_Background" if label == "Background" else "SCC"
    attrs: dict[str, Any] = {
        f"{prefix}_Time_Axis_Normalized": np.int8(1 if adjusted_count else 0),
        f"{prefix}_Nominal_Time_Resolution_s": np.int32(nominal_duration_s),
        f"{prefix}_Time_Adjusted_Profile_Count": np.int32(adjusted_count),
        f"{prefix}_Max_Time_Adjustment_s": np.int32(max_adjustment_s),
        f"{prefix}_Time_Normalization_Basis": "upstream acquisition QA; Licel whole-second header correction",
    }
    if adjusted_count:
        logger.info(
            "  -> %s SCC time axis normalized to %d s from upstream QA; adjusted %d/%d profiles (max %d s).",
            label,
            nominal_duration_s,
            adjusted_count,
            len(rows),
            max_adjustment_s,
        )
    return start_offsets, corrected_stop_offsets, attrs


def _stack_raw_lidar_data(
    tensors: Mapping[str, np.ndarray],
    channels: list[str],
    num_times: int,
    num_points: int,
) -> np.ndarray:
    stacked_tensor = np.zeros((num_times, len(channels), num_points), dtype=np.float64)
    for index, channel_name in enumerate(channels):
        stacked_tensor[:, index, :] = np.asarray(tensors[channel_name], dtype=np.float64)[:num_times, :]
    return stacked_tensor


def _channel_id(
    channel_name: str,
    hardware_map: Mapping[str, Any],
) -> int:
    if channel_name not in hardware_map:
        raise ValueError(
            f"Channel {channel_name} has no SCC channel ID in the resolved station context; "
            "SCC export must be disabled rather than writing a fabricated ID."
        )
    return int(hardware_map[channel_name])


def _channel_metadata(lidar_data: Mapping[str, Any], channel_name: str) -> Mapping[str, Any]:
    metadata = lidar_data.get("channel_metadata", {})
    if not isinstance(metadata, Mapping):
        return {}
    channel = metadata.get(channel_name, {})
    return channel if isinstance(channel, Mapping) else {}


def _is_analog_channel(channel_name: str, metadata: Mapping[str, Any]) -> bool:
    if "is_pc" in metadata:
        return not bool(metadata["is_pc"])
    return channel_name.upper().endswith(".AN")


def _channel_range_resolution_m(
    lidar_data: Mapping[str, Any],
    channel_name: str,
) -> float:
    """Return native Licel BinW metadata; no global range-resolution fallback exists."""
    metadata = _channel_metadata(lidar_data, channel_name)
    value = metadata.get("bin_width_m", np.nan)
    try:
        resolution = float(value)
    except (TypeError, ValueError):
        resolution = np.nan
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError(
            f"Channel {channel_name} lacks a positive finite Licel bin_width_m; "
            "range resolution cannot be invented by the Level 0 writer."
        )
    return resolution


def _laser_shot_matrix(lidar_data: Mapping[str, Any], num_times: int, num_channels: int) -> np.ndarray:
    values = lidar_data.get("laser_shots")
    if values is None:
        fallback = int(lidar_data.get("shots", 0))
        if fallback <= 0:
            raise ValueError("No positive laser-shot metadata available for Level 0 export.")
        return np.full((num_times, num_channels), fallback, dtype=np.int32)
    shots = np.asarray(values)
    if shots.ndim != 2 or shots.shape[1] != num_channels or shots.shape[0] < num_times:
        raise ValueError(
            "laser_shots must have shape (time, channels) conformable with Raw_Lidar_Data; "
            f"got {shots.shape}, expected at least ({num_times}, {num_channels})."
        )
    shots = shots[:num_times, :]
    if not np.all(np.isfinite(shots)) or np.any(shots <= 0):
        raise ValueError("Laser_Shots contains non-finite or non-positive values.")
    return shots.astype(np.int32)


def _create_level0_dimensions(
    ds: nc.Dataset,
    *,
    num_times: int,
    num_channels: int,
    num_points: int,
) -> None:
    ds.createDimension("time", num_times)
    ds.createDimension("channels", num_channels)
    ds.createDimension("points", num_points)
    ds.createDimension("nb_of_time_scales", 1)
    ds.createDimension("scan_angles", 1)


def _create_level0_core_variables(ds: nc.Dataset, *, include_channel_ids: bool = True) -> dict[str, nc.Variable]:
    raw_data_start = ds.createVariable("Raw_Data_Start_Time", "i4", TIME_SCALE_DIMENSIONS)
    raw_data_start.units = "s"
    raw_data_stop = ds.createVariable("Raw_Data_Stop_Time", "i4", TIME_SCALE_DIMENSIONS)
    raw_data_stop.units = "s"
    raw_lidar_data = ds.createVariable("Raw_Lidar_Data", "f8", BINARY_DIMENSIONS, zlib=True)
    raw_lidar_data.long_name = "Raw lidar signal"
    raw_lidar_data.units = RAW_SIGNAL_UNITS
    laser_pointing_angle = ds.createVariable("Laser_Pointing_Angle", "f8", ("scan_angles",))
    laser_pointing_angle.units = "degree"
    laser_pointing_angle_of_profiles = ds.createVariable(
        "Laser_Pointing_Angle_of_Profiles", "i4", TIME_SCALE_DIMENSIONS
    )
    laser_shots = ds.createVariable("Laser_Shots", "i4", ("time", "channels"))
    laser_shots.units = "shots"
    molecular_calc = ds.createVariable("Molecular_Calc", "i4")
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
    variables = {
        "raw_data_start": raw_data_start,
        "raw_data_stop": raw_data_stop,
        "raw_lidar_data": raw_lidar_data,
        "laser_pointing_angle": laser_pointing_angle,
        "laser_pointing_angle_of_profiles": laser_pointing_angle_of_profiles,
        "laser_shots": laser_shots,
        "molecular_calc": molecular_calc,
        "pressure_at_station": pressure_at_station,
        "temperature_at_station": temperature_at_station,
        "id_timescale": ds.createVariable("id_timescale", "i4", ("channels",)),
        "range_resolution": range_res,
        "background_low": bg_low,
        "background_high": bg_high,
        "channel_names": ds.createVariable("channel_string", str, ("channels",)),
    }
    if include_channel_ids:
        variables["channel_ids"] = ds.createVariable("channel_ID", "i4", ("channels",))
    return variables


def _write_channel_metadata(
    variables: Mapping[str, nc.Variable],
    channels: list[str],
    lidar_data: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    hardware_map = _hardware_map(config) if "channel_ids" in variables else {}
    background_start_m, background_stop_m = _background_window_m(config)
    for index, channel_name in enumerate(channels):
        variables["channel_names"][index] = channel_name
        if "channel_ids" in variables:
            variables["channel_ids"][index] = _channel_id(channel_name, hardware_map)
        variables["id_timescale"][index] = 0
        variables["range_resolution"][index] = _channel_range_resolution_m(lidar_data, channel_name)
        variables["background_low"][index] = background_start_m
        variables["background_high"][index] = background_stop_m


def _write_daq_range(ds: nc.Dataset, channels: list[str], lidar_data: Mapping[str, Any]) -> None:
    values = np.ma.masked_all(len(channels), dtype=np.float64)
    analog_count = 0
    for index, channel_name in enumerate(channels):
        metadata = _channel_metadata(lidar_data, channel_name)
        if not _is_analog_channel(channel_name, metadata):
            continue
        analog_count += 1
        daq_range = metadata.get("daq_range_mV", metadata.get("adc_range", np.nan))
        try:
            daq_range_mv = float(daq_range)
        except (TypeError, ValueError):
            daq_range_mv = np.nan
        if not np.isfinite(daq_range_mv) or daq_range_mv <= 0.0:
            raise ValueError(
                f"Analog channel {channel_name} lacks a positive Licel Discriminator/DAQ range required by acquisition metadata."
            )
        values[index] = daq_range_mv
    if not analog_count:
        return
    variable = ds.createVariable("DAQ_Range", "f8", ("channels",))
    variable.units = "mV"
    variable.long_name = "Analog acquisition scale"
    variable[:] = values


def _write_lr_input(ds: nc.Dataset, channels: list[str], config: Mapping[str, Any]) -> None:
    """Write SCC elastic-backscatter lidar-ratio source flags when configured."""
    raw = _resolved_station(config).get("lr_input", {})
    if not isinstance(raw, Mapping) or not raw:
        return

    unknown = sorted(set(str(name) for name in raw) - set(channels))
    if unknown:
        raise ValueError(f"LR_Input references channels not present in the exported Level 0: {unknown}")

    values = np.ma.masked_all(len(channels), dtype=np.int32)
    for index, channel_name in enumerate(channels):
        if channel_name not in raw:
            continue
        value = int(raw[channel_name])
        if value not in {0, 1}:
            raise ValueError(f"LR_Input for {channel_name} must be 0 or 1; got {value}.")
        values[index] = value

    variable = ds.createVariable("LR_Input", "i4", ("channels",))
    variable.long_name = "Lidar-ratio input source for elastic backscatter retrieval"
    variable.flag_values = "0, 1"
    variable.flag_meanings = "external_profile fixed_scc_db_value"
    variable[:] = values


def _dark_current_attributes(group_df: pd.DataFrame) -> dict:
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
    session_id: str,
    lidar_data: dict,
    group_df: pd.DataFrame,
    weather_data: dict,
    config: dict,
) -> dict:
    measurement_rows = _measurement_rows(group_df)
    timing_rows = measurement_rows if not measurement_rows.empty else group_df
    min_start_utc = pd.to_datetime(timing_rows["start_time_utc"], utc=True).min()
    max_stop_utc = pd.to_datetime(timing_rows["stop_time"], utc=True).max()
    source_files = _source_file_names(group_df)
    resolved = _resolved_station(config)
    site = resolved.get("site")
    if not isinstance(site, Mapping):
        raise ValueError("Resolved station context must provide site metadata.")
    latitude = float(site["latitude"])
    longitude = float(site["longitude"])
    if not np.isfinite(latitude) or not np.isfinite(longitude):
        raise ValueError("Resolved station latitude/longitude must be finite.")
    ready = _scc_ready(config)
    attrs = {
        "Session_ID": session_id,
        "measurement_start_time": min_start_utc.isoformat().replace("+00:00", "Z"),
        "measurement_end_time": max_stop_utc.isoformat().replace("+00:00", "Z"),
        "session_duration_seconds": float((max_stop_utc - min_start_utc).total_seconds()),
        "timezone": str(resolved["timezone"]),
        "System": str(resolved["station_name"]),
        "Processing_level": (
            "Level 0: Raw Licel to SCC-compatible NetCDF"
            if ready
            else "Level 0: Raw Licel NetCDF (SCC mapping unavailable)"
        ),
        "Pipeline": "MILGRAU",
        "SCC_Ready": np.int8(1 if ready else 0),
        "Latitude_degrees_north": latitude,
        "Longitude_degrees_east": longitude,
        "Accumulated_Shots": int(lidar_data.get("shots", 0)),
        "RawData_Start_Date": min_start_utc.strftime("%Y%m%d"),
        "RawData_Start_Time_UT": min_start_utc.strftime("%H%M%S"),
        "RawData_Stop_Time_UT": max_stop_utc.strftime("%H%M%S"),
        "Surface_Weather_Source": str(weather_data.get("source", "")),
        "Surface_Weather_Cadence": str(weather_data.get("cadence", "hourly")),
        "Source_File_Count": int(len(source_files)),
        "Source_Files": ";".join(source_files),
        "Solar_Day_Night_Threshold_deg": float(
            resolve_level0_config(config).solar_regime.day_night_threshold_deg
        ),
        "Solar_Position_Algorithm": SOLAR_POSITION_ALGORITHM,
    }
    if resolved.get("segment_id") is not None:
        attrs["Segment_ID"] = str(resolved["segment_id"])
    if resolved.get("solar_regime") is not None:
        attrs["Solar_Regime"] = str(resolved["solar_regime"])
    attrs["Station_Profile"] = str(resolved["profile_id"])
    if ready:
        attrs["SCC_Configuration_ID"] = int(resolved["scc_configuration_id"])
        attrs["SCC_Configuration_Name"] = str(resolved["scc_configuration_name"])
    attrs.update(_dark_current_attributes(group_df))
    return attrs


def _write_dark_current_availability(ds: nc.Dataset, availability: np.ndarray) -> None:
    var = ds.createVariable("Background_Profile_Available", "i1", ("channels",))
    var.long_name = "Dark-current profile availability by channel"
    var.flag_values = "0, 1"
    var.flag_meanings = "not_available available"
    var[:] = availability.astype(np.int8)


def _write_dark_current_time_axes(
    ds: nc.Dataset,
    dark_current_rows: pd.DataFrame,
    num_time_bck: int,
    config: Mapping[str, Any],
    logger: logging.Logger,
) -> None:
    if dark_current_rows.empty:
        return
    rows = dark_current_rows.sort_values("start_time_utc").reset_index(drop=True).iloc[:num_time_bck]
    reference_time = pd.to_datetime(rows["start_time_utc"], utc=True).iloc[0]
    start_offsets, stop_offsets, normalization_attrs = _scc_time_axis(
        rows,
        reference_time,
        config,
        logger,
        label="Background",
    )
    stop_time = reference_time + pd.to_timedelta(int(np.max(stop_offsets)), unit="s")
    raw_bck_start = ds.createVariable("Raw_Bck_Start_Time", "i4", BCK_TIME_SCALE_DIMENSIONS)
    raw_bck_stop = ds.createVariable("Raw_Bck_Stop_Time", "i4", BCK_TIME_SCALE_DIMENSIONS)
    raw_bck_start.units = "s"
    raw_bck_stop.units = "s"
    raw_bck_start[:, 0] = start_offsets
    raw_bck_stop[:, 0] = stop_offsets
    ds.setncattr("RawBck_Start_Date", reference_time.strftime("%Y%m%d"))
    ds.setncattr("RawBck_Start_Time_UT", reference_time.strftime("%H%M%S"))
    ds.setncattr("RawBck_Stop_Time_UT", stop_time.strftime("%H%M%S"))
    if normalization_attrs:
        ds.setncatts(normalization_attrs)


def _write_dark_current_laser_shots(
    ds: nc.Dataset,
    dc_data: Mapping[str, Any],
    channels: list[str],
    num_time_bck: int,
    logger: logging.Logger,
) -> None:
    """Persist parsed dark-acquisition NShots without inventing missing values."""
    raw_shots = dc_data.get("laser_shots")
    dc_channels = [str(name) for name in dc_data.get("channels", [])]
    if raw_shots is None or not dc_channels:
        logger.warning("  -> Dark-current NShots unavailable; Background_Laser_Shots will not be written.")
        return
    shots = np.asarray(raw_shots, dtype=np.float64)
    if shots.ndim != 2 or shots.shape[0] < num_time_bck or shots.shape[1] != len(dc_channels):
        logger.warning(
            "  -> Dark-current laser_shots shape %s is not conformable with %d profiles and %d parsed channels; "
            "Background_Laser_Shots will not be written.",
            shots.shape,
            num_time_bck,
            len(dc_channels),
        )
        return
    stacked_shots = np.full((num_time_bck, len(channels)), np.nan, dtype=np.float64)
    source_index = {name: index for index, name in enumerate(dc_channels)}
    for target_index, channel_name in enumerate(channels):
        if channel_name not in source_index:
            continue
        values = shots[:num_time_bck, source_index[channel_name]]
        valid = np.isfinite(values) & (values > 0.0)
        stacked_shots[valid, target_index] = values[valid]
        if not np.all(valid):
            logger.warning("  -> Dark-current NShots contains invalid values for channel %s; preserving them as missing.", channel_name)
    variable = ds.createVariable("Background_Laser_Shots", "f8", ("time_bck", "channels"), zlib=True)
    variable.long_name = "Laser shots accumulated for each dark-current profile and channel"
    variable.units = "shots"
    variable.description = (
        "Parsed Licel NShots for the dark acquisition. Preserved separately from measurement Laser_Shots so photon-counting "
        "dark rates can be normalized independently before evaluating nonlinear dead-time correction order."
    )
    variable[:] = stacked_shots


def write_dark_current_profile(
    ds: nc.Dataset,
    group_df: pd.DataFrame,
    channels: list[str],
    num_channels: int,
    num_points: int,
    logger: logging.Logger,
    config: Mapping[str, Any] | None = None,
) -> None:
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
            logger.warning(
                f"  -> Dark-current data missing for channel {ch_name}. Filling with NaN and flagging unavailable."
            )
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
    _write_dark_current_laser_shots(ds, dc_data, channels, num_time_bck, logger)
    _write_dark_current_time_axes(ds, df_dc, num_time_bck, config or {}, logger)
    _write_dark_current_availability(ds, availability)
    logger.info(f"  -> Successfully injected Dark Current matrix ({num_time_bck} profiles).")


def build_level0_netcdf(
    netcdf_path: str,
    session_id: str,
    lidar_data: dict,
    group_df: pd.DataFrame,
    weather_data: dict,
    config: dict,
    logger: logging.Logger,
) -> None:
    """Generate a Level-0 NetCDF from parsed Licel tensors, with optional SCC mapping."""
    try:
        tensors = lidar_data["tensors"]
        channels = lidar_data["channels"]
        num_times_tensor, num_points = validate_lidar_tensors(tensors, channels)
        num_channels = len(channels)
        measurement_rows = _measurement_rows(group_df)
        measurement_rows, num_times = _truncate_time_axis(measurement_rows, num_times_tensor, session_id, logger)
        if num_times <= 0:
            raise ValueError("No valid time profiles available after tensor/time-axis validation.")
        measurement_start_times = pd.to_datetime(measurement_rows["start_time_utc"], utc=True)
        reference_time = measurement_start_times.iloc[0]
        start_offsets, stop_offsets, normalization_attrs = _scc_time_axis(
            measurement_rows,
            reference_time,
            config,
            logger,
            label="Measurement",
        )
        laser_pointing_angle_deg = station_pointing_angle_deg_from_zenith(config)
        pressure_hpa = _surface_representative_value(weather_data, "pressure_hpa")
        temperature_c = _surface_representative_value(weather_data, "temperature_c")
        laser_shots = _laser_shot_matrix(lidar_data, num_times, num_channels)
        with nc.Dataset(netcdf_path, "w", format="NETCDF4") as ds:
            ds.setncatts(build_level0_global_attributes(session_id, lidar_data, group_df, weather_data, config))
            if normalization_attrs:
                ds.setncatts(normalization_attrs)
                normalized_stop_time = reference_time + pd.to_timedelta(int(np.max(stop_offsets)), unit="s")
                ds.setncattr("RawData_Stop_Time_UT", normalized_stop_time.strftime("%H%M%S"))
            _create_level0_dimensions(ds, num_times=num_times, num_channels=num_channels, num_points=num_points)
            variables = _create_level0_core_variables(ds, include_channel_ids=_scc_ready(config))
            _write_surface_weather_series(ds, weather_data)
            _write_solar_context(ds, measurement_rows, config)
            variables["raw_data_start"][:, 0] = start_offsets
            variables["raw_data_stop"][:, 0] = stop_offsets
            variables["raw_lidar_data"][:] = _stack_raw_lidar_data(tensors, channels, num_times, num_points)
            variables["laser_pointing_angle"][:] = np.array([laser_pointing_angle_deg], dtype=np.float64)
            variables["laser_pointing_angle_of_profiles"][:, 0] = np.zeros(num_times, dtype=np.int32)
            variables["laser_shots"][:] = laser_shots
            variables["molecular_calc"].assignValue(np.int32(0))
            variables["pressure_at_station"].assignValue(np.float64(pressure_hpa))
            variables["temperature_at_station"].assignValue(np.float64(temperature_c))
            _write_channel_metadata(variables, channels, lidar_data, config)
            _write_daq_range(ds, channels, lidar_data)
            _write_lr_input(ds, channels, config)
            write_dark_current_profile(
                ds,
                group_df,
                channels,
                num_channels,
                num_points,
                logger,
                config=config,
            )
    except Exception as exc:
        raise RuntimeError(f"Failed to build NetCDF: {exc}") from exc
