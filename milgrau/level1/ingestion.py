"""Level 0 ingestion helpers for Level 1 processing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from milgrau.io.contracts import validate_level0_contract


def _decode_level0_time_axis(ds: xr.Dataset) -> pd.DatetimeIndex:
    raw_start = ds["Raw_Data_Start_Time"]
    values = np.asarray(raw_start.values)
    if values.ndim == 2:
        values = values[:, 0]
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values, utc=True).tz_localize(None)
    raw_date = str(ds.attrs.get("RawData_Start_Date", ""))
    raw_time = str(ds.attrs.get("RawData_Start_Time_UT", ""))
    if len(raw_date) == 8 and len(raw_time) == 6:
        reference = pd.Timestamp(f"{raw_date}{raw_time}", tz="UTC")
        return pd.to_datetime(reference + pd.to_timedelta(values.astype(float), unit="s")).tz_localize(None)
    return pd.to_datetime(values.astype(float), unit="s", utc=True).tz_localize(None)


def _profile_for_measurement_time(config: Mapping[str, Any], measurement_time: pd.Timestamp) -> Mapping[str, Any]:
    catalog = config.get("_station_catalog")
    if not isinstance(catalog, Mapping):
        raise ValueError(
            "SCC raw input without channel_string requires a loaded station catalog so channel_ID values can be mapped."
        )
    profiles = catalog.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Station catalog profiles must be a non-empty list for SCC channel-ID mapping.")
    measurement_date = pd.Timestamp(measurement_time).date()
    matches: list[Mapping[str, Any]] = []
    for raw in profiles:
        if not isinstance(raw, Mapping):
            continue
        start = pd.Timestamp(raw["valid_from"]).date()
        end_raw = raw.get("valid_to")
        end = None if end_raw is None else pd.Timestamp(end_raw).date()
        if measurement_date >= start and (end is None or measurement_date <= end):
            matches.append(raw)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one station profile for SCC input date {measurement_date.isoformat()}, found "
            f"{[str(profile.get('id', '')) for profile in matches]}."
        )
    return matches[0]


def _period_mode_hint(ds: xr.Dataset) -> str | None:
    measurement_id = str(ds.attrs.get("Measurement_ID", "")).strip().lower()
    if measurement_id.endswith(("saam", "sapm")):
        return "day"
    if measurement_id.endswith("sant"):
        return "night"
    return None


def _configuration_id_hint(ds: xr.Dataset) -> int | None:
    for key in ("SCC_Configuration_ID", "scc_configuration_id"):
        if key not in ds.attrs:
            continue
        try:
            value = int(ds.attrs[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SCC configuration attribute {key} is not an integer.") from exc
        if value <= 0:
            raise ValueError(f"SCC configuration attribute {key} must be positive.")
        return value
    return None


def _channel_names_from_scc_ids(
    ds: xr.Dataset,
    config: Mapping[str, Any],
    time_index: pd.DatetimeIndex,
) -> tuple[list[str], list[str], list[int]]:
    if "channel_ID" not in ds:
        raise KeyError(
            "Level 0 input lacks channel_string and channel_ID. LIPANCORA needs either canonical channel names or "
            "numeric SCC channel IDs resolvable through station.yaml."
        )
    channel_ids_raw = np.asarray(ds["channel_ID"].values)
    if channel_ids_raw.ndim != 1 or channel_ids_raw.size != ds.sizes.get("channels", 0):
        raise ValueError("SCC channel_ID must contain exactly one value per channels entry.")
    try:
        channel_ids_float = channel_ids_raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("SCC channel_ID values must be numeric.") from exc
    rounded = np.rint(channel_ids_float)
    if (
        not np.all(np.isfinite(channel_ids_float))
        or np.any(channel_ids_float <= 0.0)
        or not np.allclose(channel_ids_float, rounded, rtol=0.0, atol=0.0)
    ):
        raise ValueError("SCC channel_ID values must be positive finite integers.")
    channel_ids = rounded.astype(np.int64)
    if np.unique(channel_ids).size != channel_ids.size:
        raise ValueError("SCC channel_ID contains duplicate channel identifiers.")
    if len(time_index) == 0:
        raise ValueError("SCC channel-ID mapping requires a non-empty measurement time axis.")

    profile = _profile_for_measurement_time(config, pd.Timestamp(time_index[0]))
    scc = profile.get("scc")
    if not isinstance(scc, Mapping):
        raise ValueError(
            f"Station profile {profile.get('id', '')!r} has no SCC mapping, so external channel_ID values cannot be canonicalized."
        )

    mode_hint = _period_mode_hint(ds)
    configuration_id_hint = _configuration_id_hint(ds)
    candidates: list[tuple[str, int, tuple[str, ...]]] = []
    for mode in ("day", "night"):
        raw_mode = scc.get(mode)
        if not isinstance(raw_mode, Mapping):
            continue
        configuration_id = int(raw_mode.get("configuration_id", 0))
        raw_channels = raw_mode.get("channels")
        if not isinstance(raw_channels, Mapping):
            continue
        reverse = {int(channel_id): str(channel) for channel, channel_id in raw_channels.items()}
        if not all(int(channel_id) in reverse for channel_id in channel_ids):
            continue
        if mode_hint is not None and mode != mode_hint:
            continue
        if configuration_id_hint is not None and configuration_id != configuration_id_hint:
            continue
        names = tuple(reverse[int(channel_id)] for channel_id in channel_ids)
        candidates.append((mode, configuration_id, names))

    if not candidates:
        qualifier = []
        if mode_hint is not None:
            qualifier.append(f"mode={mode_hint}")
        if configuration_id_hint is not None:
            qualifier.append(f"configuration_id={configuration_id_hint}")
        suffix = "" if not qualifier else f" ({', '.join(qualifier)})"
        raise ValueError(
            f"SCC channel_ID values {channel_ids.tolist()} do not resolve in station profile "
            f"{profile.get('id', '')!r}{suffix}."
        )

    unique_name_orders = {candidate[2] for candidate in candidates}
    if len(unique_name_orders) != 1:
        descriptions = [
            f"{mode}:config={configuration_id}:{list(names)}"
            for mode, configuration_id, names in candidates
        ]
        raise ValueError(
            "SCC channel_ID mapping is ambiguous between station configurations: " + "; ".join(descriptions)
        )

    names = list(next(iter(unique_name_orders)))
    modes = sorted({mode for mode, _, _ in candidates})
    configuration_ids = sorted({configuration_id for _, configuration_id, _ in candidates})
    return names, modes, configuration_ids


def _canonicalize_channel_identity(
    ds: xr.Dataset,
    config: Mapping[str, Any] | None,
    time_index: pd.DatetimeIndex,
) -> xr.Dataset:
    """Return one in-memory canonical channel identity for MILGRAU Level 1.

    Native MILGRAU Level 0 files provide ``channel_string`` directly. Standard
    SCC raw inputs commonly provide numeric ``channel_ID`` instead. For those
    files, station.yaml is the authority that maps SCC IDs to canonical physical
    channel names such as ``532.PC``. No wavelength/detector identity is guessed
    from an SCC ID alone.
    """
    if "channel_string" in ds:
        names = np.asarray(ds["channel_string"].values).astype(str)
        if names.ndim != 1 or names.size != ds.sizes.get("channels", 0):
            raise ValueError("Level 0 channel_string must contain exactly one value per channels entry.")
        if "channel_ID" in ds and config is not None:
            mapped_names, modes, configuration_ids = _channel_names_from_scc_ids(ds, config, time_index)
            if list(names) != mapped_names:
                raise ValueError(
                    "Level 0 channel_string disagrees with station.yaml mapping for the stored SCC channel_ID values."
                )
            result = ds.copy()
            result.attrs["milgrau_channel_identity_source"] = "channel_string_verified_against_station_scc_channel_ID"
            result.attrs["milgrau_scc_mapping_modes"] = ",".join(modes)
            result.attrs["milgrau_scc_mapping_configuration_ids"] = ",".join(str(value) for value in configuration_ids)
            return result
        return ds

    if config is None:
        raise ValueError(
            "Level 0 input lacks channel_string. Pass the loaded MILGRAU configuration so SCC channel_ID values can be "
            "resolved through station.yaml."
        )
    names, modes, configuration_ids = _channel_names_from_scc_ids(ds, config, time_index)
    result = ds.copy()
    result["channel_string"] = xr.DataArray(np.asarray(names, dtype=object), dims=("channels",))
    result.attrs["milgrau_level0_input_schema"] = "scc_raw_channel_ID_canonicalized"
    result.attrs["milgrau_channel_identity_source"] = "station.yaml_scc_channel_ID_mapping"
    result.attrs["milgrau_scc_mapping_modes"] = ",".join(modes)
    result.attrs["milgrau_scc_mapping_configuration_ids"] = ",".join(str(value) for value in configuration_ids)
    return result


def _native_range_resolutions(ds: xr.Dataset) -> np.ndarray:
    resolutions = np.asarray(ds["Raw_Data_Range_Resolution"].values, dtype=np.float64)
    if resolutions.ndim != 1 or resolutions.size != ds.sizes.get("channels", 0):
        raise ValueError("Raw_Data_Range_Resolution must contain one value per channel.")
    if not np.all(np.isfinite(resolutions)) or np.any(resolutions <= 0.0):
        raise ValueError("Raw_Data_Range_Resolution contains non-finite or non-positive values.")
    return resolutions


def _common_level1_altitude_grid(num_points: int, resolutions_m: np.ndarray) -> np.ndarray:
    if num_points <= 0:
        raise ValueError("Level 0 points dimension must be positive.")
    target_dz = float(np.min(resolutions_m))
    return (np.arange(num_points, dtype=np.float64) + 0.5) * target_dz


def load_and_prepare_level0(
    nc_path: str | Path,
    logger: logging.Logger,
    config: Mapping[str, Any] | None = None,
) -> tuple[xr.Dataset, np.ndarray]:
    """Load canonical MILGRAU or SCC raw Level 0 for Level 1 processing.

    Native per-channel range resolution remains available in
    ``Raw_Data_Range_Resolution`` so LIPANCORA can correct each channel on its
    native grid before interpolation. SCC files that provide ``channel_ID`` but
    no canonical ``channel_string`` are mapped through the active station.yaml
    profile before the strict Level 0 contract is evaluated.
    """
    try:
        ds = xr.open_dataset(nc_path)
        ds.load()
        time_dt = _decode_level0_time_axis(ds)
        ds = _canonicalize_channel_identity(ds, config, time_dt)
        validate_level0_contract(ds)
        ds = ds.assign_coords(time=time_dt)
        dz_values = _native_range_resolutions(ds)
        z_arr = _common_level1_altitude_grid(ds.sizes["points"], dz_values)
        if not np.allclose(dz_values, dz_values[0], rtol=0.0, atol=1e-6):
            logger.warning(
                "mixed native range resolution | %s m | common grid %.6f m",
                ", ".join(f"{value:.6f}" for value in dz_values),
                float(np.min(dz_values)),
            )
        channel_strings = ds["channel_string"].values.astype(str)
        ds = ds.rename({"points": "altitude", "channels": "channel"})
        ds = ds.assign_coords(altitude=z_arr, channel=channel_strings)
        ds["altitude"].attrs.update({"units": "m", "long_name": "Altitude above station (range-bin centers)"})
        logger.info(
            "%d profiles | %d channels | %d bins",
            ds.sizes.get("time", 0),
            ds.sizes.get("channel", 0),
            ds.sizes.get("altitude", 0),
        )
        return ds, z_arr
    except Exception as exc:
        logger.error("failed | %s | %s", Path(nc_path).name, exc)
        raise
