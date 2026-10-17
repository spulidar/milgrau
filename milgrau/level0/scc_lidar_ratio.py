"""SCC-only ancillary aerosol lidar-ratio profiles from station climatology.

SCC NetCDF input-file specification, Table 4 (v3.6): Altitude(points),
Lidar_Ratio(products, points), optional Lidar_Ratio_Error, product_ID(products),
and the Lidar_Station_Name global attribute. Never infer SCC product IDs
from channel IDs or from the configuration ID.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import netCDF4 as nc
import numpy as np
import pandas as pd

from milgrau.level0.netcdf import scc_measurement_id


def climatology_ancillary_enabled(config: Mapping[str, Any]) -> bool:
    catalog = config.get("_scc_catalog", {})
    policy = catalog.get("scc_policy", {}) if isinstance(catalog, Mapping) else {}
    lr_input = policy.get("lr_input", {}) if isinstance(policy, Mapping) else {}
    return (
        isinstance(lr_input, Mapping)
        and lr_input.get("source", "scc_db") == "climatology_ancillary"
    )


def _utc_month(measurement_rows: pd.DataFrame) -> str:
    """Require a single UTC calendar month across actual profile intervals."""
    starts = pd.to_datetime(measurement_rows["start_time_utc"], utc=True)
    stops = pd.to_datetime(measurement_rows["stop_time"], utc=True)
    if starts.empty:
        raise ValueError("SCC lidar-ratio ancillary requires measurement profiles.")
    if (stops <= starts).any():
        raise ValueError("SCC lidar-ratio ancillary requires positive profile durations.")
    # Inclusive last instant: a profile stopping at midnight belongs to the old month.
    periods = {
        timestamp.strftime("%Y%m")
        for timestamp in list(starts) + list(stops - pd.Timedelta(nanoseconds=1))
    }
    if len(periods) != 1:
        raise ValueError(
            "SCC climatology ancillary cannot represent two UTC calendar months "
            "in a single constant-LR profile; split the SCC submission first."
        )
    return starts.min().strftime("%m")


def _lr_products(
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    month: str,
) -> tuple[list[int], list[float], list[float]]:
    catalog = config["_station_catalog"]
    climatology = catalog["station"]["lidar_ratio_climatology"]
    monthly = climatology["monthly_sr"]
    std = climatology["std_sr"]
    lr_flags = context.get("lr_input", {})
    wavelengths = sorted({
        str(int(str(channel).split(".", 1)[0]))
        for channel, value in lr_flags.items() if int(value) == 0
    }, key=int)
    assigned = context.get("elastic_product_ids", {})
    product_ids: list[int] = []
    values: list[float] = []
    errors: list[float] = []
    for wavelength in wavelengths:
        raw_ids = assigned.get(wavelength, assigned.get(int(wavelength)))
        if raw_ids is None:
            raise ValueError(
                f"SCC climatology export for configuration "
                f"{context.get('scc_configuration_id')} requires true "
                f"elastic_product_ids.{wavelength} in station.yaml "
                "(SCC product_ID, not channel_ID or configuration_id)."
            )
        ids = raw_ids if isinstance(raw_ids, list) else [raw_ids]
        if not ids:
            raise ValueError(f"SCC elastic_product_ids.{wavelength} is empty.")
        try:
            lr = float(monthly[wavelength][month])
            lr_std = float(std[wavelength])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"No valid station lidar-ratio climatology for {wavelength}nm, "
                f"UTC month {month}."
            ) from exc
        if not np.isfinite(lr) or lr <= 0 or not np.isfinite(lr_std) or lr_std < 0:
            raise ValueError(f"Invalid station lidar-ratio climatology for {wavelength}nm.")
        for raw_id in ids:
            product_id = int(raw_id)
            if product_id <= 0 or product_id in product_ids:
                raise ValueError(f"Invalid or repeated SCC product_ID {product_id}.")
            product_ids.append(product_id)
            values.append(lr)
            errors.append(lr_std)
    return product_ids, values, errors


def scc_lr_file_name(measurement_id: str) -> str:
    """SCC deployment requires lr_YYYYMMDDcccNN.nc (NN is two digits).

    Use the UTC start-hour as stable two-digit suffix. A second measurement
    beginning in the same UTC hour and station/day must be rejected before
    staging rather than silently reusing its LR name.
    """
    from milgrau.level0.scc_staging import scc_raw_filename

    scc_raw_filename(measurement_id)  # validate exact ID grammar
    return f"lr_{measurement_id[:11]}{measurement_id[11:13]}.nc"


def write_scc_lidar_ratio_ancillary(
    scc_path: Path,
    measurement_rows: pd.DataFrame,
    lidar_data: Mapping[str, Any],
    config: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Path | None:
    """Write a self-contained, station-owned ancillary beside an SCC-only L0.

    Returns None when the SCC configuration has no elastic-only products.
    Does not alter MILGRAU scientific L0/L1/L2 products.
    """
    if not climatology_ancillary_enabled(config):
        return None
    month = _utc_month(measurement_rows)
    product_ids, values, errors = _lr_products(config, context, month)
    if not product_ids:
        return None

    start = pd.to_datetime(measurement_rows["start_time_utc"], utc=True).min()
    measurement_id = scc_measurement_id(start, str(context["station_id"]))
    output_path = scc_path.with_name(scc_lr_file_name(measurement_id))
    # This two-digit SCC ancillary namespace is smaller than Measurement_ID.
    # Never allow another pending measurement to claim the same ancillary
    # filename, even when its climatological data happen to be identical.
    from milgrau.level0.scc_staging import manifest_raw_name, validate_pending_scc_package

    for other in scc_path.parent.parent.glob("*"):
        if other == scc_path.parent or not other.is_dir() or other.is_symlink():
            continue
        if other.name[:11] != measurement_id[:11]:
            continue
        old_manifest = validate_pending_scc_package(other)
        if old_manifest["measurement_id"] == measurement_id:
            continue
        old_raw_name = manifest_raw_name(old_manifest)
        lr_names = set(old_manifest["files"]) - {old_raw_name}
        if output_path.name in lr_names:
            raise ValueError(
                f"SCC ancillary collision: {output_path.name} is already used "
                f"by {other.name}. Assign a different SCC LR sequence before export."
            )
    channels = [str(name) for name in lidar_data["channels"]]
    metadata = lidar_data.get("channel_metadata", {})
    bin_widths = [
        float(metadata[name]["bin_width_m"])
        for name in channels
    ]
    if not bin_widths or not np.all(np.isfinite(bin_widths)) or min(bin_widths) <= 0:
        raise ValueError("SCC lidar-ratio altitude requires positive native bin widths.")
    native_extents = []
    for channel, bin_width in zip(channels, bin_widths, strict=True):
        tensor = np.asarray(lidar_data["tensors"][channel])
        if tensor.ndim != 2 or tensor.shape[1] <= 0:
            raise ValueError(f"SCC lidar-ratio altitude requires non-empty profiles for {channel}.")
        native_extents.append(tensor.shape[1] * bin_width)
    # One common ascending grid must span every channel, including longer
    # photon-counting records with coarser native vertical resolutions.
    spacing = min(bin_widths)
    count = int(np.ceil(max(native_extents) / spacing)) + 2
    altitude = np.arange(count, dtype=np.float64) * spacing
    output_path.parent.mkdir(parents=True, exist_ok=True)

    handle, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".tmp", dir=output_path.parent,
    )
    os.close(handle)
    try:
        with nc.Dataset(temp_name, "w", format="NETCDF4_CLASSIC") as ds:
            ds.createDimension("points", altitude.size)
            ds.createDimension("products", len(product_ids))
            ds.setncattr("Lidar_Station_Name", str(context["station_id"]))
            ds.setncattr("Climatology_Source", "station.yaml:station.lidar_ratio_climatology")
            ds.setncattr("Climatology_Month_UTC", month)
            ds.setncattr("Measurement_ID", measurement_id)
            altitude_var = ds.createVariable("Altitude", "f8", ("points",))
            altitude_var.units = "m"
            altitude_var[:] = altitude
            ids_var = ds.createVariable("product_ID", "i4", ("products",))
            ids_var[:] = np.asarray(product_ids, dtype=np.int32)
            lr_var = ds.createVariable("Lidar_Ratio", "f8", ("products", "points"))
            lr_var.units = "sr"
            lr_var[:, :] = np.broadcast_to(
                np.asarray(values, dtype=np.float64)[:, None],
                (len(values), altitude.size),
            )
            err_var = ds.createVariable(
                "Lidar_Ratio_Error", "f8", ("products", "points"),
            )
            err_var.units = "sr"
            err_var[:, :] = np.broadcast_to(
                np.asarray(errors, dtype=np.float64)[:, None],
                (len(errors), altitude.size),
            )
        os.replace(temp_name, output_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return output_path


def scc_lr_ancillary_matches(
    scc_path: Path,
    raw_attrs: Mapping[str, Any],
    context: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> bool:
    """Verify ancillary identity/content so cached SCC exports cannot go stale."""
    flags = context.get("lr_input", {})
    if not any(int(value) == 0 for value in flags.values()):
        return "LR_File_Name" not in raw_attrs
    measurement_id = str(raw_attrs["Measurement_ID"])
    filename = scc_lr_file_name(measurement_id)
    if raw_attrs.get("LR_File_Name") != filename:
        return False
    ancillary_path = scc_path.with_name(filename)
    if not ancillary_path.is_file():
        return False
    month = str(raw_attrs["RawData_Start_Date"])[4:6]
    try:
        expected_ids, values, errors = _lr_products(
            {"_station_catalog": catalog}, context, month,
        )
        with nc.Dataset(ancillary_path) as ds:
            if (ds.getncattr("Lidar_Station_Name") != context["station_id"]
                or ds.getncattr("Climatology_Month_UTC") != month
                or ds.getncattr("Measurement_ID") != measurement_id):
                return False
            if ds.variables["Altitude"].dimensions != ("points",):
                return False
            if ds.variables["Lidar_Ratio"].dimensions != ("products", "points"):
                return False
            if ds.variables["product_ID"].dimensions != ("products",):
                return False
            altitude = np.asarray(ds.variables["Altitude"][:])
            if altitude.size < 2 or not np.isfinite(altitude).all():
                return False
            if not (np.diff(altitude) > 0).all():
                return False
            if not np.array_equal(ds.variables["product_ID"][:], expected_ids):
                return False
            lr = np.asarray(ds.variables["Lidar_Ratio"][:])
            err = np.asarray(ds.variables["Lidar_Ratio_Error"][:])
            if lr.shape != (len(values), altitude.size) or err.shape != lr.shape:
                return False
            if not np.allclose(lr, np.asarray(values)[:, None]):
                return False
            if not np.allclose(err, np.asarray(errors)[:, None]):
                return False
        return True
    except (OSError, KeyError, ValueError, IndexError, AttributeError):
        return False
