"""
MILGRAU Suite - Level 0: LIdar BInary Data Standardized (LIBIDS)
Reads raw Licel binary data, sanitizes spurious files, classifies measurement
periods (UTC to Local Time), and converts valid data into SCC NetCDF format.

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import os
import traceback
import pandas as pd
from statistics import mode
from typing import Tuple, Dict, Any

# Import MILGRAU core functions
from functions.core_io import (
    load_config, 
    setup_logger, 
    ensure_directories, 
    scan_raw_files, 
    read_licel_header
    fetch_surface_weather
)
from functions.physics_utils import classify_period, get_night_date

# Import SCC specific libraries
from atmospheric_lidar.licel import LicelLidarMeasurement
from atmospheric_lidar_parameters import (
    msp_netcdf_parameters_system484,
    msp_netcdf_parameters_system565,
)

# ==========================================
# GLOBAL CLASS DEFINITIONS 
# ==========================================
class LidarMeasurement484(LicelLidarMeasurement):
    extra_netcdf_parameters = msp_netcdf_parameters_system484

class LidarMeasurement565(LicelLidarMeasurement):
    extra_netcdf_parameters = msp_netcdf_parameters_system565

# ==========================================
# SEQUENTIAL PROCESSING FUNCTION
# ==========================================
def process_single_netcdf(args: Tuple[str, pd.DataFrame, Dict[str, Any]]) -> str:
    """
    Converts a grouped pandas dataframe of raw binaries into an SCC compliant NetCDF.
    """
    meas_id, group_df, config = args
    
    date_str = meas_id[:8]
    period = meas_id[8:]
    save_id = f"{date_str}sa{period}"
    year_str, month_str = save_id[:4], save_id[4:6]
    
    out_dir = os.path.join(os.getcwd(), config['directories']['processed_data'], year_str, month_str, save_id)
    netcdf_path = os.path.join(out_dir, f"{save_id}.nc")

    files_meas = group_df[group_df["meas_type"] == "measurements"]["filepath"].tolist()
    files_meas_dc = group_df[group_df["meas_type"] != "measurements"]["filepath"].tolist()

    if not files_meas:
        return f"[FAILED] No measurement files found for {save_id}"

    try:
        # Instantiate the correct SCC class based on the time of day
        # 'am' and 'pm' periods use system 565, night ('nt') uses 484
        MeasurementClass = LidarMeasurement565 if period in ["am", "pm"] else LidarMeasurement484
        my_measurement = MeasurementClass(files_meas)
        
        # Inject Dark Current measurements if they exist
        if files_meas_dc:
            my_measurement.dark_measurement = MeasurementClass(files_meas_dc)
            
        # ---------------------------------------------------------
        # Metadata Injection for SCC Compliance
        # ---------------------------------------------------------
        
        my_measurement.info["Measurement_ID"] = save_id
        
        lat = float(config['physics'].get('latitude', -23.561))
        lon = float(config['physics'].get('longitude', -46.735))
        
        # Use the mean time of the dataset to query the weather
        dt_utc_mean = group_df['start_time_utc'].iloc[len(group_df) // 2].to_pydatetime()
        weather_data = fetch_surface_weather(dt_utc_mean, lat, lon)
        
        if weather_data:
            # Required variables for SCC
            my_measurement.info["Temperature"] = str(round(weather_data['temperature_c'], 1))
            my_measurement.info["Pressure"] = str(round(weather_data['pressure_hpa'], 1))
            
            # Custom MILGRAU variables (Will be added as Global Attributes in NetCDF)
            my_measurement.info["Relative_Humidity_Percent"] = str(round(weather_data['relative_humidity_percent'], 1))
            my_measurement.info["Cloud_Cover_Percent"] = str(round(weather_data['cloud_cover_percent'], 1))
            my_measurement.info["Wind_Speed_kmh"] = str(round(weather_data['wind_speed_kmh'], 1))
            
            logger.info(
                f"  -> [{save_id}] Weather applied: "
                f"{weather_data['temperature_c']}°C, {weather_data['pressure_hpa']} hPa, "
                f"RH {weather_data['relative_humidity_percent']}%"
            )
        else:
            default_temp = str(config['physics'].get('default_surface_temp_c', 25.0))
            default_press = str(config['physics'].get('default_surface_pressure_hpa', 940.0))
            my_measurement.info["Temperature"] = default_temp
            my_measurement.info["Pressure"] = default_press
            logger.info(f"  -> [{save_id}] Weather API failed. Applied fallback: {default_temp}°C, {default_press} hPa")
        
        # Determine expected baseline from the mode of the dataset to filter outliers
        duration = mode(group_df["duration"])
        freq = mode(group_df["laser_freq"])
        shots = mode(group_df["nshots"])
        
        my_measurement.info["Accumulated_Shots"] = str(shots)
        my_measurement.info["Laser_Frequency"] = str(freq)
        my_measurement.info["Measurement_Duration"] = str(duration)

        # Save to disk
        ensure_directories(out_dir)
        my_measurement.save_as_SCC_netcdf(netcdf_path)
        
        return f"[OK] NetCDF successfully saved: {year_str}/{month_str}/{save_id}.nc"
        
    except Exception as e:
        # traceback.format_exc() captures the full red error block for debugging
        error_details = traceback.format_exc()
        return f"[ERROR] Fatal error converting {save_id}.\n{error_details}"

# ==========================================
# MAIN ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    # 1. Load Configuration & Setup Logger
    config = load_config()
    logger = setup_logger("LIBIDS", config['directories']['log_dir'])
    logger.info("=== Starting LIBIDS processing (Level 0) ===")
    
    root_dir = os.getcwd()
    raw_dir = os.path.join(root_dir, config['directories']['raw_data'])
    netcdf_dir = os.path.join(root_dir, config['directories']['processed_data'])
    
    # 2. Scan and Sanitize Input Directory
    file_paths, file_types = scan_raw_files(raw_dir, logger)
    if not file_paths:
        logger.warning(f"No valid files found in {raw_dir}. Exiting pipeline.")
        exit()

    # 3. Read Headers (Sequential and Lightweight)
    logger.info(f"Reading headers of {len(file_paths)} files sequentially...")
    
    # Fast list comprehension to extract headers
    results = [read_licel_header(f) for f in file_paths]
    start_times_utc, stop_times, durations, nshots_list, laser_freqs = zip(*results)

    # Build DataFrame mapping the raw files
    df_raw = pd.DataFrame({
        "filepath": file_paths, 
        "meas_type": file_types,
        "start_time_utc": start_times_utc, 
        "stop_time": stop_times,
        "nshots": nshots_list, 
        "duration": durations, 
        "laser_freq": laser_freqs,
    }).dropna(subset=['start_time_utc'])

    if df_raw.empty:
        logger.warning("All headers failed to read. Exiting pipeline.")
        exit()

    # 4. Timezone & Atmospheric Period Intelligence
    logger.info("Applying timezone conversions (UTC -> Local) and classifying periods...")
    
    df_raw['start_time_utc'] = pd.to_datetime(df_raw['start_time_utc']).dt.tz_localize('UTC')
    df_raw['start_time_local'] = df_raw['start_time_utc'].dt.tz_convert('America/Sao_Paulo')
    
    df_raw['flag_period'] = df_raw['start_time_local'].apply(classify_period)
    df_raw['meas_id'] = df_raw['start_time_local'].apply(get_night_date).dt.strftime('%Y%m%d') + df_raw['flag_period']

    # 5. Incremental Processing Filter
    if config['processing']['incremental']:
        logger.info("Applying early incremental filter...")
        
        def needs_processing(meas_id: str) -> bool:
            save_id = f"{meas_id[:8]}sa{meas_id[8:]}"
            expected_path = os.path.join(netcdf_dir, save_id[:4], save_id[4:6], save_id, f"{save_id}.nc")
            return not os.path.exists(expected_path)

        valid_meas_ids = [mid for mid in df_raw["meas_id"].unique() if needs_processing(mid)]
        skipped_count = len(df_raw["meas_id"].unique()) - len(valid_meas_ids)
        
        if skipped_count > 0:
            logger.info(f"[SKIPPED] {skipped_count} measurement periods already exist as NetCDF.")
            
        df_raw = df_raw[df_raw["meas_id"].isin(valid_meas_ids)]

    if df_raw.empty:
        logger.info("=== No new data to process. LIBIDS finished successfully! ===")
        exit()

    # 6. Quality Control (Laser Shots consistency per Measurement)
    logger.info("Evaluating laser shots quality and consistency per measurement...")
    good_groups = []

    for meas_id, group in df_raw.groupby("meas_id"):
        try:
            expected_shots = mode(group["nshots"])
            # Filter criteria: 0 shots or deviation greater than 0.2% of expected
            bad_condition = (group["nshots"] == 0) | (abs(group["nshots"] - expected_shots) >= 2e-3 * expected_shots)
            
            bad_group = group.loc[bad_condition]
            good_group = group.loc[~bad_condition]
            
            total_files = len(group)
            bad_files = len(bad_group)
            loss_percent = (bad_files / total_files) * 100 if total_files > 0 else 0
            
            if bad_files > 0:
                log_msg = f"  -> [{meas_id}] QA Report: {bad_files}/{total_files} files rejected ({loss_percent:.1f}% loss)."
                if loss_percent > 10.0:
                    logger.warning(log_msg)
                else:
                    logger.info(log_msg)
            else:
                logger.info(f"  -> [{meas_id}] QA Report: 100% data retention. No files rejected.")

            if not good_group.empty:
                good_groups.append(good_group)
                
        except Exception as e:
            logger.warning(f"  -> [{meas_id}] Error evaluating quality: {e}")
            
    df_good = pd.concat(good_groups).reset_index(drop=True) if good_groups else pd.DataFrame()

    # NetCDF SCC Conversion 
    if not df_good.empty:
        logger.info("Starting NetCDF SCC conversion...")
        
        # Package arguments for the processing function
        process_args = [(meas_id, group, config) for meas_id, group in df_good.groupby("meas_id")]
        
        success_count = 0
        for args in process_args:
            result = process_single_netcdf(args)
            if "[OK]" in result:
                logger.info(result)
                success_count += 1
            else:
                logger.error(result) # Now prints the full traceback if it fails
                
        if success_count == len(process_args):
            logger.info("=== LIBIDS processing finished successfully for all groups! ===")
        else:
            logger.warning(f"=== LIBIDS finished with errors. Processed {success_count}/{len(process_args)} groups. ===")
    else:
        logger.warning("No data with sufficient quality survived for NetCDF conversion.")
