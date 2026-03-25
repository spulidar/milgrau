"""
LIdar BInary Data Standardized - LIBIDS
Script to read raw lidar binary data, clean up spurious data (temp.dat, AutoSave.dpp, 
invalid laser shots), and directly convert the valid data into standardized NETCDF format 
for the Single Calculus Chain (SCC) algorithm.

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from statistics import mode
import pandas as pd

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"LIBIDS_run_{datetime.now().strftime('%Y%m%d')}.log")

logger = logging.getLogger("LIBIDS")
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
fh = logging.FileHandler(log_filename)
fh.setFormatter(formatter)
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.propagate = False

from atmospheric_lidar.licel import LicelLidarMeasurement
from atmospheric_lidar_parameters import (
    msp_netcdf_parameters_system484,
    msp_netcdf_parameters_system565,
)

# ==========================================
# GLOBAL CLASS DEFINITIONS 
# ==========================================
class LidarMeasurement_484(LicelLidarMeasurement):
    extra_netcdf_parameters = msp_netcdf_parameters_system484

class LidarMeasurement_565(LicelLidarMeasurement):
    extra_netcdf_parameters = msp_netcdf_parameters_system565

# ==========================================
# SETTINGS
# ==========================================
INCREMENTAL_PROCESSING = True  
MAX_WORKERS_IO = 4             
MAX_WORKERS_CPU = 2            

ROOT_DIR = os.getcwd()
FILES_DIR_STAND = "01-data"
NETCDF_DIR = "03-netcdf_data"

DATADIR = os.path.join(ROOT_DIR, FILES_DIR_STAND)

def make_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            logger.error(f"Failed to create directory {path}: {e}")

def readfiles(datadir_name):
    """
    Scans the directory for valid binary files. 
    It is now immune to bad folder naming, relying only on the word 'dark' 
    to flag dark current measurements.
    """
    filepath = []
    meas_type = []

    if not os.path.exists(datadir_name):
        logger.error(f"Raw data directory not found: {datadir_name}")
        return filepath, meas_type

    for dirpath, dirnames, files in os.walk(datadir_name):
        dirnames.sort()
        files.sort()
        for file in files:
            full_path = os.path.join(dirpath, file)
            
            # Clean up spurious files
            if file.endswith(".dat") or file.endswith(".dpp") or file.endswith(".zip"):
                try:
                    os.remove(full_path)
                    logger.debug(f"Spurious file deleted: {file}")
                except Exception as e:
                    logger.warning(f"Could not delete file {file}: {e}")
            else:
                filepath.append(full_path)
                
                # Check if it's a dark current measurement based purely on path string
                if "dark" in full_path.lower():
                    meas_type.append("dark_current")
                else:
                    meas_type.append("measurements")
                    
    return filepath, meas_type

def read_header(filepath):
    """Read binary file header to extract metadata. Returns UTC time."""
    try:
        with open(filepath, "rb") as f:
            _ = f.readline().decode("utf-8")
            lines = [f.readline().decode("utf-8") for _ in range(3)]
        
        start_time_str = lines[0][10:29].strip()
        stop_time_str = lines[0][30:49].strip()
        n_shots = int(lines[1][16:21])
        laser_freq = int(lines[1][22:27])
        
        start_time_utc = datetime.strptime(start_time_str, "%d/%m/%Y %H:%M:%S")
        stop_time_utc = datetime.strptime(stop_time_str, "%d/%m/%Y %H:%M:%S")
        duration = (stop_time_utc - start_time_utc).total_seconds()
        
        return start_time_utc, stop_time_utc, duration, n_shots, laser_freq
    except Exception as e:
        logger.error(f"Error reading header of file {filepath}: {e}")
        return None, None, None, None, None

def classify_period(local_dt):
    """Classifies measurement period based on local time."""
    if 6 <= local_dt.hour < 12:
        return 'am'
    elif 12 <= local_dt.hour < 18:
        return 'pm'
    else:
        return 'nt'

def get_night_date(local_dt):
    """Adjust date for night measurements (before 6 AM local belongs to the previous day)."""
    if local_dt.hour < 6:
        return local_dt - pd.Timedelta(days=1)
    return local_dt

def process_single_netcdf(args):
    meas_id, group_df = args
    date_str = meas_id[:8]
    period = meas_id[8:]
    save_id = f"{date_str}sa{period}"
    
    # Extract year and month for subfolder organization
    year_str = save_id[:4]
    month_str = save_id[4:6]
    
    # Build the new hierarchical path
    out_dir = os.path.join(ROOT_DIR, NETCDF_DIR, year_str, month_str)
    netcdf_path = os.path.join(out_dir, f"{save_id}.nc")

    files_meas = group_df[group_df["meas_type"] == "measurements"]["filepath"].tolist()
    files_meas_dc = group_df[group_df["meas_type"] != "measurements"]["filepath"].tolist()

    if not files_meas:
        logger.error(f"No measurement files found for: {save_id}")
        return f"[FAILED] {save_id}"

    try:
        MeasurementClass = LidarMeasurement_565 if period in ["am", "pm"] else LidarMeasurement_484
        my_measurement = MeasurementClass(files_meas)
        
        if files_meas_dc:
            my_dark_measurement = MeasurementClass(files_meas_dc)
            my_measurement.dark_measurement = my_dark_measurement
            
        my_measurement.info["Measurement_ID"] = save_id
        my_measurement.info["Temperature"] = "25"
        my_measurement.info["Pressure"] = "940"
        
        duration = mode(group_df["duration"])
        freq = mode(group_df["laser_freq"])
        expected_shots = int(duration * freq)
        
        my_measurement.info["Accumulated_Shots"] = str(expected_shots)
        my_measurement.info["Laser_Frequency"] = str(freq)
        my_measurement.info["Measurement_Duration"] = str(duration)

        # Ensure the new Year/Month directories exist
        make_dir(out_dir)
        my_measurement.save_as_SCC_netcdf(netcdf_path)
        
        del my_measurement 
        
        return f"[OK] NetCDF file successfully saved: {year_str}/{month_str}/{save_id}.nc"
        
    except Exception as e:
        logger.error(f"[FAILED] Fatal error converting {save_id} to NetCDF. Error: {e}", exc_info=True)
        return f"[ERROR] {save_id}"

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    logger.info("=== Starting LIBIDS processing ===")
    
    file_paths, file_types = readfiles(DATADIR)
    
    if not file_paths:
        logger.warning("No valid files found for processing. Exiting.")
        exit()

    logger.info(f"Reading headers of {len(file_paths)} files...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_IO) as executor:
        results = list(executor.map(read_header, file_paths))

    start_times_utc, stop_times, durations, nshots_list, laser_freqs = zip(*results)

    df = pd.DataFrame({
        "filepath": file_paths,
        "meas_type": file_types,
        "start_time_utc": start_times_utc,
        "stop_time": stop_times,
        "nshots": nshots_list,
        "duration": durations,
        "laser_freq": laser_freqs,
    })

    df = df.dropna(subset=['start_time_utc'])

    if df.empty:
        logger.warning("All headers failed to read. Exiting.")
        exit()

    # ==========================================
    # TIMEZONE & PERIOD INTELLIGENCE
    # ==========================================
    logger.info("Applying timezone conversions (UTC -> America/Sao_Paulo)...")
    
    # 1. Convert to Pandas Datetime and set UTC
    df['start_time_utc'] = pd.to_datetime(df['start_time_utc']).dt.tz_localize('UTC')
    
    # 2. Convert to local time (São Paulo)
    df['start_time_local'] = df['start_time_utc'].dt.tz_convert('America/Sao_Paulo')
    
    # 3. Extract the correct period (am/pm/nt) and the effective date based on LOCAL time
    df['flag_period'] = df['start_time_local'].apply(classify_period)
    df['meas_id'] = df['start_time_local'].apply(get_night_date).dt.strftime('%Y%m%d') + df['flag_period']

    # ==========================================
    # EARLY INCREMENTAL FILTER 
    # ==========================================
    if INCREMENTAL_PROCESSING:
        logger.info("Applying early incremental filter...")
        
        def needs_processing(meas_id):
            save_id = f"{meas_id[:8]}sa{meas_id[8:]}"
            year_str = save_id[:4]
            month_str = save_id[4:6]
            
            # Check for existence in the new subfolder structure
            expected_path = os.path.join(ROOT_DIR, NETCDF_DIR, year_str, month_str, f"{save_id}.nc")
            return not os.path.exists(expected_path)

        unique_meas_ids = df["meas_id"].unique()
        valid_meas_ids = [mid for mid in unique_meas_ids if needs_processing(mid)]
        skipped_count = len(unique_meas_ids) - len(valid_meas_ids)
        
        if skipped_count > 0:
            logger.info(f"[SKIPPED] {skipped_count} measurement periods already exist as NetCDF.")
            
        df = df[df["meas_id"].isin(valid_meas_ids)]

    if df.empty:
        logger.info("=== No new data to process. LIBIDS finished successfully! ===")
        exit()

    # ==========================================
    # QUALITY CONTROL
    # ==========================================
    df_good_list = []
    df_bad_list = []

    logger.info("Evaluating laser shots quality and consistency...")
    for (meas_id), group in df.groupby(["meas_id"]):
        try:
            duration = mode(group["duration"])
            freq = mode(group["laser_freq"])
            expected_shots = duration * freq

            bad_condition = (
                    (group["nshots"] == 0)
                    | (group["nshots"] <= expected_shots - 2e-3 * expected_shots)
                    | (group["nshots"] >= expected_shots + 2e-3 * expected_shots)
                )
                
            df_bad_list.append(group.loc[bad_condition])
            df_good_list.append(group.loc[~bad_condition])
            
        except Exception as e:
            logger.warning(f"Error checking file condition in group {meas_id}: {e}")
            df_bad_list.append(group)

    df_bad = pd.concat(df_bad_list).reset_index(drop=True) if df_bad_list else pd.DataFrame()
    df_good = pd.concat(df_good_list).reset_index(drop=True) if df_good_list else pd.DataFrame()

    total_files = len(df)
    bad_files = len(df_bad)

    if total_files > 0:
        loss_percent = (bad_files / total_files) * 100
        logger.info(f"Quality Report: {total_files} files evaluated.")
        logger.info(f"Bad/rejected files: {bad_files} ({loss_percent:.2f}%)")
        if loss_percent > 10:
            logger.warning("High data loss rate detected (>10%). Check hardware or atmospheric conditions.")

    # ==========================================
    # NETCDF CONVERSION
    # ==========================================
    if not df_good.empty:
        modo = "Incremental" if INCREMENTAL_PROCESSING else "Rewriting"
        logger.info(f"Starting NetCDF SCC conversion (Mode: {modo}) with {MAX_WORKERS_CPU} CPU processes...")
        
        process_args = [(meas_id, group) for meas_id, group in df_good.groupby("meas_id")]
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS_CPU) as executor:
            for result in executor.map(process_single_netcdf, process_args):
                logger.info(result)
                
        logger.info("=== LIBIDS processing finished successfully! ===")
    else:
        logger.warning("No data with sufficient quality survived for NetCDF conversion.")