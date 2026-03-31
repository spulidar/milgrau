"""
MILGRAU Suite - Core Input/Output Module
Handles configuration loading, robust logging setup, directory management,
and raw Licel binary file scanning/parsing.

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import os
import logging
import yaml
import urllib3
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path

def load_config(config_path="config.yaml"):
    """
    Loads the master configuration YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Error parsing YAML configuration: {exc}")

def setup_logger(module_name, log_dir="logs"):
    """
    Sets up a standardized logger for the MILGRAU suite.
    Prevents duplicate logging handlers if called multiple times.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{module_name}_run_{datetime.now().strftime('%Y%m%d')}.log")

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicates during interactive sessions
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler
    fh = logging.FileHandler(log_filename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console Handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.propagate = False
    return logger

def ensure_directories(*directories):
    """
    Safely creates multiple directories if they do not exist.
    """
    for d in directories:
        if not os.path.exists(d):
            try:
                os.makedirs(d)
            except OSError as e:
                logging.error(f"Failed to create directory {d}: {e}")

def scan_raw_files(datadir_name, logger=None):
    """
    Scans the directory for valid Licel binary files.
    Cleans up spurious files (.dat, .dpp, .zip) and flags dark current measurements.
    """
    filepath = []
    meas_type = []

    if not os.path.exists(datadir_name):
        if logger:
            logger.error(f"Raw data directory not found: {datadir_name}")
        return filepath, meas_type

    for dirpath, dirnames, files in os.walk(datadir_name):
        dirnames.sort()
        files.sort()
        for file in files:
            full_path = os.path.join(dirpath, file)
            
            # Clean up spurious files directly
            if file.endswith((".dat", ".dpp", ".zip")):
                try:
                    os.remove(full_path)
                    if logger:
                        logger.debug(f"Spurious file deleted: {file}")
                except Exception as e:
                    if logger:
                        logger.warning(f"Could not delete file {file}: {e}")
            else:
                filepath.append(full_path)
                
                # Check if it's a dark current measurement based purely on path string
                if "dark" in full_path.lower():
                    meas_type.append("dark_current")
                else:
                    meas_type.append("measurements")
                    
    return filepath, meas_type

def read_licel_header(filepath):
    """
    Reads the binary file header to extract vital metadata.
    Returns UTC times, duration, number of shots, and laser frequency.
    """
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
        return None, None, None, None, None


def fetch_wyoming_radiosonde(measurement_dt_utc, station_id, logger):
    """
    Dynamically fetches radiosonde data (Pressure, Altitude, Temperature) 
    from the University of Wyoming archive based on the measurement time.
    Returns a clean Pandas DataFrame or None if the server fails.
    """
    # 1. Determine the closest sounding time (00Z or 12Z) based on atmospheric logic
    hour_utc = measurement_dt_utc.hour
    if 0 <= hour_utc <= 8:
        target_dt = measurement_dt_utc
        rs_hour = '00'
    elif 9 <= hour_utc <= 20:
        target_dt = measurement_dt_utc
        rs_hour = '12'
    else: # 21 to 23 belongs to the next day's 00Z sounding
        target_dt = measurement_dt_utc + timedelta(days=1)
        rs_hour = '00'
        
    year = target_dt.strftime('%Y')
    month = target_dt.strftime('%m')
    day = target_dt.strftime('%d')
    
    url = f"http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR={year}&MONTH={month}&FROM={day}{rs_hour}&TO={day}{rs_hour}&STNM={station_id}"
    
    logger.info(f"  -> [RADIOSONDE] Fetching {year}-{month}-{day} {rs_hour}Z for station {station_id}...")
    
    try:
        # 15.0 seconds timeout protects the pipeline from infinite hanging
        http = urllib3.PoolManager(cert_reqs='CERT_NONE', timeout=15.0) 
        response = http.request('GET', url)
        
        if response.status != 200:
            logger.warning(f"  -> [RADIOSONDE ERROR] HTTP {response.status}. Wyoming server is down.")
            return None
            
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Check if the server returned a "Sorry, can't find data" message
        if soup.find('h2') is None or 'Can\'t' in soup.text:
            logger.warning("  -> [RADIOSONDE ERROR] Data not found on the server for this date.")
            return None
            
        pre_tag = soup.find('pre')
        if pre_tag is None:
            logger.warning("  -> [RADIOSONDE ERROR] Unexpected HTML format.")
            return None
            
        # 2. Extract and parse the raw text from the <pre> tag
        raw_text = pre_tag.text
        lines = raw_text.split('\n')
        
        # Find where the actual data table starts (after the dashed line)
        data_start_idx = 0
        for idx, line in enumerate(lines):
            if line.startswith("-----------------------------------------------------------------------------"):
                data_start_idx = idx + 1
                break
                
        if data_start_idx == 0:
            logger.warning("  -> [RADIOSONDE ERROR] Could not parse the table structure.")
            return None
            
        altitudes, pressures, temperatures = [], [], []
        
        for line in lines[data_start_idx:]:
            parts = line.split()
            if len(parts) >= 3: # We need at least Press, Height, Temp
                try:
                    press = float(parts[0])
                    alt = float(parts[1])
                    temp_c = float(parts[2])
                    
                    pressures.append(press)
                    altitudes.append(alt)
                    temperatures.append(temp_c + 273.15) # Convert Celsius to Kelvin
                except ValueError:
                    continue # Skip lines with text or missing values
                    
        if not altitudes:
            logger.warning("  -> [RADIOSONDE ERROR] Parsed table is empty.")
            return None
            
        # 3. Build the final dataframe and remove negative/corrupted values
        df = pd.DataFrame({'alt': altitudes, 'press': pressures, 'temp': temperatures})
        df = df[(df['alt'] > 0) & (df['press'] > 0) & (df['temp'] > 0)].dropna()
        
        logger.info("  -> [OK] Radiosonde data successfully parsed!")
        return df
        
    except Exception as e:
        logger.warning(f"  -> [RADIOSONDE ERROR] Failed to connect to Wyoming server: {e}")
        return None