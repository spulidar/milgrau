"""
MILGRAU Suite - Physics & Math Utilities
Contains core mathematical functions, statistical thresholds (e.g., IQR for clouds),
and atmospheric measurement period classifications.

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import numpy as np
import pandas as pd
import xarray as xr

# ==========================================
# TIME & MEASUREMENT PERIOD CLASSIFICATION
# ==========================================

def classify_period(local_dt):
    """
    Classifies the measurement period based on local time.
    Standardized intervals:
        - 'am': Morning (06:00 to 11:59)
        - 'pm': Afternoon (12:00 to 17:59)
        - 'nt': Night (18:00 to 05:59 of the next day)
        
    Args:
        local_dt (datetime or pd.Timestamp): The local datetime of the measurement.
        
    Returns:
        str: 'am', 'pm', or 'nt'
    """
    if 6 <= local_dt.hour < 12:
        return 'am'
    elif 12 <= local_dt.hour < 18:
        return 'pm'
    else:
        return 'nt'

def get_night_date(local_dt):
    """
    Adjusts the effective date for night measurements. 
    In Lidar continuous monitoring, a measurement taken at 04:00 AM 
    meteorologically belongs to the previous day's night cycle.
    
    Args:
        local_dt (pd.Timestamp): The local datetime of the measurement.
        
    Returns:
        pd.Timestamp: Adjusted datetime for grouping.
    """
    if local_dt.hour < 6:
        return local_dt - pd.Timedelta(days=1)
    return local_dt

# ==========================================
# STATISTICAL FILTERS & MASKS
# ==========================================

def calculate_dynamic_cloud_threshold(data_array, multiplier=10.0):
    """
    Calculates a dynamic cloud threshold using the Interquartile Range (IQR) method.
    Clouds are treated as extreme positive outliers in the Range Corrected Signal (RCS) distribution.
    
    Args:
        data_array (xarray.DataArray or numpy.ndarray): The RCS data.
        multiplier (float): The sensitivity factor. Higher means less sensitive 
                            (only very dense clouds are masked). 10.0 is the Lidar default.
                            
    Returns:
        float: The calculated threshold above which a signal is considered a cloud.
    """
    # Calculate the 25th and 75th percentiles (using numpy to handle NaNs safely)
    p25 = np.nanpercentile(data_array, 25)
    p75 = np.nanpercentile(data_array, 75)
    
    # Calculate the Interquartile Range (the spread of the 'normal' aerosol/molecular data)
    iqr = p75 - p25
    
    # Define the threshold for extreme outliers (clouds)
    threshold = p75 + (multiplier * iqr)
    
    return threshold