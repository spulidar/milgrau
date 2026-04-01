"""
MILGRAU Suite - Level 2: Lidar Elastic Backscatter and Extinction Analysis Routine (LEBEAR)
Performs optical inversion (KFS) using metadata and sounding data retrieved from Level 1.
This version is optimized for sequential processing and relies on Level 1 as the data source.

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import os
import logging
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import warnings

# Import Physics & Viz Functions
from functions.viz_utils import plot_gluing_qa, plot_molecular_qa, plot_kfs_results
from functions.physics_utils import (
    calculate_molecular_profile, 
    slide_glue_signals, 
    kfs_inversion_monte_carlo, 
    find_optimal_reference_altitude
)

# Suppress expected warnings for math on NaN slices
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# SEQUENTIAL WORKER FUNCTION
# ==========================================
def process_level2_file(args):
    nc_path, config, root_dir = args
    
    # Logger setup (Sequential safe)
    logger = logging.getLogger("LEBEAR")
    
    try:
        stem = Path(nc_path).stem.replace("_level1_rcs", "")
        year, month = stem[:4], stem[4:6]
        
        # Directory Management
        out_dir_base = config['directories']['processed_data']
        base_dir = Path(root_dir) / out_dir_base / year / month / stem
        out_dir = base_dir # Results saved in the same measurement folder
        
        level2_path = out_dir / f"{stem}_level2_optical.nc"
        
        # Check for incremental processing
        if config['processing']['incremental'] and level2_path.exists():
            return f"[SKIPPED] Level 2 already exists for {stem}"

        logger.info(f"[{stem}] Loading Level 1 RCS data...")
        with xr.open_dataset(nc_path) as ds:
            ds.load()
            
            # --- RETRIEVE ATMOSPHERIC DATA FROM LEVEL 1 ---
            # Try to reconstruct the radiosonde dataframe from Level 1 coordinates
            df_radio = None
            if "radiosonde_alt" in ds.coords:
                logger.info(f"  -> [{stem}] Radiosonde data found in Level 1 file. Using it for molecular profile.")
                df_radio = pd.DataFrame({
                    'alt': ds.radiosonde_alt.values,
                    'press': ds.Radiosonde_Pressure_hPa.values,
                    'temp': ds.Radiosonde_Temperature_K.values
                })
            else:
                logger.warning(f"  -> [{stem}] No radiosonde found in Level 1. KFS will use US Standard Atmosphere.")

            # Retrieve PBL and Tropopause from attributes (calculated in LIPANCORA)
            pbl_height_km = ds.attrs.get("pbl_height_km", -999.0)
            cpt_km = ds.attrs.get("tropopause_cpt_km", -999.0)
            lrt_km = ds.attrs.get("tropopause_lrt_km", -999.0)

            logger.info(f"[{stem}] Starting Optical Inversion Pipeline...")

            wavelengths = config['processing']['wavelengths']
            bin_m = float(np.mean(np.diff(ds.range.values)))
            alt_m = ds.range.values
            alt_km = alt_m / 1000.0

            results_beta_mean, results_beta_std = [], []
            results_ext_mean, results_ext_std = [], []
            final_channels = []

            for wl in wavelengths:
                logger.info(f"[{stem}] Processing {wl} nm...")
                
                # Identify channels for this wavelength
                ch_an = f"ch{wl}an"
                ch_ph = f"ch{wl}ph"
                
                rcs_an = ds.Range_Corrected_Signal.sel(channel=ch_an).mean(dim="time").values if ch_an in ds.channel.values else None
                rcs_pc = ds.Range_Corrected_Signal.sel(channel=ch_ph).mean(dim="time").values if ch_ph in ds.channel.values else None

                # --- A. GLUING ---
                if rcs_an is not None and rcs_pc is not None and wl != 1064:
                    logger.info(f"[{stem}] Gluing Analog and PC for {wl} nm...")
                    glued_rcs, _ = slide_glue_signals(rcs_an, rcs_pc, alt_m, config)
                    
                    if config['inversion']['interactive_qa']:
                        plot_gluing_qa(alt_km, rcs_an, rcs_pc, glued_rcs, config, f"{wl} nm", ds, root_dir, os.path.join(out_dir,'level2-plots'), stem)
                else:
                    if wl == 1064:
                        logger.info(f"[{stem}] Bypassing gluing for 1064 nm (Infrared uses Analog-only standard).")
                    glued_rcs = rcs_an if rcs_an is not None else rcs_pc

                if glued_rcs is None:
                    logger.warning(f"  -> [{stem}] Missing data for {wl} nm. Skipping.")
                    continue

                # --- B. MOLECULAR CALIBRATION & REFERENCE ---
                logger.info(f"[{stem}] Calculating Rayleigh Scattering for {wl} nm...")
                simulated_mol = calculate_molecular_profile(alt_m, wl, df_radio)
                
                # Dynamic Reference Altitude Search
                m_conf = config['inversion']
                min_alt_idx = int(m_conf['ref_alt_min_m'] / bin_m)
                max_alt_idx = min(int(m_conf['ref_alt_max_m'] / bin_m), len(glued_rcs)-1)
                
                ref_idx = find_optimal_reference_altitude(glued_rcs, simulated_mol, min_alt_idx, max_alt_idx)
                logger.info(f"  -> [{stem}] Optimal KFS reference altitude: {alt_km[ref_idx]:.2f} km.")

                if m_conf['interactive_qa']:
                    plot_molecular_qa(alt_km, glued_rcs, simulated_mol, alt_km[min_alt_idx], alt_km[max_alt_idx], config, f"{wl} nm", ds, root_dir, os.path.join(out_dir,'level2-plots'), stem)

                # --- C. KFS INVERSION (MONTE CARLO) ---
                logger.info(f"[{stem}] Running KFS Monte Carlo Inversion (100 iterations)...")
                lr_fixed = m_conf['lidar_ratio_default'].get(wl, 50.0)
                
                beta_avg, beta_std, ext_avg, ext_std = kfs_inversion_monte_carlo(
                    glued_rcs, simulated_mol, ref_idx, lr_fixed, iterations=100
                )

                if m_conf['interactive_qa']:
                    plot_kfs_results(alt_km, beta_avg, beta_std, f"{wl} nm", os.path.join(out_dir,'level2-plots'), stem)

                results_beta_mean.append(beta_avg)
                results_beta_std.append(beta_std)
                results_ext_mean.append(ext_avg)
                results_ext_std.append(ext_std)
                final_channels.append(int(wl))

            # --- D. SAVE LEVEL 2 NETCDF ---
            logger.info(f"[{stem}] Packaging and saving Level 2 NetCDF...")
            
            # Prepare dataset coordinates and variables
            out_ds = xr.Dataset(
                {
                    "Particle_Backscatter_Coefficient": (("channel", "range"), np.array(results_beta_mean)),
                    "Particle_Backscatter_Error": (("channel", "range"), np.array(results_beta_std)),
                    "Particle_Extinction_Coefficient": (("channel", "range"), np.array(results_ext_mean)),
                    "Particle_Extinction_Error": (("channel", "range"), np.array(results_ext_std))
                },
                coords={
                    "channel": final_channels,
                    "range": alt_m
                }
            )

            # Pass down Global Metadata from Level 1
            out_ds.attrs = ds.attrs
            out_ds.attrs["processing_level"] = "Level 2: Optical Inversion (KFS Monte Carlo)"
            out_ds.attrs["pbl_height_km"] = float(pbl_height_km)
            out_ds.attrs["tropopause_cpt_km"] = float(cpt_km)
            out_ds.attrs["tropopause_lrt_km"] = float(lrt_km)
            out_ds.attrs["history"] = f"{ds.attrs.get('history', '')}\nInverted with MILGRAU LEBEAR on {datetime.now(timezone.utc).isoformat()} UTC"

            # Re-inject radiosonde variables if they were present
            if df_radio is not None:
                out_ds = out_ds.assign_coords(radiosonde_alt=("radiosonde_alt", df_radio['alt'].values))
                out_ds["Radiosonde_Pressure_hPa"] = (("radiosonde_alt",), df_radio['press'].values)
                out_ds["Radiosonde_Temperature_K"] = (("radiosonde_alt",), df_radio['temp'].values)

            out_ds.to_netcdf(level2_path)
            
        return f"[OK] Level 2 processing complete for {stem}"

    except Exception as e:
        return f"[FAILED] Error processing Level 2 for {Path(nc_path).name}: {e}"

# ==========================================
# MAIN ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    from functions.core_io import load_config, setup_logger
    config = load_config()
    logger = setup_logger("LEBEAR", config['directories']['log_dir'])
    logger.info("=== Starting LEBEAR processing (Level 2 Optical Inversion) ===")
    
    root_dir = os.getcwd()
    input_dir = os.path.join(root_dir, config['directories']['processed_data'])
    
    # We look for Level 1 RCS files to invert
    files = sorted(Path(input_dir).rglob("*_level1_rcs.nc"))

    if not files:
        logger.warning(f"No Level 1 NetCDF data found in '{input_dir}'. Exiting.")
        exit()

    interactive_qa = config.get('inversion', {}).get('interactive_qa', True)
    logger.info(f"Found {len(files)} Level 1 files. Execution: Sequential (Interactive QA: {interactive_qa})")

    success_count = 0
    for f in files:
        result = process_level2_file((str(f), config, root_dir))
        if "[OK]" in result:
            logger.info(result)
            success_count += 1
        else:
            logger.error(result)

    if success_count == len(files): 
        logger.info("=== LEBEAR processing finished successfully for all files! ===")