"""
MILGRAU Suite - Visualization Utilities
Handles matplotlib configurations, standard Lidar quicklooks,
error band plotting, and aesthetic formatting (logos, footers).

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Safe for headless servers/background processing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from datetime import datetime

# Import our physics math for the cloud mask
from functions.physics_utils import calculate_dynamic_cloud_threshold

# ==========================================
# STRING & METADATA FORMATTING
# ==========================================

def extract_datetime_strings(ds):
    """
    Extracts and formats start and end times from NetCDF attributes.
    Returns strings for the plot title and footer.
    """
    try:
        dt_in_str = f"{ds.attrs['RawData_Start_Date']}{str(ds.attrs['RawData_Start_Time_UT']).zfill(6)}"
        dt_end_str = f"{ds.attrs['RawData_Start_Date']}{str(ds.attrs['RawData_Stop_Time_UT']).zfill(6)}"

        dt_in = datetime.strptime(dt_in_str, "%Y%m%d%H%M%S")
        dt_end = datetime.strptime(dt_end_str, "%Y%m%d%H%M%S")

        date_title = f"{dt_in.strftime('%d %b %Y - %H:%M')} to {dt_end.strftime('%d %b %Y - %H:%M')} UTC"
        date_footer = dt_in.strftime("%d %b %Y")
        return date_title, date_footer
    except Exception:
        return "Unknown date", "Unknown date"

def format_channel_name(raw_name):
    """
    Formats the SCC channel name for plot legends.
    Example: '00532.o_ph' -> '532nm PC'
    """
    try:
        parts = raw_name.split('.')
        wavelength = int(parts[0])
        mode = parts[1].split('_')[1].upper()
        if mode == 'PH':
            mode = 'PC' # Convert 'Photon' to standard 'Photon Counting'
        return f"{wavelength}nm {mode}"
    except Exception:
        return raw_name

# ==========================================
# AESTHETICS & LOGOS
# ==========================================

def add_footer_and_logos(fig, date_footer, root_dir):
    """
    Adds standardized footer text and institutional logos to the figure.
    """
    # Text
    fig.text(0.10, 0.03, date_footer, fontsize=12, fontweight="bold", va="center")
    fig.text(0.30, 0.03, "SPU-Lidar Station", fontsize=12, fontweight="bold", color="black", ha="right", va="center")

    # Logos configuration
    logos = [
        (os.path.join(root_dir, "img", "by-nc-nd.png"), 0.040),
        (os.path.join(root_dir, "img", "lalinet_logo2.jpeg"), 0.070),
        (os.path.join(root_dir, "img", "logo_leal.jpeg"), 0.065),
    ]

    spacing = 0.006
    y_pos = 0.005
    x_right = 0.98

    for path, height in logos:
        if not os.path.exists(path):
            continue

        img = mpimg.imread(path)
        h, w = img.shape[:2]
        width = height * (w / h) # Maintain aspect ratio
        
        x_left = x_right - width
        ax = fig.add_axes([x_left, y_pos, width, height], zorder=12)
        ax.imshow(img)
        ax.axis("off")
        
        x_right = x_left - spacing

# ==========================================
# PLOTTING ENGINES
# ==========================================

def plot_quicklook(data_slice, error_slice, max_altitude, channel_name, ds, output_folder, file_name_prefix, config, root_dir):
    """
    Generates a 2D colormap quicklook and a 1D mean profile with 1-sigma error bands.
    """
    date_title, date_footer = extract_datetime_strings(ds)
    pretty_channel = format_channel_name(channel_name)
    meas_title = f"RCS at {pretty_channel} (0 - {max_altitude} km)\n{date_title}\nSPU Lidar Station - São Paulo"

    fig = plt.figure(figsize=[15, 7.5])
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.03)
    
    # --- Colormap Axis (2D) ---
    ax0 = plt.subplot(gs[0])
    
    apply_cloud_mask = config.get("processing", {}).get("apply_cloud_mask", True)
    
    if apply_cloud_mask:
        multiplier = config.get("processing", {}).get("cloud_mask_multiplier", 10.0)
        dynamic_threshold = calculate_dynamic_cloud_threshold(data_slice, multiplier=multiplier)
        
        rcs_aerosol = data_slice.where(data_slice < dynamic_threshold)
        rcs_clouds = data_slice.where(data_slice >= dynamic_threshold)
        
        plot = rcs_aerosol.plot(x='time', y='altitude', cmap='jet', robust=True, vmin=0, add_colorbar=False, ax=ax0, add_labels=False)
        cloud_cmap = ListedColormap(['white'])
        rcs_clouds.plot(x='time', y='altitude', cmap=cloud_cmap, add_colorbar=False, ax=ax0, add_labels=False)
    else:
        plot = data_slice.plot(x='time', y='altitude', cmap='jet', robust=True, vmin=0, add_colorbar=False, ax=ax0, add_labels=False)
    
    min_altitude = 0.16 if "AN" in pretty_channel else 0.5

    ax0.set_title(meas_title, fontsize=15, fontweight="bold", loc='center')
    ax0.set_xlabel('Time (UTC)', fontsize=13, fontweight="bold")
    ax0.set_ylabel('Altitude (km a.g.l.)', fontsize=13, fontweight="bold")
    ax0.set_ylim(min_altitude, max_altitude)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # --- Mean RCS Axis (1D) ---
    ax1 = plt.subplot(gs[1], sharey=ax0)
    mean_profile = data_slice.mean(dim='time')
    
    n_profiles = error_slice.sizes['time']
    mean_error = np.sqrt((error_slice**2).sum(dim='time')) / n_profiles

    smooth_profile = mean_profile.rolling(altitude=20, min_periods=1).mean()
    smooth_error = mean_error.rolling(altitude=20, min_periods=1).mean()

    line_color = "black"
    if "532" in channel_name: line_color = "forestgreen"
    elif "355" in channel_name: line_color = "rebeccapurple"
    elif "1064" in channel_name: line_color = "crimson"
    elif "387" in channel_name: line_color = "darkblue"

    ax1.plot(smooth_profile, smooth_profile.altitude, color=line_color, linewidth=2)
    ax1.fill_betweenx(smooth_profile.altitude, smooth_profile - smooth_error, smooth_profile + smooth_error, color=line_color, alpha=0.3, edgecolor="none")

    ax1.set_xlabel('Mean RCS', fontsize=12, fontweight="bold")
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.grid(True, linestyle='--', alpha=0.6, which='both')

    # Colorbar
    plt.subplots_adjust(left=0.14, bottom=0.15, right=0.95, top=0.88)
    cb_ax = fig.add_axes([0.06, 0.15, 0.015, 0.73])
    cb = fig.colorbar(plot, cax=cb_ax, orientation='vertical')
    cb.set_label("Intensity [a.u.]", fontsize=12, fontweight="bold")
    cb_ax.yaxis.set_ticks_position('left')
    cb_ax.yaxis.set_label_position('left')

    add_footer_and_logos(fig, date_footer, root_dir)

    safe_channel_name = pretty_channel.replace(" ", "_")
    file_name = f'Quicklook_{file_name_prefix}_{safe_channel_name}_{max_altitude}km.webp'
    plt.savefig(os.path.join(output_folder, file_name), dpi=120)
    plt.close(fig)

def plot_global_mean_rcs(ds, output_folder, file_name_prefix, config, root_dir):
    """
    Plots all requested channels on a single mean RCS graph with error bands.
    """
    altitude_ranges = config.get("visualization", {}).get("altitude_ranges_km", [5, 15, 30])
    channels_to_plot = config.get("visualization", {}).get("channels_to_plot", [])
    max_altitude = max(altitude_ranges)
    
    date_title, date_footer = extract_datetime_strings(ds)
    meas_title = f"Mean RCS (0 - {max_altitude} km)\n{date_title}\nSPU Lidar Station - São Paulo"

    fig, ax = plt.subplots(figsize=(8, 9.6))
    fig.subplots_adjust(top=0.90, bottom=0.15)

    base_colors = { 355: "rebeccapurple", 387: "darkblue", 408: "darkcyan", 530: "orange", 532: "forestgreen", 1064: "crimson" }
    plotted_anything = False

    for ch in channels_to_plot:
        if ch in ds.channel.values:
            label = format_channel_name(ch)
            try:
                wavelength = int(ch.split('.')[0])
                color = base_colors.get(wavelength, "black")
            except Exception:
                color = "black"

            line_style = "-" if "an" in ch.lower() else "--"

            rc_signal = ds['Range_Corrected_Signal'].sel(channel=ch)
            rc_error = ds['Range_Corrected_Signal_Error'].sel(channel=ch)

            sig_slice = rc_signal.where(rc_signal['altitude'] <= max_altitude, drop=True)
            err_slice = rc_error.where(rc_error['altitude'] <= max_altitude, drop=True)

            mean_profile = sig_slice.mean(dim='time')
            n_profiles = err_slice.sizes['time']
            mean_error = np.sqrt((err_slice**2).sum(dim='time')) / n_profiles

            smooth_profile = mean_profile.rolling(altitude=50, min_periods=1).mean()
            smooth_error = mean_error.rolling(altitude=50, min_periods=1).mean()

            ax.plot(smooth_profile, smooth_profile.altitude, color=color, linestyle=line_style, label=label, linewidth=2)
            ax.fill_betweenx(smooth_profile.altitude, smooth_profile - smooth_error, smooth_profile + smooth_error, color=color, alpha=0.2, edgecolor="none")
            plotted_anything = True

    if not plotted_anything:
        plt.close(fig)
        return

    ax.set_title(meas_title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean RCS [a.u.]", fontsize=14, fontweight="bold")
    ax.set_ylabel("Altitude (km a.g.l.)", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim(0, max_altitude)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, which='both', alpha=0.5)

    add_footer_and_logos(fig, date_footer, root_dir)

    file_name = f'GlobalMeanRCS_{file_name_prefix}.webp'
    plt.savefig(os.path.join(output_folder, file_name), dpi=120)
    plt.close(fig)