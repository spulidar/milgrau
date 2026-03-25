"""
Lidar Elastic Backscatter and Extinction Analysis Routine - LEBEAR
This script provides tools to processes the pre-analyzed data and retrieval the backscatter and extinction profiles of the atmosphere, tunning the
aerosol optical depth values measured with the lidar and the sun-photometer
Created on Sat Feb  5 07:51:19 2022
@author: Fábio J. S. Lopes
"""

import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from functions import milgrau_function as mf
from lidar_retrievals import glue, kfs, retrieval_plots
from molecular import lidarmolfit as lmfit

from datetime import datetime

"""initial setup"""
rootdir_name = os.getcwd()
files_dir_level1 = "05-data_level1"
files_dir_to_read = "02-preprocessed_corrected"
rawinsonde_folder = "07-rawinsonde"
datadir_name = os.path.join(rootdir_name, files_dir_level1)

"""flag to calculate molecular profile """

atmospheric_flag = "radiosounding"  # 'radiosounding' for rawinsonde data or 'us_std' for US-standard atmosphere

"""Input data from user"""
lamb = 532  # elastic wavelength to be analyzed (1064, 532 and 355 nm)
glueflag = "yes"  # glueing flag --> 'yes' for glueing process, otherwise, 'no'
channelmode = "AN"  # channel mode --> analogic: 'AN' or photocounting: 'PC'
ini_molref_alt = 5000  # initial altitude range for molecular calibration
fin_molref_alt = 25000  # final altitude range for molecular calibration
optical_prop_scale = 1e6  # optical properties graphics unit in Mega-meter
altitude_scale = 1000  # altitude scale in km (a.g.l.)
altitude_min = 0.0  # minimum altitude range for bacckscatter and extinction graphics
altitude_max = 30  # minimum altitude range for bacckscatter and extinction graphics
save_plots = True


lraerosol_mean = {
    "355": {
        "01": 75.95179420295759,
        "02": 116.0781281327432,
        "03": 93.00495168575227,
        "04": 96.04858083333747,
        "05": 81.16349118630502,
        "06": 78.26140169693433,
        "07": 78.20827262376653,
        "08": 81.14378469642025,
        "09": 84.68682233680514,
        "10": 84.87031582521946,
        "11": 91.06223890036068,
        "12": 99.3530907245606,
    },
    "532": {
        "01": 58.10536753439628,
        "02": 66.06093874128817,
        "03": 61.93402744043742,
        "04": 69.49695351083719,
        "05": 62.06715278749705,
        "06": 61.72606447952922,
        "07": 58.88385500748774,
        "08": 55.05473648423215,
        "09": 55.78110226421323,
        "10": 57.83180290771037,
        "11": 57.11318566352885,
        "12": 75.16787397694915,
    },
    "1064": {
        "01": 35.93783019664728,
        "02": 25.51960975045722,
        "03": 35.37311522020933,
        "04": 40.76340051962803,
        "05": 39.80843067334443,
        "06": 43.75700933143827,
        "07": 36.91946966241792,
        "08": 30.85041991885892,
        "09": 29.74487073742008,
        "10": 33.44320194863172,
        "11": 30.8290633146269,
        "12": 41.43298173900113,
    },
}

fileinfo, subfolderinfo = mf.readfiles_meastype(datadir_name)

altitude_min_01 = 0  # minimum altitude range for scattering ratio graphic 01
altitude_max_01 = 10  # maximum altitude range for scattering ratio graphic 01
altitude_min_02 = 10  # minimum altitude range for scattering ratio graphic 02
altitude_max_02 = 30  # maximum altitude range for scattering ratio graphic 02

base_altitude = 13.8  # volcanic base plume altitude (visual)
top_altitude = 30  # volcanic top plume altitude (visual)

# tropopause = 17  # mean CPT altitude over SP from 2013 to 2024

# Comment the following lines out if you do not have tropopause.csv from 08-Tropopause yet.
tropos = pd.read_csv(rootdir_name + "/Tropopause.csv")


for i in range(len(fileinfo)):
    preprocessedsignal = []
    datafiles = []
    filenameheader = []
    preprocessedtime = []

    year = fileinfo[i][-10:-6]
    month = fileinfo[i][-6:-4]
    day = fileinfo[i][-4:-2]
    date = str(year) + "-" + str(month) + "-" + str(day)
    print("Running file ", fileinfo[i][-10:])

    # comment out if you do not have tropopause.csv from 08-Tropopause yet.
    tropopause = tropos.loc[tropos["day"] == date, 'CPT(km)'].iloc[0]

    lraerosol = lraerosol_mean[str(lamb)][month]

    for j in range(len(subfolderinfo)):
        datafiles.append(
            mf.readfiles_generic(os.path.join(fileinfo[i], subfolderinfo[j]))
        )
        if subfolderinfo[j] == files_dir_to_read:

            for filename in datafiles[j]:

                preprocessedsignal.append(
                    pd.read_csv(filename, sep=",", skiprows=range(0, 10))
                )
                filenameheader.append(mf.readdown_header(filename))
                dfdict = pd.DataFrame(filenameheader)

    for k in range(len(preprocessedsignal)):
        alt = pd.DataFrame(list(range(len(preprocessedsignal[k].index)))).mul(
            float(dfdict["vert_res"][k])
        ) + float(dfdict["vert_res"][k])
        alt.columns = ["altitude"]
        preprocessedsignalmean = pd.concat(preprocessedsignal).groupby(level=0).mean()

    """ Calling the glue function to gluing Analogic and Photocounting channels - glue.py file in lidar_retrievals folder"""

    if glueflag == "yes":
        window_length = 150
        correlation_threshold = 0.95
        intercept_threshold = 0.5
        gaussian_threshold = 0.1
        minmax_threshold = 0.5
        min_idx = 200  # 200 * 7.5 = 1500 m ,
        max_idx = 2000  # 2000 * 7.5 = 15000 m

        glued_signal, gluing_central_idx, score, c_lower, c_upper = (
            glue.glue_signals_1d(
                preprocessedsignalmean[str(lamb) + "AN"].to_numpy(),
                preprocessedsignalmean[str(lamb) + "PC"].to_numpy(),
                window_length,
                correlation_threshold,
                intercept_threshold,
                gaussian_threshold,
                minmax_threshold,
                min_idx,
                max_idx,
            )
        )

        """ Calling the Glueing Graphics Plot from LIDAR - GGPLIDAR function - retrieval_plots.py file in lidar_retrievals folder"""

        retrieval_plots.ggplidar(
            preprocessedsignalmean[str(lamb) + "AN"].to_numpy(),
            preprocessedsignalmean[str(lamb) + "PC"].to_numpy(),
            glued_signal,
            alt["altitude"],
            gluing_central_idx,
            window_length,
        )

    """ Calling the lmfit function to calculate atmospheric molecular extinction and backscatter using radiosounding data"""

    if glueflag == "yes":
        channelmode = "GL"
        dfglueing = pd.DataFrame(glued_signal[0], columns=["glued"])
        beta_molecular, simulated_signal = lmfit.lidarmolfit(
            dfdict["station"][0],
            atmospheric_flag,
            filenameheader,
            dfglueing["glued"],
            ini_molref_alt,
            fin_molref_alt,
            lamb,
            channelmode,
            rawinsonde_folder,
        )
        rcs = np.multiply(
            dfglueing["glued"].values.tolist(), np.power(alt["altitude"], 2)
        )

    else:
        beta_molecular, simulated_signal = lmfit.lidarmolfit(
            dfdict["station"][0],
            atmospheric_flag,
            filenameheader,
            preprocessedsignalmean[str(lamb) + channelmode],
            ini_molref_alt,
            fin_molref_alt,
            lamb,
            channelmode,
            rawinsonde_folder,
        )
        rcs = np.multiply(
            preprocessedsignalmean[str(lamb) + channelmode].values.tolist(),
            np.power(alt["altitude"], 2),
        )

    """KLETT-FERNALD-SASANO INVERSION"""

    reference_range = int(
        (
            fin_molref_alt / float(dfdict["vert_res"][0])
            - ini_molref_alt / float(dfdict["vert_res"][0])
        )
        / 2
    )
    index_reference = (
        int(ini_molref_alt / float(dfdict["vert_res"][0])) + reference_range
    )
    beta_aerosol_reference = 0
    bin_length = float(dfdict["vert_res"][0])
    lidar_ratio_molecular = 8.37758041
    # rcs = np.multiply(preprocessedsignalmean[str(lamb)+channelmode].values.tolist(),np.power(alt['altitude'],2))

    aerosol_backscatter = kfs.klett_backscatter_aerosol(
        rcs,
        lraerosol,
        beta_molecular,
        index_reference,
        reference_range,
        beta_aerosol_reference,
        bin_length,
        lidar_ratio_molecular,
    )
    aerosol_backscatter_smooth = savgol_filter(
        aerosol_backscatter.values.tolist(), 15, 3
    )
    aerosol_extinction_smooth = savgol_filter(
        np.multiply(aerosol_backscatter.values.tolist(), lraerosol), 15, 3
    )

    retrieval_plots.kfs_plot(
        lamb,
        dfdict,
        lraerosol,
        alt["altitude"],
        aerosol_backscatter_smooth,
        aerosol_extinction_smooth,
        altitude_min,
        altitude_max,
        optical_prop_scale,
        altitude_scale,
        channelmode,
        save_plots,
        fileinfo[i],
    )

    if glueflag == "yes":
        scattering = retrieval_plots.sr_plot(
            lamb,
            dfdict,
            alt,
            altitude_scale,
            channelmode,
            dfglueing["glued"],
            simulated_signal,
            altitude_min_01,
            altitude_max_01,
            altitude_min_02,
            altitude_max_02,
            base_altitude,
            tropopause,
            top_altitude,
            save_plots,
            fileinfo[i],
        )
    else:
        scattering = np.nan

    # Saving backscatter and extinction profiles as a csv
    mf.folder_creation(fileinfo[i] + "/06-mean_profiles")

    profiles = {
        "_scattering_mean_profile.csv": scattering if glueflag == "yes" else None,
        "_backscattering_mean_profile.csv": aerosol_backscatter_smooth,
        "_extinction_mean_profile.csv": aerosol_extinction_smooth,
    }

    for filename, data in profiles.items():
        if data is not None:
            file_path = (
                fileinfo[i] + "/06-mean_profiles/" + str(fileinfo[i][-10:]) + filename
            )

            if os.path.exists(file_path):
                existing_df = pd.read_csv(file_path)
            else:
                existing_df = pd.DataFrame({"altitude": alt["altitude"] / 1000})

            existing_df[str(lamb)] = data
            existing_df.to_csv(file_path, index=False)
