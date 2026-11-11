# Level 0 NetCDF — scientific variable and metadata dictionary

This is a description of the **current writer and product contract** for canonical LIBIDS Level 0 NetCDF. It is not a complete NetCDF or Licel standard. Presence rules below distinguish the core product from conditional dark-current and SCC interoperability fields.

The main scientific introduction is [LIBIDS](libids.md); for commands use [Operating MILGRAU](operating_milgrau.md).

## Dimensions and units

| Dimension | Meaning |
| --- | --- |
| `time` | Accepted lidar acquisition profiles |
| `channels` | Standardized spectral/acquisition channels |
| `points` | Range bins retained in the tensor |
| `nb_of_time_scales` | One source acquisition time-scale slot |
| `scan_angles` | Laser-pointing-angle slot |
| `segments` | Contiguous solar day/night segments |
| `weather_time` | Hourly historical surface-weather timestamps |
| `time_bck` | Dark-current acquisitions, present when an associated dark profile is written |
| `yaml_documents` and related provenance axes | May be present to store source recipe content (see scientific provenance policy) |

The writer requires compatible shapes and positive native bin widths for channels. Some dark-current variables are conditional and cannot be assumed present in every observation.

## Core lidar acquisition fields

| Variable | Dimensions | Unit/meaning | Presence |
| --- | --- | --- | --- |
| `Raw_Lidar_Data` | `time,channels,points` | **PC counts** or **analog mV per shot** by channel; see channel metadata | Required |
| `Raw_Data_Start_Time` | `time,nb_of_time_scales` | seconds since the acquisition reference time | Required |
| `Raw_Data_Stop_Time` | `time,nb_of_time_scales` | seconds since the acquisition reference time | Required |
| `channel_string` | `channels` | Normalized channel names such as `532.AN`, `532.PC` | Required |
| `Raw_Data_Range_Resolution` | `channels` | Native Licel BinW [m] | Required |
| `Laser_Shots` | `time,channels` | Parsed accumulated shots by profile/channel | Required |
| `Laser_Pointing_Angle` | `scan_angles` | Angle from zenith [degree] | Required |
| `Laser_Pointing_Angle_of_Profiles` | `time,nb_of_time_scales` | Pointer to angle slot | Required |
| `id_timescale` | `channels` | Acquisition time-scale identifier | Required |
| `Molecular_Calc` | scalar | SCC-compatible molecular calculation selector (written 0 by L0) | Required |
| `Background_Low` | `channels` | Intended background interval lower bound [m] | Written |
| `Background_High` | `channels` | Intended background interval upper bound [m] | Written |
| `Pressure_at_Lidar_Station` | scalar | Representative surface pressure [hPa]; may be missing | Written |
| `Temperature_at_Lidar_Station` | scalar | Representative surface temperature [°C]; may be missing | Written |

**Native signal caveat:** analog acquisition values are calculated from channel-specific ADC/DAQ metadata and NShots, whereas PC values retain counts. Treating every channel as a common raw unit will give scientifically invalid comparisons.

## Acquisition-specific and optional fields

| Variable | Dimensions | Meaning | Presence |
| --- | --- | --- | --- |
| `DAQ_Range` | `channels` | Analog acquisition DAQ scale [mV] | If any analog channel |
| `channel_ID` | `channels` | SCC channel mapping ID | Only if SCC mapping is available |
| `LR_Input` | `channels` | `0` external profile, `1` fixed SCC DB lidar ratio | Only if configured |
| `Background_Profile` | `time_bck,channels,points` | Associated raw dark-current profiles in channel-native units | Only if usable associated dark data |
| `Background_Profile_Available` | `channels` | 0 = unavailable, 1 = available | Written |
| `Raw_Bck_Start_Time` | `time_bck,nb_of_time_scales` | Dark acquisition relative start [s] | With dark profile |
| `Raw_Bck_Stop_Time` | `time_bck,nb_of_time_scales` | Dark acquisition relative stop [s] | With dark profile |
| `Background_Laser_Shots` | `time_bck,channels` | Dark-only shot counts [shots] | When parsed dark-shot metadata are usable |

A channel can have `Background_Profile_Available=0` even when other channels have valid dark profiles; absent samples are `NaN`. A file with no associated dark data can omit `Background_Profile` and its time axis while still recording per-channel unavailability.

## Solar geometry and session segments

| Variable | Dimensions | Unit/meaning |
| --- | --- | --- |
| `solar_elevation_deg` | `time` | Geometric solar-center elevation [degree] at acquisition midpoint |
| `solar_regime` | `time` | `day` or `night` |
| `segment_id` | `time` | Contiguous segment label, e.g. `seg00` |
| `Segment_Label` | `segments` | Each segment identifier |
| `Segment_Regime` | `segments` | `day` or `night` |
| `Segment_Start_Time_UTC` | `segments` | seconds since 1970-01-01 UTC |
| `Segment_End_Time_UTC` | `segments` | seconds since 1970-01-01 UTC |

Attributes `Solar_Day_Night_Threshold_deg`, `Solar_Position_Algorithm` and `Segment_Count` preserve interpretation. Segment labels correspond to contiguous regime intervals; they are not publication window IDs.

## Surface weather (native hourly cadence)

| Variable | Dimensions | Units |
| --- | --- | --- |
| `weather_time` | `weather_time` | seconds since 1970-01-01 UTC |
| `Surface_Temperature_C` | `weather_time` | °C |
| `Surface_Pressure_hPa` | `weather_time` | hPa |
| `Surface_Relative_Humidity_percent` | `weather_time` | % |
| `Surface_Cloud_Cover_percent` | `weather_time` | % |
| `Surface_Wind_Speed_kmh` | `weather_time` | km h⁻¹ |

`Surface_Weather_Source`, `Surface_Weather_Cadence` and `Surface_Weather_Scalar_Method` identify the source and summarization. Under missing-policy `nan`, hourly unavailable measurements remain missing. These surface data do not replace the vertical temperature/pressure profile generated in Level 1.

## Global metadata

The writer/provenance pass records fields such as:

| Attribute | Interpretation |
| --- | --- |
| `Session_ID` | Continuous scientific observation identifier |
| `measurement_start_time`, `measurement_end_time` | Full ISO UTC acquisition interval |
| `session_duration_seconds` | Difference between acquisition bounds |
| `timezone` | Station local-time zone |
| `System`, `Latitude_degrees_north`, `Longitude_degrees_east` | Observatory identity/location |
| `Station_Profile` | Applicable historical instrument profile |
| `RawData_Start_Date`, `RawData_Start_Time_UT`, `RawData_Stop_Time_UT` | Acquisition header time references |
| `Source_File_Count`, `Source_Files` | Original Licel source inventory |
| `Accumulated_Shots` | Representative accumulated shots metadata |
| `Dark_Current_Source_File_Count`, `Dark_Current_Source_Files` | Retained dark-source identity |
| `Dark_Current_Association_Methods`, `Dark_Current_Max_Association_Delta_hours` | Method and temporal gap |
| `SCC_Ready` | SCC mapping availability; not remote processing success |
| `Processing_level`, `Pipeline` | Product/pipeline description |

The general provenance writer also records software identity/source revision where available, configuration and station-file identity or hashes, and embedded configuration documents. Compare actual attributes in the product with the [metadata policy](netcdf_metadata_policy.md), particularly for historical products created under earlier versions.

## Only for optional SCC-specific products

SCC reexports are **separate derived files**. Fields such as `Measurement_ID`, `SCC_Configuration_ID`, `SCC_Configuration_Name`, `LR_File_Name` and SCC timestamp-normalization attributes are produced only in their appropriate SCC context.

The canonical Level 0 must not be relabeled as a remote SCC result. A local SCC `Measurement_ID` does not prove successful remote upload or ELPP/ELDA completion.

## Data integrity contract

The current Level 0 validator requires the main lidar tensor, channel/time-scale arrays, solar metadata and the five surface-weather variables, with appropriate dimensions. Dark-current arrays are checked **if they exist**. Positive finite per-channel range resolution, valid laser shots, analog DAQ metadata and consistent solar segment tables are also required.

The validator tests **structural and declared-metadata validity**, not full physical calibration, detector response, aerosol retrieval quality or the correctness of historical source station records.

## How to use this dictionary

Check `ds.variables`, `ds.dimensions` and `ds.ncattrs()` using NetCDF tools or the read-only Explorer/inspectors, then correlate field values with the known observation period and scientific recipe. Do not automatically replace missing numbers with zero or infer unavailable dark/meteorological fields from another measurement.

**Future worked example:** a named real session with source checksums, output checksum, rejected-file evidence, dark-current association, UTC/local date distinction and full Level 0 field table is still required by the tracker.
