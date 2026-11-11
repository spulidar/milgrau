# LIBIDS — Level 0 acquisition, organization and scientific traceability

LIBIDS reads original **Licel lidar acquisitions**, applies acquisition-level QA and constructs standardized **continuous scientific sessions**. Its objective is to document what the instrument measured, when and under which configuration. LIBIDS does not perform the elastic aerosol inversion.

The operational overview is in [Operating MILGRAU](operating_milgrau.md); a field-by-field account of the generated NetCDF is in the [Level 0 data dictionary](level0_schema.md).

## 1. Inputs and original acquisition evidence

LIBIDS uses the raw data tree in `directories.raw_data`, the dated instrument descriptions in `station.yaml`, and explicit policies in `config.yaml`. Each accepted Licel file supplies an acquisition interval, its channel headers and binary measurement samples.

The Licel header identifies acquisition start/stop, nominal laser information and the number of datasets. The channel header is parsed using the fields:

`Active AnalogPhoton LaserUsed DataPoints 1 HV BinW Wavelength d1 d2 d3 d4 ADCbits NShots Discriminator ID`.

Important scientific fields:

| Header field | Interpretation |
| --- | --- |
| `Active` | Channel is active and its binary payload is expected |
| `AnalogPhoton` | Detector/acquisition mode |
| `DataPoints` | Number of altitude/range samples |
| `BinW` | Native bin range resolution, in meters |
| `Wavelength` | Spectral channel identifier |
| `ADCbits`, `Discriminator` | Required acquisition information for active analog channels |
| `NShots` | Number of accumulated laser shots **per channel**, preferred over any legacy fallback |
| `ID` | Instrument/channel identifier when provided |

The parser does not invent analog ADC bit depth, discriminator/DAQ range or an active channel's bin width when absent. It also rejects duplicate active normalized channel names.

### Analog and photon-counting measurements

Channel names follow the normalized pattern `<wavelength>.AN` (analog) or `<wavelength>.PC` (photon-counting), for example `532.AN` and `532.PC`. The wavelength is obtained from the header.

**Photon-counting raw data** retain the parsed count values. **Analog raw data** are expressed in millivolts per shot, using the ADC scale from that channel's header:

[
V_{mathrm{AN}}=rac{mathrm{ADC counts}}{N_{mathrm{shots}}}
rac{mathrm{DAQ range (mV)}}{2^{mathrm{ADCbits}}-1}.
]

Thus the `Raw_Lidar_Data` array mixes **native-channel physical units** by acquisition mode; there is not one universal unit for all channels. Subsequent instrumental corrections are the responsibility of Level 1.

Raw files and their source names remain essential scientific evidence. The header index, when available, speeds discovery but does not replace the raw binary record.

## 2. Discovery, ignored files and QA

~~~bash
milgrau-libids
milgrau-libids -i 20250510
milgrau-libids -i spu_20250511-0012Z_20250511-0737Z
~~~

The `-i` selector accepts an existing file/directory, canonical session ID or station-local `YYYYMMDD` date. The raw scan is **read-only**. Entries with configured non-Licel extensions and ignored directories are excluded from candidate processing; malformed/unreadable headers are logged and skipped.

A path containing `dark` (case-insensitive) is classified as a dark-current acquisition in the current discovery implementation. Consequently, operators should confirm their directory naming convention: this is an operational classification heuristic, not evidence that a dark measurement was physically performed.

**No raw data are automatically deleted or moved to quarantine.** Quarantine is an explicit, separately audited operation that retains original identity, reason and file hash.

### Acquisition QA

LIBIDS tests measurements and dark currents separately, preserving the distinction between the two acquisition classes. In a homogeneous session/group it derives modal nominal laser shots and laser frequency and evaluates:

1. Missing, zero or invalid laser-shot counts.
2. Shot-count deviation from the modal nominal count.
3. Missing/nonpositive or inconsistent laser frequency.
4. Missing/nonpositive header duration, or incompatibility with the nominal physical duration and configured header-time jitter.

The distributed settings are:

~~~yaml
level0:
  acquisition_qa:
    laser_shot_tolerance_fraction: 0.002
    licel_header_time_jitter_s: 1.0
~~~

The shot-count criterion uses a 0.2% fractional tolerance. Rejection reasons are logged for affected original paths; an invalid group is not treated as a valid observation with invented shots or durations. A high proportion of rejected files produces a warning.

An important distinction: **the acquisition QA supports optional SCC timestamp normalization in SCC-compatible derivatives**; it does not justify silently rewriting the raw scientific acquisition timeline.

## 3. Continuous sessions and local dates

LIBIDS sorts valid acquisition headers chronologically. A **new session** begins at the first measurement, when the gap after the latest preceding measurement exceeds `level0.session.max_gap_seconds`, or when the dated station profile/calibration context changes.

The distributed session-gap threshold is **1800 seconds (30 minutes)**. A session remains continuous across midnight or month boundaries when those conditions remain satisfied. Fixed 6-hour publication windows do **not** create new scientific sessions.

The canonical ID is:

`spu_YYYYMMDD-HHMMZ_YYYYMMDD-HHMMZ`.

The ID reflects the UTC start and end at minute precision. Exact header intervals are retained as measurement metadata. The station timezone is used for selection by calendar date, not to change the UTC session identity.

## 4. Historical instrument identification

For each acquired measurement, LIBIDS resolves the applicable station profile and its calibration identity from `station.yaml`. Historical profiles have declared validity intervals, channel mappings and instrument properties. A profile or calibration change creates an explicit scientific session boundary.

The dated station catalog is the authoritative record for station coordinates, pointing angle, channel calibration and related physical metadata. Do not infer that a channel present at one date has identical detector properties or overlap at another date.

Available SCC-specific channel/product mappings remain in separate `scc.yaml`. They do not replace historical physical calibration data in `station.yaml`.

## 5. Dark-current association and limits

Dark-current data are identified separately from ordinary measurements. Each dark-current acquisition is associated with the **nearest continuous measurement session** based on its time distance from the session's start/end interval. A dark acquired during the session has zero separation.

The distributed limit is:

~~~yaml
level0:
  dark_current:
    max_association_hours: 12.0
~~~

If the nearest session lies farther away than the configured threshold, the dark current remains **unassociated**. The association method, maximum accepted temporal separation and source file names are preserved as metadata.

When suitable dark data are available, Level 0 can save `Background_Profile(time_bck,channels,points)`, `Background_Laser_Shots` and separate background acquisition timestamps. It also saves `Background_Profile_Available(channels)`.

Channels with no usable dark-current samples remain missing (`NaN`) and have availability 0. The dark-current shots, when recorded, belong to the **dark acquisition** and must not be replaced with the shots of the lidar measurement.

**Scientific limitation:** nearest-in-time association is a reproducible selection rule; it is not a guarantee that a dark profile perfectly represents detector temperature or electronic conditions during the main measurement.

## 6. Solar position and contiguous segments

LIBIDS computes geometric solar-center elevation from station coordinates and each acquisition's **midpoint**. The classification threshold in the distributed recipe is `-3.0°` (no atmospheric-refraction correction).

Every measurement has a `solar_elevation_deg`, `solar_regime` (`day` or `night`) and `segment_id` (`seg00`, `seg01`, ...). Each time the day/night classification changes, a new **contiguous scientific segment** starts.

One continuous acquisition can therefore include several segments or even more than one night segment. Segments do not change the canonical session ID. A compact `Segment_Label`, `Segment_Regime`, start/end table is saved in the NetCDF for downstream selection and interpretation.

Solar classification is geometric metadata, **not a direct cloud/sky-brightness measurement**.

## 7. Surface meteorology

LIBIDS requests hourly historical surface conditions from the **Open-Meteo Archive API** at the configured lidar station coordinates. It covers the whole session from the starting UTC hour to the next bounding hour after the end.

The separate `weather_time` coordinate contains the data's native hourly cadence. Output fields include surface temperature, pressure, relative humidity, total cloud cover and wind speed. They are not silently resampled to every lidar profile.

The default `level0.surface_weather.missing_policy: nan` preserves missing hourly values as `NaN` and issues warnings; `fail` rejects a run with unavailable required hours. A representative scalar temperature and pressure may be included for Level 0/SCC compatibility, but these are summaries and **not substitutes for the Level 1 altitude-resolved molecular atmosphere**.

## 8. Output and provenance

The canonical output is `SESSION_L0.nc`, stored under the UTC session start year/month. Its core tensor is `Raw_Lidar_Data(time, channels, points)`. Other dimensions record acquisition times, channels, segments, weather times and, when present, dark-current profiles.

Important metadata include station identity, dated profile/calibration, source Licel filenames, session start/end, total shots, surface-weather provider and cadence, dark-current association and solar threshold/algorithm. Provenance also records software/configuration identity when available.

Use the [Level 0 data dictionary](level0_schema.md) for exact field names, units, dimensions, presence rules and SCC-only fields.

## 9. How to evaluate a Level 0 file

Before passing a session to LIPANCORA:

1. Compare the NetCDF start/end with the intended Licel files.
2. Confirm channel wavelengths, detector modes, samples and shot counts.
3. Inspect rejected-file warnings, acquisition gaps and unexpected session boundaries.
4. Confirm the historical station profile and calibration identity.
5. Verify whether valid dark-current data were associated, and at what temporal distance.
6. Inspect per-profile day/night labels and any segment transitions.
7. Examine hourly surface-weather availability; `NaN` is unknown, **not zero**.
8. Confirm that SCC derivatives, if used, are stored separately from canonical Level 0.

The Level 0 file is an acquisition/standardization product. It does not yet imply a valid corrected signal or an aerosol optical retrieval.

## 10. Reproducible scientific examples — still pending

A worked real observation should identify its exact Licel filenames and hashes; acquisition/calibration dates; gaps; QA acceptance/rejection; associated dark; actual solar transition; weather completeness; L0 NetCDF checksum; and comparison to a validated earlier baseline.

That evidence is **not included in this guide yet**. The scientific tracker retains separate unchecked tasks for real-data annotated examples and historical Level 0 equivalence/performance verification.

Continue with [LIPANCORA](lipancora.md) once acquisition QA and source identity are acceptable.
