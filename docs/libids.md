# LIBIDS — Level 0: from Licel acquisition to standardized observations

LIBIDS prepares raw lidar observations for scientific analysis. Its primary scientific role is to **preserve acquisition evidence and measurement identity** while organizing reliable, continuous observation sessions. It does not retrieve aerosol optical coefficients.

## Scientific purpose

A long-term atmospheric lidar record may include different instrument configurations, channel availability, dark-current acquisitions and interruptions. Comparisons between periods require knowing **which laser, detector and calibration applied at the time of measurement**.

LIBIDS reads Licel measurements and associates them with the dated station/instrument information in `station.yaml`. The canonical Level 0 dataset preserves the measured signal and its observational context for subsequent corrections.

## Inputs and preparation

- Licel raw observations in the configured `directories.raw_data` tree.
- The station history and channel/calibration information in `station.yaml`.
- Level 0 acquisition and session settings in `config.yaml`.
- Available dark-current acquisitions and optional surface-weather context.

A persistent, reconstructible header index can accelerate scanning large historical archives. This index is an operational convenience, not a substitute for original Licel measurements or scientific provenance.

## Run Level 0

~~~bash
milgrau-libids
milgrau-libids -i 20250510
milgrau-libids -i spu_20250511-0012Z_20250511-0737Z
~~~

The `-i` selector accepts session IDs, station-local dates or paths. With `processing.incremental: true`, previously validated products can be reused; `--force` explicitly requests regeneration.

## Continuous scientific sessions

An acquisition session is identified as `station_YYYYMMDD-HHMMZ_YYYYMMDD-HHMMZ`, for example:

`spu_20250511-0012Z_20250511-0737Z`

Times in this identifier are UTC. MILGRAU uses an acquisition-continuity tolerance (configured through `level0.session.max_gap_seconds`) and the applicable instrument profile to decide where sessions begin and end. An uninterrupted observation can cross midnight, a calendar month boundary or a six-hour publication interval without splitting the scientific session.

Solar geometry is recorded for individual observations: `solar_elevation_deg(time)`, `solar_regime(time)` and `segment_id(time)`. A session can therefore contain day and night subsequences. A scientific segment (`segXX`) describes a contiguous portion with a consistent solar regime; it is **not** a new acquisition session.

## Acquisition and instrumental context

LIBIDS preserves information needed to understand the raw measurements:

- acquisition start/stop times and channel identity;
- laser shots and acquisition consistency;
- instrument profile and detector calibration context;
- association of dark-current measurements subject to the configured maximum time difference;
- solar elevation and day/night classification;
- available surface-weather context and missing-data status.

The source instrument may not have a characterized parameter for every historical channel. A missing calibration or absent dark-current observation should not be described as a measured value simply because subsequent processing remains operational.

The default surface-weather policy preserves absent values as missing, rather than inventing pressure or temperature.

## Level 0 output

The canonical file is `SESSION_L0.nc`. It contains standardized acquisition fields and metadata, including the measured lidar signal, channels, acquisition times and instrument context.

SCC-compatible derivatives may also be prepared when that **optional** integration is enabled. They are separate interoperability products and do not redefine the canonical Level 0 session or its raw physical measurements.

Each output is stored under the station, start year/month and session directory. Original raw Licel data remain the primary acquisition record.

## Quality checks before Level 1

1. Confirm the expected acquisition interval and channel list.
2. Check whether unexpected gaps split or combine observations.
3. Inspect rejected or quarantined inputs; an absent raw file cannot be silently treated as valid.
4. Check shot-count consistency and instrument/profile changes.
5. Verify dark-current association and any time offset.
6. Confirm solar-regime/segment classification.
7. Inspect source and calibration metadata before comparing different instrument eras.

`milgrau-inspect SESSION_ID` provides a quick overview. Detailed Level 0 results and flags remain in the NetCDF.

## Scientific limitations

- Level 0 does not correct all instrument effects and should not be interpreted as aerosol backscatter or extinction.
- Header indexing is not calibration, uncertainty propagation or scientific data reconstruction.
- Solar day/night labels are geometric classifications based on the configured solar-elevation threshold, not direct measurements of sky conditions.
- Instrument descriptions and historical calibrations remain limited by the evidence in the station catalog.

## Next stage

Continue with [LIPANCORA](lipancora.md), which performs instrumental corrections, background estimation and molecular-atmosphere preparation. The broader scientific division of responsibilities is in [Processing levels](processing_levels.md).
