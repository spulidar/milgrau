# SCC raw Level 0 interoperability

MILGRAU treats SCC raw NetCDF as an interoperability input format, not as a substitute for instrument calibration metadata.

## Goal

LIPANCORA should process a structurally compatible SCC raw Level 0 file from another lidar station without station-specific Python code. Instrument identity, temporal configuration, and calibration remain explicit in that station's `station.yaml`.

The intended separation is:

- SCC raw NetCDF owns acquired arrays and acquisition metadata.
- `station.yaml` owns the mapping from SCC channel identifiers to canonical physical channels and the corresponding instrument calibration history.
- `config.yaml` owns the generic processing recipe.
- Python owns schema adaptation, validation, and generic correction equations.

## Supported channel identity

MILGRAU's internal Level 1 code uses canonical channel names such as `532.AN` and `532.PC` because calibration lookup must refer to physical detector channels rather than opaque database identifiers.

LIPANCORA ingestion accepts two identity paths:

1. A Level 0 file already containing `channel_string` with canonical MILGRAU names.
2. An SCC raw file containing numeric `channel_ID`. The IDs are resolved in memory through the temporally valid SCC mapping in `station.yaml` before the strict Level 0 contract and physical corrections are applied.

When both `channel_string` and `channel_ID` are present, and a station catalog is available, LIPANCORA verifies that they agree. An SCC ID is never used to infer wavelength or detector mode without station metadata.

Resolution uses measurement time together with the station timezone and may additionally use `SCC_Configuration_ID`. The MILGRAU filename period is not used to infer SCC day/night identity. Unknown IDs, duplicate IDs, or genuinely ambiguous mappings fail explicitly.

The alternative SCC `channel_string_ID` convention is not currently a productive input identity path because MILGRAU's station catalog intentionally uses numeric SCC channel IDs. It can be added later through an explicit station-owned mapping if a real interoperability case requires it.

## Core structural requirements

After channel identity canonicalization, the file must satisfy the Level 0 acquisition contract required by LIPANCORA. Core quantities include:

- `Raw_Lidar_Data(time, channels, points)`;
- `Laser_Shots(time, channels)`;
- `Raw_Data_Range_Resolution(channels)`;
- `Raw_Data_Start_Time` and `Raw_Data_Stop_Time`;
- `Laser_Pointing_Angle` and `Laser_Pointing_Angle_of_Profiles`;
- `Molecular_Calc` and `id_timescale`;
- a positive `DAQ_Range` for analog channels;
- a resolvable measurement time basis.

Optional dark-current data may be supplied through `Background_Profile` plus SCC background timing metadata. MILGRAU's `Background_Laser_Shots(time_bck, channels)` is an extension that preserves dark-acquisition shot counts when they are available. External SCC files that lack this extension can still use the current productive Level 1 dark-profile path, but they cannot support a separately shot-normalized dark dead-time diagnostic without another traceable source for those shot counts.

A file satisfying an SCC submission schema is therefore not automatically guaranteed to satisfy every MILGRAU scientific prerequisite. LIPANCORA reports missing required acquisition or station-calibration information rather than inventing it.

## MILGRAU `_scc.nc`

LIBIDS writes `*_scc.nc` with the same Level 0 writer used for the full-channel MILGRAU Level 0, restricted to the SCC channel subset and augmented with SCC identifiers. It is therefore a valid explicit LIPANCORA input.

Use an explicit file path when the full Level 0 and SCC subset coexist, for example:

```bash
milgrau-lipancora -i /path/to/20251107_spu_12_L0_scc.nc --force
```

Automatic no-argument discovery deliberately continues to prefer the canonical full-channel Level 0 product. It does not also process the colocated SCC subset, avoiding duplicate Level 1 products from the same acquisition.

For a canonical MILGRAU SCC input, the resulting `*_L1_scc.nc` is written inside the same local-day directory. For an explicitly supplied non-canonical external SCC filename, the Level 1 product is written beside that source file instead of inventing a date hierarchy from the filename.

### Real-data regression evidence

On the historical `20251107sapm` regression case (legacy filename), the MILGRAU SCC export contains the five SCC channels `532.AN`, `532.PC`, `1064.AN`, `355.PC`, and `355.AN`. Running that `*_scc.nc` explicitly through LIPANCORA produced a five-channel Level 1 product whose corresponding scientific arrays and diagnostics were exactly equal to the same five channels extracted from the full-channel Level 1 product. The checked equality included corrected signal and uncertainty, range-corrected signal and uncertainty, PC saturation mask, correction-status diagnostics, dead-time clipping diagnostics, observed PC-rate diagnostics, bin-shift diagnostics, PBL height, altitude/time coordinates, and thermodynamic profiles.

This is end-to-end behavioral/regression evidence for the MILGRAU-generated SCC path. It is not a claim that arbitrary SCC converters or station calibrations are interchangeable without validation.

## Porting MILGRAU to another station

A new station should not require modifications to the generic correction code merely because its SCC channel IDs differ. The station package must instead provide a validated `station.yaml` containing at least:

- site and instrument profile validity periods;
- SCC channel-ID mappings for the applicable configurations;
- canonical channel identities;
- dead-time, bin-shift, background-offset, and saturation-characterization state for each productive channel;
- any station-owned atmospheric/geometry metadata needed by later stages.

This makes SCC raw data portable while keeping instrument-specific scientific assumptions explicit and auditable.

## Current scope

This interoperability layer canonicalizes input identity only. It does not claim that all SCC raw files from all converters are byte-for-byte identical, and it does not import undocumented calibration constants from SCC database state. Broader converter compatibility should be extended from real files with regression fixtures rather than by permissive guessing.
