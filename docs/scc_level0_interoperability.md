# SCC raw Level 0 interoperability

MILGRAU treats SCC raw NetCDF as an interoperability input format, not as a substitute for instrument calibration metadata.

## Goal

LIPANCORA should process a structurally compatible SCC raw Level 0 file from another lidar station without station-specific Python code. Instrument identity and calibration remain in `station.yaml`; SCC database mapping is optional and belongs to `scc.yaml`.

The intended separation is:

- SCC raw NetCDF owns acquired arrays and acquisition metadata.
- `station.yaml` owns physical channel identity, calibration, station-profile history and observational climatology.
- `scc.yaml` owns SCC channel/configuration IDs and optical product identities.
- `config.yaml` owns the generic processing recipe.
- Python owns schema adaptation, validation, and generic correction equations.

## Phase 12.1 — SCC is opt-in and separately configured

`config.yaml` sets `scc.enabled: false` by default. In that state,
MILGRAU does not load or validate `scc.yaml` and LIBIDS writes only
the canonical scientific Level 0. SCC derivatives are not part of
scientific incremental-current decisions.

With `scc.enabled: true`, `scc.config_file` is required and resolved
relative to the processing configuration. The separate SCC catalog's
station/profile IDs must match those in `station.yaml`; SCC channel and
configuration IDs are not scientific identifiers.

Phase 12.1 preserves SCC derivative placement as an interim step.
Phase 12.3 will move staging to `.cache/scc/` and provide independent
publication-state tracking.

## Supported channel identity

MILGRAU's internal Level 1 code uses canonical channel names such as `532.AN` and `532.PC` because calibration lookup must refer to physical detector channels rather than opaque database identifiers.

LIPANCORA ingestion accepts two identity paths:

1. A Level 0 file already containing `channel_string` with canonical MILGRAU names.
2. An SCC raw file containing numeric `channel_ID`. The IDs are resolved in memory through the temporally valid SCC mapping in `scc.yaml` before the strict Level 0 contract and physical corrections are applied.

When both `channel_string` and `channel_ID` are present, and a station catalog is available, LIPANCORA verifies that they agree. An SCC ID is never used to infer wavelength or detector mode without station metadata.

Resolution uses acquisition time together with the station timezone and may additionally use `SCC_Configuration_ID`. Session filenames do not encode SCC day/night identity. Unknown IDs, duplicate IDs, or genuinely ambiguous mappings fail explicitly.

The alternative SCC `channel_string_ID` convention is not currently a productive input identity path because MILGRAU's optional SCC catalog uses numeric SCC channel IDs. It can be added later through an explicit station-owned mapping if a real interoperability case requires it.

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

MILGRAU's primary Level 0 remains the complete continuous scientific session.
LIBIDS writes at most one SCC interoperability derivative for each solar regime
present in the session:

```text
spu_20251107-0900Z_20251107-1500Z_L0.nc
spu_20251107-0900Z_20251107-1500Z_day_L0_scc.nc
spu_20251107-0900Z_20251107-1500Z_night_L0_scc.nc
```

Contiguous `segXX` identities remain scientific metadata. If a long session is
`day -> night -> day`, both day segments are written into the same
`SESSION_day_L0_scc.nc`; their original timestamps and `segment_id(time)`
values are preserved, so the intervening gap is not filled or presented as
continuous acquisition. The derivative records `Solar_Regime`,
`Source_Segments`, `Source_Segment_Count`, and `Contains_Time_Gaps`.

The day/night SCC configuration is selected from `scc.yaml` using the
stored solar regime, not a fixed 06/18 clock rule.

### SCC upload measurement identity

SCC submission files contain the mandatory global attribute `Measurement_ID`.
This is **not** the MILGRAU scientific `Session_ID`. MILGRAU derives the
15-character SCC ID as `YYYYMMDDsssHHMM`, using the first **measurement**
profile of that SCC solar-regime derivative (UTC), followed by the three-character
alphanumeric SCC station code (currently `spu`) and UTC hour/minute. For example,
a regime starting at 2024-09-13 15:49:32Z is `20240913spu1549`.

The canonical `SESSION_L0.nc` does not acquire `Measurement_ID`. SCC derivatives
continue to use MILGRAU's descriptive `SESSION_day_L0_scc.nc` and
`SESSION_night_L0_scc.nc` filenames: current SCC versions do not require a
specific raw-data filename, though `<Measurement_ID>.nc` is recommended.
When incremental processing encounters an existing SCC derivative without the
correct `Measurement_ID`, that derivative is considered stale and regenerated.

For other stations, the SCC station code must be exactly three alphanumeric
characters; an incompatible station identifier must be mapped explicitly, not
truncated or fabricated. SCC database uniqueness also requires that an ID has
not already been used by another uploaded measurement.

### Optional station-climatology ancillary for SCC elastic products

SCC raw exports can use a **monthly, station-owned lidar-ratio profile**
rather than the fixed value stored in the SCC database, without modifying
scientific MILGRAU L0/L1/L2 files. In Phase 12.2 the default **inside optional
`scc.yaml`** is `scc_policy.lr_input.source: climatology_ancillary`. Normal
MILGRAU `config.yaml` still has `scc.enabled: false`; enabling SCC does **not**
send files over the network (the native client is a separate later phase).

The SCC product catalogue contains **30 distinct database product IDs**
transcribed from the six SCC product-list screenshots (systems 248, 294, 484,
565, 1046 and 1047). A product entry contains its type
(`elastic_backscatter`, `raman_backscatter`, `extinction_only` or
`lidar_ratio_extinction`), emitted wavelength, SCC `usecase`, number of
channels, and optional original parent product ID. Each historical SCC system
references these entries in its own `product_ids` list, so product IDs may
legitimately belong to more than one system. No channel or configuration ID
is ever substituted for a missing product ID.

Example using **verified current SCC IDs**:

```yaml
products:
  3683: {type: elastic_backscatter, wavelength_nm: 1064, usecase: 0, channel_count: 1}
  3684: {type: elastic_backscatter, wavelength_nm: 532, usecase: 1, channel_count: 2}
  3685: {type: elastic_backscatter, wavelength_nm: 355, usecase: 1, channel_count: 2}
profiles:
  spu-merionc-2024:
    day:
      configuration_id: 1047
      name: "Sao Paolo Lidar - Daytime-10September2024"
      channels: {"532.AN": 4069, "532.PC": 4070, "1064.AN": 4071, "355.PC": 4072, "355.AN": 4073}
      product_ids: [3683, 3684, 3685]
```

The export resolver derives `elastic_product_ids` **in memory** from typed
product entries, never as a second manually maintained YAML list. Products
with Raman companions retain the Raman processing path and are not assigned
an elastic climatological LR. For the current daytime system 1047 this selects
355, 532 and 1064 nm; for nighttime 1046 this selects **only 3683** at
1064 nm. The historical systems 248 and 294 genuinely list **both 746 and
3683** as 1064-nm elastic-backscatter products with usecase 0; the resolver
keeps both distinct IDs (rather than arbitrarily preferring one). Exact
historical SCC processing behavior still requires end-to-end validation with
the remote server.

The exporter selects the UTC calendar month of the actual SCC measurement
profiles, reads `station.lidar_ratio_climatology.monthly_sr` and `std_sr`,
and writes `lr_<Measurement_ID>.nc` beside the SCC-only raw derivative.
The ancillary follows **Table 4** of SCC's NetCDF input-file v3.6:
`Altitude(points)` in meters above the lidar station,
`Lidar_Ratio(products,points)`, optional
`Lidar_Ratio_Error(products,points)`, `product_ID(products)`, and
the global `Lidar_Station_Name`. A constant-with-altitude profile is
written per optical product, with the climatological spread reported as
`Lidar_Ratio_Error`, not as an empirical per-profile retrieval uncertainty.
A complete ascending altitude grid spans the native lidar range.

For elastic products only, the raw SCC file uses `LR_Input=0` and
`LR_File_Name="lr_<Measurement_ID>.nc"`. Wavelengths with Raman companions
remain controlled by the Raman retrieval: their `LR_Input` is left undefined.
If no elastic-only product exists, no ancillary is written. Missing real
product IDs cause a clear failure; there is **no** fallback to fabricated
channel IDs or silent use of fixed SCC_DB values.

A single constant climatology profile cannot represent two UTC calendar months:
the SCC-only export intentionally fails with an actionable diagnostic when a
day/night derivative straddles a month boundary; the scientific continuous
session is unaffected. SCC submissions for that rare case must be split into
month-specific upload measurements separately. Incremental SCC validation checks
both the referenced ancillary file and its expected product IDs, month and
climatology values, so stale or missing ancillary files invalidate the SCC export.

Upload the raw SCC NetCDF and its matching `lr_<Measurement_ID>.nc` together
using **Quick upload**, or submit the LR file separately using
**Upload ancillary**. The SCC will not start processing until the referenced
ancillary exists. Validation of the new upload path against the live SCC portal
is still required.

Official specification:
https://docs.scc.imaa.cnr.it/en/latest/_downloads/f3353a4bea47d541e7cb97788490cac6/NetCDF_input_file_v3_6.pdf
(Table 4).

Automatic LIPANCORA discovery deliberately processes only the canonical
`SESSION_L0.nc`. SCC derivatives remain interoperability products and do not
create competing canonical Level 1 products. They may still be supplied
explicitly when an SCC-specific regression/interoperability check is required:

```bash
milgrau-lipancora -i /path/to/spu_20251107-0900Z_20251107-1500Z_night_L0_scc.nc --force
```

For an explicitly supplied canonical SCC derivative, the resulting
`SESSION_night_L1_scc.nc` or `SESSION_day_L1_scc.nc` is written in the same
session directory. For an
external SCC filename, Level 1 is written beside the source file instead of
inventing a MILGRAU session hierarchy.

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
