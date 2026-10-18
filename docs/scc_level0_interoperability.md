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

Phase 12.3 stages upload packages under `.cache/scc/pending/` rather
than the scientific session directories. The SCC service client, durable
submission state, remote download and cleanup will follow in phases 12.4–12.6.

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

## MILGRAU SCC temporary export (Phase 12.3)

The canonical continuous `SESSION_L0.nc` is the scientific Level 0 product.
SCC upload NetCDFs are **disposable submission derivatives**, only generated
when `config.yaml` sets `scc.enabled: true`. Each SCC packet is partitioned by
solar regime **and UTC month**, but the scientific session/segments stay intact.

Example (one scientific session with distinct SCC publication pieces):

```text
02-processed_data/spu/2025/09/SESSION/SESSION_L0.nc

.cache/scc/pending/
  20250930spu2300/
    20250930spu2300.nc
    lr_20250930spu23.nc
    manifest.json
  20251001spu0010/
    20251001spu0010.nc
    lr_20251001spu00.nc
    manifest.json
```

`<Measurement_ID>.nc` includes the SCC global `Measurement_ID`, station/channel
identifiers, the original `Session_ID`, solar regime, source segment IDs,
and the applicable reference to `LR_File_Name`. The JSON manifest records
scientific session, SCC configuration/product IDs, original segment identities,
UTC dates, source canonical L0 path, the `scc.yaml` SHA-256 and checksums
of upload bytes. The manifest is **local staging state**, not proof the remote
SCC received anything.

SCC sounding integration is **opt-in** through
`scc.yaml scc_policy.sounding.enabled` and only for new exports.
It uses an existing observed Wyoming radiosonde cache for station 83779,
with the sounding launch-site metadata, not the lidar station location.
If available and validated, the packet also contains
`rs_YYYYMMDDcccNN.nc`, referenced by `Sounding_File_Name` with
`Molecular_Calc=1`. When disabled/missing under `missing_policy: skip`,
it retains `Molecular_Calc=0` and no sounding file.
No ERA5/USSA76 profile is labelled as observed sounding.
SCC ancillary LR names instead use the deployed Quick Upload
`lr_YYYYMMDDcccNN.nc` form; archived four-digit legacy LR names
must be repaired offline before any upload.

Packages are written in sibling temporary directories and moved atomically
to `pending/<Measurement_ID>/` only after the raw/ancillary NetCDFs and manifest
exist. Duplicate `Measurement_ID` values are rejected; existing pending
packages are **never silently overwritten**. If a raw measurement profile
*itself* straddles a UTC month boundary, staging fails explicitly rather than
guessing which monthly LR applies. Distinct profiles across the month boundary
are partitioned into month-specific SCC submissions.

The scientific L0 incremental validity check does **not** depend on SCC
staged files. Deleting `.cache/scc/` never forces canonical L0 reprocessing,
and normal incremental runs do not recreate deleted SCC packages for a
current scientific session. To explicitly regenerate SCC export packages
from archived raw data **without rewriting canonical L0**, use:

```bash
# Requires config.yaml scc.enabled: true, archived raw files and a valid L0.
milgrau-libids -i spu_20250930-2300Z_20251001-0300Z --scc-reexport
```

An existing pending package with the same Measurement_ID must be reconciled
or removed manually before a re-export. CLI `--force` instead explicitly
reprocesses the canonical scientific L0 and cannot be combined with
`--scc-reexport`.

Older `*_day_L0_scc.nc` / `*_night_L0_scc.nc` files already saved inside
`02-processed_data` remain untouched as historical evidence. The new LIBIDS
does **not** generate, migrate or delete them. Do not delete them as part of
this refactor without an independently verified retention/migration decision.

A staged packet is **not yet an uploaded packet**. Automatic upload, durable
submission state and safe cache cleanup are future phases. At present,
`scc.upload.automatic` does not initiate network communication.

### SCC upload measurement identity

SCC submission files contain the mandatory global attribute `Measurement_ID`.
This is **not** the MILGRAU scientific `Session_ID`. MILGRAU derives the
15-character SCC ID as `YYYYMMDDsssHHMM`, using the first **measurement**
profile of that SCC solar-regime derivative (UTC), followed by the three-character
alphanumeric SCC station code (currently `spu`) and UTC hour/minute. For example,
a regime starting at 2024-09-13 15:49:32Z is `20240913spu1549`.

The canonical `SESSION_L0.nc` does not acquire `Measurement_ID`.
The SCC raw file is named `<Measurement_ID>.nc` inside a directory named by its
`Measurement_ID`. The manifest binds that directory to the scientific
session and packet contents. Missing or deleted cache files never invalidate
scientific L0.

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
the SCC-only export partitions the **profile rows** by month and solar regime.
If an individual raw profile crosses a UTC calendar month, it fails with an
explicit diagnostic; a profile cannot silently receive the wrong month of
climatology. Every finalized cached packet has SHA-256 validated upload bytes;
a missing cache does not make scientific L0 stale.

Upload the raw SCC NetCDF and its matching `lr_<Measurement_ID>.nc` together
using **Quick upload**, or submit the LR file separately using
**Upload ancillary**. The SCC will not start processing until the referenced
ancillary exists. Validation of the new upload path against the live SCC portal
is still required.

Official specification:
https://docs.scc.imaa.cnr.it/en/latest/_downloads/f3353a4bea47d541e7cb97788490cac6/NetCDF_input_file_v3_6.pdf
(Table 4).

Automatic LIPANCORA discovery processes the canonical scientific
`SESSION_L0.nc` only. Cached SCC `<Measurement_ID>.nc` files are not scientific inputs.
Explicit third-party NetCDF processing remains a separate LIPANCORA
interoperability workflow and must be validated with the appropriate
station instrument metadata.

### Real-data regression evidence

On the historical `20251107sapm` regression case (legacy filename), the MILGRAU SCC export contains the five SCC channels `532.AN`, `532.PC`, `1064.AN`, `355.PC`, and `355.AN`. Running that `*_scc.nc` explicitly through LIPANCORA produced a five-channel Level 1 product whose corresponding scientific arrays and diagnostics were exactly equal to the same five channels extracted from the full-channel Level 1 product. The checked equality included corrected signal and uncertainty, range-corrected signal and uncertainty, PC saturation mask, correction-status diagnostics, dead-time clipping diagnostics, observed PC-rate diagnostics, bin-shift diagnostics, PBL height, altitude/time coordinates, and thermodynamic profiles.

This is end-to-end behavioral/regression evidence for the MILGRAU-generated SCC path. It is not a claim that arbitrary SCC converters or station calibrations are interchangeable without validation.

## Porting MILGRAU to another station

A new station should not require modifications to the generic correction code merely because its SCC channel IDs differ. The station package must instead provide a validated `station.yaml` containing at least:

- site and instrument profile validity periods;
- canonical physical channel names, with SCC channel-ID mappings separately in `scc.yaml`;
- canonical channel identities;
- dead-time, bin-shift, background-offset, and saturation-characterization state for each productive channel;
- any station-owned atmospheric/geometry metadata needed by later stages.

This makes SCC raw data portable while keeping instrument-specific scientific assumptions explicit and auditable.

## Current scope

This interoperability layer canonicalizes input identity only. It does not claim that all SCC raw files from all converters are byte-for-byte identical, and it does not import undocumented calibration constants from SCC database state. Broader converter compatibility should be extended from real files with regression fixtures rather than by permissive guessing.
