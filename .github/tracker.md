# MILGRAU tracker — session-refactor

This tracker is the single roadmap for the current MILGRAU architecture on
`session-refactor`. It keeps accepted scientific decisions, implemented work,
remaining refactor tasks, validation gates and release work in one place.

Superseded implementations are recoverable from Git history and are not
alternative productive modes.

## Legend

- [x] implemented / accepted in the current branch
- [~] partially implemented or intentionally transitional
- [ ] still actionable

## Core architecture

The project now separates three concepts:

1. **Scientific session** — one continuous, instrumentally homogeneous lidar
   acquisition. It may cross midnight and civil 6-hour boundaries.
2. **Scientific regime/segment** — internal homogeneous intervals, primarily
   solar `day` / `night`, plus any relevant instrumental/context changes.
3. **Publication window** — website-only local civil windows
   `00–06`, `06–12`, `12–18`, `18–24`.

The governing rule is:

> MILGRAU organizes science by continuous sessions and scientific regimes.
> `spulidar/measurements` organizes publication by local 6-hour windows.

`session_id != publication_window_id`.

## Current status summary

- **Phase 1 — session identity/filesystem:** essentially complete.
- **Phase 2 — continuous sessionization:** implemented; 30-minute continuity
  threshold accepted for the current SPU workflow, with broader real-data
  validation still useful.
- **Phase 3 — solar regime/segments:** implemented end to end; empirical
  validation of the current -3 degree threshold against SPU background/SNR
  remains a scientific follow-up.
- **Phase 4 — continuous Level 0:** hourly surface weather plus per-profile
  solar elevation/regime/segments implemented; SCC derivatives are written per
  solar segment when configured.
- **Phase 5 — continuous Level 1:** hourly ERA5/USSA76 atmosphere, radiosonde
  QA, inherited solar/segment context and full Level 1 figure ownership are
  implemented.
- **Phase 6 — continuous Level 2:** block-time molecular atmosphere and
  segment-homogeneous blocking are implemented in schema 8; terminology cleanup
  and figure reorganization remain.
- **Phase 7 — retire LIRACOS as an independent pipeline:** complete; Level 1
  figure ownership belongs to LIPANCORA and reusable renderers remain in
  `viz/quicklooks.py`.
- **Phase 8 — `spulidar/measurements` publication refactor:** not started here.
- **Phase 9 — Explorer / inspect:** partially session-aware.
- **Phase 10 — unified `figures/` convention:** canonical Level 1 and Level 2
  figures use the shared session `figures/` directory and semantic filenames;
  only optional retrieval-support figure scope remains open.
- **Phase 11 — documentation / final cleanup:** partial.

---

# Phase 1 — Session identity and filesystem

## Implemented

- [x] Scientific identity no longer uses `00/06/12/18`.
- [x] `session_id` is the canonical identity.
- [x] Canonical format:
  `spu_YYYYMMDD-HHMMZ_YYYYMMDD-HHMMZ`.
- [x] Example:
  `spu_20250511-0012Z_20250511-0737Z`.
- [x] Station comes first.
- [x] Session timestamps in the ID are UTC.
- [x] Filename precision is minutes; exact seconds remain product metadata.
- [x] Session parser validates ordered start/end timestamps.
- [x] Canonical station casing and `Z` handling are normalized.
- [x] Old `LOCAL_PERIOD_STARTS`, `measurement_id_parts()`,
  `build_measurement_id()` and `measurement_day_dir()` were removed from
  the session identity path.
- [x] Old `milgrau/level0/time.py` period logic was removed.
- [x] Logging and primary IO APIs use `session_id`.
- [x] Session product tree is:

```text
02-processed_data/
└── spu/
    └── YYYY/
        └── MM/
            └── spu_STARTZ_ENDZ/
                ├── spu_STARTZ_ENDZ_L0.nc
                ├── spu_STARTZ_ENDZ_L1.nc
                ├── spu_STARTZ_ENDZ_L2.nc
                └── figures/
```

- [x] Product filenames remain self-identifying when copied outside the session
  directory.
- [x] SCC-derived products preserve distinct lineage:
  `_L0_scc.nc`, `_L1_scc.nc`, `_L2_scc.nc`.
- [x] Named Level 2 variants are supported without changing session identity.
- [x] CLI selectors accept session IDs, station-local dates and explicit paths.
- [x] Date selection no longer expands into four 6-hour scientific IDs.
- [x] Date selection resolves sessions intersecting the selected local civil day.
- [x] Cross-midnight sessions are discoverable from relevant local dates.
- [x] Session paths/identity have dedicated tests.
- [x] README, processing-level docs and code inventory describe session identity.

## Remaining

- [~] Decide the exceptional policy for a scientifically valid session whose
  rounded start and end collapse into the same minute; the builder currently
  rejects it.
- [~] Define a deterministic collision policy if two distinct sessions ever
  receive the same minute-resolution canonical ID.
- [ ] Run a final repository-wide audit for obsolete scientific references to
  `measurement_id`, `_00`, `_06`, `_12`, `_18` after all later phases
  are complete.

---

# Phase 2 — Continuous sessionization

## Implemented

- [x] `build_session_inventory()` is the Level 0 inventory entry point.
- [x] Raw Licel files are ordered chronologically.
- [x] Start/stop times are normalized to UTC.
- [x] Missing/invalid stop times can be reconstructed from valid duration
  metadata when possible.
- [x] Session continuity is based on previous stop versus next start.
- [x] `level0.session.max_gap_seconds` is explicit configuration.
- [x] Current SPU continuity tolerance is **1800 s / 30 minutes**.
- [x] Gaps within 30 minutes keep one session.
- [x] Gaps greater than 30 minutes start a new session.
- [x] Crossing midnight does not split a session.
- [x] Crossing former 6-hour boundaries does not split a session.
- [x] Session inventory stores:
  `session_id`, `session_start_utc`, `session_end_utc`.
- [x] Inventory records `session_boundary_reason`.
- [x] Supported current reasons include `acquisition_start`, `time_gap` and
  `station_context_change`.
- [x] Each raw measurement file is annotated with station profile and
  calibration identity.
- [x] Profile/calibration changes split otherwise continuous acquisitions.
- [x] One Licel file crossing a profile/calibration boundary is rejected rather
  than silently assigned a heterogeneous context.
- [x] Acquisition QA does not redefine the canonical session interval.
- [x] Tests cover midnight, former 6-hour boundaries, <=30-minute continuity,
  >30-minute split and station-context changes.

## Still evaluate

- [ ] Survey a broader real SPU campaign to confirm that 30 minutes remains a
  robust operational threshold across historical acquisition patterns.
- [ ] Keep a simple gap histogram/report available for future threshold review.
- [ ] Decide whether channel-set/DAQ-structure changes must force a session
  boundary when they are not already represented by `Station_Profile`.
- [ ] Decide whether pointing/geometry changes must force a session boundary.
- [ ] Identify any hardware state not encoded in station profile/calibration
  that would make one session scientifically heterogeneous.

---

# Dark-current association

- [x] Dark current is associated with scientific sessions, not 6-hour windows.
- [x] Temporal distance is measured against the complete session interval.
- [x] A dark acquisition inside a session interval has zero temporal distance.
- [x] Outside dark acquisitions are associated to the nearest session when
  within `max_association_hours`.
- [x] Darks beyond the configured maximum remain unassociated.
- [x] Provenance records source files, method and association time delta.
- [x] Tests cover successful and rejected associations.

---

# Phase 3 — Solar regime and scientific segments

## Implemented

- [x] Compute `solar_elevation_deg(time)` from station coordinates and profile
  midpoint times using geometric solar-center elevation.
- [x] Compute `solar_regime(time)`.
- [x] Productive regime classes are only `day` and `night`.
- [x] Do not create `twilight` as a primary regime.
- [x] Keep the day/night threshold configurable.
- [x] Current configured threshold is **-3 degrees solar elevation**.
- [x] The threshold is resolved from configuration rather than hard-coded in
  scientific logic.
- [x] Preserve continuous solar elevation so classification can be revised later
  without re-deriving geometry.
- [x] Record the solar-position algorithm and threshold in product provenance.
- [x] Add per-profile `segment_id(time)`.
- [x] Start a new segment on every `day <-> night` transition.
- [x] Persist a compact segment table with label, regime, UTC start and UTC end.
- [x] Preserve day/night/day as distinct `seg00`, `seg01`, `seg02` rather
  than reusing a regime name as segment identity.
- [x] Propagate solar elevation/regime/segment identity from L0 to L1 without
  recomputing solar geometry.
- [x] Keep Level 2 clock buckets anchored to the configured wall-clock cadence
  while splitting any bucket that crosses a segment boundary.
- [x] Store `segment_id(block_time)`, `solar_regime(block_time)` and mean
  `solar_elevation_deg(block_time)` in Level 2.
- [x] Add LEBEAR selectors `--regime day|night` and `--segment segXX`.
- [x] Keep `--time-window-utc` orthogonal and allow intersection with a solar
  selector.
- [x] `--regime night` may retain multiple disjoint night segments; `--segment`
  always selects one exact contiguous segment.
- [x] Solar selectors preserve the canonical source `session_id`; derived L2
  filenames use a variant tag rather than inventing a new session identity.

## SCC solar integration

- [x] Remove productive fixed-clock 06–18 / 18–06 SCC mode selection.
- [x] `resolve_station_context` accepts an explicit resolved `day`/`night`
  mode instead of deriving SCC mode from civil clock time.
- [x] Explicit SCC raw ingestion no longer infers day/night from station-local
  clock time; it uses `Solar_Regime` / `solar_regime` or
  `SCC_Configuration_ID`, and rejects ambiguous mappings rather than guessing.
- [x] Keep the canonical full-channel Level 0 continuous across solar
  transitions.
- [x] Write one SCC Level 0 derivative per contiguous solar segment when its
  historical station configuration is exportable.
- [x] Name SCC derivatives `SESSION_segXX_L0_scc.nc`.
- [x] SCC derivatives preserve the original `Session_ID` and add
  `Segment_ID` / `Solar_Regime`.
- [x] Incremental Level 0 checks understand multiple segment SCC derivatives.
- [x] Automatic LIPANCORA discovery continues to use only the canonical
  `SESSION_L0.nc`; SCC segment products remain explicit interoperability
  derivatives.

## Scientific validation still open

- [ ] Validate the current -3 degree threshold empirically against SPU
  background/SNR transitions, especially at 355 nm.
- [ ] Document the validation sample/campaign and retain the evidence used to
  keep or revise the threshold.
- [ ] If future instrument/context states are allowed to change inside one
  session without forcing a new session, decide whether they also create
  segment boundaries. Current station-profile/calibration changes already force
  a new session.

---

# Phase 4 — Continuous Level 0 / LIBIDS

## Implemented

- [x] LIBIDS groups and processes by `session_id`.
- [x] Incremental processing is session-based.
- [x] Outputs are written inside the session directory.
- [x] Level 0 metadata includes:
  `Session_ID`, `measurement_start_time`, `measurement_end_time`,
  `session_duration_seconds`, timezone and raw source provenance.
- [x] Scientific `period=00-06` style metadata is no longer required.
- [x] Raw start/stop metadata remains available.
- [x] Station profile/calibration provenance remains available.
- [x] Dark-current provenance remains available.
- [x] SCC is an optional derivative and does not define scientific identity.

## Surface weather — implemented temporal contract

- [x] Add a native-cadence/hourly `weather_time` axis bracketing the complete
  session.
- [x] Persist time-resolved surface temperature, pressure, relative humidity,
  cloud cover and wind.
- [x] Keep Open-Meteo Archive as the current source until a better local SPU
  source is configured.
- [x] Preserve source/provenance and native/source time resolution.
- [x] Do not interpolate/duplicate hourly source values onto every lidar profile.
- [x] Missing surface weather remains missing; do not invent temperature or
  pressure defaults.
- [ ] If a reliable local weather station becomes available, prefer:
  local station -> Open-Meteo fallback, without changing the scientific schema.
- [x] Derive scalar temperature/pressure only where SCC interoperability
  explicitly requires scalar fields.
- [x] SCC scalar values use the finite median of the weather interval written
  for that SCC derivative; solar-segment SCC files therefore use segment
  weather rather than the complete-session median. These remain interoperability
  fields, not the scientific weather series.

## Solar/session context in Level 0

- [x] Add per-profile solar elevation/regime variables.
- [x] Add segment identity and segment-table metadata.
- [x] Select historical SCC day/night configuration from solar regime.
- [x] Export SCC derivatives per solar segment without splitting the canonical
  Level 0 session.

---

# Phase 5 — Continuous Level 1 / LIPANCORA

## Implemented base

- [x] LIPANCORA discovers session-based Level 0 products.
- [x] Level 1 output remains inside the session directory.
- [x] Logging and selectors use `session_id`.
- [x] Local-date selection works with sessions.
- [x] Long continuous sessions can reach Level 1 without artificial 6-hour
  splitting.
- [x] Current Level 1 still materializes the canonical atmosphere so Level 2
  performs no external atmosphere IO.

## Time-resolved atmosphere — implemented temporal contract

- [x] Add hourly `atmosphere_time` coordinates bracketing the complete session.
- [x] Materialize
  `Atmospheric_Temperature_K(atmosphere_time, altitude)`.
- [x] Materialize
  `Atmospheric_Pressure_hPa(atmosphere_time, altitude)`.
- [x] Use **ERA5 pressure-level reanalysis as the temporal backbone** for the
  canonical Level 1 atmosphere across long continuous sessions.
- [x] Preserve hourly ERA5 source/time/coverage provenance and station spatial
  metadata.
- [x] Do not abruptly replace individual ERA5 hours with radiosonde profiles.
- [x] Keep radiosondes as the local in-situ observational reference for
  comparison/QA when a suitable sounding exists.
- [x] Keep USSA76 as explicit vertical extension and full hourly fallback when
  required.
- [x] Keep solar day/night segmentation independent from atmospheric cadence.
- [x] Reuse the existing hour-indexed ERA5 cache so adjacent/repeated sessions
  do not redownload identical analysis hours.
- [x] Update Level 1 contracts/tests for time-resolved atmospheric dimensions.

## Level 2 consumption of atmosphere

- [x] Level 2 performs no ERA5/radiosonde/Open-Meteo IO.
- [x] Keep 20-minute block membership clock-anchored but define `block_time`
  as the mean timestamp of profiles actually contributing to the block.
- [x] Interpolate Level 1 temperature linearly and pressure in `log(P)` to
  each representative `block_time`.
- [x] Materialize block-resolved molecular backscatter/extinction and lidar
  ratio fields in Level 2 schema 8, including blocks without a valid retrieval.
- [x] Record source/provenance sufficiently to reproduce the molecular
  atmosphere used by each block.
- [ ] Run the complete repository test suite/CI against the time-resolved
  atmosphere contract when an execution environment is available.

## Atmospheric comparison QA / scientific figure

Create:

`SESSION_L1_AtmosphericProfile.webp`

- [x] Compare ERA5, radiosonde and the canonical profile actually used by
  MILGRAU on a common altitude grid when radiosonde is available.
- [x] Plot temperature and pressure profiles.
- [x] Plot corresponding differences.
- [x] Report temperature bias and RMSE over configurable altitude bands.
- [x] Report pressure bias/relative difference and valid vertical overlap.
- [x] Compare a retrieval-relevant derived molecular quantity such as molecular
  number density and/or molecular backscatter, so QA reflects retrieval impact
  rather than meteorological differences alone.
- [x] Record radiosonde launch/target time, ERA5 analysis time, spatial metadata
  and time offsets in figure/provenance.
- [x] Treat ERA5-versus-radiosonde as consistency/validation QA, not completely
  independent validation, because radiosonde observations may contribute to
  reanalysis assimilation.
- [x] If no suitable radiosonde exists, still generate the atmospheric figure
  with ERA5 + canonical used profile + USSA76/fallback context and explicitly
  mark radiosonde unavailable.
- [x] Define and test a maximum radiosonde time separation for QA comparison,
  independent of the production ERA5 cadence.

## Solar/segments in Level 1

- [x] Propagate `solar_elevation_deg` from Level 0.
- [x] Propagate `solar_regime` from Level 0.
- [x] Propagate `segment_id` and the segment table from Level 0.
- [x] Keep Level 1 canonical as the full session; regime/segment selection is
  applied downstream by LEBEAR rather than creating competing canonical L1
  files.

## Level 1 figures

Already prepared:

- [x] Level 1 visual output uses `figures/`.
- [x] Figure filenames include session and processing level.
- [x] RCS naming follows the new session convention.
- [x] Mean RCS naming follows the new session convention.
- [x] Default plots use the observed continuous session extent.
- [x] Former 6-hour x-axis forcing is gone.
- [x] Real temporal gaps remain visible.
- [x] Explicit UTC plotting windows remain supported.

Implemented:

- [x] LIPANCORA owns automatic RCS quicklooks, MeanRCS and AtmosphericProfile
  generation after successful Level 1 writing/validation.
- [x] Level 1 figure generation is incremental.
- [x] Figure-generation failure does not invalidate a scientifically valid L1.
- [x] Add the atmospheric profile/comparison figure above.

---

# Phase 6 — Continuous Level 2 / LEBEAR

## Accepted productive scientific policy

- [x] Ground-based vertical geometry.
- [x] Productive elastic wavelengths are 355 and 532 nm when valid L1 inputs
  exist.
- [x] Temporal block average is 20 minutes by default.
- [x] Two-sided Klett–Fernald–Sasano is the sole productive elastic inversion.
- [x] Primary molecular-reference search range is 10–15 km.
- [x] Explicit fallback reference search range is 5–20 km.
- [x] Candidate ranking uses the accepted Rayleigh/path criteria without
  rewarding altitude itself.
- [x] Rayleigh diagnostic window is 1 km.
- [x] Progressive vertical grid uses native resolution below 6 km, then the
  configured 15/30/60/100 m schedule aloft.
- [x] Nominal residual aerosol fraction at the reference is `f = 0`.
- [x] Routine Monte Carlo uses 150 configurable realizations.
- [x] Every Monte Carlo realization perturbs native signal, refits residual
  background, rebuilds the progressive grid, reruns reference selection and
  reruns the inversion.
- [x] Lidar ratio is drawn from station climatology/uncertainty and bounded by
  the configured positive minimum.
- [x] `mc_valid_fraction` and MC selection success are diagnostics, not hard
  scientific acceptance thresholds.
- [x] Missing support is not silently bridged or filled.

## Residual-background policy

- [x] L1 records acquisition/background diagnostics and corrected signal.
- [x] L2 fits a residual constant background jointly with molecular scaling over
  the configured high-altitude span using robust weighted regression.
- [x] In range-corrected space the fitted term is `B z^2`.
- [x] Nominal fitted background, formal standard error and
  calibration/background correlation are stored per block.
- [x] Monte Carlo refits background for every perturbed realization.
- [x] Fixed post-hoc background subtraction is not a productive alternative.

## Session integration already implemented

- [x] LEBEAR uses `session_id`.
- [x] Session-based Level 1 discovery is implemented.
- [x] L2 remains beside L0/L1 inside the session directory.
- [x] Exact source L1 SHA-256 provenance is preserved.
- [x] Explicit `--time-window-utc` remains supported.
- [x] Time-window variants do not change session identity.
- [x] 20-minute blocks are independent of former 6-hour publication windows.
- [x] Block membership remains wall-clock anchored while `block_time` is the
  mean observed profile time used for time-dependent atmosphere/LR evaluation.
- [x] Level 2 schema 7 stores molecular backscatter/extinction and lidar-ratio
  assumptions by `block_time`.

## Session/regime integration

- [x] Add `--regime day`.
- [x] Add `--regime night`.
- [x] Add `--segment segXX`.
- [x] Allow solar selector + `--time-window-utc` intersection.
- [x] Propagate solar regime, elevation and segment metadata into Level 2.
- [x] Prevent one Level 2 block from crossing a scientific segment boundary.
- [x] Level 2 schema 8 records solar-segment-homogeneous blocks.
- [ ] Rename ambiguous `period_*` variables/labels such as
  `period_support_fraction`.
- [ ] Prefer `session_*` or `temporal_*` names according to actual semantics.
- [ ] Remove “period mean” language where it could be confused with the old
  website periods.

## Level 2 figures

- [x] LEBEAR owns Level 2 figure orchestration in `level2/figures.py`.
- [x] Canonical Level 2 renderers live in `viz/level2.py`.
- [x] Obsolete generic `viz/level2_qa.py` and its QA-specific tests are removed.
- [x] Write Level 2 visual products into the session `figures/` directory.
- [x] Do not call every visual product “QA”.
- [x] Preserve semantic distinctions between scientific profile and diagnostic
  content through figure names/titles rather than one generic QA namespace.
- [x] Current semantic filenames include:
  `SESSION_L2_MolecularReference_355nm.webp`,
  `SESSION_L2_Gluing_355nm.webp`,
  `SESSION_L2_OpticalProfiles_355nm.webp`,
  and optional `SESSION_L2_MCReference_355nm.webp`.
- [~] Decide later whether a dedicated
  `SESSION_L2_RetrievalSupport_355nm.webp` adds value beyond the compact
  endpoint/support diagnostics already carried by OpticalProfiles and the L2
  NetCDF. Do not add a redundant figure solely to match an old list.
- [x] Retrieval-support shading remains omitted from the current compact optical
  profile view; dashed branch endpoints remain the primary compact coverage
  indicator.

---

# Phase 7 — Retire LIRACOS as an independent pipeline

## Preparation already done

- [x] LIRACOS understands session IDs.
- [x] LIRACOS no longer forces 6-hour plotting windows.
- [x] LIRACOS plots the observed session extent.
- [x] Level 1 visual outputs already target `figures/`.
- [x] Session-based figure naming already exists.

## Completed

- [x] Move Level 1 figure orchestration into LIPANCORA.
- [x] Reusable RCS/MeanRCS renderers are independent in `viz/quicklooks.py`.
- [x] LIPANCORA owns Level 1 figures and LEBEAR owns Level 2 figures.
- [x] Remove `milgrau-liracos` as a productive CLI and package entry point.
- [x] Remove `milgrau/viz/liracos.py` as an independent orchestrator.
- [x] Migrate useful LIRACOS behavior tests to `level1.figures`.
- [x] Remove LIRACOS from primary-CLI/orchestration tests.
- [x] Update README, processing-level docs and code inventory accordingly.

Target pipeline:

```text
LIBIDS     -> L0
LIPANCORA  -> L1 + figures
LEBEAR     -> L2 + figures
```

Figure failures must remain non-fatal to valid scientific NetCDF products.

---

# Phase 8 — spulidar/measurements and website publication

This work belongs primarily in `spulidar/measurements`, not the MILGRAU
scientific identity layer.

- [ ] Make `measurements` the sole owner of local publication windows:
  `00–06`, `06–12`, `12–18`, `18–24`.
- [ ] Treat those IDs explicitly as `publication_window_id`, not session IDs.
- [ ] Read MILGRAU L1/L2 directly.
- [ ] Find every scientific session intersecting each publication window.
- [ ] Select only the relevant time span for each website plot.
- [ ] Allow several sessions to contribute to one publication visualization.
- [ ] Concatenate only for visualization and preserve real gaps as missing data.
- [ ] Never scientifically merge distinct sessions.
- [ ] Reuse MILGRAU generic renderers instead of duplicating plotting/scientific
  logic.
- [ ] Keep current public/R2 naming where practical to avoid unnecessary
  historical migration.
- [ ] Keep publication state compatible with historical data.
- [ ] Separate public staging from scientific products; preferred target:
  `03-site-products/`.
- [ ] Stop recursively treating arbitrary MILGRAU scientific WEBPs as website
  publication assets.

Canonical example:

- scientific acquisition local time:
  `10/05 21:12 -> 11/05 04:37`
- one MILGRAU session:
  `spu_20250511-0012Z_20250511-0737Z`
- website publication views:
  `20250510_spu_18` uses 21:12–24:00 local,
  `20250511_spu_00` uses 00:00–04:37 local.

No scientific NetCDF is split because of website layout.

---

# Phase 9 — Explorer and inspect

## Explorer already implemented

- [x] Discovers session directories.
- [x] Finds L0/L1/L2 products for one session.
- [x] Displays the UTC session interval.
- [x] Lists available processing levels.
- [x] Does not depend on old four-period IDs.

## Explorer remaining

- [ ] Use local-time human presentation as the primary display:
  `SPU · 10/05 21:12 -> 11/05 04:37 · 7h25`.
- [ ] Show duration explicitly.
- [ ] Show `highest_available_level`.
- [ ] Show solar regime(s).
- [ ] Show segments.
- [ ] Add day/night filtering.
- [ ] Surface files in the session `figures/` directory.

## Inspect already implemented

- [x] Understands session-based L0/L1/L2 products.
- [x] Displays Session_ID.
- [x] Displays start/end and duration metadata.
- [x] Resolves date/session/path selectors.
- [x] A date can find all available processing levels for intersecting sessions.

## Inspect remaining

- [ ] Add concise local-time/human session presentation.
- [ ] Summarize solar regime and segments.
- [ ] Summarize available figures.
- [ ] Summarize explicit highest available level.

---

# Phase 10 — Unified figures convention

Accepted filesystem:

```text
spu_20250511-0012Z_20250511-0737Z/
├── spu_20250511-0012Z_20250511-0737Z_L0.nc
├── spu_20250511-0012Z_20250511-0737Z_L1.nc
├── spu_20250511-0012Z_20250511-0737Z_L2.nc
└── figures/
    ├── spu_..._L1_RCS_355AN_15km.webp
    ├── spu_..._L1_RCS_532AN_15km.webp
    ├── spu_..._L1_MeanRCS.webp
    ├── spu_..._L1_AtmosphericProfile.webp
    ├── spu_..._L2_MolecularReference_355nm.webp
    ├── spu_..._L2_Gluing_355nm.webp
    ├── spu_..._L2_OpticalProfiles_355nm.webp
    └── spu_..._L2_RetrievalSupport_355nm.webp
```

- [x] One `figures/` directory per session.
- [x] No L1/L2 subdirectories are needed.
- [x] Level 1 figure filenames carry `L1`.
- [x] RCS and MeanRCS follow session-aware naming.
- [x] Add AtmosphericProfile.
- [x] Level 2 outputs use the same `figures/` directory.
- [x] Level 2 files use semantic scientific/diagnostic names.
- [x] Remove generic `QA_` naming where the figure is not specifically QA.
- [x] Canonical renderers obey:

  `SESSIONID_LEVEL_FIGURE[_CHANNEL|WAVELENGTH][_RANGE].ext`
- [~] Dedicated RetrievalSupport figure remains optional pending demonstrated
  value; retrieval-support data themselves remain in the Level 2 product.

---

# Phase 11 — Documentation and final cleanup

Already updated:

- [x] README describes continuous sessions and canonical session IDs.
- [x] README documents session directory layout.
- [x] README explains that 6-hour windows are publication views, not scientific
  measurements.
- [x] Processing-level docs define the continuous session model.
- [x] Code inventory assigns session grouping to Level 0 inventory.
- [x] SCC interoperability docs/code acknowledge continuous sessions.

Still required after the corresponding code lands:

- [x] Document solar regime and configured threshold.
- [x] Document segment semantics.
- [x] Document time-resolved surface weather.
- [x] Document ERA5-hourly Level 1 atmosphere, radiosonde QA role and USSA76
  extension/fallback.
- [x] Replace generic QA terminology with figures terminology where
  appropriate while retaining scientifically specific QA terms.
- [x] Remove LIRACOS from productive architecture docs.
- [ ] Document `measurements` as owner of publication windows.
- [ ] Document public staging.
- [x] Update CLI examples with `--regime` / `--segment`.
- [ ] Clean obsolete `period_*` language.
- [ ] Run final repository-wide legacy-reference audit.
- [ ] Update changelog before the next release candidate.

---

# Level 2 scientific validation gates

These gates remain valid and are independent from the session refactor. Completed
items are retained here as accepted evidence; unfinished ideas are not dropped.

## Minimum scientific gate

- [x] Molecular-only truth: near-zero aerosol recovery and connected two-sided
  coverage without interpolation.
- [x] Noise sweep: bias, interval behavior, reference-selection stability and
  branch endpoints as SNR decreases.
- [x] Reference placement: controlled primary/fallback search-range cases.
- [x] Lidar-ratio mismatch: quantified backscatter/extinction response with
  conditional interpretation.
- [x] Residual aerosol at the reference: sensitivity cases retained without
  converting them into a probability prior.
- [x] Progressive-grid representation: native-versus-progressive comparison
  against common truth.
- [ ] Background truth: quantify fitted-background bias, interval coverage and
  stability for complete and incomplete temporal blocks.

## Retrieval-support engineering still required

- [ ] Add explicit block fields for backward valid, forward valid, valid through
  20 km, valid through 25 km and full connected-column status.
- [ ] Define stable reason codes for upper/lower branch termination.

## Real-data release examples

- [ ] Freeze one clean March 2024 case using the exact current recipe.
- [ ] Freeze one plume/cirrus or difficult-background case.
- [ ] Freeze one weak-signal or incomplete-block case.
- [ ] For each frozen case retain configuration, Level 2 product, figures and a
  compact table of reference, fallback, fitted background, branch endpoints and
  Monte Carlo support.

## Follow-up validation

- [ ] Sensitivity of the 1 km Rayleigh window against 0.5 and 1.5 km.
- [ ] Broader heterogeneous SPU campaign survey.
- [ ] Characterize photon-counting saturation and replace provisional guards
  with instrument evidence.
- [ ] Characterize overlap and define the validated near-field limit.
- [ ] Quantify fitted gluing slope/intercept covariance materiality.
- [ ] Compare against Raman products under matched time/altitude support.
- [ ] Compare with SCC/ELDA under matched inputs and assumptions.
- [ ] Compare column-integrated products with independent AOD where appropriate.

---

# Release engineering

- [ ] Add CI smoke test: build wheel, install it and run all productive CLI
  `--help` commands.
- [ ] Decide whether releases are source-checkout only or packaged defaults and
  assets must make the wheel independently runnable.
- [ ] Protect the release branch with the existing CI workflow as a required
  status check.
- [ ] Triage remaining test warnings; scientific/numerical warnings must be
  resolved or explicitly justified.
- [ ] Add a changelog entry covering session architecture, two-sided retrieval,
  background fit, reference policy and plotting/pipeline cleanup.

## Release language

Until minimum scientific validation and frozen real-data examples are complete,
describe the Level 2 product as an experimental two-sided elastic retrieval.
After those gates, claims may state retrieval to the highest continuously
supported altitude for each block. Do not claim universal validity to 20 or
25 km.

---

# Repository-history cleanup

Source cleanup and Git-history cleanup remain separate tasks.

- [ ] Revisit history rewriting only after the architecture/release state is
  stable.
- [ ] Before any history rewrite, create a backup tag and coordinate the forced
  update/reclone with collaborators.
- [ ] Historical quicklooks, NetCDFs and logs remain the main known contributors
  to clone size until that deliberate cleanup.

---

# Recommended execution order from current branch state

1. **Validate the implemented time-resolved meteorology/atmosphere**
   - run the full test suite/CI when available;
   - exercise a representative real SPU session with ERA5 available;
   - exercise an intentional ERA5-missing/USSA76 fallback case;
   - inspect the atmospheric comparison figure and block-resolved molecular state.

2. **Validate solar threshold on representative SPU data**
   - compare geometric elevation/background/SNR transition;
   - retain -3 degrees or revise the configured threshold from evidence.

3. **Refactor `spulidar/measurements`**
   - read L1/L2;
   - own 6-hour publication windows;
   - preserve gaps;
   - reuse renderers.

4. **Explorer / inspect completion**
   - human local-time display;
   - highest level;
   - regimes/segments;
   - figures.

5. **Scientific/release gates**
   - complete remaining L2 validation;
   - freeze real-data examples;
   - release engineering and documentation cleanup.
