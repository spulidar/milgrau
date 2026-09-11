# MILGRAU configuration refactor tracker

Branch: `new-architecture`

This file is the source of truth for the strict-configuration refactor. Update it in every implementation batch before moving to the next batch.

## Goal

Make scientific and instrumental decisions explicit and auditable:

- `config.yaml` = processing/scientific recipe.
- `station.yaml` = observational reality, hardware history, station-derived climatologies, calibration and SCC mapping.
- Python code = physical constants, equations, file-format invariants and implementation details.
- Missing required scientific/instrumental configuration must fail early unless an explicit, auditable unavailable/legacy policy is selected; production code must not invent semantic defaults.

## Global rules

- [ ] No silent scientific defaults in production paths.
- [ ] No silent instrumental defaults in production paths.
- [ ] No station-specific coordinates/timezone/IDs embedded in Python fallbacks.
- [x] No silent neutral channel correction fallback for an unknown channel in Level 1 processing; historical missing calibration follows the explicit `level1.missing_channel_calibration` policy and warns when neutral zeros are selected.
- [x] No fake SCC channel IDs; the Level 0 writer rejects a missing SCC channel mapping instead of writing `9999`.
- [ ] No algorithmic fallback that silently widens/changes a configured scientific domain.
- [x] Optional Level 2 cloud-screening behavior must be explicitly enabled/disabled in YAML.
- [x] Rayleigh molecular lidar ratio remains a versioned physical/method constant in Python rather than a user-editable YAML setting.
- [ ] Each produced NetCDF records the resolved station profile/calibration and processing configuration provenance.
- [ ] Tests cover every required-key failure and every explicit disabled/unavailable policy.

## A. Configuration ownership and schema

- [ ] Replace generic `physics` ownership with stage-oriented processing configuration. **Level 1 science controls have moved to `level1`; Level 0 acquisition/weather controls have moved to `level0`; legacy schema still requires `physics.vertical_resolution_m` although the productive Level 0 writer no longer consumes it.**
- [ ] Keep station/site/instrument metadata in `station.yaml` only. **Station-derived LR climatology is now station-owned; runtime compatibility views of site/hardware remain.**
- [ ] Stop rebuilding `physics.channels` from station data. **Productive Level 1 no longer consumes this view; loader/station compatibility still materializes it.**
- [ ] Stop rebuilding `hardware.name_to_id` as a compatibility structure.
- [ ] Remove legacy aliases injected by `normalize_config`.
- [ ] Remove legacy positional channel correction lists. **The Level 1 consumer rejects them, but loader compatibility still exists.**
- [ ] Replace `validate_config_minimum` philosophy with stage-specific strict validation. **Typed strict Level 0 and Level 1 resolvers validate productive LIBIDS/LIPANCORA before discovery/processing; legacy global validation remains.**
- [ ] Validate unknown keys with full paths.
- [ ] Add typed/resolved configuration objects or equivalent strict accessors so scientific modules do not consume raw config mappings directly. **Implemented for current Level 0 acquisition/weather/directory/discovery policy and Level 1 recipe/calibration/atmosphere paths; other stages remain.**

## B. `station.yaml`: observational reality

- [x] Keep station ID/name/institution/timezone/site coordinates in station catalog.
- [x] Keep radiosonde station identity in station catalog and remove the atmosphere-fallback boolean from station metadata.
- [x] Keep station-derived aerosol lidar-ratio climatology and uncertainty in `station.yaml`, with provenance. Repository values were moved without numerical changes.
- [x] Rename `scc_defaults` to explicit `scc_policy` terminology.
- [x] Introduce named instrument calibration sets.
- [x] Make station profiles reference a calibration ID.
- [x] Allow calibration to vary by historical profile structurally; current profiles deliberately reference the one correction set supported by existing source material.
- [x] Move `deadtime_us` to instrument calibration.
- [x] Move `bin_shift_bins` to instrument calibration.
- [x] Move `background_offset` to instrument calibration.
- [x] Add explicit PC saturation characterization/status per relevant channel.
- [ ] Represent legacy ADC/range overrides in station calibration only if real historical Licel files are demonstrated to lack valid header metadata. **No override is currently invented; active analog channels require the header values.**
- [ ] Persist resolved `station_profile_id` and `instrument_calibration_id` in products. Resolution in runtime context is implemented; NetCDF persistence remains.

## C. Level 0 strict configuration

- [x] Require raw/processed/log directories and remove productive path fallbacks. `resolve_level0_config` now resolves them explicitly and `milgrau.io.paths` no longer invents raw/processed/log roots.
- [x] Require raw discovery spurious extensions, ignore dirs and quarantine path. Productive discovery receives a resolved policy; `milgrau.io.filesystem` no longer reads config or supplies discovery defaults.
- [x] Standardize disposable caches under `.cache/`: `.cache/radiosonde`, `.cache/era5`, `.cache/weather`; caches no longer live inside `01-data`.
- [x] Define auditable quarantine layout/manifest. Quarantine remains an explicit action and uses `quarantine/YYYY/MM/DD/<reason>/` with a JSON sidecar containing UTC timestamp, reason/stage, original path, SHA-256, byte size, retained filename and optional measurement ID.
- [x] Require laser-shot tolerance through `level0.acquisition_qa.laser_shot_tolerance_fraction`.
- [x] Make Licel header timestamp jitter threshold explicit through `level0.acquisition_qa.licel_header_time_jitter_s`.
- [x] Require finite dark-current association maximum through `level0.dark_current.max_association_hours`.
- [x] Remove productive timezone fallback to `America/Sao_Paulo`; LIBIDS resolves timezone only from station catalog.
- [x] Remove hardcoded São Paulo latitude/longitude fallbacks from productive weather and Level 0 NetCDF metadata paths.
- [x] Remove fallback surface temperature 25 C / pressure 940 hPa; missing values remain NaN or fail according to explicit policy.
- [x] Use explicit missing-surface-weather policy (`nan` or `fail`) through `level0.surface_weather.missing_policy`.
- [x] Remove `DEFAULT_CHANNEL_ID = 9999`; missing SCC channel ID invalidates the SCC writer path.
- [x] Remove global 7.5 m range-resolution fallback completely. `Raw_Data_Range_Resolution` now requires positive finite native Licel `BinW`; `physics.vertical_resolution_m` cannot mask missing acquisition metadata.
- [x] Remove invented Licel analog ADC/range defaults. Active analog channels require explicit positive `ADCbits` and `Discriminator/DAQ range`; former 12-bit / 0.5-V substitutions are gone.
- [x] Remove hidden Level 0 SCC background-window literals. `Background_Low/High` are written from the explicit configured background interval rather than internal 29000/29999 m constants.

## D. Level 1 strict configuration

- [x] Photon-counting Poisson uncertainty uses observed counts before dark subtraction.
- [x] Canonical atmosphere is materialized in Level 1 with source/fallback provenance.
- [x] Missing channel calibration follows an explicit policy: `error` or `neutral_with_warning`. Repository policy uses exact zero corrections for historical processability and emits a `RuntimeWarning`; PC saturation remains `not_characterized`.
- [ ] Persist the per-channel fact that a neutral historical calibration was assumed. **The resolver exposes `ChannelCalibration.assumed_neutral`; Level 1 NetCDF persistence remains.**
- [x] Remove configurable speed of light; LIPANCORA uses exact SI `299792458 m s-1`.
- [x] Require Level 1 background window through `level1.background`.
- [x] Require dead-time numerical clipping denominator policy through `level1.photon_counting.deadtime_min_denominator`.
- [x] Separate numerical dead-time clipping from detector saturation.
- [x] Require explicit PBL reference channel.
- [x] Require PBL search interval and smoothing settings.
- [x] Remove productive PBL fallback to first available channel.
- [x] Move atmospheric source/fallback policy to `config.yaml` through `level1.atmosphere.source_priority` and explicit outside-coverage policy.
- [x] Make radiosonde temporal-selection policy explicit.
- [x] Remove hardcoded radiosonde station ID fallback.
- [x] Require complete ERA5 configuration when ERA5 is in source policy.
- [x] Do not replace invalid/missing ERA5 pressure-level configuration with an internal list.
- [x] Resolve station altitude historically for atmosphere interpolation.

## E. Level 2 strict configuration

- [x] Require wavelengths; no fallback to `[532]`.
- [x] Require temporal block averaging.
- [x] Require KFS mode explicitly.
- [x] Require Monte Carlo iterations and random seed explicitly.
- [x] Require beta-reference uncertainty and aerosol reference fraction explicitly.
- [x] Require minimum aerosol lidar ratio and negative-aerosol policy explicitly.
- [x] Remove productive aerosol LR 60 sr fallback.
- [x] Remove productive aerosol LR uncertainty 10 sr fallback.
- [ ] Require LR values for every requested wavelength/month and associated uncertainty/provenance. **Values + uncertainty are fail-fast for all 12 months. SPU climatology comes authoritatively from `station.yaml`; explicit config LR remains only compatibility fallback when a station has no climatology. Product-level provenance remains.**
- [x] Remove molecular lidar-ratio YAML knob; keep Rayleigh molecular ratio as algorithm constant.
- [x] Require complete productive gluing configuration.
- [ ] Remove low-level gluing window/search/threshold defaults.
- [ ] Do not widen an invalid configured gluing search interval to whole profile.
- [ ] Make gluing selection-score weights explicit or documented versioned algorithm constants.
- [x] Require complete productive Rayleigh-reference configuration.
- [ ] Do not select last available bin when no valid Rayleigh window exists.
- [x] Require explicit cloud-screening enabled/disabled state.
- [x] Require all cloud-screening parameters when enabled.
- [x] Make cloud baseline percentile explicit when enabled.
- [ ] Integrate cloud contamination into reference-window QA.
- [ ] Remove residual literal fallback access in `_retrieval_impl.py`.
- [ ] Later migration: express public gluing/Rayleigh spatial windows in physical units rather than bins/indices.

## F. IO/runtime/visualization

- [x] Remove filesystem raw/processed/log directory fallbacks from production config paths. Standardized cache locations remain code-level IO conventions, not scientific path fallbacks.
- [ ] Remove logging-level fallbacks when a loaded MILGRAU config is used.
- [ ] Remove visualization output-format/DPI/altitude-range fallbacks.
- [ ] Ensure `mean_profile_smooth_bins` from YAML is actually used everywhere intended.
- [ ] Make quicklook gap threshold explicit; no derived 10 min/3x-median fallback unless deliberately defined as an algorithmic mode.
- [ ] Keep display-only colors/style constants in code unless intentionally exposed as theme configuration.

## G. Scientific failure semantics

- [ ] Missing required config -> configuration error before processing starts for every stage. **Implemented for productive Level 0, Level 1 and Level 2; remaining runtime paths still need audit.**
- [x] Missing Level 1 channel calibration follows only explicit configured policy.
- [x] Unknown PC saturation characterization is explicit `not_characterized`.
- [x] Numerical dead-time clipping is not a detector saturation proxy.
- [x] Level 2 does not accept uncharacterized PC saturation as known-clear input.
- [ ] Invalid Rayleigh search -> reference selection failure.
- [ ] Invalid gluing search -> gluing failure.
- [x] Missing external atmosphere follows only explicitly configured source policy.
- [x] Missing surface weather follows only explicit `nan`/`fail` policy.
- [x] Missing/invalid active Licel `BinW`, analog ADC bits or DAQ range invalidates that input instead of applying acquisition defaults.
- [ ] Optional diagnostic failure must not silently change scientific algorithm.

## H. Provenance / FAIR

- [ ] Persist software version and Git commit in products.
- [ ] Persist scientific algorithm name/version separately from package version.
- [ ] Persist processing-config hash.
- [ ] Persist station-config hash.
- [ ] Persist resolved station profile ID.
- [ ] Persist resolved instrument calibration ID.
- [ ] Persist input file hashes or immutable input manifest.
- [ ] Persist random seed for Monte Carlo retrievals.
- [x] Persist atmosphere source, source datetime/time delta, fallback fraction, source-priority attempts and resolved station geometry.
- [ ] Persist when a neutral legacy Level 1 channel calibration was assumed.
- [ ] Persist LR source/provenance used by each retrieval.
- [x] Persist an audit sidecar for every explicit quarantined input, including SHA-256 and origin/context metadata.

## I. Tests / architecture guardrails

- [ ] Rewrite broad config tests for strict schema; remove tests that freeze legacy aliases.
- [x] Add station calibration/profile resolution tests across historical eras.
- [x] Add explicit-policy tests for missing Level 1 channel calibration.
- [x] Add failure tests for missing LR month/uncertainty.
- [x] Add tests that station LR climatology owns repository values and explicit config LR remains only compatibility fallback.
- [ ] Add failure tests for invalid gluing/Rayleigh domains. **Config-level gluing validation exists; algorithmic failure tests remain.**
- [x] Add failure tests for missing station timezone/coordinates at strict Level 0 accessors.
- [x] Add explicit-policy tests for unavailable saturation characterization.
- [x] Add Level 1 clipping-vs-saturation tests.
- [x] Add Level 2 boundary tests for missing/uncharacterized PC saturation metadata.
- [x] Add atmosphere-policy tests.
- [x] Add Level 0 tests for explicit shot tolerance, timestamp jitter, dark-current association, directories/discovery and weather policy.
- [x] Add strict Licel tests for active `BinW`, analog ADC bit depth and discriminator/DAQ range.
- [x] Add Level 0 writer regression tests proving legacy `vertical_resolution_m` cannot replace missing native `BinW` and hidden background literals are not used.
- [x] Add quarantine tests for dated reason buckets, SHA-256 sidecars, collisions and read-only discovery semantics.
- [ ] Add architectural guard against `config.get(..., semantic_literal_default)` outside config layer.
- [ ] Add regression test ensuring production config contains no undeclared semantic defaults.
- [ ] Run full test suite after each coherent implementation batch. **No CI is currently attached to the branch; isolated strict-L2 tests passed 19/19 earlier. Current full suite remains unverified in this environment.**

## J. Follow-up FAIR/release work

- [ ] Add repository software `LICENSE` after confirming institutional licensing choice.
- [ ] Unify package version and `CITATION.cff` release version.
- [ ] Restore/create documentation referenced by README.
- [ ] Add CI checks and branch protection.
- [ ] Audit NetCDF semantic metadata against current CF conventions.
- [ ] Add archived releases/DOI workflow and changelog.
- [ ] External real-data validation against independent/reference processing chain.

## Suggested improvements recorded but not implemented

- [ ] Consider carrying detector mode explicitly from Level 0 acquisition metadata through full pipeline instead of relying on canonical `.PC` / `.AN` suffixes outside resolved station calibration.
- [ ] Review laser pointing-angle ownership. The Level 0 writer still treats missing `physics.laser_pointing_angle_deg` as vertical (`0.0 deg`); determine whether this is a valid invariant for SPU-Lidar or should move to temporal station/instrument geometry before removing the fallback.

## Implementation log

### 2026-09-09 — initial strict-config tranches

- Tracker initialized from baseline `05f560f37551d8dda87b87d6585b0d92cc42c195`.
- Implemented fail-fast productive Level 2 configuration, explicit cloud policy and strict LR completeness.
- Migrated station channel corrections into named calibration sets with temporal profiles and explicit PC saturation status.
- Migrated productive Level 1 to resolved station calibration, strict background/dead-time/PBL settings and exact SI speed of light.
- Separated numerical dead-time clipping from physical PC saturation and made uncharacterized PC invalid as a scientifically known-clear Level 2 source.

### 2026-09-11 — Level 1 atmosphere and historical calibration policy

- Added explicit Level 1 atmosphere source priority, radiosonde temporal selection, strict ERA5 settings, outside-coverage policy and provenance.
- Removed hardcoded radiosonde ID and resolved historical station altitude.
- Added explicit `neutral_with_warning` historical missing-channel calibration policy using exact zero corrections only; runtime exposes `assumed_neutral` and PC remains saturation `not_characterized`.

### 2026-09-11 — Level 0 strict acquisition/weather tranche

- Added typed strict Level 0 recipe and validation before LIBIDS discovery.
- Moved shot tolerance, Licel header jitter and dark-current association limits to `level0`.
- Removed productive timezone/coordinate/weather standard-value fallbacks and added `nan|fail` surface-weather policy.
- Removed fake SCC ID `9999` and low-level São Paulo/25 C/940 hPa writer fallbacks.

### 2026-09-11 — cache, station LR, and strict Licel metadata tranche

- Commit `95ff59ea6052d428e615273dd4935d531c5cc08c`: moved unchanged SPU monthly lidar-ratio climatology and standard deviations from `config.yaml` into `station.yaml` with provenance. Loader materializes a transitional Level 2 view; explicit config LR is accepted only when station climatology is absent.
- Standardized configured caches to `.cache/radiosonde`, `.cache/era5`, and `.cache/weather`; `.cache/` and `quarantine/` are ignored by Git.
- Commit `823d354afd3a8cf3748b1dc623053d963ad22c57`: removed fake SCC channel ID and remaining coordinate/surface-value literals from Level 0 writer paths.
- Commit `a61540c5878b69a3ee39b1c531fdac1a33b31289`: changed generic IO cache fallbacks from raw-data cache folders to `.cache/weather` and `.cache/radiosonde`.
- Commit `ccd372a344fdd44d8f9b99df448860926f9079fa`: removed hidden Licel analog defaults of 12 ADC bits / 0.5 V and requires explicit positive active-channel `BinW`.

### 2026-09-11 — Level 0 filesystem/range completion tranche

- Commit `97e49bebf054cc967dba6495a44755350a9823b2`: made Level 0 raw/processed/log directories and raw-discovery policy explicit in the typed resolver; generic raw discovery now consumes explicit values and has no config/default layer of its own.
- Commit `2ddc301a7472b3708830ea22465ee1298aca6120`: removed the residual Level 0 writer 7.5-m range-resolution fallback and hidden 29000/29999-m background literals. Native Licel `BinW` is mandatory and SCC background metadata is tied to the explicit configured interval.
- Commit `409f18fd18757c51463f3db3ea69a1c796e71fd1`: added writer regression tests for native range metadata and explicit background ownership.
- Commit `8c48528848ded74b3878e1b0a0421350fd551613`: implemented dated reason-based quarantine buckets and JSON audit sidecars with SHA-256, origin, size, stage and optional measurement ID; quarantine remains manual/explicit.
- Commit `0733ea3db4295b4a1a4194665469890c6d5245b6`: added quarantine layout/manifest tests and preserved read-only raw discovery.
- Level 0 section C is now complete for the tracked strict-config scope. Remaining Level 0-adjacent work belongs to ownership cleanup/provenance/runtime audit (sections A, F, H), not hidden C defaults.
- Full repository suite is not claimed: the branch has no CI/status checks and no complete runnable snapshot is available in this environment.
