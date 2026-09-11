# MILGRAU configuration refactor tracker

Branch: `new-architecture`

This file is the source of truth for the strict-configuration refactor. Update it in every implementation batch before moving to the next batch.

## Goal

Make scientific and instrumental decisions explicit and auditable:

- `config.yaml` = processing/scientific recipe.
- `station.yaml` = observational reality, hardware history, calibration and SCC mapping.
- Python code = physical constants, equations, file-format invariants and implementation details.
- Missing required scientific/instrumental configuration must fail early unless an explicit, auditable unavailable/legacy policy is selected; production code must not invent semantic defaults.

## Global rules

- [ ] No silent scientific defaults in production paths.
- [ ] No silent instrumental defaults in production paths.
- [ ] No station-specific coordinates/timezone/IDs embedded in Python fallbacks.
- [x] No silent neutral channel correction fallback for an unknown channel in Level 1 processing; historical missing calibration follows the explicit `level1.missing_channel_calibration` policy and warns when neutral zeros are selected.
- [ ] No fake SCC channel IDs.
- [ ] No algorithmic fallback that silently widens/changes a configured scientific domain.
- [x] Optional Level 2 cloud-screening behavior must be explicitly enabled/disabled in YAML.
- [x] Rayleigh molecular lidar ratio remains a versioned physical/method constant in Python rather than a user-editable YAML setting.
- [ ] Each produced NetCDF records the resolved station profile/calibration and processing configuration provenance.
- [ ] Tests cover every required-key failure and every explicit disabled/unavailable policy.

## A. Configuration ownership and schema

- [ ] Replace generic `physics` ownership with stage-oriented processing configuration. **Level 1 science controls have moved to `level1`; Level 0 acquisition/weather controls have moved to `level0`; the transitional range-resolution field remains in `physics`.**
- [ ] Keep station/site/instrument metadata in `station.yaml` only.
- [ ] Stop rebuilding `physics.channels` from station data. **Productive Level 1 no longer consumes this view; loader/station compatibility still materializes it.**
- [ ] Stop rebuilding `hardware.name_to_id` as a compatibility structure.
- [ ] Remove legacy aliases injected by `normalize_config`.
- [ ] Remove legacy positional channel correction lists. **The Level 1 consumer now rejects them, but loader compatibility still exists.**
- [ ] Replace `validate_config_minimum` philosophy with stage-specific strict validation. **Typed strict Level 0 and Level 1 resolvers validate productive LIBIDS/LIPANCORA before discovery/processing; legacy global validation remains.**
- [ ] Validate unknown keys with full paths.
- [ ] Add typed/resolved configuration objects or equivalent strict accessors so scientific modules do not consume raw config mappings directly. **Implemented for current Level 0 acquisition/weather policy and Level 1 recipe/calibration/atmosphere paths; other stages remain.**

## B. `station.yaml`: observational reality

- [x] Keep station ID/name/institution/timezone/site coordinates in station catalog.
- [x] Keep radiosonde station identity in station catalog and remove the atmosphere-fallback boolean from station metadata.
- [x] Rename `scc_defaults` to explicit `scc_policy` terminology.
- [x] Introduce named instrument calibration sets.
- [x] Make station profiles reference a calibration ID.
- [x] Allow calibration to vary by historical profile structurally; current profiles deliberately reference the one correction set supported by existing source material.
- [x] Move `deadtime_us` to instrument calibration.
- [x] Move `bin_shift_bins` to instrument calibration.
- [x] Move `background_offset` to instrument calibration.
- [x] Add explicit PC saturation characterization/status per relevant channel.
- [ ] Represent legacy ADC/range overrides in station calibration only when raw Licel metadata cannot provide them.
- [ ] Persist resolved `station_profile_id` and `instrument_calibration_id` in products. Resolution in runtime context is implemented; NetCDF persistence remains.

## C. Level 0 strict configuration

- [ ] Require raw/processed/log directories; remove path fallbacks.
- [ ] Require raw discovery spurious extensions / ignore dirs / quarantine path when used.
- [x] Require laser-shot tolerance through `level0.acquisition_qa.laser_shot_tolerance_fraction`.
- [x] Make Licel header timestamp jitter threshold explicit through `level0.acquisition_qa.licel_header_time_jitter_s`.
- [x] Require finite dark-current association maximum through `level0.dark_current.max_association_hours`.
- [x] Remove productive timezone fallback to `America/Sao_Paulo`; LIBIDS resolves timezone only from the station catalog.
- [ ] Remove hardcoded São Paulo latitude/longitude fallback. **Productive weather retrieval now resolves coordinates only from `station.yaml`; low-level NetCDF-writer literals still need removal.**
- [ ] Remove fallback surface temperature 25 C / pressure 940 hPa. **Productive weather handling now returns `NaN` or fails according to policy; low-level NetCDF-writer literals still need removal.**
- [x] Use explicit missing-surface-weather policy (`nan` or `fail`) through `level0.surface_weather.missing_policy`.
- [ ] Remove `DEFAULT_CHANNEL_ID = 9999`; missing SCC channel ID makes SCC export unavailable/invalid.
- [ ] Remove global 7.5 m range-resolution fallback; prefer Licel metadata and explicit historical station override only when justified.
- [ ] Review Licel parser defaults for ADC bits/range and convert them to explicit format invariants or station-profile overrides.

## D. Level 1 strict configuration

- [x] Photon-counting Poisson uncertainty uses observed counts before dark subtraction (SCI-003 completed before this refactor).
- [x] Canonical atmosphere is materialized in Level 1 with source/fallback provenance.
- [x] Missing channel calibration follows an explicit policy: `error` or `neutral_with_warning`. Repository policy uses exact zero corrections for historical processability and emits a `RuntimeWarning`; PC saturation remains `not_characterized`.
- [ ] Persist the per-channel fact that a neutral historical calibration was assumed. **The resolver exposes `ChannelCalibration.assumed_neutral`; Level 1 NetCDF persistence remains.**
- [x] Remove configurable speed of light; LIPANCORA uses the exact SI value `299792458 m s-1` as a code constant.
- [x] Require Level 1 background window through `level1.background`.
- [x] Require dead-time numerical clipping denominator policy through `level1.photon_counting.deadtime_min_denominator`.
- [x] Separate numerical dead-time clipping from detector saturation; clipping never creates a physical saturation flag.
- [x] Require explicit PBL reference channel.
- [x] Require PBL search interval and smoothing settings.
- [x] Remove productive PBL fallback to first available channel; unavailable configured reference produces no substituted PBL diagnostic.
- [x] Move atmospheric source/fallback policy to `config.yaml` through `level1.atmosphere.source_priority` and explicit outside-coverage policy.
- [x] Make radiosonde temporal-selection policy explicit (`synoptic_hours_utc`, `selection`, `max_time_delta_hours`).
- [x] Remove hardcoded radiosonde station ID fallback; identity is resolved only from `station.yaml`.
- [x] Require complete ERA5 configuration when ERA5 is in the source policy.
- [x] Do not replace invalid/missing ERA5 pressure-level configuration with an internal list.
- [x] Resolve station altitude historically for atmosphere interpolation rather than using one timeless site altitude.

## E. Level 2 strict configuration

- [x] Require wavelengths; productive Level 2 no longer falls back to `[532]`.
- [x] Require temporal block averaging; productive Level 2 no longer falls back to 15 minutes.
- [x] Require KFS mode explicitly.
- [x] Require Monte Carlo iterations and random seed explicitly.
- [x] Require beta-reference uncertainty and aerosol reference fraction explicitly.
- [x] Require minimum aerosol lidar ratio and negative-aerosol policy explicitly.
- [x] Remove productive aerosol LR 60 sr fallback.
- [x] Remove productive aerosol LR uncertainty 10 sr fallback.
- [ ] Require LR values for every requested wavelength/month and associated uncertainty/provenance. **Values + uncertainty are now fail-fast for all 12 months; scientific provenance still needs a formal field.**
- [x] Remove molecular lidar-ratio YAML knob; keep Rayleigh molecular ratio as algorithm constant.
- [x] Require complete productive gluing configuration.
- [ ] Remove low-level gluing window/search/threshold defaults. Productive entry is fail-fast, but compatibility defaults still exist inside `gluing.py`.
- [ ] Do not widen an invalid configured gluing search interval to the whole profile.
- [ ] Make gluing selection-score weights explicit configuration or documented versioned algorithm constants.
- [x] Require complete productive Rayleigh-reference configuration.
- [ ] Do not select the last available bin when no valid Rayleigh window exists.
- [x] Require explicit cloud-screening enabled/disabled state.
- [x] Require all cloud-screening parameters when enabled.
- [x] Make cloud baseline percentile explicit when cloud screening is enabled.
- [ ] Integrate cloud contamination into reference-window QA.
- [ ] Remove residual literal fallback access in `_retrieval_impl.py`; productive entry is protected by full validation, but internal cleanup remains.
- [ ] Later migration: express public gluing/Rayleigh spatial windows in physical units rather than bins/indices.

## F. IO/runtime/visualization

- [ ] Remove filesystem directory fallbacks from production config paths.
- [ ] Remove logging-level fallbacks when a loaded MILGRAU config is used.
- [ ] Remove visualization output-format/DPI/altitude-range fallbacks.
- [ ] Ensure `mean_profile_smooth_bins` from YAML is actually used everywhere intended.
- [ ] Make quicklook gap threshold explicit; no derived 10 min/3x-median fallback unless deliberately defined as an algorithmic mode.
- [ ] Keep display-only colors/style constants in code unless the project intentionally exposes them as theme configuration.

## G. Scientific failure semantics

- [ ] Missing required config -> configuration error before processing starts for every stage. **Implemented for productive Level 0, Level 1 and Level 2; remaining non-stage/runtime paths still need audit.**
- [x] Missing Level 1 channel calibration follows only the explicit configured policy: fail or warned exact-zero legacy correction.
- [x] Unknown PC saturation characterization is represented explicitly as `not_characterized` in station calibration instead of assigning an invented detector limit.
- [x] Stop reusing numerical dead-time clipping as a detector saturation proxy.
- [x] Level 1 persists whether PC saturation is characterized and the rate limit when available; Level 2 does not accept an uncharacterized PC source as scientifically valid retrieval input.
- [ ] Invalid Rayleigh search -> reference selection failure.
- [ ] Invalid gluing search -> gluing failure.
- [x] Missing external atmosphere follows only the explicitly configured source policy; exhausted policies fail rather than silently selecting USSA76.
- [x] Missing surface weather follows only the explicit `nan`/`fail` Level 0 policy; productive processing no longer inserts 25 C / 940 hPa.
- [ ] Optional diagnostic failure must not silently change the scientific algorithm.

## H. Provenance / FAIR

- [ ] Persist software version and Git commit in products.
- [ ] Persist scientific algorithm name/version separately from package version.
- [ ] Persist processing-config hash.
- [ ] Persist station-config hash.
- [ ] Persist resolved station profile ID.
- [ ] Persist resolved instrument calibration ID.
- [ ] Persist input file hashes or an immutable input manifest.
- [ ] Persist random seed for Monte Carlo retrievals.
- [x] Persist atmosphere source, source datetime/time delta, fallback fraction, source-priority attempts and resolved station geometry.
- [ ] Persist when a neutral legacy Level 1 channel calibration was assumed.
- [ ] Persist LR source/provenance used by each retrieval.

## I. Tests / architecture guardrails

- [ ] Rewrite broad config tests for strict schema; remove tests that freeze legacy aliases. **Repository expectations now reflect stage ownership; general legacy-schema tests remain until tranche A.**
- [x] Add station calibration/profile resolution tests across historical eras.
- [x] Add explicit-policy tests for missing Level 1 channel calibration, including warned neutral legacy processing and strict error mode.
- [x] Add failure tests for missing LR month/uncertainty.
- [ ] Add failure tests for invalid gluing/Rayleigh domains. **Config-level gluing interval validation is tested; algorithmic failure tests remain.**
- [x] Add failure tests for missing station timezone/coordinates at strict Level 0 accessors. **Historical altitude coverage is tested in Level 1 atmosphere resolution.**
- [x] Add explicit-policy tests for unavailable saturation characterization.
- [x] Add Level 1 tests that distinguish numerical dead-time clipping from characterized physical saturation.
- [x] Add Level 2 boundary tests ensuring missing/uncharacterized PC saturation metadata is not assumed clear.
- [x] Add atmosphere-policy tests for source order, omitted sources, policy exhaustion, explicit USSA76 extension, radiosonde temporal selection and strict ERA5 pressure levels.
- [x] Add Level 0 tests for explicit shot tolerance, timestamp jitter, dark-current association and missing-weather `nan`/`fail` policy.
- [ ] Add architectural guard against `config.get(..., semantic_literal_default)` outside the config layer.
- [ ] Add regression test ensuring production config contains no undeclared semantic defaults.
- [ ] Run full test suite after each coherent implementation batch. **No CI is currently attached to the branch; isolated strict-L2 tests passed 19/19 earlier. A current full snapshot test run could not be executed from this environment.**

## J. Follow-up FAIR/release work (not part of the strict-config code batch unless touched by necessity)

- [ ] Add repository software `LICENSE` after confirming institutional licensing choice.
- [ ] Unify package version and `CITATION.cff` release version.
- [ ] Restore/create documentation referenced by README.
- [ ] Add CI checks and branch protection.
- [ ] Audit NetCDF semantic metadata against current CF conventions.
- [ ] Add archived releases/DOI workflow and changelog.
- [ ] External real-data validation against an independent/reference processing chain.

## Suggested improvements recorded but not implemented

- [ ] Consider carrying detector mode explicitly from Level 0 acquisition metadata through the full pipeline instead of relying on canonical channel-name suffixes (`.PC` / `.AN`) outside the resolved station-calibration path. This was identified during the Level 1 tranche and intentionally not implemented without separate review.

## Implementation log

### 2026-09-09 — tracker initialized

- Baseline branch head before this tracker: `05f560f37551d8dda87b87d6585b0d92cc42c195`.
- Configuration audit completed before implementation.
- No scientific/instrumental values will be invented merely to satisfy the new schema. Unknown calibration quantities must be represented explicitly as unavailable/not characterized until measured or sourced.

### 2026-09-09 — Level 2 fail-fast configuration tranche

- Added `Level2ConfigurationError` and complete productive Level 2 validation in `milgrau/level2/config.py`.
- Productive wavelengths, temporal averaging, KFS/Monte Carlo controls, gluing thresholds, Rayleigh-reference controls and LR climatology no longer receive semantic defaults from the Level 2 config helper.
- All 12 monthly LR values plus uncertainty are required for each requested productive wavelength before retrieval starts.
- Removed the molecular Rayleigh lidar-ratio setting from `config.yaml`; it remains a software/physics constant.
- Added the previously hidden gluing `gaussian_threshold` explicitly to `config.yaml`.
- Added explicit `cloud_screening.enabled: false`; enabling screening requires the complete detector configuration, including baseline percentile.
- Added strict Level 2 tests and updated cloud-screening tests. Isolated validation: `19 passed`.
- Remaining L2 cleanup is intentionally visible above: low-level defaults in `gluing.py`, arbitrary Rayleigh fallback in `molecular.py`, residual literal access in `_retrieval_impl.py`, score weights, cloud-reference integration and LR provenance.

### 2026-09-09 — station calibration catalog tranche

- Replaced global `channel_corrections` with named `calibrations` in `station.yaml`.
- Added `calibration_id` to every temporal station profile.
- Preserved every existing numerical correction exactly; no new scientific calibration number was invented.
- Renamed `scc_defaults` to `scc_policy` with explicit `fixed_value` and `raman_companions_nm` semantics.
- Removed `fallback_to_standard_atmosphere` from station identity metadata; atmospheric fallback policy still needs to move into the algorithm config in the Level 1 tranche.
- Added explicit `saturation.status: not_characterized` to every current photon-counting calibration channel because the repository does not contain defensible detector saturation limits.
- Runtime station context now resolves `calibration_id`, calibration provenance and complete channel calibration data together with `profile_id` and SCC mapping.
- A temporary `physics.channels` compatibility view remains so non-migrated consumers are not broken; productive Level 1 no longer reads it.
- Added station tests for calibration resolution, unknown calibration references and saturation characterization semantics.
- GitHub currently reports no CI/status checks on the branch, so the full repository suite has not been claimed as executed.

### 2026-09-09 — Level 1 strict calibration access tranche

- `milgrau.level1.common.get_channel_constant` now raises on missing channel calibration instead of silently applying neutral corrections.
- Positional correction lists are rejected at the Level 1 consumer; named `deadtime_us`, `bin_shift_bins`, and `background_offset` fields are required.
- Added focused tests for missing, positional, incomplete, and valid channel calibration access.
- Loader-level positional compatibility still exists and remains explicitly tracked for removal in the schema/loader tranche.

### 2026-09-09 — Level 1 strict processing recipe and saturation semantics tranche

- Added typed strict `milgrau.level1.config` resolution for background, photon-counting numerical policy, PBL settings and temporal station calibration selection.
- Moved Level 1 background/PBL/dead-time clipping controls into the explicit `level1` section of `config.yaml`.
- Removed the configurable speed of light from repository YAML; bin time now uses the exact SI constant in code.
- Productive LIPANCORA no longer consumes `physics.channels`; it resolves the calibration set associated with the Level 0 station profile/date.
- Numerical dead-time denominator clipping and physical PC saturation are now distinct masks/diagnostics.
- `not_characterized` PC calibration produces no invented physical saturation mask; Level 1 records `pc_saturation_characterized=0` and a missing rate limit explicitly.
- Level 2 treats PC without characterized saturation metadata as unavailable for scientific retrieval, while an independently valid analog fallback remains possible according to the existing policy.
- PBL uses only the configured reference channel and configured search/smoothing parameters; it does not substitute a first/532 channel when the requested channel is unavailable.
- Added/updated focused tests for strict Level 1 configuration, calibration resolution, clipping-vs-saturation semantics, PBL no-fallback behavior and the Level 1 -> Level 2 saturation-characterization boundary.
- The generic loader intentionally does not import the Level 1 validator; LIPANCORA validates before discovery/processing to avoid inverted dependencies/cycles. The old broad schema remains transitional.
- No full repository test run is claimed: the branch has no CI checks, and this environment could not obtain a runnable branch snapshot.

### 2026-09-11 — Level 1 explicit atmosphere policy tranche

- Commit `ae1c07dcc6c50eddc5be5a26b1dc72815d04b60f` introduced the atmosphere-policy code/config/test batch atomically.
- Added `level1.atmosphere.source_priority` and `external_profile_outside_coverage`; productive source selection now follows only the declared order.
- Preserved the intended source order as `radiosonde -> era5 -> ussa76`, but made every transition explicit rather than hardcoded.
- Radiosonde temporal selection is now explicit: configured synoptic UTC hours, `nearest` selection, and a finite maximum time delta.
- The station radiosonde ID/name are resolved only from `station.yaml`; the former hardcoded `83779` fallback was removed from productive code.
- ERA5 now receives a complete strict settings object. Missing/invalid pressure levels, grid, area, dataset or cache configuration raise instead of being replaced by internal defaults.
- External-profile vertical extension is explicit: `ussa76` or `fail`. USSA76 is not selected as a whole-profile fallback unless it appears in `source_priority`.
- Atmosphere provenance now records configured source priority, sources attempted, resolved station geometry, source time/time delta, source DOI when available and USSA76 extension fraction.
- Atmosphere interpolation resolves the historical station profile so the September 2024 altitude transition (766 m -> 740 m) is respected.
- Added focused tests for radiosonde target-time selection, maximum time delta, strict ERA5 settings, source-order enforcement, omitted-source behavior, exhausted policy failure, station-only radiosonde identity and historical station altitude.
- No full repository test run is claimed for this tranche because the branch still has no CI and this environment cannot fetch a runnable repository snapshot over the network.

### 2026-09-11 — explicit historical neutral channel-calibration policy

- Historical files may now process channels absent from the traceable calibration set only because `config.yaml` explicitly selects `level1.missing_channel_calibration.policy: neutral_with_warning`.
- The fallback values are required to be exactly `deadtime_us=0`, `bin_shift_bins=0`, and `background_offset=0`; non-zero values are rejected as not neutral.
- Each use emits a `RuntimeWarning` stating that the values are not a measured calibration.
- Detector mode is inferred only from canonical `.PC`/`.AN` channel suffixes in this legacy path; an unknown detector suffix still fails.
- Photon-counting fallback channels remain `saturation.status=not_characterized` and receive no invented saturation rate.
- `ChannelCalibration.assumed_neutral` records the runtime state; persistence of that flag into the Level 1 NetCDF remains tracked rather than silently implied.
- The policy can be changed back to `error` without code changes.

### 2026-09-11 — Level 0 strict acquisition/weather tranche

- Added typed strict `milgrau.level0.config` resolution and validation before LIBIDS discovery/processing.
- Moved laser-shot tolerance, Licel header timestamp jitter and dark-current association maximum from code/legacy processing keys into explicit `level0` configuration.
- Inventory timezone is now resolved only from the validated station catalog; the `America/Sao_Paulo` Python fallback was removed from the productive inventory path.
- Surface-weather coordinates are now resolved only from station metadata; productive weather retrieval no longer falls back to hardcoded São Paulo coordinates.
- Removed repository-config surface defaults `25 C` and `940 hPa`. Missing surface weather now follows `level0.surface_weather.missing_policy: nan|fail`; repository policy is `nan`.
- Updated acquisition-QA and inventory tests and added focused strict-Level-0 tests for missing/unknown keys, station metadata, dark-current limits, timestamp jitter and weather policy.
- Low-level writer cleanup is intentionally still visible in section C: `netcdf.py` retains compatibility literals for coordinates/surface values, fake SCC ID `9999`, and the 7.5 m range-resolution fallback. These are not marked complete until removed/reclassified.
- No full repository test run is claimed; the branch has no CI checks and this environment still lacks a runnable repository snapshot.
