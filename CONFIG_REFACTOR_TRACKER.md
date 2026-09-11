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
- [ ] No algorithmic fallback that silently widens/changes a configured scientific domain. **Productive gluing and Rayleigh-reference search domains are now strict; other runtime paths still require audit.**
- [x] Optional Level 2 cloud-screening behavior must be explicitly enabled/disabled in YAML.
- [x] Rayleigh molecular lidar ratio remains a versioned physical/method constant in Python rather than a user-editable YAML setting.
- [x] Produced NetCDF provenance is human-readable: software release, resolved station/calibration IDs, source YAML filenames and exact embedded YAML text propagate through the product chain. Level 2 additionally records Monte Carlo seed/iterations and the readable LR source. Input-manifest policy remains pending.
- [ ] Tests cover every required-key failure and every explicit disabled/unavailable policy.

## A. Configuration ownership and schema

- [ ] Replace generic `physics` ownership with stage-oriented processing configuration. **Level 1 science controls have moved to `level1`; Level 0 acquisition/weather controls have moved to `level0`; legacy schema still requires `physics.vertical_resolution_m` although the productive Level 0 writer no longer consumes it.**
- [ ] Keep station/site/instrument metadata in `station.yaml` only. **Station-derived LR climatology and vertical pointing geometry are now station-owned; runtime compatibility views of site/hardware remain.**
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
- [x] Record invariant SPU-Lidar vertical geometry in `station.yaml` as `station.lidar_geometry.pointing_angle_deg_from_zenith: 0.0`.
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
- [x] Persist resolved `station_profile_id` and `instrument_calibration_id` in products. Level 0 writes resolved IDs and Level 1/2 inherit them through the FAIR provenance helper.

## C. Level 0 strict configuration

- [x] Require raw/processed/log directories and remove productive path fallbacks. `resolve_level0_config` now resolves them explicitly and `milgrau.io.paths` no longer invents raw/processed/log roots.
- [x] Require raw discovery spurious extensions, ignore dirs and quarantine path. Productive discovery receives a resolved policy; `milgrau.io.filesystem` no longer reads config or supplies discovery defaults.
- [x] Standardize disposable caches under `.cache/`: `.cache/radiosonde`, `.cache/era5`, `.cache/weather`; caches no longer live inside `01-data`.
- [x] Define auditable quarantine layout/manifest. Quarantine remains an explicit action and uses `quarantine/YYYY/MM/DD/<reason>/` with a JSON sidecar containing UTC timestamp, reason/stage, original path, SHA-256, byte size, retained filename and optional measurement ID.
- [x] Require laser-shot tolerance through `level0.acquisition_qa.laser_shot_tolerance_fraction`.
- [x] Make Licel header timestamp jitter threshold explicit through `level0.acquisition_qa.licel_header_time_jitter_s`.
- [x] Require finite dark-current association maximum through `level0.dark_current.max_association_hours`.
- [x] Remove productive timezone fallback to `America/Sao_Paulo`; LIBIDS resolves timezone only from the station catalog.
- [x] Remove hardcoded São Paulo latitude/longitude fallbacks from productive weather and Level 0 NetCDF metadata paths.
- [x] Remove fallback surface temperature 25 C / pressure 940 hPa; missing values remain NaN or fail according to explicit policy.
- [x] Use explicit missing-surface-weather policy (`nan` or `fail`) through `level0.surface_weather.missing_policy`.
- [x] Remove `DEFAULT_CHANNEL_ID = 9999`; missing SCC channel ID invalidates the SCC writer path.
- [x] Remove global 7.5 m range-resolution fallback completely. `Raw_Data_Range_Resolution` now requires positive finite native Licel `BinW`; `physics.vertical_resolution_m` cannot mask missing acquisition metadata.
- [x] Remove invented Licel analog ADC/range defaults. Active analog channels require explicit positive `ADCbits` and `Discriminator/DAQ range`; former 12-bit / 0.5-V substitutions are gone.
- [x] Remove hidden Level 0 SCC background-window literals. `Background_Low/High` are written from the explicit configured background interval rather than internal 29000/29999 m constants.
- [ ] Make the Level 0 writer consume `station.lidar_geometry.pointing_angle_deg_from_zenith` directly and remove the residual low-level `0.0` fallback. **Productive LIBIDS now requires/resolves the station-owned angle when a station catalog is loaded and materializes it into the transitional writer view; only direct low-level compatibility cleanup remains.**

## D. Level 1 strict configuration

- [x] Photon-counting Poisson uncertainty uses observed counts before dark subtraction.
- [x] Canonical atmosphere is materialized in Level 1 with source/fallback provenance.
- [x] Missing channel calibration follows an explicit policy: `error` or `neutral_with_warning`. Repository policy uses exact zero corrections for historical processability and emits a `RuntimeWarning`; PC saturation remains `not_characterized`.
- [x] Persist the per-channel fact that a neutral historical calibration was assumed. Level 1 now writes `calibration_assumed_neutral(channel)` plus `neutral_legacy_calibration_channel_count`.
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
- [x] Require LR values for every requested wavelength/month and associated uncertainty/provenance. **Values + uncertainty are fail-fast for all 12 months. SPU climatology comes authoritatively from `station.yaml`; products now record the readable source path. A paper/DOI may be added later when one exists.**
- [x] Remove molecular lidar-ratio YAML knob; keep Rayleigh molecular ratio as algorithm constant.
- [x] Require complete productive gluing configuration.
- [ ] Remove low-level gluing window/search/threshold defaults. **Productive callers pass strict values; public low-level compatibility defaults remain.**
- [x] Do not widen an invalid configured gluing search interval to whole profile. The gluing kernel now rejects out-of-domain or too-narrow configured intervals.
- [ ] Make gluing selection-score weights explicit or documented versioned algorithm constants.
- [x] Require complete productive Rayleigh-reference configuration.
- [x] Do not select last available bin when no valid Rayleigh window exists. Reference selection now raises an explicit failure for insufficient or invalid configured search windows.
- [x] Require explicit cloud-screening enabled/disabled state.
- [x] Require all cloud-screening parameters when enabled.
- [x] Make cloud baseline percentile explicit when enabled.
- [ ] Integrate cloud contamination into reference-window QA.
- [ ] Remove residual literal fallback access in `_retrieval_impl.py`. **The public retrieval boundary overrides canonical atmosphere correctly, but duplicate/dead compatibility implementations remain a maintenance risk.**
- [ ] Later migration: express public gluing/Rayleigh spatial windows in physical units rather than bins/indices.

## F. IO/runtime/visualization/CLI

- [x] Remove filesystem raw/processed/log directory fallbacks from production config paths. Standardized cache locations remain code-level IO conventions, not scientific path fallbacks.
- [x] Remove logging-level fallbacks when a loaded MILGRAU config is used. `processing.console_level` and `processing.file_level` are required and validated.
- [x] Use concise contextual operator logs with `pipeline`, `save_id`, and `stage`, while retaining detailed diagnostics in the DEBUG audit file. Repository policy is console `INFO`, file `DEBUG`.
- [x] Require explicit `processing.incremental` boolean in productive L0/L1/L2/VIZ runtime helpers rather than silently defaulting to `false`.
- [x] Primary CLIs share the essential operator surface: repeatable `--input`, `--force`, and `--version`; LEBEAR additionally keeps `--time-window`.
- [x] Simplify generic execution outcomes to `OK`, `SKIPPED`, `ERROR`; scientific quality stays in QA variables/diagnostics rather than software status labels. Transitional enum aliases remain until compatibility cleanup.
- [x] Shell exit policy is now `0=normal`, `1=processing error(s)`, `2=command could not run/fatal structural error`; console summaries show processed/skipped/errors rather than recoverable/fatal terminology.
- [x] Remove visualization output-format/DPI/altitude-range fallbacks. Productive visualization now resolves these fields strictly from `visualization`.
- [x] Ensure `mean_profile_smooth_bins` from YAML is actually used for quicklook side profiles and the global mean RCS profile.
- [x] Make quicklook gap threshold explicit; no derived 10 min/3x-median fallback remains.
- [x] Keep display-only colors/style constants in code unless intentionally exposed as theme configuration. Wavelength colors and logo layout remain implementation/display constants; user-facing colormap and missing-data color stay explicit config.

## G. Scientific failure semantics

- [ ] Missing required config -> configuration error before processing starts for every stage. **Implemented for productive Level 0, Level 1 and Level 2 scientific recipes plus logging/incremental/visualization runtime controls; remaining compatibility/runtime paths still need audit.**
- [x] Missing Level 1 channel calibration follows only explicit configured policy.
- [x] Unknown PC saturation characterization is explicit `not_characterized`.
- [x] Numerical dead-time clipping is not a detector saturation proxy.
- [x] Level 2 does not accept uncharacterized PC saturation as known-clear input.
- [x] Invalid Rayleigh search -> reference selection failure. No last-bin substitute remains.
- [x] Invalid gluing search -> gluing failure. The configured domain is never widened automatically; absence of a qualifying overlap may still use the separately configured single-channel fallback policy.
- [x] Missing external atmosphere follows only explicitly configured source policy.
- [x] Missing surface weather follows only explicit `nan`/`fail` policy.
- [x] Missing/invalid active Licel `BinW`, analog ADC bits or DAQ range invalidates that input instead of applying acquisition defaults.
- [ ] Optional diagnostic failure must not silently change scientific algorithm.

## H. Provenance / FAIR

- [x] Persist software release version in products using human-readable CalVer. **MILGRAU is standardized at `2026.9`; Git commit hashes are deliberately not public NetCDF metadata.**
- [ ] Persist scientific algorithm name/version separately from package version. **Level 2 already writes versioned Fernald/molecular implementation metadata; extend/version other scientific stages deliberately before checking globally.**
- [x] Embed the exact processing YAML used for a product as scalar NetCDF string variable `processing_configuration_yaml`, with source filename and `application/yaml` metadata.
- [x] Embed the exact station/instrument YAML used for a product as scalar NetCDF string variable `station_configuration_yaml`, with source filename and `application/yaml` metadata.
- [x] Persist resolved station profile ID.
- [x] Persist resolved instrument calibration ID.
- [ ] Persist input file hashes or immutable input manifest. **User-facing product provenance should stay readable; decide whether a compact filename/size/time manifest is sufficient before adding hashes.**
- [x] Persist Level 2 Monte Carlo random seed and iteration count as `monte_carlo_random_seed` and `monte_carlo_iterations`.
- [x] Persist atmosphere source, source datetime/time delta, fallback fraction, source-priority attempts and resolved station geometry.
- [x] Persist when a neutral legacy Level 1 channel calibration was assumed, per channel and as a product-level count.
- [x] Persist LR source/provenance used by each retrieval as a readable source path (`station.yaml: station.lidar_ratio_climatology` for the current SPU recipe). **No paper/DOI is invented; that reference can be added later when it exists.**
- [x] Persist an audit sidecar for every explicit quarantined input, including SHA-256 and origin/context metadata. **Hash remains appropriate for quarantine integrity even though product NetCDF provenance is human-readable.**
- [x] Regenerated products remove legacy public `processing_config_sha256` / `station_config_sha256` attributes so old hash-oriented provenance does not leak forward through xarray inheritance.

## I. Tests / architecture guardrails

- [ ] Rewrite broad config tests for strict schema; remove tests that freeze legacy aliases.
- [x] Add station calibration/profile resolution tests across historical eras.
- [x] Add explicit-policy tests for missing Level 1 channel calibration.
- [x] Add failure tests for missing LR month/uncertainty.
- [x] Add tests that station LR climatology owns repository values and explicit config LR remains only compatibility fallback.
- [x] Add failure tests for invalid gluing/Rayleigh domains. Algorithm-level tests now prove configured gluing domains are not widened and missing Rayleigh windows fail explicitly.
- [x] Add failure tests for missing station timezone/coordinates at strict Level 0 accessors.
- [x] Add explicit-policy tests for unavailable saturation characterization.
- [x] Add Level 1 clipping-vs-saturation tests.
- [x] Add Level 2 boundary tests for missing/uncharacterized PC saturation metadata.
- [x] Add atmosphere-policy tests.
- [x] Add Level 0 tests for explicit shot tolerance, timestamp jitter, dark-current association, directories/discovery and weather policy.
- [x] Add strict Licel tests for active `BinW`, analog ADC bit depth and discriminator/DAQ range.
- [x] Add Level 0 writer regression tests proving legacy `vertical_resolution_m` cannot replace missing native `BinW` and hidden background literals are not used.
- [x] Add quarantine tests for dated reason buckets, SHA-256 sidecars, collisions and read-only discovery semantics.
- [x] Add logging tests for contextual console fields, INFO/DEBUG destination split, explicit levels and handler ownership.
- [x] Add strict visualization tests for required output/DPI/altitude/gap/smoothing settings and update LIRACOS incremental tests to the stdlib logger contract.
- [x] Add FAIR provenance tests for human-readable software/station identity and exact embedded YAML content; public SHA assertions were removed.
- [x] Add common CLI option and CalVer synchronization tests.
- [ ] Add architectural guard against `config.get(..., semantic_literal_default)` outside config layer.
- [ ] Add regression test ensuring production config contains no undeclared semantic defaults.
- [ ] Run full test suite after each coherent implementation batch. **No CI is currently attached to the branch; isolated strict-L2 tests passed 19/19 earlier. Current full suite remains unverified in this environment.**

## J. Follow-up FAIR/release work

- [ ] Add repository software `LICENSE` after confirming institutional licensing choice.
- [x] Unify package version and `CITATION.cff` release version using CalVer `2026.9`, exposed at runtime as `milgrau.__version__`.
- [ ] Restore/create documentation referenced by README.
- [ ] Add CI checks and branch protection.
- [ ] Audit NetCDF semantic metadata against current CF conventions.
- [ ] Add archived releases/DOI workflow and changelog.
- [ ] External real-data validation against independent/reference processing chain.

## Suggested improvements recorded but not implemented

- [ ] Consider carrying detector mode explicitly from Level 0 acquisition metadata through full pipeline instead of relying on canonical `.PC` / `.AN` suffixes outside resolved station calibration.
- [ ] Remove the residual direct Level 0 writer pointing-angle compatibility fallback now that productive LIBIDS resolves the station-owned angle.
- [ ] Remove transitional `SUCCESS/RECOVERABLE_FAILURE/FATAL_FAILURE` and legacy `ExitCode` aliases after remaining tests/callers have migrated to `OK/SKIPPED/ERROR`.
- [ ] Decide whether one invalid Rayleigh block should invalidate the full wavelength or be recorded as a failed block while allowing other valid blocks to continue. Current strict behavior prevents arbitrary reference substitution but is deliberately conservative.

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

- Moved unchanged SPU monthly lidar-ratio climatology and standard deviations from `config.yaml` into `station.yaml` with provenance. Loader materializes a transitional Level 2 view; explicit config LR is accepted only when station climatology is absent.
- Standardized configured caches to `.cache/radiosonde`, `.cache/era5`, and `.cache/weather`; `.cache/` and `quarantine/` are ignored by Git.
- Removed fake SCC channel ID and remaining coordinate/surface-value literals from Level 0 writer paths.
- Changed generic IO cache fallbacks from raw-data cache folders to `.cache/weather` and `.cache/radiosonde`.
- Removed hidden Licel analog defaults of 12 ADC bits / 0.5 V and requires explicit positive active-channel `BinW`.

### 2026-09-11 — Level 0 filesystem/range completion tranche

- Made Level 0 raw/processed/log directories and raw-discovery policy explicit in the typed resolver; generic raw discovery now consumes explicit values and has no config/default layer of its own.
- Removed the residual Level 0 writer 7.5-m range-resolution fallback and hidden 29000/29999-m background literals. Native Licel `BinW` is mandatory and SCC background metadata is tied to the explicit configured interval.
- Added writer regression tests for native range metadata and explicit background ownership.
- Implemented dated reason-based quarantine buckets and JSON audit sidecars with SHA-256, origin, size, stage and optional measurement ID; quarantine remains manual/explicit.
- Added quarantine layout/manifest tests and preserved read-only raw discovery.
- Level 0 section C is now complete for the tracked strict-config scope except the explicitly tracked direct-writer pointing compatibility cleanup.

### 2026-09-11 — contextual logging and strict retrieval-domain audit

- Fixed the post-C import regression in `logging_utils` caused by the removed `DEFAULT_LOG_DIR` symbol.
- Added contextual `pipeline/save_id/stage` logging with compact operator-console formatting and a richer full-date DEBUG audit file. Repository policy is `console_level: INFO`, `file_level: DEBUG`.
- L0/L1/L2 orchestration now carries canonical save IDs; per-channel/cache/transport details are moving to DEBUG while concise start/QA/station/atmosphere/wavelength/done events remain visible to operators.
- Removed logging-level and `processing.incremental=false` runtime fallbacks; loaded productive configuration must state these policies explicitly.
- Persisted `calibration_assumed_neutral(channel)` and neutral-channel count in Level 1 products.
- Removed gluing search-domain widening and arbitrary last-bin Rayleigh reference selection; added algorithm-level failure tests for both.
- Weather, radiosonde and ERA5 cache/download transport messages are DEBUG; availability/policy failures remain warnings.
- Added a strict visualization resolver; output format, DPI, altitude ranges, plotted channels, smoothing bins, gap threshold, colormap and missing-data color no longer fall back internally.
- LIRACOS now uses the same contextual logging style (`VIZ/save_id/stage`) and the configured `mean_profile_smooth_bins` drives both quicklook side profiles and global mean RCS smoothing.
- Full repository suite is not claimed: the branch still has no CI/status checks and the complete suite has not been executed in this environment.

### 2026-09-11 — FAIR config/station provenance tranche

- Added reusable `milgrau.provenance` helpers and initial configuration/station provenance propagation.
- Level 0 primary/SCC products persist resolved station profile and instrument calibration IDs; Level 1 and Level 2 inherit that historical identity.
- Initial hash-oriented provenance was subsequently replaced by the human-readable/YAML-embedded policy below.

### 2026-09-11 — CLI, outcomes, CalVer, and readable FAIR provenance tranche

- Simplified generic operational results to `OK/SKIPPED/ERROR`; fatal is now an error attribute used only to reserve shell exit 2 for commands that cannot run. Console summaries no longer expose recoverable/fatal framework terminology.
- Added repeatable `--input`, `--force`, and `--version` to LIBIDS, LIPANCORA, LIRACOS and LEBEAR; LEBEAR retains `--time-window`. LIBIDS input selection remains measurement-group based so raw-file selection cannot bypass group/dark-current context.
- Adopted CalVer `2026.9`, synchronized runtime `__version__`, `pyproject.toml`, and `CITATION.cff`.
- Replaced public NetCDF SHA metadata with human-readable provenance: software name/version, YAML filenames, resolved profile/calibration IDs, plus exact `config.yaml` and `station.yaml` contents embedded as scalar `application/yaml` variables.
- Kept SHA-256 only where it has an integrity role (for example quarantine sidecars), not as the primary human-facing scientific provenance vocabulary.
- Recorded the confirmed invariant SPU-Lidar pointing angle (`0.0° from zenith`) in `station.yaml`; productive LIBIDS now resolves/materializes it before Level 0 writing while direct low-level writer compatibility remains tracked.
- Level 2 products now persist the exact configured Monte Carlo random seed/iteration count and a readable lidar-ratio source path. No DOI/reference is fabricated; a future publication can extend this provenance naturally.
- Regenerated products explicitly remove legacy public configuration-hash attributes inherited from old files.
- Added CLI-surface, CalVer synchronization, simplified-result, and readable-provenance tests. Full repository suite is still not claimed in this environment.
