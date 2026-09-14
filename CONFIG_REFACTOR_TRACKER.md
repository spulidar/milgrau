# MILGRAU configuration refactor tracker

Branch: `new-architecture`

This file is the source of truth for the strict-configuration refactor. Update it in every coherent implementation batch.

## Goal

Make scientific and instrumental decisions explicit, auditable, and readable:

- `config.yaml` = processing/scientific recipe.
- `station.yaml` = observational reality, hardware history, station-derived climatologies, calibration and SCC mapping.
- Python = equations, physical constants, file-format invariants, validated runtime objects, and implementation details.
- Missing required scientific/instrumental configuration must fail early unless an explicit, auditable unavailable/legacy policy is selected.
- Operator logs should report scientific/processing outcomes first; detailed search/transport/debug information belongs in the DEBUG audit log.

## Global rules

- [ ] No silent scientific defaults in production paths. Productive L0/L1/L2 paths are substantially strict; residual compatibility code in Level 2 still needs audit.
- [ ] No silent instrumental defaults in production paths. Productive acquisition/calibration paths are strict; continue auditing low-level helpers.
- [x] No station-specific coordinates/timezone/IDs embedded in productive Python fallbacks.
- [x] No silent neutral channel correction fallback. Historical neutral correction is only available through explicit `level1.missing_channel_calibration` policy and is persisted in Level 1.
- [x] No fake SCC channel IDs.
- [x] Productive gluing and Rayleigh-reference domains are never silently widened/substituted.
- [x] Optional Level 2 cloud screening has an explicit enabled/disabled policy.
- [x] Rayleigh molecular lidar ratio remains a versioned physical/method constant in Python.
- [x] NetCDF provenance is human-readable: release version, resolved station/calibration IDs, exact processing/station YAML, atmosphere source, MC settings, LR source.
- [ ] Add/decide a readable immutable input-manifest policy.
- [ ] Tests cover every required-key failure and every explicit disabled/unavailable policy globally; major productive contracts are covered, full audit remains.

## A. Configuration ownership and schema

- [ ] Remove the remaining generic `physics` section completely. `physics.vertical_resolution_m` remains in repository config for historical/schema cleanup but is no longer a productive L0 range-resolution fallback.
- [x] Keep station/site/instrument identity in `station.yaml`; productive loader no longer duplicates station metadata into `site`, `radiosonde`, `physics.channels`, or `hardware.name_to_id` compatibility views.
- [x] Stop rebuilding `physics.channels` from station data.
- [x] Stop rebuilding `hardware.name_to_id` from station data.
- [x] `normalize_config` no longer injects legacy aliases.
- [x] Productive Level 1 rejects legacy positional channel-correction lists and consumes named station calibration.
- [ ] Fully retire legacy global/minimum-schema validation philosophy in favor of stage-specific strict validation. Productive L0/L1/L2 use strict resolvers; continue auditing auxiliary paths.
- [ ] Validate unknown keys with full paths across every stage. L0/L1/station sections already have strong exact-key validation in key areas.
- [ ] Extend typed/resolved configuration objects or equivalent strict accessors to every scientific/runtime stage. L0/L1 and current L2 recipe access are substantially migrated.
- [x] Loader performs only file parsing, station-catalog validation, source-path recording, and the intentional station-LR → Level-2 recipe materialization.

## B. `station.yaml`: observational reality

- [x] Station ID/name/institution/timezone/site coordinates live in station catalog.
- [x] Radiosonde station identity lives in station catalog.
- [x] SPU-Lidar invariant geometry is explicit as `station.lidar_geometry.pointing_angle_deg_from_zenith: 0.0`.
- [x] Station-derived aerosol lidar-ratio climatology and uncertainty live in station catalog with readable provenance.
- [x] SCC defaults renamed to explicit `scc_policy` terminology.
- [x] Named instrument calibration sets exist and temporal station profiles reference calibration IDs.
- [x] Calibration structure supports historical changes; current profiles deliberately reference the one correction set supported by available source material.
- [x] `deadtime_us`, `bin_shift_bins`, `background_offset`, detector mode, and PC saturation characterization are station calibration metadata.
- [ ] Add legacy ADC/range overrides only if real historical Licel evidence proves native headers are insufficient. No override is invented today.
- [x] Resolved station profile ID and calibration ID persist through L0 → L1 → L2 products.

## C. Level 0 strict configuration / acquisition

- [x] Require raw/processed/log directories.
- [x] Require raw-discovery spurious extensions, ignored directories, and quarantine path.
- [x] Ignore historical `openmeteo_cache` / `wyoming_cache` and configured non-Licel suffixes so cache files cannot be parsed as Licel data.
- [x] Standardize disposable caches under `.cache/` rather than raw-data trees.
- [x] Quarantine layout/manifest is auditable with date/reason, origin, size, SHA-256, stage, retained name, optional measurement ID.
- [x] Require acquisition laser-shot tolerance and Licel-header time-jitter policy.
- [x] Require finite dark-current association maximum.
- [x] Resolve timezone/coordinates only from station catalog.
- [x] Missing surface weather follows explicit `nan|fail` policy; no invented 25 °C / 940 hPa pair.
- [x] No fake SCC ID `9999`.
- [x] Native positive finite Licel `BinW` is mandatory; no 7.5 m writer fallback.
- [x] Analog ADC bit depth and DAQ range must be present/valid in acquisition metadata; no 12-bit / 0.5-V invention.
- [x] SCC background-window metadata comes from explicit Level 1 background configuration rather than hidden 29–30 km literals.
- [x] Level 0 writer consumes station-owned pointing angle through strict station accessors; no residual `0.0` low-level fallback remains.
- [x] Level 0 incremental currentness requires structural contract plus complete readable provenance; partially written/stale products are reprocessed.

## D. Level 1 strict configuration / atmosphere

- [x] PC Poisson uncertainty uses observed counts before dark subtraction.
- [x] Canonical thermodynamic atmosphere is materialized in Level 1 with source/fallback provenance.
- [x] Missing channel calibration follows only explicit `error` or `neutral_with_warning` policy.
- [x] Neutral historical calibration assumption persists per channel and as product count.
- [x] Exact SI speed of light is code constant, not config.
- [x] Background window, dead-time numerical clipping denominator, PBL reference channel/search interval/smoothing are explicit.
- [x] Numerical dead-time clipping is distinct from detector saturation.
- [x] Productive PBL never substitutes another channel when configured reference is unavailable.
- [x] Atmosphere source order and outside-coverage behavior are explicit in `level1.atmosphere`.
- [x] Radiosonde station identity and temporal selection are explicit/resolved; no hardcoded station-ID fallback.
- [x] ERA5 configuration is complete/fail-fast when ERA5 participates in source policy.
- [x] ERA5 pressure levels are never replaced by an internal list when missing/invalid.
- [x] Historical station altitude is used when mapping external atmosphere to the lidar grid.
- [x] ERA5 retrieval is explicitly routed to the Climate Data Store API (`https://cds.climate.copernicus.eu/api`); a machine-wide ADS endpoint can no longer silently redirect ERA5 requests to the wrong Copernicus service.
- [x] `cdsapi` optional dependency baseline is `>=0.7.7`; credentials remain outside repository/config provenance.
- [x] Atmosphere transport failures remain visible but concise on console; full exception/traceback stays in DEBUG audit log.
- [x] PBL INFO log reports computed mean height + valid profile count; search bounds/settings moved to DEBUG.
- [x] CPT/LRT reuse the same thermal tropopause kernel for radiosonde and ERA5 profiles; USSA76 remains explicitly unavailable as a tropopause source. Product attrs record `tropopause_source_type`.

## E. Level 2 strict configuration

- [x] Require wavelengths, temporal block averaging, KFS mode, MC iterations/seed, beta-reference uncertainty, aerosol-reference fraction, minimum LR, negative-aerosol policy.
- [x] No productive 60 sr LR fallback or 10 sr LR-uncertainty fallback.
- [x] LR values/uncertainty are required for every requested wavelength/month; SPU climatology is authoritative in `station.yaml` and product records readable source path.
- [x] Molecular lidar ratio is a versioned algorithm constant rather than YAML knob.
- [x] Productive gluing configuration is complete and strict.
- [x] Uncharacterized PC channels may participate only through the temporary 10% dead-time-occupancy guard when Level 1 correction succeeded and a traceable positive station dead-time is available; physical saturation remains `not_characterized`.
- [x] A numerically successful AN/PC gluing result that fails retrieval-input QA can retry configured single-channel candidates blockwise instead of suppressing a valid AN fallback.
- [ ] Remove remaining low-level/public gluing defaults retained for compatibility.
- [x] Invalid gluing search domains fail; they are never widened to the full profile.
- [ ] Document/version gluing selection-score weights as explicit algorithm constants.
- [x] Productive Rayleigh-reference config is complete and strict; no arbitrary last-bin reference.
- [x] Cloud-screening enabled/disabled state and all enabled parameters are explicit.
- [ ] Integrate cloud contamination into reference-window QA.
- [ ] Remove duplicate/dead compatibility implementations in `_retrieval_impl.py`; public strict boundary is canonical but maintenance debt remains.
- [ ] Later migration: expose gluing/Rayleigh spatial windows in physical units rather than bins/indices.

## F. IO / runtime / visualization / CLI / logging

- [x] Generic filesystem roots have no productive defaults.
- [x] Loaded config requires explicit console/file log levels and explicit incremental boolean.
- [x] Primary CLIs expose repeatable `--input`, `--force`, `--version`; LEBEAR additionally exposes `--time-window`.
- [x] Generic operation outcomes are only `OK`, `SKIPPED`, `ERROR`; old recoverable/fatal status aliases are removed.
- [x] Shell exit policy is `0=normal`, `1=processing errors`, `2=command could not run structurally`; scientific QA is separate.
- [x] Console summary uses human counts (`processed | skipped | with errors`) and duplicate per-pipeline summaries have been removed.
- [x] Console and audit file share the same compact columns: time, severity, pipeline, save ID, stage, message. Audit file adds the calendar date instead of repeating `pipeline=... save_id=... stage=...` text.
- [x] Legacy `->` presentation noise is stripped and multiline external errors are collapsed to one operator row.
- [x] Repeated calibration/saturation warnings are deduplicated on console only; the DEBUG audit file retains every per-product occurrence.
- [x] Scientific-result stages favor outcomes over method chatter: PBL/CPT/LRT values are operator-visible, while search/transport detail remains DEBUG.
- [x] L0 uses explicit operator stages for parse/station/weather/write/SCC rather than falling back to anonymous `-` context in productive group processing.
- [x] Visualization output format/DPI/altitude ranges/channels/smoothing/gap threshold/colormap/missing-data color are strict config.
- [x] Configured quicklook smoothing drives side/global profiles; no hidden gap fallback.
- [x] Display-only style constants remain code constants unless intentionally promoted to theme configuration.

## G. Scientific failure semantics

- [ ] Missing required config must fail before processing starts for every stage globally. Productive L0/L1/L2 recipes and logging/incremental/VIZ controls are covered; continue auxiliary audit.
- [x] Missing Level 1 channel calibration follows only explicit configured policy.
- [x] Unknown PC saturation remains explicit `not_characterized`.
- [x] Numerical dead-time clipping is not treated as physical detector saturation.
- [x] Level 2 does not accept uncharacterized PC saturation as known-clear input; temporary guarded operation remains distinguishable from a characterized detector limit.
- [x] Invalid Rayleigh/gluing search domains fail rather than substitute scientific domains.
- [x] Missing external atmosphere follows only configured source priority; reaching USSA76 after external-source failure is an explicit configured fallback, not an execution error.
- [x] Missing surface weather follows only explicit `nan|fail` policy.
- [x] Invalid active Licel `BinW`, ADC bits, or DAQ range invalidates input instead of applying invented acquisition defaults.
- [ ] Optional diagnostic failure must never silently change a scientific algorithm; continue audit beyond current PBL/tropopause paths.

## H. Provenance / FAIR

- [x] Human-readable CalVer software release persists in products (`2026.9`); Git commit hashes are deliberately not public scientific metadata.
- [ ] Persist/version every scientific algorithm implementation separately from package release. Fernald/molecular implementation versions already exist; extend deliberately.
- [x] Embed exact processing YAML as NetCDF string variable `processing_configuration_yaml` with source filename and `application/yaml` metadata.
- [x] Embed exact station YAML as NetCDF string variable `station_configuration_yaml` with source filename and `application/yaml` metadata.
- [x] Persist station profile ID and calibration ID.
- [ ] Decide and implement readable immutable input manifest; do not add cryptic public hashes without a concrete integrity need.
- [x] Persist Level 2 MC random seed/iterations.
- [x] Persist atmosphere source, datetime/time delta, fallback fraction, source-priority attempts, and resolved station geometry.
- [x] Persist tropopause source type (`radiosonde`, `era5`, or unavailable) alongside CPT/LRT values.
- [x] Persist neutral legacy Level 1 calibration fact.
- [x] Persist readable LR source; do not invent paper/DOI before one exists.
- [x] Quarantine sidecars retain SHA-256 because integrity verification is appropriate there.
- [x] Regenerated products remove legacy public `processing_config_sha256` / `station_config_sha256` attrs.
- [ ] If the provisional PC guard survives beyond the temporary characterization phase, persist/version its assurance policy and threshold explicitly rather than relying only on package version and audit log.

## I. Tests / architecture guardrails

- [x] Broad config tests migrated away from loader-injected station/hardware aliases; tests now validate strict ownership/resolvers.
- [x] Station calibration/profile resolution tested across historical eras.
- [x] Missing Level 1 calibration explicit-policy tests.
- [x] LR month/uncertainty and station-LR ownership tests.
- [x] Invalid gluing/Rayleigh-domain tests.
- [x] Missing station timezone/coordinates/geometry strict-accessor tests.
- [x] Saturation characterization and clipping-vs-saturation tests.
- [x] Provisional PC dead-time guard and post-gluing single-channel fallback regression tests.
- [x] Strict Licel BinW/ADC/DAQ tests.
- [x] Level 0 writer tests prove native range/background ownership and readable provenance.
- [x] Quarantine layout/sidecar/collision/read-only discovery tests.
- [x] Logging tests cover contextual columns, console-vs-file verbosity, one-row cleanup, handler ownership, and console-only warning deduplication.
- [x] Visualization strict-config and LIRACOS incremental tests.
- [x] FAIR provenance/YAML-embedding tests.
- [x] CLI option, CalVer, simplified operational-result/exit-code tests.
- [x] PBL regression verifies operator INFO reports computed mean/coverage while search settings stay DEBUG.
- [x] ERA5 regression pins Client to CDS even when environment points to ADS and verifies multiline transport errors collapse to one warning line.
- [x] ERA5 tropopause regression verifies Level 1 reuses the existing CPT/LRT kernel rather than introducing a second implementation.
- [ ] Add architectural static guard against `config.get(..., semantic_literal_default)` outside config layer.
- [ ] Add repository-wide regression asserting productive config paths contain no undeclared semantic defaults.
- [ ] Full test suite after each coherent batch. No CI/status checks are attached to this branch and the full suite cannot be claimed green from this environment.

## J. FAIR / release / repository follow-up

- [ ] Add repository `LICENSE` after confirming institutional licensing choice.
- [x] Runtime/package/CITATION release version synchronized at CalVer `2026.9`.
- [ ] Restore/create documentation referenced by README.
- [ ] Add CI checks and branch protection.
- [ ] Audit NetCDF semantic metadata against current CF conventions.
- [ ] Add archived release/DOI workflow and changelog.
- [ ] External real-data validation against independent/reference processing chain.

## Current high-value technical debt

- [ ] Characterize physical PC saturation for the operational SPU detector settings (AN/PC overlap and preferably controlled optical attenuation), then replace the Level 2 corrected-rate proxy with a raw-rate Level 1 mask and traceable `max_rate_mhz` calibration.
- [ ] Remove remaining generic `physics.vertical_resolution_m` config/schema residue.
- [ ] Remove low-level Level 2 compatibility defaults and duplicate `_retrieval_impl.py` paths.
- [ ] Version/document gluing scoring constants.
- [ ] Decide readable input manifest policy.
- [ ] Finish repository-wide semantic-default static guard.
- [ ] Add CI so the full suite becomes an enforced branch contract.

## Implementation log

### 2026-09-09 — initial strict-config tranches

- Introduced fail-fast Level 2 recipe, explicit cloud policy, strict LR completeness, station calibration sets, strict Level 1 calibration/background/dead-time/PBL configuration, and clipping-vs-saturation separation.

### 2026-09-11 — atmosphere, L0 acquisition, caches, station LR, Licel metadata

- Added explicit atmosphere source priority/radiosonde/ERA5 policy and historical station-altitude resolution.
- Added explicit historical neutral-calibration policy and persisted its use.
- Added strict L0 acquisition/weather/discovery policy, removed coordinate/weather/SCC/range/ADC invented defaults, standardized `.cache/`, and added auditable quarantine.
- Moved SPU lidar-ratio climatology into station metadata without numerical changes.

### 2026-09-11 — contextual logging, strict domains, FAIR/CLI/CalVer

- Added contextual `pipeline/save_id/stage` logs with INFO console and DEBUG audit file.
- Removed gluing-domain widening and arbitrary Rayleigh last-bin substitution.
- Added strict visualization resolver.
- Simplified operational status to `OK/SKIPPED/ERROR` and shell exit policy 0/1/2.
- Added common CLI `--input/--force/--version` surface.
- Adopted CalVer `2026.9`.
- Replaced public config hashes with readable identity + exact embedded YAML provenance.
- Added station pointing geometry, L2 MC seed/iterations, and readable LR provenance.

### 2026-09-14 — compatibility cleanup, Level 0 regressions, ERA5/CDS and operator-log polish

- Fixed Level 0 YAML-provenance VLEN writing by storing NetCDF strings with an indexed dimension; added idempotent provenance regression coverage.
- Prevented historical weather/radiosonde JSON caches from entering raw Licel discovery and made incomplete-provenance outputs stale for incremental reprocessing.
- Removed loader-created station/site/hardware compatibility views and old execution-status aliases; migrated broad orchestration/config tests to current contracts.
- Routed ERA5 explicitly to the Climate Data Store API so an ADS-configured machine cannot request ERA5 from the wrong service; raised optional `cdsapi` baseline to 0.7.7 and added endpoint regression tests.
- Unified console/audit rows, removed legacy arrows and multiline console breakage, deduplicated repeated calibration/saturation warnings only on console, and removed duplicate LIBIDS summary output.
- Changed PBL operator logging from search-window chatter to computed mean height + valid count. CPT/LRT now reuse the same thermal kernel for radiosonde and ERA5; USSA76 remains explicitly unavailable as a tropopause source.
- Full repository suite is still not claimed: this branch has no CI/status checks and this environment cannot execute the complete project test matrix.

### 2026-09-14 — provisional PC guard and Level 2 fallback recovery

- Kept SPU photon-counting physical saturation explicitly `not_characterized`; no detector `max_rate_mhz` was invented.
- Added a temporary 10% dead-time-occupancy guard for uncharacterized PC channels when Level 1 correction succeeded and traceable positive station dead-time is available. The current implementation derives an observed-rate proxy by inverting the non-paralyzable correction on the Level 1 corrected block, so it is an operational guard rather than detector characterization.
- Preserved the existing gluing behavior that ignores PC saturation masks in bins supplied by the analog source, allowing guarded PC use only where PC contributes to the glued profile.
- Added post-gluing retrieval-input recovery: if numerical gluing succeeds but scientific input QA rejects the result, configured single-channel candidates are evaluated blockwise and a valid AN/PC fallback can replace the rejected glued block.
- Added retrieval-input rejection summaries and dedicated regression tests for guarded PC use and analog fallback.
- Physical saturation characterization remains high-priority technical debt: move the guard to raw observed PC rate in Level 1 and determine a traceable limit from AN/PC linearity and preferably controlled optical attenuation before declaring `status: characterized`.
- Targeted modified files compile in this environment; the full repository suite is still not claimed because this branch has no attached CI/status checks and the complete dependency/test matrix cannot be executed here.
