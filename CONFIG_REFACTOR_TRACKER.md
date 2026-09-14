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

- [x] Remove the generic `physics` section completely. The final `physics.vertical_resolution_m` repository residue is gone; productive acquisition/range geometry comes from actual Licel/station contracts.
- [x] Keep station/site/instrument identity in `station.yaml`; productive loader no longer duplicates station metadata into `site`, `radiosonde`, `physics.channels`, or `hardware.name_to_id` compatibility views.
- [x] Stop rebuilding `physics.channels` from station data.
- [x] Stop rebuilding `hardware.name_to_id` from station data.
- [x] `normalize_config` no longer injects legacy aliases.
- [x] Productive Level 1 rejects legacy positional channel-correction lists and consumes named station calibration.
- [x] Retire the legacy global/minimum-schema validator in favor of stage-specific strict validation; obsolete `milgrau/config/schema.py` has been removed.
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
- [x] Rayleigh altitude bounds are a search interval: pre-QA requires at least one Rayleigh-sized viable window and no longer requires the complete 5–25 km band to remain finite/positive.
- [x] Productive elastic aerosol inversion uses a high-reference backward Klett–Fernald branch; forward/two-sided branches remain research diagnostics and are not a productive success requirement.
- [x] SPU productive Rayleigh width is explicitly `ref_window_m: 1000.0`; it resolves to 133 bins on the current 7.5 m common grid, preserving the validated ~1 km window while making the scientific width grid-independent.
- [x] Rayleigh reference-window width is expressed in physical units and converted from the actual uniform Level 1 altitude grid before numerical fitting/QA.
- [ ] Remove remaining low-level/public gluing defaults retained for compatibility.
- [x] Invalid gluing search domains fail; they are never widened to the full profile.
- [ ] Document/version gluing selection-score weights as explicit algorithm constants.
- [x] Productive Rayleigh-reference config is complete and strict; no arbitrary last-bin reference.
- [x] Cloud-screening enabled/disabled state and all enabled parameters are explicit.
- [ ] Integrate a validated cloud/layer mask into reference-window QA; the current preliminary detector remains disabled rather than being promoted without validation.
- [ ] Remove duplicate/dead compatibility implementations in `_retrieval_impl.py`; public strict boundary is canonical but maintenance debt remains.
- [ ] Later migration: expose gluing spatial windows/search bounds in physical units rather than bins/indices.

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
- [x] Incremental reuse of published NetCDF products requires current readable MILGRAU provenance in addition to timestamps and product-contract integrity; this policy is centralized for L0/L1/L2.
- [x] Visualization output format/DPI/altitude ranges/channels/smoothing/gap threshold/colormap/missing-data color are strict config.
- [x] Configured quicklook smoothing drives side/global profiles; no hidden gap fallback.
- [x] Display-only style constants remain code constants unless intentionally promoted to theme configuration.

## G. Scientific failure semantics

- [ ] Missing required config must fail before processing starts for every stage globally. Productive L0/L1/L2 recipes and logging/incremental/VIZ controls are covered; continue auxiliary audit.
- [x] Missing Level 1 channel calibration follows only explicit configured policy.
- [x] Unknown PC saturation remains explicit `not_characterized`.
- [x] Numerical dead-time clipping is not treated as physical detector saturation.
- [x] Level 2 does not accept uncharacterized PC saturation as known-clear input; temporary guarded operation remains distinguishable from a characterized detector limit.
- [x] Background-subtracted NaN/non-positive far-range samples are not filled or clipped and do not invalidate the complete molecular search band when another configured Rayleigh window remains viable.
- [x] Backward KFS still fails a block when invalid samples interrupt the physically sampled branch below the selected reference; outer bins outside the requested branch remain NaN.
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
- [x] Incremental NetCDF reuse rejects products missing the current readable provenance schema, forcing regeneration instead of silently carrying legacy/incomplete FAIR metadata forward.
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
- [x] Rayleigh-search regressions cover bin-shift edge NaNs, non-positive samples elsewhere in the search band, absence of any viable window, PC saturation inside/outside candidate windows, and backward KFS with an invalid upper tail.
- [x] Backward aggregation regression proves a valid backward branch produces the optical product without requiring an unrequested forward branch.
- [x] Strict Level 2 config regression pins productive `kfs_mode: backward` and rejects `two_sided` as a productive contract.
- [x] Dataset metadata regression pins a FAIR-readable backward KFS description and rejects the obsolete two-sided descriptor.
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
- [x] Physical Rayleigh-window regression proves `1000 m` resolves to 133 bins on the current 7.5 m grid and adapts to other uniform vertical grids.
- [x] Incremental regression proves NetCDF reuse requires current readable provenance even when timestamp and structural integrity checks otherwise pass.
- [ ] Add architectural static guard against `config.get(..., semantic_literal_default)` outside config layer.
- [ ] Add repository-wide regression asserting productive config paths contain no undeclared semantic defaults.
- [ ] Full test suite after each coherent batch. No CI/status checks are attached to this branch and the full suite cannot be claimed green from this environment.

## J. FAIR / release / repository follow-up

- [ ] Add repository `LICENSE` after confirming institutional licensing choice.
- [x] Runtime/package/CITATION release version synchronized at CalVer `2026.9`.
- [x] Replace the stale README that referenced removed config structures, a planned-only Level 2, radiosonde-only tropopause behavior, old status policy, and documentation pages that did not exist.
- [ ] Build the scientific documentation set tracked in Section K and make README links point only to reviewed/current pages.
- [ ] Add CI checks and branch protection.
- [ ] Audit NetCDF semantic metadata against current CF conventions.
- [ ] Add archived release/DOI workflow and changelog.
- [ ] External real-data validation against independent/reference processing chain.

## K. Scientific documentation and data-product reference

The documentation goal is broader than an installation README. The repository should let a scientist reconstruct what was measured, which assumptions were applied, how each product variable was generated, and which limitations affect interpretation without reading the Python implementation first.

### K1. README as scientific entry point

- [x] Rewrite `README.md` around the actual L0 → L1 → L2 scientific data flow rather than package architecture.
- [x] Document installation, optional ERA5/explorer extras, primary CLI usage, selectors, `--force`, and LEBEAR `--time-window`.
- [x] Explain the ownership split `config.yaml = processing recipe` versus `station.yaml = observational/instrumental truth`.
- [x] Document current repository configuration values with physical units/meaning instead of showing obsolete `physics`/`hardware` examples.
- [x] Describe historical station-profile resolution and its effect on station altitude, calibration and SCC mapping.
- [x] Document current L0/L1/L2 input/output naming and canonical processed-data hierarchy.
- [x] Provide a current variable-family reference for Level 0, Level 1 and Level 2 products, including flag meanings and multispectral completeness.
- [x] Document productive scientific choices: native-grid L1 corrections, PBL reference policy, atmosphere priority, CPT/LRT sources, 1-km Rayleigh window, origin-constrained molecular scaling, backward KFS, `R_ref=1`, station LR climatology and partial Monte Carlo scope.
- [x] Make current limitations prominent: uncharacterized PC saturation, disabled cloud screening, diagnostic-only SNR, ERA5 tropopause vertical-resolution caveat and incomplete total uncertainty budget.
- [x] Document readable FAIR YAML provenance and current Zenodo citation.

### K2. Processing/configuration reference pages

- [ ] Create `docs/processing_levels.md` with the end-to-end scientific lineage from raw Licel records through Level 0, Level 1 and Level 2, including which stage owns each transformation.
- [ ] Create `docs/configuration.md` as the authoritative `config.yaml` dictionary: full path, type, unit, allowed domain, current value, scientific effect, output metadata/variable affected, and reprocessing consequence when changed.
- [ ] Create `docs/station_catalog.md` as the authoritative `station.yaml` reference: station/site fields, timezone, historical altitude/instrument profiles, calibration sets, saturation status, SCC mapping, radiosonde identity and lidar-ratio climatology ownership.
- [ ] Explain station-profile date resolution with a worked historical example spanning at least two SPU instrument eras.
- [ ] Document which settings are scientific assumptions versus acquisition/instrument facts versus visualization-only settings.
- [ ] Document the station-config selection precedence (`explicit argument → MILGRAU_STATION_CONFIG → config.yaml`) without presenting it as a scientific source override.

### K3. NetCDF product dictionaries

- [ ] Create `docs/level0_product.md` with every dimension, variable, dtype, unit/flag, optionality rule and relevant global attribute for base and SCC-ready Level 0 products.
- [ ] Document Level 0 acquisition QA, rejected-profile semantics, dark-current association, native `BinW`, DAQ range, weather missingness and SCC readiness alongside the variable dictionary.
- [ ] Create `docs/level1_product.md` with every Level 1 signal, uncertainty, correction diagnostic, PBL variable, atmosphere variable, tropopause attribute and source/provenance field.
- [ ] Include the exact L1 correction order and uncertainty propagation equations, explicitly distinguishing numerical dead-time clipping from physical detector saturation.
- [ ] Create `docs/level2_product.md` with every Level 2 dimension/coordinate, molecular field, selected/glued signal, optical field, Rayleigh/KFS/gluing/source-selection diagnostic, completeness field and global method attribute.
- [ ] Document every integer flag/code with stable numeric meaning, including `signal_source_flag`, `retrieval_input_invalid_reason`, KFS branch flags, merge-source flags and failed-wavelength stage/code.
- [ ] Explicitly document current aggregate aliases (`aerosol_backscatter[_mean]`, `aerosol_extinction[_mean]`) and decide a future deprecation/migration path rather than leaving duplicate semantics implicit.
- [ ] State missing-value/NaN/sentinel semantics for every product family; remove or clearly justify legacy scalar `-999` conventions where CF-compatible missingness can be used instead.

### K4. Scientific methods and equations

- [ ] Create `docs/scientific_methods.md` with notation and equations for PC normalization, Poisson/dark-current/background uncertainty, non-paralyzable dead-time correction, bin shift, background subtraction and range correction.
- [ ] Document PBL gradient method and its limitations as a simple elastic-RCS boundary-layer diagnostic rather than a universal PBL truth.
- [ ] Consolidate atmosphere mapping methodology: ASL/AGL relation, historical station altitude, linear temperature interpolation, log-pressure interpolation and USSA76 outside-coverage extension.
- [ ] Update `docs/atmospheric_profiles.md` to the current nested `level1.atmosphere` configuration, source-priority contract, CDS endpoint behavior and ERA5-derived CPT/LRT support.
- [ ] Document CPT/LRT equations/criteria and explicitly distinguish radiosonde-derived versus ERA5-derived vertical information/precision.
- [ ] Document the molecular/Rayleigh equations and physical constants used by the Bucholtz-style implementation, including `8π/3 sr` molecular lidar ratio and transmission convention.
- [ ] Document the automatic Rayleigh-reference objective/QA in physical units, including how `ref_window_m` is converted on the Level 1 grid.
- [ ] Document analog/PC gluing equations, regression direction, fade weights, uncertainty propagation, candidate acceptance criteria and current score definition.
- [ ] Version/document the gluing selection-score weights before presenting them as a stable algorithm identity.
- [ ] Document the generalized Fernald/Klett–Sasano equation, exact reference-bin boundary, productive backward integration, SCI-001 backward-sign correction and research-only forward/two-sided modes.
- [ ] Document the relationship `aerosol_ref_fraction = R_ref - 1` and the current pure-molecular `R_ref=1` assumption.
- [ ] Document the partial MC ensemble exactly: perturbed quantities, distributions, lower LR bound, seed/iterations, aggregation and which uncertainty sources remain outside the budget.

### K5. FAIR provenance, QA and interpretation

- [ ] Create `docs/provenance_fair.md` covering exact embedded YAML, software release, station/calibration identity, external atmosphere provenance, LR provenance, incremental reuse semantics and the future readable input-manifest decision.
- [ ] Create `docs/quality_flags.md` as a scientist-facing interpretation guide for correction status, saturation characterization, Rayleigh/KFS validity, wavelength completeness and failure diagnostics.
- [ ] Create `docs/known_limitations.md` so provisional policies are visible outside the tracker: PC saturation, cloud mask, SNR threshold, station LR dependence, `R_ref=1`, ERA5 tropopause resolution, partial uncertainty budget and any schema metadata gaps.
- [ ] Create `docs/validation.md` distinguishing unit/synthetic numerical validation, internal product-contract tests and still-pending independent real-data validation.
- [ ] Add worked examples showing how to decide whether a Level 1 or Level 2 profile is scientifically usable from NetCDF flags alone.
- [ ] Add worked examples for complete versus partial multispectral Level 2 products and explain why partial products are not incrementally final.

### K6. Metadata/schema audit driven by documentation

- [ ] Audit every documented NetCDF physical variable for explicit `units`, `long_name`/description, dimensions and missing-value semantics before freezing the product-reference pages.
- [ ] Audit current Level 2 descriptions against the productive backward-only policy; remove residual text that still says both/two-sided branches are required for productive success.
- [ ] Reconcile versioned algorithm metadata so `elastic_inversion_algorithm_metadata()` and product-level `KFS_Mode` cannot disagree about productive integration mode.
- [ ] Audit Level 2 backscatter/extinction units and add explicit CF-readable attributes where currently implied by equations but not stored on the variable.
- [ ] Audit CF standard-name applicability conservatively; do not assign a CF `standard_name` unless the quantity exactly matches the convention definition.
- [ ] Decide whether global scalar diagnostics such as CPT/LRT should remain attrs or become typed variables with explicit missing-value metadata in a future schema version.
- [ ] Add a product-schema/version field if needed so documentation can identify intentional NetCDF contract changes independently of package CalVer.

### K7. Scientific references, examples and maintenance

- [ ] Build a verified bibliography for the exact methods actually implemented: Licel/SCC conventions where citable, Bucholtz/Rayleigh method, WMO thermal tropopause, Klett/Fernald/Sasano elastic inversion, ERA5 and any network-method comparisons used to justify productive choices.
- [ ] Cite primary/authoritative sources rather than copying thresholds or prose from secondary implementations.
- [ ] Add small xarray examples for opening L0/L1/L2 files, inspecting embedded YAML provenance, filtering valid Level 2 blocks and checking source/completeness flags.
- [ ] Add one end-to-end scientific worked example for a real SPU measurement once an anonymization/public-data policy is settled.
- [ ] Keep README concise enough to remain an entry point after dedicated docs exist; move exhaustive tables to `docs/` while retaining the scientific overview and links.
- [ ] Add a documentation review item to each scientific/schema change: equations, config reference, variable dictionary, provenance and reprocessing note must be updated together when affected.
- [ ] Add a lightweight regression/link check so README/docs cannot reference deleted pages or obsolete public config keys.

## Current high-value technical debt

- [ ] Characterize physical PC saturation for the operational SPU detector settings (AN/PC overlap and preferably controlled optical attenuation), then replace the Level 2 corrected-rate proxy with a raw-rate Level 1 mask and traceable `max_rate_mhz` calibration.
- [ ] Move the productive backward-KFS policy from the temporary Level 2 package-init adapter into the canonical `config.py`/decomposed retrieval implementation, removing the historical two-sided text path.
- [ ] Remove the temporary Level 2 package-init retrieval-QA/aggregation shims when `_retrieval_impl.py` is decomposed; keep Rayleigh-search QA and backward aggregation as explicit retrieval components.
- [ ] Validate an operational cloud/layer mask against SPU data before allowing it to exclude Rayleigh-reference windows.
- [ ] Characterize MILGRAU propagated-error SNR against SPU data before adopting a hard Rayleigh-window SNR threshold; do not copy PollyNET's threshold by analogy alone.
- [ ] Remove low-level Level 2 compatibility defaults and duplicate `_retrieval_impl.py` paths.
- [ ] Version/document gluing scoring constants.
- [ ] Decide readable input manifest policy.
- [ ] Finish repository-wide semantic-default static guard.
- [ ] Add CI so the full suite becomes an enforced branch contract.
- [ ] Complete Section K documentation and use its metadata audit to close product-description/unit inconsistencies before treating the scientific schema as documented/frozen.

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

### 2026-09-14 — contiguous Level 2 retrieval support

- Aligned pre-inversion retrieval-input QA with the existing KFS branch semantics: invalid bins may terminate only the outer edge of the physically supported profile instead of invalidating every block merely because Level 1 bin shifting produced edge NaNs.
- Defined the accepted input as one contiguous finite-positive signal / finite non-negative uncertainty run containing the complete configured Rayleigh-reference interval; no signal or uncertainty values are filled, extrapolated, or interpolated by this QA.
- Continued to reject NaNs/non-positive signal/invalid uncertainty inside the Rayleigh interval and any disjoint usable island separated by an invalid internal bin.
- Saturation QA is evaluated only on the physically supported run, so a saturation flag outside an already-invalid edge cannot invalidate otherwise usable PC support.
- Added regressions for edge NaNs, non-positive edge truncation, internal gaps, invalid Rayleigh support, saturation outside support, package integration, and KFS preservation of NaN edge bins.
- The supported-domain evaluator is installed through a small package-init compatibility shim while `_retrieval_impl.py` remains monolithic; removing that shim is tracked with the broader Level 2 decomposition debt.
- Targeted new Python files pass syntax validation in this environment; the full repository suite is still not claimed because this branch has no attached CI/status checks and the complete dependency/test matrix cannot be executed here.

### 2026-09-14 — SCC/LPP-informed Rayleigh reference and backward KFS

- Re-reviewed the elastic retrieval against EARLINET SCC/ELDA, the LPP SPU configuration/implementation, and the automated PollyNET Rayleigh-fit approach.
- Changed the productive elastic aerosol contract from two-sided to high-reference backward Klett–Fernald. The numerical forward branch remains available for research, but forward validity no longer suppresses a scientifically valid backward product.
- Reinterpreted `ref_alt_min_m`/`ref_alt_max_m` correctly as automatic Rayleigh-window search bounds. Background-subtracted non-positive samples elsewhere in 5–25 km no longer reject the full block when another configured window remains viable; no values are clipped, filled, or interpolated.
- Reduced the SPU Rayleigh window from 667 bins (~5 km) to 133 bins (~1 km on the current 7.5 m common grid), consistent in scale with SCC/Polly-style moving reference windows while retaining the broad 5–25 km search for site/season flexibility.
- Kept the final molecular calibration constrained through the origin because Level 1 already removes background; the free intercept remains diagnostic. Kept `aerosol_ref_fraction: 0.0` (`R_ref = 1`) as the explicit pure-molecular boundary assumption, consistent with the current LPP SPU configuration.
- Did not copy PollyNET's hard SNR threshold into MILGRAU: propagated-error/SNR behavior must first be characterized for SPU. SNR remains diagnostic only.
- Did not enable the preliminary MILGRAU cloud detector. Cloud/layer exclusion is scientifically desirable, as in SCC/LPP/Polly workflows, but requires validation on SPU before becoming a productive gate.
- Added operator-visible aggregate Rayleigh diagnostics (selected altitude/window, valid fraction, slope, variance, backward-valid block count), backward-only optical aggregation, and regression tests for the new search-window and branch semantics.
- Productive policy is currently installed at the Level 2 boundary while `_retrieval_impl.py` and the historical `config.py` mode text remain monolithic compatibility debt; moving the rule into the canonical decomposed config/retrieval implementation is explicitly tracked.
- Targeted new/modified Python files were syntax-validated during preparation; the full repository suite is still not claimed because this branch has no attached CI/status checks and the complete dependency/test matrix cannot be executed here.

### 2026-09-14 — backward KFS metadata consistency

- Real-data retrieval reached scientifically valid Rayleigh/backward-KFS results for both 355 and 532 nm (5/5 blocks each) but failed during dataset assembly because the legacy `KFS_Mode_Description` helper still rejected `backward`.
- Added the FAIR-readable backward-KFS description to the same Level 2 scientific-policy boundary used for the productive mode and installed it before dataset assembly.
- Added a strict-config regression proving `backward` metadata is accepted and the obsolete two-sided descriptor is rejected.
- No retrieval equations, Rayleigh selection thresholds, or detector assumptions changed in this micro-fix; it only removes a contradictory post-retrieval metadata gate.

### 2026-09-14 — schema retirement, incremental FAIR guard and physical Rayleigh width

- Removed the obsolete global `milgrau/config/schema.py` validator and the final top-level `physics.vertical_resolution_m` configuration residue; stage-specific strict resolvers remain the productive configuration authority.
- Centralized incremental FAIR integrity: every published `.nc` output now requires the current embedded MILGRAU provenance schema before timestamp/contract-based reuse, so incomplete L1/L2 products are regenerated just like incomplete L0 products.
- Replaced productive `inversion.molecular_fit.ref_window_bins` with `ref_window_m: 1000.0` and convert that width on the actual uniform Level 1 altitude grid at runtime.
- Preserved the current SPU Rayleigh method numerically: 1000 m on the 7.5 m common grid resolves to 133 bins, while coarser/finer grids retain approximately the same physical window instead of a fixed bin count.
- Added regressions for physical-width conversion, rejection of the legacy public `ref_window_bins` recipe, removal of the `physics` section, and provenance-gated NetCDF incremental reuse.
- No Rayleigh QA thresholds, search bounds, calibration equation, KFS equation, lidar-ratio assumption, or detector policy changed in this tranche.
- The full repository suite is still not claimed because this branch has no attached CI/status checks and the complete dependency/test matrix cannot be executed here.

### 2026-09-14 — scientific README and documentation roadmap

- Replaced the stale README with a current scientific/data-product reference for the `new-architecture` branch: installation, complete L0→L1→L2 lineage, configuration/station ownership, historical station context, scientific methods, NetCDF variables/flags, FAIR provenance, visualization and known limitations.
- Documented Level 2 as the current productive backward Klett–Fernald retrieval rather than a planned feature and made the elastic-extinction LR dependence, `R_ref=1` boundary, partial Monte Carlo scope and signal-source/Rayleigh QA explicit.
- Added Section K as the detailed roadmap for splitting the README into reviewed scientific method, configuration, station, product-schema, provenance, QA, validation and bibliography documents.
- Documentation review is now coupled to a metadata audit so prose cannot silently normalize current schema inconsistencies; residual two-sided wording, implied Level 2 units and duplicate aggregate aliases are tracked for explicit cleanup before a documented schema freeze.
- No processing equation, threshold, scientific configuration value or NetCDF numerical result changed in this documentation tranche.
