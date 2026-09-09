# MILGRAU configuration refactor tracker

Branch: `new-architecture`

This file is the source of truth for the strict-configuration refactor. Update it in every implementation batch before moving to the next batch.

## Goal

Make scientific and instrumental decisions explicit and auditable:

- `config.yaml` = processing/scientific recipe.
- `station.yaml` = observational reality, hardware history, calibration and SCC mapping.
- Python code = physical constants, equations, file-format invariants and implementation details.
- Missing required scientific/instrumental configuration must fail early; production code must not invent semantic defaults.

## Global rules

- [ ] No silent scientific defaults in production paths.
- [ ] No silent instrumental defaults in production paths.
- [ ] No station-specific coordinates/timezone/IDs embedded in Python fallbacks.
- [ ] No neutral channel correction fallback for an unknown channel.
- [ ] No fake SCC channel IDs.
- [ ] No algorithmic fallback that silently widens/changes a configured scientific domain.
- [x] Optional Level 2 cloud-screening behavior must be explicitly enabled/disabled in YAML.
- [x] Rayleigh molecular lidar ratio remains a versioned physical/method constant in Python rather than a user-editable YAML setting.
- [ ] Each produced NetCDF records the resolved station profile/calibration and processing configuration provenance.
- [ ] Tests cover every required-key failure and every explicit disabled/unavailable policy.

## A. Configuration ownership and schema

- [ ] Replace generic `physics` ownership with stage-oriented processing configuration.
- [ ] Keep station/site/instrument metadata in `station.yaml` only.
- [ ] Stop rebuilding `physics.channels` from station data. **Temporary compatibility view remains while Level 1 consumers are migrated.**
- [ ] Stop rebuilding `hardware.name_to_id` as a compatibility structure.
- [ ] Remove legacy aliases injected by `normalize_config`.
- [ ] Remove legacy positional channel correction lists.
- [ ] Replace `validate_config_minimum` philosophy with stage-specific strict validation.
- [ ] Validate unknown keys with full paths.
- [ ] Add typed/resolved configuration objects or equivalent strict accessors so scientific modules do not consume raw config mappings directly.

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
- [ ] Require laser-shot tolerance.
- [ ] Make Licel header timestamp jitter threshold explicit.
- [ ] Require finite dark-current association maximum or another explicit policy.
- [ ] Remove timezone fallback to `America/Sao_Paulo`.
- [ ] Remove hardcoded São Paulo latitude/longitude fallback.
- [ ] Remove fallback surface temperature 25 C / pressure 940 hPa.
- [ ] Use explicit missing-surface-weather policy (`nan` or `fail`).
- [ ] Remove `DEFAULT_CHANNEL_ID = 9999`; missing SCC channel ID makes SCC export unavailable/invalid.
- [ ] Remove global 7.5 m range-resolution fallback; prefer Licel metadata and explicit historical station override only when justified.
- [ ] Review Licel parser defaults for ADC bits/range and convert them to explicit format invariants or station-profile overrides.

## D. Level 1 strict configuration

- [x] Photon-counting Poisson uncertainty uses observed counts before dark subtraction (SCI-003 completed before this refactor).
- [x] Canonical atmosphere is materialized in Level 1 with radiosonde -> ERA5 -> USSA76 provenance (completed before this refactor).
- [ ] Remove configurable speed of light; keep exact SI constant in code.
- [ ] Require Level 1 background window.
- [ ] Require dead-time numerical clipping denominator policy.
- [ ] Separate numerical dead-time clipping from detector saturation.
- [ ] Require explicit PBL reference channel.
- [ ] Require PBL search interval and smoothing settings.
- [ ] Remove PBL fallback to first available channel.
- [ ] Move atmospheric source/fallback policy to `config.yaml`.
- [ ] Make radiosonde temporal-selection policy explicit (synoptic hours / nearest / max delta).
- [ ] Remove hardcoded radiosonde station ID fallback.
- [ ] Require complete ERA5 configuration when ERA5 is in the source policy.
- [ ] Do not replace invalid ERA5 pressure-level config with an internal list.

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

- [ ] Missing required config -> configuration error before processing starts for every stage. **Implemented for productive Level 2; other stages remain.**
- [ ] Missing channel calibration -> channel/product failure, never neutral correction.
- [x] Unknown PC saturation characterization is represented explicitly as `not_characterized` in station calibration instead of assigning an invented detector limit.
- [ ] Stop reusing numerical dead-time clipping as a detector saturation proxy.
- [ ] Invalid Rayleigh search -> reference selection failure.
- [ ] Invalid gluing search -> gluing failure.
- [ ] Missing external atmosphere follows only the explicitly configured source policy.
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
- [ ] Persist atmosphere source, source datetime/time delta and fallback fraction.
- [ ] Persist LR source/provenance used by each retrieval.

## I. Tests / architecture guardrails

- [ ] Rewrite broad config tests for strict schema; remove tests that freeze legacy aliases.
- [x] Add station calibration/profile resolution tests across historical eras.
- [ ] Add failure tests for missing channel calibration at the processing consumer.
- [x] Add failure tests for missing LR month/uncertainty.
- [ ] Add failure tests for invalid gluing/Rayleigh domains. **Config-level gluing interval validation is tested; algorithmic failure tests remain.**
- [ ] Add failure tests for missing station timezone/coordinates/altitude.
- [x] Add explicit-policy tests for unavailable saturation characterization.
- [ ] Add architectural guard against `config.get(..., semantic_literal_default)` outside the config layer.
- [ ] Add regression test ensuring production config contains no undeclared semantic defaults.
- [ ] Run full test suite after each coherent implementation batch. **No CI is currently attached to the branch; isolated strict-L2 tests passed 19/19.**

## J. Follow-up FAIR/release work (not part of the strict-config code batch unless touched by necessity)

- [ ] Add repository software `LICENSE` after confirming institutional licensing choice.
- [ ] Unify package version and `CITATION.cff` release version.
- [ ] Restore/create documentation referenced by README.
- [ ] Add CI checks and branch protection.
- [ ] Audit NetCDF semantic metadata against current CF conventions.
- [ ] Add archived releases/DOI workflow and changelog.
- [ ] External real-data validation against an independent/reference processing chain.

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
- A temporary `physics.channels` compatibility view remains so Level 1 behavior is not broken before its consumer migration; this is explicitly not considered the target architecture.
- Added station tests for calibration resolution, unknown calibration references and saturation characterization semantics.
- GitHub currently reports no CI/status checks on the branch, so the full repository suite has not been claimed as executed.
