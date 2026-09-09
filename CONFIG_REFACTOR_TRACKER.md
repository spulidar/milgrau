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
- [ ] Optional behavior must be explicitly enabled/disabled in YAML.
- [ ] Physical/mathematical constants remain versioned in Python, not user-editable YAML.
- [ ] Each produced NetCDF records the resolved station profile/calibration and processing configuration provenance.
- [ ] Tests cover every required-key failure and every explicit disabled/unavailable policy.

## A. Configuration ownership and schema

- [ ] Replace generic `physics` ownership with stage-oriented processing configuration.
- [ ] Keep station/site/instrument metadata in `station.yaml` only.
- [ ] Stop rebuilding `physics.channels` from station data.
- [ ] Stop rebuilding `hardware.name_to_id` as a compatibility structure.
- [ ] Remove legacy aliases injected by `normalize_config`.
- [ ] Remove legacy positional channel correction lists.
- [ ] Replace `validate_config_minimum` philosophy with stage-specific strict validation.
- [ ] Validate unknown keys with full paths.
- [ ] Add typed/resolved configuration objects or equivalent strict accessors so scientific modules do not consume raw config mappings directly.

## B. `station.yaml`: observational reality

- [ ] Keep station ID/name/institution/timezone/site coordinates in station catalog.
- [ ] Keep radiosonde station identity in station catalog; move fallback policy out.
- [ ] Rename `scc_defaults` to explicit SCC policy terminology.
- [ ] Introduce named instrument calibration sets.
- [ ] Make station profiles reference a calibration ID.
- [ ] Allow calibration to change by historical profile.
- [ ] Move `deadtime_us` to instrument calibration.
- [ ] Move `bin_shift_bins` to instrument calibration.
- [ ] Move `background_offset` to instrument calibration.
- [ ] Add explicit PC saturation characterization/status per relevant channel.
- [ ] Represent legacy ADC/range overrides in station calibration only when raw Licel metadata cannot provide them.
- [ ] Resolve and persist `station_profile_id` and `instrument_calibration_id`.

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

- [ ] Require wavelengths; remove `[532]` fallback.
- [ ] Require temporal block averaging; remove 15-minute fallback.
- [ ] Require KFS mode explicitly.
- [ ] Require Monte Carlo iterations and random seed explicitly.
- [ ] Require beta-reference uncertainty and aerosol reference fraction explicitly.
- [ ] Require minimum aerosol lidar ratio and negative-aerosol policy explicitly.
- [ ] Remove aerosol LR 60 sr fallback.
- [ ] Remove aerosol LR uncertainty 10 sr fallback.
- [ ] Require LR values for every requested wavelength/month and associated uncertainty/provenance.
- [ ] Remove molecular lidar-ratio YAML knob; keep Rayleigh molecular ratio as algorithm constant.
- [ ] Require complete gluing configuration.
- [ ] Remove gluing window/search/threshold defaults.
- [ ] Do not widen an invalid configured gluing search interval to the whole profile.
- [ ] Make gluing selection-score weights explicit configuration or documented versioned algorithm constants.
- [ ] Require complete Rayleigh-reference configuration.
- [ ] Do not select the last available bin when no valid Rayleigh window exists.
- [ ] Require explicit cloud-screening enabled/disabled state.
- [ ] Require all cloud-screening parameters when enabled.
- [ ] Make cloud baseline percentile explicit or document it as part of a versioned classifier.
- [ ] Integrate cloud contamination into reference-window QA.
- [ ] Later migration: express public gluing/Rayleigh spatial windows in physical units rather than bins/indices.

## F. IO/runtime/visualization

- [ ] Remove filesystem directory fallbacks from production config paths.
- [ ] Remove logging-level fallbacks when a loaded MILGRAU config is used.
- [ ] Remove visualization output-format/DPI/altitude-range fallbacks.
- [ ] Ensure `mean_profile_smooth_bins` from YAML is actually used everywhere intended.
- [ ] Make quicklook gap threshold explicit; no derived 10 min/3x-median fallback unless deliberately defined as an algorithmic mode.
- [ ] Keep display-only colors/style constants in code unless the project intentionally exposes them as theme configuration.

## G. Scientific failure semantics

- [ ] Missing required config -> configuration error before processing starts.
- [ ] Missing channel calibration -> channel/product failure, never neutral correction.
- [ ] Missing/unknown saturation threshold -> explicit `unavailable/not_characterized` state, never reuse dead-time numerical clipping as detector saturation.
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

- [ ] Rewrite config tests for strict schema; remove tests that freeze legacy aliases.
- [ ] Add station calibration/profile resolution tests across historical eras.
- [ ] Add failure tests for missing channel calibration.
- [ ] Add failure tests for missing LR/month/uncertainty.
- [ ] Add failure tests for invalid gluing/Rayleigh domains.
- [ ] Add failure tests for missing station timezone/coordinates/altitude.
- [ ] Add explicit-policy tests for unavailable saturation characterization.
- [ ] Add architectural guard against `config.get(..., semantic_literal_default)` outside the config layer.
- [ ] Add regression test ensuring production config contains no undeclared semantic defaults.
- [ ] Run full test suite after each coherent implementation batch.

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
