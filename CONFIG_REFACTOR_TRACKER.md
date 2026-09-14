# MILGRAU architecture, scientific quality and Level 2 roadmap

Branch: `new-architecture`

Audit baseline: `c6400a941964de73438132afdb7f3db9a5916f32` (2026-09-14)

This file is the source of truth for consolidating the current architecture before a future `fixing_l2` branch is recreated. It replaces a long list of partially overlapping refactor notes with one ordered roadmap: scientific correctness first, then code cleanup/architecture, then schema/FAIR hardening, and only then the larger high-column Level 2 redesign.

## Goal

MILGRAU should be scientifically defensible, FAIR, concise and easy to audit. Good quality does **not** mean keeping every historical path or adding abstraction for its own sake.

- `config.yaml` = processing/scientific recipe.
- `station.yaml` = observational reality, hardware history, calibration, station-derived climatology and SCC mapping.
- Python = equations, physical constants, validated runtime objects and implementation details.
- One scientific behavior must have one canonical productive implementation.
- No productive behavior may depend on import order, runtime monkey patching or hidden semantic defaults.
- Every retained module/function/class/public variable must have a clear role: productive API, internal implementation, validated research diagnostic, or an explicitly temporary compatibility path.
- Compatibility code without a named consumer and removal criterion should be removed.
- Duplicate outputs/aliases are allowed only under an explicit deprecation/migration contract.
- Large modules should be split when they mix responsibilities, but the repository should not be fragmented into trivial wrappers that add indirection without value.
- Comments/docstrings should explain scientific or architectural intent, not preserve obsolete implementation history.
- Missing scientific/instrumental configuration must fail early unless an explicit unavailable/legacy policy is selected.
- Operator logs should report outcomes; detailed search/transport/debug information belongs in the DEBUG audit log.
- No cleanup may silently change a scientific equation, threshold, calibration assumption or uncertainty model.

## Audit snapshot — 2026-09-14

### What is already in good shape

- [x] Generic top-level `physics` configuration residue and the obsolete global `milgrau/config/schema.py` validator are removed.
- [x] Productive L0/L1/L2 use stage-specific strict configuration in the main paths.
- [x] Station/site/instrument truth is owned by `station.yaml`; loader-created legacy station/hardware aliases are removed.
- [x] Level 0 acquisition paths are strict about native Licel `BinW`, ADC/DAQ metadata, weather policy, dark-current association and SCC metadata.
- [x] Level 1 materializes the canonical thermodynamic atmosphere and records source/fallback provenance.
- [x] CPT/LRT share one thermal kernel for radiosonde and ERA5; USSA76 is not invented as a tropopause source.
- [x] Level 2 supports strict AN/PC gluing, post-QA single-channel fallback and explicit retrieval-input reasons.
- [x] Unknown physical PC saturation remains `not_characterized`; the temporary dead-time occupancy policy is explicitly provisional.
- [x] Rayleigh altitude bounds are a search interval, not a requirement that the whole 5–25 km band remain positive.
- [x] Productive Rayleigh window width is physical (`ref_window_m: 1000.0`) and converted on the actual uniform altitude grid.
- [x] Real case `20251107sapm` reached 5/5 valid Rayleigh/backward-KFS blocks at 355 and 532 nm.
- [x] Published NetCDF reuse requires current readable provenance, not only timestamps.
- [x] README was rewritten around the current L0→L1→L2 scientific flow and known limitations.
- [x] Regression coverage is broad for configuration, acquisition, atmosphere, gluing, KFS, provenance, CLI and logging.

### P0 contradictions found by this audit

These are correctness/traceability issues and should be fixed before any new Level 2 retrieval architecture is built.

- [ ] `milgrau/level2/config.py` still defines the productive KFS contract as `two_sided`, while runtime policy is `backward`.
- [ ] `milgrau/scientific.py` still publishes `ELASTIC_BACKSCATTER_INTEGRATION_MODE = "two_sided"`, so versioned algorithm metadata can contradict the generated product.
- [ ] `milgrau/level2/__init__.py` changes `config.get_kfs_mode`, `kfs_mode_description`, `_evaluate_retrieval_input` and `retrieve_optical_blocks` at import time. Productive science must not depend on package-import side effects.
- [ ] `tests/test_imports.py` says core imports should be without side effects, but current Level 2 package import performs scientific monkey patches; make the test and implementation agree by removing the patches, not by weakening the test.
- [ ] `milgrau/level2/dataset.py` still describes `retrieval_success_flag` as requiring "both KFS-v2 branches" even though productive success is backward-only.
- [ ] `milgrau/level2/lebear.py` still emits a failure message saying "Rayleigh plus two-sided KFS optical retrieval".
- [ ] `_retrieval_impl.py` still contains a legacy atmosphere path using removed `site`/`physics` fallbacks and several semantic `config.get(..., default)` values even though the public boundary is strict.
- [ ] `milgrau/level2/retrieval.py` uses `from ..._retrieval_impl import *`, making the legacy monolith effectively public and obscuring which symbols are actually required.
- [ ] `QA_L2_Product_Status_*.txt` is still generated although the same product-completeness information already lives in the NetCDF/logs.
- [ ] Level 2 dataset currently stores duplicate aggregate aliases (`aerosol_backscatter_mean` and `aerosol_backscatter`, likewise extinction/error) without an explicit deprecation decision.
- [ ] `milgrau/level2/atmosphere.py` is only a compatibility import path to `milgrau.physics.atmosphere`; confirm remaining consumers and remove it if none remain after `_retrieval_impl.py` cleanup.
- [ ] No CI/status checks are attached to the branch, so the full test suite is not an enforced branch contract.

## Priority order

### P0 — fix scientific/metadata contradictions first

Low-to-medium effort, high value. Do before structural refactors so tests describe one truth.

- [ ] Move productive `backward` mode and its human-readable description into canonical `milgrau/level2/config.py`; delete the contradictory `two_sided` productive text.
- [ ] Make `milgrau/scientific.py` publish the same productive backward integration mode and update/bump the relevant scientific-method identity if the metadata contract changes.
- [ ] Fix stale backward/two-sided descriptions in `dataset.py`, `lebear.py`, README/docs and tests.
- [ ] Add one regression asserting `config`, scientific algorithm metadata and Level 2 NetCDF `KFS_Mode` all agree.
- [ ] Remove `QA_L2_Product_Status_*.txt` and add a QA-output regression proving only intended visual artifacts are generated.
- [ ] Define `retrieval_support_flag` and `retrieval_top_altitude_m` semantics in tests/documentation before adding them to the product schema; this is needed by the future high-column work.

Acceptance gate: one canonical statement of the productive KFS mode exists everywhere and no redundant QA status TXT is produced.

### P1 — remove import-order behavior, dead compatibility and monolithic duplication

This is the most important software-engineering work before recreating `fixing_l2`.

- [ ] Remove scientific monkey patches from `milgrau/level2/__init__.py`.
- [ ] Wire `evaluate_retrieval_input_supported_domain()` directly into the canonical retrieval implementation.
- [ ] Move backward-only aggregate-success logic into the canonical retrieval path; retire `backward_retrieval.py` if it becomes an adapter with no independent responsibility.
- [ ] Replace `from milgrau.level2._retrieval_impl import *` with explicit imports/exports.
- [ ] Decompose `_retrieval_impl.py` by responsibility: input preparation/selection, gluing orchestration, Rayleigh evaluation, KFS orchestration and aggregation. Keep numerical kernels (`gluing.py`, `molecular.py`, `kfs.py`) independently testable.
- [ ] Remove the legacy `_retrieval_impl.build_thermodynamic_profile()` path after proving the strict Level 1-atmosphere boundary is the only productive consumer.
- [ ] Remove or rewrite `_retrieval_impl.run_kfs_profile()` semantic defaults; productive KFS settings must come from the strict resolver.
- [ ] Remove duplicate legacy `_evaluate_retrieval_input()` once the supported-domain evaluator is canonical.
- [ ] Remove silent Rayleigh/gluing semantic defaults retained inside legacy orchestration (`fit_config.get(...)`, `gluing_config.get(...)`) when the strict resolver already guarantees those keys.
- [ ] Audit whether `milgrau/level2/atmosphere.py` has any remaining legitimate public consumer; delete it if it is only a stale compatibility alias.
- [ ] Review the public default `mode="two_sided"` in low-level KFS APIs. Productive calls must pass an explicit mode; research `forward/two_sided` behavior may remain available but must not look like the productive default.

Acceptance gate: importing `milgrau.level2` has no scientific side effects; no productive code depends on wildcard imports or duplicate legacy implementations.

### P2 — code-quality/static guardrails and concise public surface

- [ ] Build a repository-wide symbol/module inventory and classify each non-test symbol as public API, internal implementation, research diagnostic, compatibility, or unused.
- [ ] For every compatibility path, record its actual consumer and a removal criterion; remove paths with no consumer.
- [ ] Audit `__all__` exports and package-level re-exports; every public symbol must have a test and/or documented external purpose.
- [ ] Add a lightweight static-analysis gate (Ruff or equivalent) for unused imports/variables, obvious dead code and basic style; keep configuration minimal and scientifically neutral.
- [ ] Add a repository guard against `config.get(..., semantic_literal_default)` outside explicit resolver/compatibility code.
- [ ] Add a repository regression asserting productive config paths contain no undeclared scientific/instrumental defaults.
- [ ] Search for duplicate implementations of the same scientific equation/selection rule and keep one canonical version.
- [ ] Review broad `except Exception` blocks: retain them only at orchestration/diagnostic boundaries where failure classification is intentional.
- [ ] Review large mixed-responsibility files (`_retrieval_impl.py`, `dataset.py`, `viz/level2_qa.py`, explorer app) and split only where cohesion improves; do not create one-function wrapper modules without a clear boundary.
- [ ] Remove obsolete `legacy` helpers/comments after their replacement is validated; historical context belongs in Git history/docs, not in active code paths.
- [ ] Add CI running import checks, compile/static checks and the full pytest suite; branch protection can follow once CI is stable.

Acceptance gate: no known unused compatibility code, no wildcard productive imports, no scientific import side effects, and automated checks prevent them returning.

### P3 — schema, FAIR metadata and documentation hardening

- [ ] Reconcile duplicate Level 2 aggregate aliases and choose a deprecation/migration path before the next schema expansion.
- [ ] Audit every Level 2 physical variable for units, description/long_name, dimensions and NaN/missing semantics.
- [ ] Add explicit units for aerosol backscatter/extinction and uncertainty fields where currently implied rather than stored.
- [ ] Audit flag metadata for CF-compatible numeric `flag_values`/meanings.
- [ ] Add a product-schema/method-version field independent of package CalVer if needed for intentional scientific schema changes.
- [ ] Version/document gluing selection-score constants; the current score is algorithmic behavior and should not be an anonymous literal.
- [ ] Decide a readable immutable input-manifest policy.
- [ ] Create the focused documentation pages in Section K and then shorten README into an entry point rather than duplicating the full reference manual.

Acceptance gate: one NetCDF can be interpreted without reading source code and metadata never contradict the actual method.

### P4 — scientific evidence tasks that can run in parallel

- [ ] Characterize physical PC saturation for operational SPU detector settings using AN/PC overlap and preferably controlled attenuation; only then replace `not_characterized` with a traceable `max_rate_mhz`.
- [ ] Move any provisional PC guard that remains operational to raw observed Level 1 PC rate rather than a corrected/background-subtracted proxy.
- [ ] Characterize MILGRAU propagated-error SNR on SPU data before introducing a hard Rayleigh-window SNR threshold.
- [ ] Validate cloud/layer screening on SPU observations before it becomes a productive Rayleigh-reference gate.
- [ ] Perform independent/reference-chain real-data comparison (LPP/SCC-compatible expectations where appropriate) without claiming numerical identity.

These evidence tasks are scientifically important but should not be simulated by software defaults.

### P5 — high-column Level 2 redesign (future `fixing_l2` scope)

Do not start the ensemble/cascade implementation until P0 and the core of P1 are complete. The redesign needs a clean, single-source retrieval architecture.

The detailed objectives are in Section L below.

---

## A. Configuration ownership and schema

- [x] Remove the generic `physics` section completely.
- [x] Remove obsolete global/minimum-schema validation; stage-specific strict resolvers are authoritative.
- [x] Keep station/site/instrument identity in `station.yaml` and stop loader-created `site`/`physics.channels`/`hardware.name_to_id` compatibility views.
- [x] Productive Level 1 consumes named station calibration rather than positional correction lists.
- [ ] Validate unknown keys with full paths across every stage.
- [ ] Extend typed/resolved configuration objects or equivalent strict accessors to every productive stage.
- [ ] Close residual semantic-default paths identified in P1/P2.

## B. `station.yaml`: observational reality

- [x] Station ID/name/institution/timezone/site coordinates, radiosonde identity, lidar geometry and LR climatology live in the station catalog.
- [x] Historical profiles reference named calibration sets.
- [x] `deadtime_us`, `bin_shift_bins`, `background_offset`, detector mode and saturation characterization are calibration metadata.
- [x] Resolved station profile/calibration IDs persist through products.
- [ ] Add legacy ADC/range overrides only if real historical Licel evidence proves native headers are insufficient; do not invent overrides.
- [ ] Complete physical PC saturation characterization (P4).

## C. Level 0 acquisition

- [x] Strict raw/processed/log paths, raw discovery policy, quarantine, laser-shot tolerance, time jitter and dark-current association.
- [x] No invented weather, SCC channel ID, range resolution, ADC bits or DAQ range.
- [x] Native Licel `BinW` and acquisition metadata are mandatory where scientifically required.
- [x] SCC background-window metadata is explicit.
- [x] Incremental currentness requires structural contract plus readable provenance.
- [ ] Finish unknown-key and auxiliary semantic-default audit under P2.

## D. Level 1 corrections / atmosphere

- [x] PC Poisson uncertainty uses observed counts before dark subtraction.
- [x] Missing channel calibration follows explicit policy and persists neutral legacy use.
- [x] Background, dead-time clipping, PBL reference/search/smoothing and atmosphere-source policy are explicit.
- [x] Productive PBL never silently substitutes another channel.
- [x] Radiosonde/ERA5 selection and station geometry are explicit; ERA5 is pinned to CDS.
- [x] Canonical thermodynamics are materialized in L1; Level 2 should consume them rather than rebuilding atmosphere.
- [x] CPT/LRT share one thermal kernel.
- [ ] Audit the scientific order of PC dark-current subtraction versus nonlinear dead-time correction as a separate evidence-backed issue; do not alter it during unrelated cleanup.

## E. Current Level 2 contract

- [x] Wavelengths, averaging, KFS settings, MC settings, LR, gluing, Rayleigh fit and cloud-screen policy are required/explicit.
- [x] No productive 60 sr/10 sr LR fallback.
- [x] Post-gluing QA can recover a valid single-channel block.
- [x] Rayleigh bounds are search bounds and invalid far-range edge samples are not fabricated.
- [x] `ref_window_m` makes Rayleigh width grid-independent.
- [x] Productive retrieval is scientifically intended to be backward Klett–Fernald.
- [ ] Make the backward contract canonical in source/metadata rather than installed by runtime adapters (P0/P1).
- [ ] Remove remaining public/low-level gluing semantic defaults.
- [ ] Expose gluing spatial window/search bounds in physical units instead of bins/indices when changing that API.
- [ ] Version/document gluing score constants.
- [ ] Validate cloud/layer exclusion and SNR thresholds before enabling them.

## F. Runtime, IO, visualization and logging

- [x] Generic filesystem roots, logging levels and incremental policy are explicit.
- [x] Main CLIs expose consistent `--input`, `--force`, `--version`; LEBEAR also supports `--time-window`.
- [x] Operation statuses/exit codes are simplified and contextual logs are compact.
- [x] Published NetCDF incremental reuse requires current provenance.
- [x] Visualization configuration is strict for productive quicklooks.
- [ ] Remove redundant Level 2 QA status TXT (P0).
- [ ] Audit Level 2 QA helpers for actual use; remove display-only legacy helpers that no longer have a consumer.
- [ ] Split QA plotting by coherent plot family only if it materially improves maintainability.

## G. Scientific failure semantics

- [x] Unknown PC saturation remains explicit `not_characterized`.
- [x] Numerical dead-time clipping is distinct from detector saturation.
- [x] Invalid edge bins are not filled/interpolated merely to keep KFS alive.
- [x] Invalid search domains fail rather than silently widen/substitute domains.
- [x] Missing external atmosphere/weather follows configured policies.
- [ ] Missing required config must fail before processing starts for every remaining auxiliary path.
- [ ] Optional diagnostic failure must never silently change a scientific algorithm.

## H. FAIR provenance and scientific identity

- [x] CalVer release, exact processing/station YAML, station/calibration identity, atmosphere source, LR source and MC settings persist in products.
- [x] Legacy config hashes were removed from public science metadata in favor of readable provenance.
- [x] Incremental reuse rejects products with incomplete current provenance.
- [ ] Fix productive KFS scientific-identity contradiction in `milgrau/scientific.py` (P0).
- [ ] Persist/version each major scientific algorithm separately where needed, not only package release.
- [ ] Decide readable immutable input manifest.
- [ ] If provisional PC guard survives, persist/version its assurance policy and threshold explicitly.

## I. Tests and automated guardrails

- [x] Broad config/station/calibration/acquisition/atmosphere/gluing/KFS/provenance/CLI/logging regressions exist.
- [x] Physical Rayleigh-window regression proves 1000 m maps correctly across uniform grids.
- [x] Backward aggregation and strict backward config regressions exist.
- [x] Real operational case has been manually validated through 5/5 blocks for both elastic wavelengths.
- [ ] Add a single-source KFS metadata/config/product consistency regression (P0).
- [ ] Add synthetic support/top tests before high-column redesign.
- [ ] Add static semantic-default/dead-code guardrails (P2).
- [ ] Add CI and enforce full pytest/static checks.

## J. FAIR / release / repository follow-up

- [ ] Add repository `LICENSE` after confirming institutional licensing choice.
- [x] Runtime/package/CITATION release version is synchronized at CalVer `2026.9`.
- [x] README describes the current pipeline rather than obsolete config/planned-only L2 behavior.
- [ ] Add CI/status checks and branch protection after the suite is stable.
- [ ] Audit NetCDF metadata against relevant CF conventions conservatively.
- [ ] Add archived release/DOI workflow and changelog.
- [ ] External real-data validation against independent/reference processing chain.

## K. Scientific documentation and product reference

README is now the scientific entry point, but it should become shorter once dedicated references exist.

### K1. Completed README baseline

- [x] Current L0→L1→L2 flow, installation, CLI, config/station ownership, station-history effects, product families, productive L2 choices, known limitations and FAIR YAML provenance are documented.

### K2. Processing/configuration references

- [ ] `docs/processing_levels.md`: end-to-end lineage and transformation ownership.
- [ ] `docs/configuration.md`: authoritative config path/type/unit/domain/current value/scientific effect/reprocessing consequence.
- [ ] `docs/station_catalog.md`: station/site/history/calibration/saturation/SCC/radiosonde/LR ownership.
- [ ] Include a worked historical station-profile example and clearly distinguish scientific assumptions, instrument facts and visualization-only settings.
- [ ] Document station-config precedence without presenting it as a scientific source override.

### K3. NetCDF product dictionaries

- [ ] `docs/level0_product.md` with dimensions/variables/types/units/flags/optional fields and acquisition QA semantics.
- [ ] `docs/level1_product.md` with signals, uncertainties, correction diagnostics, atmosphere/PBL/tropopause and exact correction order.
- [ ] `docs/level2_product.md` with molecular, selected/glued signal, Rayleigh/KFS, optical, completeness and failure diagnostics.
- [ ] Document stable numeric flag meanings.
- [ ] Document/deprecate duplicate Level 2 aggregate aliases rather than leaving duplicate semantics implicit.
- [ ] Document NaN/missing/sentinel semantics and remove unjustified `-999` conventions where possible.

### K4. Scientific methods

- [ ] `docs/scientific_methods.md`: PC normalization/Poisson uncertainty/dead time/bin shift/background/range correction, PBL limits, atmosphere mapping, Rayleigh equations, gluing, KFS and MC uncertainty scope.
- [ ] Update `docs/atmospheric_profiles.md` to current nested config/CDS/ERA5 tropopause behavior.
- [ ] Document Rayleigh objective in physical units and the origin-constrained calibration/free-intercept diagnostic.
- [ ] Document gluing regression/fade/uncertainty/acceptance/score and version its score constants.
- [ ] Document generalized Fernald equation, backward productive branch, SCI-001 sign correction, research-only forward/two-sided modes and `aerosol_ref_fraction = R_ref - 1`.
- [ ] Document exactly what the partial Monte Carlo perturbs and which uncertainty sources are still missing.

### K5. Provenance, QA, limitations and validation

- [ ] `docs/provenance_fair.md`.
- [ ] `docs/quality_flags.md`.
- [ ] `docs/known_limitations.md`.
- [ ] `docs/validation.md` distinguishing unit/synthetic tests, product-contract tests and independent real-data validation.
- [ ] Add examples for judging L1/L2 scientific usability from flags and for complete versus partial multispectral products.

### K6. Metadata/schema audit

- [ ] Audit every physical NetCDF variable for units/description/dimensions/missing semantics.
- [ ] Remove residual productive two-sided wording and reconcile algorithm metadata with actual backward mode.
- [ ] Add explicit backscatter/extinction units and conservative CF metadata.
- [ ] Decide whether scalar CPT/LRT diagnostics should remain attrs or become variables in a future schema.
- [ ] Add a product-schema/method-version field if required by future intentional schema changes.

### K7. References and maintenance

- [ ] Build a verified primary-source bibliography for Licel/SCC conventions, Rayleigh/Bucholtz, WMO tropopause, Klett/Fernald/Sasano, ERA5 and network-method comparisons actually used.
- [ ] Add small xarray examples and eventually one end-to-end SPU example when public-data policy is settled.
- [ ] Keep README concise by moving exhaustive tables to dedicated docs.
- [ ] Couple every scientific/schema change to equations/config/product/provenance/reprocessing documentation review.
- [ ] Add lightweight link/reference regression for README/docs.

## L. Future high-column Level 2 redesign (objectives inherited from `fixing_l2`)

This section preserves the intended `fixing_l2` science while keeping the work blocked behind P0/P1 base cleanup.

### L0. Non-negotiable scientific rules

- [x] `target_top_altitude_m` is a target, never permission to extrapolate.
- [x] Do not turn NaN/non-positive/noisy samples into positive signal merely to extend coverage.
- [x] Do not interpolate across internal invalid intervals to make KFS look continuous.
- [x] Backward retrieval never implies values above its boundary condition.
- [x] A lower cascade segment must inherit its boundary from an accepted upper solution; never reset `SR_ref=1` at every segment.
- [x] Multiple reference solutions form a sensitivity/ensemble experiment; disagreement contributes to uncertainty.
- [x] Elastic extinction remains conditional on the assumed aerosol lidar ratio and must not be presented as independently measured extinction.
- [x] 20-minute block retrieval and long-mean/high-column retrieval are distinct products/diagnostics with distinct support/uncertainty.
- [x] Selection, merge and uncertainty policies must be explicit and persist in readable provenance.

### L1. Freeze baseline and support semantics

- [ ] Add a machine-readable regression summary for `20251107sapm` without committing raw observational files: reference altitude/window metrics, valid KFS blocks and retrieval top for both wavelengths.
- [ ] Add synthetic single-reference truncation test.
- [ ] Add synthetic molecular+aerosol truth profiles so "more finite bins" cannot be confused with "more correct bins".
- [ ] Test noisy upper tail, internal invalid gap and valid high-altitude molecular region.
- [ ] Define `retrieval_support_flag` and `retrieval_top_altitude_m` exactly for block, mean and future backbone products.

Acceptance gate: vertical support is measurable before any new coverage algorithm is introduced.

### L2. QA/output cleanup

- [ ] Remove the redundant QA product-status TXT (also P0).
- [ ] Mark optical retrieval support/top explicitly in KFS and scattering-ratio QA.
- [ ] Plot selected/accepted reference windows and do not visually present unsupported signal diagnostics as aerosol retrieval.

### L3. Physical Rayleigh-window configuration

- [x] `ref_window_m` replaces fixed productive bin width.
- [x] Convert physical width on the actual uniform altitude grid with regression coverage.
- [x] Search bounds remain in meters and invalid geometry fails.

This phase from the original `fixing_l2` plan is already complete on `new-architecture`.

### L4. Rayleigh candidate catalogue

- [ ] Replace single-best selection with a candidate evaluator returning every window that passes minimum scientific QA.
- [ ] Candidate diagnostics: start/stop/center, valid fraction, slope, variance, calibration factor, free-intercept diagnostic, uncertainty/SNR diagnostic and future validated layer flag.
- [ ] Separate pass/fail criteria from ranking criteria.
- [ ] Prefer higher altitude only among windows that already pass minimum scientific quality.
- [ ] Define a minimum separation/correlation rule so near-identical overlapping windows do not masquerade as independent ensemble evidence.
- [ ] Persist an auditable compact representation of accepted/rejected candidate reasoning.
- [ ] Do not add a hard SNR/cloud gate until P4 validation supports it.

### L5. High-column backbone

- [ ] Introduce a long-mean retrieval input separate from 20-minute block products.
- [ ] Make averaging duration explicit and persist it.
- [ ] Evaluate altitude-dependent vertical aggregation only if required by data; define widths in meters and preserve original-resolution signal.
- [ ] Select high-altitude references from the backbone profile.
- [ ] Treat 20 km as an initial target, not a success condition; a trustworthy boundary at/above the desired top is required for backward coverage there.
- [ ] If no high boundary is supported, publish a lower retrieval top rather than forcing 20 km.

### L6. Multi-reference ensemble

- [ ] Run backward KFS from multiple accepted high references.
- [ ] Keep member-specific Rayleigh and Monte Carlo diagnostics.
- [ ] Combine only members whose physical backward support includes the altitude and whose QA passed.
- [ ] Define explicit uncertainty/quality weights; no anonymous heuristic constants.
- [ ] Add between-reference spread as an uncertainty component.
- [ ] Keep member solutions available internally for audit/tests.

Acceptance gate: reference sensitivity enlarges uncertainty rather than being hidden by the mean.

### L7. Cascaded backward retrieval

- [ ] Define ordered overlapping segments from high to low.
- [ ] Solve the highest segment from a valid high/molecular boundary.
- [ ] For each lower segment, inherit `beta_total_ref`/scattering ratio from the accepted upper solution.
- [ ] Propagate inherited-boundary uncertainty into lower-segment Monte Carlo.
- [ ] Require adequate overlap/support and reject handoffs inconsistent beyond an explicit uncertainty-aware criterion.
- [ ] Never bridge unsupported internal gaps or reset `SR_ref=1` merely because a new segment begins.

Acceptance gate: cascade reproduces a well-conditioned full-column synthetic truth within defined tolerance and rejects deliberately inconsistent handoffs.

### L8. Overlap merge and uncertainty model

- [ ] Implement a documented smooth overlap merge (for example cosine/sigmoid taper with uncertainty weighting) only after segment acceptance.
- [ ] Weights sum to one where both retrievals are valid; never average valid evidence with NaN/invalid evidence.
- [ ] Check value and vertical-gradient continuity across merge regions.
- [ ] Distinguish measurement/MC, LR, boundary, reference-choice, cascade-handoff and optional averaging/aggregation uncertainty components.
- [ ] Treat covariance deliberately; do not blindly sum correlated components in quadrature.
- [ ] Ensure uncertainty increases when reference ambiguity is introduced.

### L9. Product schema / FAIR / QA for redesigned retrieval

Proposed concepts, names to be finalized only after support semantics/tests are stable:

- [ ] `retrieval_support_flag[..., altitude]`.
- [ ] `retrieval_top_altitude_m[wavelength]` plus block/backbone counterparts.
- [ ] accepted reference-member count and diagnostics.
- [ ] reference-choice/ensemble-spread uncertainty.
- [ ] cascade segment/handoff and merge-region diagnostics.
- [ ] clearly separate long-mean/backbone products from 20-minute block products.
- [ ] explicit metadata that elastic extinction uses the assumed aerosol lidar ratio.
- [ ] version the redesigned retrieval method independently of package CalVer.
- [ ] Update NetCDF contracts and make pre-redesign semantics incrementally stale.
- [ ] QA overview must show signal, molecular fit, candidate references, support top, ensemble spread and cascade/merge regions without huge unsupported upper-tail axes.

### L10. Validation matrix and merge gate

Synthetic:

- [ ] Pure molecular atmosphere → approximately zero aerosol backscatter within tolerance.
- [ ] Known aerosol layers → recover truth within defined tolerance.
- [ ] Multiple valid high references → stable ensemble.
- [ ] Contaminated reference → rejected or visibly inflates uncertainty.
- [ ] Upper-tail noise → support top falls gracefully.
- [ ] Internal invalid gap → explicit failure/no bridging.
- [ ] Segment handoff → no stitching discontinuity beyond tolerance.

Real SPU:

- [ ] `20251107sapm`: materially extend scientifically defensible coverage above current ~6 km while preserving/explaining the 0–6 km solution.
- [ ] Evaluate whether ~15 km is supported; attempt 20 km only when a valid reference/support exists above target.
- [ ] Test clear/clean, high-aerosol, cloud-contaminated, weak-signal and different AN/PC-dominance cases before merge.
- [ ] Compare against LPP where practical and SCC/ELDA methodological expectations without claiming numerical identity.

Merge success criterion: **maximize validated vertical support, expose where support ends, and remain stable/traceable under reference/noise sensitivity. Reaching 20 km by itself is not success.**

## M. Code organization and concision acceptance checklist

Use this checklist whenever cleanup/refactoring is proposed.

- [ ] A symbol/module exists because it has a current consumer or a documented public/research role.
- [ ] No productive scientific behavior is installed by import side effect.
- [ ] No wildcard import is used to define the productive public API.
- [ ] No two active functions implement the same productive scientific rule unless one is a tested low-level kernel and the other is clearly orchestration.
- [ ] Compatibility wrappers have a named consumer and planned removal point.
- [ ] Public aliases are either canonical or in an explicit deprecation window.
- [ ] Scientific constants/algorithm weights have names and versioned meaning; anonymous magic weights are avoided.
- [ ] Configuration is resolved once through strict accessors rather than repeatedly interpreted with local defaults.
- [ ] Numerical kernels are pure/testable and do not perform filesystem/config-discovery/logging policy work.
- [ ] Orchestration code coordinates stages but does not duplicate equations.
- [ ] Dataset construction defines schema/metadata but does not decide retrieval science.
- [ ] QA/visualization does not feed back into scientific retrieval decisions.
- [ ] Tests target current contracts, not obsolete compatibility behavior that we actually want to remove.
- [ ] Refactoring reduces or preserves conceptual complexity; a lower line count is not required if clarity/validation improves.
- [ ] New abstractions must remove duplication or create a real boundary; avoid wrapper layers that only rename a call.

## Recommended implementation batches on `new-architecture`

1. **P0 metadata/wording cleanup + remove QA TXT** — small, high-confidence, immediately removes contradictory public outputs.
2. **Canonical backward policy** — move backward mode/description into `config.py`, align `scientific.py`, add consistency regression.
3. **Remove Level 2 monkey patches and wildcard API** — direct-wire supported-domain QA/backward aggregation and make imports explicit.
4. **Decompose `_retrieval_impl.py` + dead-code/default audit** — remove legacy atmosphere/two-sided/input-QA paths and compatibility alias if unneeded.
5. **Level 2 schema cleanup** — duplicate aggregate aliases, units/flag metadata, algorithm/schema versioning, support/top semantics.
6. **Static analysis + CI** — unused/dead-code guard, semantic-default guard, full pytest.
7. **Focused scientific docs/validation baseline** — methods, L2 product, flags, limitations, synthetic support tests.
8. **Recreate `fixing_l2` from this cleaner base** and begin L4/L5 candidate-catalogue/high-column work.

Physical PC characterization, SNR characterization and cloud/layer validation may proceed in parallel because they depend on observational evidence rather than code cleanup.

## Current highest-value debt, in order

1. [ ] Eliminate contradictory KFS scientific identity/metadata.
2. [ ] Eliminate Level 2 import-time monkey patches and wildcard public API.
3. [ ] Decompose/remove duplicate legacy `_retrieval_impl.py` paths and semantic defaults.
4. [ ] Add static dead-code/default guardrails and CI.
5. [ ] Clean Level 2 schema metadata/duplicate aliases and define retrieval support/top.
6. [ ] Version/document gluing selection score.
7. [ ] Complete targeted scientific docs needed to freeze the current product contract.
8. [ ] Physical PC saturation characterization and raw-rate guard migration.
9. [ ] SNR/cloud validation for high-altitude reference QA.
10. [ ] Begin high-column ensemble/cascade redesign only after the base above is clean.

## Implementation log

### 2026-09-09 to 2026-09-14 — strict-config and scientific baseline

- Introduced stage-specific fail-fast L0/L1/L2 configuration, station calibration history, explicit atmosphere/weather policies, strict acquisition metadata, contextual logging, readable FAIR provenance and simplified execution status/CLI behavior.
- Corrected/validated key Level 2 behavior: guarded PC use without inventing saturation characterization, post-gluing fallback, Rayleigh search-domain semantics, backward Fernald sign/branch behavior and real-data 355/532 retrieval.
- Replaced fixed Rayleigh bins with physical `ref_window_m`, retired the obsolete global schema and final generic `physics` config residue, and gated incremental reuse on current provenance.
- Rewrote README as a current scientific entry point and created the documentation roadmap.

### 2026-09-14 — architecture/science re-audit and roadmap consolidation

- Re-audited `new-architecture` at `c6400a941964de73438132afdb7f3db9a5916f32` before recreating the future high-column branch.
- Identified a critical consistency cluster: runtime backward policy is currently installed by package-init monkey patches while canonical `config.py` and versioned `scientific.py` still contain two-sided productive semantics; dataset/log text also retains two-sided wording.
- Identified `_retrieval_impl.py` as the main cleanup target because legacy atmosphere/config-default/input-QA paths coexist with newer strict public-boundary implementations.
- Promoted code-use/dead-code/concision auditing to a first-class project goal: one canonical implementation, explicit public surface, no unexplained compatibility wrappers and automated static/CI guardrails.
- Folded the saved `fixing_l2` objectives into Section L. Physical Rayleigh width is already complete on this base; candidate catalogue, high-column backbone, reference ensemble, cascaded backward retrieval, uncertainty-aware merge and validation remain future work after P0/P1 cleanup.
