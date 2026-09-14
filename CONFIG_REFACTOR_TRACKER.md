# MILGRAU architecture, scientific quality and Level 2 roadmap

Branch: `new-architecture`

Audit baseline: `c6400a941964de73438132afdb7f3db9a5916f32` (2026-09-14)

This file is the source of truth for consolidating the current architecture before a future `fixing_l2` branch is recreated. Priorities are deliberate: scientific correctness and truthful metadata first, then removal of compatibility/dead code, then automated quality/schema hardening, and only then the larger high-column Level 2 redesign.

## Goal

MILGRAU should be scientifically defensible, FAIR, concise and easy to audit. Good quality does **not** mean retaining every historical path or adding abstraction for its own sake.

- `config.yaml` = processing/scientific recipe.
- `station.yaml` = observational reality, hardware history, calibration, station-derived climatology and SCC mapping.
- Python = equations, physical constants, validated runtime objects and implementation details.
- One productive scientific behavior must have one canonical implementation.
- Productive science must not depend on import order, monkey patching or hidden semantic defaults.
- Every retained module/function/class/public variable must have a current role: productive API, internal implementation, validated research diagnostic, or explicitly temporary compatibility path.
- Compatibility code without a named consumer and removal criterion should be removed.
- Duplicate public outputs/aliases require an explicit migration/deprecation contract.
- Split large modules only when cohesion improves; do not replace one monolith with trivial wrapper modules.
- Comments/docstrings should explain scientific or architectural intent, not preserve obsolete implementation history.
- Missing scientific/instrumental configuration must fail early unless an explicit unavailable/legacy policy is selected.
- Operator logs report outcomes; detailed transport/search/debug information belongs in the DEBUG audit log.
- Cleanup must not silently change an equation, threshold, calibration assumption or uncertainty model.

## Current audit snapshot

### Stable foundations

- [x] Obsolete generic top-level `physics` config and global `milgrau/config/schema.py` are removed.
- [x] Main L0/L1/L2 paths use stage-specific strict configuration.
- [x] Station/site/instrument truth is owned by `station.yaml`; loader-created legacy station/hardware aliases are removed.
- [x] L0 requires traceable Licel range/ADC/DAQ information and explicit weather/dark-current/SCC policies.
- [x] L1 materializes canonical thermodynamics with atmosphere-source provenance.
- [x] CPT/LRT share one thermal kernel for radiosonde and ERA5; USSA76 is not invented as a tropopause source.
- [x] L2 supports strict AN/PC gluing, post-QA single-channel fallback and explicit retrieval-input reasons.
- [x] Unknown physical PC saturation remains `not_characterized`; the 10% dead-time occupancy guard is explicitly provisional.
- [x] Rayleigh bounds are search bounds; invalid far-range samples are not fabricated.
- [x] Productive Rayleigh width is physical (`ref_window_m`) and grid-independent.
- [x] Real case `20251107sapm` reached 5/5 valid Rayleigh/backward-KFS blocks at 355 and 532 nm.
- [x] Published NetCDF reuse requires current readable provenance, not timestamps alone.
- [x] README describes the current L0→L1→L2 scientific flow and productive backward KFS baseline.

### P0 — status after lot 1

- [x] `milgrau/level2/config.py` owns productive `backward` KFS mode and description.
- [x] `milgrau/scientific.py` publishes `integration_mode = backward`.
- [x] `dataset.py` and `lebear.py` no longer describe productive success as two-sided.
- [x] Incremental Level 2 reuse rejects stale/contradictory KFS scientific metadata.
- [x] Redundant `QA_L2_Product_Status_*.txt` generation is removed.
- [x] Focused regressions pin backward identity, stale-output rejection, failure wording and absence of the redundant QA TXT.
- [x] Future `retrieval_support_flag` / `retrieval_top_altitude_m` semantics are frozen below before schema implementation.

### P1 — status after lot 2

- [x] `milgrau/level2/__init__.py` is declarative and no longer monkey-patches scientific functions.
- [x] Productive signal selection calls `evaluate_retrieval_input_supported_domain()` directly in `level2/signal_selection.py`.
- [x] Productive Rayleigh/KFS aggregation is direct in `level2/optical_retrieval.py`; success requires Rayleigh QA + backward KFS, not the forward branch.
- [x] `backward_retrieval.py` was removed after its only scientific responsibility moved into the canonical optical-retrieval path.
- [x] `scientific_policy.py` was removed after P0 made backward policy canonical in `level2/config.py`.
- [x] `retrieval.py` no longer uses `from ..._retrieval_impl import *`; productive dependencies are explicit.
- [x] Regressions now detect import-time patching and point supported-domain/backward aggregation tests at canonical modules.
- [ ] `_retrieval_impl.py` still contains duplicate legacy gluing/input-QA/optical-retrieval/atmosphere paths and semantic defaults. These are no longer productive, but must be deleted/decomposed in P1 lot 3.
- [ ] Shared retrieval dataclasses/helpers still live in `_retrieval_impl.py`; move them to cohesive modules while deleting the legacy implementations rather than creating another adapter layer.
- [ ] `milgrau/level2/atmosphere.py` compatibility path remains to be audited after legacy atmosphere removal.
- [ ] Low-level KFS research defaults still need an API audit so `two_sided` cannot look like the productive default.
- [ ] No CI/status checks are attached to this branch; focused regressions are committed but the full suite is not claimed green.

## Priority order

### P0 — scientific identity / public consistency

**Lot 1: implemented. Full-suite execution remains unverified because the branch has no CI/status checks.**

- [x] Canonicalize productive backward KFS mode/description.
- [x] Align versioned scientific metadata and public product wording.
- [x] Reject incrementally reused L2 files with stale KFS identity.
- [x] Remove redundant Level 2 QA status TXT.
- [x] Freeze future vertical-support/top semantics.

### P1 — remove import-order behavior and duplicate compatibility paths

**Lot 2: implemented. Lot 3 is now the highest-priority work.**

- [x] Remove scientific monkey patches from `milgrau/level2/__init__.py`.
- [x] Direct-wire supported-domain retrieval-input QA.
- [x] Make backward-only aggregate success canonical.
- [x] Retire `backward_retrieval.py`.
- [x] Retire superseded `scientific_policy.py`.
- [x] Replace wildcard `_retrieval_impl` import/export with explicit symbols.
- [ ] Decompose `_retrieval_impl.py` by responsibility and delete the duplicate legacy implementations now bypassed by productive code.
- [ ] Move shared dataclasses/state validation/helpers out of `_retrieval_impl.py` only where they have a clear cohesive owner.
- [ ] Remove legacy `_retrieval_impl.build_thermodynamic_profile()` after proving the canonical Level 1-atmosphere boundary is the only productive consumer.
- [ ] Remove duplicate legacy `_evaluate_retrieval_input()`.
- [ ] Remove duplicate legacy `glue_signal_blocks()` and `retrieve_optical_blocks()` after canonical module regressions cover their productive behavior.
- [ ] Remove/rewrite `_retrieval_impl.run_kfs_profile()` semantic defaults; productive KFS settings must come from strict config.
- [ ] Remove residual `fit_config.get(..., literal)` / `gluing_config.get(..., literal)` scientific defaults where strict resolvers guarantee keys.
- [ ] Audit `milgrau/level2/atmosphere.py`; delete it if no legitimate public consumer remains.
- [ ] Review low-level KFS `mode="two_sided"` defaults. Research modes may remain, but productive callers must pass explicit mode and public API must not imply two-sided production.

P1 acceptance gate: importing `milgrau.level2` has no scientific side effects; no productive path depends on wildcard imports, duplicate scientific implementations or obsolete config compatibility.

### P2 — code-use audit, concision and automated guardrails

- [ ] Build repository-wide symbol/module inventory: public API, internal implementation, research diagnostic, compatibility, unused.
- [ ] Record actual consumer + removal criterion for every compatibility path; delete compatibility with no consumer.
- [ ] Audit `__all__` and package re-exports; each public symbol needs a documented/tested purpose.
- [ ] Add lightweight Ruff (or equivalent) checks for unused imports/variables, obvious dead code and basic style; keep rules scientifically neutral.
- [ ] Add guard against `config.get(..., semantic_literal_default)` outside explicit resolver/compatibility code.
- [ ] Add repository regression for undeclared scientific/instrumental defaults in productive paths.
- [ ] Search for duplicated scientific equations/selection rules and retain one canonical implementation.
- [ ] Review broad `except Exception`; retain only intentional orchestration/diagnostic boundaries.
- [ ] Review large mixed-responsibility files (`_retrieval_impl.py`, `dataset.py`, `viz/level2_qa.py`, explorer app) and split only when cohesion improves.
- [ ] Remove obsolete `legacy` helpers/comments after replacement is validated; Git history/docs retain historical context.
- [ ] Add CI for imports, syntax/static checks and full pytest; add branch protection only after CI is stable.

P2 acceptance gate: no known unused compatibility code, no wildcard productive API, no scientific import side effects, and automated checks prevent those patterns from returning.

### P3 — Level 2 schema / FAIR metadata / focused documentation

- [ ] Decide canonical aggregate names and migration path for `aerosol_backscatter[_mean]` / `aerosol_extinction[_mean]` aliases.
- [ ] Audit every L2 physical variable for units, description/long_name, dimensions and NaN/missing semantics.
- [ ] Add explicit units for aerosol backscatter/extinction and uncertainty fields where currently implied.
- [ ] Audit numeric flag metadata for CF-compatible representation.
- [ ] Add product-schema/method-version field independent of package CalVer if intentional schema evolution needs it.
- [ ] Name/version/document gluing selection-score constants; algorithmic weights must not remain anonymous literals.
- [ ] Decide readable immutable input-manifest policy.
- [ ] Create focused docs (`processing_levels`, config, station catalog, products, methods, provenance, flags, limitations, validation) and then shorten README into an entry point.

P3 acceptance gate: a Level 2 NetCDF is scientifically interpretable without reading implementation source and its metadata cannot contradict the method used.

### P4 — observational/scientific evidence tasks (parallel)

- [ ] Characterize physical PC saturation for operational SPU detector settings using AN/PC overlap and preferably controlled attenuation; only then write traceable `max_rate_mhz`.
- [ ] Move any surviving provisional PC guard to raw observed Level 1 PC rate rather than corrected/background-subtracted proxy.
- [ ] Characterize propagated-error SNR on SPU data before introducing a hard Rayleigh-window SNR threshold.
- [ ] Validate cloud/layer screening on SPU observations before making it a productive Rayleigh gate.
- [ ] Compare real-data retrieval against an independent/reference chain (LPP and SCC/ELDA methodological expectations where appropriate) without claiming numerical identity.
- [ ] Audit Level 1 PC dark-current subtraction versus nonlinear dead-time correction as a separate evidence-backed issue.

Evidence tasks must not be replaced by invented software defaults.

### P5 — future high-column Level 2 redesign (`fixing_l2`)

Do not implement ensemble/cascade retrieval until P0 is complete and P1 duplicate/compatibility cleanup is complete. Detailed objectives are in Section L.

---

## Current processing-level contract

### A. Configuration / station ownership

- [x] `config.yaml` owns processing/scientific recipe; `station.yaml` owns station/instrument reality.
- [x] Historical station profiles reference named calibration sets; resolved profile/calibration IDs persist through products.
- [x] `deadtime_us`, `bin_shift_bins`, `background_offset`, detector mode and saturation characterization are station calibration metadata.
- [ ] Validate unknown keys with full paths across every stage.
- [ ] Extend typed/resolved strict accessors to remaining auxiliary paths.
- [ ] Add legacy ADC/range overrides only if real Licel evidence proves native headers insufficient.

### B. Level 0

- [x] Explicit raw/processed/log paths, discovery/quarantine, laser-shot tolerance, time jitter and dark-current association.
- [x] No invented weather, SCC ID, range resolution, ADC bits or DAQ range.
- [x] Incremental currentness requires structural contract plus readable provenance.
- [ ] Finish auxiliary unknown-key/semantic-default audit under P2.

### C. Level 1

- [x] PC Poisson uncertainty uses observed counts before dark subtraction.
- [x] Missing channel calibration follows explicit policy and neutral legacy use is persisted.
- [x] Background, numerical dead-time clipping, PBL and atmosphere-source policy are explicit.
- [x] Productive PBL does not substitute another channel.
- [x] Radiosonde/ERA5 selection and historical station geometry are explicit; ERA5 is pinned to CDS.
- [x] Canonical thermodynamics are materialized in L1; productive L2 consumes them.
- [x] CPT/LRT share one thermal kernel.
- [ ] PC correction-order scientific audit remains P4.

### D. Level 2 baseline

- [x] Wavelengths, temporal blocks, KFS/MC/LR/gluing/Rayleigh/cloud policy are explicit.
- [x] No productive 60 sr / 10 sr LR fallback.
- [x] Post-gluing QA can recover a valid single-channel block.
- [x] Rayleigh search tolerates invalid far-range edge samples without filling/clipping them.
- [x] `ref_window_m` makes reference width grid-independent.
- [x] Productive config/provenance/public metadata identify backward Klett–Fernald.
- [x] Productive runtime QA/aggregation is direct; package import no longer installs scientific adapters.
- [x] Productive retrieval API uses explicit imports rather than wildcard monolith exposure.
- [ ] Remove duplicate legacy retrieval functions/defaults from `_retrieval_impl.py` under P1 lot 3.
- [ ] Later migrate gluing spatial search/window settings to physical units when changing that API.
- [ ] Version gluing score constants.
- [ ] Keep cloud/SNR productive gates disabled until P4 evidence supports them.

### E. Runtime / visualization / logging

- [x] Generic filesystem roots, logging levels and incremental policy are explicit.
- [x] Main CLIs share `--input`, `--force`, `--version`; LEBEAR additionally has `--time-window`.
- [x] Operational result/exit-code model is compact and contextual logs separate INFO outcomes from DEBUG detail.
- [x] Level 2 QA no longer creates the redundant product-status TXT.
- [ ] Audit QA helpers for actual use; remove display-only legacy helpers without consumers.

### F. Failure semantics / FAIR

- [x] Unknown PC saturation remains explicit `not_characterized`.
- [x] Numerical dead-time clipping is not detector saturation characterization.
- [x] Invalid edge bins are not interpolated/fabricated to keep KFS alive.
- [x] Invalid search domains fail rather than widen silently.
- [x] Exact processing/station YAML, CalVer, station/calibration identity, atmosphere source, LR source and MC settings persist.
- [x] L2 incremental reuse rejects contradictory current KFS scientific identity.
- [ ] Missing required config must fail before processing for every remaining auxiliary path.
- [ ] Optional diagnostic failure must never silently change scientific algorithm choice.
- [ ] Decide readable immutable input manifest and additional method/schema versioning under P3.

## Tests / validation status

- [x] Broad config/station/acquisition/atmosphere/gluing/KFS/provenance/CLI/logging regressions exist.
- [x] Physical Rayleigh-window tests prove 1000 m converts correctly across uniform grids.
- [x] Backward aggregation and productive backward-config regressions exist.
- [x] P0 regressions pin config/scientific metadata/generated L2 product to backward mode and reject stale two-sided metadata.
- [x] P0 regression ensures QA does not emit `QA_L2_Product_Status_*.txt`.
- [x] P1 regressions are committed to prove legacy `_retrieval_impl` functions are not monkey-patched and canonical signal selection uses supported-domain QA directly.
- [x] P1 backward-aggregation regression now targets `optical_retrieval.py`, not the removed adapter.
- [x] Real case `20251107sapm` has been manually observed reaching 5/5 Rayleigh/backward-KFS blocks at 355/532 nm on the pre-P1 numerical baseline.
- [ ] Re-run `20251107sapm` after P1 lot 2 to confirm numerical/diagnostic equivalence through the new direct orchestration.
- [ ] Add synthetic support/top tests before implementing high-column retrieval.
- [ ] Add static semantic-default/dead-code guardrails under P2.
- [ ] Add CI/full pytest enforcement. **Until then, do not claim the complete suite is green.**

## Scientific documentation / repository follow-up

- [x] README is a current scientific entry point and describes productive backward KFS.
- [ ] Add repository `LICENSE` after institutional licensing choice is confirmed.
- [ ] `docs/processing_levels.md` — end-to-end transformation ownership.
- [ ] `docs/configuration.md` — authoritative config path/type/unit/domain/effect/reprocessing reference.
- [ ] `docs/station_catalog.md` — site/history/calibration/saturation/SCC/radiosonde/LR ownership.
- [ ] `docs/level0_product.md`, `docs/level1_product.md`, `docs/level2_product.md` — dimensions/variables/units/flags/missing semantics.
- [ ] `docs/scientific_methods.md` — corrections, PBL, atmosphere, Rayleigh, gluing, KFS, MC equations/limitations.
- [ ] Update `docs/atmospheric_profiles.md` to current nested config/CDS/ERA5-tropopause behavior.
- [ ] `docs/provenance_fair.md`, `docs/quality_flags.md`, `docs/known_limitations.md`, `docs/validation.md`.
- [ ] Build verified primary-source bibliography for methods actually implemented.
- [ ] Add small xarray examples and eventually one public/anonymized SPU worked example.
- [ ] Add archived release/DOI workflow, changelog and documentation link/reference regression.

---

## L. Future high-column Level 2 redesign — inherited `fixing_l2` objectives

### L0. Non-negotiable scientific rules

- [x] `target_top_altitude_m` is a target, never permission to extrapolate.
- [x] Do not turn NaN/non-positive/noisy samples into positive signal merely to extend coverage.
- [x] Do not interpolate across internal invalid intervals to make KFS continuous.
- [x] Backward retrieval never implies values above its boundary condition.
- [x] A lower cascade segment inherits its boundary from an accepted upper solution; never reset `SR_ref=1` at every segment.
- [x] Multiple reference solutions form a sensitivity/ensemble experiment; disagreement contributes to uncertainty.
- [x] Elastic extinction remains conditional on assumed aerosol lidar ratio.
- [x] 20-minute block retrieval and long-mean/high-column retrieval are distinct products/diagnostics.
- [x] Selection, merge and uncertainty policies must be explicit and persisted in readable provenance.

### L1. Vertical-support semantics — frozen before schema implementation

- [x] `retrieval_support_flag(block_time, wavelength, altitude)` means the **final productive optical retrieval** is scientifically supported and usable at that altitude after source selection, Rayleigh QA, KFS/candidate QA and any accepted merge. It is not merely `isfinite(product)` and does not encode source/member identity.
- [x] For the current backward-only baseline, support includes only bins actually solved on the accepted backward branch through the selected reference bin. Invalid outer bins and bins beyond a terminated branch remain unsupported/NaN; internal gaps are never bridged.
- [x] A future accepted high-column candidate/cascade may extend support only where that candidate has its own valid physical support. No extrapolated/fabricated bin can set support to `1`.
- [x] `retrieval_top_altitude_m(block_time, wavelength)` is the highest altitude where `retrieval_support_flag == 1`; NaN when no supported bin exists. Never derive it independently from the last finite array element.
- [x] Mean/backbone support and top come from the corresponding accepted product/support mask, not blindly from block products.
- [ ] Implement support variables only with the redesigned schema after synthetic support tests exist.
- [ ] Add machine-readable baseline summary for `20251107sapm` without raw observational files: reference/window metrics, valid KFS blocks and retrieval top for 355/532 nm.
- [ ] Add synthetic single-reference truncation, molecular+aerosol truth, noisy upper tail, internal-gap and valid high molecular-region tests.

Acceptance gate: vertical support is testable/measurable before introducing a coverage-extension algorithm.

### L2. QA/output cleanup

- [x] Remove redundant QA product-status TXT.
- [ ] Mark optical support/top explicitly in KFS and scattering-ratio QA after support variables enter the schema.
- [ ] Plot selected/accepted reference windows and never visually present unsupported signal diagnostics as aerosol retrieval.

### L3. Physical Rayleigh window

- [x] Productive fixed-bin width replaced by `ref_window_m`.
- [x] Physical width is converted on the actual uniform altitude grid with regression coverage.
- [x] Search bounds stay in meters and invalid geometry fails.

### L4. Rayleigh candidate catalogue

- [ ] Replace single-best selection with evaluator returning every window that passes minimum scientific QA.
- [ ] Candidate diagnostics: start/stop/center, valid fraction, slope, variance, calibration factor, free-intercept diagnostic, uncertainty/SNR diagnostic and future validated layer flag.
- [ ] Separate pass/fail criteria from ranking criteria.
- [ ] Prefer higher altitude only among already-valid windows.
- [ ] Define minimum separation/correlation so heavily overlapping windows do not masquerade as independent ensemble evidence.
- [ ] Persist compact auditable accepted/rejected reasoning.
- [ ] Do not add hard SNR/cloud gates before P4 validation.

### L5. High-column backbone

- [ ] Add long-mean retrieval input separate from 20-minute block products; averaging duration explicit/provenanced.
- [ ] Evaluate altitude-dependent vertical aggregation only if data require it; widths in meters, original-resolution signal retained.
- [ ] Select high references from the backbone profile.
- [ ] Treat 20 km as an initial target, not a success condition; publish a lower top whenever no trustworthy high boundary exists.

### L6. Multi-reference ensemble

- [ ] Run backward KFS from multiple accepted high references.
- [ ] Preserve member-specific Rayleigh/MC diagnostics.
- [ ] Combine only members whose physical backward support contains the altitude and whose QA passed.
- [ ] Define explicit uncertainty/quality weights; no anonymous heuristics.
- [ ] Add between-reference spread as an uncertainty component.
- [ ] Keep member solutions internally auditable/testable.

Acceptance gate: reference sensitivity increases exposed uncertainty rather than disappearing in an average.

### L7. Cascaded backward retrieval

- [ ] Define ordered overlapping high→low segments.
- [ ] Solve highest segment from valid high/molecular boundary.
- [ ] Each lower segment inherits `beta_total_ref` / scattering ratio from accepted upper solution.
- [ ] Propagate inherited-boundary uncertainty in lower-segment Monte Carlo.
- [ ] Require adequate overlap/support; reject uncertainty-inconsistent handoffs.
- [ ] Never bridge unsupported internal gaps or reset `SR_ref=1` merely because a new segment begins.

Acceptance gate: cascade reproduces a well-conditioned synthetic full-column truth within defined tolerance and rejects inconsistent handoffs.

### L8. Overlap merge / uncertainty model

- [ ] Implement documented smooth overlap merge only after segment acceptance; valid weights sum to one.
- [ ] Never average valid evidence with NaN/invalid evidence.
- [ ] Check value and vertical-gradient continuity across merge regions.
- [ ] Separate measurement/MC, LR, boundary, reference-choice, cascade-handoff and optional aggregation uncertainty components.
- [ ] Treat covariance deliberately; do not blindly quadrature-sum correlated terms.
- [ ] Verify uncertainty grows when reference ambiguity is introduced.

### L9. Redesigned product / FAIR / QA concepts

Names become public only after L1 support tests/schema decisions are stable.

- [ ] `retrieval_support_flag[..., altitude]`.
- [ ] `retrieval_top_altitude_m[...]` for block and backbone/aggregate products.
- [ ] accepted reference-member count/diagnostics and ensemble-spread uncertainty.
- [ ] cascade segment/handoff/merge diagnostics.
- [ ] clear separation of long-mean/backbone from 20-minute products.
- [ ] explicit elastic-extinction dependence on assumed aerosol LR.
- [ ] independent redesigned-method version and NetCDF contract update; pre-redesign semantics become incrementally stale.
- [ ] QA overview shows signal, molecular fit, candidate references, support top, ensemble spread and cascade/merge regions without unsupported upper-tail axes.

### L10. Validation / merge gate

Synthetic:

- [ ] Pure molecular → approximately zero aerosol backscatter within tolerance.
- [ ] Known aerosol layers → recover truth within defined tolerance.
- [ ] Multiple valid references → stable ensemble.
- [ ] Contaminated reference → rejected or visibly inflates uncertainty.
- [ ] Upper-tail noise → support top falls gracefully.
- [ ] Internal invalid gap → explicit failure/no bridging.
- [ ] Segment handoff → no stitching discontinuity beyond tolerance.

Real SPU:

- [ ] `20251107sapm`: materially extend defensible coverage above the current lower-column solution while preserving/explaining that solution.
- [ ] Evaluate whether ~15 km is supported; attempt 20 km only with valid boundary/support above target.
- [ ] Test clear, high-aerosol, cloud-contaminated, weak-signal and different AN/PC-dominance cases.
- [ ] Compare with LPP where practical and SCC/ELDA methodological expectations without claiming numerical identity.

Merge success criterion: **maximize validated vertical support, expose where support ends, and remain stable/traceable under reference/noise sensitivity. Reaching 20 km by itself is not success.**

---

## Code organization / concision acceptance checklist

Use this for every refactor batch.

- [ ] Every symbol/module has a current consumer or documented public/research role.
- [x] No productive scientific behavior is installed by import side effect.
- [x] No wildcard import defines the productive Level 2 public API.
- [ ] No duplicate active/legacy implementation of one productive scientific rule remains after P1 lot 3.
- [ ] Compatibility wrappers have named consumers and planned removal point.
- [ ] Public aliases are canonical or in an explicit deprecation window.
- [ ] Scientific constants/weights have names and versioned meaning; avoid anonymous magic weights.
- [ ] Configuration is resolved once through strict accessors rather than repeatedly reinterpreted with local defaults.
- [ ] Numerical kernels do not own filesystem/config-discovery/logging policy.
- [x] Productive Level 2 orchestration coordinates explicit signal-selection and optical-retrieval modules rather than package-init patching.
- [ ] Dataset construction defines schema/metadata without deciding retrieval science.
- [x] QA/visualization does not feed back into scientific retrieval decisions.
- [x] Current P1 tests target canonical modules rather than the removed compatibility adapters.
- [ ] Refactoring reduces/preserves conceptual complexity after obsolete monolith code is deleted; temporary line-count growth during extraction is not an acceptance endpoint.
- [x] New P1 modules represent real scientific boundaries (signal selection and optical retrieval), not one-call renaming wrappers.

## Recommended implementation batches

1. [x] **P0 lot 1 — backward identity + stale-output guard + QA TXT removal + regressions + support semantics.**
2. [x] **P1 lot 2 — remove Level 2 monkey patches, direct-wire supported-domain QA/backward aggregation, remove wildcard productive import and retire adapters.**
3. [ ] **P1 lot 3 — decompose/delete duplicate `_retrieval_impl.py` paths, legacy atmosphere/input-QA and semantic defaults; move shared types/helpers to cohesive owners.**
4. [ ] **P2 — repository symbol-use/dead-code/default audit, static analysis and CI.**
5. [ ] **P3 — Level 2 schema aliases/units/flags/method version + focused docs.**
6. [ ] **Validation baseline — synthetic support tests + machine-readable real-case regression summary.**
7. [ ] **Recreate `fixing_l2` from the cleaned base and start candidate-catalogue/high-column work.**

P4 physical PC/SNR/cloud evidence tasks may proceed in parallel.

## Current highest-value debt

1. [ ] Delete/decompose duplicate legacy retrieval paths in `_retrieval_impl.py` and move only genuinely shared types/helpers to cohesive modules.
2. [ ] Remove remaining semantic defaults and obsolete Level 2 atmosphere compatibility.
3. [ ] Build symbol-use/dead-code/default guardrails and CI.
4. [ ] Clean L2 duplicate aliases/units/flag metadata and method/schema versioning.
5. [ ] Version/document gluing score.
6. [ ] Add synthetic support/top validation and focused scientific docs.
7. [ ] Physical PC saturation characterization/raw-rate guard migration.
8. [ ] SNR/cloud validation for high-altitude reference QA.
9. [ ] Begin ensemble/cascade redesign only after the base above is clean.

## Implementation log

### 2026-09-09 to 2026-09-14 — strict-config and scientific baseline

- Introduced stage-specific fail-fast L0/L1/L2 configuration, station calibration history, explicit atmosphere/weather policies, strict acquisition metadata, contextual logging, readable FAIR provenance and simplified execution/CLI behavior.
- Corrected/validated guarded PC use without inventing saturation characterization, post-gluing fallback, Rayleigh search semantics and backward Fernald sign/branch behavior.
- Replaced fixed Rayleigh bins with physical `ref_window_m`, retired global schema/final generic `physics` config residue and gated incremental reuse on current provenance.
- Rewrote README as current scientific entry point.

### 2026-09-14 — architecture/science re-audit

- Re-audited `new-architecture` at `c6400a941964de73438132afdb7f3db9a5916f32`.
- Identified import-time Level 2 scientific adapters, wildcard monolith exposure, duplicated legacy retrieval paths and stale KFS metadata as the highest-value cleanup cluster.
- Promoted code-use/dead-code/concision auditing to a first-class goal and folded saved `fixing_l2` objectives into Section L.

### 2026-09-14 — P0 lot 1: backward identity and QA-output consistency

- Made productive backward KFS identity canonical in config, scientific metadata, dataset wording and operational failure diagnostics.
- Made incremental L2 reuse reject stale KFS identity.
- Removed redundant QA status TXT.
- Added focused regressions and froze future support/top semantics.
- No Fernald numerical equation, LR, MC, Rayleigh threshold, gluing threshold or PC saturation characterization was changed.
- Full repository suite is **not claimed** because no CI/status checks are attached.

### 2026-09-14 — P1 lot 2: explicit Level 2 scientific orchestration

- Removed all scientific monkey patches from `milgrau/level2/__init__.py`; package import is now declarative.
- Added `level2/signal_selection.py` as the canonical productive boundary for block gluing/source selection and direct supported-domain retrieval-input QA.
- Added `level2/optical_retrieval.py` as the canonical productive Rayleigh/backward-KFS aggregation boundary.
- Replaced `retrieval.py` wildcard monolith exposure with explicit dependencies and direct canonical orchestration.
- Removed `backward_retrieval.py` and `scientific_policy.py` after their responsibilities became canonical elsewhere.
- Added/updated regressions so legacy `_retrieval_impl` functions remain unpatched and tests target the direct productive QA/backward aggregation paths.
- This lot intentionally did **not** change NetCDF schema, Fernald equations, LR/MC settings, Rayleigh thresholds, gluing thresholds or PC guard policy.
- `_retrieval_impl.py` still contains bypassed duplicate legacy implementations and defaults; P1 lot 3 must remove them before P1 is considered complete.
- Full repository suite is **not claimed**; the immediate manual acceptance check is to re-run `20251107sapm` and confirm the same 5/5 backward retrieval behavior on both wavelengths.
