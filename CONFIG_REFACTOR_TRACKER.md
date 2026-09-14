# MILGRAU architecture, scientific quality and Level 2 roadmap

Branch: `new-architecture`

Audit baseline: `c6400a941964de73438132afdb7f3db9a5916f32` (2026-09-14)

This file is the source of truth for preparing a clean, scientifically defensible base before recreating `fixing_l2`. Priorities are intentional: correctness and truthful provenance first; one canonical implementation second; automated code-quality/schema hardening third; high-column retrieval only after those gates are stable.

## 1. Engineering/scientific rules

MILGRAU should be FAIR, concise, testable and easy to audit.

- `config.yaml` owns the processing/scientific recipe.
- `station.yaml` owns station/instrument reality, hardware history, calibration, station-derived climatology and SCC mapping.
- Python owns equations, physical constants, validated runtime objects and implementation details.
- One productive scientific behavior has one canonical implementation.
- Productive science must not depend on import order, monkey patching, wildcard imports or hidden semantic defaults.
- Every retained module/function/class/public variable needs a current role: productive API, internal implementation, validated research diagnostic, or explicitly temporary compatibility path.
- Compatibility code without a named consumer and removal criterion is deleted.
- Split files only when cohesion improves; do not replace one monolith with trivial wrappers.
- Numerical kernels stay independent of filesystem/config-discovery/orchestration policy.
- Dataset/QA code must not decide retrieval science.
- Cleanup must not silently alter an equation, threshold, calibration assumption or uncertainty model.
- Missing scientific/instrumental settings fail early unless an explicit unavailable/legacy policy exists.
- Unsupported data remain unsupported/NaN; no filling, clipping or interpolation is introduced merely to extend a retrieval.

## 2. Current audit snapshot

### Stable foundations

- [x] Stage-specific strict L0/L1/L2 configuration is in use; obsolete generic top-level `physics` config and global schema are removed.
- [x] Station/site/instrument truth is owned by `station.yaml`; historical profile/calibration IDs persist in products.
- [x] L0 requires traceable acquisition metadata and explicit weather/dark-current/SCC policies.
- [x] L1 materializes canonical pressure/temperature with atmosphere-source provenance; CPT/LRT share the thermal kernel.
- [x] L2 has explicit wavelengths, temporal blocks, LR/MC/KFS, gluing, Rayleigh and cloud-policy configuration.
- [x] AN/PC gluing, post-QA single-channel fallback and retrieval-input rejection reasons are explicit.
- [x] Unknown physical PC saturation remains `not_characterized`; the 10% dead-time occupancy rule remains explicitly provisional.
- [x] Rayleigh altitude limits are search bounds, not a requirement that the entire far-range band be positive.
- [x] Rayleigh reference width is physical (`ref_window_m`) and converted on the actual grid.
- [x] Productive elastic retrieval identity is backward Klett–Fernald across config, provenance, dataset wording and operational logs.
- [x] Incremental L2 reuse rejects stale/contradictory KFS metadata.
- [x] Redundant Level 2 QA status TXT is removed.
- [x] Real case `20251107sapm` reproduced 100% gluing and 5/5 backward-KFS blocks at 355/532 nm after P1 lot 2.
- [x] QA sparse-support statistics no longer emit NumPy all-NaN/low-DOF warnings; unsupported bins remain NaN.
- [ ] No CI/status checks are attached to this branch; do not claim the complete suite is green until CI/full pytest exists.

## 3. Priority order

### P0 — scientific identity / public consistency — COMPLETE

- [x] Canonical productive `backward` KFS mode and FAIR-readable description.
- [x] Versioned scientific metadata publishes backward integration.
- [x] Dataset/failure wording matches the productive method.
- [x] Stale two-sided outputs are not incrementally reused.
- [x] Redundant QA status TXT removed.
- [x] Future vertical-support/top semantics frozen before schema implementation.

Acceptance gate: **passed in implementation and real-data baseline.**

### P1 — remove import-order behavior, duplicate science and obsolete compatibility

#### Lot 2 — COMPLETE + REAL-DATA VALIDATED

- [x] Removed all scientific monkey patches from `milgrau.level2.__init__`.
- [x] Productive signal selection directly uses `evaluate_retrieval_input_supported_domain()`.
- [x] Productive Rayleigh/KFS aggregation directly requires Rayleigh QA + backward KFS, not the forward branch.
- [x] Removed `backward_retrieval.py` and `scientific_policy.py` after their responsibilities became canonical.
- [x] Replaced wildcard `_retrieval_impl` exposure with explicit orchestration.
- [x] Restored the pre-lot package API after a manual run exposed invalid invented cloud-screening re-exports; import regressions now pin the real cloud API.
- [x] `20251107sapm` post-lot run reproduced references 5749 m (355) and 5816 m (532), with 5/5 valid backward blocks for both wavelengths.

#### Lot 3 — IMPLEMENTATION COMPLETE; REAL-DATA RE-RUN PENDING

- [x] Deleted `_retrieval_impl.py` rather than preserving it as a compatibility wrapper: all 1245 legacy/duplicate lines are gone.
- [x] `signal_selection.py` now owns `WavelengthBlockInputs`, `BlockGluingResult`, blocking, source selection, state validation and gluing orchestration.
- [x] `optical_retrieval.py` now owns `MolecularModel`, Rayleigh acceptance/calibration helpers, branch diagnostics, strict KFS orchestration and backward aggregation.
- [x] `result_assembly.py` owns block→time expansion and construction of the public `WavelengthRetrievalResult` contract.
- [x] `block_average.py` owns generic accepted-block mean/error aggregation used by both optical products and result assembly.
- [x] `retrieval.py` is now an explicit orchestration boundary; it no longer depends on the deleted monolith.
- [x] Removed the legacy `_evaluate_retrieval_input()` path; supported-domain QA is the only productive input-QA implementation.
- [x] Removed the duplicate legacy `glue_signal_blocks()`, `retrieve_optical_blocks()`, `process_wavelength()` and result assembly implementations.
- [x] Removed legacy Level 2 atmosphere reconstruction based on obsolete `site`/`physics` config. Productive L2 only consumes atmosphere materialized by current L1.
- [x] Deleted `milgrau/level2/atmosphere.py`; the shared atmosphere kernel is `milgrau.physics.atmosphere` and no Level 2 compatibility alias is retained without a consumer.
- [x] Productive Rayleigh QA now direct-indexes strict slope/variance/min-valid settings; no local literal scientific fallbacks remain there.
- [x] Productive KFS profile orchestration obtains MC/reference/LR-bound settings from `get_kfs_config()`; no local 300/0.10/10 sr/etc. semantic fallbacks remain.
- [x] Regression verifies incomplete Rayleigh/KFS settings fail instead of silently taking local defaults.
- [x] Regression verifies productive KFS calls the multi-mode numerical kernel with explicit `mode="backward"`.
- [x] Tests that exercised old compatibility owners now import the canonical owner (`signal_selection`, `optical_retrieval`, `milgrau.physics.atmosphere`).
- [x] Import regression requires both `_retrieval_impl` and `level2.atmosphere` to be absent.
- [ ] Re-run `20251107sapm` after lot 3 and require the same baseline gluing/reference/backward-block diagnostics before closing P1 operationally.

Low-level KFS note: `level2.kfs` intentionally remains a multi-mode numerical/research kernel (`backward`, `forward`, `two_sided`). Productive L2 never relies on its direction default; the canonical wrapper passes explicit backward mode and that contract is regression-tested. Whether direct research-kernel mode should become a required argument, and which low-level symbols deserve package-level re-export, is deferred to the P2 public-API audit rather than mixed into this no-equation-change cleanup.

P1 acceptance gate: **implementation satisfied; one post-lot-3 real-data equivalence run remains.** No productive path depends on import side effects, wildcard imports, duplicate retrieval implementations, obsolete atmosphere reconstruction, or local KFS/Rayleigh semantic defaults.

### P2 — repository code-use audit, concision and automated guardrails — NEXT

Order inside P2:

1. [ ] Build a repository-wide symbol/module inventory: productive public API, internal implementation, research diagnostic, compatibility, unused.
2. [ ] Audit every `__all__`/package re-export; retain only documented/tested public symbols. In particular, decide the intended public status of low-level gluing/KFS research kernels.
3. [ ] Identify every compatibility path and its actual consumer/removal criterion; delete paths with no consumer.
4. [ ] Add lightweight Ruff (or equivalent) checks for unused imports/variables, unreachable/dead code and basic neutral style rules.
5. [ ] Add a guard against `config.get(..., scientific_literal_default)` in productive scientific paths.
6. [ ] Search for duplicated equations/selection rules and retain one canonical implementation.
7. [ ] Review broad `except Exception`; keep only intentional orchestration/optional-diagnostic boundaries.
8. [ ] Audit QA/display-only helpers for real consumers and remove obsolete legacy helpers.
9. [ ] Review remaining large mixed-responsibility files (`dataset.py`, `viz/level2_qa.py`, explorer app) and split only where cohesion demonstrably improves.
10. [ ] Add CI for imports, static checks and full pytest; only then consider branch protection.

P2 acceptance gate: no known unused compatibility code, intentional public API only, no productive hidden semantic defaults, and automated checks prevent these patterns from returning.

### P3 — Level 2 schema / FAIR metadata / focused documentation

- [ ] Choose canonical aggregate variable names and migration policy for `aerosol_backscatter[_mean]` / `aerosol_extinction[_mean]` aliases.
- [ ] Audit every L2 physical variable for units, dimensions, long_name/description and missing/NaN semantics.
- [ ] Add explicit units for aerosol backscatter/extinction and uncertainty fields wherever currently implied.
- [ ] Audit numeric flag metadata for CF-compatible representation.
- [ ] Add independent product-schema/method version if schema/method evolution needs a contract beyond package CalVer.
- [ ] Name/version/document gluing selection-score constants; no anonymous algorithmic weights.
- [ ] Decide readable immutable input-manifest policy.
- [ ] Add focused docs: processing levels, configuration, station catalog, L0/L1/L2 products, scientific methods, provenance, flags, limitations and validation; then shorten README into an entry point.
- [ ] Build a verified primary-source bibliography for methods actually implemented.

P3 acceptance gate: a Level 2 NetCDF is scientifically interpretable without reading implementation source and cannot describe a method different from the one used.

### P4 — observational/scientific evidence tasks — PARALLEL

- [ ] Characterize physical PC saturation for operational SPU settings using AN/PC overlap and preferably controlled attenuation before writing traceable `max_rate_mhz`.
- [ ] Move any surviving provisional PC guard to raw observed L1 PC rate rather than a corrected/background-subtracted proxy.
- [ ] Audit Level 1 dark-current subtraction versus nonlinear dead-time correction with evidence; do not change it as collateral cleanup.
- [ ] Characterize propagated-error SNR on SPU data before any hard Rayleigh-window SNR gate.
- [ ] Validate cloud/layer screening on SPU observations before making it a productive reference-window gate.
- [ ] Compare real retrievals against an independent/reference chain (LPP and SCC/ELDA methodological expectations where appropriate) without claiming numerical identity.

Evidence tasks must not be replaced by invented software constants.

### P5 — future high-column Level 2 redesign (`fixing_l2`)

Recreate `fixing_l2` only after P1 is real-data validated and the core P2 cleanup/guardrails are stable. Detailed scientific requirements are in Section 6.

## 4. Processing-level follow-up outside the main P0–P5 queue

### Configuration / station ownership

- [x] Processing recipe and observational reality are separated between `config.yaml` and `station.yaml`.
- [x] Historical calibration/profile identity is resolved and persisted.
- [ ] Validate unknown keys with full paths across every remaining auxiliary config path.
- [ ] Extend typed/resolved accessors to auxiliary paths that still reinterpret mappings locally.
- [ ] Add legacy ADC/range overrides only if real Licel evidence demonstrates they are needed.

### Level 0

- [x] Raw discovery/quarantine, shot tolerance, timing and dark-current association are explicit.
- [x] No invented weather, SCC ID, ADC/range or DAQ metadata.
- [x] Incremental currentness requires structural contract + readable provenance.
- [ ] Finish auxiliary semantic-default/unknown-key audit under P2.

### Level 1

- [x] PC Poisson uncertainty uses observed counts before dark subtraction.
- [x] Missing channel calibration follows explicit policy; neutral legacy use is persisted.
- [x] Background, numerical dead-time clipping, PBL and atmosphere source are explicit.
- [x] Productive PBL does not substitute another channel.
- [x] Radiosonde/ERA5/USSA76 ownership and historical geometry are explicit; ERA5 is pinned to CDS.
- [x] Canonical thermodynamics are materialized in L1 and consumed directly by L2.
- [ ] PC correction-order scientific audit remains P4.

### Level 2 baseline

- [x] Productive backward KFS identity is canonical.
- [x] No productive 60 sr / 10 sr LR fallback.
- [x] Rayleigh search permits invalid far-range edge samples without fabricating values.
- [x] Physical reference-window width is grid-independent.
- [x] Productive runtime has explicit signal-selection and optical-retrieval boundaries.
- [x] Legacy Level 2 retrieval monolith and atmosphere compatibility alias are removed.
- [ ] Migrate gluing spatial search/window settings to physical units only when that API is deliberately revised.
- [ ] Version gluing score constants under P3.
- [ ] Keep cloud/SNR productive gates disabled until P4 evidence supports them.

### Runtime / QA / logging

- [x] Main CLIs share the common execution model; LEBEAR supports explicit time windows.
- [x] INFO reports outcomes while detailed diagnostic material belongs in DEBUG.
- [x] Optional QA failure is isolated from product generation.
- [x] QA product-status TXT is removed.
- [x] QA block SEM is warning-free on sparse/unsupported upper columns without fabricating data.
- [ ] Audit QA helper usage under P2.

## 5. Validation status and immediate gate

Committed regression coverage includes strict configuration, station/acquisition, atmosphere, gluing, supported-domain QA, Rayleigh window, KFS science/MC, backward aggregation, provenance/currentness, CLI/logging and Level 2 QA statistics.

P1-specific guards now cover:

- [x] No package-import scientific monkey patching.
- [x] No `_retrieval_impl` module remains.
- [x] No `level2.atmosphere` compatibility module remains.
- [x] Signal selection uses the supported-domain QA directly.
- [x] Backward aggregation ignores the unrequested forward branch for productive success.
- [x] Rayleigh/KFS orchestration does not hide missing scientific configuration behind local defaults.
- [x] Productive KFS passes explicit backward mode to the research kernel.
- [x] Current L2 atmosphere boundary rejects old L1 files that do not contain materialized canonical thermodynamics.
- [x] Pre-lot-3 real baseline: `20251107sapm` = 355 ref 5749 m, 532 ref 5816 m, 5/5 backward blocks each.
- [ ] **Immediate manual gate:** rerun `milgrau-lebear -i 20251107sapm --force` on the lot-3 HEAD and require equivalent diagnostics with no new warning/error before marking P1 closed.
- [ ] Full pytest/static suite in CI. Until this exists, do not label the repository globally green.

## 6. Future high-column Level 2 redesign — inherited `fixing_l2` objectives

### L0. Non-negotiable scientific rules

- [x] `target_top_altitude_m` is a target, never permission to extrapolate.
- [x] Do not turn NaN/non-positive/noisy samples into positive signal merely to extend coverage.
- [x] Do not interpolate across internal invalid intervals to make KFS continuous.
- [x] Backward retrieval never implies values above its boundary condition.
- [x] A lower cascade segment inherits its boundary from an accepted upper solution; never reset `SR_ref=1` at every segment.
- [x] Multiple reference solutions are a sensitivity/ensemble experiment; disagreement contributes to uncertainty.
- [x] Elastic extinction remains conditional on assumed aerosol lidar ratio.
- [x] 20-minute block retrieval and long-mean/high-column retrieval are distinct products/diagnostics.
- [x] Selection, merge and uncertainty policies must be explicit and persisted in readable provenance.

### L1. Vertical-support semantics — FROZEN, NOT YET IN SCHEMA

- [x] `retrieval_support_flag(block_time, wavelength, altitude)` means the final productive optical retrieval is scientifically supported and usable at that bin after source selection, Rayleigh QA, KFS/candidate QA and accepted merge. It is not merely `isfinite(product)` and does not encode source/member identity.
- [x] Current backward-only support includes only bins actually solved through the accepted reference; unsupported outer bins and terminated branches remain NaN and internal gaps are never bridged.
- [x] A future high-column candidate/cascade can extend support only where it has its own valid physical support.
- [x] `retrieval_top_altitude_m` is the highest altitude with support flag 1; NaN when no supported bin exists. Never derive it independently from the last finite array element.
- [x] Backbone/mean support is derived from its own accepted product/support mask, not copied from block products.
- [ ] Add synthetic support/top tests before adding these variables to NetCDF.
- [ ] Add a machine-readable baseline summary for `20251107sapm` without committing raw observational data.

### L2. QA/output semantics

- [x] Redundant status TXT removed.
- [ ] After support enters the schema, mark retrieval support/top in KFS and scattering-ratio QA.
- [ ] Plot accepted reference windows explicitly and never present unsupported upper-tail signal diagnostics as aerosol retrieval.

### L3. Physical Rayleigh window — COMPLETE

- [x] `ref_window_m` replaces fixed productive bin width.
- [x] Width converts on the real uniform altitude grid with regression coverage.
- [x] Search bounds remain in meters and invalid geometry fails.

### L4. Rayleigh candidate catalogue

- [ ] Return every window that passes minimum scientific QA instead of a single minimum-cost candidate.
- [ ] Persist candidate start/stop/center, valid fraction, slope, variance, calibration factor, free-intercept diagnostic, SNR/uncertainty diagnostic and future validated layer flag.
- [ ] Separate pass/fail criteria from ranking criteria.
- [ ] Prefer high altitude only among already-valid candidates.
- [ ] Define minimum separation/correlation so heavily overlapping windows do not masquerade as independent ensemble evidence.
- [ ] Persist compact accepted/rejected reasoning.
- [ ] Do not add hard SNR/cloud gates before P4 validation.

### L5. High-column backbone

- [ ] Add a long-mean retrieval input distinct from 20-minute block products; averaging duration is explicit/provenanced.
- [ ] Evaluate altitude-dependent vertical aggregation only if evidence requires it; widths are physical and original-resolution signal remains available.
- [ ] Select high reference candidates from the backbone.
- [ ] Treat ~20 km as an initial target, not a success condition; publish a lower top if no trustworthy boundary exists above it.

### L6. Multi-reference ensemble

- [ ] Run backward KFS from multiple accepted high references.
- [ ] Preserve member-specific Rayleigh/MC diagnostics.
- [ ] Combine only members whose physical backward support contains the altitude and whose QA passed.
- [ ] Define explicit uncertainty/quality weights; no anonymous heuristic weights.
- [ ] Add between-reference spread as an uncertainty component.
- [ ] Keep member solutions internally auditable/testable.

Acceptance gate: reference sensitivity increases exposed uncertainty instead of disappearing in an average.

### L7. Cascaded backward retrieval

- [ ] Define ordered overlapping high→low segments.
- [ ] Solve the highest segment from a valid high/molecular boundary.
- [ ] Each lower segment inherits `beta_total_ref` / scattering ratio from the accepted upper solution.
- [ ] Propagate inherited-boundary uncertainty through lower-segment Monte Carlo.
- [ ] Require adequate overlap/support and reject uncertainty-inconsistent handoffs.
- [ ] Never bridge unsupported internal gaps or reset `SR_ref=1` because a new segment begins.

Acceptance gate: cascade reproduces a well-conditioned synthetic full-column truth within a defined tolerance and rejects inconsistent handoffs.

### L8. Overlap merge / uncertainty model

- [ ] Implement documented smooth overlap merge only after segment acceptance; valid weights sum to one.
- [ ] Never average valid evidence with NaN/invalid evidence.
- [ ] Check value and vertical-gradient continuity across merge regions.
- [ ] Separate measurement/MC, LR, boundary, reference-choice, cascade-handoff and optional aggregation uncertainty components.
- [ ] Treat covariance deliberately; do not blindly quadrature-sum correlated terms.
- [ ] Verify uncertainty increases when reference ambiguity is introduced.

### L9. Redesigned product / FAIR concepts

Names become public only after support tests/schema decisions are stable.

- [ ] `retrieval_support_flag[..., altitude]`.
- [ ] `retrieval_top_altitude_m[...]` for block and backbone/aggregate products.
- [ ] Accepted reference-member count/diagnostics and ensemble-spread uncertainty.
- [ ] Cascade segment/handoff/merge diagnostics.
- [ ] Clear separation of long-mean/backbone from 20-minute products.
- [ ] Explicit elastic-extinction dependence on assumed aerosol LR.
- [ ] Independent redesigned-method/schema version; pre-redesign outputs become incrementally stale when required.
- [ ] QA overview shows signal, molecular fit, reference candidates, support top, ensemble spread and cascade/merge regions without visually legitimizing unsupported upper tails.

### L10. Validation / merge gate

Synthetic:

- [ ] Pure molecular → approximately zero aerosol backscatter within defined tolerance.
- [ ] Known aerosol layers → recover truth within defined tolerance.
- [ ] Multiple valid references → stable ensemble.
- [ ] Contaminated reference → rejected or visibly inflates uncertainty.
- [ ] Upper-tail noise → support top falls gracefully.
- [ ] Internal invalid gap → explicit failure/no bridging.
- [ ] Segment handoff → no stitching discontinuity beyond tolerance.

Real SPU:

- [ ] `20251107sapm`: materially extend defensible coverage while preserving/explaining the current lower-column solution.
- [ ] Evaluate whether ~15 km is supported; attempt 20 km only with a valid boundary/support above the target.
- [ ] Test clear, high-aerosol, cloud-contaminated, weak-signal and different AN/PC-dominance cases.
- [ ] Compare with LPP where practical and SCC/ELDA methodological expectations without claiming numerical identity.

Merge criterion: **maximize validated vertical support, expose where support ends, and remain stable/traceable under reference/noise sensitivity. Reaching 20 km alone is not success.**

## 7. Code-organization acceptance checklist

Use this for every subsequent batch.

- [ ] Every retained symbol/module has a real consumer or documented productive/research role.
- [x] No productive scientific behavior is installed by import side effect.
- [x] No wildcard import defines productive Level 2 behavior.
- [x] No duplicate legacy Level 2 retrieval monolith remains.
- [x] No unused Level 2 atmosphere compatibility wrapper remains.
- [ ] Package/public aliases are canonical or have an explicit deprecation window.
- [ ] Scientific constants/weights have named/versioned meaning where they affect method selection.
- [ ] Configuration is resolved through strict accessors rather than local semantic literals.
- [x] Numerical KFS/gluing/molecular kernels do not own filesystem policy.
- [x] Orchestration coordinates stages without duplicating the removed retrieval implementations.
- [ ] Dataset construction defines schema/metadata without duplicated scientific decisions.
- [x] QA/visualization does not feed back into retrieval decisions.
- [x] Current tests point to canonical owner modules rather than removed compatibility adapters.
- [x] Lot 3 reduced conceptual and physical duplication rather than merely moving the monolith.
- [x] New boundaries correspond to real responsibilities: source selection, optical retrieval and result assembly.

## 8. Recommended batches from here

1. [x] **P0 lot 1 — backward identity + stale-output guard + QA TXT removal + support semantics.**
2. [x] **P1 lot 2 — remove monkey patches/wildcard behavior and direct-wire supported-domain QA/backward aggregation.**
3. [x] **P1 lot 3 implementation — delete retrieval monolith/atmosphere alias, move real responsibilities to canonical owners, remove local scientific defaults.**
4. [ ] **P1 lot 3 validation — rerun `20251107sapm` and close P1 if diagnostics remain equivalent.**
5. [ ] **P2 lot 4 — symbol/public-API/dead-code inventory + Ruff/static semantic-default guardrails.**
6. [ ] **P2 lot 5 — CI/full pytest + cleanup findings from static audit.**
7. [ ] **P3 — L2 schema aliases/units/flags/method version + focused docs.**
8. [ ] **Validation baseline — synthetic vertical-support tests + machine-readable real-case summary.**
9. [ ] **Recreate `fixing_l2` from the cleaned base and start candidate-catalogue/high-column work.**

P4 physical PC/SNR/cloud evidence work can proceed in parallel.

## 9. Implementation log

### 2026-09-09 to 2026-09-14 — strict/scientific baseline

- Introduced stage-specific fail-fast configuration, station calibration history, atmosphere/weather policies, strict acquisition metadata, contextual logging, readable FAIR provenance and simplified execution semantics.
- Corrected guarded PC use without inventing physical saturation characterization; established post-gluing fallback, Rayleigh search semantics and backward Fernald behavior.
- Replaced fixed productive Rayleigh bins with `ref_window_m`; removed global schema/generic physics residue; gated incremental reuse on current provenance.

### 2026-09-14 — architecture/science re-audit + P0

- Re-audited `new-architecture`, promoted code-use/dead-code/concision to a first-class objective and incorporated the saved `fixing_l2` plan into this tracker.
- Canonicalized backward KFS identity in config/scientific metadata/product wording.
- Removed redundant QA status TXT and added focused regressions.

### 2026-09-14 — P1 lot 2

- Removed package-init scientific monkey patches and wildcard monolith exposure.
- Created canonical signal-selection and optical-retrieval boundaries; retired transitional backward/scientific-policy adapters.
- Fixed the package-public cloud import regression exposed by manual execution.
- Real `20251107sapm` reproduced the pre-refactor 355/532 reference and 5/5 backward-block diagnostics.
- Fixed sparse-support QA SEM warnings without altering saved science.

### 2026-09-14 — P1 lot 3 implementation

- Deleted `_retrieval_impl.py` (1245 legacy/duplicate lines) rather than retaining another compatibility layer.
- Deleted the unused `level2.atmosphere` alias; shared atmosphere physics remains in `milgrau.physics.atmosphere`, while productive L2 reads the atmosphere already materialized by L1.
- Moved input/block/gluing state and selection logic into `signal_selection.py`; Rayleigh/KFS helpers and molecular retrieval state into `optical_retrieval.py`; block→public-result construction into `result_assembly.py`; generic accepted-block aggregation into `block_average.py`.
- Removed the obsolete full-band input QA, legacy atmosphere reconstruction, duplicate optical aggregation and duplicate orchestration paths with the monolith.
- Removed local Rayleigh and KFS semantic defaults from productive orchestration; strict config is now authoritative.
- Updated tests to target canonical owners and added regressions for absent legacy modules, missing-config failure and explicit productive backward kernel mode.
- No Fernald equation, Rayleigh/gluing threshold, lidar-ratio climatology, Monte Carlo uncertainty model, NetCDF schema or provisional PC policy was intentionally changed in this lot.
- Full-suite green is not claimed because the branch still has no CI/status checks; the next mandatory action is the real-data equivalence rerun listed above.
