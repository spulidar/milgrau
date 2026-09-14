# MILGRAU architecture, scientific quality and Level 2 roadmap

Branch: `new-architecture`

Audit baseline: `c6400a941964de73438132afdb7f3db9a5916f32` (2026-09-14)

This file is the source of truth for preparing a clean, scientifically defensible base before recreating `fixing_l2`. Priorities are intentional: scientific correctness and truthful provenance first; one canonical implementation second; code-use/static/schema hardening third; high-column retrieval only after those gates are stable.

## 1. Engineering and scientific rules

MILGRAU should be FAIR, concise, testable and easy to audit.

- `config.yaml` owns the processing/scientific recipe.
- `station.yaml` owns station/instrument reality, hardware history, calibration, station-derived climatology and SCC mapping.
- Python owns equations, physical constants, validated runtime objects and implementation details.
- One productive scientific behavior has one canonical implementation.
- Productive science must not depend on import order, monkey patching, wildcard imports or hidden semantic defaults.
- Every retained module/function/class/public variable needs a current role: productive API, productive internal implementation, validated research/diagnostic API, optional UI, or explicitly temporary compatibility.
- Compatibility without a named consumer and removal criterion is deleted.
- Split files only when cohesion improves; do not replace one monolith with trivial wrappers.
- Numerical kernels stay independent of filesystem/config-discovery/orchestration policy.
- Dataset and QA code describe/check products; they do not choose retrieval science.
- Cleanup must not silently alter an equation, threshold, calibration assumption or uncertainty model.
- Missing scientific/instrumental settings fail early unless an explicit unavailable/legacy policy exists.
- Unsupported data remain unsupported/NaN; no filling, clipping or interpolation is introduced merely to extend a retrieval.
- A real-data successful run validates the exercised path, not the entire scientific method or full test suite.

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
- [x] Redundant Level 2 QA product-status TXT is removed.
- [x] QA sparse-support statistics no longer emit all-NaN/low-DOF warnings; unsupported bins remain NaN.
- [x] P1 real-data baseline is reproduced after deleting the legacy Level 2 monolith.
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

### P1 — remove import-order behavior, duplicate science and obsolete compatibility — COMPLETE

#### Lot 2 — complete + real-data validated

- [x] Removed scientific monkey patches from `milgrau.level2.__init__`.
- [x] Productive signal selection directly uses `evaluate_retrieval_input_supported_domain()`.
- [x] Productive Rayleigh/KFS aggregation requires Rayleigh QA + backward KFS, not the forward branch.
- [x] Removed `backward_retrieval.py` and `scientific_policy.py` after their responsibilities became canonical.
- [x] Replaced wildcard `_retrieval_impl` exposure with explicit orchestration.
- [x] Restored the real package cloud API after manual execution exposed stale invented re-exports.
- [x] Real `20251107sapm` reproduced references 5749 m (355) and 5816 m (532), with 5/5 backward blocks for both wavelengths.

#### Lot 3 — complete + real-data validated

- [x] Deleted `_retrieval_impl.py`; all 1245 legacy/duplicate lines are gone rather than hidden behind another compatibility wrapper.
- [x] `signal_selection.py` owns `WavelengthBlockInputs`, `BlockGluingResult`, blocking, source selection, state validation and gluing orchestration.
- [x] `optical_retrieval.py` owns `MolecularModel`, Rayleigh acceptance/calibration helpers, branch diagnostics, strict KFS orchestration and backward aggregation.
- [x] `result_assembly.py` owns block→time expansion and construction of `WavelengthRetrievalResult`.
- [x] `block_average.py` owns generic accepted-block mean/error aggregation.
- [x] `retrieval.py` is an explicit one-wavelength orchestration boundary and no longer depends on the deleted monolith.
- [x] Removed legacy full-band `_evaluate_retrieval_input()`; supported-domain QA is the productive input-QA implementation.
- [x] Removed duplicate legacy gluing, optical retrieval, process-wavelength and result-assembly implementations.
- [x] Removed legacy Level 2 atmosphere reconstruction from obsolete `site`/`physics` settings.
- [x] Deleted `milgrau/level2/atmosphere.py`; shared atmosphere physics is `milgrau.physics.atmosphere`, while productive L2 consumes canonical atmosphere materialized by L1.
- [x] Productive Rayleigh QA direct-indexes strict slope/variance/min-valid settings; no local scientific threshold defaults remain there.
- [x] Productive KFS orchestration gets MC/reference/LR-bound settings from `get_kfs_config()`; no local `300`, `0.10`, `10 sr`, seed or direction fallback remains in the productive wrapper.
- [x] Regression verifies incomplete Rayleigh/KFS settings fail rather than silently taking local defaults.
- [x] Regression verifies productive KFS calls the multi-mode numerical kernel with explicit `mode="backward"`.
- [x] Tests target canonical owners and require `_retrieval_impl` and `level2.atmosphere` to remain absent.

#### P1 final operational gate — PASSED on `20251107sapm`

Manual command:

```bash
milgrau-lebear -i 20251107sapm --force
```

Observed after lot 3:

- [x] 355 nm provisional PC guard remained explicit: occupancy 0.100, nominal observed-rate proxy 50.00 MHz, physical saturation still uncharacterized.
- [x] 355 nm gluing success = 100.0%, no single-channel fallback.
- [x] 355 nm Rayleigh reference = 5749 m, window `[5254, 6244] m`, valid = 100.0%, slope = 0.000, variance = 0.001, backward KFS = 5/5 blocks.
- [x] 532 nm provisional PC guard remained explicit: occupancy 0.100, nominal observed-rate proxy 28.57 MHz, physical saturation still uncharacterized.
- [x] 532 nm gluing success = 100.0%, no single-channel fallback.
- [x] 532 nm Rayleigh reference = 5816 m, window `[5321, 6311] m`, valid = 100.0%, slope = 0.000, variance = 0.001, backward KFS = 5/5 blocks.
- [x] Product writing completed: `wavelengths=2/2`, `20251107sapm_level2_optical.nc`.
- [x] Final runtime summary reported zero errors.
- [x] No previous sparse/all-NaN variance warning reappeared in the supplied run log.

The two PC messages above are expected scientific-status warnings, not failures: they explicitly preserve the fact that physical detector saturation is still uncharacterized.

Low-level KFS note: `level2.kfs` intentionally remains a multi-mode numerical/research kernel (`backward`, `forward`, `two_sided`). Productive L2 never relies on its direction default; the canonical wrapper passes explicit backward mode. Whether direct research-kernel mode should become mandatory and whether the low-level kernels remain package-level exports is a P2 public-API decision.

P1 acceptance gate: **passed.** The exercised real-data path confirms `PC guard → gluing → Rayleigh search → backward KFS → result/dataset assembly` after the monolith deletion. This is not a claim that the full repository test suite or external scientific validation is complete.

### P2 — repository code-use audit, concision and automated guardrails — IN PROGRESS

A module/API ownership baseline now lives in `docs/code_inventory.md`.

#### Lot 4 — public API / inventory / static guardrails — STARTED

- [x] Added an explicit code-role taxonomy: productive public, productive internal, research/diagnostic, optional UI, compatibility, unused.
- [x] Added a repository module-ownership baseline and detailed Level 2 module inventory in `docs/code_inventory.md`.
- [x] Classified Level 2 low-level KFS/gluing kernels as intentionally retained research/numerical API; their availability does not alter productive backward policy.
- [x] Corrected root `milgrau.__all__`: it now advertises only the actually bound lightweight root symbol `__version__`; subpackages remain explicit imports rather than eager root imports.
- [x] Added regression that every advertised `__all__` symbol resolves to a real bound object.
- [x] Added repository AST regression rejecting wildcard imports under `milgrau/`.
- [x] Added productive-Level-2 AST regression rejecting `.get(..., fallback)` on scientific/config mappings in retrieval/selection/input-QA/optical orchestration.
- [x] Added Ruff as a development dependency with a deliberately neutral initial rule set: `E4`, `E7`, `E9`, `F`.
- [x] Identified first-pass cohesion-review candidates by responsibility/size; no file is split merely for being large.
- [ ] Run Ruff on the current branch and classify every finding before enabling it in CI.
- [ ] Complete per-symbol consumer audit for package re-exports in `io`, `level0`, `level1`, `level2` and `viz`; retain documented public/research APIs and remove accidental exports only with an explicit API decision.
- [ ] Audit compatibility/deprecation paths outside Level 2 and require a named consumer + removal criterion.
- [ ] Extend semantic-default audit beyond the canonical L2 scientific path to remaining L0/L1/auxiliary config interpretation.
- [ ] Search repository-wide for duplicate equations/selection rules and keep one canonical implementation.
- [ ] Review broad `except Exception`; retain only intentional orchestration/optional-diagnostic containment.
- [ ] Audit QA/display-only helpers for real consumers and remove obsolete helpers.
- [ ] Review `dataset.py`, `viz/level2_qa.py`, `explorer/streamlit_app.py` and other large candidates by cohesion before splitting anything.

#### Lot 5 — CI and cleanup findings — PENDING

- [ ] Fix/justify Ruff findings from lot 4.
- [ ] Run focused + full pytest in a reproducible environment.
- [ ] Add CI for import smoke tests, Ruff/static guards and full pytest.
- [ ] Add branch protection only after CI is stable enough not to create false confidence/noise.

P2 acceptance gate: no known unused compatibility code, intentional/documented public API only, no productive hidden semantic defaults, and automated checks prevent those patterns from returning.

### P3 — Level 2 schema / FAIR metadata / focused documentation

- [ ] Choose canonical aggregate names and migration policy for `aerosol_backscatter[_mean]` / `aerosol_extinction[_mean]` aliases.
- [ ] Audit every L2 physical variable for units, dimensions, `long_name`/description and missing/NaN semantics.
- [ ] Add explicit units for aerosol backscatter/extinction and uncertainty fields wherever currently implied.
- [ ] Audit numeric flag metadata for CF-compatible representation.
- [ ] Add an independent product-schema/method version if schema/method evolution needs a contract beyond package CalVer.
- [ ] Name/version/document gluing selection-score constants; no anonymous algorithmic weights.
- [ ] Decide readable immutable input-manifest policy.
- [ ] Add focused docs: processing levels, configuration, station catalog, L0/L1/L2 products, methods, provenance, flags, limitations and validation; then shorten README into an entry point.
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

Recreate `fixing_l2` only after P1 is complete and the core P2 cleanup/guardrails are stable. Detailed scientific requirements are in Section 6.

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
- [x] Post-P1 real case reaches product assembly with 5/5 backward blocks at both wavelengths.
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

## 5. Validation status

Committed regression coverage includes strict configuration, station/acquisition, atmosphere, gluing, supported-domain QA, Rayleigh window, KFS science/MC, backward aggregation, provenance/currentness, CLI/logging and Level 2 QA statistics.

P1-specific guards cover:

- [x] No package-import scientific monkey patching.
- [x] No `_retrieval_impl` module.
- [x] No `level2.atmosphere` compatibility module.
- [x] Signal selection uses supported-domain QA directly.
- [x] Backward aggregation ignores the unrequested forward branch for productive success.
- [x] Rayleigh/KFS orchestration fails on missing scientific config instead of taking local defaults.
- [x] Productive KFS passes explicit backward mode to the research kernel.
- [x] Current L2 atmosphere boundary rejects old L1 files without materialized canonical thermodynamics.
- [x] Post-lot-3 real baseline: 355 ref 5749 m, 532 ref 5816 m, 5/5 backward blocks each, product written, zero runtime errors.

P2 guardrails already committed:

- [x] Root API is explicit/minimal.
- [x] Package `__all__` entries must resolve.
- [x] Wildcard imports under `milgrau/` are rejected.
- [x] Canonical productive L2 scientific mappings may not use local `.get(..., fallback)` semantics.
- [x] Ruff configuration/dependency is present.
- [ ] Ruff clean run is not yet established.
- [ ] Full pytest/static suite in CI is not yet established.

Until the last two items exist, do not label the repository globally green.

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

- [ ] Return every window that passes minimum scientific QA instead of one minimum-cost candidate.
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

- [ ] Every retained symbol/module has a real consumer or documented productive/research role. Module-level inventory exists; per-symbol consumer audit remains.
- [x] Root package exports only a real lightweight bound symbol.
- [x] No productive scientific behavior is installed by import side effect.
- [x] No wildcard import defines productive behavior; repository AST guard prevents wildcard imports under `milgrau/`.
- [x] No duplicate legacy Level 2 retrieval monolith remains.
- [x] No unused Level 2 atmosphere compatibility wrapper remains.
- [ ] Package/public aliases are canonical or have an explicit deprecation/API decision.
- [ ] Scientific constants/weights have named/versioned meaning where they affect method selection.
- [x] Canonical productive L2 configuration is resolved through strict accessors rather than local semantic defaults; AST regression guards this boundary.
- [ ] Extend strict/default guard audit to remaining L0/L1/auxiliary paths.
- [x] Numerical KFS/gluing/molecular kernels do not own filesystem policy.
- [x] Orchestration coordinates stages without duplicating removed retrieval implementations.
- [ ] Dataset construction must continue to define schema/metadata without duplicated scientific decisions; detailed P3 audit pending.
- [x] QA/visualization does not feed back into retrieval decisions.
- [x] Current tests point to canonical owner modules rather than removed compatibility adapters.
- [x] P1 reduced conceptual and physical duplication rather than moving the monolith.
- [x] New Level 2 boundaries correspond to real responsibilities: source selection, optical retrieval and result assembly.
- [ ] Ruff/full pytest clean baseline and CI pending.

## 8. Recommended batches from here

1. [x] **P0 lot 1 — backward identity + stale-output guard + QA TXT removal + support semantics.**
2. [x] **P1 lot 2 — remove monkey patches/wildcard behavior and direct-wire supported-domain QA/backward aggregation.**
3. [x] **P1 lot 3 — delete retrieval monolith/atmosphere alias, move responsibilities to canonical owners, remove local scientific defaults.**
4. [x] **P1 lot 3 validation — `20251107sapm` reproduced the baseline and completed with zero errors.**
5. [ ] **P2 lot 4 — IN PROGRESS: module/API inventory + public-surface cleanup + Ruff/static semantic-default guardrails + consumer audit.**
6. [ ] **P2 lot 5 — fix static findings + CI/full pytest.**
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

### 2026-09-14 — P1 lot 3

- Deleted `_retrieval_impl.py` (1245 legacy/duplicate lines) rather than retaining another compatibility layer.
- Deleted unused `level2.atmosphere`; shared atmosphere physics remains in `milgrau.physics.atmosphere`, while productive L2 reads atmosphere materialized by L1.
- Moved input/block/gluing state and selection logic into `signal_selection.py`; Rayleigh/KFS helpers and molecular retrieval state into `optical_retrieval.py`; block→public-result construction into `result_assembly.py`; accepted-block aggregation into `block_average.py`.
- Removed obsolete full-band input QA, legacy atmosphere reconstruction, duplicate optical aggregation and duplicate orchestration paths.
- Removed local Rayleigh and KFS semantic defaults from productive orchestration; strict config is authoritative.
- Updated tests to target canonical owners and added regressions for absent legacy modules, missing-config failure and explicit productive backward kernel mode.
- Real-data rerun then reproduced 355 ref 5749 m / 532 ref 5816 m, 100% gluing, 5/5 backward blocks for both wavelengths, successful product writing and zero errors. P1 is therefore closed.

### 2026-09-14 — P2 lot 4 start

- Added `docs/code_inventory.md` with code-role taxonomy, package ownership, detailed Level 2 module roles, known removed compatibility paths and first-pass large-module review candidates.
- Corrected root `milgrau.__all__` to expose only `__version__`; explicit subpackages remain importable without making root import eager/heavy.
- Documented Level 2 package exports as productive API versus intentionally retained numerical/research kernels.
- Added AST regression that all advertised package exports resolve, rejects wildcard imports, and rejects scientific mapping fallbacks in canonical productive L2 orchestration.
- Added Ruff to the development extra and configured a neutral `E4/E7/E9/F` first pass.
- No Ruff/full-pytest/CI green claim is made yet; those are the next P2 gates.
