# MILGRAU architecture, scientific quality and Level 2 roadmap

Branch: `new-architecture`

Audit baseline: `c6400a941964de73438132afdb7f3db9a5916f32` (2026-09-14)

This file is the source of truth for preparing a concise, scientifically defensible base before recreating `fixing_l2`. Priority is intentional: truthful scientific identity first, one canonical implementation second, code-use/static hardening third, FAIR/schema semantics fourth, and only then the high-column redesign.

## 1. Non-negotiable engineering/scientific rules

- `config.yaml` owns processing/scientific recipe.
- `station.yaml` owns site/instrument reality, history, calibration, SCC mapping and station-derived climatology.
- Python owns equations, physical constants, validated runtime objects and implementation details.
- One productive scientific behavior has one canonical implementation.
- Productive science must not depend on import order, monkey patching, wildcard imports or hidden semantic defaults.
- Every retained symbol/module must have a current role: productive public API, productive internal, research/diagnostic API, optional UI, or temporary compatibility with a named consumer/removal criterion.
- Compatibility without a named consumer is deleted.
- Split files only when cohesion improves; do not replace monoliths with trivial wrappers.
- Numerical kernels stay independent of filesystem/config-discovery/orchestration policy.
- Dataset/QA code may describe/check products but cannot choose retrieval science.
- Cleanup must not silently alter equations, thresholds, calibration assumptions or uncertainty models.
- Missing scientific/instrument settings fail early unless an explicit unavailable/legacy policy exists.
- Unsupported data remain unsupported/NaN; no filling/interpolation is introduced merely to extend retrieval coverage.
- Tests exercise current canonical owners/contracts; obsolete tests are not a reason to restore compatibility code or hidden defaults.
- A successful real-data run validates the exercised path, not the full scientific method or repository test suite.

## 2. Priority/status overview

| Priority | Status | Purpose |
| --- | --- | --- |
| **P0** | **COMPLETE** | truthful backward-KFS identity, output/currentness consistency, remove redundant QA TXT, freeze support semantics |
| **P1** | **COMPLETE + REAL-DATA VALIDATED** | remove import-order behavior, duplicate Level 2 science, monoliths and obsolete compatibility |
| **P2** | **COMPLETE** | code-use audit, deliberate API, dead compatibility/helpers, exception/default guardrails, Ruff/full tests/CI |
| **P3** | **IN PROGRESS** | Level 2 schema/FAIR metadata, method/schema identity, provenance and focused documentation |
| **P4** | parallel evidence work | PC saturation, correction order, SNR/cloud validation, external comparison |
| **P5** | pending | recreate `fixing_l2` and implement scientifically validated high-column retrieval |

## 3. P0 — scientific identity/public consistency — COMPLETE

- [x] Productive `inversion.kfs_mode` is canonically `backward`.
- [x] Scientific metadata and dataset/log wording publish the same backward method.
- [x] Incremental reuse rejects stale/two-sided Level 2 products.
- [x] Redundant `QA_L2_Product_Status_*.txt` removed at the generator.
- [x] Future `retrieval_support_flag` / `retrieval_top_altitude_m` semantics frozen before schema implementation.

Acceptance gate: **passed in implementation and real-data baseline.**

## 4. P1 — canonical Level 2 architecture — COMPLETE

### Lot 2 — import/runtime cleanup

- [x] Removed scientific monkey patches from `milgrau.level2.__init__`.
- [x] Productive signal selection directly uses supported-domain QA.
- [x] Productive optical success requires Rayleigh QA + backward KFS, not a forward branch.
- [x] Removed transitional `backward_retrieval.py` and `scientific_policy.py`.
- [x] Removed wildcard monolith exposure; package import no longer installs science.
- [x] Import regressions pin real public symbols and prohibit reintroduced side effects.

### Lot 3 — monolith/compatibility deletion

- [x] Deleted `_retrieval_impl.py` (1245 legacy/duplicate lines), not replaced by another monolith.
- [x] Deleted obsolete `milgrau/level2/atmosphere.py`; shared atmosphere physics remains in `milgrau.physics.atmosphere`.
- [x] `signal_selection.py` owns blocking/source selection/gluing/fallback.
- [x] `optical_retrieval.py` owns Rayleigh QA/calibration and backward KFS block retrieval/aggregation.
- [x] `result_assembly.py` owns block→typed-result construction.
- [x] `block_average.py` owns generic accepted-block aggregation.
- [x] `retrieval.py` is a one-wavelength orchestration boundary.
- [x] Removed duplicate legacy QA/gluing/retrieval/result-assembly paths.
- [x] Productive Rayleigh/KFS orchestration fails on missing scientific configuration instead of using local literal defaults.
- [x] Productive caller passes explicit `mode="backward"` to the multi-mode research kernel.

### P1 real-data gate — PASSED

`milgrau-lebear -i 20251107sapm --force` after monolith deletion reproduced:

- [x] 355 nm gluing = 100%; Rayleigh reference 5749 m `[5254, 6244] m`; backward KFS 5/5.
- [x] 532 nm gluing = 100%; Rayleigh reference 5816 m `[5321, 6311] m`; backward KFS 5/5.
- [x] Product written for 2/2 wavelengths, zero runtime errors.
- [x] Sparse/all-NaN QA variance warning fixed without filling unsupported bins.
- [x] PC warnings remain explicitly provisional: physical saturation still `not_characterized`.

Acceptance gate: **passed.**

## 5. P2 — code-use audit, concision and automated guardrails — COMPLETE

Detailed ownership decisions live in `docs/code_inventory.md`.

### Lot 4A — inventory/static baseline — COMPLETE

- [x] Added role taxonomy and repository/module ownership inventory.
- [x] Root `milgrau.__all__` reduced to the real lightweight root API: `__version__`.
- [x] Exact supported `__all__` surfaces for `io`, `level0`, `level1`, `level2`, `physics`, and `viz` are regression-pinned.
- [x] AST regression rejects wildcard imports under `milgrau/`.
- [x] Productive Level 2 mappings cannot use `.get(..., scientific_literal_default)`.
- [x] Strict `level0/config.py`, `level1/config.py`, `level2/config.py` are guarded against semantic literal `.get` fallbacks.
- [x] Ruff added to dev dependencies with correctness/dead-code rules `E4`, `E7`, `E9`, `F`; `E731` deliberately excluded as style-only.
- [x] First Ruff pass removed 8 real `F401` imports; subsequent CI Ruff runs remain clean.

### Lot 4B — ownership/compatibility/dead-code/default cleanup — COMPLETE

- [x] Level 1 exports point directly to canonical owners.
- [x] Removed `level1.common.level1_output_path()` compatibility wrapper.
- [x] Removed obsolete `level1.common.get_channel_constant(..., logger)`; calibration owner is `resolve_channel_calibration()`.
- [x] Compatibility-import AST guard prevents tests/package code from reintroducing removed Level 1/2 paths.
- [x] Full-suite stale imports/fixtures were migrated to current owners/contracts instead of restoring compatibility.
- [x] `config.loader` creates no legacy `site`/`hardware`/`physics.channels` compatibility structures; station-owned LR climatology materialization is an intentional productive data view.
- [x] Historical Licel file-level laser-shot fallback retained only as a named file-format compatibility path; channel `NShots` has priority.
- [x] Removed dead QA helpers `viz/level2_qa.py::_legacy_ylim` and `_visual_scale_to_reference`; active display-only `_legacy_scale_factor` remains.
- [x] Removed inert `inversion.gluing.gaussian_threshold` from config/kernel/productive plumbing. Old configs containing it fail explicitly rather than pretending it affects selection.
- [x] Repository-wide duplicate-science review found no second productive implementation for atmosphere, corrections, gluing, Rayleigh/KFS, signal selection or L2 result assembly.
- [x] Research numerical exports/multi-mode kernels are explicitly classified and do not define productive policy.
- [x] Large files reviewed by cohesion; no split performed solely to reduce line count.

### Lot 4C — exception boundaries — COMPLETE

Broad catches are retained only at explicit containment boundaries: outer execution/file orchestration, per-wavelength partial L2 handling, per-channel L1 handling, filesystem mutation APIs returning `ExecutionResult`, product-currentness checks, and optional QA/UI presentation.

Narrowed/regression-tested helper/parser boundaries:

- [x] Level 0 acquisition QA and station-config loading propagate unexpected runtime defects.
- [x] Licel parsing contains malformed/IO failures (`ValueError`, `OSError`) but propagates unexpected runtime defects.
- [x] Open-Meteo cache/retry handles only expected IO/payload failures; an unexpected `RuntimeError` is not retried/converted to ordinary missing weather.
- [x] Level 1 min/max diagnostic reductions contain only conversion failures (`TypeError`, `ValueError`, `OverflowError`); unexpected runtime failures propagate.
- [x] Raw-tree `Path.resolve()` containment is limited to expected `OSError`/`RuntimeError`; filesystem mutation boundaries remain explicit `ExecutionResult` containment.
- [x] Human-readable `ExecutionResult` paths are stable across Windows/Linux separators.

### Lot 5 — full suite, dependencies and CI — COMPLETE

Full-suite progression:

- [x] First complete baseline: **324 passed, 23 failed, 266 warnings**.
- [x] All 23 failures classified before repair; mostly stale contracts/fixtures plus Windows path portability and historical SCC Raman mapping.
- [x] Repair batch implemented without restoring hidden defaults/compatibility.
- [x] Residual inventory failures were stale mocks only and fixed in tests.
- [x] Clean local baseline reproduced: **347 passed, 0 failed, 346 warnings**, Ruff clean, before additional exception-boundary regressions were added.
- [x] Warning baseline classified: NumPy 2.5 `ndarray.shape` deprecation is triggered inside published netCDF4 1.7.4 write paths. Warnings surfacing at MILGRAU caller lines are not a separate MILGRAU deprecation.
- [x] Invalid unpublished `netCDF4>=1.7.4.1` requirement was reverted after pip correctly rejected it.
- [x] Supported published floor is `netCDF4>=1.7.4`; do not pin NumPy backwards or hide the upstream warning merely to get warning count zero.
- [x] User-confirmed environment: NumPy **2.5.3**, netCDF4 **1.7.4**.
- [x] GitHub Actions CI runs Ruff plus full pytest on Ubuntu/Windows × Python 3.12/3.14.
- [x] CI repeatedly passed through the P2 cleanup sequence, including the final code-audit commits before tracker/documentation closure.
- [x] Branch protection/required checks is optional repository policy, not a P2 scientific/code-quality acceptance blocker; branch remains unprotected unless deliberately changed later.

P2 acceptance gate: **passed — deliberate public API, no known unused compatibility/dead helpers, no productive hidden semantic defaults, justified exception boundaries, one canonical owner per productive scientific behavior, local baseline and cross-platform CI guardrails.**

## 6. P3 — Level 2 schema / FAIR metadata / focused docs — IN PROGRESS

P3 improves product interpretation and provenance without changing the validated backward-KFS numerical baseline as collateral work. `docs/level2_schema.md` documents the current storage/support semantics; `milgrau.level2.metadata` is the canonical variable-metadata registry.

### P3.1 — canonical Level 2 schema names — COMPLETE

- [x] Inventoried every current L2 data-variable family, coordinate, completeness field and method/provenance group written by `level2.dataset`.
- [x] Schema v1 makes `aerosol_backscatter_mean`, `aerosol_backscatter_mean_error`, `aerosol_extinction_mean`, and `aerosol_extinction_mean_error` the canonical aggregate names.
- [x] Removed the four exact unsuffixed duplicate NetCDF aliases instead of retaining indefinite compatibility arrays; older/unversioned products are reprocessed.
- [x] Aggregate, block and time-expanded roles are explicit in names/dimensions; runtime dataclass field names are not treated as storage aliases.
- [x] Frozen support semantics remain deferred: `retrieval_support_flag` / `retrieval_top_altitude_m` are still absent until their synthetic tests exist.

### P3.2 — units, dimensions and missing-value semantics — COMPLETE

- [x] Added one exact metadata registry for every current L2 coordinate/data variable; dataset assembly fails if the emitted variable inventory and registry diverge.
- [x] Physical backscatter/extinction and their uncertainties have explicit SI units; lidar ratio is `sr`, dimensionless diagnostics are `1`, altitude diagnostics are `m`.
- [x] Source-dependent selected/glued signals and calibration/gluing coefficients deliberately use descriptive `unit_status` instead of fabricated absolute SI units.
- [x] `time`, `block_time`, `wavelength`, and `altitude` semantics are explicit; altitude is meters above station and positive upward.
- [x] Aerosol optical metadata states that unsupported backward bins remain NaN and are never filled/bridged; finite scattering ratio is explicitly not an altitude-support contract.
- [x] Numeric flags use integer `flag_values` arrays plus stable `flag_meanings`; aggregate/block forms share mappings.
- [x] Ambiguous zero states are documented where one bit cannot distinguish not-attempted from failed/rejected; dedicated source/input/reference/KFS diagnostics carry the stage information.
- [x] End-to-end NetCDF regression pins the exact variable inventory, units, long names, support wording and representative flag mappings.

### P3.3 — method/schema identity and algorithm provenance — COMPLETE

- [x] `level2_product_schema_version = "1"` and independent `level2_retrieval_method_version = "1"` are separate from package CalVer.
- [x] Incremental currentness rejects stale/missing schema identity, method identity, productive KFS/Fernald identity or gluing-score identity.
- [x] Gluing selection score v1 is named/versioned with exact unchanged rule `relative_rmse + abs(relative_bias) + 0.001*intercept_percent + 0.01*saturation_fraction`; all four weights are named constants and persisted metadata.
- [x] Product provenance records productive backward integration, Monte Carlo uncertainty identity, exact reference-boundary model and parameters, LR source, iteration count/random seed, and negative-aerosol/minimum-LR policy.
- [x] Metadata consistency regressions prevent a product from being considered current if method/branch/score identity differs from the code that would produce it.
- [x] P3.3 code gate passed on CI run 46: Ruff + full pytest on Ubuntu/Windows × Python 3.12/3.14.

### P3.4 — provenance/input manifest — IN PROGRESS

- [x] Chosen policy is readable/portable provenance: source Level 1 filename, stable station profile/calibration IDs, config filenames, exact processing/station YAML snapshots, and method/schema/software identity; no secrets, transient cache paths or host-specific absolute paths.
- [x] Existing provenance deliberately omits opaque configuration/Git SHA attributes; exact YAML plus stable IDs are the current reproducibility record. A future source-content hash requires a named consumer/use case rather than being added cosmetically.
- [ ] Pin an explicit cross-platform regression that the Level 2 input manifest stores the L1 filename rather than an absolute local path and that config provenance remains filename-based.
- [ ] Define and document which external atmosphere source identifiers belong in the scientific product and which cache/download details remain logs only.

### P3.5 — focused documentation — IN PROGRESS

- [x] `docs/level2_schema.md` now documents schema v1, method v1, units, flags, support/NaN semantics, gluing score identity, provenance boundary and current scientific limitations.
- [ ] Create/refresh focused docs for processing levels and config/station ownership without duplicating mutable scientific detail.
- [ ] Shorten README into an entry point after focused docs exist.
- [ ] Build a verified primary-source bibliography for methods actually implemented; distinguish historical inspiration from equations actually used.
- [x] Documented known limitations: provisional PC saturation guard, assumed aerosol LR, backward support ending at the reference, cloud screening not yet productive, no unvalidated hard SNR gate, and current NumPy/netCDF4 warning caveat.

P3 acceptance gate: **one Level 2 NetCDF is scientifically interpretable without reading implementation source and cannot describe a different method/support domain from the one actually used.**

## 7. P4 — observational/scientific evidence tasks — PARALLEL

- [ ] Characterize physical PC saturation under operational SPU settings using AN/PC overlap and preferably controlled attenuation before writing traceable `max_rate_mhz`.
- [ ] Move any surviving provisional PC guard to raw observed L1 PC rate rather than a corrected/background-subtracted proxy.
- [ ] Audit Level 1 dark-current subtraction versus nonlinear dead-time correction with evidence; do not alter it as collateral cleanup.
- [ ] Characterize propagated-error SNR on SPU data before adding a hard Rayleigh-window SNR gate.
- [ ] Validate cloud/layer screening on SPU observations before enabling it as productive reference-window rejection.
- [ ] Compare real retrievals with LPP where practical and with SCC/ELDA methodological expectations without claiming numerical identity.

Evidence tasks must not be replaced by invented constants.

## 8. P5 — future `fixing_l2` high-column redesign

Recreate `fixing_l2` only after P3 schema/FAIR groundwork is stable enough that new support/ensemble/cascade semantics have an explicit product home. The redesign must maximize **validated support**, not finite bins.

### L0 — non-negotiable scientific rules

- [x] `target_top_altitude_m` is a target, never extrapolation permission.
- [x] Do not turn NaN/non-positive/noisy samples into positive signal to extend coverage.
- [x] Do not interpolate across unsupported internal gaps to make KFS continuous.
- [x] Backward retrieval never implies values above its boundary.
- [x] A lower cascade segment inherits its boundary from an accepted upper solution; never reset `SR_ref=1` at every segment.
- [x] Multiple references are an ensemble/sensitivity experiment; disagreement contributes to uncertainty.
- [x] Elastic extinction remains conditional on assumed aerosol lidar ratio.
- [x] 20-min block retrieval and long-mean/high-column retrieval are distinct products/diagnostics.
- [x] Selection, merge and uncertainty policies must be explicit and persisted in readable provenance.

### L1 — vertical-support semantics — FROZEN, NOT YET SCHEMA

- [x] `retrieval_support_flag(block_time, wavelength, altitude)` means the final productive optical retrieval is scientifically supported at that bin after source selection, reference/KFS/candidate QA and accepted merge. It is not `isfinite(product)`.
- [x] Current backward-only support includes only bins actually solved through the accepted reference; outer unsupported bins remain NaN and internal gaps are never bridged.
- [x] A future backbone/cascade extends support only where it has its own valid physical support.
- [x] `retrieval_top_altitude_m` is the highest altitude with support flag 1; NaN when no supported bin exists.
- [ ] Add synthetic support/top tests before exposing these variables in NetCDF.
- [ ] Add machine-readable baseline summary for `20251107sapm` without committing raw observations.

### L2 — QA/output semantics

- [x] Redundant status TXT removed.
- [ ] After support enters schema, shade/mark supported optical domain and retrieval top in KFS/SR QA.
- [ ] Plot accepted reference windows explicitly and never present unsupported upper-tail signal diagnostics as valid aerosol retrieval.

### L3 — physical Rayleigh window — COMPLETE

- [x] Productive `ref_window_m` replaces fixed bin width.
- [x] Physical width converts deterministically on the real uniform altitude grid.
- [x] Search bounds remain in meters and invalid geometry fails explicitly.

### L4 — Rayleigh candidate catalogue

- [ ] Evaluate/catalogue all candidate windows passing minimum QA instead of returning one minimum-cost window.
- [ ] Persist start/stop/center, valid fraction, slope, variance, calibration factor, free-intercept diagnostic, uncertainty/SNR diagnostic and future validated layer flag.
- [ ] Separate pass/fail criteria from ranking.
- [ ] Prefer altitude only among already-valid candidates; do not choose a poor high candidate solely for coverage.
- [ ] Define minimum separation/correlation so overlapping windows do not masquerade as independent references.
- [ ] Keep cloud/SNR gates disabled until validated on SPU data.

### L5 — high-column backbone

- [ ] Build a long-mean/backbone signal separately from 20-min products.
- [ ] Make averaging/error propagation explicit and physically traceable.
- [ ] Use meter-based aggregation/windowing where grid changes require it.
- [ ] Select high reference candidates from the backbone.
- [ ] Treat ~20 km as an initial target, not a success condition; publish a lower top if no trustworthy boundary exists above it.

### L6 — multi-reference ensemble

- [ ] Run backward KFS from multiple accepted high references.
- [ ] Preserve member-specific reference and MC diagnostics.
- [ ] Combine only members whose physical backward support contains the altitude and whose QA passed.
- [ ] Define explicit uncertainty/quality weights; no anonymous heuristic weights.
- [ ] Add between-reference spread as an uncertainty component.
- [ ] Keep member solutions internally auditable/testable.

Acceptance gate: reference ambiguity increases exposed uncertainty instead of disappearing in an average.

### L7 — cascaded backward retrieval

- [ ] Define ordered overlapping high→low segments.
- [ ] Solve highest segment from a valid high/molecular boundary.
- [ ] Lower segments inherit actual `beta_total_ref` / scattering ratio from the accepted upper solution.
- [ ] Propagate inherited-boundary uncertainty through lower-segment Monte Carlo.
- [ ] Require adequate overlap/support and reject uncertainty-inconsistent handoffs.
- [ ] Never bridge unsupported internal gaps or reset `SR_ref=1` because a new segment starts.

Acceptance gate: cascade reproduces a well-conditioned synthetic full-column truth within defined tolerance and rejects inconsistent handoffs.

### L8 — overlap merge and uncertainty

- [ ] Implement documented smooth overlap merge only after segment acceptance; valid weights sum to one.
- [ ] Never average valid evidence with invalid/NaN evidence.
- [ ] Check value and vertical-gradient continuity across merge regions.
- [ ] Separate measurement/MC, LR, boundary, reference-choice, cascade-handoff and optional aggregation uncertainty components.
- [ ] Treat covariance deliberately; do not blindly quadrature-sum correlated terms.
- [ ] Verify uncertainty grows when reference ambiguity is introduced.

### L9 — redesigned FAIR product

- [ ] `retrieval_support_flag[..., altitude]`.
- [ ] `retrieval_top_altitude_m[...]` for block and backbone/aggregate products.
- [ ] Accepted reference-member count/diagnostics and ensemble-spread uncertainty.
- [ ] Cascade segment/handoff/merge diagnostics.
- [ ] Clear separation of long-mean/backbone and 20-min products.
- [ ] Explicit elastic-extinction dependence on assumed aerosol LR.
- [ ] Independent redesigned method/schema version; pre-redesign products become stale when required.
- [ ] QA overview shows signal, molecular/reference choices, support top, ensemble spread and cascade/merge regions without legitimizing unsupported upper tails.

### L10 — validation / merge gate

Synthetic:

- [ ] Pure molecular → approximately zero aerosol within defined tolerance.
- [ ] Known aerosol layers → recover truth within defined tolerance.
- [ ] Multiple valid references → stable ensemble.
- [ ] Contaminated candidate → rejected or visibly inflates uncertainty.
- [ ] Upper-tail noise → support top falls gracefully.
- [ ] Internal invalid gap → explicit failure/no bridging.
- [ ] Segment handoff → no stitch discontinuity beyond tolerance.

Real SPU:

- [ ] `20251107sapm`: materially extend defensible coverage while preserving/explaining the current lower-column solution.
- [ ] Evaluate whether ~15 km is routinely supported; attempt 20 km only with a valid boundary/support above the target.
- [ ] Validate clear, high-aerosol, cloud-contaminated, weak-signal and different AN/PC-dominance cases.
- [ ] Compare with LPP where practical and SCC/ELDA methodological behavior without claiming identity.

Merge criterion: **maximize validated vertical support, expose where support ends, and remain stable/traceable under reference/noise sensitivity. Reaching 20 km alone is not success.**

## 9. Current code-organization acceptance checklist

- [x] Every retained module/symbol has a documented productive/research/UI/compatibility role; known definition-only helpers are removed.
- [x] Root package exports only a real lightweight bound symbol.
- [x] Supported subpackage public surfaces are exact and regression-pinned.
- [x] No productive scientific behavior is installed by import side effect.
- [x] No wildcard import defines productive behavior.
- [x] No legacy Level 2 retrieval monolith/atmosphere compatibility alias remains.
- [x] Removed known Level 1 compatibility helpers with no remaining productive responsibility.
- [x] Tests are guarded against imports from known removed compatibility paths.
- [x] Canonical productive L2 and strict stage configs are guarded against local semantic defaults.
- [x] Numerical KFS/gluing/molecular kernels do not own filesystem policy.
- [x] QA/visualization does not feed back into retrieval decisions.
- [x] Cross-platform human-readable execution logs do not depend on host path separator.
- [x] Published dependency floor installs with NumPy 2.5.3 + netCDF4 1.7.4.
- [x] Full local correctness baseline is green.
- [x] GitHub Actions covers Ruff + Linux/Windows × Python 3.12/3.14 and has stayed green through the cleanup sequence.
- [x] Productive scientific behaviors have one canonical owner; retained research kernels are explicitly nonproductive policy surfaces.
- [x] Remaining broad exceptions are deliberate containment boundaries; helper/parser catches that could hide defects were narrowed and regression-tested.

## 10. Immediate next gate — P3 provenance and focused docs

1. Pin the portable L1 input/config manifest in cross-platform tests: filenames and stable IDs only, no machine-local absolute paths.
2. Define which thermodynamic source identifiers are scientific product provenance and keep external cache/download mechanics in logs.
3. Create focused processing-level and config/station-ownership docs, then make README an entry point rather than a second mutable scientific specification.
4. Build a verified primary-source bibliography for the methods actually implemented.
5. Keep P4 evidence items parallel and do not create `retrieval_support_flag` / `retrieval_top_altitude_m` or recreate `fixing_l2` before their synthetic semantics/tests are ready.
