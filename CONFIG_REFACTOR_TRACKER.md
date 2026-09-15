# MILGRAU architecture, scientific quality and Level 2 roadmap

Branch: `new-architecture`

Audit baseline: `c6400a941964de73438132afdb7f3db9a5916f32` (2026-09-14)

This file is the source of truth for preparing a concise, scientifically defensible base before recreating `fixing_l2`. Priority is intentional: truthful scientific identity first, one canonical implementation second, code-use/static/schema hardening third, and only then the high-column redesign.

## 1. Non-negotiable engineering/scientific rules

- `config.yaml` owns the processing/scientific recipe.
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
| **P2** | **IN PROGRESS; LOCAL + CI BASELINES GREEN** | code-use audit, deliberate API, dead compatibility, exception/default guardrails, Ruff/full tests/CI |
| **P3** | pending | Level 2 schema/FAIR metadata and focused documentation |
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
- [x] Previous sparse/all-NaN QA variance warnings did not return.
- [x] PC warnings remained explicitly provisional: physical saturation still `not_characterized`.

Acceptance gate: **passed.**

## 5. P2 — code-use audit, concision and automated guardrails — IN PROGRESS

Detailed ownership decisions live in `docs/code_inventory.md`.

### Lot 4A — inventory/static baseline — COMPLETE

- [x] Added role taxonomy and repository/module ownership inventory.
- [x] Root `milgrau.__all__` reduced to the real lightweight root API: `__version__`.
- [x] AST regression rejects wildcard imports under `milgrau/`.
- [x] Productive Level 2 mappings cannot use `.get(..., scientific_literal_default)`.
- [x] Strict `level0/config.py`, `level1/config.py`, `level2/config.py` are guarded against semantic literal `.get` fallbacks.
- [x] Ruff added to dev dependencies with correctness/dead-code rules `E4`, `E7`, `E9`, `F`; `E731` remains deliberately excluded as style-only.
- [x] First Ruff pass removed 8 real `F401` imports; reruns stayed clean.

### Lot 4B — ownership/compatibility/default/exception cleanup — FINAL AUDIT IN PROGRESS

Completed:

- [x] Level 1 package exports point directly to canonical owners; obsolete one-line compatibility wrappers are gone.
- [x] `level1.common.level1_output_path()` removed; canonical owner is `milgrau.io.paths.level1_output_path`.
- [x] `level1.common.get_channel_constant(..., logger)` removed; calibration owner is `resolve_channel_calibration()`.
- [x] Compatibility-import AST guard prevents tests/package code from reintroducing removed Level 1/2 paths.
- [x] Full-suite stale imports/fixtures were migrated to current owners/contracts instead of restoring compatibility.
- [x] Level 0 acquisition QA and station-config loading now contain only expected malformed-data/config errors; unexpected runtime defects propagate.
- [x] Level 1 helper catches were narrowed where broad containment could hide defects.
- [x] Licel header/group parsing now contains malformed-file/IO failures (`ValueError`, `OSError`) but propagates unexpected `RuntimeError`; regressions pin both behaviors.
- [x] Broad catches intentionally retained at outer execution boundaries, per-wavelength partial-product handling, per-channel L1 containment, filesystem action boundaries, and optional QA/UI presentation boundaries.
- [x] Human-readable `ExecutionResult` paths are stable across Windows/Linux separators.
- [x] SCC `LR_Input` policy recognizes historical 607 nm as Raman companion of 532 nm as well as 530 nm.
- [x] Exact supported `__all__` surfaces for `io`, `level0`, `level1`, `level2`, `physics`, and `viz` are now regression-pinned. New/removal exports require deliberate review instead of becoming accidental API.
- [x] `config.loader` creates no legacy `site`/`hardware`/`physics.channels` compatibility structures; station-owned LR climatology materialization is an intentional productive data view, not an alias.
- [x] Historical Licel file-level laser-shot fallback is retained deliberately as a file-format compatibility path: channel `NShots` has priority; fallback remains only while historical/custom SPU Licel files requiring it are supported.
- [x] Local functional baseline: Ruff clean and `pytest -q` **347 passed / 0 failed** on Windows/Python 3.14 before the final two Licel exception tests were added.

Still open before calling lot 4 complete:

- [ ] Remove confirmed dead QA/display helpers `viz/level2_qa.py::_legacy_ylim` and `_visual_scale_to_reference` (definition-only; `_legacy_scale_factor` remains used).
- [ ] Finish repository-wide duplicate **scientific equation/selection-rule** audit and record canonical owners; do not over-abstract generic utilities.
- [ ] Final pass over remaining `except Exception` sites: classify each as orchestration/filesystem/optional-UI containment or narrow it if it can hide a programming defect.
- [ ] Review large files by cohesion, not line count; split only where responsibility actually improves: `explorer/streamlit_app.py`, `viz/level2_qa.py`, `level2/signal_selection.py`, `level2/dataset.py`, `level0/netcdf.py`, `level1/config.py`, `level2/kfs.py`, `level2/contracts.py`, `level1/lipancora.py`, `config/station.py`.

### Lot 5 — full suite and CI — GREEN BASELINE ESTABLISHED

Full-suite progression:

- [x] First complete baseline: **324 passed, 23 failed, 266 warnings**.
- [x] All 23 failures classified before repair; mostly stale contracts/fixtures plus Windows path portability and historical SCC Raman mapping.
- [x] Repair batch implemented without restoring hidden defaults/compatibility.
- [x] Residual inventory failures were stale mocks only and fixed in tests.
- [x] Clean local baseline reproduced: **347 passed, 0 failed, 346 warnings**, Ruff clean.
- [x] Warning baseline classified: all 346 are the NumPy 2.5 `ndarray.shape` deprecation triggered inside the published netCDF4 1.7.4 write path. 285 surface through xarray's backend and 61 at MILGRAU caller lines; the latter are not a separate MILGRAU deprecation.
- [x] Upstream source changelog lists a 1.7.4.1 fix, but PyPI currently publishes through 1.7.4 only. The invalid `>=1.7.4.1` floor was reverted immediately after pip rejected it.
- [x] MILGRAU requires the latest published floor `netCDF4>=1.7.4`; do not pin NumPy backwards or suppress the known upstream warning merely to get warning count zero.
- [x] User-confirmed environment: NumPy **2.5.3**, netCDF4 **1.7.4**.
- [x] GitHub Actions CI added: Ruff on Ubuntu/Python 3.12 plus full pytest matrix on Ubuntu/Windows × Python 3.12/3.14.
- [x] CI run `34916492050` on commit `e0bed288e1d4829f0ffaa61647e1257b9b0ca5ab` completed **success**: Ruff passed and every pytest matrix job passed.
- [ ] After final lot-4 cleanup commits, require the same CI workflow to remain green on the new HEAD.
- [ ] Branch protection/required checks is optional follow-up only after the final P2 HEAD is stable; branch is currently unprotected.

P2 acceptance gate: **intentional public API, no known unused compatibility code, no productive hidden semantic defaults, justified exception boundaries, one canonical owner per scientific behavior, and local + CI guardrails preventing regression.**

## 6. P3 — Level 2 schema / FAIR metadata / focused docs — PENDING

- [ ] Choose canonical aggregate names and migration policy for `aerosol_backscatter[_mean]` / `aerosol_extinction[_mean]` aliases.
- [ ] Audit every L2 physical variable for units, dimensions, `long_name`/description and missing/NaN semantics.
- [ ] Add explicit backscatter/extinction uncertainty units where currently implied.
- [ ] Audit numeric flag metadata for CF-compatible representation.
- [ ] Introduce independent product-schema/method version if method/schema evolution requires more than package CalVer.
- [ ] Name/version/document gluing selection-score constants; no anonymous algorithmic weights.
- [ ] Decide readable immutable input-manifest policy.
- [ ] Create focused docs for processing levels, configuration/station catalog, products, methods, provenance, flags, limitations and validation; shorten README into an entry point afterward.
- [ ] Build verified primary-source bibliography for methods actually implemented.

Acceptance gate: one Level 2 NetCDF is scientifically interpretable without reading implementation source and cannot describe a different method from the one used.

## 7. P4 — observational/scientific evidence tasks — PARALLEL

- [ ] Characterize physical PC saturation under operational SPU settings using AN/PC overlap and preferably controlled attenuation before writing traceable `max_rate_mhz`.
- [ ] Move any surviving provisional PC guard to raw observed L1 PC rate rather than a corrected/background-subtracted proxy.
- [ ] Audit Level 1 dark-current subtraction versus nonlinear dead-time correction with evidence; do not alter it as collateral cleanup.
- [ ] Characterize propagated-error SNR on SPU data before adding a hard Rayleigh-window SNR gate.
- [ ] Validate cloud/layer screening on SPU observations before enabling it as productive reference-window rejection.
- [ ] Compare real retrievals with LPP where practical and with SCC/ELDA methodological expectations without claiming numerical identity.

Evidence tasks must not be replaced by invented constants.

## 8. P5 — future `fixing_l2` high-column redesign

Recreate `fixing_l2` only after P2 core cleanup/guardrails are stable. The redesign must maximize **validated support**, not finite bins.

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
- [ ] Define minimum separation/correlation so overlapping windows do not masquerade as independent ensemble evidence.
- [ ] Persist compact accepted/rejected reasoning.
- [ ] Do not add hard SNR/cloud gates before P4 validation.

### L5 — high-column backbone

- [ ] Add explicit long-mean/high-column input distinct from 20-min block retrieval; averaging duration is configured/provenanced.
- [ ] Evaluate altitude-dependent vertical aggregation only if evidence requires it; widths remain physical and original-resolution data remain available.
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

- [ ] Every retained symbol/module has a real consumer or documented productive/research role; final private-helper/duplicate audit remains.
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
- [x] GitHub Actions CI baseline is green across Ruff + Linux/Windows × Python 3.12/3.14.
- [ ] Final cleanup HEAD must rerun the same CI green before P2 is marked complete.

## 10. Immediate next gate

1. Remove the two confirmed dead private Level 2 QA display helpers only; do not touch active `_legacy_scale_factor` behavior.
2. Finish scientific-duplicate and broad-exception classification and update `docs/code_inventory.md` with canonical owners/removal criteria.
3. Run/observe CI on the final cleanup HEAD.
4. Mark P2 complete only when the final HEAD stays green; branch protection remains a separate optional repository-policy choice.
5. Then move to P3 FAIR/schema work; do not begin the high-column P5 redesign before P2 is closed.
