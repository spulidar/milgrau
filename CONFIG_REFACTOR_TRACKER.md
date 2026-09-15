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
- Tests must exercise current canonical owners/contracts; obsolete tests are not a reason to restore compatibility code or hidden defaults.
- A successful real-data run validates the exercised path, not the full scientific method or repository test suite.

## 2. Priority/status overview

| Priority | Status | Purpose |
| --- | --- | --- |
| **P0** | **COMPLETE** | truthful backward-KFS identity, output/currentness consistency, remove redundant QA TXT, freeze support semantics |
| **P1** | **COMPLETE + REAL-DATA VALIDATED** | remove import-order behavior, duplicate Level 2 science, monoliths and obsolete compatibility |
| **P2** | **IN PROGRESS** | repository code-use audit, public API ownership, dead compatibility, exception/default guardrails, Ruff/full tests/CI |
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
- [x] Regression requires every advertised `__all__` symbol to resolve.
- [x] AST regression rejects wildcard imports under `milgrau/`.
- [x] Productive Level 2 mappings cannot use `.get(..., scientific_literal_default)`.
- [x] Ruff added to dev dependencies with initial correctness/dead-code set `E4`, `E7`, `E9`, `F`.
- [x] First Ruff pass classified: 8 real `F401` dead imports removed; 3 `E731` lambda-style findings intentionally excluded from this correctness gate.
- [x] User-confirmed Ruff rerun after cleanup: **all checks passed**.
- [x] Focused architecture/cleanup tests passed locally before this sublot (`11 passed`).

### Lot 4B — ownership/compatibility/default/exception cleanup — IN PROGRESS

Completed:

- [x] Level 1 package exports now point directly to canonical owners rather than re-exporting helpers indirectly through `lipancora`.
- [x] Regression pins Level 1 package exports to `ingestion`, `pbl`, `thermodynamics` and other actual owners.
- [x] Removed one-line `level1.common.level1_output_path()` compatibility alias; LIPANCORA imports canonical `milgrau.io.paths.level1_output_path` directly.
- [x] Removed obsolete `level1.common.get_channel_constant(..., logger)` compatibility helper; current calibration ownership is `resolve_channel_calibration()`.
- [x] Regression pins both removed Level 1 compatibility helpers absent.
- [x] Narrowed `level0.quality` group containment to malformed-data exceptions (`KeyError`, `TypeError`, `ValueError`); unexpected runtime/programming failures now propagate.
- [x] Added regression proving an unexpected Level 0 QA `RuntimeError` is not silently converted into ordinary data rejection.
- [x] Narrowed station-config loader wrapping to expected configuration errors (`TypeError`, `ValueError`); unexpected validator failures propagate unchanged.
- [x] Added regression proving unexpected station-validator `RuntimeError` is not reclassified as bad YAML/config.
- [x] Narrowed `level1.common.finite_or_fill` and dark-current availability catches to expected conversion/index failures.
- [x] Extended semantic-default AST guard to strict `level0/config.py`, `level1/config.py`, `level2/config.py`: structural absence sentinels are allowed; recipe defaults are not.
- [x] Compatibility-import AST guard covers both package and tests, preventing reintroduction of removed Level 1/2 aliases.
- [x] Broad exception policy documented: retain explicit outer orchestration/partial-product/optional-QA containment; narrow helper-level catches that can hide defects.
- [x] Full-suite stale references to removed `level2.atmosphere`, `get_channel_constant`, top-level `physics`, private `lebear` gluing/Rayleigh helpers and pre-strict configuration fixtures were migrated to canonical owners/contracts rather than restoring compatibility.
- [x] Level 1 ingestion tests now use stdlib-compatible loggers and explicitly provide dead-time/saturation policy arguments; no kernel defaults were reintroduced for test convenience.
- [x] LIRACOS and LEBEAR synthetic fixtures now materialize the canonical Level 1 atmosphere and complete strict recipes instead of bypassing current product/config contracts.
- [x] Human-readable `ExecutionResult` log paths use POSIX separators for stable cross-platform diagnostics while stored `Path` semantics remain unchanged.
- [x] SCC `LR_Input` station policy now recognizes historical 607 nm as a Raman companion of 532 nm in addition to 530 nm, matching the APEL SCC channel inventory and the catalog's stated Raman-companion policy.

Still open in lot 4:

- [ ] Rerun Ruff + **full `pytest -q`** after the classified failure-repair batch; do not call the suite clean until reproduced locally.
- [ ] Finish deliberate package-public-surface decisions for `io`, `level0`, `level1`, `level2`, `physics`, `viz`; avoid accidental breaking changes to plausible external/research APIs.
- [ ] Finish compatibility/deprecation audit outside cleaned L1/L2 and require named consumers/removal criteria.
- [ ] Search repository-wide for duplicated **scientific equations/selection rules** and retain one canonical implementation; do not over-abstract tiny generic utilities.
- [ ] Finish broad-exception review in processing helpers; retained broad catches require an explicit orchestration/partial-product/UI rationale.
- [ ] Remove confirmed dead QA/display helpers. Current private candidates: `viz/level2_qa.py::_legacy_ylim` and `_visual_scale_to_reference`; edit separately because the plotting module is large.
- [ ] Review large files by cohesion, not line count: `explorer/streamlit_app.py`, `viz/level2_qa.py`, `level2/signal_selection.py`, `level2/dataset.py`, `level0/netcdf.py`, `level1/config.py`, `level2/kfs.py`, `level2/contracts.py`, `level1/lipancora.py`, `config/station.py`.

### Lot 5 — full suite and CI — PENDING

First full-suite baseline on the P2 cleanup branch:

- [x] Ruff: **all checks passed**.
- [x] `pytest -q` reached the complete suite after obsolete collection imports were removed.
- [x] Baseline result recorded rather than hidden: **324 passed, 23 failed, 266 warnings**.
- [x] All 23 failures were classified before changes: mostly stale tests/fixtures after strict contracts/canonical-owner cleanup, plus a Windows-path portability issue and a real historical SCC Raman-companion mapping inconsistency.
- [x] Repair batch implemented without restoring hidden defaults/compatibility paths.
- [ ] Reproduce the full suite after the repair batch and classify any residual failures.
- [ ] Classify warning baseline separately from test correctness: third-party xarray/netCDF4↔NumPy 2.5 warnings versus MILGRAU-owned `level0/netcdf.py` assignment deprecations.
- [ ] Establish clean full local pytest baseline in the dev environment.
- [ ] Add CI for Ruff, architecture/static guards and full pytest.
- [ ] Only after stable CI, consider branch protection/required checks.

P2 acceptance gate: **no known unused compatibility code, intentional public API, no productive hidden semantic defaults, justified exception boundaries, and automated checks preventing regression.**

## 6. P3 — Level 2 schema / FAIR metadata / focused docs — PENDING

- [ ] Choose canonical aggregate names and migration policy for `aerosol_backscatter[_mean]` / `aerosol_extinction[_mean]` aliases.
- [ ] Audit every L2 physical variable for units, dimensions, long_name/description and missing/NaN semantics.
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

- [ ] Every retained symbol/module has a real consumer or documented productive/research role; module inventory exists, per-symbol/API audit continues.
- [x] Root package exports only a real lightweight bound symbol.
- [x] No productive scientific behavior is installed by import side effect.
- [x] No wildcard import defines productive behavior.
- [x] No legacy Level 2 retrieval monolith/atmosphere compatibility alias remains.
- [x] Removed known Level 1 compatibility helpers with no remaining productive responsibility.
- [x] Tests are guarded against imports from known removed compatibility paths.
- [ ] Package/public aliases are intentional or have an explicit deprecation/API decision.
- [x] Canonical productive L2 and strict stage configs are guarded against local semantic defaults.
- [ ] Auxiliary config interpretation outside strict stage resolvers still needs final audit.
- [x] Numerical KFS/gluing/molecular kernels do not own filesystem policy.
- [x] QA/visualization does not feed back into retrieval decisions.
- [x] Cross-platform human-readable execution logs do not depend on host path separator.
- [ ] Confirm the classified full-suite repair batch with full pytest, then add CI.

## 10. Immediate next gate

The first complete full-suite run after collection cleanup produced **324 passed / 23 failed / 266 warnings** while Ruff remained clean. The failures were classified and repaired without restoring obsolete APIs or semantic defaults; the current repair batch now needs reproduction.

Run:

```bash
ruff check milgrau tests
pytest -q
```

Expected handling:

- Ruff must remain clean for the configured gate.
- Any residual pytest failure is classified before another code change: real regression, stale test/contract, optional-dependency isolation, platform portability, or test defect.
- Warnings are tracked separately from correctness. Third-party xarray/netCDF4↔NumPy 2.5 deprecations must not be confused with MILGRAU-owned `level0/netcdf.py` deprecation sites.
- Do **not** call the repository globally green until this full-suite gate is clean and subsequently reproduced in CI.

After a clean full-suite baseline, finish the remaining public-surface/QA-helper/duplicate-rule audit, then implement lot 5 CI.