# MILGRAU scientific engineering roadmap

Branch: `new-architecture`

Audit baseline before this roadmap rewrite: `83c8285bd16640b9ac978f9721c42e67aa15a2ba` (2026-09-15)

This file is the source of truth for the scientific and engineering roadmap of MILGRAU. The architectural refactor is no longer the primary problem: the current priority is to harden the validated backward-KFS baseline scientifically, complete FAIR/provenance semantics, characterize unresolved instrument behavior with evidence, and only then extend vertical support through a deliberately validated high-column redesign.

The order is intentional: truthful scientific identity -> one canonical implementation -> engineering guardrails -> current-method scientific/FAIR hardening -> instrument evidence -> high-column R&D -> release/publication readiness.

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
- Missing uncertainty is not zero uncertainty.
- A sample used in a reported mean must have uncertainty semantics consistent with the uncertainty reported for that mean.
- Correlated/model/systematic uncertainty must not be silently treated as independent noise.
- Tests exercise current canonical owners/contracts; obsolete tests are not a reason to restore compatibility code or hidden defaults.
- A successful real-data run validates the exercised path, not the full scientific method or repository test suite.
- A historical observational result is a regression baseline, not scientific ground truth.
- A target top altitude is a validation target, never permission to extrapolate, fill, bridge unsupported gaps or relax physics.

## 2. Priority/status overview

| Priority | Status | Purpose |
| --- | --- | --- |
| **P0** | **COMPLETE** | truthful backward-KFS identity and support semantics |
| **P1** | **COMPLETE + REAL-DATA VALIDATED** | canonical Level 2 architecture and removal of duplicate/hidden productive science |
| **P2** | **COMPLETE** | engineering hardening, deliberate API, exception/default guardrails and CI |
| **P3** | **IN PROGRESS** | current-method scientific hardening, FAIR schema/provenance, traceability and focused docs |
| **P4** | **PARALLEL EVIDENCE WORK** | instrument characterization and external/observational validation |
| **P5** | **PENDING** | high-column R&D: support, Rayleigh catalogue, backbone, ensemble and optional cascade |
| **P6** | **PENDING** | release/publication readiness and reproducible scientific release process |

## 3. P0 — scientific identity/public consistency — COMPLETE

- [x] Productive `inversion.kfs_mode` is canonically `backward`.
- [x] Scientific metadata and dataset/log wording publish the same backward method.
- [x] Incremental reuse rejects stale/two-sided Level 2 products.
- [x] Redundant `QA_L2_Product_Status_*.txt` removed at the generator.
- [x] Future `retrieval_support_flag` / `retrieval_top_altitude_m` semantics frozen before schema implementation.

Acceptance gate: **passed in implementation and real-data baseline.**

## 4. P1 — canonical Level 2 architecture — COMPLETE

Detailed ownership lives in `docs/code_inventory.md`.

- [x] Removed scientific monkey patches and import-order behavior.
- [x] Removed wildcard productive exposure and obsolete compatibility modules.
- [x] Deleted the legacy Level 2 retrieval monolith rather than wrapping it.
- [x] `milgrau.physics.atmosphere` is the shared atmosphere-physics owner.
- [x] `milgrau.level1.corrections` owns Level 1 corrections.
- [x] `milgrau.level1.thermodynamics` owns atmosphere-source/materialization logic.
- [x] `milgrau.level2.config` owns strict Level 2 configuration.
- [x] `milgrau.level2.gluing` owns the numerical gluing kernel.
- [x] `milgrau.level2.signal_selection` owns productive blocking/source selection/gluing/fallback.
- [x] `milgrau.level2.molecular` owns molecular physics/reference-search numerics.
- [x] `milgrau.level2.kfs` owns the KFS numerical kernel.
- [x] `milgrau.level2.optical_retrieval` owns productive Rayleigh QA and backward optical retrieval.
- [x] `milgrau.level2.retrieval` is the one-wavelength orchestration boundary.
- [x] `milgrau.level2.result_assembly` owns block -> typed-result construction.
- [x] `milgrau.level2.dataset` owns Level 2 product assembly/schema emission.
- [x] Productive caller passes explicit `mode="backward"` to the multi-mode research kernel.
- [x] Productive scientific behavior has one canonical owner; retained research kernels do not define productive policy.

### P1 real-data gate — PASSED

`milgrau-lebear -i 20251107sapm --force` after monolith deletion reproduced the productive 355/532 nm backward-KFS path with successful AN/PC gluing, accepted Rayleigh references, 5/5 backward retrieval blocks at each wavelength and a complete product with zero runtime errors.

PC saturation warnings remain explicitly provisional: physical saturation is still `not_characterized`.

Acceptance gate: **passed.**

## 5. P2 — engineering hardening and automated guardrails — COMPLETE

- [x] Repository/module role taxonomy and ownership inventory exist.
- [x] Root/subpackage public APIs are deliberate and regression-pinned.
- [x] AST guard rejects wildcard imports under `milgrau/`.
- [x] Canonical productive Level 2 code and strict stage configs are guarded against hidden scientific literal defaults.
- [x] Known dead compatibility/helpers were removed instead of retained speculatively.
- [x] Broad exception handling is retained only at explicit containment boundaries; parser/helper boundaries propagate unexpected runtime defects.
- [x] Ruff correctness/dead-code checks are active.
- [x] Full pytest suite is green.
- [x] GitHub Actions runs Ruff plus full pytest on Ubuntu/Windows x Python 3.12/3.14.
- [x] Published dependency floor installs with NumPy 2.5.3 + netCDF4 1.7.4.
- [x] The known NumPy/netCDF4 deprecation warning is classified as upstream behavior and is not hidden by pinning backwards merely to reach zero warnings.
- [x] Cross-platform human-readable paths/logs are regression-tested.

P2 acceptance gate: **passed — architecture is no longer the main roadmap risk. Broad refactoring must not displace scientific hardening.**

## 6. P3 — current-method scientific + FAIR hardening — IN PROGRESS

P3 freezes a scientifically defensible baseline before high-column R&D. It may change the current numerical baseline only when a scientific defect or undefined uncertainty/support contract is being fixed deliberately. Any such change must be versioned, tested and documented.

### P3.1 — canonical Level 2 schema names — COMPLETE

- [x] Inventoried every current L2 data-variable family, coordinate, completeness field and method/provenance group written by `level2.dataset`.
- [x] Schema v1 uses canonical aggregate optical names (`*_mean`, `*_mean_error`).
- [x] Removed exact duplicate NetCDF aliases instead of retaining indefinite compatibility arrays.
- [x] Aggregate, block and time-expanded roles are explicit in names/dimensions.
- [x] Future `retrieval_support_flag` / `retrieval_top_altitude_m` remain absent until their synthetic support semantics are tested.

### P3.2 — units, dimensions, method identity and missing-value semantics — COMPLETE

- [x] One exact metadata registry exists for every current Level 2 coordinate/data variable.
- [x] Physical backscatter/extinction and uncertainty variables have explicit SI units.
- [x] Source-dependent native/glued signals use descriptive `unit_status` instead of fabricated SI units.
- [x] Coordinate semantics are explicit; altitude is meters above station and positive upward.
- [x] Unsupported optical bins remain NaN and are never filled/bridged merely to extend support.
- [x] Finite scattering ratio is explicitly not an altitude-support contract.
- [x] Numeric flags use stable integer mappings and meanings.
- [x] `level2_product_schema_version` and `level2_retrieval_method_version` are separate from package CalVer.
- [x] Incremental currentness checks method/schema/KFS/Fernald/gluing-score identity.
- [x] Product provenance records productive backward integration, Monte Carlo identity, reference-boundary model, LR source, iteration count/random seed and current negative-aerosol/minimum-LR policies.
- [x] Gluing score identity and weights are named/versioned.

### P3.3 — current-method uncertainty/support hardening — BLOCKING

This subsection addresses issues in the present productive method; it must not be deferred to the future high-column redesign.

- [x] Replace independent signal/error reductions with a joint reduction using one scientifically defined valid mask (finite signal + finite non-negative uncertainty where an uncertainty-bearing mean is reported).
- [x] Return/store an effective sample count (`n_effective`) in the productive block inputs so means and mean uncertainties are auditable.
- [x] Add regressions proving that samples cannot contribute to the signal mean while being silently absent from its reported uncertainty denominator.
- [x] Remove the rule that converts non-finite `rcs_error` to `0.0` inside KFS Monte Carlo.
- [x] Define the current productive backward retrieval/error support as the physically sampled branch on which signal and molecular backscatter are finite/positive and signal uncertainty is finite/non-negative; missing uncertainty on that sampled path invalidates the productive branch.
- [x] Add synthetic tests where missing uncertainty inside the integration path causes explicit unsupported/rejected output rather than zero-noise Monte Carlo perturbation, and distinguish explicit `sigma=0` from missing uncertainty.
- [ ] Define which current uncertainty components are independent per block and which represent shared/correlated/systematic nuisance parameters.
- [ ] Review aggregate block uncertainty so LR/reference/model components are not blindly reduced as `sqrt(sum(sigma_i^2))/N` unless independence is justified.
- [x] State explicitly that current gluing propagated uncertainty is a partial measurement-noise propagation and excludes fitted slope/intercept uncertainty; it is not a total uncertainty budget.
- [ ] Decide whether gluing regression-parameter uncertainty is material for the productive budget; if implemented, validate with synthetic/Monte Carlo tests.
- [x] Because accepted-output/uncertainty semantics changed, increment `level2_retrieval_method_version` to `2`; incremental currentness makes pre-v2 products stale and method provenance records the change identity.

Method-v2 hardening gate: **Ruff plus full pytest passed on Ubuntu/Windows x Python 3.12/3.14 (CI run 57).**

Acceptance gate remains open: **for every productive optical value with reported uncertainty, support and uncertainty semantics are internally consistent; missing uncertainty never means zero uncertainty; aggregation assumptions are explicit and tested.** The remaining blockers are the shared/correlated uncertainty model and the decision on fitted gluing-parameter uncertainty.

### P3.4 — provenance/input identity — IN PROGRESS

Readable provenance remains primary, but the audit now has concrete consumers for content identity: incremental cache correctness and scientific input lineage.

- [x] Keep source Level 1 filename, stable station profile/calibration IDs, config filenames, exact processing/station YAML snapshots, method/schema/software identity and portable scientific source metadata.
- [x] Do not persist secrets, transient cache paths or host-specific absolute paths.
- [ ] Add `source_level1_sha256` (or an equivalently explicit content identity) to Level 2 provenance.
- [ ] Make incremental reuse compare the stored Level 1 content identity with the current input rather than relying on mtime alone for input correctness.
- [ ] Pin cross-platform tests that provenance stores portable filenames/IDs plus content identity, never machine-local source paths.
- [ ] Define stable thermodynamic source identifiers (`provider/product/version_or_release`) separately from cache filenames/download mechanics.
- [ ] Persist the stable thermodynamic source identifier and appropriate DOI/dataset identity when available.
- [ ] Define source-code identity policy: tagged scientific releases may use package version + release DOI/tag as primary identity; non-release/development products must additionally expose a repository/build revision sufficient to distinguish materially different code states sharing the same package version.
- [ ] Keep exact YAML snapshots as the human-readable configuration record; do not replace them with hashes.

Acceptance gate: **a product can identify the exact Level 1 content, scientific configuration and software/release state needed to distinguish one scientific run from another without embedding machine-local paths.**

### P3.5 — focused documentation and scientific traceability — IN PROGRESS

- [x] `docs/level2_schema.md` documents schema v1, method v2 identity, units, flags, joint signal/error support, missing-uncertainty semantics, gluing score identity, provenance boundary and current scientific limitations.
- [ ] Create/refresh focused docs for processing levels and config/station ownership without duplicating mutable scientific detail.
- [ ] Shorten README into an entry point after focused docs exist.
- [ ] Build a verified primary-source bibliography for methods actually implemented; distinguish historical inspiration from equations actually used.
- [ ] Minimum bibliography set: Klett (1981), Fernald (1984), Bucholtz (1995), photon-counting/dead-time reference(s), SCC/ELDA methodology references, FAIR4RS and the CF conventions used for claims/validation.
- [ ] Create a scientific traceability matrix linking each productive scientific claim/behavior to: canonical owner, tests, product metadata/provenance and primary literature/method reference.
- [ ] Explicitly mark elastic extinction as conditional on assumed aerosol lidar ratio and avoid implying semantic equivalence to Raman extinction products.
- [ ] Separate documentation of observational regression baselines from synthetic/analytical truth tests.

Recommended traceability-matrix columns:

| Scientific behavior | Canonical owner | Test evidence | Product metadata | Primary reference |
| --- | --- | --- | --- | --- |
| Rayleigh molecular model | `level2.molecular` | molecular/synthetic tests | molecular/method identity | Bucholtz |
| backward Fernald/KFS | `level2.kfs` + `optical_retrieval` | forward-model recovery | retrieval method/version | Fernald/Klett |
| AN/PC gluing | `level2.gluing` + `signal_selection` | synthetic + overlap tests | gluing method/score | implemented methodology |
| Rayleigh reference QA | `level2.molecular` + `optical_retrieval` | candidate/QA tests | reference diagnostics | implemented methodology |
| uncertainty model | `block_average` + `kfs` + aggregation | MC/error tests | uncertainty scope/components | implemented methodology |

### P3.6 — FAIR licensing and repository identity — BLOCKING FOR RELEASE

- [ ] Add an explicit root software `LICENSE` chosen according to project/institution policy.
- [ ] Add the corresponding software-license identity to `CITATION.cff`/package metadata where appropriate.
- [ ] Keep documentation/data licensing separate when their terms differ from the software license.
- [ ] Do not infer a code license from image assets or unrelated Creative Commons badges.
- [ ] Run a CF/compliance checker against representative Level 2 NetCDF before claiming CF compliance; document the exact convention version used for validation.

P3 acceptance gate: **one Level 2 NetCDF is scientifically interpretable and traceable without reading implementation source, cannot describe a different method/support domain from the one used, does not understate missing/undefined uncertainty, and has portable input/software provenance.**

## 7. P4 — instrument characterization and observational validation — PARALLEL EVIDENCE WORK

P4 is evidence-first. A scientifically honest result may be a characterized parameter, a rejected hypothesis, or an explicit decision to remain provisional/disabled because evidence is insufficient.

- [ ] Characterize physical photon-counting saturation under operational SPU settings using AN/PC overlap and preferably controlled attenuation before writing traceable `max_rate_mhz`.
- [ ] Move any surviving provisional PC guard to raw observed Level 1 PC rate rather than a corrected/background-subtracted proxy.
- [ ] Audit Level 1 dark-current subtraction versus nonlinear dead-time correction using instrument evidence; do not alter the order as collateral cleanup.
- [ ] Quantify the practical difference between plausible correction orders over the observed SPU count-rate/dark-rate/dead-time regime.
- [ ] Characterize propagated-error SNR on SPU data before adding a hard Rayleigh-window SNR gate.
- [ ] Validate cloud/layer screening on SPU observations before enabling it as productive reference-window rejection.
- [ ] Compare representative real retrievals with LPP where practical and with SCC/ELDA methodological expectations without claiming numerical identity.
- [ ] Evaluate current elastic-extinction presentation against SCC/ACTRIS semantics and retain explicit dependence on assumed lidar ratio.

For every P4 item, record one versioned outcome:

1. **characterized/enabled** — supported by evidence and tests;
2. **insufficient evidence/provisional** — remains disabled or explicitly provisional;
3. **rejected** — hypothesis/policy is not supported and must not enter productive configuration.

No invented constant or threshold is an acceptable way to close P4.

P4 acceptance gate: **every instrument-dependent productive threshold/correction has evidence and provenance, or remains explicitly disabled/provisional with a documented reason.**

## 8. P5 — high-column Level 2 R&D — PENDING

Start only after the P3 baseline is stable enough that new support, ensemble and optional cascade semantics have an explicit product home. The goal is to maximize **validated vertical support**, not finite bins or a fixed altitude target.

### L0 — non-negotiable scientific rules

- [x] `target_top_altitude_m` is a target, never extrapolation permission.
- [x] Do not turn NaN/non-positive/noisy samples into positive signal to extend coverage.
- [x] Do not interpolate across unsupported internal gaps to make KFS continuous.
- [x] Backward retrieval never implies values above its physical boundary.
- [x] Multiple references are an ensemble/sensitivity experiment; disagreement contributes to exposed uncertainty.
- [x] Elastic extinction remains conditional on assumed aerosol lidar ratio.
- [x] 20-min block retrieval and long-mean/high-column retrieval are distinct products/diagnostics.
- [x] Selection, merge and uncertainty policies must be explicit and persisted in readable provenance.
- [x] A cascade is a hypothesis to be justified by demonstrated need; it is not a mandatory endpoint.

### L1 — vertical-support semantics — FROZEN, NOT YET SCHEMA

- [x] `retrieval_support_flag(block_time, wavelength, altitude)` means the final productive optical retrieval is scientifically supported at that bin after source selection, reference/KFS/candidate QA and accepted merge. It is not `isfinite(product)`.
- [x] Current backward-only support includes only bins actually solved through the accepted reference; outer unsupported bins remain NaN and internal gaps are never bridged.
- [x] A future backbone/ensemble/cascade extends support only where it has its own valid physical support.
- [x] `retrieval_top_altitude_m` is the highest altitude with support flag 1; NaN when no supported bin exists.
- [ ] Add synthetic support/top tests before exposing these variables in NetCDF.
- [ ] Add a machine-readable observational regression summary for `20251107sapm` without committing raw observations.
- [ ] Label that summary explicitly as a regression baseline, not ground truth.

### L2 — validation truth hierarchy

Synthetic/analytical cases are the source of known truth:

- [ ] Pure molecular atmosphere with expected approximately zero aerosol within defined tolerance.
- [ ] Known aerosol layers generated from a controlled forward model.
- [ ] Controlled missing-error, noisy-tail and internal-gap cases.
- [ ] Multiple-reference scenarios with known perturbations/contamination.
- [ ] Segment/handoff scenarios only if cascade reaches implementation.

Observational cases are regression/behavior evidence:

- [ ] `20251107sapm` baseline summary.
- [ ] clear case.
- [ ] high-aerosol case.
- [ ] cloud/layer-contaminated case.
- [ ] weak-signal case.
- [ ] different AN/PC-dominance cases.

Observational agreement alone must not be promoted to proof of equation correctness.

### L3 — QA/output semantics

- [x] Redundant status TXT removed.
- [ ] After support enters schema, shade/mark supported optical domain and retrieval top in KFS/SR QA.
- [ ] Plot accepted reference windows explicitly and never present unsupported upper-tail signal diagnostics as valid aerosol retrieval.

### L4 — physical Rayleigh window — COMPLETE

- [x] Productive `ref_window_m` replaces fixed bin width.
- [x] Physical width converts deterministically on the real uniform altitude grid.
- [x] Search bounds remain in meters and invalid geometry fails explicitly.

### L5 — Rayleigh candidate catalogue

The present single-best-candidate workflow can reject a block when the minimum-cost candidate fails a later threshold even though another candidate might pass. Candidate validity must therefore be separated from ranking.

- [ ] Enumerate/catalogue all candidate windows satisfying minimum geometrical/data prerequisites.
- [ ] Apply explicit pass/fail QA to each candidate before ranking.
- [ ] Persist start/stop/center, valid fraction, slope, variance, calibration factor, free-intercept diagnostic, uncertainty/SNR diagnostic and future validated layer flag.
- [ ] Rank only already-valid candidates.
- [ ] Prefer altitude only among valid candidates; do not choose a poor high candidate solely for coverage.
- [ ] Define minimum separation/correlation so overlapping windows do not masquerade as independent references.
- [ ] Keep cloud/SNR gates disabled until validated on SPU data under P4.

Acceptance gate: **candidate rejection reasons are explicit, and a failing minimum-cost candidate cannot hide another scientifically valid candidate.**

### L6 — high-column backbone

- [ ] Build a long-mean/backbone signal separately from 20-min products.
- [ ] Make averaging/error propagation explicit and physically traceable using the P3 joint-support rules.
- [ ] Use meter-based aggregation/windowing where grid changes require it.
- [ ] Select high reference candidates from the backbone.
- [ ] Treat ~20 km as an initial target, not a success condition; publish a lower top if no trustworthy boundary exists above it.

### L7 — multi-reference ensemble

- [ ] Run backward KFS from multiple accepted high references.
- [ ] Preserve member-specific reference and Monte Carlo diagnostics.
- [ ] Combine only members whose physical backward support contains the altitude and whose QA passed.
- [ ] Define explicit uncertainty/quality weights; no anonymous heuristic weights.
- [ ] Add between-reference spread as an uncertainty component.
- [ ] Keep member solutions internally auditable/testable.

Acceptance gate: **reference ambiguity increases exposed uncertainty instead of disappearing in an average.**

### Decision gate A — is cascade scientifically necessary?

After L6/L7 validation, make an explicit recorded decision before implementing cascade:

- [ ] Quantify the remaining vertical-support limitation after backbone + multi-reference ensemble.
- [ ] Demonstrate whether the limitation is addressable by a cascaded boundary strategy rather than by better reference selection/support characterization.
- [ ] Compare expected coverage gain against additional boundary/handoff/covariance uncertainty and implementation complexity.
- [ ] If backbone + ensemble already meet scientific objectives, stop here and do not add cascade merely because it was planned historically.
- [ ] If cascade is justified, record the evidence and acceptance criteria before implementation.

### L8 — optional cascaded backward retrieval

Implement only if Decision gate A passes.

- [ ] Define ordered overlapping high -> low segments.
- [ ] Solve highest segment from a valid high/molecular boundary.
- [ ] Lower segments inherit actual `beta_total_ref` / scattering ratio from the accepted upper solution; never reset `SR_ref=1` merely because a segment starts.
- [ ] Propagate inherited-boundary uncertainty through lower-segment Monte Carlo.
- [ ] Require adequate overlap/support and reject uncertainty-inconsistent handoffs.
- [ ] Never bridge unsupported internal gaps.

Acceptance gate: **cascade reproduces a well-conditioned synthetic truth within defined tolerance and rejects inconsistent handoffs.**

### L9 — overlap merge and uncertainty

Applicable to ensemble and, if implemented, cascade overlaps.

- [ ] Implement documented smooth overlap merge only after member/segment acceptance; valid weights sum to one.
- [ ] Never average valid evidence with invalid/NaN evidence.
- [ ] Check value and vertical-gradient continuity across merge regions.
- [ ] Separate measurement/MC, LR, boundary, reference-choice, optional cascade-handoff and aggregation uncertainty components as far as scientifically meaningful.
- [ ] Treat covariance deliberately; do not blindly quadrature-sum correlated terms.
- [ ] Verify uncertainty grows when reference ambiguity or handoff ambiguity is introduced.

### L10 — redesigned FAIR product

- [ ] `retrieval_support_flag[..., altitude]`.
- [ ] `retrieval_top_altitude_m[...]` for block and backbone/aggregate products.
- [ ] Accepted reference-member count/diagnostics and ensemble-spread uncertainty.
- [ ] Optional cascade segment/handoff/merge diagnostics only if cascade exists.
- [ ] Clear separation of long-mean/backbone and 20-min products.
- [ ] Explicit elastic-extinction dependence on assumed aerosol LR.
- [ ] Independent redesigned method/schema version; pre-redesign products become stale when required.
- [ ] QA overview shows signal, molecular/reference choices, support top, ensemble spread and any merge/cascade regions without legitimizing unsupported upper tails.

### L11 — validation / merge gate

Synthetic:

- [ ] Pure molecular -> approximately zero aerosol within defined tolerance.
- [ ] Known aerosol layers -> recover truth within defined tolerance.
- [ ] Multiple valid references -> stable ensemble.
- [ ] Contaminated candidate -> rejected or visibly inflates uncertainty.
- [ ] Upper-tail noise -> support top falls gracefully.
- [ ] Internal invalid gap -> explicit failure/no bridging.
- [ ] If cascade exists: segment handoff -> no stitch discontinuity beyond tolerance and uncertainty reflects handoff ambiguity.

Real SPU:

- [ ] `20251107sapm`: materially extend defensible coverage while preserving/explaining the current lower-column solution.
- [ ] Evaluate whether ~15 km is routinely supported; attempt 20 km only with a valid boundary/support above the target.
- [ ] Validate clear, high-aerosol, cloud-contaminated, weak-signal and different AN/PC-dominance cases.
- [ ] Compare with LPP where practical and SCC/ELDA methodological behavior without claiming identity.

Merge criterion: **maximize validated vertical support, expose where support ends, and remain stable/traceable under reference/noise sensitivity. Reaching 20 km alone is not success.**

## 9. P6 — release/publication readiness — PENDING

P6 converts a scientifically hardened branch into a reproducible, citable release rather than treating a green development branch as a publication artifact.

- [ ] Explicit software license present and reflected in citation/package metadata.
- [ ] `CITATION.cff` version/date/DOI metadata matches the actual release state.
- [ ] Retrieval-method/schema version changes are summarized in release notes when scientific semantics change.
- [ ] Maintain a frozen reference scientific environment/constraints file for the release baseline.
- [ ] Keep a separate latest-compatible dependency CI lane for forward-compatibility detection.
- [ ] Consider a minimum-supported-dependencies CI lane for declared lower bounds.
- [ ] Build sdist/wheel in CI and test install/import from the built artifact in a clean environment.
- [ ] Produce a core-science coverage report; adopt thresholds only where they protect critical scientific modules/contracts rather than chasing a global percentage.
- [ ] Replace warning-count noise with a deliberate known-warning policy or allowlist so new warnings cannot disappear inside repeated upstream warnings.
- [ ] Run representative end-to-end Level 0 -> Level 2 processing from the release artifact/environment.
- [ ] Run CF/compliance validation on representative Level 2 products and record the convention/checker version.
- [ ] Freeze machine-readable synthetic acceptance summaries and observational regression summaries used for the release.
- [ ] Require CI checks/review policy on the release/merge path when repository governance is ready.
- [ ] Resolve branch history with `main` consciously before merge; do not cherry-pick semantically obsolete packaging changes merely to eliminate divergence.

P6 acceptance gate: **a third party can identify, install, cite and rerun the released scientific method in a documented reference environment, and release metadata uniquely identifies the code/method/schema state that produced the published products.**

## 10. Scientific traceability and acceptance checklist

### Current code-organization acceptance

- [x] Every retained productive scientific behavior has one documented canonical owner.
- [x] Root/subpackage public surfaces are deliberate and regression-pinned.
- [x] No productive scientific behavior is installed by import side effect.
- [x] No wildcard import defines productive behavior.
- [x] No legacy Level 2 retrieval monolith/atmosphere compatibility alias remains.
- [x] Strict stage configs are guarded against hidden semantic defaults.
- [x] Numerical KFS/gluing/molecular kernels do not own filesystem policy.
- [x] QA/visualization does not feed back into retrieval decisions.
- [x] Full local correctness baseline and cross-platform CI are green.

### Scientific-baseline acceptance still required

- [x] Signal/error averaging uses common support semantics.
- [x] Missing uncertainty cannot become zero-noise Monte Carlo support.
- [ ] Aggregate uncertainty states independence/correlation assumptions explicitly.
- [x] Gluing uncertainty scope is explicit and not described as total uncertainty while fitted regression-parameter uncertainty is excluded.
- [ ] Product provenance identifies exact Level 1 content.
- [ ] Development/release code identity is unambiguous.
- [ ] Scientific traceability matrix exists.
- [ ] Primary-source bibliography is verified.
- [ ] Software license is explicit.

## 11. Immediate next gate

Completed in method v2: joint signal/error averaging, `n_effective`, missing-uncertainty rejection in KFS, synthetic regressions, common-support aggregate optical reduction, method-version bump and stale-product enforcement.

Do these before recreating high-column `fixing_l2` work:

1. Define the current uncertainty-component model: which terms are independent acquisition noise and which LR/reference/model terms are shared or correlated across blocks.
2. Replace or qualify aggregate uncertainty reduction accordingly; add tests for shared versus independent nuisance components.
3. Decide whether fitted gluing slope/intercept uncertainty is material enough to propagate; keep the current scope explicitly partial until then.
4. Add Level 1 content identity to provenance and incremental currentness; keep paths portable.
5. Define stable thermodynamic source identifiers and non-release source-code revision identity.
6. Build the scientific traceability matrix and verified bibliography.
7. Add explicit software licensing and prepare CF validation criteria.
8. Keep P4 instrument characterization parallel and evidence-first.
9. Only after that, start P5 with synthetic support tests, observational regression baseline, Rayleigh candidate catalogue, backbone and ensemble. Evaluate cascade only after Decision gate A.
