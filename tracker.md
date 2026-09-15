# MILGRAU scientific engineering roadmap

Branch: `new-architecture`

Current method-v3 observational baseline: `docs/regression_baselines/20251107sapm_method_v3.json`

This file is the active source of truth for scientific/engineering work. Detailed completed history remains recoverable in Git; the tracker intentionally emphasizes current gates rather than retaining every closed implementation checkbox forever.

## Non-negotiable rules

- `config.yaml` owns the processing/scientific recipe; `station.yaml` owns site/instrument reality and station-derived observational metadata; Python owns equations/constants/implementation.
- One productive scientific behavior has one canonical implementation.
- Missing uncertainty is never zero uncertainty.
- Correlated/model/systematic uncertainty is not silently treated as independent noise.
- Unsupported data remain unsupported; no fill/interpolation is introduced merely to extend retrieval coverage.
- A successful real-data run is regression/behavior evidence, not ground truth or proof of equation correctness.
- Scientific support is not equivalent to `isfinite(product)`.
- Vertical support must respect both the upper inversion/reference boundary and the lower instrument-validity boundary.
- A target top altitude is a validation target, never permission to extrapolate, bridge gaps or relax physics.
- Instrument constants/thresholds are evidence-derived or explicitly provisional/disabled; they are never invented to close the roadmap.

## Status overview

| Priority | Status | Purpose |
| --- | --- | --- |
| P0 | COMPLETE | truthful backward-KFS identity and support semantics |
| P1 | COMPLETE + REAL-DATA VALIDATED | canonical Level 2 architecture |
| P2 | COMPLETE | engineering guardrails and cross-platform CI |
| P3 | IN PROGRESS — LICENSE/CF GATE ONLY | current-method scientific + FAIR hardening |
| P4 | PARALLEL EVIDENCE WORK | instrument characterization / observational validation |
| P5 | PENDING | altitude-resolved support + high-column R&D |
| P6 | PENDING | reproducible release/publication process |

## Completed baseline: P0–P3.5

The current productive baseline is backward Klett–Fernald–Sasano, Level 2 schema v1, retrieval method v3.

Completed and regression-guarded:

- canonical Level 2 ownership and deletion of the legacy retrieval monolith;
- explicit backward productive identity and stale-product rejection;
- canonical Level 2 schema/metadata and SI units where physically meaningful;
- joint signal/error averaging with common valid support and auditable `n_effective`;
- missing KFS uncertainty rejected rather than converted to zero Monte Carlo noise;
- aggregate optical uncertainty uses the conservative fully-correlated upper bound rather than an unjustified `1/sqrt(N)` reduction of mixed nuisance terms;
- current gluing error is explicitly only propagated measurement noise; fitted slope/intercept uncertainty is outside scope and not claimed negligible;
- exact Level 1 SHA-256 lineage and content-aware incremental reuse;
- exact YAML snapshots plus stable station/calibration/thermodynamic identities;
- portable installed-source `source_code_sha256` plus repository revision when available;
- focused processing/configuration/schema docs and scientific traceability matrix;
- verified primary references and explicit distinction between implemented physics, MILGRAU policy, observational regression, external comparison and instrument characterization;
- Ruff + full pytest on Ubuntu/Windows × Python 3.12/3.14.

## Method-v3 real-data regression baseline — COMPLETE

`20251107sapm` was rerun end-to-end from revision `842389a23c3ddc2006243db9b7d98bf200c90c0f` and frozen in `docs/regression_baselines/20251107sapm_method_v3.json`.

Observed baseline:

- Level 0 inventory: 207/209 accepted; 167 measurement profiles + 40 dark-current profiles; 12 channels; 4000 bins.
- Level 1: 167/167 valid PBL estimates; mean PBL 1.117 km; radiosonde source with 33.5% USSA76 vertical extension; CPT 18.088 km; LRT 14.255 km.
- 355 nm: AN/PC gluing 100%; 5/5 retrieval blocks; block reference centers 5.606–6.101 km; aggregate finite optical top 6.101 km; all five blocks overlap to 5.606 km.
- 532 nm: AN/PC gluing 100%; 5/5 retrieval blocks; block reference centers 5.749–6.319 km; aggregate finite optical top 6.319 km; all five blocks overlap to 5.749 km.
- Every finite aggregate aerosol optical value has a finite reported uncertainty.
- Aggregate uncertainty exactly follows method-v3 `sum(sigma_block)/n_effective` on common value/error support.
- Level 2 `source_level1_sha256` matches the uploaded Level 1 bytes exactly.

Interpretation constraint: aggregate optical arrays are finite from 3.75 m AGL, but SPU near-field geometrical overlap is not characterized in the product/configuration. That lower edge is **not** accepted as validated quantitative aerosol support.

## P3.6 — FAIR license + CF validation — BLOCKING P3 COMPLETION

- [ ] Choose an explicit software license according to project/institution policy and add a root `LICENSE`.
- [ ] Add matching software-license identity to `CITATION.cff` / package metadata where appropriate.
- [ ] Keep software, documentation and data licensing separate where their terms differ.
- [x] Representative CF-validation product selected: method-v3 `20251107sapm` Level 2 from revision `842389a...`.
- [x] Preliminary structural inspection completed; the current file has no global `Conventions` declaration, therefore MILGRAU does **not** claim CF compliance yet.
- [ ] Run an actual CF/compliance checker, record checker + CF version, fix genuine schema issues and document justified exceptions.

P3 acceptance gate: **open only on license + recorded CF validation evidence.**

## P4 — instrument characterization / observational validation — ACTIVE IN PARALLEL

Evidence tasks may finish as `characterized/enabled`, `insufficient evidence/provisional`, or `rejected`. No invented constant closes P4.

- [ ] Characterize physical photon-counting saturation under operational SPU conditions using AN/PC overlap and preferably controlled attenuation before defining a traceable `max_rate_mhz`.
- [ ] Move any surviving provisional PC guard to raw observed Level 1 PC rate rather than a corrected/background-subtracted proxy.
- [ ] Audit dark-current subtraction versus nonlinear dead-time correction with instrument evidence and quantify the practical difference over observed SPU regimes.
- [ ] Characterize near-field geometrical overlap for productive elastic channel families, or establish a validated overlap-correction product; record the lowest altitude supporting quantitative optical retrieval without inventing a generic cutoff.
- [ ] Until overlap is characterized, treat finite very-low-altitude KFS values as algorithmic output only, not validated aerosol support.
- [ ] Characterize propagated-error SNR before enabling hard Rayleigh-window SNR rejection.
- [ ] Validate cloud/layer screening before enabling it as productive reference-window rejection.
- [ ] Quantify gluing slope/intercept uncertainty and covariance with overlap resampling/bootstrap/Monte Carlo; either justify continued exclusion from the declared partial budget or introduce a new versioned propagation model.
- [ ] Compare representative retrievals with LPP where practical and with SCC/ELDA methodological expectations without claiming numerical identity.
- [ ] Keep elastic extinction explicitly conditional on assumed aerosol lidar ratio; do not present it as semantically equivalent to Raman extinction.

Current observation relevant to dead time: 355.PC had no clipped profiles in `20251107sapm`; 532.PC had clipping in 8/167 profiles, with maximum clipped-bin fraction 0.00075 per profile. Physical saturation remains uncharacterized.

P4 acceptance gate: every instrument-dependent productive threshold/correction/support boundary has evidence/provenance, or remains explicitly provisional/disabled.

## P5 — altitude-resolved support + high-column R&D — PENDING

Start implementation after P3.6 closes. Evidence for P4 may continue in parallel, but P5 support semantics must honor any unresolved P4 limitations.

### P5.1 support semantics + synthetic truth

Frozen semantics:

- `retrieval_support_flag(..., altitude)` means scientifically supported productive optical retrieval, not mere finiteness.
- Upper support ends at the actual accepted inversion/reference boundary; no backward result exists above that boundary.
- Lower support must respect validated overlap/instrument-correction domain; a finite KFS value below that domain has support 0.
- Internal unsupported gaps are never bridged.
- `retrieval_top_altitude_m` is the highest supported altitude; NaN when no supported bin exists.

To implement:

- [ ] Add synthetic support tests covering upper boundary, lower overlap boundary, noisy tail, missing uncertainty and internal gaps.
- [ ] Expose support/top only after those tests pass.
- [ ] QA must show the complete supported domain, including lower edge and upper top.

Validation truth hierarchy:

- [ ] pure molecular synthetic case -> approximately zero aerosol within defined tolerance;
- [ ] controlled aerosol layers -> recover truth within tolerance;
- [ ] controlled missing-error/noisy-tail/internal-gap/lower-overlap cases;
- [ ] multiple-reference perturbation/contamination cases;
- [x] `20251107sapm` observational regression summary frozen and explicitly labeled non-ground-truth;
- [ ] add clear, high-aerosol, cloud-contaminated, weak-signal and different AN/PC-dominance observational cases.

### P5.2 Rayleigh candidate catalogue

Current single-best-candidate selection can create false negatives when the minimum-cost candidate later fails although another candidate would pass.

- [ ] Enumerate every geometrically/data-valid candidate window.
- [ ] Apply pass/fail QA before ranking.
- [ ] Persist start/stop/center, valid fraction, slope, variance, calibration factor, free-intercept diagnostic, uncertainty/SNR diagnostic and future validated layer flag.
- [ ] Rank only candidates that already passed QA.
- [ ] Prefer altitude only among valid candidates.
- [ ] Define minimum separation/correlation so overlapping windows do not masquerade as independent references.
- [ ] Keep cloud/SNR gates disabled until P4 validates them.

### P5.3 long-mean backbone

- [ ] Build a long-mean/backbone signal separately from 20-min products.
- [ ] Use the P3 joint signal/error-support rules for averaging/error propagation.
- [ ] Use physical meter-based windowing where appropriate.
- [ ] Select high references from the backbone.
- [ ] Treat ~20 km as an initial target only; publish a lower top when no trustworthy boundary exists.

### P5.4 multi-reference ensemble

- [ ] Run backward KFS from multiple accepted references.
- [ ] Preserve member-specific reference and Monte Carlo diagnostics.
- [ ] Combine only members whose physical support contains the altitude and whose QA passed.
- [ ] Define explicit quality/uncertainty weights.
- [ ] Add between-reference spread as an exposed uncertainty component.
- [ ] Keep individual member solutions auditable.

Acceptance gate: reference ambiguity must increase exposed uncertainty instead of disappearing in an average.

### Decision gate A — cascade?

After backbone + ensemble validation:

- [ ] quantify the remaining support limitation;
- [ ] demonstrate whether cascade solves a real limitation rather than compensating for poor reference/support characterization;
- [ ] compare coverage gain against added boundary/handoff/covariance uncertainty and complexity;
- [ ] stop without cascade if backbone + ensemble meet the scientific objectives;
- [ ] if cascade is justified, record evidence and acceptance criteria before implementation.

### Optional cascade / merge

Implement only if Decision gate A passes.

- [ ] lower segments inherit the actual accepted upper-solution boundary, never an arbitrary reset to `SR=1`;
- [ ] propagate inherited-boundary uncertainty;
- [ ] reject inconsistent handoffs and never bridge unsupported gaps;
- [ ] merge only accepted members/segments with documented weights;
- [ ] treat covariance deliberately and verify uncertainty grows under reference/handoff ambiguity.

### P5 product redesign

- [ ] altitude-resolved support flag and retrieval top;
- [ ] accepted reference-member diagnostics and ensemble-spread uncertainty;
- [ ] clear separation of 20-min and long-mean/backbone products;
- [ ] optional cascade diagnostics only if cascade exists;
- [ ] explicit elastic-extinction LR dependence;
- [ ] new method/schema identity when redesigned semantics enter production;
- [ ] QA shows lower support edge, upper support top, reference choices and uncertainty without legitimizing unsupported near-field or upper tails.

P5 merge criterion: **maximize validated vertical support and expose where support begins/ends. Reaching 20 km alone is not success.**

## P6 — release/publication readiness — PENDING

- [ ] explicit software license reflected in citation/package metadata;
- [ ] `CITATION.cff` version/date/DOI aligned with the actual release;
- [ ] scientific method/schema changes summarized in release notes;
- [ ] frozen reference scientific environment/constraints;
- [ ] latest-compatible dependency CI lane, with minimum-supported lane considered separately;
- [ ] build sdist/wheel in CI and test clean installation from artifacts;
- [ ] core-science coverage report focused on critical modules/contracts;
- [ ] deliberate known-warning policy so new warnings cannot hide in repeated upstream warnings;
- [ ] end-to-end Level 0 -> Level 2 run from release artifact/environment;
- [ ] recorded CF/compliance validation;
- [ ] frozen machine-readable synthetic acceptance and observational regression summaries;
- [ ] required CI/review policy on release/merge path when governance is ready;
- [ ] consciously reconcile branch history with `main` before merge.

P6 acceptance gate: a third party can identify, install, cite and rerun the released scientific method in a documented reference environment, and release metadata uniquely identifies code/method/schema state.

## Immediate next gate

1. Run a real CF/compliance checker against the uploaded `20251107sapm` method-v3 Level 2 and fix genuine schema issues; preliminary inspection already shows missing `Conventions` metadata.
2. Resolve software-license policy; do not guess the license in code.
3. Continue P4 with geometrical-overlap characterization as a newly explicit lower-support requirement, alongside PC saturation/dead-time/SNR/cloud/gluing-fit evidence.
4. After P3.6 closes, begin P5 with synthetic lower+upper support tests and the Rayleigh candidate catalogue, then backbone and ensemble. Evaluate cascade only after Decision gate A.
