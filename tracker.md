# MILGRAU scientific engineering roadmap

Branch: `new-architecture`

Current method-v3 observational baseline: `docs/regression_baselines/20251107sapm_method_v3.json`

This file is the active source of truth for scientific/engineering work. Detailed completed history remains recoverable in Git; the tracker intentionally emphasizes current gates rather than retaining every closed implementation checkbox forever.

## Non-negotiable rules

- `config.yaml` owns the processing/scientific recipe; `station.yaml` owns site/instrument reality, station-derived observations, calibration state and provisional instrument estimates; Python owns equations and generic implementation.
- One productive scientific behavior has one canonical implementation.
- Missing uncertainty is never zero uncertainty.
- Correlated/model/systematic uncertainty is not silently treated as independent noise.
- Unsupported data remain unsupported; no fill/interpolation is introduced merely to extend retrieval coverage.
- A successful real-data run is regression/behavior evidence, not ground truth or proof of equation correctness.
- Scientific support is not equivalent to `isfinite(product)`.
- Vertical support must respect both the upper inversion/reference boundary and the lower instrument-validity boundary.
- A target top altitude is a validation target, never permission to extrapolate, bridge gaps or relax physics.
- Instrument constants/thresholds are evidence-derived or explicitly provisional/disabled; they are never invented to close the roadmap.
- Diagnostic instrument models must not silently become productive corrections.
- CF metadata practices are used for interoperability, but formal CF compliance is not claimed unless the product deliberately carries the required `Conventions` declaration and supporting validation evidence.

## Status overview

| Priority | Status | Purpose |
| --- | --- | --- |
| P0 | COMPLETE | truthful backward-KFS identity and support semantics |
| P1 | COMPLETE + REAL-DATA VALIDATED | canonical Level 2 architecture |
| P2 | COMPLETE | engineering guardrails and cross-platform CI |
| P3 | IN PROGRESS — CF METADATA QA GATE | current-method scientific + FAIR hardening |
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
- Ruff + full pytest on Ubuntu/Windows × Python 3.12/3.14 for the pre-overlap method-v3 baseline.

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

Interpretation constraint: aggregate optical arrays are finite from 3.75 m AGL, but SPU near-field geometrical overlap is not experimentally characterized. That lower edge is **not** accepted as validated quantitative aerosol support.

## P3.6 — FAIR software license + CF-aligned metadata QA — BLOCKING P3 COMPLETION

- [x] Software-license family selected: **BSD-3-Clause**.
- [x] Software copyright holders confirmed by project owner as the four code authors: Luisa Mello, Fábio J. S. Lopes, Alexandre C. Yoshida, and Alexandre Cacheffo.
- [x] Add root `LICENSE` using the canonical BSD-3-Clause terms and align `CITATION.cff` / package metadata with that software license.
- [x] Keep the BSD-3-Clause claim scoped to the software; no documentation/data license is inferred automatically from the code license.
- [x] Representative metadata-QA product selected: method-v3 `20251107sapm` Level 2 from revision `842389a...`.
- [x] Adopt an explicit **CF-aligned, not formally CF-compliant** policy. Current products intentionally do not carry `Conventions = "CF-..."`; therefore MILGRAU does not claim formal CF conformance. The policy and terminology are documented in `docs/cf_metadata_policy.md`.
- [x] Pin IOOS Compliance Checker 6.1.0 in the optional `validation` dependency group. Its current built-in CF checker covers CF 1.11 and is used as metadata QA, not as proof of scientific correctness.
- [ ] Run the checker against the representative real Level 2, record actionable findings, fix genuine metadata/schema issues, and document justified lidar-specific exceptions. Review relevant CF 1.12/1.13 requirements manually where they improve interoperability without making a formal compliance claim.

P3 acceptance gate: **open only on recorded CF-aligned metadata QA evidence. The software-license/holder gate is closed.**

## P4 — instrument characterization / observational validation — ACTIVE IN PARALLEL

Evidence tasks may finish as `characterized/enabled`, `insufficient evidence/provisional`, or `rejected`. No invented constant closes P4.

### P4 overlap — diagnostic model implemented; experimental characterization open

- [x] Add generic `coaxial_uniform_disk_geometric_v1` first-order overlap physics in `milgrau.physics.overlap`.
- [x] Keep all SPU-specific receiver/transmitter values and their uncertainty/status in `station.yaml`; no SPU numerical geometry is embedded in Python.
- [x] Support a profile-specific transmitter geometry with an explicit station fallback when that profile lacks geometry; record which source was resolved.
- [x] Keep the current model diagnostic only: `correction_policy=diagnostic_only_no_correction`; no Level 0/1 signal is divided by the model curve.
- [x] Refine the provisional receiver geometry from operator evidence: with 30 cm telescope, ~4 cm beam, 0.10 mrad divergence upper-bound value and reported ~500 m full overlap, infer a provisional **0.78 mrad full-angle FOV**. For `f=1.5 m` this implies a ~1.17 mm field stop, reasonably consistent with the operator estimate of roughly 1 mm. This is internal consistency, not independent validation.
- [x] Record the approximate ~1 mm physical diaphragm separately in `station.yaml` with explicit unverified status; do not replace the inferred angular FOV with a hidden code constant.
- [x] Document assumptions, equations, current diagnostic and campaign update path in `docs/overlap_model.md`.
- [ ] Experimentally determine FOV convention/value, field-stop diameter, beam diameter convention, wavelength-dependent divergence, alignment/separation and overlap stability.
- [ ] Use telecover/alignment mapping and preferably an independent horizontal/Raman-based method to validate the overlap curve and lower quantitative-support boundary.
- [ ] Only after characterization decide whether a productive overlap correction is justified; if introduced, version the scientific method and propagate uncertainty/support explicitly.

Other P4 evidence tasks:

- [ ] Characterize physical photon-counting saturation under operational SPU conditions using AN/PC overlap and preferably controlled attenuation before defining a traceable `max_rate_mhz`.
- [ ] Move any surviving provisional PC guard to raw observed Level 1 PC rate rather than a corrected/background-subtracted proxy.
- [ ] Audit dark-current subtraction versus nonlinear dead-time correction with instrument evidence and quantify the practical difference over observed SPU regimes.
- [ ] Characterize propagated-error SNR before enabling hard Rayleigh-window SNR rejection.
- [ ] Validate cloud/layer screening before enabling it as productive reference-window rejection.
- [ ] Quantify gluing slope/intercept uncertainty and covariance with overlap resampling/bootstrap/Monte Carlo; either justify continued exclusion from the declared partial budget or introduce a new versioned propagation model.
- [ ] Compare representative retrievals with LPP where practical and with SCC/ELDA methodological expectations without claiming numerical identity.
- [ ] Keep elastic extinction explicitly conditional on assumed aerosol lidar ratio; do not present it as semantically equivalent to Raman extinction.

Current observation relevant to dead time: 355.PC had no clipped profiles in `20251107sapm`; 532.PC had clipping in 8/167 profiles, with maximum clipped-bin fraction 0.00075 per profile. Physical saturation remains uncharacterized.

P4 acceptance gate: every instrument-dependent productive threshold/correction/support boundary has evidence/provenance, or remains explicitly provisional/disabled.

## P5 — altitude-resolved support + high-column R&D — PENDING

Start implementation after P3.6 closes. P4 evidence may continue in parallel, but P5 support semantics must honor unresolved P4 limitations.

### P5.1 support semantics + synthetic truth

Frozen semantics:

- `retrieval_support_flag(..., altitude)` means scientifically supported productive optical retrieval, not mere finiteness.
- Upper support ends at the actual accepted inversion/reference boundary; no backward result exists above that boundary.
- Lower support must respect validated overlap/instrument-correction domain; a finite KFS value below that domain has support 0.
- A provisional modeled overlap curve is diagnostic evidence, not by itself a validated support flag.
- Internal unsupported gaps are never bridged.
- `retrieval_top_altitude_m` is the highest supported altitude; NaN when no supported bin exists.
- The redesigned product must also expose the lower supported edge (for example `retrieval_bottom_altitude_m`) rather than describing support only by its top.

To implement:

- [ ] Add synthetic support tests covering upper boundary, lower overlap boundary, noisy tail, missing uncertainty and internal gaps.
- [ ] Expose altitude-resolved support and lower/upper bounds only after those tests pass.
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

P5 merge criterion: **maximize validated vertical support and expose where support begins/ends. Reaching 20 km alone is not success.**

## P6 — release/publication readiness — PENDING

- [x] explicit BSD-3-Clause software license with confirmed code-author holders reflected in root license, citation metadata and package metadata;
- [ ] `CITATION.cff` version/date/DOI aligned with the actual release;
- [ ] scientific method/schema changes summarized in release notes;
- [ ] frozen reference scientific environment/constraints;
- [ ] latest-compatible dependency CI lane, with minimum-supported lane considered separately;
- [ ] build sdist/wheel in CI and test clean installation from artifacts;
- [ ] core-science coverage report focused on critical modules/contracts;
- [ ] deliberate known-warning policy so new warnings cannot hide in repeated upstream warnings;
- [ ] end-to-end Level 0 -> Level 2 run from release artifact/environment;
- [ ] recorded CF-aligned metadata QA; formal CF compliance remains a separate future policy decision if the project chooses to claim it;
- [ ] frozen machine-readable synthetic acceptance and observational regression summaries;
- [ ] required CI/review policy on release/merge path when governance is ready;
- [ ] consciously reconcile branch history with `main` before merge.

P6 acceptance gate: a third party can identify, install, cite and rerun the released scientific method in a documented reference environment, and release metadata uniquely identifies code/method/schema state.

## Immediate next gate

1. Run IOOS Compliance Checker 6.1.0 / CF 1.11 as metadata QA against the representative `20251107sapm` method-v3 Level 2; record findings and fix genuine metadata/schema issues without adding a formal `Conventions` claim.
2. Review relevant CF 1.12/1.13 metadata requirements manually where they improve interoperability, while retaining the explicit CF-aligned/non-compliant terminology.
3. Continue P4 instrument evidence in parallel, with the 532.PC dead-time/saturation behavior as the next concrete real-data investigation and overlap characterization left experimental/diagnostic.
4. After P3.6 metadata QA closes, begin P5 with synthetic lower+upper support tests and the Rayleigh candidate catalogue, then backbone and ensemble. Evaluate cascade only after Decision gate A.
