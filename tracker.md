# MILGRAU scientific engineering roadmap

Branch: `new-architecture`

This file is the active source of truth for current scientific/engineering gates. Detailed completed history remains recoverable in Git and in the frozen evidence files under `docs/regression_baselines/`.

Current productive code identity:

- Level 2 schema: **v3** — `auditable_rayleigh_candidate_catalogue`;
- Level 2 retrieval method: **v4** — QA-first Rayleigh candidate selection with the established method-v3 uncertainty/support semantics;
- productive inversion: backward Klett–Fernald–Sasano;
- code gate for schema-v3 catalogue integration: CI green at `72133cfc446ad5b23ac1de5d9242af2d44cab9e0` on Ubuntu/Windows, Python 3.12/3.14, Ruff + full pytest matrix.

Frozen observational comparison baseline remains method v3:
`docs/regression_baselines/20251107sapm_method_v3.json`.
It is regression/behavior evidence, not ground truth.

## Non-negotiable scientific/engineering rules

- `config.yaml` owns the processing/scientific recipe; `station.yaml` owns station/instrument reality, calibration state, station-derived observations and explicitly provisional estimates; Python owns generic equations and implementation.
- Missing uncertainty is never zero uncertainty.
- Correlated/model/systematic uncertainty is not silently treated as independent noise.
- Unsupported data remain unsupported; no interpolation/fill is introduced merely to extend retrieval coverage.
- A successful real-data run is observational regression evidence, not proof of physical correctness.
- Scientific support is not equivalent to `isfinite(product)`.
- Backward retrieval never implies support above its accepted exact boundary.
- The lower scientific-support boundary requires evidence-backed instrument validity; finite near-range KFS output alone is insufficient while overlap remains uncharacterized.
- A high-altitude target is an objective, not permission to extrapolate or weaken QA.
- Long averaging must expose temporal contribution/stability and must not let a transient interval silently define an entire measurement.
- Multiple accepted reference solutions are sensitivity/ensemble evidence, not independent truths; their disagreement must remain visible.
- A KFS member is tied to the signal/molecular state at its exact local boundary.
- Elastic extinction remains conditional on the assumed aerosol lidar ratio.
- Diagnostic instrument models never silently become productive corrections.
- No physical PC saturation threshold, overlap cutoff, SNR threshold, cloud threshold or scientific tolerance is invented merely to close a gate.
- NetCDF products should be readable and self-describing, but MILGRAU currently makes no formal external metadata-convention conformance claim.

## Status overview

| Priority | Status | Purpose |
| --- | --- | --- |
| P0 | COMPLETE | truthful backward-KFS identity and support semantics |
| P1 | COMPLETE + REAL-DATA VALIDATED | canonical Level 2 architecture |
| P2 | COMPLETE | engineering guardrails and cross-platform CI |
| P3 | IN PROGRESS — RELEASE/FAIR GATE | license decision + representative metadata review |
| P4 | PARALLEL EVIDENCE WORK | instrument characterization / observational validation |
| P5.0 | COMPLETE | frozen evidence + historical audit + physical Rayleigh-window geometry |
| P5.1 | COMPLETE + REAL-DATA CHECKED | altitude-resolved inversion support |
| P5.2 | CODE COMPLETE — SCHEMA-3 REAL CHECK PENDING | QA-first Rayleigh catalogue + auditable selection |
| P5.3 | IN PROGRESS | real-data candidate experiments / interpretation |
| P5.4 | NEXT R&D GATE | temporally honest high-column backbone |
| P5.5+ | PENDING / DEFERRED | ensemble, optional cascade, merge, uncertainty extension |
| P6 | PENDING | reproducible release/publication process |

## P3 — release / FAIR hardening

### Software licensing

- [x] Code copyright holders identified by the project owner as the four code authors.
- [ ] Select the software license deliberately. The earlier provisional BSD-3-Clause choice was withdrawn and is not current project policy.
- [ ] After selection, add canonical root `LICENSE` and align `CITATION.cff` and package metadata.
- [ ] Keep software, documentation and observational-data licensing separate unless explicitly decided otherwise.

### NetCDF metadata readability

- [x] Adopt `docs/netcdf_metadata_policy.md`: readable/self-describing products without an unsupported convention-conformance claim.
- [x] Keep physical units where meaningful, explicit unit status for instrument-native quantities, numeric machine-readable flags, honest missing values and provenance.
- [ ] Record a representative Level 2 metadata review and fix genuine ambiguity/inconsistency.

P3 blocks release/FAIR completion; it does not block scientific P4/P5 development.

## P4 — instrument characterization / observational validation

### Overlap

- [x] Generic `coaxial_uniform_disk_geometric_v1` diagnostic model implemented; SPU-specific estimates remain station-owned.
- [x] Current provisional geometry is consistent with a reported ~500 m full-overlap scale under the current inferred assumptions.
- [x] Productive policy remains `diagnostic_only_no_correction`; no validated lower retrieval-support cutoff is inferred from the model.
- [ ] Experimentally determine receiver FOV/convention, field-stop diameter, beam diameter convention, wavelength-dependent divergence, alignment/separation and temporal stability.
- [ ] Validate with telecover/alignment mapping and preferably an independent horizontal/Raman-based overlap method before productive correction/support use.

### Photon-counting dead time / saturation

- [x] Preserve raw observed PC rates before dark-current subtraction/dead-time correction and keep denominator/clipping diagnostics separate from physical saturation semantics.
- [x] Preserve available dark-acquisition laser-shot counts.
- [x] `20251107sapm` observational evidence records the eight extreme 532.PC profiles and correction-order sensitivity without interpreting them as a physical detector threshold.
- [x] Keep the current productive dark-before-dead-time order pending broader evidence.
- [ ] Characterize physical PC saturation under operational SPU conditions with AN/PC overlap and preferably controlled attenuation before enabling a traceable threshold/mask.

### Other evidence

- [ ] Characterize propagated-error SNR before enabling a hard Rayleigh SNR gate.
- [ ] Validate cloud/layer screening before productive reference rejection.
- [ ] Quantify gluing slope/intercept uncertainty and covariance before expanding the uncertainty budget.
- [ ] Compare representative retrievals with LPP and SCC/ELDA expectations without treating either as ground truth.
- [ ] Keep future nighttime Raman retrieval as an independent validation/R&D path until its channel physics, overlap, uncertainty and synthetic truth are validated.

## P5.0 — frozen evidence and historical audit — COMPLETE

- [x] Freeze `20251107sapm` method-v3 observational baseline with hashes, references, block success and finite tops.
- [x] Keep synthetic truth distinct from observational regression evidence.
- [x] Audit the historical `20241219nt` archive and preserve reusable lessons/machine-readable metrics.
- [x] Record why historical apparent ~30 km coverage is not itself supported backward optical retrieval.
- [x] Express Rayleigh reference-window width physically in meters and resolve it deterministically on the actual uniform altitude grid.
- [x] Keep Rayleigh search bounds physical and fail explicitly when geometry is not representable.

## P5.1 — altitude-resolved inversion support — COMPLETE + REAL-DATA CHECKED

Implementation:

- [x] Pure backward-support contract in `milgrau.level2.support`.
- [x] Synthetic tests cover lower instrument-mask examples, exact upper boundary, noisy/missing tail, missing/negative uncertainty and internal gaps.
- [x] Support is the contiguous valid path ending at the accepted exact Rayleigh boundary; internal unsupported bins cannot be bridged.
- [x] Productive Level 2 exposes `retrieval_inversion_support_flag` and block counterpart.
- [x] Productive Level 2 exposes aggregate/block inversion top and algorithmic bottom altitude.
- [x] Productive Level 2 exposes altitude-resolved `retrieval_inversion_effective_block_count`.
- [x] NetCDF contract checks dimensions, binary flags, contiguity, top/bottom consistency and exact block-count reconstruction.
- [x] No validated lower instrument/overlap mask is claimed; future stronger `retrieval_support_flag` remains reserved.

Real observational check from the user rerun generated from source revision
`85f4bdc3cc8f17af94f7d9f9a6f10ccfcf21fad9`:

- uploaded Level 2 SHA-256: `a3d65e8b149526460bf66455183e7f307492534381f61b3eef81d6e602b8bbd0`;
- product declares schema 2 / method 4 and complete/success;
- 355 nm aggregate inversion top = **6101.25 m**; 5/5 blocks support through **5606.25 m**, while the final supported tail reaches 6101.25 m with only 1 block;
- 532 nm aggregate inversion top = **6318.75 m**; 5/5 blocks support through **5748.75 m**, while the final supported tail reaches 6318.75 m with only 1 block;
- aggregate effective-count distributions exactly reproduce the previously frozen/offline support accounting;
- algorithmic bottom remains 3.75 m in this file and is **not** interpreted as validated near-field scientific support.

This closes the P5.1 acceptance gate: “more finite bins” is now distinguishable from contiguous inversion support and from the number of blocks supporting each altitude.

## P5.2 — Rayleigh candidate catalogue: QA first, ranking second

Implementation state:

- [x] Typed candidate contract with complete window geometry and QA diagnostics.
- [x] Enumerate every fully contained candidate in the configured physical search interval.
- [x] Compute valid fraction, relative slope, relative variance, origin-constrained calibration, free-intercept diagnostic, diagnostic cost and propagated-uncertainty SNR before selection.
- [x] Preserve accepted and rejected candidates separately with explicit rejection bit mask.
- [x] SNR remains diagnostic-only; no hard SPU SNR threshold is enabled.
- [x] Productive selector is now `enumerate -> diagnose -> minimum QA -> rank accepted candidates only`.
- [x] Productive method-v4 ranking remains deliberately conservative: minimum historical `relative_slope + relative_variance` among accepted candidates, deterministic lower-grid-index tie break; **no altitude preference** is introduced.
- [x] Retrieval method identity is v4 because a rejected minimum-cost raw candidate can no longer prevent use of another QA-passing candidate.
- [x] Schema 3 persists an auditable candidate catalogue per block/wavelength/candidate.
- [x] Persist candidate geometry, QA metrics, SNR diagnostic, rejection mask, accepted flag, pre-QA unfiltered-minimum flag and productive-selected flag.
- [x] Schema-3 validation requires exactly one unfiltered minimum for each evaluated block and exactly one accepted productive selection for each successful block; persisted selected altitude must equal the actual KFS Rayleigh reference.
- [x] Generic Level 2 contract dispatch requires the catalogue for schema 3+ products.
- [x] Cross-platform schema-3 code gate green at `72133cfc446ad5b23ac1de5d9242af2d44cab9e0`.
- [ ] Inspect one real schema-3 `20251107sapm` product and verify persisted candidate counts/rejection reasons/selected flags against the known method-v4 references.
- [ ] Add cloud/layer contamination diagnostics only after validation; productive rejection remains disabled meanwhile.
- [ ] Define minimum separation/correlation rules before nearby overlapping accepted windows are treated as distinct ensemble members.

The method-v4/schema-2 observational rerun supplied by the user selected the same aggregate references as the frozen method-v3 baseline:
355 nm **5748.75 m** and 532 nm **5816.25 m**, with 5/5 KFS blocks at each wavelength.
This is regression evidence that QA-first ordering did not introduce reference drift for this case; it does not validate those references as unique physical truth.

## P5.3 — real-data candidate experiments

`20251107sapm`:

- [x] Freeze the preliminary complete candidate experiment in `docs/regression_baselines/20251107sapm_rayleigh_catalogue_preliminary.json`.
- [x] Show that minimum historical cost among QA passers reproduces the method-v3 references in all 10 block/wavelength cases.
- [x] Show that many high-altitude windows formally pass current permissive shape QA, while many have weak propagated SNR and/or high variance; “choose highest passing” is therefore unsupported.
- [x] Method-v4 real rerun keeps the same ~5.7–5.8 km aggregate references and ~6.1/~6.3 km inversion tops.
- [ ] After schema-3 rerun, use the persisted catalogue to inspect all real rejection reasons and identify whether future high-column gain is SNR/temporal-support limited rather than shape-gate limited.
- [ ] Compare future redesigned 0–6 km optical solution quantitatively against the frozen method-v3 baseline when a higher-boundary method becomes productive.

Historical `20241219nt` stress case:

- [x] Early ~20 min interval contains high-altitude molecular-like information that is lost in later blocks.
- [x] Whole ~53 min averaging can make high-altitude candidates reappear because the early interval dominates far-range contribution.
- [x] Treat this as direct motivation for temporal-support/stability diagnostics; a long mean alone cannot establish representative full-interval support.

## P5.4 — high-column backbone — NEXT R&D GATE

Do not implement a naive whole-measurement mean as a productive retrieval.

- [ ] Add a pure temporal-support/stability diagnostic contract before adding productive backbone retrieval.
- [ ] Quantify contributing profile/block count, start/stop, effective duration and altitude-resolved contribution/support fraction.
- [ ] Add sensitivity to contiguous subwindows so a transient early high-altitude contribution is visible.
- [ ] Define an auditable criterion for rejecting/splitting a backbone when far-range information is temporally non-representative; criterion must be evidence-backed, not invented to hit a target altitude.
- [ ] Test the diagnostics on synthetic temporally stable and transient cases.
- [ ] Apply the diagnostic offline to `20251107sapm` and `20241219nt` before any productive long-mean KFS.
- [ ] Only after those gates, add a distinct long-mean/high-column retrieval input with explicit duration/provenance and common signal/error support.
- [ ] Preserve original-resolution/block signals; backbone is additional state, not a destructive replacement.
- [ ] Evaluate vertical aggregation only if needed and validate its bias with synthetic truth.
- [ ] `target_top_altitude_m` may later be an R&D objective (initially 20 km), never a permission to extrapolate or weaken QA.
- [ ] If no trustworthy boundary exists at/above target, report the lower supported top.

P5.4 acceptance: longer averaging must demonstrably increase usable high-altitude information without materially biasing the lower column or hiding temporal nonstationarity.

## P5.5 — robust multi-reference ensemble — PENDING

- [ ] Require individually QA-passing narrow local reference windows.
- [ ] Define separation/correlation rules before treating candidates as distinct evidence.
- [ ] Keep each KFS member tied to its exact local boundary.
- [ ] Combine only members whose backward branch covers the altitude bin.
- [ ] Define explicit weighting from quality/uncertainty quantities.
- [ ] Add between-reference spread/reference-choice sensitivity as an uncertainty component.
- [ ] Expose effective member count separately from effective block count.

## P5.6 — cascaded backward retrieval — DEFERRED DECISION GATE

Cascade is not the default next step. Consider it only if backbone + ensemble leave a scientifically meaningful coverage gap.

If eventually justified:

- [ ] Solve ordered segments from high to low.
- [ ] Inherit each lower segment boundary from the accepted upper solution; never reset `SR_ref = 1` merely because a segment starts.
- [ ] Propagate inherited-boundary uncertainty.
- [ ] Require uncertainty-aware overlap consistency and reject inconsistent handoffs.
- [ ] Never bridge unsupported internal gaps.

## P5.7 — overlap merge — DEFERRED

Implement only if multiple accepted solutions actually need stitching. Merge weights/continuity/uncertainty behavior must be tested rather than chosen cosmetically.

## P5.8 — molecular-model and uncertainty consistency — PARALLEL

- [ ] Audit exact molecular extinction/backscatter definitions and internally consistent `S_m = alpha_mol/beta_mol` by wavelength.
- [ ] Resolve total-Rayleigh vs Cabannes/effective detected molecular component for the actual receiver/filter semantics.
- [ ] Quantify 355/532 retrieval sensitivity to the current ~1.4–1.5% molecular-lidar-ratio difference before any productive change.
- [ ] Version the retrieval method explicitly if productive molecular-lidar-ratio semantics change.
- [ ] Quantify gluing fit slope/intercept uncertainty/covariance and add only if material.
- [ ] Add future ensemble/backbone/cascade uncertainty components only with explicit dependence assumptions.

## P5.9 — Level 2 schema/provenance/QA

- [x] Schema 2 introduced altitude-resolved inversion support/top/bottom/effective-block count.
- [x] Schema 3 adds auditable Rayleigh candidate catalogue without changing method-v4 physics.
- [x] Currentness/provenance separates package version, product schema and retrieval-method identity.
- [x] Generic schema-3 contract validates the candidate catalogue rather than trusting the version attribute alone.
- [ ] Update QA plots to show effective optical retrieval top/support and selected/accepted candidate locations; finite scattering ratio must not visually imply aerosol retrieval.
- [ ] Add temporal-support panel only when backbone diagnostics become productive.
- [ ] Add ensemble/cascade panels only if those algorithms become productive.

## P5.10 — validation matrix

Synthetic already established:

- [x] Pure molecular atmosphere -> near-zero aerosol within numerical tolerance.
- [x] Controlled aerosol layers at 355/532 -> recovery within current discretization tolerance.
- [x] Vertical-grid convergence.
- [x] Missing uncertainty/internal invalid gaps fail support explicitly.
- [x] Candidate catalogue retains accepted/rejected windows and QA-first selection.
- [ ] Temporally heterogeneous sequence exposes/rejects transient-only backbone support.
- [ ] Multiple valid high references -> ensemble stability and reference-sensitivity uncertainty.
- [ ] Biased/contaminated candidate -> rejection or visible uncertainty impact.
- [ ] Upper-tail noise -> supported top falls gracefully under the future high-column method.
- [ ] Backbone averaging/aggregation -> no material lower-column truth bias.
- [ ] Molecular-lidar-ratio semantics sensitivity quantified.

Real SPU:

- [x] Current method-v4 `20251107sapm` behavior checked: complete 355/532 retrieval, unchanged ~5.7–5.8 km references, ~6.1/~6.3 km aggregate inversion tops.
- [ ] Validate schema-3 persisted real catalogue.
- [ ] Extend defensible 355/532 support materially above current tops only when evidence supports it.
- [ ] Evaluate whether 15 km is supportable for representative measurements.
- [ ] Attempt 20 km target only when a trustworthy boundary/support exists at or above the needed altitude.
- [ ] Add clear-night, high-aerosol, cloud, weak-signal, temporally changing and materially different PC/AN-dominance cases before general validation.
- [ ] Compare representative cases with LPP/SCC/ELDA under matched assumptions without treating them as truth.

P5 success criterion: **maximize defensible inversion-supported vertical coverage while exposing where/why support ends, preserving temporal representativeness, uncertainty honesty and exact boundary provenance. Success is not “reaches 20 km.”**

## P6 — release/publication readiness

- [ ] Explicit software license selected and reflected consistently in root license/citation/package metadata.
- [ ] `CITATION.cff` aligned with the actual release version/date/DOI.
- [ ] Scientific method/schema changes summarized in release notes.
- [ ] Frozen reproducible scientific environment/constraints.
- [ ] Build sdist/wheel in CI and test clean installation from artifacts.
- [ ] Core-science coverage report and deliberate warning policy.
- [ ] End-to-end Level 0 -> Level 2 run from release artifact/environment.
- [ ] Recorded representative NetCDF metadata review.
- [ ] Frozen machine-readable synthetic acceptance and observational regression summaries.
- [ ] Reconcile branch history with `main` before release/merge.

## Immediate next gate

1. Run `milgrau-lebear -i 20251107sapm --force` once from schema-3 HEAD and inspect the resulting Level 2 candidate catalogue; only the new NetCDF is required unless QA plots change unexpectedly.
2. Freeze/record that real schema-3 catalogue evidence and close the remaining observational P5.2 check.
3. Implement **temporal-support/stability diagnostics first**, with synthetic stable/transient tests, before any productive high-column backbone.
4. Apply those temporal diagnostics to `20251107sapm` and historical `20241219nt` evidence; only then decide backbone averaging strategy.
5. Audit molecular-lidar-ratio consistency and gluing fit-parameter uncertainty in parallel.
6. Proceed to productive backbone and later multi-reference ensemble only from evidence that they extend support without hiding temporal nonstationarity or boundary ambiguity.
