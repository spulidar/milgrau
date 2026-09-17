# MILGRAU scientific engineering roadmap

Branch: `new-architecture`

This is the active source of truth for current scientific/engineering gates. Detailed completed history remains recoverable in Git and in frozen evidence under `docs/regression_baselines/`.

## Current productive identity

- Level 2 schema: **v3** — auditable Rayleigh candidate catalogue.
- Level 2 retrieval method: **v4** — QA-first candidate selection; backward KFS on the native vertical grid; exact measured RCS bin remains `X_ref`.
- No productive fitted boundary, vertical aggregation, long-mean backbone, SNR threshold, cloud threshold, temporal threshold, overlap cutoff, ensemble or cascade is enabled.
- Frozen real schema-v3 product: `20251107sapm_level2_optical.nc`, SHA-256 `32b793fc7874f850ee38273f7073c957194f87dd08fed1c340e96b976917a212`, source revision `b68e4d812a37c2a003e49e30c15c2e112e4c8360`.
- Frozen method-v3 comparison: `docs/regression_baselines/20251107sapm_method_v3.json`.
- Observational comparisons are regression/behavior evidence, never ground truth.

Latest cross-platform R&D code gate covering explicit correlated fitted-boundary noise, the diagnostic high-column evidence vector and molecular-lidar-ratio sensitivity is green at `f9490dbf392748d5d40a5845b39a94703f51ad95` (Ruff + pytest, Ubuntu/Windows, Python 3.12/3.14).

## Non-negotiable rules

- `config.yaml` owns the processing/scientific recipe; `station.yaml` owns instrument/station reality and calibration history; Python owns generic equations/implementation.
- Missing uncertainty is never zero uncertainty; dependence/covariance assumptions are explicit.
- Unsupported bins remain unsupported; no interpolation/fill is introduced merely to extend coverage.
- `isfinite(product)` is not scientific support.
- Backward retrieval support never extends above its exact accepted boundary.
- Near-range finite KFS output is not validated scientific support while overlap remains uncharacterized.
- High-altitude coverage is an objective, not permission to extrapolate or weaken QA.
- Long averaging must expose temporal contribution/stability.
- Vertical aggregation is a declared resolution trade, not new information; post-retrieval cosmetic smoothing cannot create support.
- A window-fitted boundary is a different retrieval assumption and requires a new retrieval-method identity if promoted.
- A clean center bin or zero anomaly flag never certifies an entire molecular window.
- Random Monte Carlo spread never substitutes for systematic/model-bias QA; the same noisy bins are not counted twice as independent signal and boundary-fit uncertainty.
- Multiple references are sensitivity/ensemble evidence, not independent truths.
- Elastic extinction remains conditional on assumed aerosol lidar ratio.
- No physical PC saturation, overlap, SNR, cloud or temporal threshold is invented to close a gate.
- NetCDF products are self-describing/readable, but MILGRAU currently makes no formal external convention-conformance claim.

## Status overview

| Priority | Status | Purpose |
| --- | --- | --- |
| P0 | COMPLETE | truthful backward-KFS identity/support semantics |
| P1 | COMPLETE + REAL-DATA VALIDATED | canonical Level 2 architecture |
| P2 | COMPLETE | engineering guardrails + cross-platform CI |
| P3 | IN PROGRESS — RELEASE/FAIR GATE | license + representative metadata review |
| P4 | PARALLEL EVIDENCE WORK | instrument characterization / observational validation |
| P5.0 | COMPLETE | frozen evidence + legacy audit + physical Rayleigh geometry |
| P5.1 | COMPLETE + REAL-DATA CHECKED | altitude-resolved inversion support |
| P5.2 | COMPLETE + REAL-DATA CHECKED | QA-first Rayleigh catalogue + auditable selection |
| P5.3 | IN PROGRESS | real-data candidate interpretation |
| P5.4 | IN PROGRESS — HIGH-COLUMN R&D | temporal + boundary/SNR + covariance + contamination |
| P5.5 | PENDING | robust multi-reference ensemble |
| P5.6–P5.7 | DEFERRED | optional cascade / solution merge |
| P5.8 | IN PROGRESS — PARALLEL | molecular model + uncertainty consistency |
| P5.9 | IN PROGRESS | QA/schema/provenance presentation |
| P5.10 | IN PROGRESS | validation matrix |
| P6 | PENDING | reproducible release/publication |

## P3 — release / FAIR hardening

- [x] Code copyright holders identified by project owner as the four code authors.
- [ ] Deliberately select software license; earlier provisional BSD-3-Clause choice was withdrawn.
- [ ] Add root `LICENSE` only after that decision; align `CITATION.cff` and package metadata.
- [ ] Keep software/documentation/observational-data licensing distinct unless deliberately unified.
- [x] Adopt `docs/netcdf_metadata_policy.md` for readable/self-describing products without unsupported convention claims.
- [ ] Record representative Level 2 metadata review and fix genuine ambiguity/inconsistency.

P3 blocks release completion, not P4/P5 scientific work.

## P4 — instrument / observational evidence

### Overlap
- [x] Generic diagnostic geometric model exists; SPU estimates remain station-owned and non-productive.
- [x] Current provisional geometry is compatible with reported ~500 m full-overlap scale under current assumptions.
- [x] Policy remains `diagnostic_only_no_correction`; no lower support cutoff inferred.
- [ ] Experimentally determine FOV/convention, field stop, beam diameter convention, wavelength-dependent divergence, alignment/separation and stability.
- [ ] Validate with telecover/alignment mapping and preferably independent horizontal/Raman-based overlap evidence.

### Photon counting
- [x] Preserve raw observed PC rate, dead-time denominator/clipping diagnostics and dark `NShots` separately from physical saturation semantics.
- [x] `20251107sapm` correction-order sensitivity quantified; productive dark-before-dead-time order retained pending broader evidence.
- [ ] Characterize physical PC saturation with AN/PC overlap and preferably controlled attenuation before enabling a threshold/mask.

### Other evidence
- [ ] Characterize propagated-error SNR across additional regimes before any hard Rayleigh SNR gate.
- [x] Synthetic legacy-inspired anomaly-screening limits established: sharp layers can be caught, but broad smooth contamination can materially bias the fit with zero detector flag (`dc49784e27c0169c9e3c76c3259f01d6df4df63f`).
- [ ] Validate cloud/layer screening observationally before productive reference rejection/window fitting.
- [ ] Quantify gluing slope/intercept uncertainty/covariance.
- [ ] Compare representative products with LPP and SCC/ELDA under matched assumptions without treating either as truth.

## P5.0 — frozen evidence / legacy audit — COMPLETE

- [x] Freeze `20251107sapm` method-v3 baseline with hashes/references/block success/tops.
- [x] Synthetic truth kept distinct from observational regression evidence.
- [x] Audit historical `20241219nt`; preserve useful lessons and explicit provenance limitations.
- [x] Record why historical apparent ~30 km numerical coverage is not equivalent to supported backward optical retrieval.
- [x] Recover useful legacy ideas as hypotheses: long-mean backbone, distributed molecular information, scattering-ratio QA, future Raman validation; reject non-local boundary shortcuts and post-retrieval smoothing as support evidence.
- [x] Rayleigh search/window geometry expressed in physical meters on the real grid.

Evidence: `docs/legacy_level2_lessons.md`, `docs/regression_baselines/20241219nt_legacy_level2_audit.json`.

## P5.1 — altitude-resolved inversion support — COMPLETE + REAL-DATA CHECKED

- [x] Pure backward support contract; internal invalid gaps cannot be bridged.
- [x] Aggregate/block inversion support, top/bottom and altitude-resolved effective block count persisted/validated.
- [x] No validated lower instrument/overlap mask claimed.

`20251107sapm`: 355 top 6101.25 m (5/5 through 5606.25 m); 532 top 6318.75 m (5/5 through 5748.75 m); final tails are one-block support. Algorithmic bottom 3.75 m is not interpreted as validated near-field support.

## P5.2 — Rayleigh candidate catalogue — COMPLETE + REAL-DATA CHECKED

- [x] Enumerate every fully contained physical candidate window.
- [x] Persist valid fraction, relative slope/variance, calibration diagnostics, diagnostic cost, bin-wise propagated SNR, accepted/rejected state and rejection bits.
- [x] Productive selection is `enumerate -> minimum QA -> rank accepted only`.
- [x] Method-v4 ranking remains historical `relative_slope + relative_variance`; no altitude preference or SNR gate.
- [x] Schema 3 validates one unfiltered minimum + one accepted productive selection per successful block/wavelength and exact equality with KFS reference.

Real schema-v3 audit: 25,340/25,340 candidate slots evaluated, 16,224 pass enabled minimum-QA gates; all 10 selected candidates are accepted minimum-cost accepted candidates; unfiltered minima are already accepted in all 10 cases. Aggregate references remain 5748.75 m (355) and 5816.25 m (532).

Evidence: `docs/regression_baselines/20251107sapm_method_v4_schema3_catalogue.json`.

Remaining for future ensemble/cloud QA:
- [ ] Validate observational contamination diagnostics.
- [ ] Define candidate separation/correlation rules before nearby overlapping windows count as distinct ensemble members.

## P5.3 — real-data candidate interpretation — IN PROGRESS

For `20251107sapm`:
- [x] Shape QA permits many candidates up to ~23.6–24.5 km; therefore shape pass != supported optical retrieval.
- [x] Median accepted-candidate bin-wise SNR falls from ~14.8/15.9 (355/532) at 5–10 km to ~3.13/3.59 at 10–15 km, ~1.20/1.32 at 15–20 km and ~0.90/0.87 at 20–25 km.
- [x] Window-level calibration uncertainty contains much more information than one-bin SNR; see P5.4.
- [x] Limitation is now decomposed into boundary noise, covariance/resolution trade, temporal representativeness and contamination/model bias instead of one opaque top-altitude failure.
- [ ] Compare a future redesigned retrieval against frozen lower-column baseline before promotion.

## P5.4 — temporally honest high-column backbone R&D

### Temporal evidence
- [x] Explicit block weights/support, dominant-contribution fraction, contiguous subwindow comparison and high-candidate persistence diagnostics.
- [x] `20251107sapm`: all five blocks have finite signal/error through inspected <30 km; below 10 km dominant block contribution ~0.25–0.26; at 15–20 km median ~0.385 and ~13–14% of bins exceed 50% single-block contribution.
- [x] `20251107sapm`: accepted-candidate persistence ≥15/20 km is 1.0 for both wavelengths.
- [x] Historical `20241219nt`: only first of three ~20-min blocks has ≥15/20 km candidates; weighted persistence 37/105 ≈ 0.352.

Evidence: `20251107sapm_temporal_support_preliminary.json`, `high_column_candidate_persistence_comparison.json`.

### Vertical aggregation / smoothing R&D
- [x] Non-overlapping pre-retrieval aggregation with strict support and explicit independent-vs-fully-correlated uncertainty limits.
- [x] Empirical PC lag-1 residual correlation in real far range is modest (~0.13 at 355, ~0.10 at 532); longer lags near zero in this event.
- [x] Diagnostic correlation model predicts ~2.5× SNR gain at 60 m and ~3.4–3.5× at 120 m in 15–20 km; fully correlated limit gives essentially no gain.
- [x] Synthetic KFS truth at 15/30/60/120/240 m stays within current <5% R&D guards relative to truth represented on that coarse grid while explicitly exposing narrow-layer peak loss at 240 m.
- [ ] Do not choose any productive aggregation width from one event; broaden synthetic layer/noise cases and real regimes.

Evidence: `20251107sapm_vertical_aggregation_preliminary.json`. Gate: `4a71eed6b9cb235b3d31959c68c74ce3a05fb83d`.

### Rayleigh-window boundary R&D
- [x] Window calibration uncertainty implemented under independent, fully correlated and explicit lag-autocorrelation assumptions.
- [x] Real median window-factor SNR under current empirical short-range covariance diagnostic is ~9.1/10.4 at 15–20 km and ~6.3/6.9 at 20–25 km (355/532), despite bin-wise SNR near 1.
- [x] Exact-center-bin vs same-altitude window-fit mismatch grows from ~4–5% at 5–10 km to ~43–47% at 15–20 km and ~47–50% at 20–25 km; current selected ~5.6–6.3 km references remain much closer (~0.5–7.6%).
- [x] Clean synthetic window-fit boundary reduces strong exact-bin noise without globally smoothing the profile.
- [x] Contaminated synthetic window can bias fitted boundary despite aerosol-free center bin.
- [x] Independent-noise fitted-boundary MC recomputes the fit from the same perturbed bins, matches analytic fit uncertainty and avoids double counting (`9b88493322b8cac29707bf9246a703351a4d0b82`).
- [x] Explicit caller-supplied full window correlation matrix is supported/validated; no AR(1) or covariance law is inferred. Positive correlation demonstrably reduces naive averaging gain (`9250f6f29fc2c1d2b167db59a988c95777fc92f6`, included in green `f9490db...` gate).
- [x] MC counterexample proves systematic contamination bias can exceed random MC spread by many sigma; random uncertainty does not certify molecular purity.
- [ ] Extend noise amplitudes, asymmetric contamination and boundary-placement offsets.
- [ ] Validate contamination observables on additional real regimes.
- [ ] If promoted, bump retrieval method and expose boundary estimator, uncertainty/dependence model and contamination QA.

Evidence: `20251107sapm_rayleigh_window_uncertainty_preliminary.json`, `20251107sapm_exact_vs_window_boundary_preliminary.json`.

### High-column evidence vector / decision rule
- [x] Add typed diagnostic `HighColumnEvidence` record with separate fields for shape-QA state, bin-wise SNR, window SNR under multiple dependence assumptions, temporal persistence, block dominance, subwindow disagreement, contamination fraction, effective resolution, boundary estimator and noise model.
- [x] The evidence record deliberately contains **no composite score, high-column pass/fail, ranking or target-altitude preference**; unevaluated evidence remains NaN rather than a favorable default.
- [ ] Populate the vector for real schema-v3 candidates in an offline analysis helper.
- [ ] Only after additional evidence, formulate an experimental decision rule without tuning it to 15/20 km.
- [ ] Run offline non-productive high-boundary retrievals on `20251107sapm`; compare 0–6 km, uncertainty and support against frozen baseline before any promotion.
- [ ] Require at least one additional real SPU regime with materially different high-altitude/temporal behavior before choosing thresholds or aggregation width.
- [ ] Only then consider a productive long-mean/high-column input; preserve native block/native-resolution state.

P5.4 acceptance: added averaging/aggregation/boundary denoising must increase usable information without materially biasing lower column, hiding temporal nonstationarity, concealing contamination or disguising resolution loss.

## P5.5 — robust multi-reference ensemble — PENDING

- [ ] Require individually QA-passing local reference windows.
- [ ] Define separation/correlation rules.
- [ ] Keep each KFS member tied to its exact local boundary.
- [ ] Combine only members covering each altitude bin.
- [ ] Define explicit quality/uncertainty weighting.
- [ ] Add between-reference spread/reference-choice sensitivity uncertainty.
- [ ] Expose effective member count separately from effective block count.

## P5.6 — cascaded backward retrieval — DEFERRED

Consider only if backbone + ensemble leave a meaningful coverage gap. Any future cascade must inherit upper-solution boundary/uncertainty, reject inconsistent handoffs and never bridge unsupported gaps.

## P5.7 — overlap merge — DEFERRED

Implement only if multiple accepted solutions actually require stitching; weights/continuity/uncertainty behavior must be tested, not cosmetic.

## P5.8 — molecular model / uncertainty consistency — PARALLEL

- [x] Confirm current molecular definitions: `alpha_mol` is total Rayleigh volume scattering/extinction; `beta_mol` is angular 180° backscatter with wavelength-dependent depolarization/phase function.
- [x] Under those exact coded definitions, model-implied `alpha_mol/beta_mol` is ~8.503663 sr at 355 nm and ~8.496626 sr at 532 nm, versus productive `8π/3 = 8.377580 sr` (+1.505%/+1.421%).
- [x] Synthetic forward/inverse sensitivity test confirms a signal generated with model-implied molecular LR is recovered more faithfully when inversion uses the matching semantics at both wavelengths; productive code is unchanged (`f9490db...`).
- [ ] Resolve total-Rayleigh vs Cabannes/effective detected molecular component for actual receiver/filter semantics before changing `S_m`.
- [ ] Quantify representative real-SPU retrieval sensitivity under the physically appropriate receiver semantics.
- [ ] Version retrieval method if productive molecular semantics change.
- [ ] Quantify gluing fit slope/intercept uncertainty/covariance and include only if material.

Evidence: `docs/regression_baselines/molecular_lidar_ratio_semantics_rnd.json`.

## P5.9 — Level 2 schema/provenance/QA

- [x] Schema 2: inversion support/top/bottom/effective block count.
- [x] Schema 3: auditable Rayleigh candidate catalogue; method v4 remains distinct from schema identity.
- [x] Generic schema-v3 validation checks catalogue invariants rather than trusting version attribute.
- [x] `docs/level2_schema.md` documents schema 3 / method 4 and diagnostic-only SNR.
- [ ] QA plots: show effective optical retrieval top/support explicitly and prevent finite high-altitude scattering ratio from visually implying aerosol retrieval.
- [ ] QA molecular-fit plot: expose selected/accepted candidate locations without turning candidate density into validation.
- [ ] Add temporal/aggregation/fitted-boundary/ensemble panels only when corresponding diagnostics become productive or when clearly labeled R&D output is deliberately requested.

## P5.10 — validation matrix

### Synthetic established
- [x] Pure molecular -> near-zero aerosol; controlled aerosol 355/532 recovery; vertical-grid convergence.
- [x] Missing uncertainty/internal gaps fail support explicitly.
- [x] QA-first candidate catalogue retains accepted/rejected windows.
- [x] Temporal heterogeneity/candidate persistence distinguish persistent vs transient high-range information.
- [x] Vertical aggregation preserves coarse-grid KFS truth while exposing resolution loss.
- [x] Clean fitted-window boundary reduces exact-bin noise; contaminated-window counterexample exposes bias.
- [x] Independent and explicit-correlated fitted-boundary Monte Carlo match corresponding analytic uncertainty models without double counting.
- [x] Molecular-lidar-ratio semantic mismatch is exercised in an independent forward-model sensitivity test.
- [ ] Broader covariance/noise/contamination/boundary-placement matrix.
- [ ] Evidence-backed temporal reject/split rule.
- [ ] Multiple high references -> ensemble stability/reference-sensitivity uncertainty.
- [ ] Future high-column upper-tail noise -> supported top falls gracefully.
- [ ] Future backbone -> no material lower-column truth bias.

### Real SPU
- [x] Current method-v4 `20251107sapm` behavior and schema-v3 catalogue frozen/checked.
- [x] Temporal support/contribution/persistence evidence frozen.
- [x] Vertical correlation/SNR-gain brackets frozen.
- [x] Window calibration uncertainty and exact-vs-fit boundary instability frozen.
- [ ] Populate high-column evidence vector from real candidate catalogue.
- [ ] Offline high-boundary fitted-window/aggregation experiment against frozen lower-column baseline.
- [ ] Evaluate whether ~15 km is defensible in representative cases; attempt ~20 km only when trustworthy boundary/support exists.
- [ ] Add clear-night, high-aerosol, cloud, weak-signal, temporally changing and different PC/AN-dominance cases.
- [ ] Compare representative cases with LPP/SCC/ELDA under matched assumptions.

P5 success criterion: **maximize defensible inversion-supported coverage while exposing where/why support ends, with temporal representativeness, uncertainty honesty, declared resolution and exact boundary provenance. Success is not “reaches 20 km.”**

## P6 — release / publication readiness

- [ ] Select/record software license; align citation/package metadata.
- [ ] Align `CITATION.cff` with actual release version/date/DOI.
- [ ] Summarize scientific method/schema changes in release notes.
- [ ] Freeze reproducible scientific environment/constraints.
- [ ] Build/test sdist + wheel in CI and clean installation from artifacts.
- [ ] Core-science coverage report + deliberate warning policy.
- [ ] End-to-end Level 0 -> Level 2 from release artifact/environment.
- [ ] Representative NetCDF metadata review.
- [ ] Freeze machine-readable synthetic acceptance + observational regression summaries.
- [ ] Reconcile branch history with `main` before release/merge.

## Immediate next gate

1. Populate `HighColumnEvidence` from the existing `20251107sapm` schema-v3 catalogue and temporal diagnostics **offline**, keeping every evidence dimension separate.
2. Extend fitted-boundary synthetics across noise amplitudes, asymmetric contamination and reference-placement offsets; zero anomaly flag must remain a known non-certification case.
3. Build the first offline, non-productive higher-boundary comparison for `20251107sapm` only after the evidence vector can explain each candidate; compare 0–6 km, uncertainty and supported top against frozen method-v3/v4 baselines.
4. Implement the QA-plot support/top annotation so scattering-ratio diagnostics above the optical-retrieval top are visually explicit.
5. Keep 60–120 m aggregation as parallel R&D, not a selected productive width.
6. Seek at least one additional real SPU regime before promoting thresholds/boundary estimator/aggregation width.
7. Resolve receiver molecular semantics and gluing fit-parameter uncertainty in parallel.
8. Only after those gates consider a productive high-column backbone, then multi-reference ensemble; cascade remains a later decision gate.
