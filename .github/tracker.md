# MILGRAU scientific engineering tracker

Branch: `new-architecture`
Tracker snapshot: 2026-09-17

This is the active scientific/engineering source of truth. Detailed evidence remains in Git, executable tests and `docs/regression_baselines/`.

The tracker keeps a compact status summary **and** actionable checklists. The first tracker version contained detailed checklists; commit `ef342d3b972420d84ed715d716a509700f93b81d` consolidated ~858 lines into ~296 and removed much of that task-level detail. The checklists below restore the useful structure without restoring obsolete work.

---

# 0. Non-negotiable scientific rules

* Separate analytical/synthetic truth, real-data regression, external-chain comparison and instrument characterization.
* Real measurements are behavior/regression evidence, not exact aerosol optical truth.
* Missing uncertainty is never zero uncertainty; unsupported bins stay unsupported.
* Backward KFS cannot extend above its accepted boundary or jump invalid internal gaps.
* High altitude is an objective, not permission to weaken QA.
* Long averaging must expose temporal contribution; vertical aggregation must expose resolution loss.
* A fitted/window or vertically aggregated boundary is a new retrieval assumption.
* Clean center bin != clean molecular window; narrow Monte Carlo spread != absence of bias.
* Rayleigh-window shape compatibility is not proof that `beta_aer(ref)=0`.
* Thresholds are not chosen to reach a desired altitude.
* Productive semantic changes require explicit method/schema/provenance/baseline versioning.

---

# 1. Productive identity — unchanged

* Level-2 schema **v3**: auditable Rayleigh candidate catalogue.
* Retrieval method **v4**: QA-first Rayleigh selection + backward Klett–Fernald–Sasano.
* Productive boundary: exact measured RCS bin at selected reference altitude.
* Boundary assumption: `aerosol_ref_fraction = 0`, therefore `beta_total(ref)=beta_mol(ref)`.
* Candidate rank after minimum QA: `relative_slope + relative_variance`, lower grid index as deterministic tie-breaker.
* Rayleigh SNR is diagnostic-only.
* Aerosol extinction is conditional on assumed aerosol lidar ratio.
* Productive optical support is inversion support, not generic finite-value support.

Not productive yet: inferred non-zero boundary aerosol, fitted/window boundary, vertically aggregated high-column retrieval, adaptive high-column resolution, relaxed MC-support semantics, hard SNR/cloud/temporal gates, overlap cutoff, physical PC saturation threshold, ensemble, cascade/stitching and Raman correction.

---

# 2. Active baseline and evidence set

Primary baseline: `20251107sapm`.

Active identity:

* L2 SHA `bba3b454387ef34c9b4bee3b0ab69e286e611cc44f9a4a6358b779f6d9f236d8`
* source L1 SHA `7b49c9deb93e89b103dba95455d854fcebb97b81292a18269edd1bea7a5ee342`
* source revision `fa1cae20ec1ac28bf2955deb5a93f9a3ed56e663`
* schema/method 3/4
* 25,340 candidate slots; 16,331 accepted; 10 selected
* temporal weights 23 / 39 / 39 / 40 / 26

Historical evidence for the previous checksum is retained, but its original NetCDF is unavailable after workstation migration. Historical, active and campaign products are not claimed input-equivalent unless their input hashes match.

Key frozen evidence includes:

* `20251107sapm_active_baseline_p54_summary.json`
* `20251107sapm_first_high_boundary_experiment.json`
* `20251107sapm_path_failure_diagnostic.json`
* `20251107sapm_vertical_aggregation_support_edge.json`
* `p54_covariance_leaveout_synthetic.json`
* `p5_4_observational_campaign_20260917.json`
* `p5_4_campaign_aggregation_same_boundary_20260917.json`
* `p5_4_campaign_high_boundary_aggregation_20260917.json`
* `p5_4_campaign_native_grid_boundary_only_20260917.json`
* `p5_4_aggregation_effect_decomposition_20260917.json`
* `p5_4_boundary_existing_diagnostic_association_20260917.json`
* `p5_4_synthetic_residual_aerosol_boundary_sweep_20260917.json`
* `p5_4_synthetic_boundary_residual_noise_lr_20260917.json`
* `p5_4_campaign_boundary_fraction_sensitivity_20260917.json`
* `p3_level2_metadata_audit_20260917.json`

Raman feasibility evidence remains preserved but is **deferred** as a future independent validation path rather than a current method-v5 prerequisite.

---

# 3. Status overview

| Priority | Status | Meaning |
| --- | --- | --- |
| P0–P2 | COMPLETE | productive architecture/support/engineering guardrails |
| P3 | **CORE FAIR COMPLETE** | MIT selected; recipe/provenance strong; release polish moves to P6 |
| P4 | PARALLEL EVIDENCE | overlap, detector and gluing characterization |
| P5.0–P5.2 | FROZEN FOUNDATION | baseline, support semantics, QA-first catalogue |
| P5.3 | MOSTLY COMPLETE | candidate diagnostics interpreted; no new purity score justified |
| P5.4 | **ACTIVE PRIMARY R&D** | define and validate elastic high-column method-v5 semantics |
| P5.5 | PENDING | ensemble only if method-v5 evidence establishes need |
| P5.6–P5.7 | DEFERRED | cascade / stitching |
| P5.8 | PARALLEL R&D | molecular semantics / uncertainty consistency |
| P5.9 | COMPLETE | support-aware QA implemented and real-image reviewed |
| P5.10 | IN PROGRESS | heterogeneous validation matrix |
| Raman | DEFERRED | future independent validation / nighttime retrieval path |
| P6 | PENDING | reproducible release/publication |

---

# 4. P5.4 — established scientific findings

## 4.1 Candidate/path/contamination evidence

Across baseline, synthetics and 11 heterogeneous/seasonal real L2 products:

* candidate existence or persistence does not imply a usable KFS boundary;
* accepted candidates can sit above internal invalid backward paths;
* native high paths can be nominally complete yet MC-fragile because many weak isolated bins accumulate failure probability;
* `20250509sant` has a strong layer near 14.36–14.37 km and accepted/persistent candidate islands reappear above it;
* fitted/window boundaries can denoise yet remain biased under broad contamination;
* leave-part-out and placement sensitivity expose some localized contamination but are not molecular-purity certificates;
* no SNR, cloud, persistence, continuity, leave-out or placement threshold is authorized.

All successful supplied campaign retrieval blocks use post-QA **analog fallback**, so glued/PC high-column behavior remains an evidence gap, not a blocker for defining elastic method-v5 semantics.

## 4.2 Covariance and aggregation

From `20250629sant` Level 1, block-demeaned uncertainty-normalized residuals over 5–20 km give approximately:

* lag-1 / 7.5 m correlation **0.10–0.13**;
* 60 m / 8-bin SNR gain **2.47–2.52**;
* 120 m / 16-bin SNR gain **3.31–3.40**.

At the existing productive boundary:

* 60 m aggregation: median 0.6–6 km relative L2 change ~**5%**, p95 ~**20%**;
* 120 m aggregation: median ~**6%**, p95 ~**20%**;
* deterministic coarse representation + boundary-cell shift contributes median ~**4.6%** / **5.6%**, with p95 ~21–22%.

Aggregation improves high-path support but is not lower-column neutral. No width is yet productive.

## 4.3 Moving the boundary is the larger sensitivity

Native-grid boundary-only tests near 10 km isolate boundary movement from aggregation:

* 159/161 input-valid runs have usable exact high-reference bins;
* 111/159 are all-300 MC complete;
* median 0.6–6 km relative L2 change vs method v4: **13.1%**;
* p95: **47.2%**;
* median absolute integrated-column difference: **16.5%**; p95 **61.2%**.

Slope, variance, SNR, diagnostic cost, calibration factor and path completeness show weak/inconsistent within-case association with this lower-column sensitivity. There is no demonstrated elastic score that validates a higher boundary.

## 4.4 Residual aerosol is a demonstrated boundary mechanism

Controlled synthetic truth shows:

* Rayleigh slope/variance/valid-fraction QA can pass while `beta_aer(ref)>0`;
* using the true total-backscatter boundary recovers lower-column truth;
* forcing `beta_aer(ref)=0` creates monotonic lower-column bias as residual aerosol grows;
* signal noise adds dispersion but does not remove systematic boundary bias;
* lidar-ratio mismatch can amplify or partially cancel boundary bias.

Across 150 successful campaign controls at the existing reference, declared sensitivity scenarios `f=0.02` and `f=0.05` change the 0.6–6 km retrieval by median ~**5.3%** and ~**13.9%** respectively. These are sensitivity scenarios, not inferred aerosol fractions.

**Current conclusion:** the high-column problem is principally a boundary-model + support-statistics problem, not a candidate-score problem.

---

# 5. P5.4 — restored actionable checklist for method v5

Full decision contract: `docs/high_column_method_v5_decisions.md`.

## 5.1 Scientific product definition

* [ ] Decide that method v5 is explicitly an **elastic retrieval conditional on declared boundary and lidar-ratio assumptions**, rather than an independently observed aerosol truth product.
* [ ] Decide whether the central upper boundary remains `beta_aer(ref)=0` with a systematic residual-aerosol sensitivity envelope, or whether a different bounded boundary prior is adopted.
* [ ] Decide whether the high-column extension applies to aerosol backscatter only or to backscatter + extinction under the same assumed/climatological lidar ratio.

Recommended minimal path: keep a central molecular-boundary solution for comparability, but publish boundary sensitivity separately and do not describe molecular purity as independently verified.

## 5.2 Boundary estimator

* [ ] Choose exact native-bin, explicit Rayleigh-window estimator, or another synthetically validated estimator.
* [ ] If a window estimator is chosen, propagate estimator uncertainty separately from signal noise.
* [ ] Keep broad-contamination bias as a systematic failure mode; denoising is not purity validation.

## 5.3 Vertical representation

* [ ] Choose native, fixed-coarse or adaptive-resolution high-column representation.
* [ ] If aggregation is used, declare effective resolution per altitude.
* [ ] Use an explicit covariance/dependence model for aggregated uncertainty.
* [ ] Do not interpolate through invalid internal gaps.
* [ ] Compare the chosen representation against native synthetic truth and the method-v4 lower column.

## 5.4 MC / support semantics

* [ ] Decide whether method v5 still requires 100% valid MC paths or uses a pre-declared valid-realization fraction.
* [ ] If a fraction is used, define it from statistical meaning before checking achieved altitude.
* [ ] Store/report valid-realization fraction so support confidence is auditable.
* [ ] Mark bins with insufficient realizations unsupported rather than filling them.

## 5.5 Temporal semantics

* [x] Block contribution and candidate persistence diagnostics exist.
* [ ] Decide whether method v5 remains block-native or introduces a long-mean high-column signal.
* [ ] If long averaging is used, preserve block contribution/persistence in the product.
* [ ] Do not let a long mean hide a transient layer/cloud event.

Recommended first implementation: remain block-native; postpone a long-mean backbone until the boundary/support decisions are already validated.

## 5.6 Admissible path

* [x] Internal invalid gaps cannot be bridged.
* [x] Local Rayleigh QA alone is insufficient for path acceptance.
* [ ] Define one deterministic path-admissibility contract before reference ranking.
* [ ] Keep cloud/layer state diagnostic unless a separate validated veto is established.
* [ ] Add characterized saturation/instrument masks when P4 evidence exists; do not fabricate them now.

## 5.7 Lower-column preservation and truth validation

* [x] Current real-data lower-column sensitivity is quantified.
* [x] Residual-aerosol boundary bias exists in controlled truth.
* [ ] Define synthetic-truth acceptance in terms of bias + uncertainty coverage.
* [ ] Define real-data regression compatibility using uncertainty-normalized differences, not an arbitrary percent selected after seeing results.
* [ ] Report both profile-shape and integrated-column differences.
* [ ] Keep achieved top altitude out of the acceptance metric.

## 5.8 Deterministic reference selection

* [ ] Only after sections 5.1–5.7 are fixed, define the high-reference selection rule.
* [ ] Keep candidate shape QA, path support, boundary sensitivity, temporal state and effective resolution as separate quantities unless controlled evidence justifies combining them.
* [ ] Do not introduce a new composite “purity score” from current evidence.

## 5.9 Method/schema promotion

* [ ] Implement experimental method-v5 path behind an explicit method switch or R&D entry point.
* [ ] Freeze method-v4 and method-v5 outputs on identical Level-1 inputs.
* [ ] Bump retrieval method to v5 only after validation gates pass.
* [ ] Decide whether schema v3 can represent boundary model, effective resolution and MC-support fraction; bump schema only if needed.
* [ ] Record boundary model, estimator, resolution, support fraction and sensitivity assumptions in NetCDF.

---

# 6. Minimum method-v5 validation matrix

* [ ] molecular-only synthetic truth;
* [x] residual-aerosol boundary synthetic truth;
* [x] weak-signal / many-isolated-bin path case;
* [x] localized and broad contamination counterexamples;
* [x] high-cloud/layer real case available (`20250509sant`);
* [x] heterogeneous 11-product SPU campaign available;
* [ ] chosen method-v5 algorithm run across the heterogeneous campaign;
* [ ] 355 and 532 evaluated separately;
* [ ] chosen high-column resolution compared against native truth/regression;
* [ ] lower-column compatibility evaluated independently of top altitude;
* [ ] at least one glued/PC-dominant successful regime before instrument-wide generalization.

---

# 7. Raman — deferred, not deleted

Raman channel identity/feasibility evidence is preserved in the repository, but quantitative Raman retrieval is no longer a blocker for the first elastic method-v5 experiment.

* [x] Current acquisition contains 387/408/530 channels.
* [x] SCC mapping for 387/530 current nighttime configuration is known.
* [ ] Current receiver/filter spectral response remains unresolved.
* [ ] Quantitative Raman retrieval remains future work.

Future role: independent boundary validation and/or independent nighttime extinction retrieval after the elastic method-v5 design is stable.

---

# 8. P3 — FAIR / provenance checklist

## Core FAIR state

* [x] Code authors identified.
* [x] Software license selected: **MIT**.
* [x] Root `LICENSE` added.
* [x] `pyproject.toml` license metadata aligned.
* [x] `CITATION.cff` license metadata aligned.
* [x] Readable/self-describing NetCDF metadata policy adopted.
* [x] Representative Level-2 metadata audit completed.
* [x] Exact `config.yaml` and `station.yaml` recipes embedded in products.
* [x] Repository revision, source-code SHA and source-Level1 SHA preserved.
* [x] Machine-readable station profile/calibration lineage fixed for new products.
* [x] Scientific traceability aligned to schema v3 / method v4.

## Deferred curation / release polish

* [ ] Final creator/contact/title/summary/keywords policy for release artifacts.
* [ ] Formal CF compliance review before claiming a CF convention.
* [ ] Bit-for-bit dependency/environment artifact under P6.

Current station labels intentionally remain simple:

* lidar-ratio table: `climatology`;
* channel correction set: `experimental`.

Detailed historical ancestry is not an active blocker and can be curated later.

---

# 9. P4 — parallel instrument evidence checklist

## Overlap

* [x] Diagnostic-only geometry exists.
* [x] No productive overlap correction/cutoff is inferred.
* [ ] Telecover/alignment/overlap characterization.
* [ ] Wavelength-dependent geometry when experimentally available.

## Photon counting

* [x] Raw observed rate preserved.
* [x] Numerical dead-time clipping separated from physical saturation.
* [ ] Physical saturation characterization using AN/PC overlap and/or controlled attenuation.

## Gluing

* [x] Productive window selection and diagnostics explicit/versioned.
* [x] Measurement-noise propagation through fade weights implemented.
* [ ] Quantify fitted slope/intercept uncertainty and covariance.
* [ ] Test a valid glued/PC-dominant regime.

P4 evidence improves generality but does not block defining the first conditional elastic method-v5 R&D path.

---

# 10. Closed engineering / QA gates

* [x] Exact-reference failures are block-local rather than wavelength-fatal (`tests/test_level2_kfs_block_failure.py`; CI `35248016374`).
* [x] Support-aware SR/KFS QA implemented and heterogeneous real-image reviewed (CI `35241784472`).
* [x] Scientific traceability regression contract protects schema v3 / method v4 (CI `35240916329`).
* [x] Station/calibration machine-readable provenance regression tested; CI `35257631767` passed Ruff and cross-platform pytest.

---

# 11. P5.5–P5.8 / P6 deferred checklist

## P5.5 Ensemble

* [ ] Revisit only if one validated method-v5 boundary still leaves material reference ambiguity.

## P5.6–P5.7 Cascade / stitching

* [ ] Keep deferred until a single-path high-column method is scientifically characterized.

## P5.8 Molecular semantics

* [ ] Keep productive molecular lidar ratio `8*pi/3` unless receiver/filter semantics justify a deliberate change.

## P6 Release

* [ ] Reproducible dependency/environment artifact.
* [ ] Warning/coverage/build review.
* [ ] Release metadata polish.
* [ ] Reconcile `new-architecture` with release branch/main.
* [ ] Required checks / branch protection policy.
* [ ] Immutable scientific tag for the released method/schema pair.

---

# 12. Immediate next scientific decisions

The next work is **not another score sweep**. Before implementing productive high-column integration, close these decisions in order:

1. [ ] D1 — accept method v5 as a conditional elastic retrieval and choose the boundary model/sensitivity semantics.
2. [ ] D2 — choose the numerical boundary estimator.
3. [ ] D3 — choose native/fixed/adaptive high-column vertical representation.
4. [ ] D4 — choose MC-validity/support semantics.
5. [ ] D5 — keep block-native retrieval or introduce long averaging.
6. [ ] D7 — define synthetic-truth and real-regression promotion criteria.
7. [ ] Implement the experimental method-v5 path.
8. [ ] Run the full synthetic + heterogeneous SPU validation matrix.
9. [ ] Decide method-v5 promotion and schema impact.

Recommended starting position for the first experiment:

* conditional elastic product;
* central `beta_aer(ref)=0` retained for comparability, with explicit residual-boundary sensitivity reported separately;
* block-native temporal processing;
* no cloud veto and no interpolation;
* test fixed 60 m versus adaptive high-altitude aggregation rather than assuming a winner;
* replace all-300-valid semantics only if a pre-declared statistical support rule is justified;
* promote based on synthetic truth/uncertainty coverage and uncertainty-normalized lower-column compatibility, never on achieved altitude alone.

---

# 13. Current handoff

MILGRAU productive Level 2 remains **schema v3 / method v4**.

P5.4 has established that moving the upper boundary is a larger sensitivity than aggregation, current elastic candidate diagnostics do not independently validate `beta_aer(ref)=0`, and residual boundary aerosol can create systematic lower-column bias even when Rayleigh-window QA passes. This does **not** prevent an elastic-only method v5 if its scientific claim is explicitly conditional on declared boundary and lidar-ratio assumptions and its boundary sensitivity is carried honestly.

Raman is deferred as future independent validation. FAIR core work is sufficiently closed for the present retrieval-development phase: MIT is selected, metadata/provenance are strong, and climatology/calibration labels are intentionally simple. The active scientific problem is now the explicit method-v5 decision checklist in section 5 and `docs/high_column_method_v5_decisions.md`.
