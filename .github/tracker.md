# MILGRAU scientific engineering tracker

Branch: `new-architecture`
Tracker snapshot: 2026-09-17

This is the active scientific/engineering source of truth. Detailed evidence remains in Git, executable tests and `docs/regression_baselines/`.

The tracker keeps a compact status summary **and** actionable checklists. A checked item means the implementation/evidence actually exists; promotion by project decision does not retroactively close remaining scientific validation work.

---

# 0. Non-negotiable scientific rules

* Separate analytical/synthetic truth, real-data regression, external-chain comparison and instrument characterization.
* Real measurements are behavior/regression evidence, not exact aerosol optical truth.
* Missing uncertainty is never zero uncertainty; unsupported bins stay unsupported.
* Backward KFS cannot extend above its accepted boundary or jump missing/masked/instrument-invalid internal gaps.
* Finite signed background-subtracted RCS is measurement information and may be averaged at a declared coarser resolution; the aggregated KFS cell itself must be finite and positive.
* High altitude is an objective, not permission to weaken QA.
* Long averaging must expose temporal contribution; vertical aggregation must expose resolution loss.
* A fitted/window or vertically aggregated boundary is a retrieval assumption and must remain explicit.
* Clean center bin != clean molecular window; narrow Monte Carlo spread != absence of bias.
* Rayleigh-window shape compatibility is not proof that `beta_aer(ref)=0`.
* Thresholds are not chosen to reach a desired altitude.
* Productive semantic changes require explicit method/schema/provenance/baseline versioning.

---

# 1. Productive identity — method v5

The `new-architecture` branch now treats method v5 as the only productive/default Level-2 path. Backward compatibility with productive method v4 is no longer a requirement for this branch.

* Level-2 schema **v4**.
* Retrieval method **v5**.
* 20-minute temporal blocks.
* Native 7.5 m lower-column representation below 6 km, then progressive vertical aggregation.
* Productive progressive-grid schedule: 7.5 m below 6 km; 15 m at 6–10 km; 30 m at 10–15 km; 60 m at 15–25 km; <=100 m requested above 25 km, realized as 97.5 m on the current 7.5 m SPU grid.
* Native-grid Rayleigh QA remains separate from the aggregated numerical KFS boundary.
* Reference policy: try declared tiers **10 -> 9 -> 8 -> 6 km**; use the first tier containing a Rayleigh-accepted, path-admissible cell; within that tier minimize `relative_slope + relative_variance`; exact ties prefer lower altitude / lower cell index.
* Tier/fallback altitude is a search/support policy, not a molecular-purity claim.
* Nominal boundary remains `f = beta_aer(ref)/beta_mol(ref) = 0` as a declared assumption.
* Productive boundary-sensitivity scenarios are `f = 0`, `0.02`, `0.05`.
* Random uncertainty propagates native-signal perturbations through progressive aggregation, Rayleigh QA, path admissibility, tier/reference selection and KFS.
* `beta_ref_relative_std = 0` by default unless independent evidence later justifies a separate random boundary-estimator term.
* `mc_valid_fraction(z)` is a diagnostic/support quantity; no hard MC-survival cutoff is imposed.
* Aerosol extinction remains conditional on assumed/climatological aerosol lidar ratio.
* Productive optical support is inversion support, not generic finite-value support.

Still not established as productive physical claims: hard SNR/cloud/temporal gates, overlap cutoff, physical PC saturation threshold, universal instrument-wide glued/PC high-column behavior, cascade/stitching and Raman correction.

---

# 2. Baselines, evidence and current implementation gate

## 2.1 Historical method-v4 regression baseline

`20251107sapm` remains useful as historical/reference evidence, not as the productive identity of this branch.

* L2 SHA `bba3b454387ef34c9b4bee3b0ab69e286e611cc44f9a4a6358b779f6d9f236d8`
* source L1 SHA `7b49c9deb93e89b103dba95455d854fcebb97b81292a18269edd1bea7a5ee342`
* source revision `fa1cae20ec1ac28bf2955deb5a93f9a3ed56e663`
* schema/method 3/4
* 25,340 candidate slots; 16,331 accepted; 10 selected
* temporal weights 23 / 39 / 39 / 40 / 26

Historical, active and campaign products are not claimed input-equivalent unless their input hashes match.

## 2.2 Productive method-v5 implementation gate

* Last green pre-tracker-refresh implementation commit: `79262e397739310a66e0e8d6dc11c00e30e3f9a3`.
* CI run `35282690421`: Ruff + pytest successful on Ubuntu/Windows and Python 3.12/3.14.
* Ubuntu Python 3.12 suite: **541 passed**.
* Productive `config.yaml` declares schema-4/method-v5 behavior and the 10/9/8/6 km tier policy.
* Productive LEBEAR writes schema-4 method-v5 NetCDF and validates it with `milgrau.level2.schema_v5`.
* No frozen real-data method-v5 Level-2 baseline has yet been accepted from the newly supplied Level-1 validation files.

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
* `p5_4_method_v5_progressive_grid_campaign_20260917.json`
* `p5_4_method_v5_selector_synthetic_20260917.json`
* `p3_level2_metadata_audit_20260917.json`

Raman feasibility evidence remains preserved but is **deferred** as a future independent validation path rather than a current method-v5 prerequisite.

---

# 3. Status overview

| Priority | Status | Meaning |
| --- | --- | --- |
| P0–P2 | COMPLETE | architecture/support/engineering guardrails |
| P3 | **CORE FAIR COMPLETE** | MIT selected; recipe/provenance strong; schema-4/method-v5 traceability active; release polish moves to P6 |
| P4 | PARALLEL EVIDENCE | overlap, detector and gluing characterization |
| P5.0–P5.2 | FROZEN FOUNDATION | baseline, support semantics, QA-first catalogue |
| P5.3 | COMPLETE FOR V5 POLICY | existing Rayleigh diagnostics retained; no new purity score justified |
| P5.4 | **PRODUCTIVE V5 IMPLEMENTED / REAL-L1 VALIDATION OPEN** | progressive grid, tiered selector, selection-aware MC, schema 4 and productive LEBEAR are implemented |
| P5.5 | PENDING / CONDITIONAL | ensemble only if remaining ambiguity exceeds explicit `f` sensitivity |
| P5.6–P5.7 | DEFERRED | cascade / stitching |
| P5.8 | PARALLEL R&D | molecular semantics / uncertainty consistency |
| P5.9 | COMPLETE | support-aware QA implemented and real-image reviewed |
| P5.10 | **IN PROGRESS** | real Level-1 / heterogeneous validation matrix |
| Raman | DEFERRED | future independent validation / nighttime retrieval path |
| P6 | PENDING | reproducible release/publication and branch policy |

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

The previously supplied successful campaign retrieval blocks used post-QA analog fallback. Glued/PC-dominant high-column behavior therefore remains an evidence gap for instrument-wide generalization.

## 4.2 Boundary movement dominates the new sensitivity

Native-grid boundary-only tests near 10 km found:

* 159/161 input-valid runs with usable exact high-reference bins;
* 111/159 all-300 MC complete;
* median 0.6–6 km relative L2 change vs historical method v4: **13.1%**;
* p95: **47.2%**;
* median absolute integrated-column difference: **16.5%**; p95 **61.2%**.

Existing slope, variance, SNR, cost, calibration factor and path completeness do not consistently predict this sensitivity. There is no demonstrated elastic score that validates molecular purity at a higher boundary.

## 4.3 Residual aerosol is a demonstrated boundary mechanism

Controlled truth shows that Rayleigh QA can pass while `beta_aer(ref)>0`; true total-backscatter boundary recovers truth; forcing `f=0` creates increasing lower-column bias; LR mismatch may amplify or partially compensate it.

Across 150 real controls at the historical reference, declared scenarios `f=0.02` and `f=0.05` changed the 0.6–6 km retrieval by median ~**5.3%** and ~**13.9%**. These remain sensitivity scenarios, not inferred aerosol fractions.

## 4.4 Progressive-grid campaign evidence

On 161 available 20-min block×wavelength states, continuous positive aggregated path tops from the established lower column were:

* minimum 6.77 km; p10 9.80 km; p25 12.89 km;
* median **14.57 km**; p75 17.07 km; p90 21.71 km; maximum 24.99 km;
* 143/161 reach >=10 km, 133/161 >=12 km, 69/161 >=15 km, 18/161 >=20 km.

A mandatory 20–25 km boundary was rejected: candidate existence there was much more common than a continuous admissible path. The productive v5 implementation therefore uses explicit tier fallback rather than requiring one fixed high altitude.

Longer temporal averaging is not presently justified as the fix: 20/40/60 min probes gave similar median continuous tops. The productive v5 baseline remains 20 min.

---

# 5. P5.4 — actionable method-v5 checklist

Full scientific-decision history: `docs/high_column_method_v5_decisions.md`.

## 5.1 Scientific product definition

* [x] Method v5 is an **elastic retrieval conditional on declared boundary, lidar-ratio and effective-resolution assumptions**.
* [x] Nominal boundary remains `beta_aer(ref)=0` / `f=0` for comparability.
* [x] `f=0` is an assumption, not an observation of molecular purity.
* [x] Residual `f` is treated as a separate systematic sensitivity dimension.
* [x] Backscatter and extinction are produced; extinction remains conditional on assumed/climatological LR.

## 5.2 Boundary estimator

* [x] High boundary is represented by the same progressive-grid cell used by KFS.
* [x] Effective cell width and source-bin count remain explicit.
* [x] Native 1-km Rayleigh QA remains separate from the aggregated-cell numerical boundary.
* [x] Broad-contamination bias remains a systematic failure mode; averaging is not purity validation.

## 5.3 Vertical representation

* [x] Preserve native 7.5 m representation below 6 km.
* [x] Productive grid: 15 m at 6–10 km, 30 m at 10–15 km, 60 m at 15–25 km, <=100 m above 25 km.
* [x] Current 7.5 m SPU grid realizes the <=100 m cap as 97.5 m.
* [x] No interpolation, padding or source-bin reuse.
* [x] Missing/masked/instrument-invalid source sample invalidates its progressive cell.
* [x] Finite signed RCS may be averaged; aggregated KFS cell must be finite and positive.
* [x] Effective resolution/source count are retained and written to schema-4 output.
* [ ] Complete a direct controlled-truth **native-vs-progressive retrieval representation-error** study, not only path feasibility.

Implementation: `milgrau/level2/adaptive_grid.py`; productive assembly: `milgrau/level2/method_v5_product.py`.

## 5.4 MC / support semantics

* [x] V5 does not define physical support as `300/300` MC survival.
* [x] Nominal deterministic path support is separate from random-MC robustness.
* [x] Nested boundary-scenario MC reports altitude-resolved valid fraction per `f` scenario.
* [x] Selection-aware MC reruns progressive aggregation + Rayleigh QA + selector inside each noisy realization.
* [x] Fixed-reference versus selection-aware interval-coverage experiments were run.
* [x] `mc_valid_fraction(z)` is persisted in the schema-4 NetCDF and explicitly declared diagnostic-only.
* [x] No hard valid-fraction cutoff is imposed; existing coverage evidence does not justify using survival fraction as uncertainty calibration.
* [ ] Decide whether an explicit **valid-count** variable should also be persisted in the final schema in addition to fraction + iteration count.

Implementation: `milgrau/level2/selection_aware_mc_rnd.py`, `milgrau/level2/method_v5_product.py`, `milgrau/level2/schema_v5.py`.

## 5.5 Boundary `f` uncertainty semantics

* [x] Outer caller-declared `f` scenarios = epistemic/systematic boundary sensitivity.
* [x] Inner MC = random signal + declared LR uncertainty + reference re-selection.
* [x] Same perturbed measurement/selection draws are paired across `f` scenarios.
* [x] No arbitrary probability distribution assigned to `f`.
* [x] Legacy extra `beta_ref_relative_std` is zero by default because independent meaning is not established once signal noise and selection are propagated.
* [ ] Only marginalize `f` into a total probabilistic uncertainty if independent evidence later supplies a defensible distribution.

## 5.6 Temporal semantics

* [x] Keep 20-min blocks for productive v5.
* [x] 40/60-min observational probes completed; median continuous top barely changes.
* [x] Do not trade temporal resolution for altitude without a demonstrated benefit.
* [ ] Revisit adaptive temporal averaging only if real-v5 validation still shows a scientifically justified need.

## 5.7 Admissible path

* [x] Local Rayleigh QA alone is insufficient for path acceptance.
* [x] Continuous nominal progressive-cell path is diagnosed separately.
* [x] Missing/masked/instrument-invalid gaps are not bridged.
* [x] Cloud/layer state remains diagnostic unless a validated veto is established.
* [ ] Add characterized physical saturation/overlap instrument masks when P4 evidence exists.

## 5.8 Lower-column preservation and truth validation

* [x] Historical real-data lower-column sensitivity to boundary movement is quantified.
* [x] Residual-aerosol boundary bias exists in controlled truth.
* [x] Synthetic selector studies retain profile-shape relative-L2 and integrated-column error metrics.
* [x] Keep achieved top altitude out of the acceptance metric.
* [ ] Complete molecular-only progressive-grid truth.
* [ ] Complete direct native-vs-progressive retrieval truth comparison.
* [ ] Define synthetic-truth acceptance in terms of deterministic bias + interval coverage rather than top altitude.
* [ ] Run the real-Level1 v5 harness and define uncertainty-normalized real-data compatibility diagnostics.
* [ ] Add an integrated-column difference to the real-Level1 validation harness; it currently records lower-column profile relative-L2 but not the column metric.

## 5.9 Rayleigh/reference selection

* [x] Rayleigh QA for v5 candidate cells remains on the native grid using a physical 1-km window.
* [x] Progressive cell is the numerical KFS boundary representation.
* [x] “Select the highest accepted candidate” is rejected by synthetic truth.
* [x] Deterministic selector exists and applies QA/path first, then declared support tiers.
* [x] Productive tier policy is **10 -> 9 -> 8 -> 6 km**; tier fallback is explicit metadata.
* [x] Within the winning tier, minimize existing Rayleigh cost `relative_slope + relative_variance`; exact ties prefer lower altitude / lower cell index.
* [x] Explicit cloud and stratospheric-aerosol synthetic cases exist.
* [x] 355 and 532 nm selector behavior has been challenged separately under normalized synthetic noise.
* [x] No new composite molecular-purity score is introduced.
* [ ] Validate tier/fallback/reference-distribution behavior with real Level-1 355/532 noise and covariance.

Implementation: `milgrau/level2/high_column_selector.py`, `milgrau/level2/method_v5_rnd.py`.

## 5.10 Method/schema promotion

* [x] End-to-end selected-reference v5 execution exists.
* [x] Productive retrieval method bumped to **v5**.
* [x] Productive schema bumped to **v4**.
* [x] Productive `milgrau-lebear` uses method v5 by default.
* [x] Schema-4 validator enforces dimensions/support/version identity.
* [x] NetCDF records progressive resolution/source count, nominal reference/tier/fallback, selector diagnostics, selected-reference MC distribution, `f` scenarios, supported top, selection-success fraction and altitude-resolved `mc_valid_fraction`.
* [x] Project decision removed the requirement to maintain productive v4 compatibility on this branch; historical v4 outputs remain evidence only.
* [ ] Run productive v5 on the supplied real Level-1 validation files and freeze the first accepted method-v5 real-data baseline.
* [ ] Run productive v5 across the broader heterogeneous campaign once the real-Level1 gate is stable.

---

# 6. Minimum method-v5 validation matrix

* [ ] molecular-only progressive-grid synthetic truth;
* [x] residual-aerosol boundary synthetic mechanism;
* [x] weak-signal / many-isolated-bin path case;
* [x] localized and broad contamination counterexamples;
* [x] explicit high-cloud synthetic/real challenge exists;
* [x] explicit stratospheric-aerosol truth in candidate region;
* [x] 355 and 532 synthetic selector truth separately;
* [x] heterogeneous 11-product SPU path/resolution feasibility;
* [x] 20/40/60-min temporal path comparison;
* [x] valid-MC fraction versus confidence-interval coverage studied;
* [x] reference-selection uncertainty identified and propagated inside selection-aware MC;
* [ ] native vs progressive **retrieval** truth comparison;
* [ ] productive v5 retrieval on supplied real Level-1 files;
* [ ] real-Level1 lower-column compatibility independently of top altitude;
* [ ] real 355/532 high-altitude noise/covariance characterization and comparison with synthetic assumptions;
* [ ] executable productive v5 across the broader heterogeneous campaign;
* [ ] at least one glued/PC-dominant successful regime before instrument-wide generalization.

---

# 7. Raman — deferred, not deleted

Raman channel identity/feasibility evidence is preserved, but quantitative Raman retrieval is not a blocker for the first elastic v5 product.

* [x] Current acquisition contains 387/408/530 channels.
* [x] SCC mapping for 387/530 current nighttime configuration is known.
* [ ] Current receiver/filter spectral response remains unresolved.
* [ ] Quantitative Raman retrieval remains future work.

Future role: independent boundary validation and/or independent nighttime extinction retrieval after elastic v5 is stable on real Level-1 input.

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
* [x] Scientific traceability regression contract aligned to schema **v4** / method **v5**.

## Deferred curation / release polish

* [ ] Final creator/contact/title/summary/keywords policy for release artifacts.
* [ ] Formal CF compliance review before claiming a CF convention.
* [ ] Bit-for-bit dependency/environment artifact under P6.

Current station labels intentionally remain simple: lidar-ratio table `climatology`; channel correction set `experimental`. Detailed historical ancestry is not an active blocker.

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
* [x] Schema-4 method-v5 output persists gluing start/split/stop and fit-quality diagnostics.
* [ ] Quantify fitted slope/intercept uncertainty and covariance.
* [ ] Test a valid glued/PC-dominant high-column regime.

P4 evidence improves instrument-wide generality but does not invalidate the current conditional method-v5 semantics.

---

# 10. Closed engineering / QA gates

* [x] Exact-reference failures are block-local rather than wavelength-fatal (`tests/test_level2_kfs_block_failure.py`; CI `35248016374`).
* [x] Support-aware SR/KFS QA implemented and heterogeneous real-image reviewed (CI `35241784472`).
* [x] Station/calibration machine-readable provenance regression tested; CI `35257631767` passed Ruff and cross-platform pytest.
* [x] Tiered selector + selection-aware MC implementation exists and is covered by executable tests.
* [x] Productive schema-4/method-v5 LEBEAR path implemented and schema validated.
* [x] Latest productive method-v5 implementation CI `35282690421` passed Ruff and pytest on Ubuntu/Windows, Python 3.12/3.14; Ubuntu 3.12 reported **541 passed**.

---

# 11. Engineering cleanup created by v5 promotion

Promotion made some old R&D/compatibility labels stale even though they do not change runtime behavior.

* [ ] Update `docs/high_column_method_v5_decisions.md` so it no longer says productive Level 2 remains schema v3 / method v4.
* [ ] Remove or rewrite stale docstrings such as “not wired into productive method v4” in modules now used by productive v5.
* [ ] Decide whether to rename productive modules/files still carrying `_rnd` suffixes (`method_v5_rnd`, `selection_aware_mc_rnd`, etc.) or keep the names with an explicit historical note.
* [ ] Remove/archive legacy v4-only orchestration helpers once no remaining tests/tools depend on them; do not retain compatibility code solely for compatibility.
* [ ] Update any schema-v3/method-v4 wording remaining in user-facing docs/examples.

---

# 12. P5.5–P5.8 / P6 deferred checklist

## P5.5 Ensemble

* [ ] Revisit only if one validated v5 boundary still leaves material reference ambiguity beyond explicit boundary-scenario sensitivity.

## P5.6–P5.7 Cascade / stitching

* [ ] Keep deferred until the single-path productive v5 method is scientifically characterized on real Level-1 input.

## P5.8 Molecular semantics

* [ ] Keep current molecular lidar-ratio semantics unless receiver/filter physics justify a deliberate versioned change.

## P6 Release

* [ ] Reproducible dependency/environment artifact.
* [ ] Warning/coverage/build review.
* [ ] Release metadata polish.
* [ ] Reconcile `new-architecture` with release branch/main.
* [ ] Required checks / branch protection policy (`new-architecture` is currently not protected).
* [ ] Immutable scientific tag for the released schema-4/method-v5 pair.

---

# 13. Immediate next scientific gates

The productive architecture decision is closed; the next work is validation/cleanup, not another selector redesign.

1. [ ] Run the supplied real Level-1 files through productive method v5 and the real-Level1 validation harness.
2. [ ] Characterize actual 355/532 high-altitude noise/covariance from those Level-1 files and compare it with the synthetic noise model.
3. [ ] Complete molecular-only progressive-grid truth and direct native-vs-progressive retrieval representation-error tests.
4. [ ] Add integrated-column comparison to the real-Level1 validation harness and define uncertainty-normalized lower-column regression diagnostics.
5. [ ] Freeze the first accepted real method-v5 Level-2 baseline with exact Level-1 hash, code identity, schema/method identity and configuration provenance.
6. [ ] Extend the executable method-v5 run to the broader heterogeneous campaign.
7. [ ] Obtain at least one valid glued/PC-dominant high-column regime before claiming instrument-wide detector-mode generality.
8. [ ] Clean stale R&D/v4 naming and documentation now that v5 is productive.

No target altitude, MC fraction, SNR value or percent lower-column agreement is preselected as an acceptance threshold.

---

# 14. Current handoff

MILGRAU `new-architecture` productive Level 2 is now **schema v4 / method v5**. The implementation gate is green: CI `35282690421` passed Ruff plus the full cross-platform pytest matrix, with 541 tests passing on Ubuntu/Python 3.12.

The productive elastic method is conditional: nominal `f=0`, explicit `f=0.02/0.05` systematic scenarios, selection-aware random MC, native Rayleigh QA, tiered 10/9/8/6 km reference fallback, progressive KFS grid preserving <6 km at 7.5 m and coarsening to <=100 m aloft, and a 20-min temporal baseline. Reference altitude and achieved top remain measurement-dependent and are not success metrics by themselves.

The main scientific work still open is **real Level-1 validation**, direct progressive-representation truth testing, real detector/noise characterization, real lower-column regression diagnostics and a glued/PC-dominant validation case. The main engineering work still open is cleanup of stale R&D/v4 naming/documentation and eventual P6 release hardening.
