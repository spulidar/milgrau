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
* Backward KFS cannot extend above its accepted boundary or jump missing/masked/instrument-invalid internal gaps.
* Finite signed background-subtracted RCS is measurement information and may be averaged at a declared coarser resolution; the aggregated KFS cell itself must be finite and positive.
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

Not productive yet: method-v5 progressive grid, aggregated-cell boundary, residual-`f` scenario ensemble, relaxed MC-support semantics, automatic high-reference selector, hard SNR/cloud/temporal gates, overlap cutoff, physical PC saturation threshold, cascade/stitching and Raman correction.

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
* `p5_4_method_v5_progressive_grid_campaign_20260917.json`
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
| P5.4 | **ACTIVE / V5 R&D EXECUTABLE** | progressive grid + nested boundary scenarios implemented; selector/coverage validation open |
| P5.5 | PENDING | ensemble only if method-v5 evidence establishes need beyond explicit `f` sensitivity |
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

## 4.2 Boundary movement dominates the new sensitivity

Native-grid boundary-only tests near 10 km:

* 159/161 input-valid runs have usable exact high-reference bins;
* 111/159 are all-300 MC complete;
* median 0.6–6 km relative L2 change vs method v4: **13.1%**;
* p95: **47.2%**;
* median absolute integrated-column difference: **16.5%**; p95 **61.2%**.

Existing slope, variance, SNR, cost, calibration factor and path completeness do not consistently predict this sensitivity. There is no demonstrated elastic score that validates a higher boundary.

## 4.3 Residual aerosol is a demonstrated boundary mechanism

Controlled truth shows that Rayleigh QA can pass while `beta_aer(ref)>0`; true total-backscatter boundary recovers truth; forcing `f=0` creates increasing lower-column bias; LR mismatch may amplify or partially compensate it.

Across 150 real controls at the existing reference, declared scenarios `f=0.02` and `f=0.05` change the 0.6–6 km retrieval by median ~**5.3%** and ~**13.9%**. These are sensitivity scenarios, not inferred aerosol fractions.

## 4.4 Progressive-grid campaign evidence

Current first-prototype grid:

| altitude | requested resolution | SPU realization |
| --- | ---: | ---: |
| <6 km | 7.5 m | 1 native bin |
| 6–10 km | 15 m | 2 bins |
| 10–15 km | 30 m | 4 bins |
| 15–25 km | 60 m | 8 bins |
| >=25 km | <=100 m | 97.5 m / 13 bins |

On 161 available 20-min block×wavelength states, continuous positive aggregated path tops from the established lower column are:

* minimum 6.77 km; p10 9.80 km; p25 12.89 km;
* median **14.57 km**; p75 17.07 km; p90 21.71 km; maximum 24.99 km;
* 143/161 reach >=10 km, 133/161 >=12 km, 69/161 >=15 km, 18/161 >=20 km.

A mandatory 20–25 km boundary is rejected: 117/161 cases contain a native Rayleigh-accepted candidate there, but only 18/161 have a continuous nominal path to that region under the <=100 m grid cap. **20–25 km is a preferred region when supported, not a required boundary altitude.**

Longer temporal averaging is not presently justified as the fix: 20/40/60 min windows give median continuous tops ~14.81/14.87/14.96 km respectively. The first v5 prototype therefore keeps 20 min.

**Current conclusion:** preserve the validated lower column exactly, coarsen progressively above 6 km, let the high supported top remain measurement-dependent, and carry boundary sensitivity explicitly.

---

# 5. P5.4 — actionable method-v5 checklist

Full contract: `docs/high_column_method_v5_decisions.md`.

## 5.1 Scientific product definition

* [x] Method v5 is an **elastic retrieval conditional on declared boundary, lidar-ratio and effective-resolution assumptions**.
* [x] Nominal boundary remains `beta_aer(ref)=0` / `f=0` for comparability.
* [x] `f=0` is an assumption, not an observation of molecular purity.
* [x] Residual `f` is treated as a separate systematic sensitivity dimension.
* [x] Backscatter and extinction may both be produced; extinction remains conditional on assumed/climatological LR.

## 5.2 Boundary estimator

* [x] High boundary is represented by the same progressive-grid cell used by KFS.
* [x] Effective cell width and source-bin count remain explicit.
* [x] Native 1-km Rayleigh QA remains separate from the aggregated-cell numerical boundary.
* [x] Broad-contamination bias remains a systematic failure mode; averaging is not purity validation.

## 5.3 Vertical representation

* [x] Preserve native 7.5 m representation through the established lower column below 6 km.
* [x] First R&D grid: 15 m at 6–10 km, 30 m at 10–15 km, 60 m at 15–25 km, <=100 m above 25 km.
* [x] Current 7.5 m SPU grid realizes the <=100 m cap as 97.5 m.
* [x] No interpolation, padding or source-bin reuse.
* [x] Missing/masked/instrument-invalid source sample invalidates its progressive cell.
* [x] Finite signed RCS may be averaged; aggregated KFS cell must be finite and positive.
* [x] Effective resolution/source count are retained by the R&D grid utility.
* [ ] Validate representation error and retrieval coverage in controlled truth before promotion.

Implementation: `milgrau/level2/adaptive_grid.py`; tests: `tests/test_adaptive_grid_rnd.py`.

## 5.4 MC / support semantics

* [x] V5 will not define physical support as `300/300` MC survival.
* [x] Nominal deterministic path support is separate from random-MC robustness.
* [x] Nested boundary-scenario MC reports valid count/fraction per `f` scenario.
* [x] No new valid-fraction cutoff is imposed yet.
* [ ] Derive any future cutoff from synthetic confidence-interval coverage, never from desired altitude.
* [ ] Add final valid-count/fraction fields to the v5 NetCDF/schema contract.

Implementation: `boundary_fraction_monte_carlo_sensitivity` in `milgrau/level2/boundary_sensitivity.py`.

## 5.5 Boundary `f` uncertainty semantics

* [x] Outer caller-declared `f` scenarios = epistemic/systematic boundary sensitivity.
* [x] Inner MC = random signal + declared LR/reference-estimator perturbations.
* [x] Same seed reused across `f` scenarios for paired comparison.
* [x] No arbitrary probability distribution assigned to `f`.
* [ ] Only marginalize `f` into a total probabilistic uncertainty if independent evidence later supplies a defensible distribution.

## 5.6 Temporal semantics

* [x] Keep current 20-min blocks for the first v5 prototype.
* [x] 40/60-min observational probes completed; median continuous top barely changes.
* [x] Do not trade temporal resolution for altitude without a demonstrated benefit.
* [ ] Revisit adaptive temporal averaging after grid + MC coverage validation if still needed.

## 5.7 Admissible path

* [x] Local Rayleigh QA alone is insufficient for path acceptance.
* [x] Continuous nominal progressive-cell path is diagnosed separately.
* [x] Missing/masked/instrument-invalid gaps are not bridged.
* [x] Cloud/layer state remains diagnostic unless a validated veto is established.
* [ ] Add characterized saturation/instrument masks when P4 evidence exists.

## 5.8 Lower-column preservation and truth validation

* [x] Current real-data lower-column sensitivity is quantified.
* [x] Residual-aerosol boundary bias exists in controlled truth.
* [ ] Define synthetic-truth acceptance in terms of bias + uncertainty coverage.
* [ ] Define real-data regression compatibility using uncertainty-normalized differences.
* [ ] Report both profile-shape and integrated-column differences.
* [x] Keep achieved top altitude out of the acceptance metric.

## 5.9 Rayleigh/reference selection

* [x] Rayleigh QA for v5 candidate cells remains on the native grid using a physical 1-km window.
* [x] Progressive cell is the numerical KFS boundary representation.
* [x] 20–25 km is preferred only when Rayleigh-compatible **and** path-admissible.
* [x] R&D catalogue can expose accepted/admissible cells without auto-ranking them.
* [ ] Determine the minimum altitude where v5 may depart from the v4 reference regime.
* [ ] Determine deterministic final ranking/tie-break among admissible high cells using synthetic truth.
* [ ] Verify selector behavior with explicit stratospheric-aerosol and cloud truth cases.
* [x] Do not introduce a composite molecular-purity score.

Implementation: `milgrau/level2/high_column_rnd.py`.

## 5.10 Method/schema promotion

* [x] Experimental v5 R&D execution exists for an explicitly selected reference cell.
* [ ] Run the executable v5 retrieval across the heterogeneous campaign.
* [ ] Freeze v4 and v5 outputs on identical Level-1 inputs.
* [ ] Bump productive retrieval method to v5 only after validation gates pass.
* [ ] Decide whether schema v3 can represent progressive altitude/resolution + scenario dimensions cleanly; otherwise bump schema.
* [ ] Record boundary model, `f` scenario, resolution/source count, supported top and MC valid fraction in NetCDF.

---

# 6. Minimum method-v5 validation matrix

* [ ] molecular-only progressive-grid synthetic truth;
* [x] residual-aerosol boundary synthetic mechanism;
* [x] weak-signal / many-isolated-bin path case;
* [x] localized and broad contamination counterexamples;
* [x] high-cloud/layer real case available (`20250509sant`);
* [x] heterogeneous 11-product SPU path/resolution feasibility;
* [x] 20/40/60-min temporal path comparison;
* [ ] executable v5 retrieval across heterogeneous campaign;
* [ ] explicit stratospheric-aerosol truth in candidate region;
* [ ] 355 and 532 synthetic truth separately;
* [ ] native vs progressive retrieval truth comparison;
* [ ] valid-MC fraction versus confidence-interval coverage;
* [ ] lower-column compatibility independently of top altitude;
* [ ] at least one glued/PC-dominant successful regime before instrument-wide generalization.

---

# 7. Raman — deferred, not deleted

Raman channel identity/feasibility evidence is preserved, but quantitative Raman retrieval is not a blocker for the first elastic v5 experiment.

* [x] Current acquisition contains 387/408/530 channels.
* [x] SCC mapping for 387/530 current nighttime configuration is known.
* [ ] Current receiver/filter spectral response remains unresolved.
* [ ] Quantitative Raman retrieval remains future work.

Future role: independent boundary validation and/or independent nighttime extinction retrieval after elastic v5 is stable.

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
* [ ] Quantify fitted slope/intercept uncertainty and covariance.
* [ ] Test a valid glued/PC-dominant regime.

P4 evidence improves generality but does not block the first conditional elastic v5 R&D path.

---

# 10. Closed engineering / QA gates

* [x] Exact-reference failures are block-local rather than wavelength-fatal (`tests/test_level2_kfs_block_failure.py`; CI `35248016374`).
* [x] Support-aware SR/KFS QA implemented and heterogeneous real-image reviewed (CI `35241784472`).
* [x] Scientific traceability regression contract protects schema v3 / method v4 (CI `35240916329`).
* [x] Station/calibration machine-readable provenance regression tested; CI `35257631767` passed Ruff and cross-platform pytest.
* [ ] Latest progressive-grid / nested-`f` / executable-v5 R&D commits: CI pending at this tracker snapshot.

---

# 11. P5.5–P5.8 / P6 deferred checklist

## P5.5 Ensemble

* [ ] Revisit only if one validated v5 boundary still leaves material reference ambiguity beyond explicit boundary-scenario sensitivity.

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

# 12. Immediate next scientific gates

The architecture decisions D0–D6 are sufficiently closed for R&D implementation. Next gates are now:

1. [ ] Run molecular-only and aerosol-known synthetic truth on the progressive grid and quantify representation bias.
2. [ ] Map `mc_valid_fraction` to actual confidence-interval coverage in controlled truth; derive a cutoff only if coverage requires one.
3. [ ] Add explicit stratospheric-aerosol truth in the candidate-reference region.
4. [ ] Use those synthetics to choose the minimum v5 reference altitude and final deterministic ranking among admissible cells.
5. [ ] Run the executable selected-cell v5 retrieval across the heterogeneous SPU campaign.
6. [ ] Compare v5 vs v4 lower column on identical Level-1 input as regression/sensitivity, not truth.
7. [ ] Decide schema-v3 extension versus schema-v4.
8. [ ] Only then promote productive retrieval method to v5.

No target altitude, MC fraction, SNR value or percent lower-column agreement is preselected as an acceptance threshold.

---

# 13. Current handoff

MILGRAU productive Level 2 remains **schema v3 / method v4**.

The first elastic method-v5 R&D architecture is now concrete: conditional product; nominal `f=0`; explicit outer residual-`f` scenarios; inner random Monte Carlo; native Rayleigh QA; progressive KFS grid preserving <6 km at 7.5 m and coarsening to <=100 m aloft; 20-min temporal baseline; nominal path support separated from MC survival.

The campaign evidence shows that this representation commonly supports paths into ~12–18 km but cannot guarantee 20–25 km. High reference altitude must therefore remain measurement-dependent. `20–25 km` is a preferred region when genuinely supported, not a success requirement.

Raman remains deferred as future independent validation. The remaining method-v5 blockers are synthetic uncertainty coverage, the final high-reference selector, heterogeneous executable retrieval validation and schema/product representation—not another Rayleigh score sweep.
