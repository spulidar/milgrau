# MILGRAU scientific engineering tracker

Branch: `new-architecture`
Tracker snapshot: 2026-09-17

This is the active scientific/engineering source of truth. Detailed history remains in Git, executable tests and `docs/regression_baselines/`.

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
* Thresholds are not chosen to reach a desired altitude.
* Productive semantic changes require explicit method/schema/provenance/baseline versioning.

---

# 1. Productive identity — unchanged

* Level-2 schema **v3**: auditable Rayleigh candidate catalogue.
* Retrieval method **v4**: QA-first candidate selection + backward Klett–Fernald–Sasano.
* Productive boundary: exact measured RCS bin at the selected reference altitude.
* Candidate rank: `relative_slope + relative_variance`, lower grid index as deterministic tie-breaker after minimum QA.
* Rayleigh SNR is diagnostic-only.
* Aerosol extinction is conditional on assumed aerosol lidar ratio.
* Productive optical support is inversion support, not generic finite-value support.

Not productive: fitted/window boundary, vertically aggregated high-column backbone, smoothing as support extension, hard SNR/cloud/temporal/continuity gates, overlap cutoff, physical PC saturation threshold, ensemble, cascade/stitching and model-implied wavelength-dependent molecular lidar ratio.

---

# 2. Active observational baseline

Primary baseline case: `20251107sapm`.

Active identity:

* L2 SHA `bba3b454387ef34c9b4bee3b0ab69e286e611cc44f9a4a6358b779f6d9f236d8`
* source L1 SHA `7b49c9deb93e89b103dba95455d854fcebb97b81292a18269edd1bea7a5ee342`
* revision `fa1cae20ec1ac28bf2955deb5a93f9a3ed56e663`
* source-code SHA `091d733499a9dd3026af6f7869ea1db2a1e57546ebde9d58c2009985ddb033e5`
* schema/method 3/4
* 25,340 candidate slots; 16,331 accepted; 10 selected
* profile-count weights 23 / 39 / 39 / 40 / 26

Historical evidence for the previous checksum is retained, but its original NetCDF is unavailable after workstation migration. Historical, active and campaign products are not claimed input-equivalent unless input hashes match.

Current P5.4 evidence files include:

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

---

# 3. Status overview

| Priority | Status | Meaning |
| --- | --- | --- |
| P0–P2 | COMPLETE | productive architecture/support/engineering guardrails |
| P3 | IN PROGRESS | FAIR/release/license/provenance |
| P4 | PARALLEL EVIDENCE | instrument characterization / external optical comparison |
| P5.0–P5.2 | FROZEN FOUNDATION | baselines, support semantics, QA-first catalogue |
| P5.3 | IN PROGRESS | real candidate interpretation |
| P5.4 | **ACTIVE PRIMARY R&D** | boundary-condition validity is now the main scientific blocker |
| P5.5 | PENDING | ensemble only if P5.4 establishes need |
| P5.6–P5.7 | DEFERRED | cascade / stitching |
| P5.8 | PARALLEL R&D | molecular semantics / uncertainty consistency |
| P5.9 | **IMPLEMENTED + REAL-IMAGE REVIEWED** | support-aware QA gate closed |
| P5.10 | IN PROGRESS | heterogeneous/seasonal validation matrix |
| P6 | PENDING | reproducible release/publication |

---

# 4. P5.4 established foundation

The active baseline and synthetic work established:

* candidate existence/persistence does not imply a usable KFS boundary;
* native paths near ~12 km can be nominally complete yet MC-fragile because many isolated weak bins accumulate failure probability;
* accepted candidates can exist above internal invalid backward paths;
* 60/120 m aggregation can improve support but trades vertical resolution and depends on covariance;
* fitted/window boundaries can denoise yet remain biased under contamination;
* leave-part-out can detect localized contamination but cannot certify molecular purity under broad common-mode contamination;
* no SNR, cloud, persistence, continuity or leave-out threshold is authorized.

---

# 5. Heterogeneous/seasonal campaign — observational regime gate met

Evidence: `p5_4_observational_campaign_20260917.json`.

The reviewed package contains 11 L2 products and 256 QA/quicklook images, plus Level 1 for `20250629sant`. It spans biomass-burning-labelled, low-cloud, high-cloud, clean seasonal/daypart and adjacent temporal records.

Key findings:

* “clean” does not imply easy high-column support;
* `20250509sant` high cloud shows accepted/persistent candidate islands reappearing above a strong layer, proving that shape QA + persistence cannot certify molecular purity;
* favorable `20250628/29` and fragile `20250521sapm` / `20250716sapm` controls separate atmospheric topology from statistical path fragility;
* all successful supplied campaign blocks use post-QA **analog fallback**, so glued/PC high-column behavior remains unvalidated.

The “second real regime” blocker is closed.

---

# 6. Empirical vertical dependence — first Level-1 estimate completed

From `20250629sant_level1_rcs.nc`, block-demeaned uncertainty-normalized residuals give, between 5 and 20 km:

* lag-1 / 7.5 m correlation ~**0.10–0.13**;
* most longer lags only a few percent;
* implied 60 m SNR gain ~**2.47–2.52**;
* implied 120 m gain ~**3.31–3.40**.

This one clean-night analog case lies much closer to independence than full correlation, but it is not a station-wide covariance law. Replication and source-chain diversity remain open.

---

# 7. Aggregation-only at the productive boundary — completed

Evidence: `p5_4_campaign_aggregation_same_boundary_20260917.json`.

The offline KFS implementation was first anchored by exact reproduction of all **150** successful persisted block/wavelength backscatter profiles at native group size 1.

With the existing productive boundary mapped to the coarse cell:

* 60 m independent: all 150 paths complete; median 0.6–6 km relative L2 difference **5.1%**, p95 **20.6%**; integrated-column p95 **26.9%**;
* 60 m fully-correlated error amplitude: all paths complete; median L2 **4.8%**, p95 **19.9%**;
* 120 m independent: all paths complete; median L2 **6.3%**, p95 **20.4%**; integrated-column p95 **33.5%**;
* 120 m fully-correlated amplitude: all paths complete; median L2 **6.1%**, p95 **20.6%**.

Decision: aggregation improves support but is **not lower-column neutral** and does not uniformly satisfy the existing loose 5% synthetic R&D preservation guard.

---

# 8. Aggregation effect decomposition — completed

Evidence: `p5_4_aggregation_effect_decomposition_20260917.json`.

At the productive boundary:

* deterministic coarse vs grouped native deterministic gives median relative L2 **4.6%** for 60 m and **5.6%** for 120 m, with p95 ~**21–22%**;
* aggregated MC vs aggregated deterministic has median difference ~**13%**;
* persisted native MC vs native deterministic also has median difference ~**13.4%**.

Interpretation: the nonlinear partial-MC mean already differs materially from deterministic KFS on the productive native grid; that is not a new aggregation artifact. The aggregation-specific structural/discretization + boundary-cell effect is smaller in median but strongly event dependent.

Decision: do not tune aggregation width to compensate for high-boundary bias. Aggregation and boundary validity are separate R&D dimensions.

---

# 9. High-boundary + aggregation stress experiment — completed, non-productive

Evidence: `p5_4_campaign_high_boundary_aggregation_20260917.json`.

Using an existing all-block accepted probe near 10 km:

* 60 m independent: 152/161 runs all-300 complete; median lower-column L2 difference **11.9%**, p95 **43.6%**;
* 120 m independent: 157/161 all-complete; median L2 **10.4%**, p95 **46.0%**;
* fully-correlated error amplitudes retain many incomplete runs;
* integrated-column p95 differences are ~**50–59%** depending on width/dependence case.

Support can improve dramatically while the lower-column optical product changes materially. This experiment mixes aggregation and boundary change, so it demonstrates that support gain is insufficient but does not isolate the dominant cause.

No backbone or width is authorized.

---

# 10. Native-grid boundary-only experiment — completed; main new result

Evidence: `p5_4_campaign_native_grid_boundary_only_20260917.json`.

The same accepted high probes were tested on the **native 7.5 m grid**, changing only the exact KFS boundary.

Across 161 input-valid block/wavelength runs:

* 2 exact reference bins are unusable, preserving the known window-QA vs exact-bin distinction;
* 159 runs have valid exact reference bins;
* only 111/159 are all-300 complete;
* median 0.6–6 km relative L2 difference vs productive method v4 is **13.1%**;
* p95 lower-column L2 difference is **47.2%**;
* median absolute integrated-column difference is **16.5%**, p95 **61.2%**.

Even favorable complete-path cases can change strongly. Example: `20250629sant/532` at 10.001 km is 14/14 all-complete but has median lower-column L2 difference ~**24.6%** and integrated-column p95 ~**69%**.

**Central result:** the dominant unresolved problem is not merely weak-bin support or rebinning. A higher accepted Rayleigh-like window does not establish validity of the physical boundary assumption `beta_aer(ref)=0`.

---

# 11. Do existing candidate diagnostics predict boundary risk? — no reliable evidence

Evidence: `p5_4_boundary_existing_diagnostic_association_20260917.json`.

142 native-grid boundary-only runs with valid productive lower-column controls were compared against existing candidate metrics.

Pooled correlations show some moderate associations, but they are confounded by regime/probe altitude. More importantly, within measurement/wavelength groups (17 groups with >=4 comparable blocks):

* median Spearman rho for relative slope vs lower-column change ~**-0.20**;
* relative variance ~**0.00**;
* candidate SNR ~**0.10–0.14**;
* diagnostic cost ~**0.00**;
* calibration factor ~**0.20–0.26**;
* signs vary across cases.

Path completeness also fails to rank lower-column change consistently.

Decision: there is no basis to solve the boundary problem by merely reweighting the current Rayleigh score or introducing a tuned composite threshold. Existing diagnostics characterize shape/precision/support; they do not validate the aerosol-free boundary condition.

---

# 12. Engineering/QA gates already closed

Exact-reference robustness:

* historical wavelength-fatal cases: `20240621sant/532`, `20240902sant/355`;
* KFS exact-bin rejection is now block-local without changing method-v4 Rayleigh semantics;
* implementation commit `f90c6f770e7b8d369e775d1ed812c17fbe1266e3`;
* `tests/test_level2_kfs_block_failure.py`;
* CI `35248016374` passed.

P5.9 support-aware QA:

* inversion top/support semantics are explicit in SR/KFS plots;
* `tests/test_level2_qa_support_context.py`;
* CI `35241784472` passed;
* heterogeneous human image review passed, including high-cloud `20250509sant`.

---

# 13. Parallel open work

P3:

* deliberate license decision + root `LICENSE` + metadata alignment;
* representative L2 metadata review;
* SPU lidar-ratio climatology provenance;
* historical calibration provenance.

P4:

* overlap/telecover/alignment evidence;
* physical PC saturation characterization;
* gluing fit covariance/materiality;
* a real glued/PC-dominant regime;
* true external optical comparison. `20250629sant_scc.nc` is SCC-ready **Level 0**, not SCC/ELDA optical L2.

P5.8:

* productive molecular lidar ratio remains `8*pi/3` until receiver/filter semantics are resolved.

P6:

* reproducible environment/build checks, warnings/coverage, release metadata, branch protection/reconciliation and immutable scientific tags.

---

# 14. Next scientific gate before any method v5

The evidence now points to **boundary-condition validity** as the primary P5.4 problem.

Next work must:

1. Build controlled synthetic cases with known **non-zero aerosol at the reference altitude** while keeping Rayleigh-window shape approximately molecular. Quantify how residual `beta_aer(ref)` propagates into the 0–6 km inversion.
2. Sweep residual aerosol fraction at the boundary separately from signal noise, vertical aggregation and lidar-ratio uncertainty.
3. Test whether any observable, physically motivated diagnostic can constrain that residual without using the retrieved aerosol itself circularly.
4. Use high-cloud `20250509sant` as a mandatory contamination-topology counterexample: accepted islands above the cloud cannot certify a molecular boundary.
5. Replicate vertical covariance on more Level-1 cases and obtain glued/PC-dominant evidence before instrument-wide aggregation claims.
6. Define lower-column preservation from physics/uncertainty/validation needs, not by tuning a tolerance to make a desired altitude pass.
7. Only then decide whether a method-v5 high-column backbone is scientifically justified. Ensemble remains after that decision; cascade/stitching remain deferred.

---

# 15. Current handoff

MILGRAU productive Level 2 remains **schema v3 / method v4**.

P5.4 has progressed from “can we find a higher candidate?” to a much sharper result: **higher Rayleigh-like candidates can be statistically precise, persistent and path-complete yet still alter the lower-column KFS retrieval because the aerosol-free boundary assumption is not validated**. Multi-case aggregation improves support but is not neutral; native-grid boundary-only experiments show even larger lower-column sensitivity; existing Rayleigh shape/SNR/cost diagnostics do not reliably rank that risk.

Therefore the next R&D is not another candidate score or altitude threshold. It is a controlled investigation of residual aerosol at the boundary and how, if at all, that physical assumption can be constrained observably. No hard threshold, cloud veto, fitted boundary, aggregation width, high-column backbone, ensemble, cascade or method-v5 promotion is authorized.
