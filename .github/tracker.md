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
* A fitted/window boundary or a vertically aggregated boundary is a new retrieval assumption.
* Clean center bin != clean molecular window; narrow Monte Carlo spread != absence of bias.
* Thresholds are not chosen to reach a desired altitude.
* Productive semantic changes require explicit method/schema/provenance/baseline versioning.

---

# 1. Current productive identity — unchanged

* Level 2 product schema: **v3**, auditable Rayleigh candidate catalogue.
* Retrieval method: **v4**, QA-first candidate selection + backward Klett–Fernald–Sasano.
* Productive boundary: exact measured RCS bin at the selected reference altitude.
* Candidate policy: enumerate complete windows -> apply configured minimum QA -> rank accepted candidates by `relative_slope + relative_variance`, lower grid index as deterministic tie-breaker.
* Propagated Rayleigh SNR is diagnostic-only.
* Aerosol extinction is conditional on assumed aerosol lidar ratio.
* Productive optical support means inversion support, not generic finite-value support.

Still **not productive**: fitted/window boundary, vertically aggregated high-column backbone, smoothing as support extension, hard SNR/cloud/temporal gates, overlap cutoff, physical PC saturation threshold, multi-reference ensemble, cascade/stitching and model-implied wavelength-dependent molecular lidar ratio.

---

# 2. Observational baseline policy

Primary baseline case: `20251107sapm`.

Historical derived evidence is preserved for L2 SHA `32b793fc7874f850ee38273f7073c957194f87dd08fed1c340e96b976917a212`, source L1 SHA `52663cbfd7863db6d38bc6596616c672c573c19bc55a3c5e72403abbc4ba0b3f`, revision `b68e4d812a37c2a003e49e30c15c2e112e4c8360`. The original historical NetCDF is unavailable after workstation migration and is not a current blocker.

Active baseline identity:

* L2 SHA `bba3b454387ef34c9b4bee3b0ab69e286e611cc44f9a4a6358b779f6d9f236d8`
* source L1 SHA `7b49c9deb93e89b103dba95455d854fcebb97b81292a18269edd1bea7a5ee342`
* product source revision `fa1cae20ec1ac28bf2955deb5a93f9a3ed56e663`
* source-code SHA `091d733499a9dd3026af6f7869ea1db2a1e57546ebde9d58c2009985ddb033e5`
* schema/method 3/4
* 25,340 candidate slots evaluated; 16,331 accepted; 10 selected
* profile-count weights 23 / 39 / 39 / 40 / 26
* aggregate reference ~6078.75 m (355) / ~5846.25 m (532)
* aggregate inversion top ~6198.75 m (355) / ~6281.25 m (532)

Historical, active-baseline and campaign products are not claimed input-equivalent unless their input hashes match.

Primary P5.4 evidence files now include:

* `20251107sapm_active_baseline_p54_summary.json`
* `20251107sapm_first_high_boundary_experiment.json`
* `20251107sapm_path_failure_diagnostic.json`
* `20251107sapm_vertical_aggregation_support_edge.json`
* `p54_covariance_leaveout_synthetic.json`
* `p5_4_heterogeneous_spu_cases_20260917.json`
* `p5_4_observational_campaign_20260917.json`
* `p5_4_campaign_aggregation_same_boundary_20260917.json`
* `p5_4_campaign_high_boundary_aggregation_20260917.json`

---

# 3. Status overview

| Priority | Status | Meaning |
| --- | --- | --- |
| P0–P2 | COMPLETE | productive architecture/support/engineering guardrails |
| P3 | IN PROGRESS | FAIR/release/license/provenance |
| P4 | PARALLEL EVIDENCE | instrument characterization and true external optical comparison |
| P5.0–P5.2 | FROZEN FOUNDATION | baselines, support semantics, QA-first catalogue |
| P5.3 | IN PROGRESS | real candidate interpretation |
| P5.4 | **ACTIVE PRIMARY R&D** | multi-case boundary/aggregation/support evidence; no method-v5 yet |
| P5.5 | PENDING | ensemble only if P5.4 establishes need |
| P5.6–P5.7 | DEFERRED | cascade / stitching |
| P5.8 | PARALLEL R&D | molecular semantics / uncertainty consistency |
| P5.9 | **IMPLEMENTED + REAL-IMAGE REVIEWED** | support-aware QA gate closed |
| P5.10 | IN PROGRESS | heterogeneous/seasonal validation matrix |
| P6 | PENDING | reproducible release/publication |

---

# 4. P5.4 foundation already established

The active baseline established that candidate existence is not equivalent to a usable KFS boundary. At ~12 km the native path can be nominally complete yet become Monte-Carlo fragile because many isolated/short weak bins accumulate non-positive probability. Accepted/persistent candidates can exist still higher while the backward path contains internal invalid bins.

Synthetic/R&D evidence additionally established:

* independent-window SNR can greatly overstate precision when vertical dependence is ignored;
* 60/120 m aggregation can improve path completeness but loses vertical resolution;
* explicit correlation families can remove most apparent aggregation gain under strong correlation;
* leave-part-out detects some localized contamination but cannot certify purity under broad common-mode contamination;
* fitted/window boundaries can be precise and biased;
* no SNR, cloud, persistence, continuity or leave-out threshold is authorized from these experiments.

---

# 5. Heterogeneous/seasonal observational campaign — regime gate met

Evidence: `p5_4_observational_campaign_20260917.json`.

The reviewed user package SHA is `017be4afc3607b4a4df417ca510d672bffb0ecc52733f55f7a2dfc256707cf43`. It contains 11 Level-2 products and 256 QA/quicklook images, plus Level 1 and SCC-ready Level 0 material for `20250629sant`.

Campaign coverage includes biomass-burning-labelled, low-cloud, high-cloud, user-labelled clean seasonal/daypart controls and adjacent temporal records. Labels are observational annotations, not independent aerosol/cloud truth.

Key findings:

* “clean” does not imply easy high-column support: `20250628/29` are favorable near 10 km, while `20250521sapm`, `20250716sapm` and the supplied `20251107sapm` lineage remain substantially more fragile;
* high-cloud `20250509sant` has a strong SR layer near 14.36–14.37 km; the first connected all-block accepted region stops around 12.1–12.3 km, but disconnected accepted/persistent candidate islands reappear above the cloud around 15.3 km and higher;
* therefore shape QA + persistence cannot certify molecular purity;
* low-cloud and biomass-burning-labelled cases stress lower-column structure and path support differently from clean controls;
* candidate topology is useful as a diagnostic but is not a productive boundary rule;
* all successfully retrieved blocks in the supplied campaign use post-QA **analog single-channel fallback**. This campaign therefore does not validate glued or photon-counting high-column behavior.

The user-listed 2024-05-22 high-cloud morning and January-27 clean-summer cases are not present in the supplied ZIP; the January year is also unspecified. The user-labelled 2025-05-20 clean-autumn case is represented by canonical `Measurement_ID=20250521sapm`.

Decision: the “second real regime” blocker is closed.

---

# 6. Empirical vertical dependence — first Level-1 estimate completed

Using `20250629sant_level1_rcs.nc`, block-demeaned profile residuals normalized by supplied Level-1 uncertainty were analyzed with 20-minute floor blocks and the existing `vertical_noise_autocorrelation` R&D semantics.

For `355.AN` and `532.AN` between 5 and 20 km:

* lag-1 / 7.5-m correlation is about **0.10–0.13**;
* most longer lags are a few percent;
* inferred equal-variance stationary SNR gain is about **2.47–2.52** for 60 m / 8 bins;
* gain is about **3.31–3.40** for 120 m / 16 bins.

This is materially closer to the independent limit than to full correlation for this one clean-night analog case, but it is not a station-wide covariance law and unresolved atmospheric variability can contribute to the residual correlation.

Decision: the generic “no empirical covariance evidence” gate is closed; replication and source-chain diversity remain open.

---

# 7. Multi-case aggregation at the existing productive boundary — completed

Evidence: `p5_4_campaign_aggregation_same_boundary_20260917.json`.

Experiment design: strict fixed-origin non-overlapping 60/120 m pre-retrieval aggregation, while retaining each block's own productive method-v4 boundary by mapping it to the containing coarse cell. The aggregated result is compared to the corresponding coarse representation of persisted native-grid method-v4 aerosol backscatter over 0.6–6 km.

Validation anchor: group size 1 reproduces persisted `aerosol_backscatter_block` **exactly** for all 150 successful campaign block/wavelength retrievals; maximum and median 0.6–6 km relative L2 difference are 0.

Results across those 150 controls:

* 60 m, independent-within-aggregate error: all paths complete; median lower-column relative L2 difference **5.1%**, p95 **20.6%**; median absolute integrated-column difference **6.7%**, p95 **26.9%**;
* 60 m, fully-correlated error amplitude: all paths complete; median L2 **4.8%**, p95 **19.9%**;
* 120 m, independent: all paths complete; median L2 **6.3%**, p95 **20.4%**; integrated-column p95 **33.5%**;
* 120 m, fully-correlated error amplitude: all paths complete; median L2 **6.1%**, p95 **20.6%**.

The event dependence is large. For example, 60 m is comparatively benign for `20241202saam/532`, but `20250629sant/532` has median lower-column relative L2 difference ~14.4% even though it is one of the favorable high-column controls.

**Decision:** aggregation is not lower-column-neutral. Improved path completeness does not justify promotion. The current loose synthetic 5% R&D preservation guard is not met uniformly by real-data behavior relative to method-v4 controls.

---

# 8. Multi-case high-boundary + aggregation experiment — completed, non-productive

Evidence: `p5_4_campaign_high_boundary_aggregation_20260917.json`.

Experiment design: for each processed wavelength, take the pre-existing all-block accepted candidate nearest 10 km (or the highest all-block candidate when the persistent region does not reach 10 km), map it to the fixed-origin 60/120 m aggregate grid, and run backward KFS MC. Rayleigh QA is **not** rerun on the aggregated signal; this is a support/bias stress experiment, not a selection method.

Global results over 161 input-valid block/wavelength runs:

* 60 m independent: 152/161 runs have all 300 simulations complete; median lower-column L2 difference vs productive v4 **11.9%**, p95 **43.6%**; integrated-column p95 **52.4%**;
* 60 m fully-correlated amplitude: only 124/161 all-complete; lower-column p95 L2 **45.3%**;
* 120 m independent: 157/161 all-complete; median lower-column L2 **10.4%**, p95 **46.0%**;
* 120 m fully-correlated amplitude: only 122/161 all-complete; lower-column p95 L2 **45.5%**.

Important examples:

* `20250521sapm/532` remains incomplete even after aggregation in several blocks; under the fully-correlated amplitude neither width gives any all-300-complete set across all nine input-valid blocks;
* `20250716sapm` changes from native MC fragility near 10 km to complete paths under independent aggregation, but covariance dependence remains substantial;
* `20251107sapm/532` becomes 5/5 complete at 60 m under independent aggregation, yet its lower-column median L2 difference is ~25% and integrated-column p95 difference is ~62%; under fully-correlated amplitude no block is all-300 complete;
* `20250629sant/532` is already a favorable native ~10-km case, yet a 60 m high-boundary experiment still changes the lower column materially (median L2 ~19%, integrated-column p95 ~70%).

**Central result:** higher completeness is not equivalent to preservation of the optical product. These runs intentionally combine two changes — vertical aggregation and higher boundary — and therefore demonstrate that support gain alone is insufficient, not which change dominates each lower-column difference.

**Decision:** no 60/120 m high-column backbone, no width choice and no method-v5 promotion.

---

# 9. Exact-reference KFS robustness fix — validated

Historical campaign products exposed a wavelength-fatal orchestration failure in `20240621sant/532` and `20240902sant/355`: a Rayleigh window may pass window QA while the exact center bin required by productive KFS is non-finite or non-positive.

KFS correctly rejects such a boundary. The implementation now keeps this rejection **block-local** rather than aborting the whole wavelength, while preserving method-v4 Rayleigh semantics and exact-bin KFS semantics.

* implementation commit `f90c6f770e7b8d369e775d1ed812c17fbe1266e3`
* regression test `tests/test_level2_kfs_block_failure.py`
* cross-platform CI run `35248016374` passed

A current-code real Level-1 rerun remains desirable if those historical Level-1 files become available.

---

# 10. P5.9 support-aware QA — gate closed

`milgrau/viz/level2_qa_support.py` marks the algorithmic inversion top and distinguishes the region above it from supported aerosol retrieval while retaining scattering ratio as a diagnostic.

`tests/test_level2_qa_support_context.py` guards explicit top handling and support-flag fallback. Cross-platform CI run `35241784472` passed.

Human review covered low cloud, high cloud, biomass-burning-labelled, clean/adjacent and seasonal/daypart cases. The critical `20250509sant` panels visibly separate finite high-altitude SR from supported aerosol retrieval.

Decision: the principal P5.9 visualization gate is closed. Future R&D panels must remain explicitly labeled R&D; candidate density remains diagnostic, not validation.

---

# 11. External-comparison status

`20250629sant_scc.nc` is MILGRAU SCC-compatible **Level 0 raw input**, not SCC/ELDA optical Level 2 output. It cannot be used as an external aerosol backscatter/extinction validation.

P4/P5.10 still require matched SCC/ELDA/LPP optical output with compatible wavelength/product semantics when feasible.

---

# 12. Remaining P3 / P4 / P5.8 / P6 work

P3 FAIR/release:

* deliberate software-license decision and root `LICENSE`;
* license metadata alignment;
* representative Level-2 metadata review;
* provenance for SPU lidar-ratio climatology;
* defensible historical calibration provenance.

P4 instrument evidence:

* overlap / telecover / alignment evidence;
* physical PC saturation characterization;
* gluing slope/intercept uncertainty/covariance/materiality;
* real source-chain diversity beyond universal analog fallback;
* external optical-chain comparison.

P5.8 molecular semantics:

* productive molecular lidar ratio remains `8*pi/3` until receiver/filter molecular semantics are resolved;
* matched-semantic synthetic improvement remains R&D evidence only.

P6 reproducibility/release:

* reproducible environment/build artifact checks;
* warnings/coverage cleanup;
* release metadata;
* branch reconciliation/protection;
* immutable scientific tags.

---

# 13. Gate before any method-v5 proposal

The previous “run multi-case 60/120 m aggregation” gate is now **complete**. It does not support promotion.

Next experiments must decompose the observed lower-column changes rather than combine assumptions:

1. **Boundary-only experiment on native 7.5 m grid:** keep native resolution, move only the exact KFS boundary to controlled accepted probes. Quantify 0.6–6 km change vs method v4 and path completeness.
2. **Aggregation-only experiment:** already executed at the productive boundary; extend diagnostics to separate deterministic discretization/boundary-cell shift from Monte Carlo noise effects.
3. **Combined experiment only after 1–2:** compare whether any support gain remains scientifically useful once lower-column preservation is enforced.
4. Use `20250509sant` as a mandatory contamination-topology stress case; accepted islands above the cloud must never be interpreted as molecular purity by themselves.
5. Replicate vertical dependence on additional Level-1 cases and obtain at least one valid glued/PC-dominant regime before instrument-wide generalization.
6. Define a lower-column preservation criterion from physics/uncertainty and validation needs — **not** by tuning a tolerance to make a desired high boundary pass.
7. Only then decide whether a method-v5 high-column backbone is scientifically justified. Multi-reference ensemble remains after that decision; cascade/stitching remain deferred.

---

# 14. Current handoff

MILGRAU productive Level 2 remains **schema v3 / method v4**.

P5.4 now has a heterogeneous/seasonal real campaign, a first empirical Level-1 vertical-dependence estimate, synthetic covariance/contamination counterexamples, multi-case aggregation at the productive boundary and multi-case high-boundary aggregation stress tests. The new evidence changes the question: 60/120 m aggregation clearly improves support in many weak-bin cases, but it is **not a neutral extension of the existing retrieval**. Even without raising the boundary, real campaign lower-column differences are material and event dependent; when a ~10 km boundary is added, support often improves further while 0.6–6 km differences can become large.

Therefore the immediate scientific task is to isolate boundary effects from aggregation effects and establish a defensible lower-column preservation requirement. No hard threshold, cloud veto, fitted boundary, aggregation width, high-column backbone, ensemble, cascade or method-v5 promotion is authorized. Source-chain diversity beyond analog fallback remains an explicit evidence gap.
