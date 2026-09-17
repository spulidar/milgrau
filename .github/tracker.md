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
* A fitted Rayleigh-window boundary is a new retrieval assumption.
* Clean center bin != clean molecular window; narrow Monte Carlo spread != absence of bias.
* Thresholds are not chosen to reach a desired altitude.
* Productive semantic changes require explicit method/schema/provenance/baseline versioning.

---

# 1. Current productive identity — unchanged

* Level 2 product schema: **v3**, auditable Rayleigh candidate catalogue.
* Retrieval method: **v4**, QA-first candidate selection + backward Klett–Fernald–Sasano.
* Productive boundary: exact measured RCS bin at the selected reference altitude.
* Candidate policy: enumerate complete windows -> apply configured minimum QA -> rank accepted candidates by `relative_slope + relative_variance`, with lower grid index as deterministic tie-breaker.
* Propagated Rayleigh SNR is diagnostic-only.
* Aerosol extinction remains conditional on assumed aerosol lidar ratio.
* Productive optical support is inversion support, not generic finite-value support.

Not productive: fitted boundary, high-column backbone, productive vertical aggregation, smoothing as support extension, hard SNR/cloud/temporal gates, overlap cutoff, physical PC saturation threshold, ensemble, cascade/stitching and model-implied wavelength-dependent molecular lidar ratio.

---

# 2. Observational baseline policy

Primary case: `20251107sapm`.

Historical derived evidence is preserved for L2 SHA `32b793fc7874f850ee38273f7073c957194f87dd08fed1c340e96b976917a212`, source L1 `52663cbfd7863db6d38bc6596616c672c573c19bc55a3c5e72403abbc4ba0b3f`, revision `b68e4d812a37c2a003e49e30c15c2e112e4c8360`. Its original NetCDF is unavailable after workstation migration and is not a current development blocker.

Active reproducible baseline:

* L2 SHA: `bba3b454387ef34c9b4bee3b0ab69e286e611cc44f9a4a6358b779f6d9f236d8`
* source L1 SHA: `7b49c9deb93e89b103dba95455d854fcebb97b81292a18269edd1bea7a5ee342`
* product source revision: `fa1cae20ec1ac28bf2955deb5a93f9a3ed56e663`
* source-code SHA: `091d733499a9dd3026af6f7869ea1db2a1e57546ebde9d58c2009985ddb033e5`
* schema / method: 3 / 4
* candidate slots: 25,340 evaluated; 16,331 accepted; 10 selected
* independently reconstructed profile-count weights: **23 / 39 / 39 / 40 / 26**
* aggregate reference: ~6078.75 m (355), ~5846.25 m (532)
* aggregate inversion top: ~6198.75 m (355), ~6281.25 m (532)

Active evidence files include:

* `20251107sapm_current_head_regeneration_preliminary.json`
* `20251107sapm_active_baseline_p54_summary.json`
* `20251107sapm_first_high_boundary_experiment.json`
* `20251107sapm_path_failure_diagnostic.json`
* `20251107sapm_vertical_aggregation_support_edge.json`
* `p54_covariance_leaveout_synthetic.json`
* `p5_4_heterogeneous_spu_cases_20260917.json`

Historical and active products are not claimed input-equivalent.

---

# 3. Status overview

| Priority | Status | Meaning |
| --- | --- | --- |
| P0–P2 | COMPLETE | productive architecture/support/engineering guardrails |
| P3 | IN PROGRESS | FAIR/release/license; method-v4 traceability drift corrected and guarded |
| P4 | PARALLEL EVIDENCE | instrument characterization / external comparisons |
| P5.0–P5.2 | FROZEN FOUNDATION | baselines, support semantics, QA-first catalogue |
| P5.3 | IN PROGRESS | real candidate interpretation |
| P5.4 | **ACTIVE PRIMARY R&D** | high-column support/boundary/aggregation evidence |
| P5.5 | PENDING | ensemble only if P5.4 establishes need |
| P5.6–P5.7 | DEFERRED | cascade / stitching |
| P5.8 | PARALLEL R&D | molecular semantics / uncertainty consistency |
| P5.9 | IN PROGRESS | support-aware QA implemented; heterogeneous visual review pending |
| P5.10 | IN PROGRESS | validation matrix |
| P6 | PENDING | reproducible release/publication |

---

# 4. P5.4 HighColumnEvidence — observational gate populated

`HighColumnEvidence` keeps shape QA, bin SNR, window SNR dependence limits, exact-window persistence, dominant temporal contribution, subwindow disagreement, contamination status, effective resolution and boundary/dependence identity physically separate. There is no composite score or preferred target altitude.

Completed on the active baseline:

* [x] all 25,340 slots analyzed;
* [x] 16,331 accepted / 10 selected reconciled;
* [x] temporal weights independently reconstructed;
* [x] altitude distributions and missingness inspected;
* [x] diagnostic redundancy assessed;
* [x] compact auditable evidence frozen.

Findings:

* independent-window SNR can be ~10–12x bin SNR, while the fully-correlated limit returns close to bin SNR;
* bin/window SNR are partly redundant as noise-strength indicators;
* persistence and subwindow disagreement expose different failure modes;
* same-window persistence falls with altitude even if some high candidate exists in every block;
* subwindow agreement cannot establish purity because broad common-mode contamination can bias both halves.

No hard threshold, score or method-v5 change is authorized.

---

# 5. First high-boundary experiment — completed, non-productive

Evidence: `20251107sapm_first_high_boundary_experiment.json`.

The offline analysis first reproduced persisted productive method-v4 block backscatter means and standard deviations exactly for all 10 block/wavelength cases, then altered only experimental boundary choices.

* Moderate all-five accepted probe: 10001.25 m at both wavelengths.
* Nominal support-edge probe: 12176.25 m (355), 12393.75 m (532).
* Extreme-high accepted/persistent stress probe: 15386.25 m (355), 18873.75 m (532).

Central result: **local candidate QA + temporal persistence != usable KFS boundary**.

Extreme-high candidates are accepted in every block yet have zero complete nominal backward paths. At 532 nm even positive center bins do not help because internal non-positive bins break support. No gap filling is permitted.

At the nominal ~12 km support edge, the native-grid path is nominally complete but MC perturbations are highly fragile: worst complete fractions are ~19.3% (355) and ~13.3% (532) for exact boundaries, with ~0.33% worst fitted-boundary completeness.

Decision: no higher/fitted productive boundary and no method-v5 promotion.

---

# 6. Weak-bin failure mechanism — characterized

Evidence: `20251107sapm_path_failure_diagnostic.json`.

Near the ~12 km support edge, worst blocks contain 27 weak-risk bins at 355 and 32 at 532 with >1% individual non-positive probability under the current independent Gaussian per-bin perturbation model. Maximum contiguous risk clusters are only 3 and 2 bins respectively.

Interpretation: support fragility is distributed across many isolated/short clusters, not one extended missing layer. This justifies testing strict aggregation as a resolution/precision trade; it does not justify interpolation.

---

# 7. 60/120 m support-edge aggregation — completed, R&D only

Evidence: `20251107sapm_vertical_aggregation_support_edge.json`.

Strict non-overlapping aggregation was tested with no padding/interpolation:

* 60 m = 8 native bins;
* 120 m = 16 native bins.

Under independent source-bin error combination, both widths yielded 100% complete 300-member MC paths in all blocks at the ~12 km probes. Under a fully-correlated-within-group uncertainty amplitude, support remains dramatically better than native but worst block completeness is ~98.7–99.7%, so current all-simulations-valid productive semantics are not automatically met.

This event does not identify a 60 vs 120 m winner. Existing controlled synthetic truth tests show the coarsened inversion can remain within the current loose 5% R&D truth guard while separately exposing narrow-layer resolution loss.

Decision: aggregation remains promising R&D, not a productive width choice.

---

# 8. Explicit covariance families — synthetic gate executable-complete

Executable evidence: `tests/test_vertical_aggregation_correlation_family_rnd.py`.
Frozen evidence: `p54_covariance_leaveout_synthetic.json`.

Equicorrelation gain families:

* 60 m / 8 bins: 2.83 at rho=0 -> 2.17 at 0.1 -> 1.61 at 0.3 -> 1.24 at 0.6 -> 1.05 at 0.9 -> 1.0 at 1;
* 120 m / 16 bins: 4.0 -> 2.53 -> 1.71 -> 1.26 -> 1.05 -> 1.0.

Exponential-lag correlation gives intermediate gains but preserves the same dependence trend. Strong correlation can erase almost all extra precision of the wider aggregate while resolution loss remains.

Cross-platform CI containing these tests and the leave-part-out tests passed in run `35240513895`.

These are explicit synthetic brackets, not an inferred SPU covariance law.

---

# 9. Leave-part-out contamination sensitivity — synthetic gate executable-complete

Executable evidence: `tests/test_rayleigh_window_leave_out_rnd.py`.

* clean molecular window stays invariant under removal of each contiguous quarter;
* localized asymmetric contamination gives a biased full fit (~1.1005) and ~7.4% leave-quarter-out factor range;
* broad symmetric contamination gives a strongly biased full fit (~1.227) while leave-quarter-out range stays ~1%.

Therefore leave-out sensitivity can expose localized contamination but **stable leave-out behavior is not a molecular-purity certificate**. No universal veto threshold is authorized.

---

# 10. Heterogeneous SPU stress cases — second-real-regime gate met

Evidence: `p5_4_heterogeneous_spu_cases_20260917.json`.

User-supplied observational annotations are treated as context, not independent atmospheric ground truth:

* `20240620sant`: plume / night;
* `20240902sant`: biomass-burning / morning;
* `20241202saam`: low cloud / morning;
* `20250509sant`: high cloud / night;
* `20240621sant`: unlabeled adjacent-day stress case.

All supplied products are schema v3 / method v4 generated from source revision `831ac2f4666c670fb3fec877fd581903a397a60d`.

Important source-selection limitation: **all successfully retrieved blocks in these products use post-QA analog single-channel fallback**. These cases therefore expand atmospheric/regime coverage for Rayleigh/KFS but do not validate glued or photon-counting high-column behavior.

Key real-event findings:

* high-cloud `20250509sant` has a strong scattering-ratio layer near 14.36–14.37 km; candidate acceptance is strongly disrupted through the layer, yet all-block accepted candidate islands reappear above it up to ~15.51 km (355) and ~17.50 km (532);
* therefore shape QA plus exact-window persistence is not sufficient to certify a molecularly clean high boundary, consistent with the synthetic broad-contamination counterexample;
* low-cloud `20241202saam` shows strong low-altitude scattering-ratio structure while the productive references remain near ~5.6–5.9 km;
* biomass-burning-labelled `20240902sant` strongly stresses lower-column aerosol structure and has only 4/7 valid backward 532-nm blocks;
* plume `20240620sant` retains 18/18 valid 355-nm blocks and 16/18 at 532 nm, while some 532-nm Rayleigh-selected references reach ~15.4–15.5 km but fail backward support;
* accepted candidates above 20 km occur in several cases at low diagnostic SNR; their existence is not validation of a usable KFS boundary.

Decision: the requirement for materially different real SPU regimes is no longer the primary P5.4 blocker. The next scientific gates are path contamination/layer handling without a tuned cloud veto, lower-column preservation under any aggregation/backbone experiment, explicit covariance behavior, and source-selection diversity beyond analog fallback.

---

# 11. Exact-reference KFS failure — engineering robustness fix

Two supplied cases exposed a method-v4 orchestration failure:

* `20240621sant` / 532 nm;
* `20240902sant` / 355 nm.

The Rayleigh window definition can satisfy configured valid-fraction/slope/variance QA even when the **exact center bin** required by productive KFS is non-finite or non-positive. KFS correctly rejects such a boundary, but the previous call path propagated the `ValueError` and aborted the whole wavelength.

Engineering decision: preserve method-v4 Rayleigh semantics and exact-bin KFS semantics; treat this expected KFS input rejection as **block-local**, leave its KFS-valid flag false, and continue evaluating other blocks. Do not add a new center-bin Rayleigh gate without an explicit scientific method decision.

Implementation commit: `f90c6f770e7b8d369e775d1ed812c17fbe1266e3`.
Regression test: `tests/test_level2_kfs_block_failure.py`, commit `9829e79e68814092fe8c1afa2626d4082efb2e86`.
CI run `35248016374` is the validation run for this change; at this tracker write it is still executing.

A current-code re-run of the two source Level 1 cases is still required to verify the observed wavelength-level failure becomes block-local in the real data path.

---

# 12. P3 scientific traceability — drift corrected

`docs/scientific_traceability.md` explicitly documents productive schema v3 / method v4, current candidate-catalogue ownership, inherited uncertainty/support semantics, R&D-only P5.4 ownership and active baseline evidence.

`tests/test_scientific_traceability_contract.py` guards canonical version identity against the known method-v3 documentation drift. Full cross-platform CI run `35240916329` completed successfully.

Remaining P3: deliberate software license, root `LICENSE`, license metadata alignment, representative metadata review, lidar-ratio climatology provenance and defensible historical calibration provenance.

---

# 13. P5.9 support-aware QA — implemented, visual review pending

The Level 2 QA path now distinguishes scattering-ratio context from supported aerosol optical retrieval. `milgrau/viz/level2_qa_support.py` marks the algorithmic inversion top and visually identifies the region above it as outside supported aerosol retrieval while preserving scattering ratio as a diagnostic.

`tests/test_level2_qa_support_context.py` guards both explicit `retrieval_top_altitude_m` handling and fallback through `retrieval_inversion_support_flag`. Cross-platform CI run `35241784472` passed.

Remaining P5.9 gate: human inspection on heterogeneous real cases, especially low/high cloud, to ensure finite high-altitude scattering ratio cannot visually read as retrieved aerosol and that molecular/candidate context is not misleading.

---

# 14. Validation state before any method v5

Synthetic established: molecular/aerosol recovery, grid convergence, missing uncertainty/gap semantics, QA-first catalogue, temporal persistence/transience, vertical aggregation/resolution trade, fitted-window denoising, contaminated-window counterexamples, independent/correlated fitted-boundary MC, molecular-lidar-ratio sensitivity, wider noise amplitudes, asymmetric/broad contamination, boundary placement, explicit covariance families and leave-part-out counterexamples.

Real SPU established: active `20251107sapm` baseline plus plume, biomass-burning-labelled, low-cloud, high-cloud and adjacent-day stress cases; candidate catalogues; temporal evidence; SNR/dependence brackets; exact-vs-window evidence; populated HighColumnEvidence; first high-boundary comparison; weak-bin failure decomposition; 60/120 m support-edge aggregation; real cloud-layer evidence that accepted/persistent candidate islands can reappear above a strong layer.

Still required before any method-v5 proposal:

* lower-column truth preservation for a future aggregated/backbone method under controlled synthetics and explicit covariance;
* path-contamination/layer tests that do not rely on a tuned hard cloud veto;
* empirical dependence/covariance characterization where feasible;
* at least one real regime with materially different signal-source behavior, ideally valid glued and/or photon-counting contribution rather than universal analog fallback;
* current-code re-run of the two exact-reference failure cases;
* visual QA review on the heterogeneous cases;
* matched LPP/SCC/ELDA comparisons when feasible.

P5 success is defensible inversion-supported coverage with honest uncertainty, temporal representativeness, declared resolution and exact provenance — not “reach 20 km”.

---

# 15. Parallel open work

P4: overlap/telecover evidence, physical PC saturation, gluing fit covariance/materiality, additional AN/PC regimes, real cloud/layer validation and external-chain comparison.

P5.8: productive molecular lidar ratio stays `8*pi/3` until actual receiver/filter molecular semantics are resolved.

P5.9: support-aware rendering is implemented; heterogeneous real-image review remains open. Candidate density remains diagnostic, not validation; R&D panels remain labeled R&D.

P6: reproducible environment/build artifact tests, warnings/coverage, release metadata, branch reconciliation/protection and immutable scientific tags remain pending.

---

# 16. Immediate next gate

1. Review Level 2 QA images for `20240620sant`, `20240902sant`, `20241202saam` and especially `20250509sant`; prioritize ScatteringRatio, KFS and Molecular panels at both wavelengths.
2. Re-run `20240621sant` and `20240902sant` from Level 1 on current HEAD after the block-local KFS fix and verify one bad exact reference no longer aborts the whole wavelength.
3. Use the high-cloud case as a real-event contamination stress test: experimental high-boundary logic must not infer molecular purity merely because accepted/persistent candidate islands reappear above the cloud layer.
4. Extend the 60/120 m R&D comparison to heterogeneous cases only after defining a path-safe, contamination-aware experiment and preserving 0–6 km products against the method-v4 control.
5. Seek a real case with different source-selection behavior (valid glued/PC contribution) before generalizing high-column conclusions to the instrument chain.
6. Only after those gates decide whether a method-v5 high-column backbone is scientifically justified. Multi-reference ensemble remains after that decision; cascade/stitching remain deferred.

---

# 17. Current handoff

MILGRAU productive Level 2 remains schema v3 / method v4. P5.4 no longer depends on a single observational regime: the new heterogeneous matrix adds plume, biomass-burning-labelled, low-cloud and high-cloud stress cases. The high-cloud case supplies particularly strong real-event evidence that local Rayleigh shape QA and temporal persistence can reappear above a strong layer and therefore do not certify a clean molecular boundary. All supplied successful blocks use analog fallback, so source-chain diversity remains an explicit evidence gap. Two cases also exposed a block-local exact-reference KFS robustness defect; the implementation now preserves Rayleigh acceptance but rejects only the unusable KFS block, with a focused regression test and CI validation in progress. Support-aware QA plotting has passed cross-platform CI and now needs visual review on these real cases. No hard threshold, cloud veto, fitted boundary, aggregation width, ensemble, cascade or method-v5 promotion is authorized.
