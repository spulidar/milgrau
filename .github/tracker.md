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

Primary baseline case: `20251107sapm`.

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
* `p5_4_observational_campaign_20260917.json`

Historical, active-baseline and campaign products are not claimed input-equivalent unless their input hashes match.

---

# 3. Status overview

| Priority | Status | Meaning |
| --- | --- | --- |
| P0–P2 | COMPLETE | productive architecture/support/engineering guardrails |
| P3 | IN PROGRESS | FAIR/release/license; method-v4 traceability corrected and guarded |
| P4 | PARALLEL EVIDENCE | instrument characterization / true external-chain comparisons still open |
| P5.0–P5.2 | FROZEN FOUNDATION | baselines, support semantics, QA-first catalogue |
| P5.3 | IN PROGRESS | real candidate interpretation |
| P5.4 | **ACTIVE PRIMARY R&D** | multi-case high-column support/boundary/aggregation evidence |
| P5.5 | PENDING | ensemble only if P5.4 establishes need |
| P5.6–P5.7 | DEFERRED | cascade / stitching |
| P5.8 | PARALLEL R&D | molecular semantics / uncertainty consistency |
| P5.9 | **IMPLEMENTED + REAL-IMAGE REVIEWED** | support-aware QA semantics now visually checked on heterogeneous cases |
| P5.10 | IN PROGRESS | validation matrix expanded to heterogeneous/seasonal campaign |
| P6 | PENDING | reproducible release/publication |

---

# 4. P5.4 HighColumnEvidence — active-baseline gate populated

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

# 7. 60/120 m support-edge aggregation — single-case experiment completed, R&D only

Evidence: `20251107sapm_vertical_aggregation_support_edge.json`.

Strict non-overlapping aggregation was tested with no padding/interpolation:

* 60 m = 8 native bins;
* 120 m = 16 native bins.

Under independent source-bin error combination, both widths yielded 100% complete 300-member MC paths in all blocks at the ~12 km probes. Under a fully-correlated-within-group uncertainty amplitude, support remains dramatically better than native but worst block completeness is ~98.7–99.7%, so current all-simulations-valid productive semantics are not automatically met.

This event does not identify a 60 vs 120 m winner. Existing controlled synthetic truth tests show the coarsened inversion can remain within the current loose 5% R&D truth guard while separately exposing narrow-layer resolution loss.

Decision: aggregation remains promising R&D, not a productive width choice. The next aggregation gate is explicitly multi-case and must check 0–6 km preservation.

---

# 8. Explicit covariance families — synthetic gate executable-complete

Executable evidence: `tests/test_vertical_aggregation_correlation_family_rnd.py`.
Frozen evidence: `p54_covariance_leaveout_synthetic.json`.

Equicorrelation gain families:

* 60 m / 8 bins: 2.83 at rho=0 -> 2.17 at 0.1 -> 1.61 at 0.3 -> 1.24 at 0.6 -> 1.05 at 0.9 -> 1.0 at 1;
* 120 m / 16 bins: 4.0 -> 2.53 -> 1.71 at 0.3 -> 1.26 at 0.6 -> 1.05 at 0.9 -> 1.0 at 1.

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

# 10. Heterogeneous/seasonal SPU campaign — observational regime gate met

Evidence: `p5_4_heterogeneous_spu_cases_20260917.json` and `p5_4_observational_campaign_20260917.json`.

The reviewed user package SHA is `017be4afc3607b4a4df417ca510d672bffb0ecc52733f55f7a2dfc256707cf43`. It contains 11 Level-2 products and 256 QA/quicklook images, plus Level 1 and SCC-ready Level 0 material for `20250629sant`.

Observed campaign identities include biomass-burning-labelled, low-cloud, high-cloud, user-labelled clean seasonal/daypart controls and adjacent records. User atmospheric labels are treated as observational annotations rather than exact aerosol/cloud truth.

Important source-selection limitation: **all successfully retrieved wavelength blocks in this supplied campaign use post-QA analog single-channel fallback**. The campaign therefore provides strong atmospheric/Rayleigh/KFS stress coverage but does not validate glued or photon-counting high-column behavior.

Cross-case findings:

* “clean” does not imply easy high-column support: `20250628/29` are favorable near 10 km, while `20250521sapm`, `20250716sapm` and the campaign lineage of `20251107sapm` can remain MC-fragile despite no dominant obvious cloud in the quicklook;
* high-cloud `20250509sant` has a strong scattering-ratio layer near 14.36–14.37 km. The first connected all-block accepted candidate run ends near 12.1–12.3 km, but disconnected accepted/persistent islands reappear above the cloud at ~15.3 km and higher;
* therefore local shape QA + temporal persistence cannot certify molecular purity;
* biomass-burning-labelled `20240902sant` has very poor nominal path support near 10 km despite accepted candidates there;
* low-cloud `20241202saam` keeps productive references near ~5.6–5.9 km and becomes highly uncertainty-fragile before 10 km;
* adjacent clean-night records `20250628/29` provide the most favorable multi-block controls in the supplied campaign, with nominal ~10-km paths intact in every block;
* `20250716saam` is visually favorable but has only one block, so it cannot establish temporal persistence; the same-day later `20250716sapm` record is substantially more fragile;
* candidate topology — especially the first connected all-block accepted run — is scientifically informative as a threshold-free diagnostic, but is **not** authorized as a productive boundary rule because isolated gaps and strict all-block intersection can also truncate otherwise informative regions.

The user-listed 2024-05-22 high-cloud morning and January-27 clean-summer case are not present in the supplied ZIP; the latter also lacks a specified year. The user-labelled 2025-05-20 clean-autumn case is represented by canonical file `Measurement_ID=20250521sapm`.

Decision: the “second real regime” blocker is closed. The primary P5.4 question is now whether an aggregation/backbone can preserve the lower column across favorable and unfavorable real regimes while remaining contamination-aware and covariance-honest.

---

# 11. Empirical vertical dependence — first real Level-1 estimate completed

Evidence is frozen inside `p5_4_observational_campaign_20260917.json` from `20250629sant_level1_rcs.nc`.

Method: block-demeaned Level-1 range-corrected-signal residuals divided by supplied per-profile uncertainty, using the configured 20-minute floor blocks and the existing `vertical_noise_autocorrelation` R&D semantics. Because the corresponding Level-2 retrieval selected analog fallback, diagnostics were evaluated for `355.AN` and `532.AN`.

Between 5 and 20 km:

* lag-1 / 7.5-m correlation is approximately **0.10–0.13**;
* most longer-lag correlations are only a few percent;
* autocorrelation-adjusted SNR gain for 60 m / 8 bins is approximately **2.47–2.52**, below the independent limit sqrt(8)=2.83;
* corresponding 120 m / 16-bin gain is approximately **3.31–3.40**, below the independent limit 4.0 and far above the fully-correlated limit 1.0.

Interpretation: for this one clean-night analog case, the observed vertical dependence is materially closer to the independent limit than to the fully-correlated limit. This is **not** a station-wide covariance law: unresolved atmospheric variability can contribute to the residual correlation, and replication is still required for other Level-1 regimes and signal sources.

Decision: the generic “no empirical covariance evidence” gate is closed. Replication/source-chain diversity remains open.

---

# 12. Exact-reference KFS failure — engineering robustness fix validated

Historical campaign products exposed the old orchestration failure in:

* `20240621sant` / 532 nm;
* `20240902sant` / 355 nm.

The Rayleigh window can satisfy configured valid-fraction/slope/variance QA even when the **exact center bin** required by productive KFS is non-finite or non-positive. KFS correctly rejects that boundary, but the old call path propagated the `ValueError` and aborted the whole wavelength.

Engineering decision: preserve method-v4 Rayleigh semantics and exact-bin KFS semantics; treat this expected KFS input rejection as **block-local**, leave its KFS-valid flag false, and continue evaluating other blocks. Do not add a new center-bin Rayleigh gate without an explicit scientific method decision.

Implementation commit: `f90c6f770e7b8d369e775d1ed812c17fbe1266e3`.
Regression test: `tests/test_level2_kfs_block_failure.py`, commit `9829e79e68814092fe8c1afa2626d4082efb2e86`.
Cross-platform CI run `35248016374` completed successfully.

A current-code **real Level-1 re-run** of the two historical failure cases remains desirable, but their Level-1 files are not in the supplied campaign package and this no longer blocks the unit-level engineering fix.

---

# 13. P5.9 support-aware QA — implemented and heterogeneous real-image reviewed

`milgrau/viz/level2_qa_support.py` marks the algorithmic inversion top and visually identifies the region above it as outside supported aerosol retrieval while preserving scattering ratio as a diagnostic.

`tests/test_level2_qa_support_context.py` guards explicit `retrieval_top_altitude_m` handling and fallback through `retrieval_inversion_support_flag`. Cross-platform CI run `35241784472` passed.

Human review covered low-cloud, high-cloud, biomass-burning-labelled, clean/adjacent controls and seasonal/daypart examples. The support-aware SR/KFS panels visibly distinguish finite high-altitude scattering ratio from supported aerosol retrieval, including the critical `20250509sant` high-cloud case.

Decision: the principal P5.9 support-visualization gate is closed. Candidate density remains diagnostic rather than validation, and any future R&D panels must remain explicitly labeled R&D.

---

# 14. External-comparison status

The supplied `20250629sant_scc.nc` is **MILGRAU Level 0 SCC-compatible raw input** (`SCC_Ready=1`), not an SCC/ELDA optical Level-2 retrieval. It therefore cannot serve as an external aerosol backscatter/extinction comparison.

P4/P5.10 external-chain comparison remains open and requires matched SCC/ELDA/LPP optical output with compatible wavelength/product semantics.

---

# 15. P3 scientific traceability — drift corrected

`docs/scientific_traceability.md` explicitly documents productive schema v3 / method v4, current candidate-catalogue ownership, inherited uncertainty/support semantics, R&D-only P5.4 ownership and active baseline evidence.

`tests/test_scientific_traceability_contract.py` guards canonical version identity against the known method-v3 documentation drift. Full cross-platform CI run `35240916329` completed successfully.

Remaining P3: deliberate software license, root `LICENSE`, license metadata alignment, representative metadata review, lidar-ratio climatology provenance and defensible historical calibration provenance.

---

# 16. Validation state before any method v5

Synthetic established: molecular/aerosol recovery, grid convergence, missing uncertainty/gap semantics, QA-first catalogue, temporal persistence/transience, vertical aggregation/resolution trade, fitted-window denoising, contaminated-window counterexamples, independent/correlated fitted-boundary MC, molecular-lidar-ratio sensitivity, wider noise amplitudes, asymmetric/broad contamination, boundary placement, explicit covariance families and leave-part-out counterexamples.

Real SPU established: active baseline; heterogeneous/seasonal campaign; candidate catalogues; temporal evidence; SNR/dependence brackets; exact-vs-window evidence; populated HighColumnEvidence; first high-boundary comparison; weak-bin failure decomposition; single-case 60/120 m support-edge aggregation; high-cloud candidate re-entry above a strong layer; favorable and unfavorable clean controls; first empirical Level-1 vertical-dependence estimate; support-aware QA human review.

Still required before any method-v5 proposal:

* multi-case 60/120 m aggregation/backbone experiments spanning favorable and unfavorable real regimes;
* explicit **0–6 km preservation** against method-v4 controls whenever a high-column experiment changes vertical resolution or boundary treatment;
* contamination/layer stress tests that do not rely on a tuned hard cloud veto;
* replication of empirical dependence on additional Level-1 cases and, critically, a materially different signal-source regime such as valid glued or photon-counting contribution;
* matched external optical-chain comparisons when feasible;
* current-code real Level-1 rerun of historical exact-reference failures when those source files become available.

P5 success is defensible inversion-supported coverage with honest uncertainty, temporal representativeness, declared resolution and exact provenance — not “reach 20 km”.

---

# 17. Parallel open work

P4: overlap/telecover evidence, physical PC saturation, gluing fit covariance/materiality, source-chain diversity and true external optical comparison.

P5.8: productive molecular lidar ratio stays `8*pi/3` until actual receiver/filter molecular semantics are resolved.

P5.9: primary support-aware visualization gate is closed; maintain semantic guards for future panels.

P6: reproducible environment/build artifact tests, warnings/coverage, release metadata, branch reconciliation/protection and immutable scientific tags remain pending.

---

# 18. Immediate next gate

1. Run a **multi-case strict 60/120 m aggregation experiment** using favorable controls (`20250628sant`, `20250629sant`, and usable `20240621sant/355`) and unfavorable/stress controls (`20250521sapm`, `20250716sapm`, `20241202saam`, `20250509sant`).
2. For every aggregation experiment, preserve the method-v4 native-grid result as control and quantify differences in aerosol backscatter/extinction/support over **0–6 km** separately from any high-column gain.
3. Carry at least independent, fully-correlated and the observational `20250629sant` autocorrelation-informed dependence cases where scientifically meaningful; never present the latter as a universal instrument covariance.
4. Keep `20250509sant` as the contamination topology stress case: no experimental boundary logic may infer molecular purity merely because accepted/persistent candidates reappear above the high cloud.
5. Seek/retain a real Level-1 case with valid glued or photon-counting-dominant retrieval before generalizing a backbone to the instrument chain.
6. Only after these gates decide whether a method-v5 high-column backbone is scientifically justified. Multi-reference ensemble remains after that decision; cascade/stitching remain deferred.

---

# 19. Current handoff

MILGRAU productive Level 2 remains schema v3 / method v4. P5.4 is now supported by a heterogeneous and seasonal observational campaign rather than a single event. The campaign confirms two independent high-column limitations: atmospheric/topological contamination can create accepted/persistent candidate islands above a strong layer, while visually clean records can still be uncertainty-fragile. `20250629sant` additionally provides a first Level-1 empirical vertical-dependence estimate, with 5–20 km lag-1 correlations around 0.10–0.13 and 60/120 m SNR gains around 2.5/3.3–3.4, materially closer to independence than full correlation but not yet generalizable. Support-aware QA passed CI and heterogeneous human review. The exact-reference block-local robustness fix passed cross-platform CI. All successful supplied campaign retrievals still use analog fallback, so source-chain diversity remains an explicit evidence gap. The next decisive P5.4 step is a multi-case 60/120 m experiment with explicit 0–6 km preservation and covariance sensitivity. No hard threshold, cloud veto, fitted boundary, aggregation width, ensemble, cascade or method-v5 promotion is authorized.