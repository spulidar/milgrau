# MILGRAU scientific engineering tracker

Branch: `new-architecture`
Tracker snapshot: 2026-09-17

This is the active scientific/engineering source of truth. Detailed history remains recoverable from Git, executable tests and `docs/regression_baselines/`.

---

# 0. Non-negotiable rules

* Keep analytical/synthetic truth, observational regression, external-chain comparison and instrument characterization distinct.
* Real measurements are behavioral/regression evidence, not aerosol optical ground truth.
* Missing uncertainty is never zero uncertainty; unsupported bins remain unsupported.
* Backward KFS never extends above its accepted boundary and never jumps internal invalid gaps.
* High altitude is an objective, not permission to weaken QA.
* Long averaging must expose temporal contribution; aggregation must expose resolution loss.
* A fitted Rayleigh-window boundary is a new retrieval assumption.
* Clean center bin != clean molecular window; narrow MC spread != absence of systematic/model bias.
* Thresholds are not chosen to reach a desired altitude.
* Any productive semantic change requires deliberate method/schema/provenance/baseline versioning.

---

# 1. Productive identity — unchanged

* Level 2 schema: **v3** — auditable Rayleigh candidate catalogue.
* Retrieval method: **v4** — QA-first Rayleigh candidate selection + backward Klett–Fernald–Sasano.
* Productive boundary: exact measured RCS bin at the selected reference altitude.
* Candidate ranking: enumerate complete windows -> apply minimum QA -> rank accepted candidates by `relative_slope + relative_variance`; lower grid index is deterministic tie-breaker.
* Rayleigh SNR is diagnostic only.
* Aerosol extinction remains conditional on assumed aerosol lidar ratio.
* Productive optical support is inversion support, not generic `isfinite()` support.

Not productive: fitted boundary, high-column backbone, productive vertical aggregation, smoothing as support extension, hard SNR/cloud/temporal thresholds, overlap cutoff, physical PC saturation threshold, multi-reference ensemble, cascade/stitching and model-implied wavelength-dependent molecular lidar ratio.

---

# 2. Observational baseline policy

Primary case: `20251107sapm`.

Historical derived evidence is retained under historical L2 SHA-256 `32b793fc7874f850ee38273f7073c957194f87dd08fed1c340e96b976917a212`, source L1 `52663cbfd7863db6d38bc6596616c672c573c19bc55a3c5e72403abbc4ba0b3f`, revision `b68e4d812a37c2a003e49e30c15c2e112e4c8360`. The original historical NetCDF is unavailable after workstation migration and is no longer a development blocker.

Active reproducible baseline:

* L2 SHA-256: `bba3b454387ef34c9b4bee3b0ab69e286e611cc44f9a4a6358b779f6d9f236d8`
* source L1 SHA-256: `7b49c9deb93e89b103dba95455d854fcebb97b81292a18269edd1bea7a5ee342`
* product source revision: `fa1cae20ec1ac28bf2955deb5a93f9a3ed56e663`
* source-code SHA-256: `091d733499a9dd3026af6f7869ea1db2a1e57546ebde9d58c2009985ddb033e5`
* schema / method: 3 / 4
* catalogue: 25,340 evaluated; 16,331 accepted; 10 selected
* independently reconstructed temporal weights: **23 / 39 / 39 / 40 / 26**
* aggregate reference: ~6078.75 m (355), ~5846.25 m (532)
* aggregate inversion top: ~6198.75 m (355), ~6281.25 m (532)

Active evidence files:

* `20251107sapm_current_head_regeneration_preliminary.json`
* `20251107sapm_active_baseline_p54_summary.json`
* `20251107sapm_first_high_boundary_experiment.json`
* `20251107sapm_path_failure_diagnostic.json`
* `20251107sapm_vertical_aggregation_support_edge.json`
* `p54_covariance_leaveout_synthetic.json`

Historical and active products are not claimed input-equivalent.

---

# 3. Status overview

| Priority | Status | Meaning |
| --- | --- | --- |
| P0–P2 | COMPLETE | productive architecture/support/engineering guardrails |
| P3 | IN PROGRESS | FAIR/release/license/traceability |
| P4 | PARALLEL EVIDENCE | instrument characterization / external comparisons |
| P5.0–P5.2 | FROZEN FOUNDATION | baselines, support semantics, QA-first catalogue |
| P5.3 | IN PROGRESS | real candidate interpretation |
| P5.4 | **ACTIVE PRIMARY R&D** | high-column support/boundary/aggregation evidence |
| P5.5 | PENDING | ensemble only if P5.4 establishes need |
| P5.6–P5.7 | DEFERRED | cascade / stitching |
| P5.8 | PARALLEL R&D | molecular semantics / uncertainty consistency |
| P5.9 | IN PROGRESS | QA/schema/provenance presentation |
| P5.10 | IN PROGRESS | validation matrix |
| P6 | PENDING | reproducible release/publication |

---

# 4. P5.4 HighColumnEvidence — observational gate populated

The evidence vector keeps shape QA, bin SNR, window SNR under explicit dependence assumptions, exact-window persistence, dominant temporal contribution, subwindow disagreement, contamination status, effective resolution and estimator identity physically separate. There is no composite score or high-column pass/fail.

Established on the active baseline:

* [x] all 25,340 candidate slots joined/analyzed;
* [x] 16,331 accepted / 10 selected reconciled;
* [x] weights reconstructed from embedded timestamps;
* [x] altitude distributions and diagnostic redundancy inspected;
* [x] compact auditable summary frozen.

Findings:

* independent-bin window SNR is ~10–12x single-bin SNR, but the fully-correlated limit returns close to single-bin SNR;
* bin and window SNR are partly redundant noise-strength indicators;
* exact-window persistence and subwindow disagreement expose different failure modes;
* same-window persistence falls with altitude even when some high candidate exists in every block;
* small subwindow disagreement cannot certify purity because broad contamination can bias both halves together.

No hard SNR threshold, score, target altitude or method-v5 proposal is justified.

---

# 5. First high-boundary retrieval experiment — completed, non-productive

Evidence: `20251107sapm_first_high_boundary_experiment.json`.

The offline analysis first reproduced all 10 persisted productive method-v4 block backscatter means and standard deviations exactly on supported bins, then changed only the experimental boundary semantics.

Probes:

* moderate: all-five accepted candidate nearest 10 km — 10001.25 m at both wavelengths;
* nominal support edge: highest all-five accepted candidate with complete nominal exact backward path — 12176.25 m (355), 12393.75 m (532);
* extreme-high stress: highest all-five accepted candidate irrespective of path completeness — 15386.25 m (355), 18873.75 m (532).

Key result: **local candidate QA + temporal persistence != usable KFS boundary.**

Extreme-high candidates are accepted in all five blocks, yet no block has a complete nominal backward path. At 532 nm the center itself is positive in all blocks, but internal non-positive bins still break every path. No gap filling was introduced.

At the nominal ~12 km support edge, uncertainty perturbations are also extremely fragile: worst complete 300-member MC fractions are ~19.3% / 13.3% for exact boundaries and ~0.33% for fitted boundaries at 355 / 532 nm.

Decision: no higher/fitted productive boundary and no method-v5 promotion.

---

# 6. Weak-bin path failure — characterized

Evidence: `20251107sapm_path_failure_diagnostic.json`.

Under the current independent per-bin Gaussian signal perturbation model:

* near 10 km, no native bin has >1% individual risk of becoming non-positive, but accumulated path/ensemble tail risk can still break an all-300-valid requirement;
* near the ~12 km support edge, the worst blocks contain 27 weak-risk bins (355) and 32 (532) with >1% individual non-positive probability;
* maximum contiguous risk clusters are only 3 bins (355) and 2 bins (532).

Interpretation: the failure is distributed across many weak isolated/short-cluster bins, not one extended missing layer. That supports testing strict aggregation as a precision/resolution trade; it does **not** justify interpolation or gap filling.

---

# 7. Vertical aggregation support-edge experiment — completed, still R&D

Evidence: `20251107sapm_vertical_aggregation_support_edge.json`.

Strict non-overlapping 60 m (8-bin) and 120 m (16-bin) pre-retrieval aggregation was tested at the ~12 km support-edge probes. Signal is arithmetic mean; incomplete/invalid groups remain invalid; no padding/interpolation occurs.

Results:

* under independent source-bin errors inside each group, both 60 and 120 m achieved 100% complete 300-member MC paths in all blocks;
* under the fully-correlated-within-group uncertainty amplitude, support remains much better than native but worst block completeness is ~98.7–99.7%, so current all-simulations-valid productive semantics are not automatically met;
* both widths strongly improve support relative to native exact support-edge completeness (~19.3% at 355, ~13.3% at 532);
* this event does not establish a 60 m vs 120 m winner.

Existing controlled synthetic KFS tests show 60/120 m coarsening can preserve coarse-grid truth within the current loose 5% R&D guard while separately exposing narrow-layer resolution loss. The real event is not ground truth.

Decision: vertical aggregation is a scientifically plausible P5.4 R&D direction, not a productive width choice.

---

# 8. Covariance family — synthetic gate expanded

Executable evidence: `tests/test_vertical_aggregation_correlation_family_rnd.py`.
Frozen summary: `p54_covariance_leaveout_synthetic.json`.

Explicit equicorrelation families show aggregation gain decreasing monotonically with positive correlation:

* 60 m / 8 bins: gain 2.83 at rho=0 -> 2.17 at 0.1 -> 1.61 at 0.3 -> 1.24 at 0.6 -> 1.05 at 0.9 -> 1.0 at rho=1;
* 120 m / 16 bins: 4.0 -> 2.53 -> 1.71 -> 1.26 -> 1.05 -> 1.0.

Explicit exponential lag families (`rho_k=rho^k`) retain more gain than equicorrelation but show the same monotonic dependence. Under strong correlation, doubling from 60 to 120 m can yield almost no additional precision while still losing resolution.

These are synthetic brackets, not an inferred SPU covariance law.

---

# 9. Leave-part-out contamination sensitivity — synthetic gate expanded

Executable evidence: `tests/test_rayleigh_window_leave_out_rnd.py`.

* clean molecular window: factor exactly stable when each contiguous quarter is removed;
* localized asymmetric contamination: full factor ~1.1005 and leave-quarter-out relative factor range ~7.4%; this diagnostic can expose localized contamination;
* broad symmetric contamination: full factor ~1.227 while leave-quarter-out relative range stays ~1%; stable leave-out behavior therefore **does not certify molecular purity**.

Decision: leave-out state may be useful diagnostic evidence but cannot be converted into a universal contamination veto from these cases.

CI for the newest correlation/leave-out commits must be green before these implementation gates are marked executable-complete.

---

# 10. Validation state before any method v5

Synthetic established: molecular/aerosol recovery, grid convergence, missing uncertainty/gap semantics, QA-first catalogue, temporal support, aggregation resolution trade, fitted-window denoising, contamination counterexamples, independent/correlated fitted-boundary MC, molecular-lidar-ratio sensitivity, wider noise amplitudes, asymmetric/broad contamination, placement sensitivity, covariance families and leave-part-out counterexamples.

Real SPU established: active `20251107sapm` baseline, catalogue, temporal evidence, SNR/dependence brackets, exact-vs-window evidence, populated HighColumnEvidence, first high-boundary comparison, path-failure decomposition and 60/120 m support-edge aggregation experiment.

Still required:

* CI confirmation for latest synthetic additions;
* at least one materially different real SPU Level 2 product;
* lower-column truth preservation for any future aggregated/backbone method under controlled synthetics;
* explicit empirical covariance/dependence evidence where available;
* clear/high-aerosol/cloud/weak-signal/temporally-changing/different AN-PC regimes;
* matched LPP/SCC/ELDA comparisons when feasible.

P5 success means defensible inversion-supported coverage with honest uncertainty, temporal representativeness, declared resolution and exact provenance — not “reach 20 km”.

---

# 11. Parallel P3/P4/P5.8/P5.9/P6

P3: choose/add license, align CFF/package/release metadata, finish Level 2 metadata review, update `docs/scientific_traceability.md` fully to schema 3 / method 4, improve aerosol lidar-ratio provenance.

P4: overlap/telecover evidence, physical PC saturation, gluing fit covariance/materiality, additional AN/PC regimes, real cloud/layer validation and external-chain comparison.

P5.8: keep productive molecular lidar ratio at `8*pi/3` until actual receiver/filter molecular semantics are resolved.

P5.9: QA must distinguish diagnostic finite high-altitude quantities from supported aerosol retrieval; candidate density is not validation; R&D panels stay labeled R&D.

P6: reproducible environment/build tests, warnings/coverage, release metadata, branch reconciliation/protection and immutable scientific tags remain pending.

---

# 12. Immediate next gate

Proceed in this order:

1. Confirm CI for `tests/test_vertical_aggregation_correlation_family_rnd.py` and `tests/test_rayleigh_window_leave_out_rnd.py`.
2. Obtain **one materially different current SPU Level 2 schema-v3/method-v4 NetCDF** and run the same HighColumnEvidence -> path-support -> experimental-boundary workflow. Preferred stress regimes: temporal change, cloud/layer, weak signal, high aerosol or substantially different AN/PC dominance.
3. Do not derive thresholds from `20251107sapm` alone, even if the second experiment improves top altitude.
4. If the contrasting case also supports aggregation, build the next controlled synthetic experiment around lower-column truth preservation under explicit covariance + 60/120 m resolution trade.
5. Only then decide whether an experimental high-column backbone deserves a method-v5 proposal.
6. Multi-reference ensemble remains after that decision; cascade/stitching remain deferred.

---

# 13. Current handoff

MILGRAU productive Level 2 remains schema v3 / method v4. P5.4 now shows that the high-column limitation in `20251107sapm` is not simply candidate selection: locally accepted persistent windows may sit above internal non-positive KFS path bins, and nominal ~12 km paths become fragile under native-grid signal perturbations. Strict 60/120 m aggregation strongly improves path robustness in this event, but its gain is covariance-dependent and no width is preferred. Synthetic covariance families show strong correlation can erase nearly all extra precision from wider aggregation, while leave-part-out diagnostics can reveal localized contamination but can miss broad common-mode bias. No threshold, fitted boundary, aggregation width, ensemble or method-v5 promotion is authorized. The next real scientific gate is replication on a materially different current SPU product.
