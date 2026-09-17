# MILGRAU scientific engineering tracker

Branch: `new-architecture`
Tracker snapshot: 2026-09-17

This file is the active scientific/engineering source of truth for current MILGRAU development. It is deliberately a current-state tracker, not a complete history; completed implementation history remains in Git, tests and `docs/regression_baselines/`.

---

# 0. Operating rules

Before changing productive science:

1. Read this tracker, `milgrau/scientific.py` and `docs/level2_schema.md`.
2. Distinguish analytical/synthetic truth, observational regression, external-chain comparison and instrument characterization.
3. State the scientific question and exit condition before changing code.
4. Do not promote a diagnostic because it improves altitude coverage in one event.
5. Missing uncertainty is never zero uncertainty; unsupported bins remain unsupported.
6. Productive backward retrieval support never extends above its accepted boundary and never bridges internal invalid gaps.
7. If productive semantics change, explicitly decide method/schema/provenance/baseline versioning.
8. Close scientific tasks with: question, evidence, finding, decision, scope, versioning and remaining uncertainty.

---

# 1. Current productive identity

* Level 2 product schema: **v3** — auditable Rayleigh candidate catalogue.
* Level 2 retrieval method: **v4** — QA-first candidate selection + backward Klett–Fernald–Sasano.
* Productive KFS boundary: exact measured RCS bin at the selected reference altitude.
* Productive Rayleigh ranking: enumerate complete candidates -> apply configured minimum QA -> rank accepted candidates by `relative_slope + relative_variance`; lower grid index is deterministic tie-breaker.
* Propagated Rayleigh SNR is diagnostic only.
* Aerosol extinction remains conditional on assumed aerosol lidar ratio.
* Productive optical support is inversion support, not generic `isfinite()` support.

Explicitly **not productive**: fitted/window-derived KFS boundary, high-column backbone, vertical aggregation for productive retrieval, post-retrieval support extension, hard Rayleigh SNR/cloud/temporal thresholds, overlap cutoff, physical PC saturation threshold, multi-reference ensemble, cascade/stitching and model-implied wavelength-dependent molecular lidar ratio.

Package CalVer remains `2026.9`; development identity therefore also requires source/repository provenance.

---

# 2. Observational baseline policy

Primary observational case remains `20251107sapm`. Real measurements are regression/behavior evidence, **not aerosol optical ground truth**.

## 2.1 Historical frozen evidence — retained, binary unavailable

Historical Level 2 SHA-256:

`32b793fc7874f850ee38273f7073c957194f87dd08fed1c340e96b976917a212`

Historical source Level 1 SHA-256:

`52663cbfd7863db6d38bc6596616c672c573c19bc55a3c5e72403abbc4ba0b3f`

Historical source revision:

`b68e4d812a37c2a003e49e30c15c2e112e4c8360`

The original historical NetCDF is no longer available after workstation migration. Its derived regression summaries remain valid historical records and must not be rewritten as if the current Level 1 were input-equivalent.

## 2.2 Active reproducible observational baseline

Current reproducible Level 2 SHA-256:

`bba3b454387ef34c9b4bee3b0ab69e286e611cc44f9a4a6358b779f6d9f236d8`

Current source Level 1 SHA-256:

`7b49c9deb93e89b103dba95455d854fcebb97b81292a18269edd1bea7a5ee342`

Product source revision:

`fa1cae20ec1ac28bf2955deb5a93f9a3ed56e663`

Product source-code SHA-256:

`091d733499a9dd3026af6f7869ea1db2a1e57546ebde9d58c2009985ddb033e5`

Active catalogue state:

* 25,340 / 25,340 candidate slots evaluated;
* 16,331 candidates accepted by current minimum QA;
* 10 productive selections;
* 2,534 candidates per block/wavelength case;
* embedded 167 profile timestamps reproduce block weights **23 / 39 / 39 / 40 / 26** for the configured 20-minute floor blocks;
* aggregate reference altitudes: ~6078.75 m at 355 nm and ~5846.25 m at 532 nm;
* aggregate inversion tops: ~6198.75 m at 355 nm and ~6281.25 m at 532 nm.

Frozen current evidence:

* `docs/regression_baselines/20251107sapm_current_head_regeneration_preliminary.json`
* `docs/regression_baselines/20251107sapm_active_baseline_p54_summary.json`

Policy: historical derived evidence stays preserved; the current product is the active reproducible observational baseline. There is **no claim of input equivalence** between them.

---

# 3. Non-negotiable scientific rules

* `config.yaml` owns processing/scientific recipe; `station.yaml` owns station/instrument reality and calibration history; Python owns generic equations.
* Near-range finite KFS values are not validated instrument support while overlap is uncharacterized.
* Long averaging must expose temporal contribution/stationarity.
* Vertical aggregation is an explicit resolution/precision trade, not creation of information.
* A fitted Rayleigh-window boundary is a new retrieval assumption.
* Clean center bin does not imply a clean molecular window.
* Narrow Monte Carlo spread does not exclude systematic/model bias.
* The same samples must not be counted twice as independent signal and boundary-fit uncertainty.
* Multiple references are sensitivity/ensemble evidence, not independent truths.
* Thresholds are not chosen to hit a desired altitude.
* Product metadata must not claim calibration or convention compliance that has not been demonstrated.

---

# 4. Status overview

| Priority | Status | Current meaning |
| --- | --- | --- |
| P0–P2 | COMPLETE | productive architecture/support/engineering guardrails |
| P3 | IN PROGRESS | FAIR/release metadata, license, traceability drift |
| P4 | PARALLEL EVIDENCE | instrument/observational characterization |
| P5.0–P5.2 | FROZEN FOUNDATION | baselines, inversion support, QA-first catalogue |
| P5.3 | IN PROGRESS | interpretation of candidate catalogue |
| P5.4 | **ACTIVE PRIMARY R&D** | high-column evidence and boundary robustness |
| P5.5 | PENDING | multi-reference ensemble, only if P5.4 justifies it |
| P5.6–P5.7 | DEFERRED | cascade / solution stitching |
| P5.8 | PARALLEL R&D | molecular semantics / uncertainty consistency |
| P5.9 | IN PROGRESS | QA/schema/provenance presentation |
| P5.10 | IN PROGRESS | validation matrix |
| P6 | PENDING | reproducible release/publication |

---

# 5. P3 / P4 parallel work

P3 remains release-blocking but does not block P5 research. Open items: deliberately select/add software license; align package/CFF/release licensing; representative Level 2 metadata review; bring `docs/scientific_traceability.md` fully to schema 3 / method 4; improve aerosol lidar-ratio climatology provenance; distinguish defensible historical calibration from inherited values.

P4 open evidence remains instrument-specific: overlap/telecover characterization, physical PC saturation, gluing fit-parameter covariance/materiality, additional AN/PC regimes, real cloud/layer validation and matched LPP/SCC/ELDA comparisons.

---

# 6. P5.4 — ACTIVE PRIMARY R&D

Scientific question: can MILGRAU use more information already present in the measurement to extend defensible backward optical support **without** biasing the established lower column, hiding nonstationarity/contamination/resolution loss, double-counting uncertainty or inventing thresholds from one favorable case?

## 6.1 Temporal evidence

Established:

* block support/contribution diagnostics exist;
* dominant contribution and exact-window candidate persistence exist;
* historical/current `20251107sapm` shows broad high-candidate availability;
* `20241219nt` remains the contrasting transient-support stress case;
* current active baseline independently reproduces weights 23/39/39/40/26.

New active-baseline finding: exact-window persistence declines strongly with altitude even when some accepted high candidate exists in every block. Therefore exact-window persistence is not interchangeable with generic “candidate available above altitude”.

No productive temporal threshold is authorized.

## 6.2 Vertical aggregation R&D

Established: strict non-overlapping pre-retrieval aggregation, explicit independent/fully-correlated uncertainty limits, real-event high-range SNR gain at 60–120 m and synthetic resolution-loss checks.

Open: broader narrow/broad-layer, covariance/noise and real-regime matrix. No productive aggregation width is selected; any future width must expose effective vertical resolution.

## 6.3 Rayleigh-window boundary R&D

Established:

* window calibration uncertainty helper;
* independent / fully correlated / supplied covariance assumptions kept distinct;
* fitted-boundary Monte Carlo does not double-count fit noise;
* clean synthetics show denoising potential;
* contamination can yield precise biased fitted boundaries;
* exact-bin/window disagreement grows strongly aloft;
* broad smooth contamination can evade a sharp-layer detector.

Active-baseline evidence by accepted-candidate altitude band shows:

| Band | 355 nm bin SNR | 355 nm window SNR independent / fully corr. | 532 nm bin SNR | 532 nm window SNR independent / fully corr. |
| --- | ---: | ---: | ---: | ---: |
| 5–10 km | 14.77 | 171.60 / 14.88 | 15.93 | 184.76 / 16.03 |
| 10–15 km | 3.14 | 36.63 / 3.18 | 3.57 | 41.74 / 3.62 |
| 15–20 km | 1.20 | 12.89 / 1.31 | 1.30 | 14.62 / 1.39 |
| 20–25 km | 0.91 | 8.88 / 1.02 | 0.87 | 8.91 / 0.99 |

Interpretation: the apparent ~10–12× window-SNR gain under independent bins nearly disappears under the fully-correlated limit. The scientific value of window fitting is therefore strongly dependence-model contingent.

## 6.4 HighColumnEvidence observational gate — POPULATED

`HighColumnEvidence` keeps shape QA, bin SNR, window SNR dependence limits, exact-window persistence, dominant block contribution, subwindow disagreement, contamination status, effective resolution, estimator identity and dependence model physically separate.

Completed on the active reproducible baseline:

* [x] candidate-level export logic exercised for all 25,340 slots;
* [x] current block weights independently reconstructed from embedded timestamps;
* [x] 16,331 accepted / 10 selected reconciled;
* [x] auditable compact summary frozen as `20251107sapm_active_baseline_p54_summary.json`;
* [x] candidate CSV generated locally with 25,340 rows; its SHA-256 and size are frozen in the summary rather than committing ~11 MB of derived rows;
* [x] altitude-stratified evidence inspected;
* [x] diagnostic redundancy qualitatively assessed.

Key findings:

1. Bin-wise SNR and both window-SNR limits are strongly rank-correlated through most of 10–20 km; they are partly redundant as noise-strength indicators.
2. Persistence and subwindow disagreement are less redundant with SNR and represent different failure modes.
3. Subwindow disagreement grows with altitude, but a small value does **not** certify molecular purity because broad smooth contamination can bias both halves together.
4. No composite score, hard SNR threshold, preferred target altitude, productive aggregation width or method-v5 change is justified by this event.

Scope: diagnostic/R&D only. Productive schema 3 / method 4 remain unchanged.

## 6.5 Synthetic robustness matrix — EXPANDED 2026-09-17

Added `tests/test_rayleigh_window_evidence_matrix_rnd.py` (implementation commit `7c2b449350fc84a3c06d28de93d99a8044782951`) covering:

* clean independent noise sweep from 1% to 30%;
* asymmetric contamination producing fit bias + large half-window disagreement;
* broad symmetric contamination producing >20% fit bias while half-window disagreement remains <1%;
* fitted-scale sensitivity to moving a 1-km window across a fixed contaminated structure.

The numerical assertions were checked against the analytical estimator before commit. These tests are R&D evidence only and define no productive threshold.

Remaining synthetic gates:

* correlated-noise families beyond independent/full-correlation brackets;
* explicit leave-part-of-window-out variants;
* retrieval-level lower-column preservation under candidate/boundary changes;
* temporal reject/split experiments;
* graceful support failure for future high-column experiments.

---

# 7. First offline high-boundary experiment

Now authorized as the **next non-productive experiment**, not as a productive method change.

Requirements:

* use existing schema-v3 candidates; do not alter productive method v4;
* choose a small number of explicitly experimental candidate boundaries based on transparent evidence dimensions, not a composite score or desired altitude;
* record exact candidate evidence and boundary estimator;
* compare against productive method-v4 retrieval.

Minimum comparison:

* lower column approximately 0–6 km: aerosol backscatter/extinction differences, uncertainty, support and profile structure;
* high column: supported top, uncertainty growth, boundary/window SNR dependence, temporal persistence, resolution and contamination limitations.

A higher retrieval top alone is not success. No permanent tolerance is to be invented before the comparison distribution is understood.

---

# 8. P5.8 / P5.9 / P5.10

P5.8: productive molecular lidar ratio remains `8π/3` until actual SPU receiver/filter molecular semantics are resolved; the model-implied difference remains an R&D sensitivity (~1.4–1.5%).

P5.9: QA must show optical support/top explicitly; finite high-altitude scattering ratio must not look like supported aerosol retrieval; candidate density is not validation; R&D panels remain labeled R&D. `docs/scientific_traceability.md` still needs full method-v4/schema-v3 cleanup.

P5.10 synthetic established: molecular-only recovery, aerosol recovery, grid convergence, missing uncertainty/gaps, QA-first catalogue, temporal support, aggregation trade, clean fitted-window denoising, contaminated-window counterexample, fitted-boundary MC, correlated MC and molecular-lidar-ratio sensitivity. The new evidence matrix adds wider noise, asymmetric/broad contamination and placement sensitivity.

Real-SPU established: active `20251107sapm` baseline, candidate catalogue, temporal evidence, SNR/dependence brackets, exact-vs-window evidence and populated high-column evidence vector. Still required: first higher-boundary comparison, at least one materially different SPU regime, clear/high-aerosol/cloud/weak-signal/temporally-changing/AN-PC-different cases and matched external-chain comparisons.

P5 success criterion: maximize **defensible inversion-supported coverage** while exposing why support ends, with honest uncertainty, temporal representativeness, resolution and exact boundary provenance. Success is not “reach 20 km”.

---

# 9. P6 release/publication readiness

Still pending: license, CFF/release consistency, reproducible constraints/environment, build/test sdist+wheel, end-to-end release-artifact run, frozen machine-readable acceptance summaries, warning policy, core-science coverage, package metadata/clean install, `new-architecture`/`main` reconciliation, branch protection/required checks and immutable scientific release tags.

Unfinished P5 R&D must not be forced into a release method.

---

# 10. Immediate next gate

Proceed in this order unless new evidence invalidates it:

1. Confirm CI for the new synthetic evidence matrix.
2. Run the first **offline non-productive high-boundary retrieval comparison** on the active reproducible `20251107sapm` baseline.
3. Compare every experiment with method v4, especially the 0–6 km lower column and uncertainty/support changes.
4. Add correlated-noise and leave-part-out synthetic tests if the experiment is sensitive to dependence/contamination ambiguity.
5. Repeat the evidence logic on at least one materially different SPU event before deriving any threshold/rule.
6. Keep vertical aggregation as a parallel declared-resolution experiment.
7. In parallel, fix scientific-traceability drift, improve lidar-ratio provenance, characterize gluing covariance and resolve receiver molecular semantics.
8. Only after these gates decide whether a method-v5 proposal is scientifically justified.
9. Multi-reference ensemble comes after that decision; cascade/stitching remain deferred.

---

# 11. Stop conditions

Pause a line of R&D if it cannot be separated from noise/model bias, is supported only by one favorable event, improves altitude while degrading the lower column, requires an arbitrary target-driven threshold, has unknown uncertainty dependence, hides temporal intermittency/resolution loss, relies on uncharacterized instrument behavior or can be answered by a simpler experiment first.

---

# 12. Current handoff

MILGRAU productive Level 2 remains schema v3 / method v4 with QA-first Rayleigh selection and backward KFS using the exact measured RCS bin as the boundary. The unavailable historical NetCDF is now explicitly historical evidence rather than a development blocker; the current checksum-identified `20251107sapm` product is the active reproducible observational baseline. Its P5.4 `HighColumnEvidence` gate is populated: 25,340 slots, 16,331 accepted candidates, explicit temporal weights, altitude-stratified SNR/dependence/persistence/subwindow evidence and no composite score. Window fitting shows real precision potential under independent noise, but its advantage collapses toward single-bin behavior under full correlation and broad smooth contamination can remain precise yet biased. The synthetic matrix now covers wider noise, asymmetric/broad contamination and placement sensitivity. The next task is the first non-productive high-boundary retrieval comparison against method v4 with strict lower-column preservation analysis; no threshold, estimator, aggregation width, ensemble or method-v5 promotion is authorized yet.
