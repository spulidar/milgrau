# MILGRAU scientific engineering tracker

Branch: `new-architecture`
Tracker snapshot: 2026-09-17

This file is the active scientific/engineering source of truth for current MILGRAU development.

It is intentionally **not** a complete project history. Completed implementation history remains recoverable from Git, tests and frozen evidence under `docs/regression_baselines/`.

This tracker exists to answer five questions clearly:

1. What is productive today?
2. What scientific assumptions are currently accepted?
3. What remains R&D or unvalidated?
4. What is the next evidence gate?
5. What must happen before an R&D result can change the productive method?

---

# 0. How to use this tracker

## For future development / AI handoff

Before proposing or implementing a scientific change:

1. Read **Current productive identity** and **Frozen decisions**.
2. Read `milgrau/scientific.py` and `docs/level2_schema.md` for current method/schema identity.
3. Read the evidence files referenced by the relevant active gate.
4. Distinguish:

   * analytical/synthetic truth;
   * observational regression evidence;
   * external-chain comparison;
   * instrument characterization.
5. State the scientific question being tested before changing code.
6. Do not convert an R&D diagnostic into a threshold, score or productive rule merely because it appears useful in one case.
7. If productive semantics change, explicitly decide whether retrieval-method version, product-schema version, metadata and regression baselines must change.
8. When closing an R&D task, record:

   * finding;
   * interpretation;
   * decision;
   * evidence file/test/commit;
   * next gate if unresolved.

Avoid vague items such as “improve high altitude”, “investigate uncertainty” or “optimize QA”. Every active item should have an observable question or exit condition.

---

# 1. Current productive identity

## Canonical Level 2 state

* Level 2 product schema: **v3**

  * auditable Rayleigh candidate catalogue.
* Level 2 retrieval method: **v4**

  * QA-first Rayleigh candidate selection;
  * backward Klett–Fernald–Sasano;
  * native vertical grid;
  * exact measured RCS bin at the selected reference altitude remains productive `X_ref`.
* Productive Rayleigh ranking:

  * enumerate all complete candidates;
  * apply configured minimum QA;
  * rank accepted candidates by `relative_slope + relative_variance`;
  * lower grid index is deterministic tie-breaker.
* Propagated Rayleigh SNR is diagnostic only.
* Aerosol extinction remains conditional on assumed aerosol lidar ratio.
* Productive optical support is backward-inversion support, not generic `isfinite()` support.

## Explicitly **not productive**

The following are currently R&D, diagnostic-only or uncharacterized and must not be silently enabled:

* fitted/window-derived KFS boundary;
* high-column backbone;
* vertical aggregation for productive retrieval;
* post-retrieval smoothing as support extension;
* hard Rayleigh SNR threshold;
* cloud/layer rejection threshold;
* temporal persistence threshold;
* lower overlap/instrument cutoff;
* physical PC saturation threshold;
* multi-reference ensemble;
* cascaded retrieval;
* solution stitching/merge;
* model-implied wavelength-dependent molecular lidar ratio.

## Current development snapshot

* Active branch: `new-architecture`.
* Tracker HEAD:
  `7fa2760101281d01059e3a20b60cc08b3d92cde5`.
* Package CalVer remains `2026.9`; development identity must therefore also use source/repository provenance.
* Cross-platform CI is green on the current development state.
* Deprecation warnings remain technical debt.
* `new-architecture` is effectively the scientific development line and must be deliberately reconciled with `main` before formal release.
* Branch protection / required checks remain a P6 governance item.

This commit snapshot is descriptive only. Scientific identity is defined by the versioned method/schema/provenance contracts.

---

# 2. Frozen observational reference state

Primary observational regression case:

`20251107sapm`

Frozen schema-v3 Level 2 product:

`20251107sapm_level2_optical.nc`

SHA-256:

`32b793fc7874f850ee38273f7073c957194f87dd08fed1c340e96b976917a212`

Source revision:

`b68e4d812a37c2a003e49e30c15c2e112e4c8360`

Important interpretation:

**This measurement is observational regression/behavior evidence, not aerosol ground truth.**

Current catalogue evidence:

* 25,340 / 25,340 candidate slots evaluated;
* 16,224 candidates pass enabled minimum QA;
* current selected references remain approximately 5.6–6.3 km;
* shape-QA-passing candidates can exist above 20 km;
* candidate acceptance therefore does not imply a trustworthy high-altitude KFS boundary;
* finite scattering ratio above the accepted backward boundary is not supported aerosol retrieval.

Frozen evidence includes:

* `20251107sapm_method_v3.json`
* `20251107sapm_method_v4_schema3_catalogue.json`
* `20251107sapm_temporal_support_preliminary.json`
* `20251107sapm_vertical_aggregation_preliminary.json`
* `20251107sapm_rayleigh_window_uncertainty_preliminary.json`
* `20251107sapm_exact_vs_window_boundary_preliminary.json`
* `high_column_candidate_persistence_comparison.json`
* `molecular_lidar_ratio_semantics_rnd.json`

Historical `20241219nt` remains useful as a contrasting observational case, especially for temporal intermittency, but its legacy retrieval output is not ground truth.

---

# 3. Non-negotiable scientific rules

* `config.yaml` owns processing/scientific recipe.
* `station.yaml` owns station/instrument reality, configuration history and calibration evidence.
* Python owns generic equations and implementations.
* Missing uncertainty is never zero uncertainty.
* Dependence/covariance assumptions must be explicit.
* Unsupported bins remain unsupported.
* No interpolation/filling is introduced merely to extend apparent coverage.
* `isfinite(product)` is not scientific support.
* Backward optical support never extends above that retrieval's accepted boundary.
* Internal invalid gaps cannot be jumped.
* Near-range finite KFS values are not validated instrument support while overlap remains uncharacterized.
* High altitude is an objective, not permission to weaken QA.
* Long averaging must expose temporal contribution and stationarity.
* Vertical aggregation is a declared information/resolution trade, not creation of new information.
* Post-retrieval cosmetic smoothing cannot create scientific support.
* A fitted Rayleigh-window boundary is a new retrieval assumption.
* Clean center bin != clean molecular window.
* Narrow Monte Carlo spread != absence of systematic/model bias.
* The same noisy samples must not be counted twice as independent signal and fitted-boundary uncertainty.
* Multiple references are sensitivity/ensemble evidence, not independent truths.
* Thresholds are not invented because they produce convenient altitude coverage.
* Product metadata must not claim a convention or physical calibration that has not been demonstrated.

---

# 4. Evidence hierarchy

### 1. Analytical / synthetic truth

Used to test equation correctness, controlled support behavior, estimator bias, uncertainty propagation and known physical states.

Primary evidence for algorithm correctness.

### 2. Real-data regression

Used to test whether the actual SPU processing path behaves consistently on representative measurements.

Real measurements do not provide exact aerosol optical ground truth.

### 3. External-chain comparison

LPP / SCC / ELDA comparisons test interoperability and methodological consistency under matched assumptions.

Agreement does not make another chain the truth standard.

### 4. Instrument characterization

Required for detector saturation, dead-time ordering, overlap, calibration history and other instrument-specific physical claims.

A plausible value or historical default is not equivalent to characterization.

---

# 5. Frozen decisions until new evidence exists

These decisions should not be reopened casually during unrelated work.

## Productive KFS

Keep backward KFS as the productive elastic retrieval.

Forward/two-sided branches remain research diagnostics unless a separate scientific case is established.

## Rayleigh selection

Keep method-v4 QA-first candidate selection.

Do not add altitude preference or hard SNR rejection from the current single-event evidence.

## High-column retrieval

Do not promote fitted boundary, vertical aggregation, long-mean backbone or candidate ensemble yet.

First establish contamination robustness, temporal representativeness and lower-column preservation.

## Overlap

Keep `diagnostic_only_no_correction`.

No lower cutoff is scientifically validated yet.

## Photon-counting saturation

Keep physical saturation `not_characterized`.

Numerical dead-time denominator clipping is not physical detector saturation.

## Molecular lidar ratio

Do not replace `8π/3` in productive KFS until actual receiver/filter molecular semantics are resolved and real-SPU sensitivity is evaluated.

## External metadata conventions

Do not declare formal CF or other convention compliance before deliberate checker-based validation against a named released convention.

---

# 6. Status overview

| Priority  | Status             | Current meaning                                                    |
| --------- | ------------------ | ------------------------------------------------------------------ |
| P0–P2     | COMPLETE           | productive architecture, support semantics, engineering guardrails |
| P3        | IN PROGRESS        | FAIR/release metadata and licensing                                |
| P4        | PARALLEL EVIDENCE  | instrument and observational characterization                      |
| P5.0–P5.2 | FROZEN FOUNDATION  | baselines, inversion support, QA-first catalogue                   |
| P5.3      | IN PROGRESS        | interpretation of real candidate catalogue                         |
| P5.4      | ACTIVE PRIMARY R&D | high-column information / boundary / temporal evidence             |
| P5.5      | PENDING            | multi-reference ensemble                                           |
| P5.6–P5.7 | DEFERRED           | cascade / solution stitching                                       |
| P5.8      | PARALLEL R&D       | molecular semantics + uncertainty consistency                      |
| P5.9      | IN PROGRESS        | QA/schema/provenance presentation                                  |
| P5.10     | IN PROGRESS        | validation matrix                                                  |
| P6        | PENDING            | reproducible release/publication                                   |

**Primary scientific focus remains P5.4.**

Do not start P5.5/P5.6 simply because implementation is interesting. They exist only if P5.4 evidence demonstrates a remaining scientific need.

---

# 7. P3 — FAIR / release hardening

P3 blocks formal release completion but does not block P4/P5 research.

* [x] Code authors/copyright holders identified.
* [ ] Deliberately select software license.
* [ ] Add root `LICENSE`.
* [ ] Align license metadata in package/CFF/release metadata.
* [ ] Keep software/documentation/observational-data licensing distinctions explicit.
* [x] Adopt readable/self-describing NetCDF metadata policy.
* [ ] Perform representative Level 2 metadata review and correct genuine ambiguity.
* [ ] Correct documentation drift where `docs/scientific_traceability.md` still describes the productive baseline as retrieval method v3; canonical state is schema 3 / method 4.
* [ ] Ensure all scientific traceability entries reflect current catalogue/method-v4 ownership.
* [ ] Add provenance for the SPU aerosol lidar-ratio climatology beyond “migrated from legacy config”:

  * source data;
  * retrieval method;
  * period;
  * sample population;
  * filtering/selection;
  * publication/DOI or internal evidence identifier when available.
* [ ] Decide what historical calibration information is scientifically defensible versus merely inherited.

---

# 8. P4 — instrument / observational evidence

## Overlap

* [x] Generic diagnostic geometry exists.
* [x] Current estimates are station-owned and explicitly unvalidated.
* [x] No productive correction/cutoff is inferred.
* [ ] Experimentally determine receiver FOV convention, field stop, beam diameter convention, wavelength-dependent divergence, alignment/separation and temporal stability.
* [ ] Validate with telecover/alignment mapping.
* [ ] Prefer independent horizontal and/or Raman-based evidence when available.

## Photon counting

* [x] Preserve raw observed rate.
* [x] Separate numerical dead-time clipping from physical saturation.
* [x] Preserve dark-current shot information and uncertainty.
* [x] `20251107sapm` correction-order sensitivity quantified.
* [ ] Characterize actual physical saturation using AN/PC overlap and preferably controlled attenuation.
* [ ] Revisit dark-current vs nonlinear dead-time ordering only with broader instrument evidence.

## Gluing

* [x] Productive window selection and diagnostics are explicit/versioned.
* [x] Measurement-noise propagation through fade weights is implemented.
* [ ] Quantify fitted slope/intercept uncertainty and covariance.
* [ ] Decide whether that contribution is material enough for productive uncertainty propagation.
* [ ] Test representative regimes with differing AN/PC dominance.

## Other observational characterization

* [ ] Broaden propagated-error SNR evidence before any hard threshold.
* [x] Demonstrate synthetic failure of simple anomaly detection for broad smooth contamination.
* [ ] Validate cloud/layer observables on real events before productive veto.
* [ ] Compare representative products with LPP/SCC/ELDA under matched assumptions.

---

# 9. Frozen P5 foundation — P5.0 to P5.2

These stages are complete and should generally be treated as invariants rather than active work.

* **P5.0 — baselines / legacy audit:** frozen observational baselines exist; synthetic truth is kept distinct from observational regression; legacy ideas were retained only as hypotheses; historical numerical high-altitude coverage is not interpreted as supported optical retrieval.
* **P5.1 — inversion support:** backward support is altitude-resolved, contiguous and cannot bridge invalid gaps. For `20251107sapm`, aggregate supported tops are approximately 6101 m at 355 nm and 6319 m at 532 nm. Near-range algorithmic support is not equivalent to validated overlap support.
* **P5.2 — Rayleigh catalogue:** schema v3 enumerates and preserves all complete candidate windows and their diagnostics. Productive method v4 follows `enumerate -> minimum QA -> rank accepted only`, using `relative_slope + relative_variance`, with no SNR gate or altitude preference.

Do not reopen these stages unless new evidence contradicts an invariant or a future method/schema change deliberately supersedes them.

---

# 10. P5.3 — real-data candidate interpretation

Current main finding:

**The present shape QA is useful as minimum consistency QA but is not sufficient to establish a high-column KFS boundary.**

For `20251107sapm`, approximate median accepted-candidate bin-wise SNR:

| altitude | 355 nm | 532 nm |
| -------- | -----: | -----: |
| 5–10 km  |   14.8 |   15.9 |
| 10–15 km |   3.13 |   3.59 |
| 15–20 km |   1.20 |   1.32 |
| 20–25 km |   0.90 |   0.87 |

Therefore:

* high candidate existence != high boundary trustworthiness;
* shape pass != supported aerosol retrieval;
* single-bin boundary noise is a serious high-column limitation;
* window-level information may be materially stronger than one-bin information;
* contamination/model bias remains a separate problem from random uncertainty.

Open:

* [ ] Any future redesigned retrieval must be compared against the frozen lower-column baseline before promotion.
* [ ] No target altitude such as 15 or 20 km is an acceptance criterion by itself.

---

# 11. P5.4 — ACTIVE PRIMARY R&D: defensible high-column information

## Scientific question

Can MILGRAU use more of the information already present in the measurement to extend backward optical retrieval support **without**:

* biasing the established lower column;
* hiding temporal nonstationarity;
* mistaking contamination for molecular signal;
* overstating resolution;
* double-counting uncertainty;
* turning one favorable event into a general threshold?

This is the main current research question.

---

## 11.1 Temporal evidence

Established:

* [x] Block contribution/support diagnostics exist.
* [x] Dominant contribution fraction exists.
* [x] Candidate persistence exists.
* [x] `20251107sapm` high candidates are temporally persistent across all five blocks.
* [x] `20241219nt` demonstrates a contrasting transient high-column case.

Interpretation:

A long-mean high-altitude candidate can only represent the measurement if its contributing time intervals are themselves documented.

Open:

* [ ] Define a temporal decision rule only after more heterogeneous synthetic and real cases.
* [ ] Any future long-mean backbone must retain native block state and contribution information.

---

## 11.2 Vertical aggregation R&D

Established:

* [x] Strict pre-retrieval non-overlapping aggregation exists.
* [x] Independent and fully-correlated uncertainty limits are explicit.
* [x] Real far-range short-lag residual correlation is modest in `20251107sapm`.
* [x] 60–120 m aggregation can improve high-range SNR in this event.
* [x] Synthetic tests show expected resolution loss while preserving coarse-grid KFS truth under current guards.

Interpretation:

Aggregation can trade resolution for precision, but cannot create information.

Open:

* [ ] Expand synthetic narrow/broad layer, noise and covariance cases.
* [ ] Evaluate additional real regimes.
* [ ] Do not select 60 m or 120 m as productive width from `20251107sapm` alone.
* [ ] Any productive width must expose `effective_vertical_resolution_m`.

---

## 11.3 Rayleigh-window boundary R&D

Established:

* [x] Window calibration uncertainty implemented.
* [x] Independent, fully correlated and explicitly supplied covariance assumptions remain distinct.
* [x] Fitted-boundary MC uses the same perturbed samples for the fit and retrieval.
* [x] No boundary-fit noise is double-counted.
* [x] Clean synthetic fitted boundary reduces exact-bin noise.
* [x] Contaminated windows can produce precise but biased fitted boundaries.
* [x] Positive correlation reduces naive averaging gain.
* [x] Exact-bin/window-fit disagreement becomes large at high altitude.

Approximate observed median absolute exact-vs-fit disagreement:

* 5–10 km: ~4–5%;
* 10–15 km: ~19–23%;
* 15–20 km: ~43–47%;
* 20–25 km: ~47–50%.

Interpretation:

The exact single bin becomes a noisy estimator of local molecular scale aloft, but replacing it with a window estimate is not justified until contamination/model bias is controlled.

Open next:

* [ ] Extend synthetic noise amplitude range.
* [ ] Add asymmetric contamination.
* [ ] Add broad smooth contamination.
* [ ] Shift contaminating structure relative to candidate center.
* [ ] Vary reference placement within otherwise similar windows.
* [ ] Evaluate subwindow disagreement / leave-part-of-window-out behavior.
* [ ] Compare current unweighted estimator with weighted/robust alternatives only as R&D.
* [ ] Do not choose an estimator because it reaches a higher altitude.
* [ ] Validate promising contamination diagnostics in additional real regimes.

---

## 11.4 High-column evidence vector

Implemented diagnostic record:

`HighColumnEvidence`

Fields remain physically separate:

* shape-QA state;
* candidate bin-wise SNR;
* window calibration SNR under multiple dependence assumptions;
* temporal persistence;
* dominant signal/block contribution;
* subwindow relative disagreement;
* contamination fraction/diagnostic;
* effective vertical resolution;
* boundary estimator identity;
* noise/dependence model.

Critical design decision:

**There is currently no composite score, no high-column pass/fail and no preferred target altitude.**

Open next:

* [x] Implement an offline helper joining schema-v3 candidate slots to `HighColumnEvidence`: `milgrau/level2/high_column_export.py` and `milgrau/cli/high_column_evidence.py`.
* [ ] Execute the helper on the checksum-matched frozen `20251107sapm` NetCDF; individual observational records have not yet been exported.
* [ ] Persist/export the analysis in an auditable table/summary.
* [ ] Inspect how evidence dimensions behave with altitude before defining any decision rule.
* [ ] Determine which diagnostics are redundant and which identify genuinely different failure modes.
* [ ] Do not replace missing evidence with favorable defaults.

---

### Offline export implementation — 2026-09-17

* **Question:** can each persisted candidate be joined to separate uncertainty,
  temporal and subwindow diagnostics without changing selection or inventing
  missing evidence?
* **Implementation:** CSV/strict-JSON candidate tables plus altitude-stratified
  summaries; original selected/rejected/unevaluated states are retained. Explicit
  block weights, resolution declaration, input SHA-256 and exporter source
  identity accompany the analysis. See `docs/high_column_evidence_export.md`.
* **Evidence:** `tests/test_high_column_export.py` covers analytical clean-window
  SNR, unequal temporal weights, rejected/unevaluated slots, missing error,
  exact native geometry, export round trip, input checksum and no-overwrite.
  Local syntax/whitespace checks passed; full pytest/Ruff validation is pending
  repository CI because scientific/test dependencies were unavailable locally.
* **Finding:** source code and aggregate frozen summaries are available, but the
  original NetCDF is not in the repository/workspace. Aggregates cannot recover
  per-candidate evidence. No new observational finding is claimed.
* **Decision / scope:** diagnostic-only infrastructure; the observational gate
  remains open. Contamination and empirical covariance diagnostics remain null.
  Exact-window persistence is explicitly distinct from availability of any
  candidate above a target altitude. Subwindow agreement does not prove purity.
* **Versioning:** productive schema 3 / method 4 unchanged; no new thresholds,
  estimator promotion or resolution choice.
* **Next gate:** obtain the frozen NetCDF, run the documented checksum-guarded
  command, reconcile counts/selected references and inspect distributions before
  any experimental high-boundary retrieval.

---

## 11.5 First offline high-boundary retrieval experiment

Only begin once the evidence vector can explain candidate state.

The first experiment remains **non-productive**.

For selected experimental candidate(s):

* use existing schema-v3 catalogue;
* do not modify productive method v4;
* record exact candidate evidence;
* run experimental boundary estimator;
* compare with productive method-v4 result.

Required comparison:

### Lower column

Evaluate at minimum approximately 0–6 km:

* aerosol backscatter difference;
* aerosol extinction difference;
* uncertainty change;
* support change;
* profile structure.

### High column

Record:

* supported top;
* uncertainty growth;
* candidate/boundary SNR;
* temporal evidence;
* resolution;
* contamination diagnostics.

### Acceptance logic

A higher retrieval top is **not** sufficient.

The experiment is interesting only if extra supported information is obtained without material degradation/bias of the established lower-column solution.

No permanent percentage tolerance should be invented before the comparison distribution is understood.

---

## 11.6 Minimum evidence before any method-v5 proposal

Do not propose productive high-column semantics until all are available:

* [ ] controlled clean synthetic cases;
* [ ] controlled contaminated synthetic cases;
* [ ] covariance/noise sensitivity;
* [ ] boundary-placement sensitivity;
* [ ] temporal heterogeneity case;
* [ ] `20251107sapm`;
* [ ] at least one additional materially different SPU regime;
* [ ] lower-column preservation analysis;
* [ ] explicit resolution accounting;
* [ ] uncertainty/dependence provenance;
* [ ] documented contamination QA limitations.

If productive semantics change:

* bump retrieval method;
* update `milgrau/scientific.py`;
* update `docs/level2_schema.md`;
* update `docs/scientific_traceability.md`;
* update product metadata;
* add synthetic acceptance tests;
* freeze new observational regression evidence;
* invalidate incompatible incremental products.

---

# 12. P5.5 — multi-reference ensemble — PENDING

Do not begin as the current main task.

Only proceed if P5.4 demonstrates that multiple individually credible references provide useful independent/sensitivity information.

Required future questions:

* [ ] What candidate separation is enough to avoid treating overlapping windows as independent?
* [ ] How is covariance between references represented?
* [ ] How are members weighted?
* [ ] How is reference-choice spread represented?
* [ ] What is effective member count by altitude?
* [ ] Does ensemble behavior preserve lower-column truth in synthetics?

Each member must remain tied to its own exact local boundary semantics.

---

# 13. P5.6 / P5.7 — cascade and solution merge — DEFERRED

These are not next steps.

Consider only if:

1. a defensible backbone exists;
2. multi-reference evidence exists;
3. a scientifically meaningful coverage gap still remains.

Any cascade or merge must:

* propagate inherited uncertainty;
* reject inconsistent handoffs;
* never bridge unsupported gaps;
* expose source/member identity;
* demonstrate continuity without cosmetic smoothing.

---

# 14. P5.8 — molecular model / uncertainty consistency

Established:

* [x] `alpha_mol` = total Rayleigh volume scattering/extinction.
* [x] `beta_mol` = angular 180° molecular backscatter using wavelength-dependent depolarization/phase function.
* [x] Model-implied `alpha_mol/beta_mol` differs from productive `8π/3` by about 1.4–1.5%.
* [x] Synthetic sensitivity demonstrates that matched forward/inverse molecular semantics improve controlled recovery.
* [x] Productive method remains unchanged.

Open:

* [ ] Determine total-Rayleigh vs Cabannes/effective detected molecular component for actual SPU receiver/filter semantics.
* [ ] Document filter spectral response and polarization implications if available.
* [ ] Quantify representative real-SPU retrieval sensitivity.
* [ ] Change productive `S_m` only after the physical receiver semantics are resolved.
* [ ] Any such change requires retrieval-method versioning.

This task runs in parallel with P5.4 and must not block unrelated high-column evidence work unless the expected effect becomes material to the tested conclusion.

---

# 15. P5.9 — Level 2 QA / schema / provenance presentation

Established:

* [x] schema-v3 catalogue is validated structurally;
* [x] inversion support/top/bottom are explicit;
* [x] schema and retrieval-method identities are separated.

Open:

* [ ] QA plots must show optical retrieval support/top explicitly.
* [ ] Finite high-altitude scattering ratio must not visually look like retrieved aerosol optical product.
* [ ] Molecular-fit QA should show selected candidate and candidate acceptance context.
* [ ] Candidate density should not be presented as validation.
* [ ] Temporal, aggregation, fitted-boundary or ensemble panels enter normal QA only when productive; otherwise they must be labeled R&D.
* [ ] Bring `docs/scientific_traceability.md` fully into method-v4/schema-v3 state.

---

# 16. P5.10 — validation matrix

## Synthetic established

* [x] pure molecular -> near-zero aerosol;
* [x] controlled aerosol 355/532 recovery;
* [x] vertical-grid convergence;
* [x] missing uncertainty fails support;
* [x] internal gaps break support;
* [x] QA-first candidate catalogue behavior;
* [x] persistent vs transient temporal support;
* [x] vertical aggregation/resolution trade;
* [x] clean fitted-window denoising;
* [x] contaminated fitted-window bias counterexample;
* [x] independent fitted-boundary MC;
* [x] explicitly correlated fitted-boundary MC;
* [x] molecular-lidar-ratio semantics sensitivity.

Still required:

* [ ] wider noise levels;
* [ ] correlated-noise families;
* [ ] broad/smooth contamination;
* [ ] asymmetric contamination;
* [ ] boundary placement sensitivity;
* [ ] temporal reject/split experiments;
* [ ] multi-reference dependence;
* [ ] future high-column graceful support failure;
* [ ] lower-column truth preservation for any future backbone.

## Real SPU

Established:

* [x] `20251107sapm` method-v4/schema-v3 baseline;
* [x] candidate catalogue;
* [x] temporal support;
* [x] vertical covariance/SNR brackets;
* [x] exact-vs-fitted boundary evidence.

Required next:

* [ ] populated high-column evidence vector;
* [ ] first offline higher-boundary comparison;
* [ ] second contrasting real case;
* [ ] clear-night case;
* [ ] higher-aerosol case;
* [ ] cloud/layer case;
* [ ] weak-signal case;
* [ ] temporally changing case;
* [ ] differing AN/PC dominance;
* [ ] external LPP/SCC/ELDA comparisons under matched assumptions.

P5 success criterion:

**maximize defensible inversion-supported coverage while exposing where and why support ends, with honest uncertainty, temporal representativeness, declared vertical resolution and exact boundary provenance.**

Success is **not** “reach 20 km”.

---

# 17. P6 — reproducible release / publication readiness

## FAIR / release

* [ ] select software license;
* [ ] add root license and package metadata;
* [ ] verify CFF/release version/date/DOI consistency;
* [ ] write release notes for scientific method/schema changes;
* [ ] freeze reproducible environment/constraints;
* [ ] build sdist + wheel in CI;
* [ ] install and test from built artifacts rather than only editable checkout;
* [ ] run end-to-end L0 -> L2 from release artifact/environment;
* [ ] freeze machine-readable synthetic acceptance summaries;
* [ ] freeze observational regression summaries.

## Engineering quality

* [ ] reduce current NumPy/xarray/netCDF deprecation warnings deliberately;
* [ ] define warning policy so new important warnings are visible;
* [ ] generate core-science coverage report;
* [ ] evaluate whether a minimum coverage gate is useful without incentivizing meaningless tests;
* [ ] test package metadata and clean installation.

## Governance

* [ ] reconcile `new-architecture` and `main`;
* [ ] decide final default/release branch strategy;
* [ ] require relevant CI checks before release merges;
* [ ] protect release branch as appropriate;
* [ ] create immutable tags/releases for published scientific states.

Release preparation must not force unfinished P5 R&D into the productive method.

---

# 18. Immediate next gate

Do these in order unless new evidence invalidates the sequence:

1. **Run the implemented offline `HighColumnEvidence` exporter** on the checksum-matched `20251107sapm` schema-v3 NetCDF (not versioned in this repository); see `docs/high_column_evidence_export.md`. The real-data gate remains open.
2. Produce an auditable candidate/evidence summary without creating a composite score or threshold.
3. Extend synthetic fitted-boundary tests with:

   * wider noise range;
   * asymmetric contamination;
   * broad/smooth contamination;
   * reference-placement offsets;
   * subwindow disagreement diagnostics.
4. Use that evidence to choose a small number of experimental candidate boundaries for the **first non-productive high-boundary retrieval comparison**.
5. Compare the experimental result against method v4, especially the established 0–6 km column, uncertainty and supported top.
6. Keep 60–120 m vertical aggregation as a parallel experiment, not a productive choice.
7. Run the same evidence logic on at least one contrasting SPU measurement before deriving thresholds/rules.
8. In parallel:

   * resolve receiver molecular semantics;
   * quantify gluing parameter uncertainty;
   * fix method-v3/method-v4 documentation drift;
   * improve aerosol-lidar-ratio provenance.
9. Only after those gates decide whether a **method v5** high-column estimator/backbone is scientifically justified.
10. Multi-reference ensemble comes after that decision; cascade/solution merge remain deferred.

---

# 19. Stop conditions / anti-scope-creep rules

Pause a line of R&D rather than expanding it when:

* its result cannot be distinguished from noise/model bias;
* only one favorable real event supports it;
* it improves top altitude but materially distorts the lower column;
* it requires an arbitrary threshold chosen to hit a desired altitude;
* its uncertainty assumption is unknown;
* it hides temporal intermittency;
* it hides loss of vertical resolution;
* it relies on uncharacterized instrument behavior;
* a simpler experiment can answer the same scientific question first.

Do not add a new retrieval architecture merely because the previous experiment is incomplete.

---

# 20. Definition of a scientifically complete tracker item

A scientific task is ready to be marked complete when the tracker can state:

**Question:** what was being tested?
**Evidence:** which tests/data/baselines support the result?
**Finding:** what was observed?
**Decision:** what does MILGRAU do because of it?
**Scope:** productive, diagnostic-only, R&D or rejected?
**Versioning:** did method/schema/provenance change?
**Remaining uncertainty:** what is still not known?

If those answers are unavailable, the item is still open even if code exists.

---

# 21. Current one-paragraph handoff

MILGRAU currently has a stable productive Level 2 baseline at schema v3 / retrieval method v4 using QA-first Rayleigh candidate selection and backward KFS with the exact measured RCS bin as the boundary. Altitude-resolved inversion support, candidate provenance and partial uncertainty semantics are explicit and tested. The current primary R&D question is whether higher-altitude information already present in real measurements can be used through fitted-window boundary estimation and/or declared pre-retrieval vertical aggregation without biasing the established lower column, hiding contamination, temporal nonstationarity or resolution loss. `20251107sapm` shows many high-altitude shape-QA candidates but weak single-bin SNR and large exact-vs-window boundary disagreement aloft; fitted-window information is promising but contamination can produce precise biased boundaries. The offline `HighColumnEvidence` exporter is implemented with auditable tables and synthetic tests, but the frozen observational NetCDF was unavailable in this checkout. The immediate task is therefore to run that exporter on the checksum-matched schema-v3 product, inspect the candidate distributions, strengthen contamination/noise synthetic evidence, and only then perform the first offline non-productive higher-boundary retrieval comparison. No high-column threshold, estimator, aggregation width, ensemble, cascade or method-v5 change has yet been authorized.

