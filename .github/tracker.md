# MILGRAU scientific engineering tracker

Branch: `new-architecture`
Tracker snapshot: 2026-09-17

This is the active scientific/engineering source of truth. It records current productive identity, accepted evidence, unresolved R&D and the next evidence gate. Detailed history stays in Git, tests and `docs/regression_baselines/`.

---

# 0. Rules that must not drift

* Separate analytical/synthetic truth, real-data regression, external-chain comparison and instrument characterization.
* Real lidar measurements are observational evidence, not aerosol optical ground truth.
* Missing uncertainty is never zero uncertainty; unsupported bins remain unsupported.
* Backward KFS support cannot extend above the accepted boundary or jump internal invalid gaps.
* High altitude is an objective, not permission to weaken QA.
* Long averaging must expose temporal contribution; aggregation must expose resolution loss.
* A fitted Rayleigh-window boundary is a new retrieval assumption.
* Clean center bin != clean molecular window; narrow Monte Carlo spread != absence of bias.
* Thresholds are not chosen to reach a desired altitude.
* Productive semantic changes require deliberate method/schema/provenance/baseline versioning.

---

# 1. Productive Level 2 identity

* Product schema: **v3** — auditable Rayleigh candidate catalogue.
* Retrieval method: **v4** — QA-first candidate selection + backward Klett–Fernald–Sasano.
* Productive boundary: exact measured RCS bin at the selected reference altitude.
* Candidate policy: enumerate complete windows -> minimum QA -> rank accepted candidates by `relative_slope + relative_variance`; lower grid index breaks ties.
* Rayleigh SNR is diagnostic only.
* Aerosol extinction is conditional on assumed aerosol lidar ratio.
* Productive optical support is inversion support, not `isfinite(product)`.

Not productive: fitted boundary, high-column backbone, productive vertical aggregation, smoothing as support extension, hard SNR/cloud/temporal gates, overlap cutoff, physical PC saturation threshold, multi-reference ensemble, cascade/stitching and model-implied wavelength-dependent molecular lidar ratio.

---

# 2. Observational baseline policy

Primary case: `20251107sapm`.

## Historical derived evidence

Historical L2 SHA-256 `32b793fc7874f850ee38273f7073c957194f87dd08fed1c340e96b976917a212`, source L1 `52663cbfd7863db6d38bc6596616c672c573c19bc55a3c5e72403abbc4ba0b3f`, revision `b68e4d812a37c2a003e49e30c15c2e112e4c8360`.

The original historical NetCDF was lost during workstation migration. Its derived summaries remain historical regression evidence and are not rewritten.

## Active reproducible baseline

* L2 SHA-256: `bba3b454387ef34c9b4bee3b0ab69e286e611cc44f9a4a6358b779f6d9f236d8`
* source L1 SHA-256: `7b49c9deb93e89b103dba95455d854fcebb97b81292a18269edd1bea7a5ee342`
* product source revision: `fa1cae20ec1ac28bf2955deb5a93f9a3ed56e663`
* product source-code SHA-256: `091d733499a9dd3026af6f7869ea1db2a1e57546ebde9d58c2009985ddb033e5`
* schema/method: 3 / 4
* candidate slots: 25,340 evaluated; 16,331 accepted; 10 productively selected
* temporal profile-count weights independently reconstructed from embedded timestamps: **23 / 39 / 39 / 40 / 26**
* aggregate reference altitudes: ~6078.75 m (355), ~5846.25 m (532)
* aggregate inversion tops: ~6198.75 m (355), ~6281.25 m (532)

Active evidence:

* `docs/regression_baselines/20251107sapm_current_head_regeneration_preliminary.json`
* `docs/regression_baselines/20251107sapm_active_baseline_p54_summary.json`
* `docs/regression_baselines/20251107sapm_first_high_boundary_experiment.json`

Policy: historical evidence is preserved; the current checksum-identified product is the active reproducible baseline. No input-equivalence claim is made.

---

# 3. Status overview

| Priority | Status | Meaning |
| --- | --- | --- |
| P0–P2 | COMPLETE | productive architecture/support/engineering guardrails |
| P3 | IN PROGRESS | FAIR/release/license/traceability |
| P4 | PARALLEL EVIDENCE | instrument characterization / external comparisons |
| P5.0–P5.2 | FROZEN FOUNDATION | baselines, support semantics, QA-first catalogue |
| P5.3 | IN PROGRESS | real candidate interpretation |
| P5.4 | **ACTIVE PRIMARY R&D** | high-column boundary/support evidence |
| P5.5 | PENDING | ensemble only if P5.4 establishes need |
| P5.6–P5.7 | DEFERRED | cascade / stitching |
| P5.8 | PARALLEL R&D | molecular semantics / uncertainty consistency |
| P5.9 | IN PROGRESS | QA/schema/provenance presentation |
| P5.10 | IN PROGRESS | validation matrix |
| P6 | PENDING | reproducible release/publication |

---

# 4. P5.4 evidence vector — OBSERVATIONAL GATE POPULATED

`HighColumnEvidence` keeps separate: shape QA, bin SNR, window SNR dependence limits, exact-window temporal persistence, dominant contribution, subwindow disagreement, contamination status, resolution, estimator identity and dependence model. There is no composite score/pass/fail or preferred target altitude.

Completed on the active baseline:

* [x] all 25,340 candidate slots exported/analyzed;
* [x] 16,331 accepted and 10 selected reconciled;
* [x] block weights reconstructed independently;
* [x] compact auditable evidence frozen;
* [x] altitude distributions inspected;
* [x] diagnostic redundancy assessed.

Key findings:

* independent-bin window SNR is roughly 10–12x single-bin SNR, but the fully-correlated limit is close to single-bin SNR; precision gain is dependence-model contingent;
* bin SNR and window SNR are strongly rank-correlated through most of 10–20 km and are partly redundant noise-strength indicators;
* exact-window persistence and subwindow disagreement expose different failure modes;
* persistence of the same window falls with altitude even when some high candidate exists in every block;
* small subwindow disagreement cannot certify molecular purity because broad contamination can bias both halves together.

No SNR threshold, score, altitude target or method-v5 change is authorized.

---

# 5. P5.4 synthetic robustness — EXPANDED AND CI GREEN

`tests/test_rayleigh_window_evidence_matrix_rnd.py` adds controlled cases for:

* independent clean noise sweep from 1% to 30%;
* asymmetric contamination -> biased fit + large half-window disagreement;
* broad symmetric contamination -> >20% fit bias while half-window disagreement stays <1%;
* boundary/window placement sensitivity over fixed contamination.

Implementation commit: `7c2b449350fc84a3c06d28de93d99a8044782951`.

CI run `35239430137` passed with this matrix included.

Still open synthetically: correlated-noise families beyond independent/full-correlation brackets, leave-part-out diagnostics, temporal reject/split tests, retrieval-level lower-column truth preservation and future high-column graceful support failure under controlled truth.

---

# 6. First offline high-boundary experiment — COMPLETED, NON-PRODUCTIVE

Evidence: `docs/regression_baselines/20251107sapm_first_high_boundary_experiment.json`.

Before changing any boundary, the offline analysis exactly reproduced persisted productive method-v4 block backscatter means and standard deviations for all 10 block/wavelength cases (zero relative L2 difference on supported bins). This validates the comparison path.

Three explicit probes were used; none is a productive gate:

1. **Moderate probe**: candidate accepted in all five blocks nearest 10 km — 10001.25 m at both wavelengths.
2. **Support-edge probe**: highest all-five accepted candidate whose nominal exact backward path is complete in every block — 12176.25 m at 355 and 12393.75 m at 532.
3. **Extreme-high graceful-failure probe**: highest all-five accepted candidate regardless of path completeness — 15386.25 m at 355 and 18873.75 m at 532.

## Moderate probe

Median fitted/exact boundary-signal ratio: 1.019 (355), 1.051 (532).

Median 0–6 km aerosol-backscatter relative L2 difference versus method v4:

* exact high boundary: 2.8% (355), 12.6% (532);
* fitted-window boundary: 0.9% (355), 6.8% (532).

These are sensitivity differences, **not truth errors**; being closer to method v4 does not validate an estimator.

Monte Carlo completeness across blocks:

* 355: worst exact 99.67%, fitted 96.67%; under current productive semantics requiring all requested simulations valid, at least one 355 case would fail;
* 532: exact and fitted remained 100% complete for the 10-km probe.

## Support-edge probe

The nominal path is complete, but uncertainty perturbations make it highly fragile:

* 355 worst complete MC fraction: exact 19.3%, fitted 0.33%;
* 532 worst complete MC fraction: exact 13.3%, fitted 0.33%.

Therefore nominal path completeness alone is insufficient; perturbation robustness becomes a major support constraint near ~12 km in this event.

## Extreme-high graceful failure

All five blocks accept the local candidate window, yet **zero blocks have a complete nominal backward path**.

* 355 at 15386.25 m: the exact center itself is non-positive in 2/5 blocks and additional path gaps exist.
* 532 at 18873.75 m: the center is positive in all blocks, but internal non-positive path bins still break every backward inversion.

This is direct observational evidence that:

**local candidate QA + temporal persistence != usable KFS boundary.**

No gap filling/interpolation was introduced. Failure is retained as scientific support information rather than forced retrieval.

Decision: do **not** promote a higher/fitted boundary or method v5 from this event.

---

# 7. Current P5.4 interpretation

The first experiment changes the research emphasis. The immediate limitation is not merely choosing a better high candidate; it is the combination of:

* contiguous backward-path support;
* weak-signal perturbation robustness;
* dependence/covariance assumptions;
* contamination/model bias;
* preservation of the lower-column solution.

A fitted window may reduce random single-bin sensitivity, but it cannot repair an invalid path and broad contamination may remain precise but biased.

Vertical aggregation remains a legitimate parallel R&D option because it may reduce weak-signal sign failures, but it must be treated as a declared resolution trade and tested against lower-column truth before any promotion.

---

# 8. Validation matrix

Synthetic established: molecular-only/aerosol recovery, vertical-grid convergence, missing uncertainty/gap semantics, QA-first catalogue, temporal persistence/transience, vertical aggregation trade, clean fitted-window denoising, contaminated-window bias, independent/correlated fitted-boundary MC, molecular-lidar-ratio sensitivity, wider noise, asymmetric/broad contamination and placement sensitivity.

Real SPU established: active method-v4/schema-v3 baseline, catalogue, temporal evidence, SNR/dependence brackets, exact-vs-window evidence, populated high-column vector and first non-productive high-boundary comparison.

Still required before any method-v5 proposal:

* controlled correlated-noise families / placement / graceful support-failure truth tests;
* at least one materially different SPU regime;
* clear/high-aerosol/cloud/weak-signal/temporally-changing/different AN-PC cases;
* lower-column truth preservation for any proposed backbone/aggregation strategy;
* explicit resolution accounting and uncertainty/dependence provenance;
* matched LPP/SCC/ELDA comparison when feasible.

P5 success means defensible inversion-supported coverage with honest uncertainty, temporal representativeness, resolution and exact provenance — not “reach 20 km”.

---

# 9. Parallel P3/P4/P5.8/P5.9/P6 work

P3: choose/add license, align CFF/package/release metadata, finish Level 2 metadata review, update `docs/scientific_traceability.md` to schema 3 / method 4, improve lidar-ratio provenance.

P4: overlap/telecover evidence, physical PC saturation, gluing fit covariance/materiality, additional AN/PC regimes, real cloud/layer validation and external-chain comparison.

P5.8: keep productive molecular lidar ratio at `8*pi/3` until actual receiver/filter molecular semantics are resolved.

P5.9: QA must visually distinguish diagnostic finite high-altitude quantities from supported aerosol retrieval; candidate density is not validation; R&D panels stay labeled R&D.

P6: reproducible environment/build artifact tests, warnings/coverage, release metadata, branch reconciliation/protection and immutable scientific tags remain pending.

---

# 10. Immediate next gate

Proceed in this order:

1. Quantify **why** high-boundary Monte Carlo paths fail near 10–12 km: identify weak/non-positive bins, their altitude distribution and whether failures are dominated by isolated bins or extended regions.
2. Test whether declared pre-retrieval vertical aggregation (60/120 m R&D only) reduces those path failures while preserving known synthetic lower-column truth and exposing effective resolution.
3. Add correlated-noise and leave-part-out contamination diagnostics where they materially change the conclusion.
4. Repeat the evidence/high-boundary logic on at least one materially different SPU event before any threshold/rule.
5. Only after those gates decide whether a method-v5 high-column backbone is scientifically justified.
6. Multi-reference ensemble remains after that decision; cascade/stitching remain deferred.

---

# 11. Stop conditions

Stop/pause an R&D branch if it cannot be separated from noise/model bias, only one favorable event supports it, it improves altitude while materially degrading the lower column, it needs an arbitrary target-driven threshold, uncertainty dependence is unknown, it hides intermittency/resolution loss, it relies on uncharacterized instrument behavior or a simpler experiment answers the question first.

---

# 12. Current handoff

MILGRAU productive Level 2 remains schema v3 / method v4. The active reproducible `20251107sapm` baseline now has a populated P5.4 evidence vector and a completed first non-productive high-boundary comparison. The key new result is that accepted/persistent high Rayleigh windows do not guarantee a usable KFS boundary: internal non-positive RCS bins break contiguous backward support, and even nominally complete ~12-km paths become highly fragile under the existing Monte Carlo perturbations. A 10-km probe is more stable but already shows wavelength-dependent lower-column sensitivity and, at 355 nm, incomplete MC robustness in at least one block. Window fitting can reduce single-bin random sensitivity but cannot bridge invalid support and remains vulnerable to broad contamination and unknown dependence. The next scientific gate is therefore support-failure characterization plus declared-resolution vertical-aggregation experiments, followed by replication on a contrasting SPU case. No high-column threshold, fitted boundary, aggregation width, ensemble, cascade or method-v5 promotion is authorized.
