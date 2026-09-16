# MILGRAU scientific engineering roadmap

Branch: `new-architecture`

This file is the active source of truth for current scientific/engineering gates. Detailed completed history remains recoverable in Git and in the frozen evidence files under `docs/regression_baselines/`.

Current productive code identity:

- Level 2 schema: **v3** — `auditable_rayleigh_candidate_catalogue`;
- Level 2 retrieval method: **v4** — QA-first Rayleigh candidate selection with the established method-v3 uncertainty/support semantics;
- productive inversion: backward Klett–Fernald–Sasano;
- schema-v3 catalogue code gate: CI green at `72133cfc446ad5b23ac1de5d9242af2d44cab9e0` on Ubuntu/Windows, Python 3.12/3.14, Ruff + full pytest matrix;
- temporal-support/candidate-persistence R&D gate: CI green at `831c5e7eead3bceb00830312bfae72109ecaa506` on the same cross-platform matrix;
- vertical-aggregation uncertainty/covariance R&D gate: CI green at `360f8d7bf4a445b878f939135b44bc2cdb80c04a` on the same cross-platform matrix;
- current observational schema-v3 evidence: `docs/regression_baselines/20251107sapm_method_v4_schema3_catalogue.json`;
- current temporal evidence: `docs/regression_baselines/20251107sapm_temporal_support_preliminary.json` and `docs/regression_baselines/high_column_candidate_persistence_comparison.json`;
- current vertical-aggregation evidence: `docs/regression_baselines/20251107sapm_vertical_aggregation_preliminary.json`.

Frozen method-v3 comparison baseline remains `docs/regression_baselines/20251107sapm_method_v3.json`. Observational files are regression/behavior evidence, not ground truth.

## Non-negotiable scientific/engineering rules

- `config.yaml` owns the processing/scientific recipe; `station.yaml` owns station/instrument reality, calibration state, station-derived observations and explicitly provisional estimates; Python owns generic equations and implementation.
- Missing uncertainty is never zero uncertainty.
- Correlated/model/systematic uncertainty is not silently treated as independent noise.
- Unsupported data remain unsupported; no interpolation/fill is introduced merely to extend retrieval coverage.
- Scientific support is not equivalent to `isfinite(product)`.
- Backward retrieval never implies support above its accepted exact boundary.
- The lower scientific-support boundary requires evidence-backed instrument validity; finite near-range KFS output alone is insufficient while overlap remains uncharacterized.
- A high-altitude target is an objective, not permission to extrapolate or weaken QA.
- Long averaging must expose temporal contribution/stability and must not let a transient interval silently define an entire measurement.
- Vertical smoothing/aggregation is a resolution trade, not new information; its effective resolution and uncertainty dependence model must be explicit.
- Post-retrieval cosmetic smoothing must never be used to claim additional inversion support.
- Multiple accepted reference solutions are sensitivity/ensemble evidence, not independent truths; their disagreement must remain visible.
- A KFS member is tied to the signal/molecular state at its exact local boundary.
- Elastic extinction remains conditional on the assumed aerosol lidar ratio.
- Diagnostic instrument models never silently become productive corrections.
- No physical PC saturation threshold, overlap cutoff, SNR threshold, cloud threshold, temporal-stability threshold or scientific tolerance is invented merely to close a gate.
- NetCDF products should be readable and self-describing, but MILGRAU currently makes no formal external metadata-convention conformance claim.

## Status overview

| Priority | Status | Purpose |
| --- | --- | --- |
| P0 | COMPLETE | truthful backward-KFS identity and support semantics |
| P1 | COMPLETE + REAL-DATA VALIDATED | canonical Level 2 architecture |
| P2 | COMPLETE | engineering guardrails and cross-platform CI |
| P3 | IN PROGRESS — RELEASE/FAIR GATE | license decision + representative metadata review |
| P4 | PARALLEL EVIDENCE WORK | instrument characterization / observational validation |
| P5.0 | COMPLETE | frozen evidence + historical audit + physical Rayleigh-window geometry |
| P5.1 | COMPLETE + REAL-DATA CHECKED | altitude-resolved inversion support |
| P5.2 | COMPLETE + REAL-DATA CHECKED | QA-first Rayleigh catalogue + auditable selection |
| P5.3 | IN PROGRESS | real-data candidate interpretation |
| P5.4 | IN PROGRESS — TEMPORAL + VERTICAL-SNR R&D | temporally honest high-column backbone R&D |
| P5.5+ | PENDING / DEFERRED | ensemble, optional cascade, merge, uncertainty extension |
| P6 | PENDING | reproducible release/publication process |

## P3 — release / FAIR hardening

### Software licensing

- [x] Code copyright holders identified by the project owner as the four code authors.
- [ ] Select the software license deliberately. The earlier provisional BSD-3-Clause choice was withdrawn and is not current project policy.
- [ ] After selection, add canonical root `LICENSE` and align `CITATION.cff` and package metadata.
- [ ] Keep software, documentation and observational-data licensing separate unless explicitly decided otherwise.

### NetCDF metadata readability

- [x] Adopt `docs/netcdf_metadata_policy.md`: readable/self-describing products without an unsupported convention-conformance claim.
- [x] Keep physical units where meaningful, explicit unit status for instrument-native quantities, numeric machine-readable flags, honest missing values and provenance.
- [ ] Record a representative Level 2 metadata review and fix genuine ambiguity/inconsistency.

P3 blocks release/FAIR completion; it does not block scientific P4/P5 development.

## P4 — instrument characterization / observational validation

### Overlap

- [x] Generic `coaxial_uniform_disk_geometric_v1` diagnostic model implemented; SPU-specific estimates remain station-owned.
- [x] Current provisional geometry is consistent with a reported ~500 m full-overlap scale under the current inferred assumptions.
- [x] Productive policy remains `diagnostic_only_no_correction`; no validated lower retrieval-support cutoff is inferred from the model.
- [ ] Experimentally determine receiver FOV/convention, field-stop diameter, beam diameter convention, wavelength-dependent divergence, alignment/separation and temporal stability.
- [ ] Validate with telecover/alignment mapping and preferably an independent horizontal/Raman-based overlap method before productive correction/support use.

### Photon-counting dead time / saturation

- [x] Preserve raw observed PC rates before dark-current subtraction/dead-time correction and keep denominator/clipping diagnostics separate from physical saturation semantics.
- [x] Preserve available dark-acquisition laser-shot counts.
- [x] `20251107sapm` observational evidence records the eight extreme 532.PC profiles and correction-order sensitivity without interpreting them as a physical detector threshold.
- [x] Keep the current productive dark-before-dead-time order pending broader evidence.
- [ ] Characterize physical PC saturation under operational SPU conditions with AN/PC overlap and preferably controlled attenuation before enabling a traceable threshold/mask.

### Other evidence

- [ ] Characterize propagated-error SNR before enabling a hard Rayleigh SNR gate.
- [ ] Validate cloud/layer screening before productive reference rejection.
- [ ] Quantify gluing slope/intercept uncertainty and covariance before expanding the uncertainty budget.
- [ ] Compare representative retrievals with LPP and SCC/ELDA expectations without treating either as ground truth.

## P5.0 — frozen evidence and historical audit — COMPLETE

- [x] Freeze `20251107sapm` method-v3 observational baseline with hashes, references, block success and finite tops.
- [x] Keep synthetic truth distinct from observational regression evidence.
- [x] Audit the historical `20241219nt` archive and preserve reusable lessons/machine-readable metrics.
- [x] Record why historical apparent ~30 km coverage is not itself supported backward optical retrieval.
- [x] Express Rayleigh reference-window width physically in meters and resolve it deterministically on the actual uniform altitude grid.
- [x] Keep Rayleigh search bounds physical and fail explicitly when geometry is not representable.

## P5.1 — altitude-resolved inversion support — COMPLETE + REAL-DATA CHECKED

- [x] Pure backward-support contract in `milgrau.level2.support`.
- [x] Synthetic tests cover exact upper boundary, lower instrument-mask examples, noisy/missing tail, missing/negative uncertainty and internal gaps.
- [x] Productive Level 2 exposes aggregate/block inversion support, top/bottom altitude and altitude-resolved effective block count.
- [x] Internal unsupported gaps cannot be bridged.
- [x] No validated lower instrument/overlap mask is claimed.

Real method-v4 check for `20251107sapm`:

- 355 nm aggregate inversion top = **6101.25 m**; 5/5 blocks support through **5606.25 m**, final tail supported by one block;
- 532 nm aggregate inversion top = **6318.75 m**; 5/5 blocks support through **5748.75 m**, final tail supported by one block;
- algorithmic bottom remains 3.75 m and is not interpreted as validated near-field scientific support.

## P5.2 — Rayleigh candidate catalogue — COMPLETE + REAL-DATA CHECKED

Implementation:

- [x] Enumerate every fully contained candidate in the configured physical search interval.
- [x] Compute valid fraction, relative slope, relative variance, calibration diagnostics, diagnostic cost and propagated-uncertainty SNR before selection.
- [x] Preserve accepted/rejected candidates and explicit rejection bit masks.
- [x] Productive selection is `enumerate -> diagnose -> minimum QA -> rank accepted candidates only`.
- [x] Method-v4 ranking remains minimum historical `relative_slope + relative_variance` among accepted candidates, with deterministic lower-grid-index tie break and no altitude preference.
- [x] Schema 3 persists candidate geometry, QA metrics, SNR diagnostic, rejection mask, accepted flag, pre-QA unfiltered-minimum flag and productive-selected flag.
- [x] Schema-3 validation enforces exactly one unfiltered minimum and one accepted productive selection for each successful block/wavelength and exact equality with the KFS reference altitude.
- [x] SNR remains diagnostic-only; no hard threshold is enabled.

Real schema-v3 check from uploaded `20251107sapm_level2_optical.nc`:

- file SHA-256 `32b793fc7874f850ee38273f7073c957194f87dd08fed1c340e96b976917a212`;
- source revision `b68e4d812a37c2a003e49e30c15c2e112e4c8360`, schema 3, method 4, product success;
- **25,340/25,340** candidate slots evaluated across 5 blocks × 2 wavelengths × 2,534 candidate centers;
- **16,224** candidates pass all currently enabled minimum-QA gates;
- accepted flag exactly equals zero rejection mask in every slot;
- exactly one unfiltered minimum and one productive selection exist in every block/wavelength case;
- every selected candidate is accepted, is the minimum diagnostic cost among accepted candidates and exactly equals the persisted block KFS reference altitude;
- the unfiltered minimum is already accepted and equals the productive selection in all 10 cases, so QA-first ordering does not change this measurement's references;
- aggregate references remain **5748.75 m (355)** and **5816.25 m (532)**; aggregate inversion tops remain **6101.25 m** and **6318.75 m**;
- evidence frozen in `docs/regression_baselines/20251107sapm_method_v4_schema3_catalogue.json`.

Remaining catalogue R&D is not required to close P5.2:

- [ ] Validate cloud/layer contamination diagnostics before productive rejection.
- [ ] Define separation/correlation rules before nearby overlapping accepted windows are treated as distinct ensemble members.

## P5.3 — real-data candidate interpretation — IN PROGRESS

`20251107sapm` now shows directly from the persisted schema-v3 catalogue:

- [x] Current minimum-cost selection reproduces the frozen lower references in all 10 block/wavelength cases.
- [x] Many high-altitude windows pass the permissive shape/valid-fraction gates; highest accepted candidates occur at roughly **23.6–24.5 km** depending on block/wavelength.
- [x] Passing shape QA at high altitude is therefore not the limiting condition and must not be interpreted as supported KFS optical retrieval.
- [x] Accepted-candidate SNR declines strongly with altitude: median SNR is ~14.8/15.9 in 5–10 km (355/532), ~3.13/3.59 in 10–15 km, ~1.20/1.32 in 15–20 km and ~0.90/0.87 in 20–25 km.
- [x] Every accepted candidate in the 15–20 km and 20–25 km bands has SNR < 3 in this measurement; this is descriptive evidence only, not a proposed hard SNR threshold.
- [x] Rejections are dominated by excess relative slope and/or variance, with some insufficient-valid-fraction combinations; no candidate was rejected for invalid calibration in this case.
- [ ] Separate future high-column limitation into propagated-SNR, temporal representativeness and cloud/layer contamination before enabling any new productive gate.
- [ ] Compare a future redesigned lower-column solution against the frozen method-v3 baseline when a higher-boundary method becomes productive.

## P5.4 — high-column backbone — TEMPORAL + VERTICAL-SNR R&D

Do not implement a naive whole-measurement mean or cosmetic post-retrieval smoothing as a productive retrieval.

Implemented temporal diagnostics:

- [x] Pure signal/error temporal-support diagnostics with explicit block weights.
- [x] Altitude-resolved supporting-block count and supported-weight fraction on common finite signal/non-negative uncertainty support.
- [x] Per-block absolute signal-contribution fractions, dominant contribution fraction/index and contiguous-subwindow state diagnostics.
- [x] Explicit high-altitude candidate-persistence diagnostic: for a caller-specified target altitude, report accepted-candidate counts per block, block availability, highest accepted candidate and weighted block-persistence fraction without declaring the target scientifically valid.
- [x] Synthetic tests cover stable blocks, unequal weights, transient-only support, single-block dominance, missing uncertainty, early/late state change and first-block-only high-candidate persistence.
- [x] Cross-platform CI green for the candidate-persistence extension at `831c5e7eead3bceb00830312bfae72109ecaa506`.

Real `20251107sapm` temporal diagnostic with explicit profile-count weights `[23, 39, 39, 40, 26]`:

- [x] All five blocks have finite RCS and finite non-negative propagated error through the inspected <30 km domain at both wavelengths; finite/error support therefore does **not** explain why productive KFS stops near 6 km.
- [x] Below 10 km the dominant block contribution is well balanced: median dominant fraction ~0.25–0.26 and no bins exceed 50% single-block contribution.
- [x] At 15–20 km median dominant fraction rises to ~0.385 (355/532), with ~13–14% of bins exceeding 50% single-block contribution but <1% exceeding 70%.
- [x] Two-/three-block subwindows diverge much more around 15–18 km than around 10 km, consistent with increasing noise and/or temporal variability; no acceptance tolerance is inferred from that alone.
- [x] Evidence frozen in `docs/regression_baselines/20251107sapm_temporal_support_preliminary.json`.

Comparative high-candidate persistence evidence:

- [x] For `20251107sapm`, every one of the five current blocks contains at least one already-accepted Rayleigh candidate at or above **15 km** and **20 km** for both 355 and 532 nm; profile-count-weight persistence is 1.0.
- [x] For historical `20241219nt`, only the first of three 20-minute blocks contains accepted candidates at or above 15/20 km; later blocks collapse to highest accepted candidates near 6.4 and 5.5 km. The legacy profile-count-weight persistence is 37/105 ≈ 0.352.
- [x] This contrast is frozen in `docs/regression_baselines/high_column_candidate_persistence_comparison.json`.
- [x] Candidate persistence is therefore an auditable temporal-representativeness observable, but **not yet a productive acceptance criterion**. Historical uncertainty is also not numerically identical to the current propagated-uncertainty model.

Vertical aggregation / smoothing R&D:

- [x] Add pure **non-overlapping vertical aggregation** diagnostics on a uniform grid; no productive signal is changed.
- [x] Preserve strict support: one missing/invalid source bin invalidates its aggregate; no interpolation or gap bridging is allowed.
- [x] Carry both independent-bin uncertainty and perfect-positive-correlation uncertainty explicitly instead of silently claiming a `sqrt(N)` gain.
- [x] Add empirical block-demeaned, uncertainty-scaled vertical residual autocorrelation and a stationary autocorrelation-adjusted theoretical SNR-gain diagnostic.
- [x] Synthetic tests verify identity at one bin, expected independent/correlated limits, missing-bin rejection, linear-profile center preservation, no edge padding and correlated-residual detection.
- [x] Cross-platform CI green for vertical-aggregation diagnostics at `360f8d7bf4a445b878f939135b44bc2cdb80c04a`.
- [x] In real `20251107sapm` far-range selected signal, the gluing source is PC in the inspected 15–25 km bands. Level-1 PC residual lag-1 correlation is modest (~0.13 at 355 nm and ~0.10 at 532 nm) and longer lags are near zero.
- [x] Under the empirical stationary/equal-variance correlation approximation, expected SNR gain is ~2.5× for 60 m aggregation and ~3.4–3.5× for 120 m aggregation in 15–20 km, below the ideal `sqrt(N)` but materially above one.
- [x] Direct block-error bracketing shows the scientific dependence assumption clearly: at 15–20 km, 60–120 m aggregation can move median SNR from ~1.1–1.2 toward ~2.2–3.8 under an independence-like model, while the perfect-correlation limit gives essentially no gain.
- [x] At 20–25 km even aggressive 120–240 m aggregation remains weak in this case; aggregation cannot manufacture high-column information from an intrinsically noise-dominated signal.
- [x] Evidence frozen in `docs/regression_baselines/20251107sapm_vertical_aggregation_preliminary.json`.
- [ ] Run synthetic KFS truth experiments on 15/30/60/120/240 m pre-retrieval aggregation and quantify layer-amplitude bias, vertical displacement, boundary sensitivity and lower-column bias.
- [ ] Quantify calibration-window uncertainty using the full 1 km Rayleigh window, because current candidate SNR is bin-wise and may understate the information in a window-level calibration estimate.
- [ ] Prefer explicit pre-retrieval aggregation/rebinning over Savitzky–Golay post-processing if a productive resolution trade is eventually justified.

Remaining backbone gate:

- [ ] Combine persistence, propagated-SNR behavior, contribution dominance, subwindow disagreement and any validated aggregation resolution into an evidence-backed experimental decision rule without tuning it to a desired top altitude.
- [ ] Validate that rule on more synthetic boundary cases and at least additional observational regimes before making it productive.
- [ ] Only then add a distinct long-mean/high-column retrieval input with explicit duration/provenance and common signal/error support.
- [ ] Preserve original-resolution/block signals; backbone is additional state, not a destructive replacement.

P5.4 acceptance: longer averaging/controlled aggregation must demonstrably increase usable high-altitude information without materially biasing the lower column, hiding temporal nonstationarity or disguising loss of vertical resolution.

## P5.5 — robust multi-reference ensemble — PENDING

- [ ] Require individually QA-passing narrow local reference windows.
- [ ] Define separation/correlation rules before treating candidates as distinct evidence.
- [ ] Keep each KFS member tied to its exact local boundary.
- [ ] Combine only members whose backward branch covers the altitude bin.
- [ ] Define explicit weighting from quality/uncertainty quantities.
- [ ] Add between-reference spread/reference-choice sensitivity as an uncertainty component.
- [ ] Expose effective member count separately from effective block count.

## P5.6 — cascaded backward retrieval — DEFERRED DECISION GATE

Cascade is not the default next step. Consider it only if backbone + ensemble leave a scientifically meaningful coverage gap. If eventually justified, lower segments inherit their boundary from the accepted upper solution, inherited-boundary uncertainty is propagated, inconsistent handoffs are rejected and unsupported gaps are never bridged.

## P5.7 — overlap merge — DEFERRED

Implement only if multiple accepted solutions actually need stitching. Merge weights/continuity/uncertainty behavior must be tested rather than chosen cosmetically.

## P5.8 — molecular-model and uncertainty consistency — PARALLEL

- [ ] Audit exact molecular extinction/backscatter definitions and internally consistent `S_m = alpha_mol/beta_mol` by wavelength.
- [ ] Resolve total-Rayleigh vs Cabannes/effective detected molecular component for actual receiver/filter semantics.
- [ ] Quantify 355/532 retrieval sensitivity to the current molecular-lidar-ratio difference before productive change.
- [ ] Version retrieval method explicitly if productive molecular semantics change.
- [ ] Quantify gluing fit slope/intercept uncertainty/covariance and add only if material.
- [ ] Add future ensemble/backbone/cascade uncertainty components only with explicit dependence assumptions.

## P5.9 — Level 2 schema/provenance/QA

- [x] Schema 2 introduced altitude-resolved inversion support/top/bottom/effective-block count.
- [x] Schema 3 adds auditable Rayleigh candidate catalogue without changing method-v4 KFS physics.
- [x] Currentness/provenance separates package version, product schema and retrieval-method identity.
- [x] Generic schema-3 contract validates the candidate catalogue rather than trusting the version attribute alone.
- [x] `docs/level2_schema.md` documents schema 3 / method 4 and diagnostic-only SNR policy.
- [ ] Update QA plots to show effective optical retrieval top/support and selected/accepted candidate locations; finite scattering ratio must not visually imply aerosol retrieval.
- [ ] Add temporal-support panel only when backbone diagnostics become productive.
- [ ] Add aggregation/effective-resolution metadata and QA only if vertical aggregation becomes productive.
- [ ] Add ensemble/cascade panels only if those algorithms become productive.

## P5.10 — validation matrix

Synthetic established:

- [x] Pure molecular atmosphere -> near-zero aerosol within numerical tolerance.
- [x] Controlled aerosol layers at 355/532 -> recovery within current discretization tolerance.
- [x] Vertical-grid convergence.
- [x] Missing uncertainty/internal invalid gaps fail support explicitly.
- [x] Candidate catalogue retains accepted/rejected windows and QA-first selection.
- [x] Temporally heterogeneous synthetic sequence exposes transient-only far-range support and single-block dominance.
- [x] Candidate-persistence synthetic cases distinguish persistent high-candidate availability from first-block-only availability without applying an acceptance threshold.
- [x] Vertical-aggregation helper exposes independent and fully correlated uncertainty limits and never bridges missing source bins.
- [ ] KFS truth under controlled 15/30/60/120/240 m vertical aggregation -> quantify bias versus effective resolution and noise gain.
- [ ] Evidence-backed backbone reject/split criterion on temporal non-representativeness.
- [ ] Multiple valid high references -> ensemble stability and reference-sensitivity uncertainty.
- [ ] Biased/contaminated candidate -> rejection or visible uncertainty impact.
- [ ] Upper-tail noise -> supported top falls gracefully under the future high-column method.
- [ ] Backbone averaging/aggregation -> no material lower-column truth bias.
- [ ] Molecular-lidar-ratio semantics sensitivity quantified.

Real SPU:

- [x] Current method-v4 `20251107sapm` behavior checked: complete 355/532 retrieval, unchanged lower references and ~6.1/~6.3 km aggregate inversion tops.
- [x] Schema-3 persisted real catalogue validated and frozen as machine-readable observational evidence.
- [x] Preliminary real temporal support/contribution/subwindow diagnostics recorded for `20251107sapm`.
- [x] Candidate-persistence contrast recorded between `20251107sapm` and historical `20241219nt`.
- [x] Preliminary vertical-correlation/SNR-gain brackets recorded for `20251107sapm`; they motivate aggregation experiments but do not authorize productive smoothing.
- [ ] Extend defensible 355/532 support materially above current tops only when evidence supports it.
- [ ] Evaluate whether 15 km is supportable for representative measurements.
- [ ] Attempt 20 km target only when trustworthy boundary/support exists at or above the needed altitude.
- [ ] Add clear-night, high-aerosol, cloud, weak-signal, temporally changing and materially different PC/AN-dominance cases before general validation.
- [ ] Compare representative cases with LPP/SCC/ELDA under matched assumptions without treating them as truth.

P5 success criterion: **maximize defensible inversion-supported vertical coverage while exposing where/why support ends, preserving temporal representativeness, uncertainty honesty, declared effective resolution and exact boundary provenance. Success is not “reaches 20 km.”**

## P6 — release/publication readiness

- [ ] Explicit software license selected and reflected consistently in root license/citation/package metadata.
- [ ] `CITATION.cff` aligned with actual release version/date/DOI.
- [ ] Scientific method/schema changes summarized in release notes.
- [ ] Frozen reproducible scientific environment/constraints.
- [ ] Build sdist/wheel in CI and test clean installation from artifacts.
- [ ] Core-science coverage report and deliberate warning policy.
- [ ] End-to-end Level 0 -> Level 2 run from release artifact/environment.
- [ ] Recorded representative NetCDF metadata review.
- [ ] Frozen machine-readable synthetic acceptance and observational regression summaries.
- [ ] Reconcile branch history with `main` before release/merge.

## Immediate next gate

1. Run **synthetic KFS truth experiments** with explicit non-overlapping 15/30/60/120/240 m pre-retrieval aggregation; quantify SNR gain, effective-resolution loss, layer-amplitude/position bias, boundary sensitivity and lower-column bias.
2. Add a window-level Rayleigh calibration-uncertainty diagnostic so bin-wise SNR is not confused with the uncertainty of a ~1 km calibration estimate; carry dependence assumptions explicitly.
3. Combine persistence, calibration/SNR evidence, contribution dominance and subwindow disagreement into an **experimental**, non-productive backbone decision rule; do not tune thresholds to 15/20 km.
4. Seek at least one additional real SPU case with materially different temporal/high-altitude behavior before promoting any threshold or aggregation width to production.
5. Audit molecular-lidar-ratio consistency and gluing fit-parameter uncertainty in parallel.
6. Only after those gates, introduce a productive long-mean/high-column backbone and later a multi-reference ensemble if they demonstrably extend support without hiding temporal nonstationarity, boundary ambiguity or loss of resolution.
