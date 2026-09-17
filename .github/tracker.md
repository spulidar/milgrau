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
* Rayleigh-window shape compatibility is not proof that `beta_aer(ref)=0`.
* Thresholds are not chosen to reach a desired altitude.
* Productive semantic changes require explicit method/schema/provenance/baseline versioning.

---

# 1. Productive identity — unchanged

* Level-2 schema **v3**: auditable Rayleigh candidate catalogue.
* Retrieval method **v4**: QA-first candidate selection + backward Klett–Fernald–Sasano.
* Productive boundary: exact measured RCS bin at the selected reference altitude.
* Productive boundary assumption: `aerosol_ref_fraction = 0`, i.e. `beta_total(ref)=beta_mol(ref)`.
* Candidate rank: `relative_slope + relative_variance`, lower grid index as deterministic tie-breaker after minimum QA.
* Rayleigh SNR is diagnostic-only.
* Aerosol extinction is conditional on assumed aerosol lidar ratio.
* Productive optical support is inversion support, not generic finite-value support.

Not productive: non-zero inferred boundary aerosol, fitted/window boundary, vertically aggregated high-column backbone, smoothing as support extension, hard SNR/cloud/temporal/continuity gates, overlap cutoff, physical PC saturation threshold, ensemble, cascade/stitching and model-implied wavelength-dependent molecular lidar ratio.

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

Current P5.4 evidence includes:

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
* `p5_4_raman_companion_feasibility_20250629.json`

---

# 3. Status overview

| Priority | Status | Meaning |
| --- | --- | --- |
| P0–P2 | COMPLETE | productive architecture/support/engineering guardrails |
| P3 | IN PROGRESS | FAIR/release/license/provenance |
| P4 | PARALLEL EVIDENCE | instrument characterization / external optical comparison / Raman metadata |
| P5.0–P5.2 | FROZEN FOUNDATION | baselines, support semantics, QA-first catalogue |
| P5.3 | IN PROGRESS | real candidate interpretation |
| P5.4 | **ACTIVE PRIMARY R&D** | boundary-condition validity; Raman-independent constraint is the leading path |
| P5.5 | PENDING | ensemble only if P5.4 establishes need |
| P5.6–P5.7 | DEFERRED | cascade / stitching |
| P5.8 | PARALLEL R&D | molecular semantics / uncertainty consistency |
| P5.9 | **IMPLEMENTED + REAL-IMAGE REVIEWED** | support-aware QA gate closed |
| P5.10 | IN PROGRESS | heterogeneous/seasonal validation matrix |
| P6 | PENDING | reproducible release/publication |

---

# 4. P5.4 support/aggregation foundation

The active baseline, heterogeneous campaign and synthetic work established:

* candidate existence/persistence does not imply a usable KFS boundary;
* native paths near ~12 km can be nominally complete yet MC-fragile because many isolated weak bins accumulate failure probability;
* accepted candidates can exist above internal invalid backward paths;
* 60/120 m aggregation can improve support but trades vertical resolution and depends on covariance;
* fitted/window boundaries can denoise yet remain biased under contamination;
* leave-part-out and boundary-placement sensitivity can detect some localized contamination but broad common-mode contamination can remain stable;
* no SNR, cloud, persistence, continuity, leave-out or placement-stability threshold is authorized.

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

This clean-night analog case lies much closer to independence than full correlation, but it is not a station-wide covariance law. Replication and source-chain diversity remain open.

---

# 7. Multi-case aggregation — support gain does not preserve the product

Aggregation-only evidence: `p5_4_campaign_aggregation_same_boundary_20260917.json`.

The offline KFS implementation reproduces all **150** successful persisted block/wavelength backscatter profiles exactly at native group size 1.

At the existing productive boundary:

* 60 m: all 150 paths complete; median 0.6–6 km relative L2 change ~**5%**, p95 ~**20%**;
* 120 m: all paths complete; median change ~**6%**, p95 ~**20%**;
* integrated-column p95 changes reach ~**27%** (60 m) and ~**33%** (120 m).

Decomposition (`p5_4_aggregation_effect_decomposition_20260917.json`) shows deterministic coarse representation + boundary-cell shift contributes median ~**4.6%** at 60 m and **5.6%** at 120 m, with p95 ~21–22%. The partial-MC mean already differs materially from deterministic KFS on the native productive grid, so that nonlinear MC behavior is not a new aggregation artifact.

Decision: aggregation improves support but is not lower-column neutral. No width is authorized.

---

# 8. Higher boundary is the larger new sensitivity

High-boundary + aggregation (`p5_4_campaign_high_boundary_aggregation_20260917.json`) improves completeness in many cases but produces campaign p95 lower-column changes around 44–46%.

Native-grid boundary-only evidence (`p5_4_campaign_native_grid_boundary_only_20260917.json`) isolates the boundary change:

* 159/161 input-valid runs have usable exact high-reference bins;
* only 111/159 are all-300 complete;
* median 0.6–6 km relative L2 difference vs productive v4 is **13.1%**;
* p95 is **47.2%**;
* median absolute integrated-column difference is **16.5%**, p95 **61.2%**.

Even complete-path cases can change strongly. `20250629sant/532` at ~10 km is 14/14 all-complete but changes the lower-column retrieval materially.

Decision: high-column limitation is not primarily a candidate-ranking or vertical-rebinning problem. The physical boundary condition is now the main scientific blocker.

---

# 9. Existing elastic candidate diagnostics do not validate the boundary

Evidence: `p5_4_boundary_existing_diagnostic_association_20260917.json`.

Within measurement/wavelength groups, candidate slope, variance, SNR, diagnostic cost, calibration factor and path completeness have weak/inconsistent association with lower-column sensitivity to a higher accepted boundary. Signs vary by event.

Decision: do not solve P5.4 by reweighting the current Rayleigh score or by inventing a composite threshold. Existing diagnostics characterize shape, precision and numerical support; they do not establish `beta_aer(ref)=0`.

---

# 10. Residual aerosol at the boundary — synthetic mechanism executable-complete

Evidence:

* `p5_4_synthetic_residual_aerosol_boundary_sweep_20260917.json`
* `p5_4_synthetic_boundary_residual_noise_lr_20260917.json`
* `tests/test_kfs_residual_aerosol_boundary_rnd.py`
* `docs/boundary_condition_rnd.md`

Controlled synthetic truth prescribes a broad smooth aerosol contribution around a ~10 km reference while retaining Rayleigh-like local window shape.

Key results:

* the current slope/variance/valid-fraction QA can still pass when `beta_aer(ref)>0`;
* with the **true** total-backscatter boundary, KFS recovers the controlled lower-column truth to numerical accuracy;
* forcing `beta_aer(ref)=0` creates monotonic lower-column bias as residual aerosol increases;
* at 532 nm in this controlled family, `beta_aer(ref)/beta_m(ref)=0.10` gives ~8% lower-column L2 error and ~10% integrated-column underestimation; 0.20 gives ~15% / ~19%; 0.50 gives ~31% / ~39%; these are mechanism-test values, **not operational thresholds**;
* signal noise increases dispersion but does not remove the systematic boundary bias;
* lidar-ratio mismatch is independent and can either amplify or partially cancel boundary bias, so apparent lower-column agreement cannot prove a correct boundary;
* localized contamination produces strong retrieval sensitivity to boundary placement, but broad contamination can remain nearly placement-stable while still biased. Placement stability is diagnostic, not a purity certificate.

Cross-platform CI run `35255804893` passed Ruff and pytest on Ubuntu/Windows Python 3.12/3.14 with these R&D tests included.

Decision: boundary residual aerosol is an explicit unresolved physical assumption, not a missing weight in the existing Rayleigh score.

---

# 11. Explicit boundary sensitivity helper — R&D only

`milgrau/level2/boundary_sensitivity.py` provides `boundary_fraction_sensitivity_profiles`.

For caller-declared scenario fractions `f`, it evaluates

`beta_total(ref) = beta_mol(ref) * (1 + f)`

with the signal, grid and lidar ratio fixed. It returns deterministic backward-KFS profiles and deliberately provides **no inferred fraction, probability, score, preferred scenario or pass/fail decision**.

`tests/test_boundary_sensitivity_rnd.py` guards this exact semantic contract.

Real-campaign sensitivity evidence: `p5_4_campaign_boundary_fraction_sensitivity_20260917.json`.

Across all 150 successful campaign block/wavelength controls at their existing productive reference altitude:

* declared `f=0.02` changes the 0.6–6 km solution by median ~**5.3%** (p95 ~13.3%);
* declared `f=0.05` gives median ~**13.9%** (p95 ~34.6%);
* integrated-column sensitivity is also material.

These fractions are sensitivity scenarios only. The experiment does **not** estimate that real residual aerosol equals 2% or 5%.

---

# 12. Raman companions — physically independent boundary-validation path identified

Evidence/documentation:

* `p5_4_raman_companion_feasibility_20250629.json`
* `docs/spu_raman_channel_provenance.md`
* `station.yaml` current `spu-merionc-2024` SCC channel mapping

Published SPU instrumentation identifies:

* 387 nm as nitrogen Raman associated with 355 nm;
* 408 nm as water-vapor Raman associated with 355 nm;
* 530 nm as nitrogen Raman associated with 532 nm, described in prior SPU instrumentation work as rotational Raman.

The current MerionC night SCC mapping contains 387 and 530 AN/PC channels. The supplied `20250629sant` Level 1 contains corrected 387/530 signals with useful block SNR through the P5.4 altitude range; e.g. median block SNR at 10 km is ~11.9/31.7 for 387 AN/PC and ~32.7/57.8 for 530 AN/PC.

This establishes **feasibility**, not a Raman retrieval. Current-receiver passband/filter response, species/cross-section treatment, calibration/overlap semantics and required wavelength-dependence assumptions must be explicit before quantitative use. Historical published filter bandwidths are not silently promoted to the post-2024 MerionC profile.

Decision: prioritize companion-channel metadata/physics as the leading independent route to boundary validation. Do not use companion-channel presence or SNR as an undocumented cloud/molecular-purity flag.

---

# 13. Engineering/QA gates already closed

Exact-reference robustness:

* historical wavelength-fatal cases: `20240621sant/532`, `20240902sant/355`;
* KFS exact-bin rejection is now block-local without changing method-v4 Rayleigh semantics;
* implementation commit `f90c6f770e7b8d369e775d1ed812c17fbe1266e3`;
* `tests/test_level2_kfs_block_failure.py`;
* CI `35248016374` passed.

P5.9 support-aware QA:

* inversion top/support semantics explicit in SR/KFS plots;
* `tests/test_level2_qa_support_context.py`;
* CI `35241784472` passed;
* heterogeneous human image review passed, including high-cloud `20250509sant`.

---

# 14. Parallel open work

P3:

* deliberate license decision + root `LICENSE` + metadata alignment;
* representative L2 metadata review;
* SPU lidar-ratio climatology provenance;
* historical calibration provenance.

P4:

* overlap/telecover/alignment evidence;
* physical PC saturation characterization;
* gluing fit covariance/materiality;
* a real glued/PC-dominant elastic regime;
* current Raman receiver/filter spectral provenance;
* true external optical comparison. `20250629sant_scc.nc` is SCC-ready **Level 0**, not SCC/ELDA optical L2.

P5.8:

* productive molecular lidar ratio remains `8*pi/3` until receiver/filter semantics are resolved.

P6:

* reproducible environment/build checks, warnings/coverage, release metadata, branch protection/reconciliation and immutable scientific tags.

---

# 15. Next scientific gate before any method v5

The residual-boundary mechanism itself is now established. The next question is whether independent information can constrain it.

1. **Raman metadata gate:** establish current MerionC effective 387/530 detection wavelengths/passbands, filter/transmission response, overlap/calibration semantics and any post-2024 hardware changes from traceable instrument evidence.
2. **Raman forward/retrieval truth gate:** implement only after item 1. Start with controlled synthetic molecular + aerosol truth and explicit Raman physics; validate extinction/boundary sensitivity before touching real data.
3. **Real Raman feasibility gate:** use `20250629sant` first because Level-1 387/530 signal is available and strong; compare independent Raman evidence with elastic candidate/reference regions without feeding the elastic retrieval back into its own validation.
4. **Boundary sensitivity fallback:** until an independent constraint exists, keep non-zero `f` as declared R&D sensitivity scenarios, never as inferred corrections or productive uncertainty without justification.
5. Replicate vertical covariance on additional Level-1 cases and obtain glued/PC-dominant elastic evidence before instrument-wide aggregation claims.
6. Define lower-column preservation from physics/uncertainty/validation needs, not by tuning tolerance to make a desired altitude pass.
7. Only after those gates decide whether a method-v5 high-column backbone is scientifically justified. Ensemble remains after that decision; cascade/stitching remain deferred.

---

# 16. Current handoff

MILGRAU productive Level 2 remains **schema v3 / method v4**.

P5.4 now has a much sharper scientific diagnosis. Weak high-altitude signal and covariance matter, and aggregation can improve numerical support, but neither is the main unresolved assumption. Higher Rayleigh-like candidates can be precise, persistent, placement-stable and path-complete while still altering the lower-column retrieval because `beta_aer(ref)=0` has not been independently validated. Controlled truth tests now reproduce this mechanism directly, and the heterogeneous real campaign shows that the retrieval is materially sensitive even to small declared boundary-residual scenarios.

The project therefore should not pursue another elastic composite score as the next method-v5 step. The SPU instrument already provides a more promising independent route: corrected 387/530 Raman companion signals are present in the current night data. The immediate R&D priority is to make their current instrument semantics and Raman physics traceable, then test whether they can independently constrain the boundary assumption.

No hard threshold, cloud veto, fitted boundary, aggregation width, inferred residual fraction, high-column backbone, ensemble, cascade or method-v5 promotion is authorized.
