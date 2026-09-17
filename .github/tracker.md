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
* Historical instrument numbers are not current metadata unless continuity is traceable.
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

Not productive: inferred non-zero boundary aerosol, fitted/window boundary, vertically aggregated high-column backbone, Raman boundary correction, hard SNR/cloud/temporal/continuity gates, overlap cutoff, physical PC saturation threshold, ensemble, cascade/stitching and model-implied wavelength-dependent molecular lidar ratio.

---

# 2. Active baseline and frozen P5.4 evidence

Primary baseline: `20251107sapm`.

Active identity:

* L2 SHA `bba3b454387ef34c9b4bee3b0ab69e286e611cc44f9a4a6358b779f6d9f236d8`
* source L1 SHA `7b49c9deb93e89b103dba95455d854fcebb97b81292a18269edd1bea7a5ee342`
* source revision `fa1cae20ec1ac28bf2955deb5a93f9a3ed56e663`
* schema/method 3/4
* 25,340 candidate slots; 16,331 accepted; 10 selected
* profile-count weights 23 / 39 / 39 / 40 / 26

Historical evidence for the previous checksum is retained, but its original NetCDF is unavailable after workstation migration. Historical, active and campaign products are not claimed input-equivalent unless input hashes match.

Key evidence files:

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
* `p5_4_raman_metadata_gate_20260917.json`

---

# 3. Status overview

| Priority | Status | Meaning |
| --- | --- | --- |
| P0–P2 | COMPLETE | productive architecture/support/engineering guardrails |
| P3 | IN PROGRESS | FAIR/release/license/provenance |
| P4 | PARALLEL EVIDENCE | instrument characterization / external optical comparison / Raman receiver metadata |
| P5.0–P5.2 | FROZEN FOUNDATION | baselines, support semantics, QA-first catalogue |
| P5.3 | IN PROGRESS | real candidate interpretation |
| P5.4 | **ACTIVE PRIMARY R&D** | boundary-condition validity; independent Raman constraint is leading path |
| P5.5 | PENDING | ensemble only if P5.4 establishes need |
| P5.6–P5.7 | DEFERRED | cascade / stitching |
| P5.8 | PARALLEL R&D | molecular semantics / uncertainty consistency |
| P5.9 | **IMPLEMENTED + REAL-IMAGE REVIEWED** | support-aware QA gate closed |
| P5.10 | IN PROGRESS | heterogeneous/seasonal validation matrix |
| P6 | PENDING | reproducible release/publication |

---

# 4. Support, contamination and aggregation foundation

Established across baseline, synthetics and the heterogeneous campaign:

* candidate existence/persistence does not imply a usable KFS boundary;
* native high paths can be nominally complete yet MC-fragile because many weak isolated bins accumulate failure probability;
* accepted candidates can exist above internal invalid backward paths;
* `20250509sant` demonstrates accepted/persistent candidate islands reappearing above a strong high-cloud layer;
* fitted/window boundaries can denoise yet remain biased under broad contamination;
* leave-part-out and placement sensitivity detect some localized contamination but are not molecular-purity certificates;
* no SNR, cloud, persistence, continuity, leave-out or placement-stability threshold is authorized.

The heterogeneous/seasonal campaign contains 11 L2 products and 256 QA/quicklook images. All successful supplied campaign retrieval blocks use post-QA **analog fallback**, so glued/PC high-column behavior remains an explicit evidence gap.

---

# 5. Empirical vertical dependence and aggregation results

From `20250629sant_level1_rcs.nc`, block-demeaned uncertainty-normalized residuals between 5–20 km give approximately:

* lag-1 / 7.5 m correlation **0.10–0.13**;
* 60 m / 8-bin SNR gain **2.47–2.52**;
* 120 m / 16-bin SNR gain **3.31–3.40**.

This is one clean-night analog case, not a station-wide covariance law.

At the productive boundary, multi-case aggregation gives:

* 60 m: median 0.6–6 km relative L2 change ~**5%**, p95 ~**20%**;
* 120 m: median ~**6%**, p95 ~**20%**;
* deterministic coarse representation + boundary-cell shift alone contributes median ~**4.6%** (60 m) and **5.6%** (120 m), with p95 ~21–22%.

Decision: aggregation can improve support but is not lower-column neutral. No productive aggregation width is authorized.

---

# 6. Higher boundary is the dominant new sensitivity

Native-grid boundary-only evidence isolates the effect of moving the exact KFS reference without vertical aggregation:

* 159/161 input-valid runs have usable exact high-reference bins;
* 111/159 are all-300 MC complete;
* median 0.6–6 km relative L2 difference vs productive v4: **13.1%**;
* p95: **47.2%**;
* median absolute integrated-column difference: **16.5%**; p95 **61.2%**.

Even all-complete high-boundary cases can materially change the lower column.

Existing candidate diagnostics — slope, variance, SNR, diagnostic cost, calibration factor and path completeness — have weak/inconsistent within-case association with this lower-column sensitivity. Their signs change among events.

Decision: P5.4 is not primarily a score-tuning or vertical-rebinning problem. The unresolved physical boundary assumption is the main blocker.

---

# 7. Residual aerosol boundary mechanism — established in controlled truth

Evidence:

* `p5_4_synthetic_residual_aerosol_boundary_sweep_20260917.json`
* `p5_4_synthetic_boundary_residual_noise_lr_20260917.json`
* `tests/test_kfs_residual_aerosol_boundary_rnd.py`
* `docs/boundary_condition_rnd.md`

Controlled synthetic truth demonstrates:

* current Rayleigh slope/variance/valid-fraction QA can pass with `beta_aer(ref)>0`;
* the true total-backscatter boundary recovers known lower-column truth;
* forcing `beta_aer(ref)=0` creates monotonic lower-column bias as residual aerosol grows;
* signal noise adds dispersion but does not remove systematic boundary bias;
* lidar-ratio mismatch can amplify or partially cancel boundary bias, so apparent agreement cannot prove boundary correctness.

Mechanism-test fractions are **not operational thresholds**.

Cross-platform CI containing these R&D tests passed in run `35255804893`.

---

# 8. Explicit boundary-sensitivity helper — R&D only

`milgrau/level2/boundary_sensitivity.py` exposes caller-declared scenarios

`beta_total(ref) = beta_mol(ref) * (1 + f)`

with no inferred fraction, probability, score, preferred scenario or pass/fail decision.

Across 150 successful campaign block/wavelength controls at the existing productive reference:

* declared `f=0.02`: median 0.6–6 km change ~**5.3%**, p95 ~13.3%;
* declared `f=0.05`: median ~**13.9%**, p95 ~34.6%.

These are sensitivity scenarios only. They do not estimate real residual aerosol.

---

# 9. Raman independent-constraint path — feasibility established, spectral metadata still open

Published current-system evidence supports:

* 100 Hz SPU system;
* elastic 355/532/1064 nm;
* 387 nm nitrogen Raman companion of 355 nm;
* 408 nm water-vapor Raman companion of 355 nm;
* 530 nm nitrogen rotational-Raman companion associated with 532 nm.

Current repository/SCC evidence:

* `spu-merionc-2024` night SCC configuration 1046 exposes `530.PC=4074`, `530.AN=4075`, `387.AN=4076`, `387.PC=4077`;
* supplied `20250629sant` Level 1 contains corrected 387/530 AN/PC signals with useful high-altitude SNR;
* supplied `20250629sant_scc.nc` SHA `8d52f534525c73e64badec5d4007cdf4ea3609e5e962a9045a1bdaf178314af0` confirms configuration/channel IDs but contains **no passband, bandwidth, effective-wavelength or spectral-response fields**.

Metadata changes:

* `station.yaml` now records `raman_detection` for `spu-merionc-2024` with channel identity separate from unresolved current spectral response;
* 387/408/530 current spectral response is explicitly `current_instrument_evidence_required`;
* 408 is published as a current system channel but is not exposed in SCC 1046; this is recorded as an unresolved mapping/scope fact, not guessed;
* `tests/test_spu_raman_metadata_contract.py` prevents silent insertion of numeric MerionC passbands/effective wavelengths while evidence is unresolved;
* `docs/spu_raman_channel_provenance.md` documents the evidence hierarchy;
* `docs/raman_boundary_validation_rnd.md` defines the R&D physics contract.

Historical SPU filter values remain historical provenance only. They are not promoted into the post-2024 MerionC profile because public historical receiver/FOV/overlap details are not demonstrably continuous with the current 100 Hz configuration.

CI run `35256468442` validates the Raman metadata contract; Ruff and Ubuntu 3.12/3.14 were green at this tracker update, with Windows jobs still completing.

---

# 10. Raman physics gate

No real Raman retrieval is authorized yet.

Scientific contract:

* 387/355 vibrational-Raman extinction is the preferred first quantitative path once current 387 spectral/overlap metadata is traceable;
* the logarithmic-derivative Raman method requires explicit molecular extinction, N2 density, overlap semantics and aerosol wavelength-dependence treatment;
* absolute range-independent Raman calibration can cancel from the derivative, but overlap/range-dependent response does not;
* 530/532 is rotational Raman and requires explicit temperature-dependent effective cross section plus active filter/transmission response; it must not be processed with an undocumented 387-like simplification;
* a Raman diagnostic must remain independent of elastic candidate scoring and must not be tuned to reproduce KFS.

Minimum controlled synthetic families before real retrieval: molecular-only, aerosol below boundary, weak residual aerosol at boundary, layer crossing boundary, Ångström sensitivity, rotational-Raman temperature/filter sensitivity, overlap-transition contamination, and noise/smoothing-resolution sensitivity.

---

# 11. Engineering/QA gates closed

Exact-reference robustness:

* historical wavelength-fatal cases `20240621sant/532` and `20240902sant/355` are now block-local KFS rejections;
* implementation `f90c6f770e7b8d369e775d1ed812c17fbe1266e3`;
* `tests/test_level2_kfs_block_failure.py`;
* CI `35248016374` passed.

P5.9 support-aware QA:

* inversion top/support semantics explicit in SR/KFS plots;
* heterogeneous real-image review passed, including high-cloud `20250509sant`;
* CI `35241784472` passed.

---

# 12. Parallel open work

P3:

* deliberate license decision + root `LICENSE` + metadata alignment;
* representative L2 metadata review;
* SPU lidar-ratio climatology provenance;
* historical calibration provenance.

P4:

* overlap/telecover/alignment evidence;
* physical PC saturation characterization;
* gluing fit covariance/materiality;
* real glued/PC-dominant elastic regime;
* **current MerionC 387/530 filter/transmission provenance and Raman-vs-elastic overlap/calibration semantics**;
* true external optical comparison. `20250629sant_scc.nc` is SCC-ready Level 0, not SCC/ELDA optical L2.

P5.8:

* productive molecular lidar ratio remains `8*pi/3` until receiver/filter semantics are resolved.

P6:

* reproducible environment/build checks, warnings/coverage, release metadata, branch protection/reconciliation and immutable scientific tags.

---

# 13. Next scientific gate before any method v5

1. Obtain traceable current MerionC evidence for the 387/530 receiver path: filter manufacturer/model or measured transmission curves, effective passbands/wavelengths, receiver changes/continuity since 2024, and Raman-vs-elastic overlap/calibration semantics.
2. Do **not** substitute the historical SPU instrument-page filter values for missing current evidence.
3. Once item 1 is resolved, implement the first controlled **387/355 vibrational-Raman synthetic truth** experiment under the contract in `docs/raman_boundary_validation_rnd.md`.
4. Keep 530/532 later unless its rotational-Raman filter/temperature physics is explicitly characterized.
5. Use `20250629sant` as the first real feasibility case only after synthetic validation; compare Raman evidence with elastic candidate regions without feeding KFS results back into Raman validation.
6. Replicate empirical vertical covariance on additional Level-1 cases and obtain glued/PC-dominant elastic evidence before instrument-wide aggregation claims.
7. Define lower-column preservation from physics/uncertainty/validation needs, not by tuning tolerance to make a desired altitude pass.
8. Only after these gates decide whether a method-v5 high-column backbone is scientifically justified. Ensemble remains after that decision; cascade/stitching remain deferred.

---

# 14. Current handoff

MILGRAU productive Level 2 remains **schema v3 / method v4**.

P5.4 has moved from “how do we find a higher Rayleigh-looking window?” to the sharper question “how do we independently validate the physical boundary condition?”. Aggregation improves numerical support but is not lower-column neutral; moving the exact boundary is the larger sensitivity; current elastic candidate diagnostics do not validate `beta_aer(ref)=0`; and controlled truth directly reproduces the residual-aerosol boundary mechanism.

The leading independent path is Raman, but the repository now explicitly refuses to manufacture current MerionC spectral metadata from historical receiver documentation. Current channel identities and SCC mappings are traceable; current 387/530 spectral response and Raman-specific overlap/calibration semantics remain the key external evidence gap. The generic Raman R&D physics contract is documented, but no real Raman retrieval or method-v5 change is authorized until that instrument gate is resolved.

No hard threshold, cloud veto, fitted boundary, aggregation width, inferred residual fraction, Raman correction, high-column backbone, ensemble, cascade or method-v5 promotion is authorized.
