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
* Retrieval method **v4**: QA-first Rayleigh selection + backward Klett–Fernald–Sasano.
* Productive boundary: exact measured RCS bin at selected reference altitude.
* Boundary assumption: `aerosol_ref_fraction = 0`, therefore `beta_total(ref)=beta_mol(ref)`.
* Candidate rank after minimum QA: `relative_slope + relative_variance`, lower grid index as deterministic tie-breaker.
* Rayleigh SNR is diagnostic-only.
* Aerosol extinction is conditional on assumed aerosol lidar ratio.
* Productive optical support is inversion support, not generic finite-value support.

Not productive: inferred non-zero boundary aerosol, fitted/window boundary, vertically aggregated backbone, Raman boundary correction, hard SNR/cloud/temporal/continuity gates, overlap cutoff, physical PC saturation threshold, ensemble, cascade/stitching and model-implied wavelength-dependent molecular lidar ratio.

---

# 2. Active baseline and evidence set

Primary baseline: `20251107sapm`.

Active identity:

* L2 SHA `bba3b454387ef34c9b4bee3b0ab69e286e611cc44f9a4a6358b779f6d9f236d8`
* source L1 SHA `7b49c9deb93e89b103dba95455d854fcebb97b81292a18269edd1bea7a5ee342`
* source revision `fa1cae20ec1ac28bf2955deb5a93f9a3ed56e663`
* schema/method 3/4
* 25,340 candidate slots; 16,331 accepted; 10 selected
* temporal weights 23 / 39 / 39 / 40 / 26

Historical evidence for the previous checksum is retained, but its original NetCDF is unavailable after workstation migration. Historical, active and campaign products are not claimed input-equivalent unless their input hashes match.

Key frozen evidence includes:

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
* `p3_level2_metadata_audit_20260917.json`

---

# 3. Status overview

| Priority | Status | Meaning |
| --- | --- | --- |
| P0–P2 | COMPLETE | productive architecture/support/engineering guardrails |
| P3 | **IN PROGRESS / METADATA REVIEW COMPLETE** | FAIR recipe provenance strong; release governance + legacy scientific provenance remain |
| P4 | PARALLEL EVIDENCE | instrument characterization / Raman receiver metadata / external optical comparison |
| P5.0–P5.2 | FROZEN FOUNDATION | baseline, support semantics, QA-first catalogue |
| P5.3 | IN PROGRESS | candidate interpretation |
| P5.4 | **ACTIVE PRIMARY R&D** | boundary-condition validity; independent Raman constraint is leading path |
| P5.5 | PENDING | ensemble only if P5.4 establishes need |
| P5.6–P5.7 | DEFERRED | cascade / stitching |
| P5.8 | PARALLEL R&D | molecular semantics / uncertainty consistency |
| P5.9 | **IMPLEMENTED + REAL-IMAGE REVIEWED** | support-aware QA gate closed |
| P5.10 | IN PROGRESS | heterogeneous/seasonal validation matrix |
| P6 | PENDING | reproducible release/publication |

---

# 4. P5.4 — established scientific findings

## 4.1 Candidate/path/contamination evidence

Across baseline, synthetics and 11 heterogeneous/seasonal real L2 products:

* candidate existence or persistence does not imply a usable KFS boundary;
* accepted candidates can sit above internal invalid backward paths;
* native high paths can be nominally complete yet MC-fragile because many weak isolated bins accumulate failure probability;
* `20250509sant` has a strong layer near 14.36–14.37 km and accepted/persistent candidate islands reappear above it;
* fitted/window boundaries can denoise yet remain biased under broad contamination;
* leave-part-out and placement sensitivity expose some localized contamination but are not molecular-purity certificates;
* no SNR, cloud, persistence, continuity, leave-out or placement threshold is authorized.

All successful supplied campaign retrieval blocks use post-QA **analog fallback**, so glued/PC high-column behavior remains an explicit evidence gap.

## 4.2 Covariance and aggregation

From `20250629sant` Level 1, block-demeaned uncertainty-normalized residuals over 5–20 km give approximately:

* lag-1 / 7.5 m correlation **0.10–0.13**;
* 60 m / 8-bin SNR gain **2.47–2.52**;
* 120 m / 16-bin SNR gain **3.31–3.40**.

This is one clean-night analog case, not a station-wide covariance law.

At the existing productive boundary:

* 60 m aggregation: median 0.6–6 km relative L2 change ~**5%**, p95 ~**20%**;
* 120 m aggregation: median ~**6%**, p95 ~**20%**;
* deterministic coarse representation + boundary-cell shift contributes median ~**4.6%** / **5.6%**, with p95 ~21–22%.

Decision: aggregation improves support but is not lower-column neutral. No width is productive.

## 4.3 Moving the boundary is the larger sensitivity

Native-grid boundary-only tests near 10 km isolate boundary movement from aggregation:

* 159/161 input-valid runs have usable exact high-reference bins;
* 111/159 are all-300 MC complete;
* median 0.6–6 km relative L2 change vs method v4: **13.1%**;
* p95: **47.2%**;
* median absolute integrated-column difference: **16.5%**; p95 **61.2%**.

Slope, variance, SNR, diagnostic cost, calibration factor and path completeness show weak/inconsistent within-case association with this lower-column sensitivity. There is no demonstrated hidden elastic score that validates the higher boundary.

## 4.4 Residual aerosol is a demonstrated boundary mechanism

Controlled synthetic truth shows:

* Rayleigh slope/variance/valid-fraction QA can pass while `beta_aer(ref)>0`;
* using the true total-backscatter boundary recovers lower-column truth;
* forcing `beta_aer(ref)=0` creates monotonic lower-column bias as residual aerosol grows;
* signal noise adds dispersion but does not remove systematic boundary bias;
* lidar-ratio mismatch can amplify or partially cancel boundary bias, so apparent agreement cannot prove boundary correctness.

Executable evidence: `tests/test_kfs_residual_aerosol_boundary_rnd.py`; cross-platform CI `35255804893` passed.

`milgrau/level2/boundary_sensitivity.py` exposes only caller-declared scenarios. Across 150 successful campaign controls at the existing reference, declared `f=0.02` changes 0.6–6 km by median ~**5.3%** and `f=0.05` by ~**13.9%**. These are sensitivity scenarios, not inferred aerosol fractions.

**P5.4 conclusion:** the principal unresolved problem is physical validation of the KFS boundary assumption, not candidate-score tuning or vertical rebinning.

---

# 5. Raman independent-constraint path

Current evidence establishes channel identity and acquisition, not yet quantitative Raman retrieval.

* Published current SPU system: 100 Hz; elastic 355/532/1064; Raman 387 N2, 408 H2O and 530 N2.
* Current Level 0/1 `20250629sant` contains **387/408/530 AN+PC** channels.
* SCC configuration 1046 exposes 387/530 AN+PC; 408 is present in acquisition but not exposed by that SCC configuration.
* `20250629sant_scc.nc` confirms channel IDs but contains no passband, bandwidth, effective wavelength or spectral-response fields.
* Historical SPU filter values are not promoted to the MerionC profile because continuity of receiver/filter/FOV metadata is not demonstrated.

Repository safeguards:

* `station.yaml` separates Raman channel identity from unresolved current spectral response;
* `tests/test_spu_raman_metadata_contract.py` prevents silent insertion of numerical MerionC passbands/effective wavelengths while unresolved;
* `docs/spu_raman_channel_provenance.md` records the evidence hierarchy;
* `docs/raman_boundary_validation_rnd.md` defines the R&D physics contract.

Scientific ordering:

1. 387/355 vibrational Raman is the preferred first quantitative path once current 387 spectral/overlap metadata is traceable.
2. 530/532 is rotational Raman and requires explicit temperature-dependent effective cross section plus active filter/transmission response; an undocumented 387-like formula is not acceptable.
3. A real Raman diagnostic must remain independent of elastic candidate scoring and be synthetically validated first.

Current external blocker: current MerionC filter/transmission/effective passband plus Raman-vs-elastic overlap/calibration semantics.

---

# 6. P3 FAIR / provenance state

Representative metadata audit: `docs/regression_baselines/p3_level2_metadata_audit_20260917.json`.

Across all 11 campaign L2 products:

* schema/method identity, repository revision, source-code SHA and source-Level1 SHA are present;
* KFS boundary/MC/lidar-ratio assumptions are recorded;
* exact `config.yaml` and exact `station.yaml` are embedded as YAML in every NetCDF;
* therefore the scientific recipe is strongly preserved and legacy config hashes are unnecessary redundancy.

One machine-discoverability gap was found: these campaign products expose legacy `Station_Profile` but not normalized `station_profile_id` / `instrument_calibration_id` global attributes.

Current-code fix:

* `milgrau/provenance.py` commit `590cf00c211b441aae1e58bbc1c4aeeb816d79f0` normalizes an explicitly recorded source `Station_Profile` and resolves its calibration only when that named profile exists in the loaded station catalog; it performs **no date-based profile inference**;
* regression test commit `b25045722c0d405728c68945ab601bce69ee739d`;
* CI run `35257631767`: Ruff and Ubuntu Python 3.12/3.14 passed; Windows jobs were still running at this tracker snapshot.

Release/discovery metadata remains separate:

* software/data license + root `LICENSE` require deliberate governance choice;
* creator/contact/citation/title/summary/keywords need a stable publication policy;
* do not claim CF Conventions until actual CF compliance is reviewed;
* bit-for-bit dependency environment belongs to P6 release artifacts.

## Lidar-ratio recipe provenance

`docs/lidar_ratio_climatology_provenance.md` records the current status.

The exact 36 monthly 355/532/1064 values already exist in the oldest tracked `config.yaml` revision (`96c22196c73e1ce5733e21c3877f031d8da34d97`, 2026-03-31). The tracked history contains no derivation script, source dataset, sample period, uncertainty derivation or citation for that table. Published SPU literature establishes seasonal lidar-ratio variability but does not establish these exact numbers.

Status: **legacy empirical recipe; scientific derivation unresolved**. The values remain the declared productive assumption; replacing them or their uncertainty semantics is a scientific change requiring validation.

---

# 7. Engineering/QA gates closed

Exact-reference robustness:

* historical wavelength-fatal cases `20240621sant/532` and `20240902sant/355` are now block-local KFS rejections;
* implementation `f90c6f770e7b8d369e775d1ed812c17fbe1266e3`;
* `tests/test_level2_kfs_block_failure.py`;
* CI `35248016374` passed.

P5.9 support-aware QA:

* inversion top/support semantics explicit in SR/KFS plots;
* heterogeneous real-image review passed, including high-cloud `20250509sant`;
* CI `35241784472` passed.

Scientific traceability:

* documentation aligned to schema v3 / method v4;
* regression contract prevents method-v3 drift;
* CI `35240916329` passed.

---

# 8. Parallel open work

P3:

* deliberate license decision + root `LICENSE` + metadata alignment;
* recover original lidar-ratio monthly-table derivation if possible;
* strengthen historical channel-calibration provenance.

P4:

* overlap/telecover/alignment evidence;
* physical PC saturation characterization;
* gluing fit covariance/materiality;
* real glued/PC-dominant elastic regime;
* current MerionC 387/530 filter/transmission provenance and Raman-vs-elastic overlap/calibration semantics;
* true external optical comparison (`20250629sant_scc.nc` is SCC-ready Level 0, not SCC/ELDA optical L2).

P5.8:

* productive molecular lidar ratio remains `8*pi/3` until receiver/filter semantics are resolved.

P6:

* reproducible dependency/environment artifact;
* warnings/coverage/build checks;
* release metadata;
* branch protection/reconciliation;
* immutable scientific tags.

---

# 9. Next gates before any method v5

1. Obtain traceable current MerionC evidence for the 387/530 receiver path: filter manufacturer/model or measured transmission, effective passband/wavelength, receiver changes/continuity and Raman-vs-elastic overlap/calibration semantics.
2. Do **not** substitute historical SPU filter values for missing current evidence.
3. Once item 1 is resolved, implement controlled 387/355 vibrational-Raman synthetic truth under `docs/raman_boundary_validation_rnd.md`; only then use `20250629sant` as first real feasibility case.
4. Keep 530/532 later unless rotational-Raman temperature/filter physics is explicit.
5. Replicate empirical vertical covariance on additional Level-1 cases and obtain a valid glued/PC-dominant elastic regime before instrument-wide aggregation claims.
6. Recover lidar-ratio and historical calibration derivations where possible; preserve unresolved provenance explicitly otherwise.
7. Define any lower-column preservation criterion from physics/uncertainty/validation needs, never by tuning a tolerance to make a desired altitude pass.
8. Only after these gates decide whether a method-v5 high-column backbone is scientifically justified. Ensemble remains after that decision; cascade/stitching remain deferred.

---

# 10. Current handoff

MILGRAU productive Level 2 remains **schema v3 / method v4**.

P5.4 has moved from “find a higher Rayleigh-looking window” to the scientifically sharper problem “independently validate the physical KFS boundary condition”. Aggregation can increase support but is not lower-column neutral; moving the boundary is the larger sensitivity; current elastic diagnostics do not validate `beta_aer(ref)=0`; and controlled truth reproduces residual-aerosol boundary bias.

Raman is the leading independent path, but current MerionC spectral/overlap metadata is intentionally not fabricated from historical documentation. FAIR recipe provenance is already strong because exact YAML recipes and source/code identities are embedded. The representative L2 metadata audit is complete, machine-readable station lineage is fixed for new products, and the remaining P3 gaps are primarily governance/release metadata plus unresolved scientific ancestry of legacy empirical assumptions.

No hard threshold, cloud veto, fitted boundary, aggregation width, inferred residual fraction, Raman correction, high-column backbone, ensemble, cascade or method-v5 promotion is authorized.
