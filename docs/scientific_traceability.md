# MILGRAU scientific traceability

This document maps the current productive scientific behavior to canonical code ownership, executable evidence, product provenance and the literature or standard that actually supports each claim. External references are an implementation basis only when MILGRAU implements the corresponding physics or method; FAIR, comparison and interoperability references are identified separately.

## Current productive Level 2 identity

The productive Level 2 state on `new-architecture` is:

* product schema **v4** — progressive-grid method-v5 state, support, selected-reference diagnostics, systematic boundary scenarios and selection-aware uncertainty;
* retrieval method **v5** — native-grid Rayleigh QA, progressive-path admissibility, tiered high-reference selection and backward elastic Klett–Fernald–Sasano retrieval;
* 20-minute temporal blocks by default;
* exact native vertical representation below 6 km for the current SPU 7.5 m grid, followed by explicit progressive aggregation above 6 km;
* default reference tiers **10 -> 9 -> 8 -> 6 km**, attempted in that order; within the first supported tier the accepted/admissible candidate with minimum `relative_slope + relative_variance` is selected;
* reference altitude is a search preference and **not** a molecular-purity or aerosol-free claim;
* nominal boundary `f = beta_aer(ref)/beta_mol(ref) = 0`, with caller-declared non-zero `f` values stored as outer systematic sensitivity scenarios rather than sampled as a probability distribution;
* signal noise is propagated through progressive aggregation, Rayleigh QA, tier fallback, reference selection and KFS in a selection-aware Monte Carlo;
* altitude-resolved `mc_valid_fraction` is a diagnostic support quantity and is not converted to a hard pass/fail threshold;
* Rayleigh propagated-error SNR remains diagnostic-only.

Canonical version ownership is `milgrau/scientific.py`. Product assembly is owned by `milgrau/level2/method_v5_product.py`; schema-4 validation is owned by `milgrau/level2/schema_v5.py`; active method decisions are documented in `docs/high_column_method_v5_decisions.md`.

## Method-v5 product semantics

Method v5 reports the following conditional scientific statement:

> Given the declared aerosol lidar-ratio assumption, upper-boundary condition and effective vertical representation, this is the elastic solution; this is the random dispersion obtained when measurement noise is propagated through reference re-selection; and these are the declared boundary-condition sensitivity scenarios.

The elastic retrieval does not claim that aerosol lidar ratio, residual aerosol at the reference, or molecular purity were independently observed. Extinction remains conditional on the assumed/climatological aerosol lidar ratio.

The productive progressive-grid schedule is currently:

| Altitude | Requested effective resolution | SPU 7.5 m realization |
| --- | ---: | ---: |
| < 6 km | 7.5 m | 1 native bin |
| 6–10 km | 15 m | 2 bins |
| 10–15 km | 30 m | 4 bins |
| 15–20 km | 60 m | 8 bins |
| 20–25 km | 60 m | 8 bins |
| >=25 km | <=100 m | 97.5 m / 13 bins on the current grid |

Cells are strict contiguous native-bin groups. NaN, masked or instrument-invalid samples are never bridged. Finite signed background-subtracted signal may participate in an arithmetic cell mean, but a cell used by KFS must itself be finite and positive.

## Traceability matrix

| Scientific behavior / claim | Canonical owner | Executable evidence / diagnostics | Product provenance / status |
| --- | --- | --- | --- |
| Product schema and retrieval method are separately versioned | `milgrau.scientific` | scientific traceability/currentness tests | `level2_product_schema_version=4`, `level2_retrieval_method_version=5` |
| Molecular Rayleigh backscatter/extinction and molecular signal are wavelength dependent | `milgrau.level2.molecular` | `tests/test_molecular.py`, KFS synthetic tests | molecular state plus implementation identity; Bucholtz (1995) implementation basis |
| Thermodynamic pressure/temperature are materialized on the Level-1 lidar grid | `milgrau.level1.thermodynamics` | atmosphere-source and NetCDF tests | `thermodynamic_profile_*` lineage; ERA5/radiosonde/USSA76 identity |
| Backward elastic inversion uses an explicit upper boundary and assumed aerosol lidar ratio | `milgrau.level2.kfs` | analytical/synthetic KFS recovery tests | `integration_mode=backward`; Klett (1981), Fernald (1984) implementation basis |
| Progressive vertical representation is strict and gap-preserving | `milgrau.level2.adaptive_grid`, `high_column_rnd` | adaptive-grid/high-column tests | `effective_vertical_resolution_m`, `source_bin_count`, progressive schedule metadata |
| Native Rayleigh QA precedes high-column reference selection | `milgrau.level2.rayleigh_candidates`, `high_column_rnd` | candidate/selector tests | selected reference slope, variance, valid fraction, cost and SNR diagnostics |
| Reference selection uses the highest supported declared tier, then minimum existing Rayleigh cost | `milgrau.level2.high_column_selector` | synthetic cross-wavelength selector tests, real-profile diagnostics | reference altitude/tier/index/fallback per block |
| Altitude is not a purity certificate | method-v5 decision contract | cloud, broad-residual and stratospheric-aerosol synthetic counterexamples | explicit `rayleigh_reference_altitude_policy` and tier interpretation metadata |
| Nominal boundary is `f=0`, while non-zero `f` values are outer systematic scenarios | `method_v5_product`, `selection_aware_mc_rnd` | boundary-fraction and selection-aware tests | `residual_fraction`, boundary scenario metadata |
| Random uncertainty propagates signal noise through reference selection itself | `milgrau.level2.selection_aware_mc_rnd` | selection-aware MC tests and coverage R&D | MC mean/std/quantiles, selected-reference MC distribution, selection-success fraction |
| `mc_valid_fraction(z)` diagnoses support but does not define physical acceptance | method-v5 schema/decision contract | coverage tests | altitude-resolved `mc_valid_fraction`; no cutoff |
| Period means are finite-only altitude-by-altitude and must be read with temporal support | `method_v5_product`, `schema_v5` | schema-4 contract tests | `period_support_count`, `period_support_fraction` |
| Gluing remains explicit and auditable | `milgrau.level2.gluing`, `signal_selection` | gluing tests | block start/split/stop, slope/intercept, correlation, RMSE and bias |
| Gluing error propagates measurement-noise terms; fitted slope/intercept covariance is excluded | `milgrau.level2.gluing.propagate_glued_error` | gluing uncertainty test | `gluing_uncertainty_scope` |
| Exact Level-1 bytes are part of Level-2 scientific lineage | `milgrau.provenance.file_sha256` | source-identity tests | `source_level1_sha256` |
| Source-code state is separately traceable from package version | `milgrau.provenance.source_code_provenance` | source-provenance tests | source-code SHA/repository revision where available |
| Partial multispectral output is explicit | productive LEBEAR/schema-4 | orchestration/schema tests | requested/processed/failed wavelengths and completeness/status |

## Current validation evidence and interpretation

The v5 architecture was developed from a sequence of controlled synthetic and observational experiments. The important scientific conclusions retained in the productive method are:

* choosing the highest accepted candidate is rejected; high altitude alone can increase bias and does not certify molecular purity;
* no fixed tested altitude floor (8, 10 or 12 km) guarantees a clean boundary under broad aerosol contamination;
* candidate existence at 20–25 km is substantially more common than a continuous admissible KFS path to that altitude;
* progressive vertical aggregation can extend the continuous usable path, but the achieved top remains measurement dependent;
* fixed-reference Monte Carlo omits material reference-selection uncertainty; selection-aware Monte Carlo is therefore part of method v5;
* Monte-Carlo survival and interval calibration are different quantities, so no survival cutoff is used as a scientific acceptance rule;
* residual aerosol at the boundary is not assigned an invented random prior; it remains an explicit systematic scenario dimension.

Frozen evidence is retained under `docs/regression_baselines/`, including progressive-grid campaign, synthetic selector, residual-boundary and selection-aware coverage studies.

A real-profile QA example from 2024-09-02 21:00–23:08 UTC at 532 nm produced block references spanning roughly 8.1–10.3 km under the tiered policy, with altitude-resolved temporal support decreasing above the highest consistently supported region. This is interpreted as block-dependent support, not as a requirement that all blocks reach one common top.

Further extension toward higher references remains an optimization target inside method v5. Raising the achieved reference altitude must not weaken the existing path, QA, uncertainty or boundary-sensitivity semantics, and retrieval top is not itself a truth metric.

## Evidence hierarchy

1. **Analytical / synthetic truth** — primary evidence for equation correctness, estimator behavior, controlled bias and random-uncertainty behavior.
2. **Real-data regression and QA** — verifies behavior and support of the actual SPU acquisition path; it does not provide exact aerosol optical truth.
3. **External-chain comparison** — tests interoperability and consistency under matched assumptions; agreement does not make another chain ground truth.
4. **Instrument characterization** — required for physical detector saturation, overlap, calibration history and other instrument-specific claims.

## Product-language constraints

* Elastic aerosol extinction is **conditional on the assumed aerosol lidar ratio** and is not equivalent to Raman extinction.
* `f=0` is a declared nominal boundary condition, not evidence that the reference is aerosol-free.
* Non-zero `f` values are systematic sensitivity scenarios unless independent evidence later supplies a probability model.
* Finite signal outside nominal/MC support is not a supported aerosol optical retrieval.
* Unsupported gaps are not interpolated or bridged to increase apparent coverage.
* A higher reference or retrieval top is not an acceptance criterion by itself.
* Physical photon-counting saturation remains uncharacterized where instrument evidence is absent; provisional Level-2 guards must not be described as detector characterization.
* Current uncertainty is conditional/partial rather than a complete metrological budget.

## Verified bibliography

### Implemented physical / retrieval methods

* Klett, J. D. (1981). *Stable analytical inversion solution for processing lidar returns*. Applied Optics, 20(2), 211–220. DOI `10.1364/AO.20.000211`.
* Fernald, F. G. (1984). *Analysis of atmospheric lidar observations: some comments*. Applied Optics, 23(5), 652. DOI `10.1364/AO.23.000652`.
* Bucholtz, A. (1995). *Rayleigh-scattering calculations for the terrestrial atmosphere*. Applied Optics, 34(15), 2765–2773. DOI `10.1364/AO.34.002765`.
* Donovan, D. P., Whiteway, J. A., & Carswell, A. I. (1993). *Correction for nonlinear photon-counting effects in lidar systems*. Applied Optics, 32(33), 6742–6753. DOI `10.1364/AO.32.006742`.

### External lidar-chain methodology / comparison

* D'Amico, G., Amodeo, A., Mattis, I., Freudenthaler, V., & Pappalardo, G. (2016). *EARLINET Single Calculus Chain – technical – Part 1: Pre-processing of raw lidar data*. Atmospheric Measurement Techniques, 9, 491–507. DOI `10.5194/amt-9-491-2016`.
* Mattis, I., D'Amico, G., Baars, H., Amodeo, A., Madonna, F., & Iarlori, M. (2016). *EARLINET Single Calculus Chain – technical – Part 2: Calculation of optical products*. Atmospheric Measurement Techniques, 9, 3009–3029. DOI `10.5194/amt-9-3009-2016`.

### FAIR / metadata guidance

* Barker, M., Chue Hong, N. P., Katz, D. S., et al. (2022). *Introducing the FAIR Principles for research software*. Scientific Data, 9, 622. DOI `10.1038/s41597-022-01710-x`.
* CF Metadata Conventions: compliance requires deliberate validation against a named released convention/checker; MILGRAU does not currently claim full CF compliance.

## Deferred / evidence-required claims

The following remain open and do not block the current method-v5 identity:

* extending supported references higher more often without relaxing current path/QA semantics;
* SPU physical photon-counting saturation threshold and nonlinear regime;
* overlap cutoff/correction;
* any hard Rayleigh SNR threshold;
* productive cloud/layer veto beyond existing path/QA behavior;
* fitted gluing slope/intercept covariance materiality;
* independent probability model for residual aerosol fraction at the boundary;
* receiver-specific molecular lidar-ratio semantics;
* Raman retrieval and Raman-based validation of the elastic lidar-ratio/boundary assumptions.
