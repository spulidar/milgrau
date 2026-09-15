# MILGRAU scientific traceability

This document maps current productive scientific behavior to its canonical code owner, executable test evidence, product provenance/metadata, and the literature or standard that actually supports the claim. It is intentionally conservative: an external reference is listed as an **implementation basis** only when MILGRAU implements the corresponding physics/method. References used only for comparison, validation practice, interoperability, or FAIR guidance are labeled accordingly.

The productive Level 2 baseline described here is backward elastic Klett–Fernald–Sasano retrieval with Level 2 retrieval method version 3. Future P5 high-column/backbone/ensemble/cascade work is not part of this matrix until its own acceptance tests and product semantics exist.

## Traceability matrix

| Scientific behavior / claim | Canonical owner | Executable evidence | Product metadata / provenance | Literature / status |
| --- | --- | --- | --- | --- |
| Wavelength-dependent molecular Rayleigh backscatter/extinction and molecular signal | `milgrau.level2.molecular` | `tests/test_molecular.py`, `tests/test_kfs_scientific.py` | molecular variables; molecular atmosphere implementation identity | **Implementation basis:** Bucholtz (1995), DOI `10.1364/AO.34.002765` |
| Thermodynamic pressure/temperature are materialized on the Level 1 lidar altitude grid before L2 | `milgrau.level1.thermodynamics`; fallback physics in `milgrau.physics.atmosphere` | `tests/test_atmosphere_sources.py`, `tests/test_netcdf_contract.py` | `thermodynamic_profile_*`, molecular source identity | ERA5 dataset DOI when used; US Standard Atmosphere 1976 for explicit fallback |
| Stable atmosphere-source identity is separate from cache/download mechanics | `milgrau.provenance.thermodynamic_source_provenance` | `tests/test_thermodynamic_provenance.py` | `thermodynamic_profile_provider`, `thermodynamic_profile_product`, `thermodynamic_profile_version_or_release`, `thermodynamic_profile_source_id` | **FAIR/provenance policy**, not a retrieval equation |
| Backward elastic inversion with explicit boundary value and assumed aerosol lidar ratio | `milgrau.level2.kfs`; productive orchestration in `milgrau.level2.optical_retrieval` | `tests/test_kfs_scientific.py`, `tests/test_level2_backward_retrieval.py` | `elastic_backscatter_inversion_method`, `integration_mode=backward`, Fernald implementation identity | **Implementation basis:** Klett (1981), DOI `10.1364/AO.20.000211`; Fernald (1984), DOI `10.1364/AO.23.000652` |
| Corrected backward molecular exponential sign | `milgrau.level2.kfs` | `test_removed_backward_sign_explicitly_fails_known_nonzero_profile`; known-profile recovery tests at 355/532 nm | `fernald_implementation_version`, `fernald_scientific_change` | **Equation validation:** same Klett/Fernald family; old-sign failure retained as quantitative regression evidence |
| Elastic extinction is derived conditionally from assumed aerosol lidar ratio | `milgrau.level2.kfs` / `milgrau.level2.optical_retrieval` | KFS synthetic recovery tests | `lidar_ratio_assumed_sr`, `lidar_ratio_std_sr`, `lidar_ratio_source`; extinction variable descriptions | **Method assumption:** elastic inversion; must not be described as direct Raman extinction |
| Signal/error temporal means use one common finite signal + finite non-negative uncertainty support | `milgrau.level2.block_average`; productive use in `milgrau.level2.signal_selection` | `tests/test_level2_uncertainty_support.py` | method-v3 identity; documented missing-value/support semantics | **MILGRAU uncertainty contract**, not imported from an external algorithm |
| Missing signal uncertainty is unsupported and never silently converted to zero Monte Carlo noise | `milgrau.level2.kfs` | missing-error vs explicit-zero regressions in `tests/test_level2_uncertainty_support.py` | method-v3 identity; uncertainty/support documentation | **MILGRAU uncertainty contract** |
| Profile-to-block measurement-noise reduction uses independent-error quadrature | `milgrau.level2.block_average.mean_and_error_of_mean` | grouped/common-mask tests | `uncertainty_component_dependence` | **Declared statistical model**; applies only to retained profile-level measurement-noise terms |
| Block-to-aggregate optical uncertainty does not assume mixed KFS nuisance terms are independent | `milgrau.level2.block_average.mean_and_correlated_error_bound`; called by optical aggregation | correlated-bound regressions in `tests/test_level2_uncertainty_support.py` | `optical_block_uncertainty_correlation_policy`, exact aggregation formula | **MILGRAU conservative covariance policy:** full-positive-correlation upper bound until components are decomposed |
| KFS Monte Carlo includes signal perturbation, lidar-ratio perturbation and reference-boundary perturbation but is not a total uncertainty budget | `milgrau.level2.kfs.kfs_inversion_monte_carlo` | `tests/test_kfs_scientific.py`, `tests/test_level2_uncertainty_support.py` | `uncertainty_method=Monte Carlo`, partial uncertainty scope, MC seed/iterations, boundary model | **Declared partial model**; no claim of total metrological uncertainty |
| Analog/photon-counting gluing candidate selection and score | `milgrau.level2.gluing`; productive selection in `milgrau.level2.signal_selection` | `tests/test_gluing.py`, `tests/test_lebear_gluing_uncertainty.py` | gluing score version/formula/weights and diagnostics | **MILGRAU implementation-specific policy**; not attributed to Klett/Fernald/SCC |
| Glued-signal error propagates input measurement-noise terms through slope/fade weights; fitted slope/intercept covariance is excluded | `milgrau.level2.gluing.propagate_glued_error` | `tests/test_lebear_gluing_uncertainty.py`, method-v3 metadata tests | `gluing_uncertainty_scope` | **Declared partial model.** Fit-parameter materiality remains P4 evidence work |
| Photon-counting dead-time correction/saturation treatment is instrument-dependent and remains provisional where not characterized | `milgrau.level1.corrections`; provisional L2 guard in `milgrau.level2.retrieval` | `tests/test_level2_pc_guarded_fallback.py`, `tests/test_level2_pc_saturation_contract.py` | saturation/dead-time diagnostics and explicit provisional wording | **Physics/validation context:** Donovan, Whiteway & Carswell (1993), DOI `10.1364/AO.32.006742`; instrument characterization still required under P4 |
| Rayleigh reference-window search/QA uses explicit valid fraction, slope and variance criteria | search numerics in `milgrau.level2.molecular`; productive QA in `milgrau.level2.optical_retrieval` | `tests/test_lebear_rayleigh_qa.py`, `tests/test_rayleigh_window.py` | reference altitude/window, slope, variance, valid fraction, success diagnostics | **MILGRAU implementation-specific QA.** P5 candidate-catalogue redesign remains pending |
| A finite scattering ratio does not imply supported aerosol retrieval | `milgrau.level2.optical_retrieval`; schema/metadata in `milgrau.level2.dataset` and `milgrau.level2.metadata` | NetCDF/schema contract tests | scattering-ratio descriptions; retrieval block flags | **Product semantics**, not a literature equation |
| Exact Level 1 content is part of Level 2 scientific lineage | `milgrau.provenance.file_sha256`; `milgrau.level2.lebear.level2_output_is_current` | `tests/test_level2_source_identity.py`, currentness regressions | `source_level1_sha256` plus readable source filename | **FAIR/provenance implementation:** content identity has named consumers (lineage and cache correctness) |
| Different installed MILGRAU source states sharing one package version are distinguishable | `milgrau.provenance.package_source_sha256` / `source_code_provenance` | `tests/test_source_code_provenance.py` | `source_code_sha256`, `source_code_identity`, optional repository/build revision | **FAIR/provenance implementation:** source-content identity is primary for dev states; Git/build revision is supplementary when available |
| Level 2 complete/partial/failed multispectral state is explicit and partial products are not incrementally reusable | `milgrau.level2.completeness`, `milgrau.level2.lebear` | `tests/test_level2_completeness.py`, `tests/test_lebear_round1.py` | requested/processed/failed wavelengths, failure stage/code, product completeness/status | **MILGRAU product contract** |
| SCC/ELDA comparison is methodological/interoperability validation, not a claim of numerical identity | external comparison work under P4/P5 | not yet an acceptance gate | future validation summaries only | **Comparison references:** D'Amico et al. (2016) SCC preprocessing; Mattis et al. (2016) SCC optical products |
| Research-software metadata, license and provenance should satisfy FAIR4RS goals | repository/release engineering | provenance tests; license still pending | CFF/Zenodo/provenance/release metadata | **FAIR guidance:** Barker et al. (2022), DOI `10.1038/s41597-022-01710-x` |
| CF claims require validation against an identified released convention | schema/release validation under P3/P6 | compliance check still pending | convention/checker version must be recorded when run | **Community standard:** CF Conventions, current roadmap target release 1.13; no MILGRAU claim of full CF compliance before checker evidence |

## Evidence hierarchy

Scientific truth tests and observational regression tests serve different purposes:

1. **Analytical/synthetic truth:** known molecular-only or forward-generated aerosol states are the primary evidence for equation correctness and controlled support/uncertainty behavior.
2. **Real-data regression:** cases such as `20251107sapm` verify stability and observational behavior of an exercised path, but they are not ground truth for aerosol optical profiles.
3. **External-chain comparison:** LPP/SCC/ELDA comparisons test interoperability, expected behavior and methodological consistency. Agreement does not make one chain the numerical truth of another.
4. **Instrument characterization:** saturation, dead-time ordering, SNR/cloud gates and gluing-fit parameter uncertainty require SPU-specific evidence. A plausible constant is not evidence.

## Product-language constraints

The following wording constraints are part of scientific traceability:

- Aerosol extinction from the elastic retrieval is **conditional on the assumed aerosol lidar ratio**. It is not semantically equivalent to a directly retrieved Raman extinction product.
- `uncertainty_scope` remains partial. The current Monte Carlo and propagation terms must not be described as a complete uncertainty budget.
- A finite diagnostic signal or scattering ratio outside the accepted backward inversion support is not a supported aerosol optical retrieval.
- The target upper altitude is never permission to extrapolate, interpolate across unsupported gaps, or fabricate positive signal.
- Current physical photon-counting saturation is not considered characterized until P4 supplies instrument evidence.

## Verified bibliography

### Implemented physical / retrieval methods

- Klett, J. D. (1981). *Stable analytical inversion solution for processing lidar returns*. Applied Optics, 20(2), 211–220. DOI: `10.1364/AO.20.000211`. Role: elastic-lidar inversion family and boundary sensitivity.
- Fernald, F. G. (1984). *Analysis of atmospheric lidar observations: some comments*. Applied Optics, 23(5), 652. DOI: `10.1364/AO.23.000652`. Role: molecular/aerosol elastic inversion formulation used by the productive KFS family.
- Bucholtz, A. (1995). *Rayleigh-scattering calculations for the terrestrial atmosphere*. Applied Optics, 34(15), 2765–2773. DOI: `10.1364/AO.34.002765`. Role: wavelength-dependent molecular Rayleigh scattering physics.
- Donovan, D. P., Whiteway, J. A., & Carswell, A. I. (1993). *Correction for nonlinear photon-counting effects in lidar systems*. Applied Optics, 32(33), 6742–6753. DOI: `10.1364/AO.32.006742`. Role: photon-counting nonlinearity/dead-time scientific context; not evidence that the current SPU saturation threshold is characterized.

### External lidar-chain methodology / comparison

- D'Amico, G., Amodeo, A., Mattis, I., Freudenthaler, V., & Pappalardo, G. (2016). *EARLINET Single Calculus Chain – technical – Part 1: Pre-processing of raw lidar data*. Atmospheric Measurement Techniques, 9, 491–507. DOI: `10.5194/amt-9-491-2016`. Role: quality-controlled preprocessing/traceability and uncertainty methodology used as external comparison context.
- Mattis, I., D'Amico, G., Baars, H., Amodeo, A., Madonna, F., & Iarlori, M. (2016). *EARLINET Single Calculus Chain – technical – Part 2: Calculation of optical products*. Atmospheric Measurement Techniques, 9, 3009–3029. DOI: `10.5194/amt-9-3009-2016`. Role: external optical-product retrieval/quality methodology and synthetic-validation comparison context.

### FAIR / metadata standards

- Barker, M., Chue Hong, N. P., Katz, D. S., et al. (2022). *Introducing the FAIR Principles for research software*. Scientific Data, 9, 622. DOI: `10.1038/s41597-022-01710-x`. Role: FAIR4RS basis for persistent identity, rich metadata, license and detailed provenance requirements.
- CF Metadata Conventions. Release 1.13 is the current released convention targeted by the roadmap for explicit compliance checking. Role: domain metadata/community-standard validation. MILGRAU must record the convention and checker version when a compliance claim is made.

## Deferred / evidence-required items

The following intentionally do **not** receive a literature-backed “complete” status merely because a related paper exists:

- SPU photon-counting physical saturation rate and any productive threshold;
- dark-current subtraction versus nonlinear dead-time ordering for the actual SPU acquisition chain;
- hard Rayleigh-window propagated-error SNR threshold;
- cloud/layer veto policy for productive reference selection;
- fitted gluing slope/intercept uncertainty materiality and covariance;
- future high-column reference catalogue/backbone/ensemble/cascade behavior.

Each remains subject to its P4/P5 acceptance gate and must be versioned if it changes productive scientific semantics.
