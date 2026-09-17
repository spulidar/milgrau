# MILGRAU scientific traceability

This document maps current productive scientific behavior to canonical code ownership, executable evidence, product provenance and the literature or standard that actually supports each claim. It is intentionally conservative: an external reference is an **implementation basis** only when MILGRAU implements the corresponding physics/method; comparison, FAIR and interoperability references are labeled separately.

## Current productive Level 2 identity

The productive Level 2 state is:

* product schema **v3** — auditable Rayleigh candidate catalogue;
* retrieval method **v4** — QA-first Rayleigh candidate selection followed by backward elastic Klett–Fernald–Sasano retrieval;
* native vertical grid;
* exact measured RCS bin at the selected reference altitude remains the productive KFS signal boundary;
* accepted candidates are ranked by `relative_slope + relative_variance`, with lower grid index as deterministic tie-breaker;
* propagated Rayleigh SNR remains diagnostic-only and is not a productive threshold.

Method v4 deliberately retains the uncertainty/support semantics established before the candidate-selection change. P5 high-column fitted-boundary, aggregation, ensemble and cascade work remains R&D until a separate acceptance/versioning decision is made.

Canonical version ownership is `milgrau/scientific.py`; schema details are in `docs/level2_schema.md`; active scientific-development decisions are in `.github/tracker.md`.

## Traceability matrix

| Scientific behavior / claim | Canonical owner | Executable evidence | Product metadata / provenance | Literature / status |
| --- | --- | --- | --- | --- |
| Level 2 product schema and retrieval-method identities are distinct and explicitly versioned | `milgrau.scientific`; dataset/metadata writers | schema/currentness/provenance tests | `level2_product_schema_version=3`, `level2_retrieval_method_version=4`, source-code identity | **MILGRAU scientific-version contract** |
| Wavelength-dependent molecular Rayleigh backscatter/extinction and molecular signal | `milgrau.level2.molecular` | `tests/test_molecular.py`, `tests/test_kfs_scientific.py` | molecular variables and molecular implementation identity | **Implementation basis:** Bucholtz (1995), DOI `10.1364/AO.34.002765` |
| Thermodynamic pressure/temperature are materialized on the Level 1 lidar altitude grid before Level 2 | `milgrau.level1.thermodynamics`; fallback physics in `milgrau.physics.atmosphere` | `tests/test_atmosphere_sources.py`, `tests/test_netcdf_contract.py` | `thermodynamic_profile_*`, molecular-source identity | ERA5 dataset identity when used; US Standard Atmosphere 1976 for explicit fallback |
| Stable atmosphere-source identity is separate from cache/download mechanics | `milgrau.provenance.thermodynamic_source_provenance` | `tests/test_thermodynamic_provenance.py` | provider/product/version/source ID | **FAIR/provenance policy**, not a retrieval equation |
| Backward elastic inversion with explicit boundary value and assumed aerosol lidar ratio | `milgrau.level2.kfs`; productive orchestration in `milgrau.level2.optical_retrieval` | `tests/test_kfs_scientific.py`, `tests/test_level2_backward_retrieval.py` | inversion method, `integration_mode=backward`, Fernald implementation identity | **Implementation basis:** Klett (1981), Fernald (1984) |
| Corrected backward molecular exponential sign | `milgrau.level2.kfs` | known-profile recovery and explicit old-sign failure regression | `fernald_implementation_version`, `fernald_scientific_change` | **Equation validation** within Klett/Fernald family |
| Elastic extinction is conditional on assumed aerosol lidar ratio | `milgrau.level2.kfs`, `milgrau.level2.optical_retrieval` | KFS synthetic recovery tests | `lidar_ratio_assumed_sr`, `lidar_ratio_std_sr`, lidar-ratio source | **Method assumption**; not direct Raman extinction |
| Productive Rayleigh catalogue enumerates complete candidate windows, applies minimum QA, then ranks accepted candidates only | `milgrau.level2.rayleigh_candidates`; orchestration in `milgrau.level2.optical_retrieval`; persistence in `milgrau.level2.rayleigh_catalogue_dataset` / `rayleigh_catalogue_output` | `tests/test_rayleigh_candidates.py`, `tests/test_level2_rayleigh_catalogue_dataset.py`, `tests/test_lebear_rayleigh_qa.py` | schema-v3 candidate catalogue, evaluated/accepted/selected flags, rejection masks, candidate metrics | **MILGRAU method-v4 policy**; implementation-specific QA |
| Productive candidate ranking uses `relative_slope + relative_variance` after QA, with deterministic lower-grid-index tie break | `milgrau.level2.rayleigh_candidates` | candidate-selection regressions | retrieval-method-v4 identity and selection-policy metadata | **MILGRAU method-v4 policy** |
| Rayleigh propagated-error SNR is diagnostic-only with no hard productive threshold | candidate catalogue / metadata | catalogue/high-column diagnostic tests | SNR policy metadata and candidate SNR fields | **Current scientific decision:** insufficient multi-regime evidence for a threshold |
| Productive KFS uses the exact measured RCS bin at the selected reference altitude | `milgrau.level2.kfs`, `milgrau.level2.optical_retrieval` | KFS and boundary R&D tests | exact selected reference altitude and method-v4 identity | **Productive method-v4 boundary semantics** |
| Window-fitted boundary, aggregation and high-column evidence are R&D-only | `milgrau.level2.window_boundary_mc`, `vertical_aggregation`, `high_column_evidence`, `high_column_export` | `tests/test_kfs_window_boundary*_rnd.py`, `tests/test_kfs_vertical_aggregation_rnd.py`, `tests/test_high_column_evidence.py`, `tests/test_high_column_export.py`, P5.4 synthetic tests | frozen R&D summaries under `docs/regression_baselines/`; no productive method change | **R&D evidence**, not method-v4 behavior |
| Signal/error temporal means use one common finite-signal + finite non-negative-uncertainty support | `milgrau.level2.block_average`; productive use in `signal_selection` | `tests/test_level2_uncertainty_support.py` | method-v4 identity with inherited uncertainty/support contract | **MILGRAU uncertainty contract** |
| Missing signal uncertainty is unsupported and never silently converted to zero MC noise | `milgrau.level2.kfs` | missing-error vs explicit-zero regressions | uncertainty/support documentation | **MILGRAU uncertainty contract** |
| Profile-to-block measurement-noise reduction uses independent-error quadrature | `milgrau.level2.block_average.mean_and_error_of_mean` | grouped/common-mask tests | `uncertainty_component_dependence` | **Declared statistical model** for retained profile-level measurement noise |
| Block-to-aggregate optical uncertainty does not assume mixed KFS nuisance terms are independent | `milgrau.level2.block_average.mean_and_correlated_error_bound`; optical aggregation | correlated-bound regressions | `optical_block_uncertainty_correlation_policy`, aggregation formula | **Conservative covariance policy:** full-positive-correlation upper bound until decomposition exists |
| KFS MC includes signal perturbation, lidar-ratio perturbation and reference-boundary perturbation but is not a total uncertainty budget | `milgrau.level2.kfs.kfs_inversion_monte_carlo` | `tests/test_kfs_scientific.py`, `tests/test_level2_uncertainty_support.py` | `uncertainty_method=Monte Carlo`, partial scope, seed/iterations/boundary model | **Declared partial model** |
| Backward retrieval support is contiguous and cannot bridge invalid signal/molecular/uncertainty gaps | `milgrau.level2.kfs`, `milgrau.level2.support`, retrieval output assembly | support/gap regressions | inversion support/top/bottom, branch flags | **Product semantics**; finite diagnostic quantities outside support are not aerosol optical retrieval |
| Analog/photon-counting gluing candidate selection and score are explicit/versioned | `milgrau.level2.gluing`; productive selection in `signal_selection` | `tests/test_gluing.py`, `tests/test_lebear_gluing_uncertainty.py` | gluing score/formula/diagnostics | **MILGRAU implementation-specific policy** |
| Glued-signal error propagates input measurement-noise terms through slope/fade weights; fitted slope/intercept covariance is excluded | `milgrau.level2.gluing.propagate_glued_error` | `tests/test_lebear_gluing_uncertainty.py` | `gluing_uncertainty_scope` | **Declared partial model**; fit-parameter materiality remains P4 work |
| Photon-counting dead-time correction and physical saturation are distinct; physical saturation remains uncharacterized | `milgrau.level1.corrections`; L2 guard in `milgrau.level2.retrieval` | PC saturation/fallback tests | saturation/dead-time diagnostics and provisional wording | **Physics context:** Donovan, Whiteway & Carswell (1993); instrument characterization still required |
| A finite scattering ratio does not imply supported aerosol retrieval | `milgrau.level2.optical_retrieval`; schema/metadata writers | NetCDF/schema/support tests | scattering-ratio descriptions; retrieval support/branch flags | **Product semantics** |
| Exact Level 1 content is part of Level 2 scientific lineage | `milgrau.provenance.file_sha256`; Level 2 currentness logic | `tests/test_level2_source_identity.py` | `source_level1_sha256` plus source filename | **FAIR/provenance implementation** |
| Different source states sharing one package version remain distinguishable | `milgrau.provenance.package_source_sha256`, `source_code_provenance` | `tests/test_source_code_provenance.py` | source-code SHA/identity plus optional repository revision | **FAIR/provenance implementation** |
| Complete/partial/failed multispectral Level 2 state is explicit and partial products are not silently reused | `milgrau.level2.completeness`, `milgrau.level2.lebear` | `tests/test_level2_completeness.py`, `tests/test_lebear_round1.py` | requested/processed/failed wavelengths, failure stage/code, completeness/status | **MILGRAU product contract** |
| SCC/ELDA comparison is methodological/interoperability validation, not numerical ground truth | external-comparison work under P4/P5 | acceptance gate not yet complete | future comparison summaries | **Comparison basis:** D'Amico et al. (2016); Mattis et al. (2016) |
| FAIR research-software metadata, licensing and provenance should be explicit | repository/release engineering | provenance tests; license decision still pending | CFF/Zenodo/release metadata | **FAIR4RS guidance:** Barker et al. (2022) |
| CF compliance is not claimed before checker-based validation against a named released convention | P3/P6 schema/release validation | compliance check pending | convention/checker version must be recorded | **Community-standard validation**, not currently a compliance claim |

## Current P5.4 observational evidence status

The active reproducible `20251107sapm` product is identified by Level 2 SHA-256 `bba3b454387ef34c9b4bee3b0ab69e286e611cc44f9a4a6358b779f6d9f236d8`. Its source Level 1 differs from the unavailable historical baseline; historical derived summaries are retained but are not treated as input-equivalent.

P5.4 evidence currently establishes, for this event only:

* candidate acceptance/persistence does not by itself guarantee a contiguous usable backward KFS path;
* native-grid weak-bin perturbations can make nominally complete ~12 km paths highly fragile;
* strict 60/120 m pre-retrieval aggregation can materially improve path robustness in this event, but gain depends strongly on covariance assumptions and no productive width is selected;
* window fitting can reduce single-bin random sensitivity but cannot repair invalid path support;
* subwindow/leave-part-out agreement cannot certify molecular purity because broad common-mode contamination can bias all subwindows together.

Primary frozen evidence is in:

* `docs/regression_baselines/20251107sapm_active_baseline_p54_summary.json`;
* `docs/regression_baselines/20251107sapm_first_high_boundary_experiment.json`;
* `docs/regression_baselines/20251107sapm_path_failure_diagnostic.json`;
* `docs/regression_baselines/20251107sapm_vertical_aggregation_support_edge.json`;
* `docs/regression_baselines/p54_covariance_leaveout_synthetic.json`.

None of those files changes productive schema v3 / method v4.

## Evidence hierarchy

1. **Analytical / synthetic truth** — primary evidence for equation correctness, estimator behavior, known physical states, controlled bias and uncertainty/support semantics.
2. **Real-data regression** — verifies behavior/stability of the actual SPU path; does not provide exact aerosol optical truth.
3. **External-chain comparison** — tests interoperability and methodological consistency under matched assumptions; agreement does not make another chain ground truth.
4. **Instrument characterization** — required for physical saturation, overlap, calibration history, detector behavior and other instrument-specific claims.

## Product-language constraints

* Elastic aerosol extinction is **conditional on the assumed aerosol lidar ratio** and is not semantically equivalent to Raman extinction.
* `uncertainty_scope` is partial; current MC/propagation terms must not be described as a complete metrological uncertainty budget.
* Finite scattering ratio or signal outside accepted inversion support is not a supported aerosol optical retrieval.
* Unsupported gaps are not interpolated merely to improve apparent coverage.
* A higher retrieval top is not an acceptance criterion by itself.
* Physical photon-counting saturation remains uncharacterized until P4 supplies instrument evidence.
* High-column R&D diagnostics must remain explicitly labeled R&D until productive semantics are versioned and accepted.

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

The following remain open even when related literature exists:

* SPU physical photon-counting saturation threshold and actual nonlinear regime;
* overlap cutoff/correction;
* dark-current versus nonlinear dead-time ordering for the real SPU chain;
* any hard Rayleigh SNR threshold;
* productive cloud/layer veto;
* fitted gluing slope/intercept covariance materiality;
* productive fitted Rayleigh boundary;
* productive vertical aggregation width;
* high-column backbone, ensemble, cascade or solution stitching;
* receiver-specific molecular lidar-ratio semantics.

Each requires its own P4/P5 evidence gate and explicit versioning if productive science changes.
