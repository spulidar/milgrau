# MILGRAU — Scientific and operational work tracker

This tracker prioritizes **scientific validity, user operation and complete documentation** of MILGRAU. Items are checked when the stated task has actually been reviewed and completed; a passing automated test or a finished prose page is not proof of independent scientific validation.

**Current branch:** `main`. The SCC/atmosphere integration was fast-forward merged on 11 November 2026 at commit `1bfec134e8a0246d733dbe0ec31bd120383f2559`. Future documentation improvements and real-data scientific validation continue on `main`.

## Current processing state

| Area | Current operational state | Still to validate |
| --- | --- | --- |
| **LIBIDS / Level 0** | Continuous Licel sessions, historical instrument context, acquisition QA and reconstructible header cache | Equivalence and performance on representative real historical acquisitions |
| **LIPANCORA / Level 1** | Corrected signals, RCS, uncertainty and hourly thermodynamic atmosphere | Instrument-sensitive cases, real ERA5/Wyoming/IGRA2 comparisons and fallback behavior |
| **Atmospheric source selection** | ERA5 → Wyoming → local IGRA2 → USSA76, including explicit no-CDS mode and public queue heuristic | Field operation, queue-threshold suitability and scientific effects of source changes |
| **LEBEAR / Level 2** | Elastic KFS optical retrieval with molecular reference, uncertainty and validity diagnostics | Representative cross-era optical comparisons and instrument-limit evidence |
| **SCC interoperability** | Opt-in local staging/status/download tools; observed sounding only, with provider/hash verification | Authorized real-data SCC input/output comparison and completed ELPP/ELDA case |
| **Documentation** | Reorganized README and initial user/scientific guides | Extensive method-by-method technical/scientific confirmation and examples |

For the current processing behavior, read [MILGRAU documentation](../docs/README.md). For the original SCC development log and the superseded experimental GRIB/CDS planning, see the [archived development tracker](../docs/history/scc_integration_development_tracker.md).

## Integration completed; outstanding scientific validation gates

- [x] Create temporary integration branch and preserve the production `main`.
- [x] Integrate current atmosphere selection and the SCC/LIBIDS work without the superseded experimental monthly GRIB planner.
- [x] Verify automated CI on the integration candidate after scientific integration changes (Linux/Windows, Python 3.12/3.14, Ruff, Explorer).
- [x] Validate the exact integrated commit in cross-platform CI (6/6 checks, 780 Linux tests).
- [ ] Run representative **real-data** Level 0/1/2 scientific comparisons under identical source settings.
- [ ] Validate the source-specific atmosphere and timestamp/vertical-coverage provenance in real observations.
- [ ] Verify SCC packages, local checksums and non-destructive scientific source retention.
- [ ] Obtain separately authorized remote SCC validation on a suitable ID with completed ELPP/ELDA and downloaded outputs.
- [ ] Confirm no credentials, private datasets, generated NetCDF files, caches, logs or operational journals were committed.
- [x] Merge integrated code and documentation by fast-forward into `main`, preserving commit history.
- [ ] Perform final real-data scientific and remote SCC acceptance before claiming validated production use.

No remotely destructive SCC action or new CDS download is implied by any unchecked gate. An SCC upload acknowledgement alone does not complete the ELPP/ELDA validation.

---

# Extensive scientific documentation checklist

**Goal:** create a complete, independently reviewed user and scientific reference for MILGRAU, from the Licel acquisition to interpretation of aerosol optical products. Documentation must reflect the **implemented** methods and their limits; it must not imply a parameter, observational evidence or scientific comparison that has not been established.

**Documentation format:** each guide should progressively cover scientific objective, input and conditions, physical foundations, operator commands, outputs/units, important settings, QA diagnostics, assumptions/limits, a reproducible real observation and literature. The initial guide pages are now present, but the deeper evidence and example review below remain future work.

## A. General documentation and operation

- [x] Reorganize the README around scientific use and processing workflow.
- [x] Create a documentation index and a first-run guide.
- [x] Publish a code-reviewed operational command/selection guide for the current main branch.
- [x] Publish a first-pass scientific configuration reference with current distributed values and cautions.
- [ ] Verify installation and first-run instructions on a clean environment and representative workstation.
- [ ] Document all operator-relevant configuration choices with scientific meanings and valid settings.
- [ ] Review complete-session, single-session, local-date and file-selection examples against real data.
- [ ] Document incremental reuse, forced reprocessing and figure-only operation with before/after evidence.
- [ ] Verify output folder layout and NetCDF file naming across instrument periods and SCC modes.
- [ ] Standardize scientific terms, symbols, units, times (UTC/local) and missing-value terminology.
- [ ] Review accessibility, cross-links and examples across all main guides.

## B. LIBIDS — Level 0

- [x] Fully describe Licel input measurements, structure and acquisition fields.
- [x] Explain channel identification, detector modes and standardized channel mapping.
- [x] Document continuous-session construction, actual gap thresholds and date-boundary cases.
- [x] Explain dark-current association, time tolerance, rejected cases and propagation of source identity.
- [x] Describe all acquisition-quality tests, rejection criteria, and quarantined-input handling.
- [x] Explain dated instrument configuration and historical channel/calibration provenance.
- [ ] Verify solar elevation, day/night classification and contiguous segment interpretation.
- [x] Document surface-weather data sources, available measurements and missing-data semantics.
- [x] Produce the full Level 0 NetCDF dictionary: coordinates, variables, units, flags and global attributes.
- [ ] Provide a reproducible, scientifically annotated real Level 0 processing example.
- [ ] Compare Level 0 output and runtime performance against a validated historical baseline.

**Block A/B documentation batch (11 November 2026):** Added
[operating commands](../docs/operating_milgrau.md),
[scientific configuration reference](../docs/configuration_reference.md),
[expanded LIBIDS guide](../docs/libids.md) and the
[Level 0 NetCDF dictionary](../docs/level0_schema.md).
Completed B items above mean **code-verified descriptions**, not real-data
experimental validation. Clean-environment execution, all-era sample runs,
independent solar numerical checks, baseline equivalence/performance and
cross-guide editorial acceptance remain unchecked.

## C. LIPANCORA — Level 1

- [ ] Explain scientific objectives and required input quality.
- [ ] Document each applicable instrumental correction and where historical parameters come from.
- [ ] Explain analog and photon-counting treatment and channel alignment.
- [ ] Describe dead-time assumptions, bin shifts and the limits of saturation characterization.
- [ ] Explain robust background estimation (range, MAD, outliers, count and uncertainty diagnostics).
- [ ] Document propagation of corrected-signal/RCS uncertainty and unsupported cases.
- [ ] Provide the range-corrected-signal derivation, definitions and unit conventions.
- [ ] Document PBL/tropopause diagnostics and their failure/availability semantics.
- [ ] Explain hourly molecular atmosphere materialization and altitude/time reference.
- [ ] Document and interpret every Level 1 figure and diagnostic.
- [ ] Produce the full Level 1 NetCDF dictionary with units, flags and relevant provenance.
- [ ] Provide a reproducible, scientifically annotated real Level 1 example.
- [ ] Assess instrumental edge cases and scientific differences across historical station profiles.

## D. Molecular atmosphere and meteorological sources

- [ ] Explain the relationship of pressure/temperature to molecular density and Rayleigh terms.
- [ ] Document ERA5 fields, temporal/vertical mapping, coverage and reanalysis limitations.
- [ ] Explain University of Wyoming observed-sounding selection and station representativeness.
- [ ] Explain local historical NOAA IGRA2 use, archive preparation and observed launch identity.
- [ ] Explain USSA76 as a modeled whole-hour fallback or an explicit vertical extension.
- [ ] Verify source preference and per-hour fallback behavior under realistic missing-data conditions.
- [ ] Document global CDS queue observation, its non-guaranteed nature and threshold calibration.
- [ ] Validate the practical `--radiosonde-only` workflow including 12-hour launch selection.
- [ ] Demonstrate the atmospheric time/height interpolation with an actual observation.
- [ ] Verify per-hour source labels, radiosonde cache/provider identity, source offset and USSA76 fraction.
- [ ] Compare ERA5, Wyoming, IGRA2 and USSA76 in real cases and discuss resulting molecular differences.
- [ ] Demonstrate how a change in atmosphere source affects Level 1 and Level 2 scientific outputs.

## E. LEBEAR — Level 2

- [ ] Define physical backscatter, extinction and scattering ratio with units and interpretation.
- [ ] Explain time-block reduction, segment rules, support and progressive vertical resolution.
- [ ] Document analog/photon-counting gluing and allowed single-channel fallback.
- [ ] Derive molecular backscatter/extinction calculations from saved Level 1 conditions.
- [ ] Document molecular Rayleigh reference search and scientific acceptance diagnostics.
- [ ] Explain the physical/mathematical basis of two-sided Klett–Fernald–Sasano inversion.
- [ ] Explain backward/forward valid paths, endpoints, gaps and fixed-height limitations.
- [ ] Document monthly aerosol lidar-ratio assumptions and their unresolved original derivation.
- [ ] Explain the effect of lidar-ratio/reference assumptions on retrieved extinction/backscatter.
- [ ] Document selection-aware Monte Carlo and interpretation of random versus systematic limitations.
- [ ] Explain finite signals versus valid optical retrievals, complete/partial/failed products.
- [ ] Produce a full Level 2 dictionary of variable/coordinate/attribute units and quality semantics.
- [ ] Provide an annotated real example of retrieved profiles and uncertainty interpretation.
- [ ] Compare selected real observations with supported independent or external-chain evidence.

## F. SCC — scientific interoperability

- [ ] Explain the SCC purpose and differences between its processing and MILGRAU scientific products.
- [ ] Document SCC inputs produced from long continuous sessions and day/night partitions.
- [ ] Explain lidar-ratio and observed-sounding ancillary inputs with actual scientific source identity.
- [ ] Validate Wyoming/IGRA2 sounding selection, station/time/provenance and retained hash evidence.
- [ ] Demonstrate safe local SCC input validation and dry-run.
- [ ] Document intentional uploads, remote status and finished-result download with one approved ID.
- [ ] Explain ELPP/ELDA outputs and success/incomplete/failure states using actual remote examples.
- [ ] Document historical data retention and no-silent-replacement rules for operators.
- [ ] Prepare a real MILGRAU-versus-SCC scientific comparison, with matched input/method assumptions.

## G. Results, visualization and reproducibility

- [ ] Document the available L0/L1/L2 figures and how to read axes, units and support.
- [ ] Validate Explorer operation and its interpretation limits using real products.
- [ ] Explain all quality flags, absent values, incomplete products and `NaN` semantics.
- [ ] Describe scientific provenance fields and procedures for inspecting source identity.
- [ ] Summarize physical and methodological limitations of each reported product.
- [ ] Provide reproducible examples covering different historical station configurations.
- [ ] Verify every published example command against the integrated branch.
- [ ] Check references, equations, physical symbols, units and DOI provenance.
- [ ] Perform a complete independent scientific review of the documentation.
- [ ] Mark each documentation guide "reviewed" only after real-case validation and reference checking.

## Required structure and completion criteria

A section should be marked complete only after it provides, where appropriate:

- Scientific objective and physical basis.
- Necessary observations, instrument context and configuration.
- Operator commands that were tested.
- Outputs, units and interpretation.
- QA criteria, assumptions and limitations.
- At least one representative real observation and reproducible result.
- Supporting scientific references and review date.

**Status:** first-pass guides and revised README written; extensive validation, detailed variable dictionaries and real-data worked examples remain pending.

---

## Cleanup and deprecation review (non-destructive)

The [cleanup review register](../docs/cleanup_review.md) is maintained alongside
documentation. Do **not** delete code merely because a module is unused by
the current CLI: it may be a supported scientific API or retained method.

- [ ] Determine whether `milgrau/level0/solar.py` duplicates the productive
  solar/segment logic, including all imports and downstream compatibility.
- [ ] Determine whether preliminary `level2.cloud_screening` should remain
  an explicitly disabled research API, be integrated or be retired.
- [ ] Review historical SCC/ERA5/GRIB documentation links and maintain
  archived scientific provenance rather than deleting the history.
- [ ] Review `Processing_level` attribute wording for canonical L0 versus SCC
  interoperability while preserving product-reader compatibility.
- [ ] Check any classification ambiguity caused by raw paths containing
  the string `dark` before changing the observed data-selection behavior.
- [ ] Prepare any code cleanup as a separate, test-backed change with explicit
  approval; no remote SCC records, raw data or provenance archives removed.

## Historical development evidence

The October SCC refactor chronology and superseded GRIB/CDS experiments have been retained **unchanged** in the [archived tracker](../docs/history/scc_integration_development_tracker.md). They are historical records, not the current user runbook or active atmospheric policy.

The [historical integration merge plan](../docs/scc_integration_merge_plan.md) records the former merge gates. Remaining **real-data scientific and SCC validation** gates stay active here even though the code is now on `main`.
