# MILGRAU — Scientific and operational work tracker

This tracker prioritizes **scientific validity, user operation and complete documentation** of MILGRAU. Items are checked when the stated task has actually been reviewed and completed; a passing automated test or a finished prose page is not proof of independent scientific validation.

**Current work branch:** `integration/scc-atmosphere-20261111`. The production `main` remains separate until final acceptance.

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

## Acceptance gates before production merge

- [x] Create temporary integration branch and preserve the production `main`.
- [x] Integrate current atmosphere selection and the SCC/LIBIDS work without the superseded experimental monthly GRIB planner.
- [x] Verify automated CI on the integration candidate after scientific integration changes (Linux/Windows, Python 3.12/3.14, Ruff, Explorer).
- [ ] Review the final complete diff and verify all CI checks again on the **exact final merge candidate**.
- [ ] Run representative **real-data** Level 0/1/2 scientific comparisons under identical source settings.
- [ ] Validate the source-specific atmosphere and timestamp/vertical-coverage provenance in real observations.
- [ ] Verify SCC packages, local checksums and non-destructive scientific source retention.
- [ ] Obtain separately authorized remote SCC validation on a suitable ID with completed ELPP/ELDA and downloaded outputs.
- [ ] Confirm no credentials, private datasets, generated NetCDF files, caches, logs or operational journals were committed.
- [ ] Decide on final production merge explicitly; retain scientific/operational evidence.

No remotely destructive SCC action or new CDS download is implied by any unchecked gate. An SCC upload acknowledgement alone does not complete the ELPP/ELDA validation.

---

# Extensive scientific documentation checklist

**Goal:** create a complete, independently reviewed user and scientific reference for MILGRAU, from the Licel acquisition to interpretation of aerosol optical products. Documentation must reflect the **implemented** methods and their limits; it must not imply a parameter, observational evidence or scientific comparison that has not been established.

**Documentation format:** each guide should progressively cover scientific objective, input and conditions, physical foundations, operator commands, outputs/units, important settings, QA diagnostics, assumptions/limits, a reproducible real observation and literature. The initial guide pages are now present, but the deeper evidence and example review below remain future work.

## A. General documentation and operation

- [x] Reorganize the README around scientific use and processing workflow.
- [x] Create a documentation index and a first-run guide.
- [ ] Verify installation and first-run instructions on a clean environment and representative workstation.
- [ ] Document all operator-relevant configuration choices with scientific meanings and valid settings.
- [ ] Review complete-session, single-session, local-date and file-selection examples against real data.
- [ ] Document incremental reuse, forced reprocessing and figure-only operation with before/after evidence.
- [ ] Verify output folder layout and NetCDF file naming across instrument periods and SCC modes.
- [ ] Standardize scientific terms, symbols, units, times (UTC/local) and missing-value terminology.
- [ ] Review accessibility, cross-links and examples across all main guides.

## B. LIBIDS — Level 0

- [ ] Fully describe Licel input measurements, structure and acquisition fields.
- [ ] Explain channel identification, detector modes and standardized channel mapping.
- [ ] Document continuous-session construction, actual gap thresholds and date-boundary cases.
- [ ] Explain dark-current association, time tolerance, rejected cases and propagation of source identity.
- [ ] Describe all acquisition-quality tests, rejection criteria, and quarantined-input handling.
- [ ] Explain dated instrument configuration and historical channel/calibration provenance.
- [ ] Verify solar elevation, day/night classification and contiguous segment interpretation.
- [ ] Document surface-weather data sources, available measurements and missing-data semantics.
- [ ] Produce the full Level 0 NetCDF dictionary: coordinates, variables, units, flags and global attributes.
- [ ] Provide a reproducible, scientifically annotated real Level 0 processing example.
- [ ] Compare Level 0 output and runtime performance against a validated historical baseline.

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

## Historical development evidence

The October SCC refactor chronology and superseded GRIB/CDS experiments have been retained **unchanged** in the [archived tracker](../docs/history/scc_integration_development_tracker.md). They are historical records, not the current user runbook or active atmospheric policy.

The [integration merge plan](../docs/scc_integration_merge_plan.md) records the production gates for the current branch. Any GitHub PR or source branch from earlier isolated development is not a substitute for the final integrated scientific validation.
