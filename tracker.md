# MILGRAU scientific engineering roadmap

Branch: `new-architecture`

Current method-v3 observational baseline: `docs/regression_baselines/20251107sapm_method_v3.json`

This file is the active source of truth for current scientific/engineering gates. Detailed completed history remains recoverable in Git.

## Non-negotiable rules

- `config.yaml` owns the processing/scientific recipe; `station.yaml` owns site/instrument reality, station-derived observations, calibration state and provisional instrument estimates; Python owns equations and generic implementation.
- Missing uncertainty is never zero uncertainty.
- Correlated/model/systematic uncertainty is not silently treated as independent noise.
- Unsupported data remain unsupported; no fill/interpolation is introduced merely to extend retrieval coverage.
- A successful real-data run is regression/behavior evidence, not ground truth or proof of equation correctness.
- Scientific support is not equivalent to `isfinite(product)`.
- Vertical support must respect both the upper inversion/reference boundary and the lower instrument-validity boundary.
- Instrument constants/thresholds are evidence-derived or explicitly provisional/disabled; they are never invented to close the roadmap.
- Diagnostic instrument models must not silently become productive corrections.
- NetCDF products should be readable and self-describing, but MILGRAU currently makes no formal metadata-convention conformance/alignment claim.
- Interoperable file syntax does not replace station/instrument knowledge: external SCC raw data must resolve through explicit station-owned channel identity and calibration metadata rather than permissive guessing.

## Status overview

| Priority | Status | Purpose |
| --- | --- | --- |
| P0 | COMPLETE | truthful backward-KFS identity and support semantics |
| P1 | COMPLETE + REAL-DATA VALIDATED | canonical Level 2 architecture |
| P2 | COMPLETE | engineering guardrails and cross-platform CI |
| P3 | IN PROGRESS — LICENSE + METADATA QA | current-method scientific + FAIR hardening |
| P4 | PARALLEL EVIDENCE WORK | instrument characterization / observational validation |
| P5 | IN PROGRESS — SUPPORT CONTRACT | altitude-resolved support + high-column R&D |
| P6 | PENDING | reproducible release/publication process |

## Current productive baseline

The productive baseline is backward Klett–Fernald–Sasano, Level 2 schema v1, retrieval method v3. The real-data regression baseline `20251107sapm` is frozen from revision `842389a23c3ddc2006243db9b7d98bf200c90c0f` and is explicitly observational, not ground truth.

Key current limitations remain: elastic extinction is conditional on assumed lidar ratio; physical photon-counting saturation is uncharacterized; fitted gluing slope/intercept uncertainty is outside the current partial budget; cloud/SNR Rayleigh gates are not yet evidence-backed; near-field overlap is not experimentally characterized.

## P3.6 — FAIR software license + NetCDF metadata QA

### Software license

- [x] Code copyright holders identified by project owner as the four code authors: Luisa Mello, Fábio J. S. Lopes, Alexandre C. Yoshida, and Alexandre Cacheffo.
- [ ] Select the software license deliberately. The earlier provisional BSD-3-Clause choice was withdrawn on 2026-09-15 and is not current project policy.
- [ ] After selection, add the canonical root `LICENSE` and align `CITATION.cff` and package metadata.
- [ ] Keep software, documentation, and data licensing separate unless the project explicitly chooses otherwise.

### NetCDF metadata policy

- [x] Adopt `docs/netcdf_metadata_policy.md`: products should be readable, self-describing, and technically unambiguous without claiming conformance or alignment with an external metadata convention.
- [x] Do not add a global `Conventions` attribute unless a future project decision deliberately adopts a specific convention/version and its maintenance obligations.
- [x] Keep clear coordinate semantics, physical units where meaningful, explicit unit status for instrument-native quantities, machine-readable flags, honest missing values, and readable provenance.
- [ ] Audit representative Level 2 metadata for ambiguity/inconsistency and fix genuine readability/interoperability problems. External checkers may be used as optional linting aids, not as a project certification gate.

P3 acceptance gate: **open on deliberate software-license selection plus a recorded representative NetCDF metadata review. This gate does not block scientific P4/P5 development; it blocks release/FAIR completion.**

## P3.7 — SCC raw Level 0 interoperability

- [x] Document the interoperability contract in `docs/scc_level0_interoperability.md`.
- [x] Keep MILGRAU canonical physical channel names (`532.PC`, `355.AN`, etc.) as the calibration-facing internal identity while accepting numeric SCC `channel_ID` at ingestion.
- [x] Canonicalize external numeric SCC channel IDs through the temporally valid `station.yaml` SCC mapping before Level 0 validation and Level 1 corrections; do not infer wavelength/detector identity from opaque IDs alone.
- [x] Verify `channel_string` against `channel_ID` when both are present, as in MILGRAU's own `*_scc.nc` output.
- [x] Reject unknown IDs, duplicate IDs and genuinely ambiguous day/night/configuration mappings; optional `Measurement_ID` and `SCC_Configuration_ID` hints may disambiguate without guessing.
- [x] Wire the station catalog into the productive LIPANCORA ingestion path.
- [x] Preserve default discovery behavior so a colocated full Level 0 and `_scc.nc` are not both processed automatically. The SCC subset is an explicit input when both products coexist.
- [x] Keep canonical `*_scc.nc` Level 1 output inside the original measurement directory; do not create a separate `_scc` measurement tree. Explicit non-canonical external SCC files write Level 1 beside the source rather than deriving a fake date hierarchy from the filename.
- [x] Validate MILGRAU's own `20251107sapm_scc.nc` through an end-to-end explicit LIPANCORA run. The resulting five-channel Level 1 product (`532.AN`, `532.PC`, `1064.AN`, `355.PC`, `355.AN`) is exactly equal, channel-for-channel, to the corresponding arrays/diagnostics in the full-channel Level 1 product, including corrected signal/error, RCS/error, PC masks and rate/clipping diagnostics, bin-shift diagnostics, PBL, altitude/time coordinates, and thermodynamic profiles. This is regression/behavior evidence for the MILGRAU SCC path, not proof that arbitrary SCC converters are equivalent.
- [ ] Add broader external-converter fixtures only from real SCC raw files. `channel_string_ID` remains outside the productive mapping path until a traceable station-owned use case requires it.
- [ ] External SCC dark profiles that lack `Background_Laser_Shots` remain usable by the current productive dark-profile path, but cannot support separately shot-normalized dark dead-time diagnostics without another traceable dark-shot source.

## P4 — instrument characterization / observational validation

### Overlap

- [x] Generic `coaxial_uniform_disk_geometric_v1` diagnostic model implemented; SPU-specific values remain in `station.yaml`.
- [x] Current provisional receiver geometry uses 30 cm telescope, ~4 cm beam, 0.10 mrad divergence model value, and inferred 0.78 mrad full-angle FOV, giving ~500 m model full overlap. With `f=1.5 m`, this implies ~1.17 mm field stop, consistent with the operator estimate of roughly 1 mm.
- [x] Current overlap remains `estimated_unvalidated` and `diagnostic_only_no_correction`; no signal correction or validated lower retrieval-support boundary is inferred from the model.
- [ ] Experimentally determine FOV convention/value, field-stop diameter, beam diameter convention, wavelength-dependent divergence, alignment/separation, and temporal stability.
- [ ] Validate with telecover/alignment mapping and preferably an independent horizontal/Raman-based overlap method before any productive correction/support boundary is introduced.

### Photon-counting dead time / saturation

- [x] Preserve the maximum directly observed PC rate per profile in Level 1 before dark-current subtraction, dead-time correction, bin shift or background subtraction.
- [x] Evaluate and persist dead-time denominator/clipping diagnostics on that raw observed rate separately from the current productive dark-subtracted path.
- [x] A future characterized physical saturation mask uses the observed pre-dark PC rate. No SPU saturation limit is invented or enabled by this change.
- [x] Preserve parsed dark-acquisition `NShots` as optional `Background_Laser_Shots(time_bck, channels)` in newly generated Level 0 files, so dark PC rates can be normalized independently.
- [x] Re-analysis of `20251107sapm` confirms 532.PC observed-rate maxima near 133–134 MHz for ordinary profiles and approximately 350–351 MHz for eight anomalous profiles. Raw-rate and current productive denominator clipping identify the same 8/167 profiles. This is observational evidence of extreme acquired count rates, not proof of a physical detector-saturation threshold.
- [x] Real rerun from revision `1194efc9227d31f95ce7670cd2258745dd7201cd` preserves measurement and dark shot counts (both 3001–3002 shots). 532.PC dark observed rate reaches about 15.75 MHz and produces no numerical dead-time clipping.
- [x] Quantified `20251107sapm` correction-order sensitivity. Comparing the current productive `DT(measurement - mean_dark)` path with separately shot-normalized/corrected `DT(measurement) - mean(DT(dark))`, after the current bin shift and background removal, gives median absolute difference ~8.5e-7 MHz, 99th percentile ~4.2e-4 MHz and maximum ~0.051 MHz. Where the current corrected signal magnitude exceeds 1 MHz, the relative-difference median is ~5.5e-7, 99th percentile ~2.5e-5 and maximum ~2.0e-4 (0.020%). This is one-day observational sensitivity evidence, not a universal detector model.
- [x] Retain the existing productive dark-before-dead-time order for method v3 for now: this representative dataset provides no material retrieval-range benefit from changing order, while a change would create a new scientific method/provenance state. Continue to preserve the diagnostics needed for broader evidence.
- [ ] Characterize physical photon-counting saturation under operational SPU conditions using AN/PC overlap and preferably controlled attenuation before defining a traceable maximum rate.

### Other instrument evidence

- [ ] Characterize propagated-error SNR before enabling a hard Rayleigh SNR gate.
- [ ] Validate cloud/layer screening before productive reference-window rejection.
- [ ] Quantify gluing slope/intercept uncertainty and covariance before expanding the declared uncertainty budget.
- [ ] Compare representative retrievals with LPP and SCC/ELDA expectations without treating those comparisons as ground truth.

## P5 — altitude-resolved support + high-column R&D

Scientific P5 development may proceed in parallel with the unresolved P3 license/metadata release work. P4 evidence remains authoritative: unresolved instrument boundaries must stay provisional/unsupported rather than being invented for P5.

### P5.1 support semantics + synthetic truth

- [x] Existing independent elastic forward-model tests already cover pure-molecular truth, nonzero controlled aerosol layers at 355/532 nm, variable lidar ratio, exact-boundary recovery and vertical-grid convergence with explicit numerical tolerances. These are synthetic truth tests, not observational golden-product checks.
- [x] Add a pure backward-support contract in `milgrau.level2.support` without yet exposing new NetCDF schema fields.
- [x] Synthetic support tests cover a known lower instrument boundary, upper inversion boundary, noisy/missing upper tail, missing uncertainty, negative uncertainty and internal value gaps.
- [x] Frozen backward semantics: support is the contiguous valid path ending at the accepted upper inversion/reference boundary; an unsupported internal bin cannot be bridged to declare lower bins supported.
- [x] Support requires finite value, finite non-negative uncertainty and any supplied validated instrument mask. `instrument_valid=None` is explicitly not a claim that overlap/instrument limits are characterized.
- [ ] Integrate the tested support contract into productive Level 2 assembly only after CI is green and the exact block/aggregate support source is reviewed.
- [ ] Expose `retrieval_support_flag(..., altitude)`, lower supported altitude and upper supported altitude only after productive integration tests demonstrate that schema output matches the frozen support semantics.

### P5.2 Rayleigh candidate catalogue

- [ ] Enumerate all geometrically/data-valid reference candidates.
- [ ] Apply pass/fail QA before ranking, then rank only accepted candidates.
- [ ] Persist candidate window, valid fraction, slope, variance, calibration, intercept, uncertainty/SNR and future layer diagnostics.
- [ ] Define separation/correlation rules so overlapping windows do not masquerade as independent references.

### P5.3 backbone + ensemble

- [ ] Build a long-mean/backbone signal using the established common signal/error support rules.
- [ ] Select multiple accepted high references and run auditable backward KFS ensemble members.
- [ ] Combine only members whose physical support contains the altitude and whose QA passed.
- [ ] Expose between-reference spread as uncertainty rather than averaging ambiguity away.
- [ ] Evaluate cascade only after backbone + ensemble demonstrate a remaining scientific need.

## P6 — release/publication readiness

- [ ] Explicit software license selected and reflected consistently in root license/citation/package metadata.
- [ ] `CITATION.cff` version/date/DOI aligned with the actual release.
- [ ] Scientific method/schema changes summarized in release notes.
- [ ] Frozen reference scientific environment/constraints.
- [ ] Build sdist/wheel in CI and test clean installation from artifacts.
- [ ] Core-science coverage report and deliberate warning policy.
- [ ] End-to-end Level 0 -> Level 2 run from release artifact/environment.
- [ ] Recorded representative NetCDF metadata review.
- [ ] Frozen machine-readable synthetic acceptance and observational regression summaries.
- [ ] Reconcile branch history with `main` before release/merge.

## Immediate next gate

1. Let the SCC-output-path and P5 support-contract tests pass the full cross-platform CI; fix genuine regressions before product-schema integration.
2. Integrate the tested support contract into Level 2 assembly and add schema/output tests before exposing support/bottom/top variables.
3. Refactor Rayleigh selection into an all-candidates -> QA -> rank pipeline after the support contract is product-integrated.
4. Quantify gluing fit-parameter uncertainty, then proceed to long-mean backbone and multi-reference ensemble work.
