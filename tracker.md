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

## Status overview

| Priority | Status | Purpose |
| --- | --- | --- |
| P0 | COMPLETE | truthful backward-KFS identity and support semantics |
| P1 | COMPLETE + REAL-DATA VALIDATED | canonical Level 2 architecture |
| P2 | COMPLETE | engineering guardrails and cross-platform CI |
| P3 | IN PROGRESS — LICENSE + METADATA QA | current-method scientific + FAIR hardening |
| P4 | PARALLEL EVIDENCE WORK | instrument characterization / observational validation |
| P5 | READY FOR SCIENTIFIC R&D | altitude-resolved support + high-column R&D |
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
- [x] Re-analysis of the existing `20251107sapm` Level 0 shows 532.PC observed-rate maxima near 133–134 MHz for ordinary profiles and approximately 351 MHz for the eight anomalous profiles. Raw-rate denominator clipping identifies the same 8/167 profiles and the same maximum clipped-bin fraction (~0.00075) as the current productive diagnostic. This is observational evidence of extreme acquired count rates, not proof of a physical detector-saturation threshold.
- [ ] Regenerate `20251107sapm` Level 0/Level 1 with the new diagnostics, verify real dark `NShots`, and quantify the difference between the current productive order (`dark subtraction -> dead-time correction`) and separately normalized/corrected measurement and dark rates.
- [ ] Change the productive correction order only if the quantified evidence supports it; any such change requires explicit scientific-method/provenance versioning and regression tests.
- [ ] Characterize physical photon-counting saturation under operational SPU conditions using AN/PC overlap and preferably controlled attenuation before defining a traceable maximum rate.

### Other instrument evidence

- [ ] Characterize propagated-error SNR before enabling a hard Rayleigh SNR gate.
- [ ] Validate cloud/layer screening before productive reference-window rejection.
- [ ] Quantify gluing slope/intercept uncertainty and covariance before expanding the declared uncertainty budget.
- [ ] Compare representative retrievals with LPP and SCC/ELDA expectations without treating those comparisons as ground truth.

## P5 — altitude-resolved support + high-column R&D

Scientific P5 development may proceed in parallel with the unresolved P3 license/metadata release work. P4 evidence remains authoritative: unresolved instrument boundaries must stay provisional/unsupported rather than being invented for P5.

### P5.1 support semantics + synthetic truth

- [ ] Add synthetic cases for upper inversion boundary, lower instrument/overlap boundary, noisy tail, missing uncertainty, and internal gaps.
- [ ] `retrieval_support_flag(..., altitude)` must mean scientifically supported retrieval, not mere finiteness.
- [ ] Expose both lower and upper supported bounds; internal unsupported gaps are never bridged.
- [ ] Add pure-molecular and controlled aerosol-layer truth cases with explicit tolerances.

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

1. Complete the `20251107sapm` real-data dead-time/dark-order comparison using newly preserved dark `NShots`; do not alter the productive correction order before that evidence exists.
2. Begin P5 synthetic support/truth tests in parallel with P4 evidence work; license and metadata policy remain release gates rather than scientific-development blockers.
3. Refactor Rayleigh selection into an all-candidates -> QA -> rank pipeline after the support/truth harness is established.
4. Quantify gluing fit-parameter uncertainty, then proceed to long-mean backbone and multi-reference ensemble work.
