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
- A requested high-altitude target is a target, not permission to extrapolate. The product stops lower whenever the data do not support a valid inversion above that altitude.
- Backward retrieval above its accepted boundary is never implied. A finite signal/scattering-ratio diagnostic above the boundary is not an aerosol optical retrieval.
- A future cascade must inherit its lower-segment boundary from an accepted upper solution; it must never reset `SR_ref = 1` merely because a new segment begins.
- Multiple accepted reference solutions are a sensitivity/ensemble experiment, not independent truths. Their disagreement must remain visible and contribute to uncertainty.
- Elastic extinction is conditional on the assumed aerosol lidar ratio and is never presented as an independently measured extinction profile.
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
| P5 | IN PROGRESS — HIGH-COLUMN REDESIGN | altitude-resolved support + Rayleigh catalogue + backbone/ensemble R&D |
| P6 | PENDING | reproducible release/publication process |

## Current productive baseline

The productive baseline is backward Klett–Fernald–Sasano, Level 2 schema v1, retrieval method v3. The real-data regression baseline `20251107sapm` is frozen from revision `842389a23c3ddc2006243db9b7d98bf200c90c0f` and is explicitly observational, not ground truth.

For the baseline case, the accepted block references are near 5.6–6.3 km and the aggregate finite optical-product tops are 6101.25 m (355 nm) and 6318.75 m (532 nm). Those tops are a consequence of the current single-reference backward solution and are not instrument-validated near-field support claims.

Key current limitations remain: elastic extinction is conditional on assumed lidar ratio; physical photon-counting saturation is uncharacterized; fitted gluing slope/intercept uncertainty is outside the current partial budget; cloud/SNR Rayleigh gates are not yet evidence-backed; near-field overlap is not experimentally characterized.

The high-column redesign goal is to maximize **validated vertical optical support**, with an initial design target of 20 km when the measurement genuinely supports it. Reaching 20 km is not itself an acceptance criterion.

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

## P5 — altitude-resolved support + high-column Level 2 redesign

Scientific P5 development may proceed in parallel with unresolved P3 release work. P4 evidence remains authoritative: unresolved instrument boundaries must stay provisional/unsupported rather than being invented for P5.

### P5.0 frozen evidence + physical Rayleigh-window geometry

- [x] Freeze the real `20251107sapm` method-v3 observational baseline in `docs/regression_baselines/20251107sapm_method_v3.json`, including product hashes, selected references, valid KFS block counts and current finite tops.
- [x] Keep synthetic truth separate from observational regression evidence.
- [x] Express the Rayleigh reference-window width physically as `inversion.molecular_fit.ref_window_m` rather than as a scientific bin-count knob.
- [x] Convert physical window width deterministically on the actual uniform altitude grid; 1000 m resolves to 133 bins at 7.5 m spacing and 67 bins at 15 m spacing.
- [x] Keep search bounds in meters and fail explicitly when the physical window cannot be represented on the grid.

### P5.1 support semantics + measurable vertical coverage

- [x] Existing independent elastic forward-model tests cover pure-molecular truth, nonzero controlled aerosol layers at 355/532 nm, variable lidar ratio, exact-boundary recovery and vertical-grid convergence with explicit numerical tolerances. These are synthetic truth tests, not observational golden-product checks.
- [x] Add a pure backward-support contract in `milgrau.level2.support`.
- [x] Synthetic support tests cover a known lower instrument boundary, upper inversion boundary, noisy/missing upper tail, missing uncertainty, negative uncertainty and internal value gaps.
- [x] Frozen backward semantics: support is the contiguous valid path ending at the accepted upper inversion/reference boundary; an unsupported internal bin cannot be bridged to declare lower bins supported.
- [x] Support requires finite value, finite non-negative uncertainty and any supplied validated instrument mask. `instrument_valid=None` is explicitly not a claim that overlap/instrument limits are characterized.
- [x] Full cross-platform CI for the support contract and SCC-output-path work is green at HEAD `5d03b84c6a4512f917358cd8b628ef51639713b1`.
- [ ] Integrate **inversion support** into productive Level 2 assembly at block and aggregate level.
- [ ] Expose `retrieval_inversion_support_flag(..., altitude)` for the contiguous backward path supported by retrieval value/error and the accepted upper boundary. This flag must explicitly **not** claim a characterized near-field instrument boundary.
- [ ] Expose `retrieval_top_altitude_m[wavelength]` from aggregate inversion support and block counterparts where useful.
- [ ] Expose `retrieval_inversion_bottom_altitude_m` only as an algorithmic/inversion-domain diagnostic; do not call it validated scientific near-field support while overlap is uncharacterized.
- [ ] Expose an altitude-resolved effective successful-block count so high-altitude coverage supported by 1/5 blocks is distinguishable from coverage supported by 5/5 blocks.
- [ ] Reserve a future stricter `retrieval_support_flag` for the conjunction of inversion support **and validated instrument support**. Do not publish this stronger claim until the lower instrument-validity boundary is evidence-backed.
- [ ] Add schema/contract tests proving that product support fields exactly reproduce the frozen support semantics and never bridge gaps.

P5.1 acceptance gate: vertical-coverage metrics distinguish “more finite bins” from “more inversion-supported bins” and do not misrepresent the unresolved overlap boundary.

### P5.2 Rayleigh candidate catalogue — QA first, ranking second

Current limitation: the productive selector minimizes a cost over geometrically/data-valid windows and applies full configured Rayleigh QA only to that one selected window. This can hide acceptable higher reference regions and can also fail after selecting one candidate even when another candidate would pass.

- [ ] Introduce a typed Rayleigh candidate contract containing at least start/stop/center altitude, valid-bin count/fraction, relative slope, relative variance, origin-constrained calibration factor, free-intercept diagnostic, and explicit acceptance/rejection reasons.
- [ ] Enumerate **all** geometrically/data-valid reference candidates inside the configured search interval.
- [ ] Compute diagnostics for every candidate before selection.
- [ ] Apply minimum scientific pass/fail QA before ranking; rejected candidates must never participate in ranking.
- [ ] Rank only accepted candidates. Higher altitude may be preferred only after minimum quality gates pass; do not chase altitude by ranking a poor high window over a demonstrably molecular lower one.
- [ ] Persist an auditable candidate catalogue or compact reproducible representation sufficient to explain why references were accepted/rejected and why selected references were preferred.
- [ ] Add uncertainty/SNR diagnostics to candidates using propagated uncertainties, but do not enable a hard SNR gate until SPU evidence validates the criterion.
- [ ] Add cloud/layer contamination diagnostics when available, but keep productive rejection disabled until validated on SPU observations.
- [ ] Define minimum separation/correlation rules before treating nearby overlapping accepted windows as distinct ensemble members.

P5.2 acceptance gate: for any block/backbone profile, MILGRAU can explain every accepted/rejected reference candidate and the final ranking without hidden heuristic defaults.

### P5.3 real-data candidate experiment before algorithmic complexity

Primary case: `20251107sapm`.

- [ ] Run the full candidate catalogue on each 20-minute block at 355 and 532 nm using the existing 5–25 km search interval and 1 km physical window.
- [ ] Record the highest QA-passing candidate, the best-ranked candidate, candidate density versus altitude, and the limiting diagnostic for rejected high-altitude windows.
- [ ] Determine whether the current ~6 km top is primarily caused by signal quality or by the current single-minimum-cost selector.
- [ ] Compare redesigned candidate selection below the current ~6 km baseline; differences must be physically explainable and within defined synthetic/uncertainty tolerances.

P5.3 acceptance gate: do not implement a backbone/cascade merely because it was in an old design. First show from candidate evidence what actually limits the real measurement.

### P5.4 high-column backbone

Use only if the block-level candidate experiment shows that high-altitude support is limited by block SNR/variance rather than by an absence of usable molecular information.

- [ ] Add a distinct long-mean/high-column retrieval input separate from the 20-minute block products.
- [ ] Make backbone averaging duration/configuration explicit and persist it in provenance. Do not choose the duration by convenience alone.
- [ ] Build the long-mean signal and error on common support using the established missing-uncertainty rules.
- [ ] Preserve the original-resolution signal. A backbone or vertically aggregated profile is an additional retrieval input/diagnostic, not a destructive replacement.
- [ ] Evaluate altitude-dependent vertical aggregation only if needed; define aggregation widths in meters, preserve uncertainty semantics and validate bias with synthetic truth.
- [ ] Select high-altitude Rayleigh candidates from the backbone catalogue.
- [ ] Introduce `target_top_altitude_m` as a design target/configurable objective, initially 20000 m for R&D. It is never permission to extrapolate or lower QA.
- [ ] Permit the reference search to extend above the target (currently up to 25 km), because a backward solution intended to reach 20 km needs a trustworthy boundary at or above the desired top.
- [ ] If no accepted boundary exists above the target, report the lower supported top rather than forcing 20 km.

P5.4 acceptance gate: synthetic and real evidence shows that longer averaging increases usable high-altitude support without materially biasing the accepted lower-column solution.

### P5.5 multi-reference ensemble + reference-choice uncertainty

- [ ] Run backward KFS from multiple accepted, sufficiently separated high-altitude references rather than treating one reference as uniquely true.
- [ ] Retain each member’s candidate diagnostics and Monte Carlo uncertainty internally and, where practical, in auditable output/provenance.
- [ ] At each altitude, combine only members whose backward branch physically covers that bin and passed all relevant QA.
- [ ] Define ensemble weighting from explicit quality/uncertainty quantities; no hidden constants.
- [ ] Add between-reference spread/reference-choice sensitivity as an uncertainty component instead of hiding it by averaging.
- [ ] Ensure uncertainty increases rather than decreases when reference ambiguity is intentionally introduced in synthetic tests.
- [ ] Expose effective member count versus altitude and distinguish reference-member count from block count.

P5.5 acceptance gate: moving an accepted reference within a clean molecular region changes the ensemble result within a predefined tolerance, while uncertainty expands when reference sensitivity is large.

### P5.6 cascaded backward retrieval — deferred decision gate

Cascade is **not** the next step by default. Evaluate it only if backbone + ensemble still leave a scientifically meaningful coverage gap that cannot be addressed by a single accepted high boundary.

If cascade is justified:

- [ ] Define ordered overlapping altitude segments from high to low.
- [ ] Solve the highest segment from a valid high-altitude boundary.
- [ ] For each lower segment, derive `beta_total_ref` / scattering ratio at its upper boundary from the already accepted upper solution.
- [ ] Propagate inherited-boundary uncertainty into the lower-segment Monte Carlo.
- [ ] Require an overlap region with adequate common support before accepting a handoff.
- [ ] Reject handoffs whose upper/lower solutions disagree beyond an explicit uncertainty-aware criterion.
- [ ] Never reset `SR_ref = 1` simply because a new segment begins.
- [ ] Preserve NaN/unsupported state where a branch cannot be supported; never bridge internal gaps.

P5.6 acceptance gate: a cascade reproduces a single well-conditioned full-column synthetic inversion within tolerance, and deliberately inconsistent handoffs fail explicitly.

### P5.7 overlap merge — only if multiple accepted solutions require stitching

- [ ] Implement an explicit smooth merge only for genuinely overlapping accepted solutions.
- [ ] Candidate methods may include cosine/sigmoid taper combined with documented uncertainty weighting; choose from tests rather than aesthetics.
- [ ] Merge weights must sum to one wherever both profiles are valid.
- [ ] Never average a valid retrieval with invalid/NaN output as though both had evidence.
- [ ] Check value and vertical-gradient continuity across the merge region.
- [ ] Persist merge start/stop and method/provenance.

P5.7 acceptance gate: merged synthetic profiles contain no artificial step/kink attributable to the stitching method within defined tolerance.

### P5.8 uncertainty model extension

Current method-v3 already propagates measurement uncertainty, shared lidar-ratio nuisance and reference-boundary uncertainty through Monte Carlo, and uses a conservative fully-correlated upper bound when aggregating block uncertainty. Missing uncertainty is rejected rather than treated as zero.

Still required for the redesigned high-column product:

- [ ] Quantify gluing slope/intercept fit uncertainty and covariance; add it only if material, otherwise retain evidence justifying exclusion.
- [ ] Add reference-choice/ensemble spread when ensemble retrieval is introduced.
- [ ] Add cascade handoff/inherited-boundary uncertainty if cascade is introduced.
- [ ] Add averaging/vertical-aggregation sampling uncertainty if backbone aggregation makes it scientifically relevant.
- [ ] Decide which uncertainty components are persisted separately versus summarized in a total/partial budget.
- [ ] Keep covariance assumptions explicit; do not add correlated components in quadrature by convenience.
- [ ] Keep identical uncertainty semantics across 355/532 even when their supported tops differ.

### P5.9 Level 2 schema/provenance + QA redesign

As features become productive, version the schema/method deliberately and keep old products stale under changed semantics.

- [ ] Add inversion-support/top/block-count fields from P5.1 with explicit metadata distinguishing algorithmic inversion support from future instrument-validated scientific support.
- [ ] Add reference-candidate/member diagnostics required to audit P5.2/P5.5.
- [ ] Add backbone variables with unambiguous names distinct from 20-minute block products.
- [ ] Add ensemble spread/member-count variables if ensemble becomes productive.
- [ ] Add cascade segment/handoff and merge-region diagnostics only if those algorithms become productive.
- [x] Existing metadata explicitly states that elastic extinction is computed using the assumed aerosol lidar ratio.
- [x] Scientific retrieval-method versioning is independent of package CalVer.
- [ ] Update NetCDF contract validation and incremental currentness for every schema/method semantic change.
- [ ] QA plots must show the effective optical retrieval top/support and all selected/accepted reference locations rather than implying that finite scattering ratio equals retrieved aerosol.
- [ ] Backbone/ensemble/cascade QA panels are added only when the corresponding productive feature exists.

### P5.10 validation matrix for high-column redesign

#### Synthetic validation

- [x] Pure molecular atmosphere -> aerosol backscatter approximately zero within numerical tolerance.
- [x] Known aerosol layers at 355/532 -> recover known profiles within current discretization tolerance.
- [x] Vertical-grid convergence -> finer grid reduces numerical recovery error as expected.
- [x] Missing uncertainty/internal invalid gap -> support fails explicitly rather than bridging.
- [ ] Multiple valid high references -> ensemble stable and uncertainty reflects reference sensitivity.
- [ ] Biased/contaminated candidate -> rejected or increases uncertainty rather than silently biasing output.
- [ ] Upper-tail noise -> supported top falls gracefully.
- [ ] Backbone averaging/aggregation -> does not bias known lower-column truth beyond defined tolerance.
- [ ] Cascade handoff truth case -> no discontinuity and inconsistent boundary is rejected, if cascade is adopted.

#### Real SPU validation

Primary case: `20251107sapm`.

- [ ] 355 nm: extend defensible inversion support materially above the current ~6.1 km aggregate finite top when data support it.
- [ ] 532 nm: extend defensible inversion support materially above the current ~6.3 km aggregate finite top when data support it.
- [ ] Preserve/compare the redesigned 0–6 km solution against the frozen method-v3 baseline; changes must be physically explainable and consistent with the declared uncertainty/tolerance.
- [ ] Evaluate whether 15 km is supportable for this measurement.
- [ ] Attempt the 20 km design target only if a valid reference/support exists at or above the required boundary.
- [ ] Inspect wavelength consistency of aerosol-layer structure without forcing agreement.

Additional cases required before calling the redesign generally validated:

- [ ] clear/clean night with strong high-altitude molecular signal;
- [ ] high aerosol loading;
- [ ] cloud-contaminated case;
- [ ] weak-signal case;
- [ ] case with materially different PC/AN dominance or single-channel fallback behavior.

#### External comparison

- [ ] Compare representative retrievals against LPP when practical, recording time averaging, vertical resolution, LR and reference strategy.
- [ ] Compare methodological behavior against SCC/ELDA expectations without claiming numerical identity or ground truth.

P5 acceptance gate: **success is not “reaches 20 km.” Success is “maximizes validated/inversion-supported vertical coverage, exposes where and why support ends, and remains stable/traceable under synthetic truth, reference and noise sensitivity tests.”**

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

1. Integrate the tested backward **inversion-support** contract into block/aggregate Level 2 assembly; expose top/support diagnostics without pretending the unresolved overlap boundary is characterized.
2. Add contract/schema/output tests and version the Level 2 schema/currentness if new public NetCDF fields are introduced.
3. Refactor Rayleigh selection from “single minimum-cost candidate -> QA” into “enumerate all -> diagnose -> QA -> rank accepted candidates only”.
4. Run that catalogue on `20251107sapm` before introducing a long-mean backbone, and record what actually limits high-altitude candidates.
5. Quantify gluing fit-parameter uncertainty in parallel; proceed to backbone + ensemble only from evidence that they are needed.
