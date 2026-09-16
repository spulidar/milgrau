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
- Long averaging must not hide temporal state changes. A high-column backbone requires explicit temporal support/stability evidence, not merely a finite whole-measurement mean.
- A KFS boundary is local to an exact altitude/bin. A broad-column statistic may inform a robust consensus/ensemble, but must not replace the physical signal and molecular state at the member's exact boundary.
- Molecular extinction, molecular backscatter and the molecular lidar ratio used by KFS must describe a mutually consistent detected molecular component; total-Rayleigh versus Cabannes semantics must be resolved explicitly rather than assumed.
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

Key current limitations remain: elastic extinction is conditional on assumed lidar ratio; physical photon-counting saturation is uncharacterized; fitted gluing slope/intercept uncertainty is outside the current partial budget; cloud/SNR Rayleigh gates are not yet evidence-backed; near-field overlap is not experimentally characterized; and the molecular lidar-ratio constant used by KFS still needs an explicit consistency audit against the depolarization-aware molecular model and the receiver's detected molecular component.

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
- [ ] Treat a future nighttime Raman retrieval as an independent validation/R&D path only after channel-pair physics, overlap, uncertainty and synthetic truth are explicitly validated; do not port the untested legacy Raman implementation as productive code.

## P5 — altitude-resolved support + high-column Level 2 redesign

Scientific P5 development may proceed in parallel with unresolved P3 release work. P4 evidence remains authoritative: unresolved instrument boundaries must stay provisional/unsupported rather than being invented for P5.

### P5.0 frozen evidence + historical audit + physical Rayleigh-window geometry

- [x] Freeze the real `20251107sapm` method-v3 observational baseline in `docs/regression_baselines/20251107sapm_method_v3.json`, including product hashes, selected references, valid KFS block counts and current finite tops.
- [x] Keep synthetic truth separate from observational regression evidence.
- [x] Audit the user-supplied historical MILGRAU archive and preserve reusable scientific lessons in `docs/legacy_level2_lessons.md` plus machine-readable metrics in `docs/regression_baselines/20241219nt_legacy_level2_audit.json`. The legacy case is evidence about past behavior, not a golden implementation.
- [x] Reconstruct the archived `20241219nt` 355.PC backscatter behavior closely enough to identify the historical mechanism (correlation ~0.99996; relative L2 ~1.1%) while recording that source code and saved products in the archive are not a perfectly synchronized provenance snapshot.
- [x] Document why legacy apparent 30 km coverage is not itself support: the old workflow used a ~53 min whole-measurement mean, a nominal 15 km boundary, a broad ~5–25 km signal-reference median, two-sided KFS and post-inversion smoothing; most saved bins above 15 km are negative.
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

Current limitation: the productive selector still minimizes a cost over geometrically/data-valid windows and applies full configured Rayleigh QA only to that one selected window. The new catalogue is deliberately diagnostic/non-productive until ranking and high-altitude quality gates are validated.

- [x] Introduce a typed `RayleighReferenceCandidate` contract with start/stop/center altitude, valid-bin count/fraction, relative slope, relative variance, origin-constrained calibration factor, free-intercept diagnostic, diagnostic cost, SNR diagnostic and explicit rejection mask.
- [x] Enumerate **all** fully contained reference candidates inside the configured search interval without silently reducing them to one result.
- [x] Compute shape/calibration diagnostics for every candidate before selection.
- [x] Preserve accepted and rejected candidates separately so pass/fail QA is inspectable before any ranking policy is applied.
- [x] Add propagated-uncertainty SNR diagnostics to candidates; keep SNR diagnostic-only because no evidence-backed hard SPU threshold has been selected.
- [x] Add synthetic catalogue tests for a clean molecular scaling, contaminated high-altitude gradient, missing samples without interpolation and preservation of rejected candidates.
- [ ] Refactor the **productive selector** to use `enumerate -> diagnose -> minimum QA -> rank accepted candidates only`; rejected candidates must never participate in productive ranking.
- [ ] Define the productive ranking policy. Higher altitude may be preferred only after minimum quality gates pass; do not chase altitude by ranking a poor high window over a demonstrably molecular lower one.
- [ ] Persist an auditable candidate catalogue or compact reproducible representation sufficient to explain why references were accepted/rejected and why selected references were preferred.
- [ ] Add cloud/layer contamination diagnostics when available, but keep productive rejection disabled until validated on SPU observations.
- [ ] Define minimum separation/correlation rules before treating nearby overlapping accepted windows as distinct ensemble members.

P5.2 acceptance gate: for any block/backbone profile, MILGRAU can explain every accepted/rejected reference candidate and the final ranking without hidden heuristic defaults.

### P5.3 real-data candidate experiments before algorithmic complexity

Primary current-method case: `20251107sapm`.

- [x] Run the diagnostic full candidate catalogue on each existing 20-minute block at 355 and 532 nm using the 5–25 km search interval and 1 km physical window; freeze the preliminary evidence in `docs/regression_baselines/20251107sapm_rayleigh_catalogue_preliminary.json`.
- [x] Verify that the accepted candidate with minimum historical diagnostic cost reproduces the productive method-v3 selected reference in all 10 block/wavelength cases, confirming that the catalogue is comparing the same current selection logic.
- [x] Show that many formally passing high-altitude candidates exist under the present slope/variance/valid-fraction thresholds, including candidates above 20 km, but that many have weak propagated SNR (~order 1) and/or variance near the permissive current limit. Therefore “choose the highest passing window” is not scientifically justified.
- [x] Establish that the present ~6 km references are strongly influenced by minimum-cost ranking, while current high-altitude QA is too permissive to replace that ranking with a naive altitude preference.
- [ ] Compare the eventual redesigned productive candidate selection below the current ~6 km baseline; differences must be physically explainable and within defined synthetic/uncertainty tolerances.

Historical stress case: `20241219nt` from the legacy archive.

- [x] Use the archived 355.PC profiles to test the high-column concept observationally. The first ~20 min block has many molecular-like high-altitude candidates, while the later ~20 min blocks lose acceptable high-altitude shape support near ~6.4 and ~5.5 km under the exploratory current-style diagnostics.
- [x] Show that the whole ~53 min mean can recover many high-altitude candidates because the early high-altitude signal dominates the average; around 15 km, about 97% of the signed whole-mean signal contribution comes from the first 20 min block. This is evidence that a backbone needs temporal-support/stability diagnostics.

P5.3 acceptance gate: do not implement a backbone/cascade merely because it was in an old design. First show from candidate evidence what actually limits each real measurement and whether longer averaging represents a stable atmospheric/instrument state.

### P5.4 high-column backbone

Use only if candidate experiments show that high-altitude support is limited by short-block SNR/variance rather than by an absence of usable molecular information.

- [ ] Add a distinct long-mean/high-column retrieval input separate from the 20-minute block products.
- [ ] Make backbone averaging duration/configuration explicit and persist it in provenance. Do not choose the duration by convenience alone.
- [ ] Build the long-mean signal and error on common support using the established missing-uncertainty rules.
- [ ] Add explicit temporal support/stability diagnostics before a backbone is accepted: contributing profile/block count, start/stop time, effective duration, contribution/support fraction versus altitude, and sensitivity to contiguous subwindows.
- [ ] Reject or split a backbone when high-altitude evidence is confined to a transient subperiod that is not representative of the full interval. Do not let a strong early block silently create a 50-minute high-column claim.
- [ ] Compare whole-measurement averaging with evidence-driven contiguous stable windows; any adaptive time-window selection must be auditable and must not optimize against the desired aerosol result.
- [ ] Preserve the original-resolution signal. A backbone or vertically aggregated profile is an additional retrieval input/diagnostic, not a destructive replacement.
- [ ] Evaluate altitude-dependent vertical aggregation only if needed; define aggregation widths in meters, preserve uncertainty semantics and validate bias with synthetic truth.
- [ ] Select high-altitude Rayleigh candidates from the backbone catalogue.
- [ ] Introduce `target_top_altitude_m` as a design target/configurable objective, initially 20000 m for R&D. It is never permission to extrapolate or lower QA.
- [ ] Permit the reference search to extend above the target (currently up to 25 km), because a backward solution intended to reach 20 km needs a trustworthy boundary at or above the desired top.
- [ ] If no accepted boundary exists above the target, report the lower supported top rather than forcing 20 km.

P5.4 acceptance gate: synthetic and real evidence shows that longer averaging increases usable high-altitude support without materially biasing the accepted lower-column solution **and without hiding temporal loss/change of high-altitude support**.

### P5.5 robust reference consensus + multi-reference ensemble

The legacy 5–25 km median reference is retained only as motivation for robustness. Do **not** copy its non-local boundary: in the audited case the broad signal statistic and exact 15 km molecular boundary were inconsistent and did not preserve the configured zero-aerosol boundary.

- [ ] Use multiple narrow physical Rayleigh candidates that individually pass QA; each KFS member remains tied to its own exact local reference bin/state.
- [ ] Define separation/correlation rules and, if useful, a robust calibration consensus from sufficiently separated accepted windows before KFS; do not average strongly overlapping windows as independent evidence.
- [ ] Run backward KFS from multiple accepted, sufficiently separated high-altitude references rather than treating one reference as uniquely true.
- [ ] Retain each member’s candidate diagnostics and Monte Carlo uncertainty internally and, where practical, in auditable output/provenance.
- [ ] At each altitude, combine only members whose backward branch physically covers that bin and passed all relevant QA.
- [ ] Define ensemble weighting from explicit quality/uncertainty quantities; no hidden constants.
- [ ] Add between-reference spread/reference-choice sensitivity as an uncertainty component instead of hiding it by averaging.
- [ ] Ensure uncertainty increases rather than decreases when reference ambiguity is intentionally introduced in synthetic tests.
- [ ] Expose effective member count versus altitude and distinguish reference-member count from block count.

P5.5 acceptance gate: moving an accepted reference within a clean molecular region changes the ensemble result within a predefined tolerance, while uncertainty expands when reference sensitivity is large. No ensemble member may use a non-local signal statistic as though it were the exact member boundary.

### P5.6 cascaded backward retrieval — deferred decision gate

Cascade is **not** the next step by default. Evaluate it only if backbone + ensemble still leave a scientifically meaningful coverage gap that cannot be addressed by a single accepted high boundary.

The current forward/two-sided KFS kernels remain research diagnostics. The legacy `20241219nt` output shows why finite forward bins cannot be counted as high-column support: the saved backscatter above the nominal 15 km boundary is predominantly negative/noisy. Productive high-column coverage remains backward-boundary based unless a future method contract is independently validated.

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

### P5.8 molecular-model consistency + uncertainty extension

Current method-v3 already propagates measurement uncertainty, shared aerosol lidar-ratio nuisance and reference-boundary uncertainty through Monte Carlo, and uses a conservative fully-correlated upper bound when aggregating block uncertainty. Missing uncertainty is rejected rather than treated as zero.

The legacy audit exposed a separate model-consistency question: current molecular extinction/backscatter are depolarization-aware, while KFS still defaults to `8*pi/3` for the molecular lidar ratio. From the current molecular `alpha_mol/beta_mol`, the phase-function-consistent values are approximately 8.504 sr at 355 nm and 8.497 sr at 532 nm, around 1.4–1.5% above `8*pi/3`. Do not silently change this constant: first resolve whether the actual detected elastic molecular component should be treated as total Rayleigh or Cabannes for the receiver/filter response.

- [ ] Audit the exact definition used by current `molecular_extinction` and `molecular_backscatter` and derive the internally consistent `S_m = alpha_mol/beta_mol` for each wavelength.
- [ ] Determine from optical/filter/detector semantics whether KFS should use total-Rayleigh, Cabannes or another effective detected molecular lidar ratio; document assumptions and references.
- [ ] Quantify the retrieval sensitivity of 355/532 backscatter to the ~1.4–1.5% molecular-lidar-ratio difference on synthetic truth and representative real profiles before any productive change.
- [ ] If the productive molecular lidar-ratio semantics change, version the retrieval method and provenance explicitly; do not hide the change under package CalVer.
- [ ] Quantify gluing slope/intercept fit uncertainty and covariance; add it only if material, otherwise retain evidence justifying exclusion.
- [ ] Add reference-choice/ensemble spread when ensemble retrieval is introduced.
- [ ] Add cascade handoff/inherited-boundary uncertainty if cascade is introduced.
- [ ] Add averaging/vertical-aggregation sampling uncertainty if backbone aggregation makes it scientifically relevant.
- [ ] Add temporal-window/backbone selection sensitivity if adaptive backbone windows are introduced.
- [ ] Decide which uncertainty components are persisted separately versus summarized in a total/partial budget.
- [ ] Keep covariance assumptions explicit; do not add correlated components in quadrature by convenience.
- [ ] Keep identical uncertainty semantics across 355/532 even when their supported tops differ.

### P5.9 Level 2 schema/provenance + QA redesign

As features become productive, version the schema/method deliberately and keep old products stale under changed semantics.

- [ ] Add inversion-support/top/block-count fields from P5.1 with explicit metadata distinguishing algorithmic inversion support from future instrument-validated scientific support.
- [ ] Add reference-candidate/member diagnostics required to audit P5.2/P5.5.
- [ ] Add backbone variables with unambiguous names distinct from 20-minute block products, including temporal support/provenance when a backbone exists.
- [ ] Add ensemble spread/member-count variables if ensemble becomes productive.
- [ ] Add cascade segment/handoff and merge-region diagnostics only if those algorithms become productive.
- [x] Existing metadata explicitly states that elastic extinction is computed using the assumed aerosol lidar ratio.
- [x] Scientific retrieval-method versioning is independent of package CalVer.
- [ ] Update NetCDF contract validation and incremental currentness for every schema/method semantic change.
- [ ] QA plots must show the effective optical retrieval top/support and all selected/accepted reference locations rather than implying that finite scattering ratio equals retrieved aerosol.
- [ ] Add a high-column temporal-support panel when a backbone is productive so a whole-interval result cannot hide that only one subperiod carries the far-range signal.
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
- [ ] Temporally heterogeneous synthetic sequence -> whole-interval backbone must expose or reject transient-only high-altitude support rather than treating it as uniformly representative.
- [ ] Molecular-lidar-ratio consistency case -> retrieval impact of total-Rayleigh/Cabannes/effective `S_m` assumptions is quantified and versioned if productive semantics change.
- [ ] Cascade handoff truth case -> no discontinuity and inconsistent boundary is rejected, if cascade is adopted.

#### Real SPU validation

Primary current-method case: `20251107sapm`.

- [ ] 355 nm: extend defensible inversion support materially above the current ~6.1 km aggregate finite top when data support it.
- [ ] 532 nm: extend defensible inversion support materially above the current ~6.3 km aggregate finite top when data support it.
- [ ] Preserve/compare the redesigned 0–6 km solution against the frozen method-v3 baseline; changes must be physically explainable and consistent with the declared uncertainty/tolerance.
- [ ] Evaluate whether 15 km is supportable for this measurement.
- [ ] Attempt the 20 km design target only if a valid reference/support exists at or above the required boundary.
- [ ] Inspect wavelength consistency of aerosol-layer structure without forcing agreement.

Historical stress case: `20241219nt`.

- [x] Preserve the legacy audit evidence and provenance caveat; do not use the old 30 km profile as truth.
- [ ] Reprocess the underlying measurement through a future backbone/candidate implementation and verify that high-altitude support is attributed to the correct temporal interval rather than inherited blindly from the ~53 min legacy mean.
- [ ] Compare the redesigned lower-column solution against the reconstructed legacy behavior and, if an independently archived SCC/ELDA product becomes available, compare all three with matched averaging/LR/reference settings without treating SCC as ground truth.

Additional cases required before calling the redesign generally validated:

- [ ] clear/clean night with strong high-altitude molecular signal;
- [ ] high aerosol loading;
- [ ] cloud-contaminated case;
- [ ] weak-signal case;
- [ ] temporally changing/transient high-altitude layer case;
- [ ] case with materially different PC/AN dominance or single-channel fallback behavior.

#### External comparison

- [ ] Compare representative retrievals against LPP when practical, recording time averaging, vertical resolution, LR and reference strategy.
- [ ] Compare methodological behavior against SCC/ELDA expectations without claiming numerical identity or ground truth.
- [ ] If a validated nighttime Raman product is developed, use it as an independent elastic-retrieval validation axis for extinction/backscatter/lidar-ratio sensitivity rather than as a replacement for elastic QA.

P5 acceptance gate: **success is not “reaches 20 km.” Success is “maximizes validated/inversion-supported vertical coverage, exposes where and why support ends, remains temporally representative of the interval it claims to describe, and is stable/traceable under synthetic truth, reference, molecular-model and noise sensitivity tests.”**

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
3. Refactor the productive Rayleigh selector to consume the existing auditable candidate catalogue as `enumerate -> diagnose -> QA -> rank accepted candidates only`; keep SNR diagnostic-only until its criterion is evidence-backed.
4. Before implementing the high-column backbone, design and test temporal-support/stability diagnostics using both `20251107sapm` and the legacy `20241219nt` stress case.
5. Audit molecular-lidar-ratio consistency (`alpha_mol/beta_mol`, wavelength dependence and total-Rayleigh-vs-Cabannes detected component) in parallel with gluing fit-parameter uncertainty.
6. Proceed to backbone + robust multi-reference ensemble only from evidence that they extend support without hiding temporal nonstationarity or boundary ambiguity.
