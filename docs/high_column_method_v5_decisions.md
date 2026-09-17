# High-column elastic retrieval — method-v5 decision gate

## Goal

Extend the useful altitude range of the MILGRAU elastic backward retrieval without claiming information that is not observed. Raman retrieval is deferred; method v5 is therefore defined as an **elastic retrieval conditional on declared lidar-ratio, boundary-condition and effective-resolution assumptions**.

The productive method remains schema v3 / method v4 until the R&D implementation below passes synthetic-truth and heterogeneous-real-data validation.

## Decisions closed for the first method-v5 prototype

### D0 — Scientific meaning of the product — CLOSED

Method v5 will report the following conditional statement:

> Given this aerosol lidar-ratio assumption, this upper boundary condition and this effective vertical resolution, this is the elastic solution; this is its random/Monte-Carlo dispersion; and this is its sensitivity to the boundary condition.

The elastic retrieval does not claim that the lidar ratio or residual aerosol at the boundary were independently observed.

### D1 — Meaning of the upper boundary — CLOSED FOR PROTOTYPE

- [x] Nominal boundary: `beta_aer(ref) = 0`, equivalently `f = beta_aer(ref)/beta_mol(ref) = 0`.
- [x] High-reference search domain for the first prototype: **20–25 km**.
- [x] `f=0` is an explicit conditional assumption, not a claim that 20–25 km is physically aerosol-free.
- [x] Residual-aerosol boundary uncertainty is retained as a separate systematic sensitivity dimension.

Stratospheric aerosol can exist in the 20–30 km region, including background sulfate and volcanic/wildfire perturbations. Therefore altitude alone cannot certify molecular purity. A high Rayleigh-compatible reference in 20–25 km is used because it is physically preferable to a lower tropospheric reference for the nominal zero-aerosol assumption, while the unresolved residual aerosol is kept explicit.

### D2 — Numerical boundary estimator — CLOSED FOR PROTOTYPE

- [x] The high boundary is represented by the **same effective vertical cell used by the high-column retrieval**, not by pretending that a multi-bin estimator is a native 7.5 m point measurement.
- [x] The RCS, molecular backscatter and altitude assigned to the reference are the strict aggregated-cell quantities.
- [x] The effective cell resolution and source-bin count must be written to diagnostics/product metadata.
- [x] Aggregation never bridges an invalid native bin. A source cell containing unsupported required samples remains unsupported.

This reduces single-bin random noise while keeping the estimator's physical resolution honest. It does not establish aerosol-free purity.

### D3 — Progressive vertical grid — CLOSED FOR FIRST R&D GRID

The first method-v5 prototype uses a deterministic progressive resolution schedule built from contiguous native Level-1 bins:

| altitude | requested resolution | SPU 7.5 m native-grid realization |
| --- | ---: | ---: |
| below 10 km | 7.5 m | 1 bin |
| 10–15 km | 15 m | 2 bins |
| 15–20 km | 30 m | 4 bins |
| 20–25 km | 60 m | 8 bins |
| >=25 km | 100 m maximum requested | 97.5 m = 13 bins |

The 100 m value is a **maximum requested physical resolution**, not permission to invent a non-native cell. For the current 7.5 m SPU grid the nearest strict contiguous realization not exceeding 100 m is 97.5 m.

This schedule is an initial R&D policy, not yet a productive threshold law. Its transition heights will be challenged with synthetic coverage and heterogeneous measurements before promotion. The implementation must expose the schedule so later work can test an uncertainty-driven selector without rewriting the KFS core.

No interpolation, padding or gap filling is allowed. The KFS integration already supports a strictly increasing nonuniform altitude grid, so the adaptive representation can remain explicit rather than being resampled back to a fake uniform 7.5 m grid.

### D4 — Monte-Carlo/support semantics — CLOSED CONCEPTUALLY

- [x] Method v5 will no longer define physical support as `300/300` Monte-Carlo realizations surviving.
- [x] Nominal deterministic path support and Monte-Carlo robustness are separate quantities.
- [x] Save the valid-realization count and fraction for each requested branch/profile.
- [x] Do not yet impose a new valid-fraction cutoff.
- [x] Any future cutoff is derived from synthetic-truth uncertainty-coverage behavior, not from the altitude reached.

Method v4 semantics remain unchanged while v5 is R&D.

### D4b — Residual aerosol `f` and Monte Carlo — CLOSED FOR PROTOTYPE

Residual aerosol at the boundary is an epistemic/systematic uncertainty, not presently a characterized random variable. Therefore the first prototype uses a **nested uncertainty design**:

1. outer, caller-declared `f` scenarios describe boundary-condition sensitivity;
2. within each `f` scenario, the ordinary Monte Carlo propagates signal noise and declared lidar-ratio uncertainty;
3. random dispersion and between-`f` sensitivity are stored separately.

No probability density is assigned to `f` yet. If independent evidence later supports a probability distribution for `f`, the nested ensemble may be marginalized into a declared total conditional uncertainty. Until then, combining arbitrary `f` draws with random measurement noise would overstate what is known.

### D5 — Temporal strategy — BASELINE CLOSED / EXTENSION OPEN

- [x] Preserve the current 20 min block product as the primary temporal state for the first v5 prototype.
- [x] Vertical aggregation is tested before introducing longer temporal averaging, so the two information trades can be identified separately.
- [ ] Evaluate shorter/longer candidate temporal blocks on the heterogeneous observations after the progressive-grid prototype is operational.
- [ ] If a longer high-column mean is adopted, preserve the underlying block-level contribution/persistence diagnostics.

The current `block_average_minutes: 20` is therefore retained initially rather than silently lengthened to obtain high-altitude coverage.

### D6 — Admissible atmospheric path — CLOSED INVARIANTS

- [x] All required cells must come from finite usable source samples.
- [x] Internal unsupported gaps are not bridged.
- [x] Instrument/saturation masks are respected when characterized.
- [x] Cloud/layer presence remains diagnostic unless a separately validated veto is adopted.

A candidate above an invalid or opaque segment is not automatically usable merely because its local Rayleigh window passes QA.

### D7 — Lower-column preservation criterion — VALIDATION DESIGN CLOSED

Promotion is based on two distinct evidence classes:

- [x] **synthetic truth:** bias and uncertainty coverage against known aerosol truth;
- [x] **real-data regression:** difference from method v4 in the established lower column, interpreted as compatibility/sensitivity rather than truth.

No fixed percentage is selected merely because it allows a preferred high-altitude solution. Integrated-column and profile-shape differences are both retained.

### D8 — Lidar-ratio scope — CLOSED FOR PROTOTYPE

- [x] Backscatter and extinction may both be generated under the same declared lidar-ratio assumption.
- [x] Extinction remains explicitly conditional on the assumed/climatological aerosol lidar ratio.
- [x] A higher supported extinction profile is not an independent lidar-ratio retrieval.

### D9 — Higher-reference selection — PARTIALLY OPEN

The first prototype searches for Rayleigh-compatible high-reference cells in **20–25 km** after constructing the progressive grid and confirming nominal path admissibility.

Still open before promotion:

- [ ] determine the deterministic tie-break/selection policy among multiple admissible 20–25 km cells;
- [ ] verify that the policy is not acting as a proxy for an unobserved `f`;
- [ ] validate the selected-reference behavior under stratospheric-aerosol and cloud synthetic cases.

Candidate shape QA, path support, estimator precision and effective resolution remain separate diagnostics; no composite molecular-purity score is introduced.

### D10 — Method/product versioning — DECIDED IN PRINCIPLE

If the prototype passes promotion gates:

- [x] retrieval method becomes **v5**;
- [ ] decide whether schema v3 can cleanly represent all new altitude-grid and uncertainty dimensions or schema v4 is required;
- [ ] freeze method-v4 and method-v5 regression products from identical Level-1 inputs;
- [ ] expose boundary model, `f` scenario identity, boundary cell resolution/source-bin count, effective vertical resolution and valid-MC fraction.

## Progressive-grid implementation contract

The grid utility must satisfy all of the following before being connected to productive retrieval code:

- contiguous source-bin groups only;
- no overlap between output cells;
- no native sample used twice;
- no output cell spanning an unsupported source gap;
- arithmetic cell mean for signal/state quantities unless a different estimator is explicitly declared;
- uncertainty aggregation with an explicit dependence model;
- source count and effective cell width retained;
- molecular and measured profiles represented on exactly the same cells;
- output altitude strictly increasing so the existing generalized KFS integral can operate directly on the nonuniform grid;
- exact native-grid identity below the first transition.

## Minimum validation matrix before promotion

- [ ] controlled molecular-only synthetic truth;
- [ ] controlled residual-aerosol boundary truth;
- [ ] weak-signal / many-isolated-bin path case;
- [ ] narrow and broad contamination cases;
- [ ] cloud/layer case;
- [ ] explicit stratospheric-aerosol case in the 20–25 km reference region;
- [ ] at least several heterogeneous current SPU measurements;
- [ ] 355 and 532 nm separately;
- [ ] native vs progressive high-column grid;
- [ ] lower-column compatibility reported independently of achieved top altitude;
- [ ] valid-MC fraction versus synthetic confidence-interval coverage;
- [ ] temporal 20 min baseline versus any later proposed temporal averaging.

## Current interpretation

The first v5 prototype is now sufficiently specified to implement the vertical-grid layer without waiting for Raman. The remaining scientific work is validation rather than choosing an arbitrary target altitude: verify the progressive grid, quantify its representation error, determine MC coverage behavior, and then finalize the high-reference tie-break within 20–25 km.