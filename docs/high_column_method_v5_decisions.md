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
- [x] `f=0` is an explicit conditional assumption, not a claim that a given high altitude is physically aerosol-free.
- [x] Residual-aerosol boundary uncertainty is retained as a separate systematic sensitivity dimension.
- [x] **20–25 km is a preferred high-reference region when supported, not a mandatory target.**

Stratospheric aerosol can exist in the 20–30 km region, including background sulfate and volcanic/wildfire perturbations. Altitude alone therefore cannot certify molecular purity. The prototype should seek the highest Rayleigh-compatible reference that is also physically/numerically admissible below the configured maximum search altitude, rather than forcing every block to reach 20 or 25 km.

Campaign evidence in `docs/regression_baselines/p5_4_method_v5_progressive_grid_campaign_20260917.json` shows why this distinction matters: among 161 block×wavelength cases, 117 contain a native Rayleigh-accepted candidate in 20–25 km, but only 18 have a continuous nominal backward path to such a high reference under the current <=100 m progressive-grid cap. Candidate existence is therefore much more common than usable high-boundary support.

### D2 — Numerical boundary estimator — CLOSED FOR PROTOTYPE

- [x] The high boundary is represented by the **same effective vertical cell used by the high-column retrieval**, not by pretending that a multi-bin estimator is a native 7.5 m point measurement.
- [x] The RCS, molecular backscatter and altitude assigned to the reference are the strict aggregated-cell quantities.
- [x] The effective cell resolution and source-bin count must be written to diagnostics/product metadata.
- [x] Aggregation never bridges a missing/masked/instrument-invalid native sample.
- [x] Finite signed background-subtracted RCS samples may be averaged; the resulting aggregated RCS cell must itself be finite and positive before KFS uses it.

A finite negative native RCS sample at high altitude is not automatically a physical gap; it can be a noisy background-subtracted measurement. Rejecting every aggregate containing one such sample defeats the purpose of estimating the signal at the declared coarser resolution. This distinction does **not** permit interpolation across NaNs, masks or instrument-invalid samples.

### D3 — Progressive vertical grid — CLOSED FOR FIRST R&D GRID

The current first-prototype schedule preserves the established lower-column representation exactly and begins aggregation above it:

| altitude | requested resolution | SPU 7.5 m native-grid realization |
| --- | ---: | ---: |
| below 6 km | 7.5 m | 1 bin |
| 6–10 km | 15 m | 2 bins |
| 10–15 km | 30 m | 4 bins |
| 15–20 km | 60 m | 8 bins |
| 20–25 km | 60 m | 8 bins |
| >=25 km | 100 m maximum requested | 97.5 m = 13 bins |

The 100 m value is a **maximum requested physical resolution**, not permission to invent a non-native cell. For the current 7.5 m SPU grid the largest strict contiguous native-bin realization not exceeding 100 m is 97.5 m.

Why aggregation begins at 6 km rather than 10 km: campaign tests showed that retaining 7.5 m all the way to 10 km leaves many weak cells in the 6–10 km path and prevents a high boundary even when the far-range cells themselves are sufficiently averaged. The productive method-v4 lower-column validation region is approximately 0.6–6 km, so the prototype preserves that region exactly and treats the region above it as the high-column information trade.

This schedule is an R&D policy, not yet a universal station law. Its transition heights will be challenged with synthetic coverage and heterogeneous measurements before promotion. The implementation exposes the schedule so a future uncertainty-driven selector can replace fixed altitude bands without rewriting KFS.

No interpolation, padding or gap filling is allowed. The generalized KFS integrator already uses local `dz` values and therefore accepts a strictly increasing nonuniform altitude grid directly.

### D3a — Campaign evidence for the current grid

Frozen evidence: `docs/regression_baselines/p5_4_method_v5_progressive_grid_campaign_20260917.json`.

For the 161 available 20 min block×wavelength states, the continuous positive aggregated path from the lower column reaches approximately:

- minimum: **6.77 km**;
- p10: **9.80 km**;
- p25: **12.89 km**;
- median: **14.57 km**;
- p75: **17.07 km**;
- p90: **21.71 km**;
- maximum: **24.99 km**.

Counts are 143/161 reaching at least 10 km, 133/161 at least 12 km, 69/161 at least 15 km and 18/161 at least 20 km.

Interpretation: the <=100 m grid materially extends usable elastic path information for many measurements, but it does **not** justify promising a fixed 20–25 km boundary. The achievable top remains measurement-dependent and must be reported as such.

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

### D5 — Temporal strategy — CLOSED FOR FIRST PROTOTYPE

- [x] Preserve the current **20 min block product** as the primary temporal state for the first v5 prototype.
- [x] Vertical aggregation is tested independently of temporal averaging so the two information trades remain identifiable.
- [x] Real-data probes at 40 and 60 min did not materially improve the campaign-wide median continuous top.
- [ ] Revisit temporal adaptation only after the progressive-grid retrieval and MC-coverage semantics are validated.

Frozen real-data comparison:

- 20 min: 159 windows, median continuous top **14.81 km**, 70 reach >=15 km, 18 reach >=20 km;
- 40 min: 87 windows, median **14.87 km**, 41 reach >=15 km, 9 reach >=20 km;
- 60 min: 59 windows, median **14.96 km**, 29 reach >=15 km, 6 reach >=20 km.

The small median change does not justify silently degrading temporal resolution to obtain altitude coverage. If a future longer high-column mean is adopted, the underlying block contribution/persistence state must remain available.

### D6 — Admissible atmospheric path — CLOSED INVARIANTS

- [x] All required aggregated cells must come from finite usable source samples.
- [x] Internal missing/masked/instrument-invalid gaps are not bridged.
- [x] An aggregated RCS cell used by KFS must be finite and positive.
- [x] Instrument/saturation masks are respected when characterized.
- [x] Cloud/layer presence remains diagnostic unless a separately validated veto is adopted.

A candidate above an invalid or opaque segment is not automatically usable merely because its local Rayleigh window passes QA.

### D7 — Lower-column preservation criterion — VALIDATION DESIGN CLOSED

Promotion is based on two distinct evidence classes:

- [x] **synthetic truth:** bias and uncertainty coverage against known aerosol truth;
- [x] **real-data regression:** difference from method v4 in the established lower column, interpreted as compatibility/sensitivity rather than truth.

No fixed percentage is selected merely because it allows a preferred high-altitude solution. Integrated-column and profile-shape differences are both retained.

High-reference campaign experiments have already shown that moving the boundary can alter the lower column materially. Therefore successful high-path support cannot by itself validate a v5 solution; boundary sensitivity remains a required output dimension.

### D8 — Lidar-ratio scope — CLOSED FOR PROTOTYPE

- [x] Backscatter and extinction may both be generated under the same declared lidar-ratio assumption.
- [x] Extinction remains explicitly conditional on the assumed/climatological aerosol lidar ratio.
- [x] A higher supported extinction profile is not an independent lidar-ratio retrieval.

### D9 — Higher-reference selection — PARTIALLY OPEN

The first prototype should select a reference only after constructing the progressive grid and confirming nominal path admissibility.

Current direction:

- [x] maximum configured search altitude remains 25 km for the prototype;
- [x] 20–25 km is preferred when an admissible Rayleigh-compatible path actually exists;
- [x] if 20–25 km is not supported, a lower high-column reference may be used rather than forcing failure at an arbitrary target altitude;
- [ ] determine the minimum altitude at which the v5 high-reference selector is allowed to depart from the v4 reference regime;
- [ ] determine the deterministic tie-break among multiple admissible high cells;
- [ ] verify with synthetic truth that the selector is not acting as a proxy for unobserved `f`;
- [ ] validate selected-reference behavior under explicit stratospheric-aerosol and cloud cases.

Candidate shape QA, nominal path support, estimator precision, MC robustness and effective resolution remain separate diagnostics; no composite molecular-purity score is introduced.

### D10 — Method/product versioning — DECIDED IN PRINCIPLE

If the prototype passes promotion gates:

- [x] retrieval method becomes **v5**;
- [ ] decide whether schema v3 can cleanly represent all new altitude-grid and uncertainty dimensions or schema v4 is required;
- [ ] freeze method-v4 and method-v5 regression products from identical Level-1 inputs;
- [ ] expose boundary model, `f` scenario identity, boundary cell resolution/source-bin count, effective vertical resolution, nominal supported top and valid-MC fraction.

## Progressive-grid implementation contract

The grid utility must satisfy all of the following before being connected to productive retrieval code:

- contiguous source-bin groups only;
- no overlap between output cells;
- no native sample used twice;
- no output cell spanning a missing/masked/instrument-invalid source gap;
- finite signed background-subtracted RCS may contribute to an arithmetic cell mean;
- any aggregated RCS cell passed to KFS must itself be finite and positive;
- arithmetic cell mean for signal/state quantities unless a different estimator is explicitly declared;
- uncertainty aggregation with an explicit dependence model;
- source count and effective cell width retained;
- molecular and measured profiles represented on exactly the same cells;
- output altitude strictly increasing so the existing generalized KFS integral can operate directly on the nonuniform grid;
- exact native-grid identity through the established lower-column region below 6 km.

Current implementation: `milgrau/level2/adaptive_grid.py`, with executable R&D contract in `tests/test_adaptive_grid_rnd.py`.

## Minimum validation matrix before promotion

- [ ] controlled molecular-only synthetic truth;
- [ ] controlled residual-aerosol boundary truth;
- [ ] weak-signal / many-isolated-bin path case;
- [ ] narrow and broad contamination cases;
- [ ] cloud/layer case;
- [ ] explicit stratospheric-aerosol case in the high-reference region;
- [x] heterogeneous-current-SPU path/resolution feasibility;
- [ ] 355 and 532 nm synthetic truth separately;
- [ ] native vs progressive high-column retrieval, not only path feasibility;
- [ ] lower-column compatibility reported independently of achieved top altitude;
- [ ] valid-MC fraction versus synthetic confidence-interval coverage;
- [x] 20/40/60 min observational path comparison.

## Current interpretation

The vertical representation is now sufficiently specified for a first R&D retrieval implementation. The campaign evidence rejects two tempting but unsupported shortcuts: keeping native 7.5 m all the way to 10 km is unnecessarily restrictive for a high-column path, while forcing a 20–25 km boundary is too aggressive under a <=100 m resolution cap. The first v5 prototype therefore preserves the established lower column exactly, progressively coarsens above 6 km, retains 20 min temporal blocks, and allows the supported high-reference altitude to remain measurement-dependent.
