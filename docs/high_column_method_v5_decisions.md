# High-column elastic retrieval — method-v5 decision gate

## Goal

Extend the useful altitude range of the MILGRAU elastic backward retrieval without claiming information that is not observed. Raman retrieval is explicitly deferred; method v5 may therefore be an **elastic retrieval conditional on declared boundary and lidar-ratio assumptions**.

The productive method remains schema v3 / method v4 until the decisions below are closed and validated.

## Scientific decisions still required

### D1 — Meaning of the upper boundary

Choose the method-v5 scientific claim:

- [ ] **Strict molecular boundary:** set `beta_aer(ref)=0` and treat the selected upper reference as aerosol-free by assumption.
- [ ] **Bounded residual-aerosol boundary:** retain a central boundary assumption but carry an explicit systematic sensitivity/envelope for `beta_aer(ref)>0`.

The current elastic diagnostics can select Rayleigh-compatible regions but cannot independently observe `beta_aer(ref)`. If strict molecular purity is required as an observed fact, an independent constraint is needed. If the product is explicitly conditional, an elastic-only method can proceed.

### D2 — Boundary estimator

Choose the quantity used as the numerical KFS boundary:

- [ ] exact native-grid measured bin;
- [ ] Rayleigh-window estimator with explicitly propagated estimator uncertainty;
- [ ] other estimator supported by synthetic truth.

A window estimator may reduce random noise but does not remove common-mode aerosol contamination. It must therefore be evaluated separately for precision and bias.

### D3 — Vertical representation above the current robust region

Choose whether the high-column signal is represented on:

- [ ] the native 7.5 m grid;
- [ ] a fixed coarser grid, with 60 m and 120 m currently characterized as R&D examples;
- [ ] an adaptive-resolution grid with explicit effective resolution per altitude.

No interpolation is allowed to bridge invalid paths. Aggregation must propagate uncertainty with an explicit dependence/covariance model and must expose the resolution loss in the product.

### D4 — Monte-Carlo/support semantics

Method v4 effectively treats complete Monte-Carlo path survival very strictly. High paths accumulate failure probability across many weak bins.

Choose the scientific meaning of support for method v5:

- [ ] require all simulations valid;
- [ ] require a pre-declared valid-realization fraction and report it;
- [ ] derive altitude-resolved uncertainty only from valid realizations while marking insufficient-realization bins unsupported.

Any threshold must be justified statistically and fixed before examining whether it reaches a preferred altitude.

### D5 — Temporal strategy

Choose whether the high-column retrieval is performed from:

- [ ] the existing native temporal blocks;
- [ ] a longer averaged signal used as a high-column backbone;
- [ ] both, with the long mean never replacing block-level temporal diagnostics.

If long averaging is used, the product must preserve contribution fractions / persistence so that transient high-altitude structure is not presented as stationary.

### D6 — Admissible atmospheric path

Define what makes a backward path physically/numerically admissible before reference ranking:

- [ ] all required signal bins finite and usable;
- [ ] no internal unsupported gaps;
- [ ] saturation/instrument masks respected when characterized;
- [ ] cloud/layer presence remains diagnostic unless a separately validated veto is adopted.

A candidate above an invalid or opaque segment is not automatically a usable boundary merely because its local Rayleigh window passes QA.

### D7 — Lower-column preservation criterion

A higher-boundary retrieval must not be accepted only because it extends farther.

Define the promotion criterion using two distinct evidence classes:

- [ ] **synthetic truth:** bias and uncertainty coverage against known aerosol truth;
- [ ] **real-data regression:** difference from method v4 in the established lower column, interpreted as compatibility/sensitivity rather than truth.

Prefer an uncertainty-normalized compatibility criterion over an arbitrary percent chosen after seeing results. Integrated-column differences and profile-shape differences should both be reported.

### D8 — Lidar-ratio scope

Decide what product is being extended:

- [ ] aerosol backscatter only, with extinction remaining explicitly conditional on climatological/assumed lidar ratio;
- [ ] backscatter + extinction under the same declared lidar-ratio assumption and uncertainty/sensitivity treatment.

Do not interpret higher extinction support as independently retrieved lidar ratio.

### D9 — Reference selection rule

Only after D1–D8 are fixed, define the deterministic selector for a higher reference. Candidate scoring must remain separate from physical support and boundary-assumption uncertainty.

Possible selector inputs already available include minimum Rayleigh shape QA, path admissibility, estimator uncertainty, temporal contribution diagnostics and effective vertical resolution. Existing evidence does not support a new composite score as a proxy for molecular purity.

### D10 — Method/product versioning

If method v5 changes boundary semantics, vertical representation, support semantics or uncertainty dimensions:

- [ ] bump retrieval method to v5;
- [ ] decide whether schema v3 can represent all new quantities or whether schema v4 is required;
- [ ] freeze method-v4 and method-v5 regression products on the same Level-1 inputs;
- [ ] expose the boundary model, boundary estimator, effective resolution, valid-MC fraction and boundary-sensitivity assumptions in NetCDF metadata/variables.

## Minimum validation matrix before promotion

- [ ] controlled molecular-only synthetic truth;
- [ ] controlled residual-aerosol boundary truth;
- [ ] weak-signal / many-isolated-bin path case;
- [ ] narrow and broad contamination cases;
- [ ] cloud/layer case;
- [ ] at least several heterogeneous current SPU measurements;
- [ ] 355 and 532 nm separately;
- [ ] native vs chosen high-column resolution;
- [ ] lower-column compatibility reported independently of achieved top altitude.

## Current interpretation

The main unresolved scientific choice is no longer “which Rayleigh score reaches highest”. It is **what conditional boundary model and uncertainty semantics MILGRAU is willing to publish for an elastic-only high-column retrieval**. Once D1–D5 and D7 are fixed, implementation can be completed without waiting for Raman; Raman can later serve as independent validation rather than a prerequisite for the first elastic method-v5 experiment.
