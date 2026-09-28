# Scientific traceability

This document maps the current scientific behavior to code ownership,
executable evidence and product diagnostics. MILGRAU exposes one current
Level 2 method; superseded experiments are not selectable alternatives.

## Scientific statement of the product

> Given the measured signal and its propagated uncertainty, the materialized
> molecular atmosphere, the declared aerosol lidar-ratio distribution, the
> selected molecular-reference range and boundary condition, MILGRAU reports a
> two-sided elastic solution and the dispersion obtained when measurement noise
> is propagated through background fitting, reference re-selection and the
> inversion.

Elastic extinction is conditional on the assumed aerosol lidar ratio. A
selected reference altitude is a supported search result, not direct proof of
an aerosol-free layer.

## Current scientific decisions

- 20-minute temporal blocks by default.
- Native vertical representation below 6 km and strict progressive aggregation
  above it.
- Native-grid Rayleigh QA with a 1 km physical window.
- Primary reference range 10–15 km; fallback 5–20 km.
- First supported range wins; minimum `relative_slope + relative_variance`
  selects within that range.
- Robust weighted joint fit of molecular scaling and constant residual
  background over the configured high-altitude span.
- Two-sided Klett–Fernald–Sasano integration.
- Routine nominal boundary `f=0`.
- 150 configurable selection-aware Monte Carlo realizations.
- Lidar ratio sampled from the resolved station climatology and uncertainty.
- No hard threshold on propagated-error SNR, Monte Carlo selection success or
  altitude-resolved finite-realization fraction.
- No interpolation, padding or gap bridging to extend apparent coverage.

## Traceability matrix

| Scientific behavior | Canonical owner | Executable evidence / output |
| --- | --- | --- |
| Molecular Rayleigh backscatter/extinction and signal are wavelength dependent | `milgrau.level2.molecular` | molecular/KFS synthetic tests; saved molecular fields |
| Pressure and temperature are materialized at Level 1 | `milgrau.level1.thermodynamics` | atmosphere-source and NetCDF tests; `thermodynamic_profile_*` provenance |
| Signal values and uncertainties use common support | `milgrau.level2.block_average`, `signal_selection` | uncertainty-support tests; invalid-reason flags |
| Progressive aggregation is contiguous and gap preserving | `milgrau.level2.adaptive_grid` | adaptive-grid/high-column tests; resolution and source-bin variables |
| Residual background and molecular scale are jointly fitted robustly | `milgrau.level2.rayleigh_candidates` | candidate/background truth tests; fitted-background diagnostics |
| Native Rayleigh QA precedes reference selection | `milgrau.level2.rayleigh_candidates`, `high_column` | candidate/selector tests; slope, variance, valid fraction and SNR |
| Reference ranges are prioritized explicitly | `milgrau.level2.high_column_selector` | selector tests; selected range and fallback flag |
| Two-sided KFS uses one explicit reference and assumed aerosol lidar ratio | `milgrau.level2.kfs`, `two_sided_retrieval` | analytical/synthetic branch recovery; branch endpoints |
| Measurement noise propagates through reference re-selection | `milgrau.level2.uncertainty_mc` | selection-aware MC tests; selected-reference samples and quantiles |
| Residual aerosol remains an explicit boundary scenario | `two_sided_retrieval`, `uncertainty_mc` | boundary-sensitivity tests; `residual_fraction` coordinate |
| MC finite fraction is diagnostic, not a gate | Level 2 schema and tests | altitude-resolved `mc_valid_fraction` |
| Gluing is selected and audited per block | `milgrau.level2.gluing`, `signal_selection` | gluing tests and fit diagnostics |
| Exact Level 1 bytes identify upstream input | `milgrau.provenance.file_sha256` | source-identity tests; `source_level1_sha256` |
| Exact installed Python content identifies code state | `milgrau.provenance.source_code_provenance` | source-provenance tests; source-content SHA |

## Background model

The fitted model in range-corrected space is

\[
RCS(z) = A\,RCS_{mol}(z) + Bz^2,
\]

where `A` is molecular scaling and `B` is a constant offset before range
correction. The broad-span fit is weighted by signal uncertainty and uses Huber
iteratively reweighted regression. Negative noisy observations are retained in
the fit rather than clipped.

After fitting, `Bz²` is removed before progressive aggregation and inversion.
Each Monte Carlo realization perturbs the native signal and refits both
nuisance parameters. Background uncertainty is therefore coupled to reference
selection rather than appended as an independent post-hoc term.

## Vertical representation and support

The configured schedule is:

| Altitude | Requested effective resolution |
| --- | ---: |
| below 6 km | native |
| 6–10 km | 15 m |
| 10–15 km | 30 m |
| 15–25 km | 60 m |
| above 25 km | at most 100 m |

The achieved width is recorded for every cell. An endpoint is the last finite
cell reached continuously from the reference in the corresponding KFS branch.
The target of 20–25 km is therefore evaluated per block and cannot be inferred
from coordinate extent alone.

## Uncertainty interpretation

The reported random dispersion includes:

- Level 1 signal uncertainty;
- background refitting;
- progressive aggregation;
- stochastic reference acceptance/selection;
- lidar-ratio sampling;
- reference-boundary random dispersion configured in KFS.

It does not yet constitute a complete metrological budget. In particular:

- the aerosol lidar-ratio model remains an assumption;
- gluing slope/intercept covariance is not fully propagated;
- overlap and physical detector saturation are not fully characterized;
- residual aerosol at the reference has no independently justified probability
  prior.

## Evidence hierarchy

1. Analytical and synthetic truth for equations, estimators, bias and interval
   behavior.
2. Real-data regression and QA for behavior on SPU acquisitions.
3. External-chain comparison under matched inputs and assumptions.
4. Instrument characterization for saturation, overlap and calibration claims.

Real observations do not supply exact aerosol truth. SCC/ELDA agreement is a
consistency result, not automatic ground truth.

## Language constraints

- Do not describe elastic extinction as independently measured Raman
  extinction.
- Do not describe `f=0` as proof of an aerosol-free boundary.
- Do not interpret finite signal as supported optical retrieval without branch
  and Monte Carlo support.
- Do not describe altitude as a molecular-purity score.
- Do not call a provisional photon-counting guard a detector characterization.
- Do not claim universal retrieval to 20 or 25 km.

## Implemented physical and retrieval references

- Klett JD. Stable analytical inversion solution for processing lidar returns.
  *Applied Optics*. 1981;20(2):211–220. DOI: `10.1364/AO.20.000211`.
- Fernald FG. Analysis of atmospheric lidar observations: some comments.
  *Applied Optics*. 1984;23(5):652. DOI: `10.1364/AO.23.000652`.
- Bucholtz A. Rayleigh-scattering calculations for the terrestrial atmosphere.
  *Applied Optics*. 1995;34(15):2765–2773. DOI: `10.1364/AO.34.002765`.
- Donovan DP, Whiteway JA, Carswell AI. Correction for nonlinear
  photon-counting effects in lidar systems. *Applied Optics*.
  1993;32(33):6742–6753. DOI: `10.1364/AO.32.006742`.

## External-chain methodology

- D'Amico G, Amodeo A, Mattis I, Freudenthaler V, Pappalardo G. EARLINET
  Single Calculus Chain – technical – Part 1. *Atmos Meas Tech*.
  2016;9:491–507. DOI: `10.5194/amt-9-491-2016`.
- Mattis I, D'Amico G, Baars H, Amodeo A, Madonna F, Iarlori M. EARLINET
  Single Calculus Chain – technical – Part 2. *Atmos Meas Tech*.
  2016;9:3009–3029. DOI: `10.5194/amt-9-3009-2016`.

## Open evidence

The tracker is authoritative for pending gates. The principal open claims are
background-fit coverage, native-versus-progressive representation error,
explicit connected-column/20 km/25 km status, heterogeneous real-data behavior,
instrument saturation/overlap and independent Raman/SCC/AOD comparison.
