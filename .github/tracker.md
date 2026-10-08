# MILGRAU tracker

This tracker records only the current product, accepted scientific decisions
and work that is still actionable. Superseded implementations and experiments
remain recoverable from Git history and are not alternative processing modes.

## Current objective

Deliver a reliable SPU-Lidar elastic Level 2 product with the highest column
supported by each measurement. The immediate validation target is useful
retrieval through 20 km and, when the signal and continuous numerical path
permit, through 25 km. No fixed top altitude is promised and unsupported bins
remain missing.

## Current processing policy

- Ground-based vertical geometry.
- Elastic wavelengths: 355 and 532 nm when their Level 1 inputs are valid.
- Temporal blocks: 20 minutes by default.
- Two-sided Klett–Fernald–Sasano integration around one selected molecular
  reference.
- Primary molecular-reference search range: 10–15 km.
- Explicit fallback search range: 5–20 km.
- Within the first supported range, select the Rayleigh-accepted,
  path-admissible candidate with minimum
  `relative_slope + relative_variance`; altitude is not a ranking reward.
- Rayleigh diagnostic window: 1 km.
- Progressive vertical grid: native resolution below 6 km, then 15 m at
  6–10 km, 30 m at 10–15 km, 60 m at 15–25 km and at most 100 m above 25 km.
- Nominal residual aerosol fraction at the reference: `f = 0`.
- Routine uncertainty ensemble: 150 configurable Monte Carlo realizations.
- Every realization perturbs the native signal, refits the residual background,
  rebuilds the progressive grid, reruns reference selection and then reruns the
  inversion.
- Lidar ratio is drawn from the configured station climatology/uncertainty and
  is bounded by the configured positive minimum.
- `mc_valid_fraction` and Monte Carlo selection success are diagnostics, not
  hard scientific acceptance thresholds.
- Missing value or uncertainty support is never filled, bridged or silently
  treated as zero uncertainty.

## Background policy

- Level 1 records the acquisition/background diagnostics and corrected signal.
- Level 2 estimates a residual constant background jointly with molecular
  scaling over the configured high-altitude fit span using robust weighted
  regression.
- In range-corrected space the fitted term is `B z²`.
- The nominal fitted background, formal standard error and calibration/background
  correlation are stored per block.
- Monte Carlo refits the background for every perturbed realization.
- Fixed post-hoc subtraction is not an alternative productive mode.

## Product and QA policy

- The NetCDF records the progressive grid, source-bin counts, molecular state,
  selected reference and its search range, fallback use, fitted background,
  two-sided branch endpoints, optical products and Monte Carlo statistics.
- Aerosol extinction is conditional on the assumed aerosol lidar ratio.
- Negative/noisy high-altitude estimates may exist; scientific support must be
  read together with branch endpoints and Monte Carlo validity.
- QA figures are optical/scientific inspection products, not only software
  diagnostics. The current panels are signal/gluing, molecular reference and
  aerosol optical profiles.
- Retrieval support shading is not plotted; the dashed branch endpoint remains
  the compact coverage indicator.

## Completed consolidation

- [x] Two-sided retrieval is the sole productive elastic path.
- [x] Old progressive reference tiers were replaced by the 10–15 km primary
  range and 5–20 km fallback.
- [x] Robust fitted residual background is used by the nominal retrieval and
  refitted inside Monte Carlo.
- [x] Routine Monte Carlo is configurable and set to 150 realizations.
- [x] Routine residual-aerosol boundary is `f = 0`.
- [x] LIRACOS and LEBEAR accept explicit UTC time windows.
- [x] Current QA panels replaced the superseded plotting stack.
- [x] Productive modules no longer use experiment/revision names.
- [x] Backward-only retrieval assembly and comparison CLIs were removed from
  the installed package.
- [x] Documentation describes one current product rather than parallel method
  generations.

## Session refactor: meteorology and atmosphere

Accepted architecture for continuous sessions:

- [x] Session continuity tolerance is 30 minutes (`level0.session.max_gap_seconds = 1800`).
- [ ] Level 0 surface weather becomes a time-resolved series rather than one
  midpoint snapshot.
- [ ] Surface-weather cadence is hourly on its native/source cadence; do not
  duplicate/interpolate hourly source values onto every lidar profile.
- [ ] Persist time-resolved surface temperature, pressure, relative humidity,
  cloud cover and wind with explicit source/provenance metadata.
- [ ] Missing surface weather remains missing; do not invent default
  temperature/pressure values.
- [ ] SCC scalar surface temperature/pressure, when required, are derived only
  for SCC interoperability and do not replace the scientific time series.
- [ ] Level 1 atmosphere becomes time resolved as
  `Atmospheric_Temperature_K(atmosphere_time, altitude)` and
  `Atmospheric_Pressure_hPa(atmosphere_time, altitude)`.
- [ ] ERA5 pressure-level reanalysis is the temporal backbone for the canonical
  Level 1 atmosphere because it provides consistent hourly atmospheric state
  across long continuous sessions.
- [ ] Radiosondes remain the preferred local in-situ observational reference
  for validation/QA when a suitable sounding is available near the session.
- [ ] Radiosonde availability does not force abrupt piecewise replacement of
  the hourly ERA5 backbone.
- [ ] USSA76 remains explicit vertical extension and full fallback when the
  configured external sources cannot provide the required atmosphere.
- [ ] Level 2 performs no external meteorological IO; it interpolates the
  materialized Level 1 atmosphere to each retrieval `block_time`.
- [ ] Solar day/night segments and atmospheric time resolution remain
  independent concepts.

### Atmospheric comparison QA

- [ ] Add a Level 1 atmospheric comparison figure to `figures/`, with a
  session/level-identifying filename such as
  `SESSION_L1_AtmosphericProfile.webp`.
- [ ] When radiosonde is available, compare three profiles on a common altitude
  grid: ERA5, radiosonde and the canonical profile actually used by MILGRAU.
- [ ] Plot temperature and pressure profiles and the corresponding differences.
- [ ] Report compact quantitative metrics over configurable altitude bands:
  temperature bias/RMSE, pressure bias/relative difference and valid overlap.
- [ ] Also compare the derived molecular state relevant to lidar retrieval
  (molecular number density/backscatter or an equivalent directly traceable
  quantity) so QA measures scientific retrieval impact, not only meteorological
  differences.
- [ ] Record radiosonde launch/target time, ERA5 analysis time, spatial source
  metadata and time offsets in the figure/provenance.
- [ ] Treat ERA5-versus-radiosonde agreement as a consistency/validation QA,
  not as a fully independent validation, because radiosonde observations may
  contribute to the reanalysis assimilation system.
- [ ] If no suitable radiosonde exists, generate the atmospheric figure with
  ERA5 + canonical used profile + USSA76/fallback context and mark radiosonde
  as unavailable rather than silently omitting provenance.
- [ ] Define and test the maximum radiosonde time separation used for the QA
  comparison independently of the production ERA5 atmosphere cadence.

## Required before the next release candidate

### Engineering

- [ ] Add explicit block fields for backward valid, forward valid, valid through
  20 km, valid through 25 km and full connected-column status.
- [ ] Define stable reason codes for an upper or lower branch stopping.
- [ ] Add a CI smoke test that builds the wheel, installs it and runs all primary
  CLI `--help` commands.
- [ ] Decide whether the release is source-checkout only or whether packaged
  defaults/assets must make the wheel independently runnable.
- [ ] Add a changelog entry describing the two-sided retrieval, background fit,
  prioritized reference ranges and code cleanup.
- [ ] Protect the release branch with the existing CI workflow as a required
  status check.
- [ ] Triage the remaining test warnings; scientific/numerical warnings must be
  resolved or explicitly justified.

### Minimum scientific gate

- [x] Molecular-only truth: verify near-zero aerosol recovery and connected
  two-sided coverage without interpolation.
- [x] Noise sweep: verify bias, interval behavior, reference-selection stability
  and branch endpoints as signal-to-noise decreases.
- [x] Reference placement: verify controlled cases inside the primary and
  fallback ranges.
- [x] Lidar-ratio mismatch: quantify backscatter/extinction response and retain
  conditional language.
- [x] Residual aerosol at the reference: run sensitivity cases without turning
  them into a probability prior.
- [ ] Background truth: quantify fitted-background bias, interval coverage and
  stability for complete and incomplete temporal blocks.
- [x] Progressive-grid representation: compare retrieval on native and
  progressive grids against common truth.

### Real-data release examples

- [ ] Freeze one clean March 2024 case processed with the exact current recipe.
- [ ] Freeze one plume/cirrus or difficult-background case.
- [ ] Freeze one weak-signal or incomplete-block case.
- [ ] For each case retain configuration, Level 2 product, QA figures and a
  compact table of reference, fallback, background, branch endpoints and
  Monte Carlo support.

## Follow-up validation after the release candidate

- [ ] Sensitivity of the 1 km Rayleigh window against 0.5 and 1.5 km.
- [ ] Broader heterogeneous SPU campaign survey.
- [ ] Characterize photon-counting saturation and replace provisional guards
  with instrument evidence.
- [ ] Characterize overlap and define the validated near-field limit.
- [ ] Quantify fitted gluing slope/intercept covariance materiality.
- [ ] Compare against Raman products under matched time/altitude support.
- [ ] Compare with SCC/ELDA under matched inputs and assumptions.
- [ ] Compare column-integrated products with independent AOD where appropriate.

## Release language

Until the minimum scientific gate and real-data examples are frozen, describe
the product as an experimental two-sided elastic retrieval. After those gates,
the release may claim retrieval to the highest continuously supported altitude
for each block. It must not claim universal validity to 20 or 25 km.

## Repository-history cleanup

Source cleanup and Git-history cleanup are separate tasks. Old quicklooks,
NetCDF files and logs remain in repository history and are the main reason for
the large clone size. Rewriting history can be considered after the release,
with a backup tag and a coordinated forced update/reclone for collaborators.
