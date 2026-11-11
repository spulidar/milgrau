# LIPANCORA — Level 1: corrected lidar signal and molecular atmosphere

LIPANCORA transforms standardized Level 0 acquisition records into scientifically usable lidar signals, with instrumental diagnostics, uncertainty information and a time-resolved thermodynamic atmosphere.

## Scientific purpose

The raw intensity registered by a lidar is affected by detector response, background light, channel behavior, acquisition configuration and geometric range. Those contributions must be treated before aerosol inversion.

LIPANCORA applies the **implemented and configured instrumental corrections**, records the correction status and calculates signal quantities needed by Level 2. When historical instrumental parameters are uncertain, the product should retain that limitation rather than imply a fully characterized response.

## Inputs

- A validated `SESSION_L0.nc` file.
- Applicable channel calibration and station profile from `station.yaml`.
- Level 1 settings in `config.yaml`.
- Temperature/pressure information derived from ERA5, observed radiosondes or USSA76.

Level 1 retains the session and solar-segment identities supplied by Level 0.

## Run Level 1

~~~bash
milgrau-lipancora
milgrau-lipancora -i SESSION_ID
milgrau-lipancora -i SESSION_ID --radiosonde-only
~~~

For figures only:

~~~bash
milgrau-lipancora --figures-only -i SESSION_ID
~~~

The `--force` option refreshes selected products (or all selected figures together with `--figures-only`). Preserve baseline NetCDFs before changing scientific settings or atmospheric modes.

## Instrumental corrections

The current processing path includes, where applicable:

- detector dead-time correction for photon-counting channels;
- channel-specific bin alignment/shift;
- background estimation and subtraction;
- application of relevant historical calibration context;
- propagation of available measurement uncertainty;
- range correction and output of diagnostic quantities.

**Photon counting.** A dead-time model corrects observed count rates within its applicable assumptions. A numerical guard or saturation flag is **not** equivalent to having experimentally characterized the physical saturation response of every detector. Inspect the saved warning and saturation diagnostics.

**Channel alignment.** Applied bin shifts must correspond to station-supported calibration. Missing or provisional values can affect subsequent interpretation.

**Background estimation.** MILGRAU evaluates a configured altitude interval using a robust median/MAD-based estimator and records diagnostics such as usable-bin count and outlier fraction. A stable background estimate is important because a small residual can become significant after range correction.

## Corrected signals and RCS

The range-corrected signal is conceptually

\[
\mathrm{RCS}(r,t)=\bigl[P(r,t)-B(t)\bigr]r^2,
\]

where \(\,P(r,t)\,\) is the measured signal and \(B(t)\) is the background estimate, with instrument-specific corrections and channel calibration applied as configured.

The reported RCS is not automatically an absolute aerosol backscatter coefficient. It is a corrected lidar signal used in later retrieval, and its uncertainty and valid support must be considered.

LIPANCORA retains additional pre-background and corrected-signal fields where available, allowing reviewers to distinguish acquisition intensity, background treatment and final RCS.

## Molecular atmosphere and source selection

Level 1 materializes an hourly temperature and pressure atmosphere on the lidar altitude grid. The normal operational priority is:

**ERA5 → Wyoming observation → NOAA IGRA2 archive → USSA76.**

A validated ERA5 cache hour is retained even if the public CDS queue is congested. When the queue indicator exceeds the configured heuristic threshold, no new CDS request is initiated for the missing data; acceptable observed radiosondes are then considered before USSA76.

`--radiosonde-only` explicitly disables ERA5/CDS and requests the observation-first policy. Wyoming is attempted before IGRA2; if neither provides a suitable observation, USSA76 is identified as a model fallback rather than an observation.

A source can vary between atmosphere hours. The NetCDF records `Atmospheric_Source_Type`, radiosonde-provider identity in `Atmospheric_Observation_Source`, time offsets and USSA76 fractions. It also preserves context about the queue decision.

Interpolation of external profiles to the lidar vertical grid requires attention to height reference, vertical coverage and thermodynamic consistency. Outside the observed/model input's vertical support, the configured USSA76 extension is recorded explicitly.

For exact source rules, tolerances and scientific cautions, see [Atmospheric profiles](atmospheric_profiles.md).

## Other diagnostic outputs

Where configured and supported, Level 1 includes:

- planetary boundary-layer (PBL) estimates;
- tropopause diagnostics from the available atmosphere;
- channel/correction success and instrument-quality flags;
- robust background measurements and uncertainty estimates;
- hourly atmospheric evolution and comparisons with a radiosonde QA reference.

These estimates can fail or be unavailable without invalidating every corrected signal. They are scientific diagnostics, not automatic evidence of aerosol optical retrieval.

## Scientific output and figures

The canonical product is `SESSION_L1.nc`, which includes corrected signal/RCS with uncertainty, the complete time-resolved atmosphere and provenance from Level 0.

Automatically generated figures can include RCS quicklooks, mean RCS and atmospheric profile/evolution views. Figures are visualizations of saved scientific data; they are not a substitute for inspecting masks, units, flags and full-resolution NetCDF arrays.

## Quality checks before Level 2

1. Confirm the Level 0 source and historical channel calibration.
2. Review background behavior and the range over which it was estimated.
3. Check whether any photon-counting correction or saturation warning limits interpretation.
4. Examine signal uncertainty, gaps and valid vertical coverage.
5. Verify the actual atmospheric source **for each hour**, its age and its USSA76 contribution.
6. Check differences between the chosen molecular atmosphere and an independent-looking observational QA sounding cautiously: observations may also have entered reanalysis assimilation.
7. Confirm that Level 1 provenance and its source identity match the intended scientific run.

The absence of a valid Level 1 uncertainty should not be interpreted as zero uncertainty.

## Next stage

[LEBEAR](lebear.md) takes corrected signals, uncertainties and the Level 1 molecular atmosphere into elastic aerosol retrieval. See also [Scientific quality](scientific_quality.md).
