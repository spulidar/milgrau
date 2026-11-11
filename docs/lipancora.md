# LIPANCORA — Level 1: corrected lidar signal and molecular atmosphere

LIPANCORA transforms a standardized Level 0 Licel acquisition into **instrumentally corrected lidar signals** and a **time-resolved molecular atmosphere**. It retains quality information and uncertainty needed to decide whether a signal is suitable for later elastic aerosol inversion.

**LIPANCORA does not retrieve aerosol backscatter or extinction.** Those optical coefficients are conditional outputs of [LEBEAR / Level 2](lebear.md).

## 1. Input and scientific context

The required input is a validated canonical `SESSION_L0.nc` containing acquisition times, native per-channel range resolution, `Raw_Lidar_Data`, `Laser_Shots`, channel identity, historical instrument context and Level 0 solar/segment fields. Associated dark-current measurements are used only when their Level 0 availability fields support them.

The station and calibration record in `station.yaml` determines detector mode, dead-time constant, bin shift, background offset and, **only if characterized**, physical photon-counting saturation rate. The scientific recipe in `config.yaml` determines background and PBL settings and the atmosphere selection policy.

A calibration missing from the dated instrument record can use the explicit `neutral_with_warning` policy, which applies neutral numerical values and sets `calibration_assumed_neutral=1`. This is **not an experimentally determined calibration**.

## 2. Run Level 1

~~~bash
milgrau-lipancora
milgrau-lipancora -i spu_20250511-0012Z_20250511-0737Z
milgrau-lipancora -i SESSION_ID --force
~~~

For explicit processing without ERA5:

~~~bash
milgrau-lipancora -i SESSION_ID --radiosonde-only
~~~

To regenerate scientific plots **without recomputing Level 1**:

~~~bash
milgrau-lipancora --figures-only -i SESSION_ID
milgrau-lipancora --figures-only -i SESSION_ID --force
~~~

`--radiosonde-only` cannot be combined with `--figures-only`. The same canonical `SESSION_L1.nc` filename is used for different atmosphere policies. Preserve existing products before comparing two scientific recipes, and reprocess Level 2 after changing Level 1.

The full selector, incremental and output-handling guide is [Operating MILGRAU](operating_milgrau.md).

## 3. Channel-native correction before interpolation

Level 0 channels can have different native range-bin widths. LIPANCORA performs **instrumental corrections on each channel's own native range grid** before interpolating corrected numerical arrays onto a shared Level 1 grid.

For a native bin spacing \(\Delta r\), its range-bin centers are

$$
r_i=(i+\\tfrac{1}{2})\\Delta r.
$$

The common Level 1 grid uses the **finest** native channel resolution in the input. Numerical signals are interpolated in altitude; diagnostic boolean masks use nearest-neighbor interpretation. This produces a common coordinate for visualization and L2, but interpolation does **not** create independently resolved physical information.

The current Level 1 coordinate `altitude` is labeled as range-bin centers above the station, in meters. Actual observation geometry must be assessed when interpreting this coordinate as vertical height.

## 4. Dark-current correction

When Level 0 provides an accepted dark-current profile for a channel, LIPANCORA computes the mean over the available dark acquisitions, \(D(r)\), and an estimated standard error from their spread:

$$
\\sigma_D(r)=\\frac{\\operatorname{std}_t D_t(r)}{\\sqrt{n_D}}.
$$

The software subtracts the dark-current mean **before** subsequent detector corrections. A dark-current association based on time tolerance is documented in [LIBIDS](libids.md); it is not proof that the detector conditions were identical to the main measurement.

If no usable dark acquisition exists, processing may continue without subtraction, with `dark_current_used(channel)=0`. The missing physical correction must not be interpreted as zero true dark current.

## 5. Photon counting, dead time and saturation

The raw photon-counting signal contains accumulated counts \(N(r,t)\). The observed photon-counting rate in MHz is computed using the **per-profile channel shot count** \(N_{\rm shots}\) and the native round-trip bin time

$$
\\Delta t_{\\rm bin,\\mu s}=\\frac{2\\Delta r}{c}\\times10^6,
\\qquad
q_{\\rm obs}=\\frac{N}{N_{\\rm shots}\\,\\Delta t_{\\rm bin,\\mu s}}.
$$

The rate diagnostic `pc_observed_rate_mhz_max(time,channel)` uses the **directly observed counts before dark-current subtraction**, shift, dead-time correction or background subtraction.

### Current productive correction order

The current code subtracts the native dark-current counts first, converts to a rate and applies the calibrated nonparalyzable dead-time model:

$$
q_{\\rm corr}=\\frac{q_{\\rm dark\\,sub}}{1-\\tau q_{\\rm dark\\,sub}},
$$

where \(\tau\) is the channel dead-time constant expressed in microseconds. The numerical implementation uses a configured positive denominator floor (`deadtime_min_denominator`, distributed value 0.05). **This numerical clip is not proof of physical detector saturation or correction accuracy.**

The observed-rate denominator is also evaluated separately for diagnostics, before dark-current subtraction.

The current order, **dark current → dead time → bin shift → robust background**, is the implemented procedure; whether it is physically optimal for each historical instrument configuration requires real detector evidence and is an open review item. Do not silently reorder the correction in a scientific analysis.

### Physical saturation is a separate concept

A pixel is flagged by `pc_saturation_mask` only if a **characterized** physical maximum rate exists in the corresponding instrument calibration and the raw observed rate exceeds it. When `pc_saturation_characterized=0`, a zero saturation mask does **not** establish unsaturated operation.

The `deadtime_clipping_fraction` and `deadtime_raw_clipping_fraction` fields describe numerical denominator behavior, not experimentally observed detector nonlinearity.

## 6. Signal uncertainty

For photon counting, the raw-count Poisson contribution uses the **observed raw nonnegative counts**, not the potentially negative dark-subtracted counts:

$$
\\sigma_{q,\\rm count}=\\frac{\\sqrt{\\max(N,0)}}{N_{\\rm shots}\\Delta t_{\\rm bin,\\mu s}}.
$$

When a dark-current uncertainty is available, its rate-scaled variance is added in quadrature:

$$
\\sigma_{q}^2=
\\sigma_{q,\\rm count}^2+
\\left(\\frac{\\sigma_D}{N_{\\rm shots}\\Delta t_{\\rm bin,\\mu s}}\\right)^2.
$$

The nonparalyzable dead-time calculation propagates the uncertainty through the same denominator using the implemented derivative, with the numerical floor if triggered.

For analog channels, the Level 0 input is in mV per shot and LIPANCORA does **not** apply photon-counting dead time. The current analog uncertainty estimate uses the robust far-range background scale, combined in quadrature with available dark-current uncertainty.

These numerical uncertainties represent **the uncertainty terms included in the current calculation**. They do not cover unknown instrument-response calibration, unmeasured physical saturation or all potential systematic errors.

## 7. Bin alignment and robust far-range background

A configured integer `bin_shift_bins` aligns the corrected channel. Shift-induced missing edge bins are kept `NaN`, and their fraction is recorded. The software does not fill them with fabricated signal.

After alignment, the atmospheric background is estimated for each time/channel using the configured far-range window. The distributed window is 29,000–29,999 m; whether it is appropriate for a particular observation is **a scientific assumption that must be checked**.

The robust estimator uses the median \(m\) and median absolute deviation (MAD):

$$
m=\\operatorname{median}(x_i),\\qquad
\\sigma_{\\rm robust}=1.4826\\operatorname{median}(|x_i-m|).
$$

The reported standard error of the median, under its Gaussian approximation, is

$$
\\sigma_m\\approx\\frac{1.2533\\,\\sigma_{\\rm robust}}{\\sqrt{n}}.
$$

The current implementation requires at least eight valid bins in the configured window. Samples beyond 4.5 robust standard deviations are **counted as outliers for diagnostics**; the recorded background location is the median of the supported window, not a new trimmed mean.

A zero MAD is retained as zero rather than silently replaced with a non-robust standard deviation. This is numerically robust but does not independently prove negligible systematic background error.

The configured channel background offset is applied to the location estimate before subtraction. The saved `background_estimate`, `background_standard_error`, `background_robust_scale`, `background_valid_bins` and `background_outlier_fraction` expose the method's diagnostics.

## 8. Corrected signal and range correction

After detector/shift correction, let \(S_{\\rm preBG}\) denote the pre-background signal and \(B(t)\) the estimated far-range background. LIPANCORA produces

$$
S_{\\rm corrected}(r,t)=S_{\\rm preBG}(r,t)-B(t),
$$

$$
\\mathrm{RCS}(r,t)=r^2S_{\\rm corrected}(r,t).
$$

The saved 1σ error for the corrected signal combines the preceding signal error and the uncertainty in the estimated background location in quadrature. The RCS error is multiplied by \(r^2\) in the same numerical path.

**Important:** the background-location error is shared across the altitude bins of a profile/channel, and the current metadata identify this correlation. Treating every RCS altitude sample as statistically independent would underestimate uncertainty in some later averages.

The RCS uses channel-native corrected signal units multiplied by m². It is **not an absolute aerosol backscatter coefficient**.

## 9. Solar context, boundary layer and tropopause

LIPANCORA copies the measured-time solar elevation, day/night regime and contiguous segment identity from Level 0 **without recomputing them**. The geometrical solar algorithm and its timestamp-unit correction are covered by [Solar regimes](solar_regimes.md).

For the planetary boundary layer (PBL), the current diagnostic uses the configured RCS channel (distributed: `532.AN`) in the 500–4000 m search interval. It smooths the vertical signal (15 bins by default) and identifies the strongest **negative vertical gradient**, avoiding simple edge-padding artifacts.

`PBL_Height_km(time)` is a **gradient-method estimate**, not a direct physical measurement of mixing height. When the reference channel is absent or correction failed, the PBL diagnostic is omitted with a warning rather than silently substituting another channel. Unsupported estimates are `NaN`.

The hourly atmospheric profiles also support cold-point and WMO-style lapse-rate tropopause diagnostics (`Tropopause_CPT_km` and `Tropopause_LRT_km`). Those diagnostics depend on the selected source's vertical temperature coverage and should not be treated as independently observed lidar tropopause heights. Missing scalar compatibility summaries may use the documented sentinel; use time-resolved arrays and source flags for research.

## 10. Atmospheric temperature and pressure

LIPANCORA builds `Atmospheric_Temperature_K(atmosphere_time,altitude)` and `Atmospheric_Pressure_hPa(atmosphere_time,altitude)` with normally hourly cadence.

Its standard source order is **ERA5 → observed Wyoming → local NOAA IGRA2 → USSA76**, including explicit source/vertical fallback tracking. `--radiosonde-only` bypasses cached and network ERA5. Each hour records `Atmospheric_Source_Type`, and a radiosonde hour records `Atmospheric_Observation_Source` as Wyoming or IGRA2. An individual session may mix sources.

External profiles are mapped from height above sea level to the lidar's above-ground-level range grid; standard atmosphere contributions outside source coverage remain labeled. LEBEAR obtains the molecular atmosphere from this saved Level 1 product, rather than downloading a new source.

Read [Atmospheric profiles](atmospheric_profiles.md) for exact time windows, CDS queue behavior, observed source selection, and source-provenance interpretation.

## 11. Figures and scientific review

The canonical scientific file is `SESSION_L1.nc`. LIPANCORA can generate:

| Figure | Scientific question |
| --- | --- |
| Time–height RCS quicklook, per selected channel and altitude range | Where does corrected signal exist, and how does its structure evolve in time? |
| Global mean RCS profile | What is the session-average corrected signal by altitude/channel? |
| Atmospheric profile | What thermodynamic conditions, source identities and radiosonde comparison support the session? |
| Atmospheric evolution | How do temperature/pressure and the selected atmospheric state vary with time? |

Plots may use kilometers or display aggregation while the NetCDF retains the full saved grid. Figure failures are **nonfatal** to an already validated scientific Level 1; review figure status separately from Level 1 success.

Before using the file for Level 2, inspect: channel status; `calibration_assumed_neutral`; dark-current use; raw observed PC rates and actual saturation characterization; numerical dead-time clipping; shift-related gaps; background count/error/outliers; valid RCS uncertainty; actual atmosphere by hour; and PBL/tropopause availability.

See the [Level 1 variable dictionary](level1_schema.md) and [Scientific quality](scientific_quality.md).

## 12. Scientific limitations and pending real-data validation

This first-pass scientific guide is grounded in the productive methods and available tests, **not yet in representative real instrument measurements**. A complete validation should compare the same Licel session under identical calibration and atmospheric policies across the historical instrument eras, including saturated/low-count channels, weak signal, available/missing dark, mixed native resolutions and suitable/unsuitable background windows.

A scientifically annotated Level 1 worked example requires exact original source identity, station/calibration record, computed signal and error/diagnostics, chosen atmosphere per hour, output file checksum and independently reviewed differences.

**Do not mark that real-data validation complete from synthetic CI alone.** Submit source material as described in [Level 0 validation workbook](level0_validation_workbook.md), then assess the Level 1 method case by case.
