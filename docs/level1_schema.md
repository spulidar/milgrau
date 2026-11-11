# Level 1 NetCDF — scientific variables, units and QA flags

This dictionary describes the **current LIPANCORA output** at the level of physical interpretation. The exact field set can depend on available channels, the selected meteorological source, optional diagnostics and the saved software revision. Inspect the NetCDF's coordinate/variable metadata for a specific publication.

Start with [LIPANCORA](lipancora.md) and [Atmospheric profiles](atmospheric_profiles.md).

## Coordinates and scientific grouping

| Coordinate/dimension | Meaning | Unit |
| --- | --- | --- |
| `time` | Original Level 0 acquisition profile timestamps, normalized to UTC in the saved NetCDF | Datetime UTC |
| `channel` | Normalized acquisition channel, e.g. `532.AN`, `532.PC` | Text |
| `altitude` | Shared range-bin-center coordinate on the finest native Level 0 resolution | m AGL/range |
| `atmosphere_time` | Time-resolved thermodynamic sampling (normally hourly) | Datetime UTC |
| `segments` | Contiguous Level 0 solar day/night segment table | Dimensionless index |

Signal variables contain **time, channel and altitude** axes (saved order may be inspected for each file). Background and detector diagnostics can be per-channel or per-time/channel; molecular atmosphere varies by atmosphere time and altitude. Do not assume a raw acquisition's `time` equals an `atmosphere_time`.

## Instrument-corrected signal products

| Variable | Meaning | Saved units / convention |
| --- | --- | --- |
| `signal_pre_background` | Signal after optional dark-current subtraction, PC dead time and bin shift, but before robust far-range background subtraction | Channel-native corrected units |
| `signal_pre_background_error` | One-sigma uncertainty before far-range background subtraction | Channel-native corrected units |
| `corrected_signal` | Corrected signal after far-range background estimate | Channel-native corrected units |
| `corrected_signal_error` | One-sigma uncertainty of corrected signal | Channel-native corrected units |
| `range_corrected_signal` | `corrected_signal` × range² | `a.u. m^2` (NetCDF label) |
| `range_corrected_signal_error` | One-sigma RCS uncertainty | `a.u. m^2` |
| `pc_saturation_mask` | 0/1 mask where a **characterized** observed-rate physical PC limit was exceeded, otherwise not flagged | Dimensionless; interpret with characterization flag |

These signals are *not* calibrated aerosol optical backscatter/extinction. The acquisition's original analog and photon-counting signal modes have different native physical units.

## Channel status, calibration and detector diagnostics

| Variable | Typical axes | Meaning |
| --- | --- | --- |
| `channel_correction_success` | `channel` | 1 = correction succeeded; 0 = failed |
| `dark_current_used` | `channel` | 1 if usable associated dark-current profile contributed |
| `calibration_assumed_neutral` | `channel` | 1 = explicit fallback instead of traceable channel calibration |
| `deadtime_correction_applied` | `channel` | Whether PC dead-time correction was active |
| `bin_shift_bins` | `channel` | Applied integer bin shift |
| `bin_shift_invalid_fraction` | `time,channel` | Fraction of masked bins introduced by shift |
| `deadtime_clipping_fraction` | `time,channel` | Fraction numerically clipped **after dark subtraction** |
| `deadtime_raw_clipping_fraction` | `time,channel` | Diagnostic fraction clipped using directly observed PC rate |
| `deadtime_min_denominator_observed` | `channel` | Minimum productive denominator, when applicable |
| `deadtime_raw_min_denominator_observed` | `channel` | Minimum denominator from raw observed rate |
| `deadtime_min_denominator_allowed` | `channel` | Configured numerical denominator floor |
| `pc_observed_rate_mhz_max` | `time,channel` | Maximum **raw observed** PC rate before dark subtraction [MHz] |
| `pc_saturation_characterized` | `channel` | 1 only when a measured physical saturation limit is known |
| `pc_saturation_rate_limit_mhz` | `channel` | Physical PC limit [MHz], otherwise `NaN` |
| `pc_saturation_fraction` | `time,channel` | Fraction exceeding characterized physical limit |

**Essential distinction:** `deadtime_clipping_fraction > 0` indicates numerical regularization, not by itself detector saturation. A zero `pc_saturation_mask` with `pc_saturation_characterized=0` is **not evidence of unsaturated operation**.

## Robust atmospheric-background diagnostics

| Variable | Typical axes | Interpretation |
| --- | --- | --- |
| `background_estimate` | `time,channel` | Median-based location minus configured offset |
| `background_standard_error` | `time,channel` | Gaussian-equivalent uncertainty of median location |
| `background_robust_scale` | `time,channel` | \(1.4826\times\mathrm{MAD}\), channel-corrected units |
| `background_valid_bins` | `time,channel` | Number of finite far-range bins |
| `background_outlier_fraction` | `time,channel` | Fraction outside the configured robust-scale threshold |

The background-location uncertainty is common to the altitude bins within one acquisition/channel and is documented as correlated. The distributed estimator is `median_mad`, with far-range window and minimum-bin parameters in `config.yaml`.

## Time-resolved molecular atmosphere

| Variable | Axes | Unit / meaning |
| --- | --- | --- |
| `Atmospheric_Temperature_K` | `atmosphere_time,altitude` | Applied temperature [K] |
| `Atmospheric_Pressure_hPa` | `atmosphere_time,altitude` | Applied pressure [hPa] |
| `Atmospheric_Source_Type` | `atmosphere_time` | `era5`, `radiosonde`, `ussa76` |
| `Atmospheric_Observation_Source` | `atmosphere_time` | `wyoming_siphon`, `igra2` or `none` |
| `Atmospheric_Source_Time_Delta_hours` | `atmosphere_time` | Offset of the selected source [h] |
| `Atmospheric_USSA76_Fallback_Fraction` | `atmosphere_time` | Fraction of altitude grid using standard-atmosphere values |
| `Atmospheric_Source_Min_Altitude_ASL_m` | `atmosphere_time` | External source's lower altitude [m ASL] |
| `Atmospheric_Source_Max_Altitude_ASL_m` | `atmosphere_time` | External source's upper altitude [m ASL] |
| `Tropopause_CPT_km` | `atmosphere_time` | Cold-point tropopause [km], if supported |
| `Tropopause_LRT_km` | `atmosphere_time` | Lapse-rate tropopause [km], if supported |

An atmosphere hour may be supported by an external source at some heights and **USSA76 vertical extension** at other heights. Inspect the fallback fraction and actual provider by hour, not just the summary global attribute. Thermodynamic profiles are positive/finitely validated before writing.

## Optional radiosonde QA and PBL fields

| Variable | Axes | Meaning |
| --- | --- | --- |
| `Radiosonde_QA_Temperature_K` | `altitude` | One mapped observed reference sounding [K], if available |
| `Radiosonde_QA_Pressure_hPa` | `altitude` | Its observed pressure reference [hPa], if available |
| `PBL_Height_km` | `time` | Gradient-based PBL diagnostic [km] for configured reference channel |

A radiosonde QA reference is **not** an additional productive atmospheric hour unless the source selection actually used that observation. The PBL field may be absent if the configured reference channel is unavailable or its correction failed; unsupported heights can be `NaN`.

## Solar/segment context copied from Level 0

`solar_elevation_deg(time)`, `solar_regime(time)`, `segment_id(time)`, `Segment_Label(segments)`, `Segment_Regime(segments)`, `Segment_Start_Time_UTC(segments)` and `Segment_End_Time_UTC(segments)` are inherited from the **exact upstream Level 0**, without recomputation.

This distinction matters after the documented November 2026 pandas timestamp-unit solar correction: products created under an older software revision should be audited before comparing segments. See [Solar regimes](solar_regimes.md).

## Global scientific provenance

Important attributes include `Processing_level`, `Pipeline`, `Input_Level0_File`, `Altitude_units`, `Altitude_grid_convention`, `background_estimator`, `neutral_legacy_calibration_channel_count`, `thermodynamic_source_priority`, `thermodynamic_sources_present`, `thermodynamic_profile_altitude_reference`, `thermodynamic_station_altitude_m`, `atmosphere_selection_reason`, `cds_queue_state`, `era5_network_allowed_at_selection` and `level1_atmosphere_mode`.

When radiosonde QA exists, attributes include observed provider, actual selected sounding time, time offset, archive/cache identity, checksum and vertical extension. Software/configuration and upstream source provenance are appended by the general provenance writer.

`level1_atmosphere_mode=radiosonde_only` describes explicit operator selection, not a claim that every hour used a measured sounding: USSA76 can still be the last resort.

## How to inspect a saved Level 1

~~~bash
milgrau-inspect SESSION_ID --validate
milgrau-inspect SESSION_ID --full
~~~

These commands inspect stored fields and validate their structural contract. A structural pass does not prove the dark-current calibration, detector saturation behavior, molecular representativeness or validity of a downstream aerosol inversion.

## Remaining scientific evidence

Real representative Level 1 runs are still needed to verify the uncertainty behavior, saturation/shift effects, far-range background suitability and mixed-atmosphere sensitivity across historical SPU profiles. See [LIPANCORA](lipancora.md) and [Scientific quality](scientific_quality.md).
