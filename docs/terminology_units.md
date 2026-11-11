# Scientific terminology, units and time conventions

This reference fixes the terms used in MILGRAU's user guides. The definitions here describe the current scientific processing and its evidence limits.

## Time, sessions and sources

| Term | Meaning |
| --- | --- |
| **Acquisition** | One original Licel measurement file and its header-defined start/stop |
| **Continuous scientific session** | Chronologically connected acquisitions under an applicable station profile/calibration; may cross midnight or month |
| **Session ID** | `spu_YYYYMMDD-HHMMZ_YYYYMMDD-HHMMZ`, start/end in **UTC** with minute precision |
| **Station-local date** | `YYYYMMDD` selector interpreted in the station timezone from `station.yaml`; not a UTC day |
| **Solar regime** | `day` or `night` based on *geometric* solar-center elevation at the acquisition midpoint |
| **Solar segment** | Contiguous profiles with one regime; `seg00`, `seg01`, ...; not a separate session |
| **SCC Measurement_ID** | Separate external interoperability identity, not the canonical scientific session |
| **UTC month partition** | Used for SCC-specific lidar-ratio ancillary selection; does not split the scientific session |
| **ERA5** | Reanalysis atmospheric profile, not a locally measured radiosonde |
| **Wyoming / IGRA2** | Observational radiosonde providers; document actual observation time and launch station |
| **USSA76** | Standard *modeled* atmosphere used as stated fallback or vertical extension, not an observed sounding |
| **CDS queue** | Operational global congestion indicator; not the age/quality of the atmospheric data |

`NaN` and absent coordinates mean **missing or unsupported**, never physical zero.

## Core physical quantities and units

| Symbol / field | Meaning | Recommended unit |
| --- | --- | --- |
| `r` | Lidar slant range or propagation distance | m |
| `z` | Altitude relative to station, as defined by product coordinates | m AGL |
| `solar_elevation_deg` | Geometric solar-center elevation, not refracted | ° |
| `T`, `Atmospheric_Temperature_K` | Atmospheric temperature | K |
| `P`, `Atmospheric_Pressure_hPa` | Atmospheric pressure | hPa |
| Surface temperature | Surface weather or station scalar | °C |
| `Raw_Data_Range_Resolution` | Native range bin spacing | m |
| Analog Level 0 signal | Converted from channel ADC counts and shot count | mV per shot |
| Photon-counting Level 0 signal | Raw integer counts, per channel/bin/acquisition | counts |
| `Laser_Shots` | Shots used in an acquisition | shots |
| `RCS` | Range-corrected lidar signal | product-specific signal × m² |
| `β_m` / `β_a` | Molecular/aerosol backscatter coefficient | m⁻¹ sr⁻¹ |
| `α_m` / `α_a` | Molecular/aerosol extinction coefficient | m⁻¹ |
| `S_a=α_a/β_a` | Assumed aerosol lidar ratio for elastic retrieval | sr |
| `SR` | Ratio of total to molecular backscatter | dimensionless |
| `Background_Profile` | Raw dark-current acquisition, not the high-altitude fitted atmospheric background | native channel units |

Scientific users must check actual NetCDF variable metadata: some plots convert vertical units to km and optical coefficients to scaled units for readability. The table gives the physical meaning, not a blanket guarantee that every stored variable has exactly the same display unit.

## Scientific distinctions that affect interpretation

**Acquisition quality is not optical retrieval quality.** LIBIDS verifies source data and instrument context. LIPANCORA evaluates correction/uncertainty and molecular profiles. LEBEAR evaluates the supported inverse solution and uncertainty.

**Dark current is not fitted atmospheric background.** The first is a separate detector acquisition, optionally associated to a session; the second is a statistical estimate from the lidar signal at selected altitude.

**RCS is not an aerosol backscatter measurement.** The corrected signal still requires molecular context, a suitable reference and inversion assumptions.

**Retrieved elastic extinction is conditional.** The assumed aerosol lidar ratio and molecular reference influence the result. It should not be described as directly measured Raman extinction.

**Valid, finite, available and scientifically justified are different.** A NetCDF can be present even if an optical profile is partial; a finite scattering ratio can exist outside continuous valid aerosol retrieval support.

**Weather source is not scientific validation.** A cached ERA5 hour can be suitable but reanalysis, observed sounding and standard atmosphere may diverge. The public CDS queue status is an operational choice and not a measure of profile representativeness.

## Important November 2026 timestamp correction

The solar ephemeris formerly converted `DatetimeIndex.asi8` values to Unix seconds assuming the backing datetime unit was always **nanoseconds**. Under newer pandas versions, an index can retain **microsecond** storage. That could alter the apparent date supplied to the solar astronomy equations and therefore change computed solar elevations and possibly assigned day/night segments.

The current `main` explicitly normalizes timestamps to nanoseconds first and tests results against a separate NOAA approximation. This fixes the time representation, **not** the chosen geometric solar method.

**Previously generated Level 0 files must not be assumed unaffected:** where older processing used microsecond datetime storage, solar labels and downstream segment selections may need scientific comparison or reprocessing. Do not silently rewrite historical products; verify source-code identity and compare affected sessions with the [Level 0 verification workbook](level0_validation_workbook.md).

See [Solar regimes](solar_regimes.md), [LIBIDS](libids.md), [Level 0 NetCDF variables](level0_schema.md), [LIPANCORA](lipancora.md) and [LEBEAR](lebear.md).
