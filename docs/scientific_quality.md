# Scientific quality, uncertainty and interpretation

MILGRAU aims to make scientifically relevant assumptions and validity information visible in its products. **An executed command, generated figure or finite number does not alone certify a valid retrieval.**

This guide describes how to evaluate products before using them in reports, papers or external comparison.

## A practical review sequence

| Stage | What to check first | Why it matters |
| --- | --- | --- |
| Level 0 / LIBIDS | Measurement interval, channel configuration, dark current, acquisition flags | Defines which observations actually contributed |
| Level 1 / LIPANCORA | Corrections, background, detector status, RCS uncertainty, molecular source | Affects corrected signal and molecular extinction/backscatter |
| Level 2 / LEBEAR | Reference range, gluing, lidar ratio, branch validity, Monte Carlo and temporal support | Determines which optical values are interpretable |
| SCC comparison | Same observations, time interval, atmosphere, ancillary data and method | Differences may arise from different scientific assumptions |

## 1. Availability versus scientific validity

A NetCDF file may be present while some channels, blocks, wavelengths or altitudes remain unusable. MILGRAU can preserve partial results and explicit failure information for scientific diagnosis.

Distinguish the following:

- **Available:** a variable or file exists.
- **Finite:** a stored numerical value is not missing.
- **Supported:** the underlying measurements and required path/uncertainty checks were satisfied.
- **Scientifically interpretable:** method assumptions, calibration and remaining physical limitations are acceptable for the research question.

These concepts are not interchangeable.

## 2. Quality of the raw acquisition

Confirm the correct station profile, laser and detector configuration for the measurement date. Check whether dark-current data were associated within the configured time allowance and whether channels are complete.

When measurements have large gaps, unusual shot counts or missing calibration evidence, record the limitation. A retained warning should not be rewritten as proof of normal detector behavior.

## 3. Corrected signal quality

Background, dead-time correction, bin shifts and channel alignment can affect the Level 1 signal and its uncertainty. Review the saved background estimate and diagnostics before interpreting apparent weak signal at high altitude.

A numerical photon-counting guard cannot replace laboratory characterization of a detector's saturation response. Near-field overlap remains a separate physical limitation.

Missing uncertainty is **unknown**, not zero. Do not use zero dispersion as a stand-in for unavailable signal-noise information.

## 4. Molecular atmosphere and provenance

The Level 1 atmosphere may use ERA5, Wyoming, NOAA IGRA2 or the US Standard Atmosphere, including a vertical USSA76 extension to an external profile.

For each relevant period, record:

- the effective `Atmospheric_Source_Type`;
- `Atmospheric_Observation_Source` for radiosonde hours;
- observation/analysis time offset and available vertical support;
- standard-atmosphere contribution;
- the source-selection context, especially an operational CDS queue fallback.

The public CDS queue indicator is an **operational heuristic**, not a scientific quality index. ERA5, observed soundings and USSA76 may yield different temperature and pressure profiles. A radiosonde QA comparison is useful, but not necessarily independent of data assimilated into ERA5.

If a later normal run replaces a fallback atmosphere with newly available ERA5, treat the resulting L1 and downstream L2 as a changed scientific product; preserve and compare the former baseline where needed.

## 5. Retrieval assumptions

Elastic retrieval requires a reference boundary and an aerosol lidar ratio. The monthly lidar-ratio values are declared empirical assumptions; their original derivation remains incompletely documented. They should not be presented as independently observed values for each night.

A high reference region is not automatic proof of purely molecular atmosphere. Examine Rayleigh fit quality, selection metrics and continuous valid path.

Aerosol extinction from elastic inversion is conditional on the assumed lidar ratio and must not be described as an independent Raman extinction observation.

## 6. Uncertainty and support

LEBEAR reports nominal solutions and Monte Carlo summary measures, subject to the implemented perturbation scheme. Interpret uncertainty together with sample counts, valid realizations and altitude-specific support.

- `NaN` means missing/unsupported, not zero aerosol.
- Branch endpoints describe continuous validity from the reference, not a fixed grid top.
- Temporal averages may be valid at different altitudes for different numbers of observations.
- A scattering-ratio diagnostic may exist where aerosol inversion is unsupported.
- Systematic uncertainty from unknown instrument parameters or model assumptions is not necessarily represented by random Monte Carlo spread.

## 7. Visualizations

Scientific figures and the Explorer summarize stored data and are suitable for screening, but they may aggregate or reduce samples **for display only**. Read exact values, units, quality flags and provenance from the corresponding NetCDF when drawing scientific conclusions.

Changing a color scale is not a change in the measurement or retrieval.

## 8. Comparisons with SCC or independent observations

For a meaningful comparison, align observation intervals, instrument/channel assumptions, altitude reference, averaging, molecular atmosphere, lidar ratio, reference conditions and valid support. SCC acceptance or production of an ELDA result is valuable interoperability evidence, not proof that numerical outputs must match MILGRAU.

## Suggested record for a publishable example

Record the session ID, UTC interval, instrument profile, source raw measurements, Level 0/1/2 file identity, processing recipe, molecular source per hour, wavelength/segment, reference and lidar ratio, valid altitude range, retrieved value with uncertainty/support and any limitations.

## Further reading

[LIBIDS](libids.md) · [LIPANCORA](lipancora.md) · [Atmospheric profiles](atmospheric_profiles.md) · [LEBEAR](lebear.md) · [Level 2 variables](level2_schema.md) · [Scientific references](scientific_traceability.md)
