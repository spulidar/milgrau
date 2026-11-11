# Atmospheric profiles — molecular state for lidar processing

MILGRAU constructs the time-resolved thermodynamic atmosphere during **LIPANCORA / Level 1**. The Level 2 retrieval then uses the saved Level 1 temperature and pressure to calculate molecular backscatter and extinction for each retrieval time; it does not select or download meteorological data again.

## Why molecular atmosphere matters

Rayleigh molecular scattering and atmospheric density depend on temperature and pressure. Using an inappropriate time, height reference or assumed profile can propagate into the molecular reference and aerosol optical retrieval. For reproducibility, a Level 1 result reports **which atmosphere was actually applied** rather than only listing possible data providers.

## Operational source order

For each required hourly atmosphere:

1. **ERA5 pressure levels**, including validated cached hours.
2. **University of Wyoming upper-air observations**, when suitable.
3. **NOAA IGRA2 historical radiosonde archive**, when an acceptable Wyoming observation is unavailable and the local archive is present.
4. **US Standard Atmosphere 1976 (USSA76)**, explicitly identified as a modeled fallback if observational/reanalysis sources are missing or unusable.

The processing recipe includes:

~~~yaml
level1:
  atmosphere:
    time_resolution_minutes: 60
    source_priority: ["era5", "radiosonde", "ussa76"]
    external_profile_outside_coverage: "ussa76"
    radiosonde:
      synoptic_hours_utc: [0, 12]
      selection: "nearest"
      max_time_delta_hours: 6.0
~~~

The exact numerical values in the installed `config.yaml` govern a run. Solar day/night segments are independent of the atmosphere-hour cadence.

**Important:** the provider order is not a claim of scientific equivalence. An ERA5 reanalysis, a measured sounding and a standardized atmosphere can represent different environmental conditions.

## ERA5 and CDS congestion

When a valid required ERA5 hour is already in the local cache, LIPANCORA uses it even if the CDS queue is congested.

Before a normal run, the CLI can read the public [CDS live dashboard](https://cds.climate.copernicus.eu/live) to obtain an approximate count of globally queued requests. This is a **best-effort reading of a public page**, not a documented stable machine-readable API and not the position or predicted waiting time of an individual MILGRAU request.

The integration uses a **heuristic default threshold of 8,000 queued requests**:

- **At or below the threshold:** ordinary ERA5 cache/download behavior remains available.
- **Above the threshold:** do not create new CDS requests; retain already cached ERA5 hours and use the remaining fallback sequence for missing hours.
- **Dashboard unavailable/unreadable:** record queue state as `unknown`, rather than misreporting "low"; the ordinary ERA5 policy remains available.

Control this behavior at runtime:

~~~bash
milgrau-lipancora -i SESSION_ID --cds-queue-threshold 8000
milgrau-lipancora -i SESSION_ID --no-cds-queue-check
~~~

The threshold is an operational starting value that requires calibration against real use. Reading the dashboard does not cancel or retry older CDS jobs. ERA5 access uses externally configured `cdsapi` credentials; credentials must not be stored in the scientific recipe or NetCDF.

## Explicit no-ERA5 mode

~~~bash
milgrau-lipancora -i SESSION_ID --radiosonde-only
~~~

This disables new and cached ERA5 use for that run and seeks **Wyoming → local IGRA2 → USSA76**. The explicit mode uses an observational selection window up to **12 hours**, accommodating historical observations with 12 UTC launches only. The high-queue fallback also uses this wider observational window. The normal low-queue setting retains the configured six-hour window unless the scientific recipe is changed.

If neither observed source is available, USSA76 is not relabeled as radiosonde: its status is recorded as `ussa76`. This mode uses the same canonical `SESSION_L1.nc` filename; preserve prior results if making a before/after comparison.

## Radiosonde observations and historical archive

MILGRAU first tries suitable Wyoming observations, including an identified Wyoming local cache. The observed launch time must satisfy the configured temporal tolerance. A nominal closest 00/12 UTC launch without a real profile must not be manufactured.

The **NOAA IGRA2** fallback reads an operator-provided historical archive if available locally, for example:

~~~text
.cache/radiosonde/BRM00083779-data.txt
~~~

The original fixed-width historical archive is not automatically downloaded or rewritten by MILGRAU. Validated observation subsets can be cached as CSV files and accompanying JSON metadata. When Wyoming and IGRA2 observations coexist for the same nominal launch, they are kept distinguishable, including separate file identities and source metadata.

An observation at 12 UTC may serve a 19 UTC target if its actual seven-hour age is permitted by the active policy. The **actual selected observation time and offset** must be checked rather than assuming that the nearest theoretical launch exists.

## Time and height representation

The Level 1 atmospheric time axis covers the entire lidar session at the configured cadence. For each atmosphere hour, temperature and pressure are mapped to the lidar altitude grid.

External-profile geometric altitude above sea level must be reconciled with the lidar station altitude and its above-ground-level coordinate. Within profile coverage, temperature is interpolated linearly in height and pressure in logarithmic space. Where source coverage ends and policy permits, USSA76 provides an **explicit vertical extension** rather than an unreported assumption.

For a Level 2 time block between Level 1 atmosphere hours, LEBEAR interpolates temperature linearly in time and pressure in $\log(P)$ before calculating molecular scattering. It does not make a separate source choice.

## How the source appears in Level 1

The main Level 1 variables include:

| Variable | Interpretation |
| --- | --- |
| `Atmospheric_Temperature_K(atmosphere_time, altitude)` | Applied temperature [K] |
| `Atmospheric_Pressure_hPa(atmosphere_time, altitude)` | Applied pressure [hPa] |
| `Atmospheric_Source_Type(atmosphere_time)` | `era5`, `radiosonde` or `ussa76` |
| `Atmospheric_Observation_Source(atmosphere_time)` | `wyoming_siphon` or `igra2` for radiosonde hours; `none` otherwise |
| `Atmospheric_Source_Time_Delta_hours(atmosphere_time)` | Temporal offset of the selected source |
| `Atmospheric_USSA76_Fallback_Fraction(atmosphere_time)` | Portion of the profile supplied by the standard atmosphere |

Global attributes also describe the selection reason, CDS queue status, sources used across the session and associated observational archives where applicable. A single Level 1 may mix different sources across hours; global summaries never replace per-hour evidence.

LEBEAR inherits key source metadata into Level 2 along with the exact upstream Level 1 file identity.

## Atmospheric QA and figures

Where an appropriate radiosonde observation is available, LIPANCORA can save mapped `Radiosonde_QA_Temperature_K` and `Radiosonde_QA_Pressure_hPa` fields, with its actual provider/time and identified cache. This is one observational comparison, not necessarily an independent truth: upper-air observations may also influence reanalysis assimilation.

The Level 1 atmospheric profile/evolution figures help inspect source differences, atmospheric variation during a session and molecular implications. They are diagnostic views; verify exact per-hour identities and scientific masks in NetCDF before publishing.

## Scientific review checklist

- What source contributed to each relevant hour?
- Was a radiosonde launch actually observed, and how far was it from lidar acquisition time?
- Is the launch station representative of the lidar location?
- Was some or all of the vertical column extended by USSA76?
- Did the queue policy skip a new ERA5 request, or was ERA5 already cached?
- Were the molecular reference and derived aerosol coefficients interpreted with the same Level 1 atmosphere?
- If processing was repeated with a different atmosphere, were previous NetCDFs preserved for comparison?

See [LIPANCORA](lipancora.md), [LEBEAR](lebear.md) and [Scientific quality](scientific_quality.md).
