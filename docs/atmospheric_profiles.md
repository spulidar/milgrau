# Time-resolved molecular atmosphere

MILGRAU materializes the thermodynamic atmosphere in **Level 1** so Level 2
never fetches meteorological data or silently chooses a new source.

## Productive atmosphere policy

For continuous sessions the Level 1 atmosphere used by MILGRAU is time resolved:

- **ERA5 pressure levels** are the hourly temporal backbone when available.
- **US Standard Atmosphere 1976 (USSA76)** is the explicit vertical extension
  outside external-profile coverage and the full fallback for an hour when ERA5
  is unavailable.
- **Radiosonde** is retained as a local observational comparison/QA reference.
  It does not replace isolated ERA5 hours in the productive time series.

The productive source order is explicit in `config.yaml`. The current
repository recipe uses:

```yaml
level1:
  atmosphere:
    time_resolution_minutes: 60
    source_priority: ["era5", "ussa76"]
    external_profile_outside_coverage: "ussa76"
```

Solar day/night segmentation is independent from atmosphere cadence.

## Level 1 contract

A successful Level 1 contains:

- `atmosphere_time`
- `Atmospheric_Temperature_K(atmosphere_time, altitude)`
- `Atmospheric_Pressure_hPa(atmosphere_time, altitude)`
- `Atmospheric_Source_Type(atmosphere_time)`
- `Atmospheric_Source_Time_Delta_hours(atmosphere_time)`
- `Atmospheric_USSA76_Fallback_Fraction(atmosphere_time)`

It also records source-profile altitude coverage and time-resolved tropopause
diagnostics where available.

The atmosphere time axis brackets the complete lidar session at the configured
cadence. This guarantees that every Level 2 retrieval block can obtain a
thermodynamic state without extrapolating in time.

## Vertical interpolation and fallback

ERA5 source heights are interpreted as geometric altitude above mean sea level.
The resolved station altitude aligns the external ASL profile with the lidar
AGL grid.

Within source coverage:

- temperature is interpolated linearly with altitude;
- pressure is interpolated in `log(P)`.

Outside external-profile coverage, USSA76 is used when
`external_profile_outside_coverage: "ussa76"`. The fraction of each hourly
profile supplied by USSA76 is stored explicitly.

If ERA5 is unavailable for a required hour and `ussa76` is present in
`source_priority`, the complete hourly profile is materialized from USSA76.
No synthetic surface temperature/pressure constants are invented.

## ERA5 IO, global prefetch and reconstructible cache

ERA5 requests use the configured CDS pressure-level dataset and request only the
molecular-atmosphere fields required by MILGRAU: temperature and geopotential.

The downloaded NetCDF and JSON sidecar are IO cache artifacts. The Level 1
NetCDF is the scientific product. LIPANCORA preflights **all pending Level 0 files selected by the current
invocation** before processing any Level 1 file. It deduplicates only the
required UTC hours, including session-bracketing hours. A single selected
session requests only its missing hours; a bulk run plans all pending sessions
together, without requesting every hour of every calendar day.

ERA5 CDS requests group days with **identical lists of required hours** within
one UTC month; this prevents unintended day × time Cartesian-product
overfetch. The initial ceiling is **80,000 CDS fields per request**
(2 variables × the configured number of pressure levels × actual day-hour
combinations). Requests are issued sequentially to avoid saturating CDS.
The CDS may still reject or queue requests below this ceiling.

The validated NetCDF and JSON sidecar for each hourly profile remain under
`.cache/era5/YYYY/MM/`. Reconstructible request metadata lives under
`.cache/era5/requests/`; it never replaces the actual data. Missing hours
are requested only when the cache has no validated compatible file.
Hourly files from completed requests survive a later failed batch.
An uncertain submitted CDS request is **not blindly resubmitted**.

A strictly configured `level1.atmosphere.missing_policy` chooses
`require_era5` (the experimental branch default: fail rather than finalize
scientific L1 with missing ERA5) or `allow_fallback` (explicitly use the
configured USSA76 source if ERA5 is unavailable). The fallback source is
always recorded in the science output.

`cdsapi` is a core MILGRAU dependency because ERA5 is the productive
time-resolved atmosphere backbone. CDS credentials remain outside the repository
and are read by `cdsapi`.

## Radiosonde comparison reference

When the station catalog defines a radiosonde station and a suitable sounding
exists near the session, LIPANCORA maps one comparison sounding to the Level 1
altitude grid and stores:

- `Radiosonde_QA_Temperature_K(altitude)`
- `Radiosonde_QA_Pressure_hPa(altitude)`

with sounding time/offset and coverage provenance.

This sounding is **QA evidence**, not a piecewise productive replacement for
ERA5. ERA5-versus-radiosonde comparison is a consistency/validation check, not
fully independent validation, because radiosonde observations may contribute
to reanalysis assimilation.

Level 1 maintains two complementary atmospheric figures:

- `SESSION_L1_AtmosphericProfile.webp` compares the source actually used in
  Level 1 with the radiosonde reference, showing temperature difference,
  pressure difference and molecular-number-density impact. The plot states the
  source used explicitly (ERA5, ERA5 + USSA76 extension, or USSA76 fallback)
  without duplicating coincident source curves.
- `SESSION_L1_AtmosphericEvolution.webp` shows the hourly temperature anomaly
  through the complete session and the corresponding molecular-number-density
  change relative to the first hourly profile.

The radiosonde comparison remains consistency/QA evidence rather than a fully
independent validation because radiosonde observations may contribute to ERA5
assimilation.

## Level 2 temporal interpolation

For each Level 2 `block_time`:

1. temperature is linearly interpolated between neighboring
   `atmosphere_time` profiles;
2. pressure is interpolated in `log(P)`;
3. molecular backscatter/extinction are calculated for that exact block time;
4. the molecular state is aggregated onto the progressive Level 2 grid.

Therefore the Level 2 product stores block-resolved molecular state:

- `molecular_backscatter(block_time, wavelength, altitude)`
- `molecular_extinction(block_time, wavelength, altitude)`

Level 2 performs no ERA5, radiosonde or Open-Meteo network IO.

## Provenance

Level 1 records the atmosphere cadence, sources present, ERA5 DOI when used,
per-hour source type/time offset, USSA76 fallback fraction and resolved station
coordinates.

If the complete session falls back to USSA76, provenance identifies USSA76
rather than incorrectly presenting the product as ERA5-derived.

Existing Level 1/Level 2 files from the former single-profile atmosphere
contract must be regenerated for the current schema.
