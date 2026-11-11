# Time-resolved molecular atmosphere

MILGRAU materializes the thermodynamic atmosphere in **Level 1** so Level 2
never fetches meteorological data or silently chooses a new source.

## Productive atmosphere policy

For continuous sessions the Level 1 atmosphere used by MILGRAU is time resolved:

- **ERA5 pressure levels** are the hourly temporal backbone when available.
- **US Standard Atmosphere 1976 (USSA76)** is the explicit vertical extension
  outside external-profile coverage and the full fallback for an hour when ERA5
  is unavailable.
- **Radiosonde** is retained as a local observational comparison/QA reference
  in standard processing; it does not replace isolated ERA5 hours in that
  workflow. The explicit `milgrau-lipancora --radiosonde-only` option instead
  makes observational profiles the productive L1 atmosphere for that run.

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

If ERA5 is unavailable for a required hour, an observed Wyoming sounding is
preferred, followed by a local NOAA IGRA2 observation when available. When
neither is available and `ussa76` is configured, the complete hourly profile
is materialized from USSA76.
No synthetic surface temperature/pressure constants are invented.

## ERA5 IO and cache

ERA5 requests use the configured CDS pressure-level dataset and request only the
molecular-atmosphere fields required by MILGRAU: temperature and geopotential.

The downloaded NetCDF and JSON sidecar are IO cache artifacts. The Level 1
NetCDF is the scientific product. Missing hours are requested from CDS in
UTC-day batches, then split back into hour-indexed cache files. Adjacent
sessions therefore reuse the same downloaded ERA5 hours without repeating
network requests.

`cdsapi` is a core MILGRAU dependency because ERA5 is the productive
time-resolved atmosphere backbone. CDS credentials remain outside the repository
and are read by `cdsapi`.

## Operational source preference, CDS queue and radiosonde-only mode

The normal productive Level 1 preference is **ERA5 → Wyoming → NOAA IGRA2
archive → USSA76**. This order is per hourly atmospheric profile. A missing
external source is never represented as an actual observation. ERA5 cached hours
are preferred even if the CDS global queue is congested.

### Public CDS queue: best-effort signal

Before normal CLI processing, MILGRAU tries one short read of the public CDS
dashboard at https://cds.climate.copernicus.eu/live. It extracts the global
"Queued requests" metric, which is **not** an official machine-readable queue
API and does **not** indicate an individual job's wait time.

The heuristic threshold defaults to **8,000 queued requests** and can be
adjusted with `--cds-queue-threshold N`; this is an operational starting value,
not a scientifically validated queue-performance cutoff. Above that count,
MILGRAU skips **new** CDS requests while retaining access to validated cached
ERA5 hours. Missing hours then use Wyoming, IGRA2 or USSA76. The high-queue
path permits observed soundings up to 12 hours from their nominal UTC launch.
The original six-hour window remains in the ordinary low-queue setting.

The public dashboard may change or become unavailable. If its metric cannot be
read, the log records **unknown queue**, and the normal ERA5 path is retained;
unknown is never reported as a low queue. Disable this experimental observation
with `--no-cds-queue-check`. Existing submitted/uncertain CDS requests are not
cancelled, retried or silently resubmitted because of this check. The queue
decision is captured in L1 provenance.

### Explicit no-CDS mode

`milgrau-lipancora --radiosonde-only -i SESSION_ID` always disables ERA5/CDS
and uses **Wyoming → IGRA2 → USSA76** for the hourly atmosphere. The 12-hour
observational selection tolerance is explicit. If neither observational source
is usable, USSA76 is a declared model fallback, not an observed sounding.
This flag does not change the persisted configuration. LEBEAR accepts the
resulting L1 and propagates atmospheric source provenance to L2.

Historical SPU IGRA2 source data may be installed at:

```text
.cache/radiosonde/BRM00083779-data.txt
```

This operator-supplied archive is never downloaded automatically or rewritten.
The NOAA fixed-width profiles are quality filtered and may be cached as hourly
CSV and JSON sidecars with their genuine observation source. A successful
Wyoming retrieval never overwrites a previous IGRA2 cache: when both exist
for the same nominal launch, a separate `*_wyoming.csv` is retained.

The observations are searched within the explicit time tolerance; absent
00Z profiles are not fabricated. A valid 12Z observation can be used for a
19Z profile under the 12-hour high-queue or no-CDS policy, with its seven-hour
offset recorded.

### Source identity and incremental processing

The Level 1 NetCDF stores `Atmospheric_Source_Type(atmosphere_time)` as
`era5`, `radiosonde` or `ussa76`. For radiosonde hours it additionally
stores `Atmospheric_Observation_Source(atmosphere_time)` as
`wyoming_siphon` or `igra2`, and stores the exact observational archive in
provenance where applicable. Queue status, selection reason and whether ERA5
network IO was permitted are retained as metadata. USSA76 vertical extension
fractions remain explicit and must not be confused with fully observed profiles.

A previous high-queue L1 may be upgraded on a later ordinary run when new ERA5
downloads are permitted. Explicit radio-only and normal processing keep the
canonical filename `SESSION_ID_L1.nc`, so preserve baseline files in separate
folders when conducting scientific comparisons. L2 products identify their
source L1 by exact file hash and carry the selected observation-origin metadata.

This operational fallback is not evidence that ERA5, radiosonde and USSA76
are scientifically interchangeable. Always inspect observation age, station,
vertical coverage and the adopted atmosphere before interpreting aerosol
retrievals.

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
