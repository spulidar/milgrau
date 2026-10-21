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

## Optional productive radiosonde-only processing (main branch)

Run `milgrau-lipancora --radiosonde-only -i SESSION_ID` to process
a selected L0 from radiosonde observations without requesting ERA5 or
waiting for CDS. Without `-i`, the same switch applies to every eligible L0
in the Level 1 queue. This CLI override is ephemeral: `config.yaml` and the
normal ERA5/USSA76 source priority remain unchanged.

**Offline archive before Siphon:** place the unmodified NOAA/NCEI IGRA2
historical data file for SPU/WMO station 83779 at:

```text
.cache/radiosonde/BRM00083779-data.txt
```

MILGRAU first checks the preexisting per-hour
`.cache/radiosonde/YYYY/MM/radiosonde_83779_YYYYMMDD_HHZ.csv`.
Otherwise, it locates the **exact station and nominal UTC sounding time** in
the IGRA2 archive. The historical file is indexed once **per process**, in
memory, using byte offsets, so repeated L1 hours do not reread the full
multi-year text file. The IGRA2 PRESS (Pa), GPH (m ASL) and TEMP (0.1 °C)
fields are converted to `pressure` (hPa), `height` (m) and
`temperature` (°C). Missing `-9999` and quality-removed `-8888`
measurements are excluded. Only a usable vertical profile is accepted.
If no match or valid profile exists, MILGRAU **falls back to
University of Wyoming via Siphon** using the unchanged selection policy.
Valid IGRA2 observations are cached as hourly CSV plus JSON metadata for
faster subsequent runs. The local archive itself is not rewritten or
committed to Git (the repository excludes `.cache/`).

A native IGRA2 archive is **not a Siphon/Wyoming download**. Scientific
provenance records whether the radiosonde hours came from NOAA/NCEI IGRA2,
Wyoming/Siphon, or both, and retains the archive basename when applicable.
The nominal UTC hour is used to match the configured synoptic hour; the
actual release time in IGRA2 may differ from that nominal hour.

A sounding is selected **independently for each hourly L1 atmosphere time**.
This explicit provisional `--radiosonde-only` mode expands the temporal
window to **12 hours** (whereas the ordinary ERA5/QA processing retains its
configured **six-hour** selection). Cached observations at nominal UTC 00Z
and 12Z are searched in order of their **actual age**, not just by closest
nominal synoptic time. This matters for historical station 83779: dates with
only a **12Z** archived sounding can use that same real profile at **19Z**
(age **7 hours**) instead of repeatedly requesting a nonexistent **00Z**
launch. The actual selected timestamp and age are recorded **per hourly
profile**. The program does **not** synthesize a missing 00Z sounding,
pretend that 12Z is equally representative of the evening, or silently
extend the window beyond 12 hours. If no valid local observation exists
within the window, Wyoming/Siphon is tried; if that also fails, the session
**fails** instead of generating a full-USSA76 substitute.
Outside the observed vertical range of a valid sounding, the established
`external_profile_outside_coverage: ussa76` setting extends the profile
vertically, with the fallback fraction recorded.

**Output naming:** the radiosonde-only run deliberately writes the **normal**
`SESSION_ID_L1.nc` and may **replace an existing ERA5 L1**. The NetCDF
records `level1_atmosphere_mode=radiosonde_only`,
`thermodynamic_source_priority=radiosonde`, and
`Atmospheric_Source_Type=radiosonde`. This is an explicitly
**non-productive-for-L2, provisional L1** in the current workflow. The
LEBEAR L2 pipeline rejects such input even if a pre-existing L2 would
otherwise qualify for incremental reuse; its error instructs the operator
to rerun ordinary `milgrau-lipancora -i SESSION_ID` without
`--radiosonde-only`. Incremental L1 selection recognizes mode changes
so a normal run reprocesses radio-only L1 instead of skipping it as current.
A normal run without the flag restores ERA5/USSA76 processing, provided
the required ERA5 data can be obtained. Check its atmospheric source
provenance, because the normal branch also allows an explicit USSA76
fallback during CDS outages.

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
