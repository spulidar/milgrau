# Molecular atmosphere sources

MILGRAU resolves the thermodynamic profile used by the molecular/Rayleigh calculation in this order:

1. **Radiosonde** — preferred when a valid Wyoming Upper Air sounding is available.
2. **ERA5 pressure levels** — optional IO fallback when `era5.enabled: true`.
3. **US Standard Atmosphere 1976** — deterministic, network-free final fallback implemented with the standard stratified layers through 84.852 km geopotential altitude.

The source hierarchy is scientific provenance, not only an IO detail: changing pressure or temperature changes molecular number density, molecular backscatter/extinction, Rayleigh calibration, and therefore the elastic inversion.

## Processing contract

Thermodynamic source selection belongs to **Level 1**. Every successfully written Level 1 product contains a complete atmosphere already mapped to the Level 1 lidar `altitude` coordinate:

- `Atmospheric_Temperature_K(altitude)`
- `Atmospheric_Pressure_hPa(altitude)`

Level 2 reads these two variables directly. It performs no radiosonde/ERA5 IO, no source selection, no vertical interpolation and no hidden standard-atmosphere fallback. A Level 1 file that does not contain the canonical atmosphere is intentionally rejected and must be reprocessed.

For radiosonde and ERA5 profiles, source heights are interpreted as geometric altitude above mean sea level (ASL), the station altitude is used to align them with the lidar AGL grid, and interpolation is performed in Level 1. Temperature is interpolated linearly with altitude; pressure is interpolated in `log(P)` because its vertical behavior is approximately exponential. Bins outside the vertical coverage of the external profile are filled with USSA76. The fraction of bins filled this way is recorded as `thermodynamic_profile_standard_fallback_fraction`.

When neither external source is usable, USSA76 is evaluated directly on the full lidar grid and the fallback fraction is `1.0`.

## ERA5 configuration

```yaml
era5:
  enabled: false
  cache_dir: ".cache/milgrau/era5"
  dataset: "reanalysis-era5-pressure-levels"
  grid_deg: 0.25
  area_half_width_deg: 0.25
```

ERA5 support is optional:

```bash
pip install -e ".[era5]"
```

The CDS API credentials are intentionally **not** stored in `config.yaml`. Configure `cdsapi` using the user's `~/.cdsapirc` file according to the Copernicus Climate Data Store instructions.

MILGRAU requests only the pressure-level variables required by the molecular atmosphere: **temperature** and **geopotential**. The downloaded NetCDF is an IO cache only; a JSON sidecar records source, dataset, ERA5 DOI, analysis time, measurement-time offset, requested coordinates and download time. The scientific processing product remains the Level 1 NetCDF.

## Level 1 provenance

Global attributes include:

- `thermodynamic_profile_available`
- `thermodynamic_profile_source_type`
- `thermodynamic_profile_source`
- `thermodynamic_profile_datetime_utc`
- `thermodynamic_profile_time_delta_hours`
- `thermodynamic_profile_station_id`
- `thermodynamic_profile_doi`
- `thermodynamic_profile_standard_fallback_fraction`
- `thermodynamic_profile_grid`
- `thermodynamic_profile_altitude_reference`

For external sources, the original source-profile ASL coverage is also recorded with:

- `thermodynamic_source_profile_min_altitude_asl_m`
- `thermodynamic_source_profile_max_altitude_asl_m`

There are no `Radiosonde_*` compatibility aliases in the Level 1 product. Source identity belongs in provenance, not in the variable name.

## Reprocessing note

The former fallback clipped a tropospheric lapse-rate profile at 216.65 K while continuing the same pressure law aloft. The current fallback uses the stratified USSA76 layers. In addition, the canonical Level 1 atmosphere contract is now mandatory. Existing Level 1 products created before this contract must be regenerated before current Level 2 processing.
