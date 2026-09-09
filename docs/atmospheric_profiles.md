# Molecular atmosphere sources

MILGRAU resolves the thermodynamic profile used by the molecular/Rayleigh calculation in this order:

1. **Radiosonde** — preferred when a valid Wyoming Upper Air sounding is available.
2. **ERA5 pressure levels** — optional IO fallback when `era5.enabled: true`.
3. **US Standard Atmosphere 1976** — deterministic, network-free final fallback implemented with the standard stratified layers through 84.852 km geopotential altitude.

The source hierarchy is scientific provenance, not only an IO detail: changing pressure or temperature changes molecular number density, molecular backscatter/extinction, Rayleigh calibration, and therefore the elastic inversion.

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

MILGRAU requests only the pressure-level variables required by the molecular atmosphere: **temperature** and **geopotential**. The downloaded NetCDF is cached locally and a JSON sidecar records source, dataset, ERA5 DOI, analysis time, measurement-time offset, requested coordinates and download time.

## Level 1 provenance

Level 1 stores canonical external-profile variables when a radiosonde or ERA5 profile is available:

- `atmospheric_altitude`
- `Atmospheric_Temperature_K`
- `Atmospheric_Pressure_hPa`

and global attributes including:

- `thermodynamic_profile_source_type`
- `thermodynamic_profile_source`
- `thermodynamic_profile_datetime_utc`
- `thermodynamic_profile_time_delta_hours`
- `thermodynamic_profile_station_id`
- `thermodynamic_profile_doi`

The legacy `Radiosonde_*` variables are temporarily retained as compatibility aliases for the current Level 2 reader. Their metadata explicitly identifies the real source; ERA5 data must not be interpreted as a physical radiosonde observation.

If neither external source is available, Level 1 records `thermodynamic_profile_source_type = "ussa76"` and Level 2 evaluates USSA76 directly on the lidar altitude grid.

## Reprocessing note

The former fallback clipped a tropospheric lapse-rate profile at 216.65 K while continuing the same pressure law aloft. The new fallback uses the stratified USSA76 layers. Level 2 products that relied on the old standard-atmosphere fallback are therefore not scientifically identical to products generated after this change and should be reprocessed when strict comparability is required.
