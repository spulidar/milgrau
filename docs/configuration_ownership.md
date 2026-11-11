# Configuring MILGRAU for a scientific observation

This guide explains the three principal YAML files from the **user's perspective**: what information belongs in each file, which settings affect processing, and how to preserve reproducibility.

For an initial run, see [Getting started](getting_started.md). This document is not an exhaustive list of every allowed parameter; use the distributed YAML files and the per-level scientific guides when changing a numerical setting.

## What to configure

| File | User-facing role | When to change |
| --- | --- | --- |
| `config.yaml` | Data directories, processing choices, QA criteria, atmosphere preference and inversion/figure settings | When conducting a new processing run or sensitivity experiment |
| `station.yaml` | Station coordinates, historical instrument configurations, channel identity/calibration and station climatology | When verified instrument/observational evidence changes |
| `scc.yaml` | Optional external SCC configuration/channel/product identifiers and ancillary policies | When preparing SCC-compatible input for an authorized SCC account |

The standard installation keeps SCC **disabled** unless `scc.enabled: true` is deliberately set in `config.yaml`.

## 1. Data locations and processing control

Under `directories`, identify the original raw Licel tree and the location where products will be written. Output is organized by station, UTC year/month of the session start and canonical continuous session ID.

`processing.incremental: true` allows reuse of a current validated product. If an experimental run changes atmosphere source, instrument treatment or inversion assumptions, preserve prior results and use explicit reprocessing as appropriate.

An operational success status does not replace scientific QA. Warnings and stage diagnostics should be examined before considering results suitable for research.

## 2. Level 0 settings

Under `level0`, relevant controls include:

- `session.max_gap_seconds`: how long an acquisition interruption may be before a new continuous session begins;
- `acquisition_qa`: tolerances for shot-count and header-time consistency;
- `dark_current.max_association_hours`: allowable time separation when associating dark-current records;
- `solar_regime.day_night_threshold_deg`: geometric solar-elevation classification boundary;
- `surface_weather.missing_policy`: treatment of absent surface-weather data.

These are **processing policies**, not measurements of the real instrument. Distinguish them from dated calibration and channel facts supplied in `station.yaml`.

## 3. Level 1 settings and atmosphere

`level1` controls background estimation, photon-counting correction guards, PBL diagnostics, handling of missing historical channel calibration and atmospheric materialization.

The current normal atmospheric preference in the recipe is:

~~~yaml
level1:
  atmosphere:
    time_resolution_minutes: 60
    source_priority: ["era5", "radiosonde", "ussa76"]
    external_profile_outside_coverage: "ussa76"
~~~

For radiosondes, the normal configured selection window is six hours; the explicit `--radiosonde-only` and high-CDS-queue fallback paths use a 12-hour observed selection window. Wyoming is sought before the locally supplied NOAA IGRA2 archive.

ERA5 uses a local cache and optionally CDS network retrieval. Its credentials must be provided through the supported `cdsapi` configuration **outside** the repository. The public CDS queue threshold is a command-line operational option, not a permanent statement of data quality.

The actual source by atmosphere hour must be read from the generated Level 1 NetCDF. For detailed rules and comparisons, consult [Atmospheric profiles](atmospheric_profiles.md).

## 4. Level 2 settings

LEBEAR's scientific recipe is under `inversion` in `config.yaml`. Particularly relevant choices are:

- `wavelengths_to_process`: elastic wavelengths requested;
- `block_average_minutes`: temporal averaging;
- `molecular_fit`: Rayleigh reference QA and fitted interval settings;
- `retrieval.reference_search_ranges_m`: ordered molecular-reference search ranges;
- `retrieval.progressive_grid_schedule`: altitude-dependent vertical resolution;
- `gluing`: conditions for joining analog and photon-counting channels;
- `monte_carlo_iterations` and related settings: stochastic propagation and reproducibility;
- `aerosol_ref_fraction`, `retrieval.residual_aerosol_fractions` and related quantities: explicit inversion boundary assumptions.

Changing reference ranges, lidar ratio, input selection, grid or uncertainty settings may change the derived aerosol optical products and therefore requires a documented scientific reason and baseline comparison.

The SPU monthly **aerosol lidar-ratio** table resides in `station.yaml`. Its original empirical derivation remains unresolved; do not describe these values as a fully characterized observed climatology. See [Lidar-ratio provenance](lidar_ratio_climatology_provenance.md).

## 5. Instrument and station history

`station.yaml` describes the station and dated configurations used to interpret measured channels. It includes applicable calibration identities and parameters such as detector dead time and bin shifts where known.

Historical records may have missing or provisional values. A configured neutral fallback with a warning enables processing, but must not be represented as an experimentally determined calibration.

The configured launch station for radiosonde observations may be **different from the lidar station**. A sounding's station identity and height reference must be understood when interpreting molecular profiles.

## 6. SCC is optional

Only when `scc.enabled: true` does MILGRAU use the separate `scc.yaml` catalog to prepare SCC-specific inputs. This catalog identifies SCC configurations, product/channel mappings and observed-sounding ancillary requirements. It does **not** replace the station instrument record or the scientific Level 0/1/2 results.

Remote upload requires an explicit SCC operator command; changing the YAML to enabled does not authorize remote submission. See the [SCC operator guide](scc_batch_operator_guide.md).

## 7. Scientific figures

`visualization` controls which plots are produced and their display choices, including limits, channels, logo images and figure formats. Visualization downsampling is for display and does not alter scientific NetCDF resolution.

A changed color scale or plotted altitude span is not a change in the physics or retrieval.

## Reproducible processing record

For every analyzed session, retain:

1. Source Licel observations and canonical session ID.
2. Relevant `station.yaml` profile/calibration identity.
3. The `config.yaml` values used for Level 0/1/2.
4. ERA5/radiosonde/USSA76 source and observed launch context.
5. Generated NetCDF and its upstream provenance.
6. Scientific QA decisions and the reason for any changed processing settings.

MILGRAU embeds important recipe and source metadata in scientific NetCDF products; this should be inspected, not reconstructed from memory after a run.

See [Scientific quality](scientific_quality.md), [LIBIDS](libids.md), [LIPANCORA](lipancora.md) and [LEBEAR](lebear.md).
