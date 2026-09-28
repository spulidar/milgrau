# Configuration and scientific ownership

MILGRAU separates the run recipe, station/instrument reality and implemented
physics. A value has one authoritative owner so the provenance of a result is
unambiguous.

| Owner | Responsibility |
| --- | --- |
| `config.yaml` | processing choices for one run |
| `station.yaml` | dated station, instrument and calibration facts |
| Python | equations, constants, validation and implementation mechanics |
| external source | radiosonde/ERA5 content |
| NetCDF provenance | resolved record of the run, inputs and source code |

## `config.yaml`

The processing recipe owns:

- incremental/overwrite policy and directory roles;
- Level 0 acquisition QA;
- Level 1 correction and atmosphere-source priority;
- Level 2 wavelengths and temporal block duration;
- gluing search and QA settings;
- molecular-reference QA window and thresholds;
- ordered molecular-reference search ranges;
- progressive vertical-grid schedule;
- two-sided KFS and Monte Carlo settings;
- residual-aerosol boundary scenarios;
- cloud-screening enablement;
- visualization and QA settings.

Scientific readers are strict: a missing required setting fails rather than
creating a silent default.

## `station.yaml`

The station catalogue owns:

- station identity, coordinates, altitude and timezone;
- dated hardware/acquisition profiles;
- laser and channel definitions;
- dead time, bin shift and characterized saturation properties;
- calibration identity and SCC channel mapping;
- station lidar-ratio climatology and its declared uncertainty;
- overlap geometry and the status of provisional estimates.

The dated station profile is resolved before processing. Downstream code uses
that resolved state and does not independently rediscover instrument history.

## Python

Code owns:

- Rayleigh/molecular atmosphere equations;
- signal and uncertainty propagation;
- robust background and molecular-scaling estimators;
- progressive aggregation and missing-data semantics;
- reference admissibility and deterministic selection mechanics;
- Klett–Fernald–Sasano integration;
- selection-aware Monte Carlo;
- schemas, flags, validation and provenance algorithms.

Mutable station facts do not become Python constants. Equations and universal
physical constants do not become editable station configuration.

## Current Level 2 decision locations

| Decision | Owner |
| --- | --- |
| 20-minute blocks | `config.yaml: inversion.block_average_minutes` |
| two-sided integration | `config.yaml: inversion.kfs_mode` |
| 10–15 km then 5–20 km reference ranges | `config.yaml: inversion.retrieval.reference_search_ranges_m` |
| 1 km Rayleigh window | `config.yaml: inversion.molecular_fit.ref_window_m` |
| progressive-grid schedule | `config.yaml: inversion.retrieval.progressive_grid_schedule` |
| 150 Monte Carlo realizations | `config.yaml: inversion.monte_carlo_iterations` |
| nominal `f=0` | `config.yaml: inversion.retrieval.residual_aerosol_fractions` |
| lidar-ratio monthly values/dispersion | resolved station catalogue |
| reference-selection equations and tie break | Python |
| background fit and Monte Carlo refit | Python |

## Secrets and paths

Credentials never belong in YAML committed to the repository or in scientific
NetCDF provenance. Provider-supported credential mechanisms are used instead.

Published provenance stores portable configuration snapshots, source hashes
and software/source-code identity. Host-specific absolute paths are operational
details, not scientific identity.

## Visualization boundary

QA modules may smooth or rescale data only for display. They cannot change the
saved retrieval, invent support, select a different reference or silently apply
scientific thresholds.
