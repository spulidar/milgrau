# MILGRAU processing levels

This document describes the stable responsibility boundary between MILGRAU processing levels. It intentionally avoids copying mutable numerical thresholds from `config.yaml` or instrument values from `station.yaml`; those files remain the authoritative runtime recipe and observational record.

A successful command means the software stage completed according to its contract. It does **not** by itself certify every scientific diagnostic. Scientific acceptance remains represented by the product variables, flags, provenance and validation tests.

## Pipeline overview

| Level / tool | Primary input | Main responsibility | Primary product |
| --- | --- | --- | --- |
| Level 0 — `milgrau-libids` | raw Licel acquisition and associated acquisition context | standardize the acquisition record and preserve raw/instrument traceability | Level 0 NetCDF |
| Level 1 — `milgrau-lipancora` | Level 0 NetCDF | apply channel corrections, propagate signal uncertainty and materialize ancillary atmospheric state | Level 1 corrected-signal/RCS NetCDF |
| Visualization — `milgrau-liracos` | Level 1 NetCDF | generate diagnostic quicklooks without changing scientific retrieval decisions | figures / QA views |
| Level 2 — `milgrau-lebear` | Level 1 NetCDF | select/glue elastic signals, build molecular profiles, perform Rayleigh-reference QA and productive support-aware two-sided elastic retrieval | Level 2 optical NetCDF |
| Explorer — `milgrau-explorer` | processed products | interactive inspection only | interactive UI |

## Level 0 — LIBIDS

### Responsibility

Level 0 turns one continuous raw Licel session into a standardized, traceable acquisition dataset. It owns parsing/acquisition-level consistency and the mapping from the historical instrument record to standardized channels.

### Inputs

- raw Licel files grouped into continuous sessions by acquisition continuity and station context;
- station/instrument history resolved from `station.yaml`;
- processing policy from `config.yaml`;
- dark-current/acquisition context where applicable;
- hourly surface-weather context on its own `weather_time` axis according to
  the configured missing-data policy.

### Outputs

The Level 0 product preserves the standardized raw acquisition and the metadata required by downstream Level 1 corrections. It may also expose SCC-oriented metadata/mapping where configured, but SCC compatibility does not define MILGRAU scientific truth.

### Session identity

Level 0 session identity is independent of civil-day publication windows. A session
may cross midnight and former 00/06/12/18 boundaries. A new session begins after
a configured acquisition gap or when the temporally valid station profile/calibration
changes. The canonical identifier is
`station_YYYYMMDD-HHMMZ_YYYYMMDD-HHMMZ`, using the raw acquisition interval
in UTC; QA does not rename the underlying acquisition.

### Boundary

Level 0 does not perform aerosol optical retrieval. Instrument facts belong in the station catalog rather than being recreated as hidden constants in parser code.

## Level 1 — LIPANCORA

### Responsibility

Level 1 converts standardized acquisition channels into corrected lidar signals with propagated uncertainty and a complete thermodynamic atmosphere on the lidar altitude grid.

The canonical correction owner is `milgrau.level1.corrections`; thermodynamic source selection/materialization is owned by `milgrau.level1.thermodynamics`.

### Inputs

- valid Level 0 NetCDF;
- station-resolved calibration/instrument parameters;
- Level 1 processing recipe;
- hourly ERA5 pressure-level atmosphere plus explicit USSA76 fallback;
- optional radiosonde sounding for observational atmosphere QA.

### Outputs

The productive Level 1 contract includes, as applicable:

- corrected signal;
- corrected-signal one-sigma uncertainty;
- range-corrected signal and propagated uncertainty;
- pre-background signal and uncertainty for auditable reconstruction;
- per-profile median/MAD background estimate, standard error, robust scale,
  valid-bin count and outlier fraction;
- photon-counting saturation diagnostics/status;
- correction-success status per channel;
- PBL/tropopause diagnostics;
- `Atmospheric_Temperature_K(atmosphere_time, altitude)` and
  `Atmospheric_Pressure_hPa(atmosphere_time, altitude)` fully materialized;
- per-hour atmosphere source/fallback metadata;
- optional mapped radiosonde QA reference.

### Thermodynamic boundary

Level 2 does not rediscover meteorology. It interpolates the canonical
time-resolved Level 1 temperature/pressure state to each retrieval block time.
Source cache filenames are operational details, not scientific source
identities.

### Photon-counting boundary

The current correction path and provisional guard are not equivalent to a characterized detector saturation limit. Physical saturation and dark-current/dead-time ordering remain P4 instrument-evidence items.

## Level 2 — LEBEAR

### Retrieval identity

The Level 2 method is a two-sided elastic Klett–Fernald–Sasano retrieval. The backward branch below the selected reference and the forward branch above it both terminate at the first unsupported input or numerical failure. The product exposes branch validity, endpoints and altitude-resolved Monte Carlo support and never promises a fixed top altitude.

The product records the installed software version and a source-code hash. The technical NetCDF schema identifier exists only to validate the file contract; no separate retrieval-method version is maintained.

### Inputs

- validated Level 1 corrected signals/RCS and their uncertainties;
- complete Level 1 pressure/temperature profiles;
- explicit Level 2 processing recipe from `config.yaml`;
- station-derived assumptions such as the resolved lidar-ratio climatology where applicable.

### Processing sequence

For each configured elastic wavelength, Level 2 currently performs:

1. channel discovery and temporal block reduction;
2. analog/photon-counting gluing or explicitly allowed single-channel fallback;
3. retrieval-input QA;
4. block-resolved molecular Rayleigh calculation from the time-interpolated
   Level 1 thermodynamic atmosphere;
5. robust weighted residual-background fit over the broad Rayleigh search span,
   followed by local Rayleigh reference calibration and explicit QA;
6. two-sided KFS Monte Carlo retrieval with separate backward/forward support diagnostics;
7. block acceptance and aggregate product construction;
8. complete/partial/failed multispectral product accounting;
9. FAIR provenance and optional QA visualization.

The canonical ownership map for these steps is maintained in `docs/code_inventory.md`.

### Support semantics

- Missing scientific support remains `NaN`; it is not filled to extend vertical coverage.
- Signal means and reported signal uncertainties use common sample support.
- Missing signal uncertainty is not interpreted as zero uncertainty.
- A finite scattering ratio is a diagnostic and does not by itself establish aerosol-retrieval support.
- Backward and forward support are contiguous and branch-specific; finite diagnostic values do not bridge an unsupported cell.
- The current aggregate optical uncertainty uses a conservative correlation policy for mixed block-level nuisance terms; see `docs/level2_schema.md`.
- Elastic extinction is conditional on the assumed aerosol lidar ratio.

### Completeness semantics

A wavelength belongs to the scientific Level 2 wavelength coordinate only when it produced a usable scientific result. Requested, processed and failed wavelengths are stored separately with stable failure stage/code information. Partial products may be written for diagnosis but are not reused incrementally as complete products.

## Visualization and explorer

Visualization modules describe products; they do not choose productive retrieval science. A plotting or Streamlit change must not alter source selection, Rayleigh QA, KFS acceptance or uncertainty semantics.

## Provenance across levels

Published NetCDF provenance is intended to answer four distinct questions without machine-local paths:

1. **Which software state ran?** Package version plus installed source-content identity and optional repository/build revision.
2. **Which recipe ran?** Exact processing/station YAML snapshots plus stable station/calibration IDs.
3. **Which upstream scientific input was used?** Readable filenames/source metadata; Level 2 additionally records exact Level 1 SHA-256 content identity.
4. **Which scientific method produced the values?** Explicit schema/retrieval-method identities and method-specific metadata.

See `docs/level2_schema.md` and `docs/scientific_traceability.md` for the detailed Level 2 contract.

## Validation hierarchy

Synthetic/analytical tests provide known truth for equations and controlled support behavior. Real observations are regression and behavior evidence. External chains such as LPP/SCC/ELDA provide methodological/interoperability comparison, not automatic ground truth for MILGRAU numerical output.

The roadmap and acceptance gates are maintained in `tracker.md`.
