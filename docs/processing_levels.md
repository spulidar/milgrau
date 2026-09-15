# MILGRAU processing levels

This document describes the stable responsibility boundary between MILGRAU processing levels. It intentionally avoids copying mutable numerical thresholds from `config.yaml` or instrument values from `station.yaml`; those files remain the authoritative runtime recipe and observational record.

A successful command means the software stage completed according to its contract. It does **not** by itself certify every scientific diagnostic. Scientific acceptance remains represented by the product variables, flags, provenance and validation tests.

## Pipeline overview

| Level / tool | Primary input | Main responsibility | Primary product |
| --- | --- | --- | --- |
| Level 0 — `milgrau-libids` | raw Licel acquisition and associated acquisition context | standardize the acquisition record and preserve raw/instrument traceability | Level 0 NetCDF |
| Level 1 — `milgrau-lipancora` | Level 0 NetCDF | apply channel corrections, propagate signal uncertainty and materialize ancillary atmospheric state | Level 1 corrected-signal/RCS NetCDF |
| Visualization — `milgrau-liracos` | Level 1 NetCDF | generate diagnostic quicklooks without changing scientific retrieval decisions | figures / QA views |
| Level 2 — `milgrau-lebear` | Level 1 NetCDF | select/glue elastic signals, build molecular profiles, perform Rayleigh-reference QA and productive backward elastic retrieval | Level 2 optical NetCDF |
| Explorer — `milgrau-explorer` | processed products | interactive inspection only | interactive UI |

## Level 0 — LIBIDS

### Responsibility

Level 0 turns raw Licel files into a standardized, traceable acquisition dataset. It owns parsing/acquisition-level consistency and the mapping from the historical instrument record to standardized channels.

### Inputs

- raw Licel files selected by measurement ID/discovery rules;
- station/instrument history resolved from `station.yaml`;
- processing policy from `config.yaml`;
- dark-current/acquisition context where applicable;
- optional surface-weather context according to the configured missing-data policy.

### Outputs

The Level 0 product preserves the standardized raw acquisition and the metadata required by downstream Level 1 corrections. It may also expose SCC-oriented metadata/mapping where configured, but SCC compatibility does not define MILGRAU scientific truth.

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
- atmospheric source data according to configured priority: radiosonde, ERA5 and/or explicit US Standard Atmosphere 1976 fallback.

### Outputs

The productive Level 1 contract includes, as applicable:

- corrected signal;
- corrected-signal one-sigma uncertainty;
- range-corrected signal and propagated uncertainty;
- photon-counting saturation diagnostics/status;
- correction-success status per channel;
- PBL/tropopause diagnostics;
- `Atmospheric_Temperature_K` and `Atmospheric_Pressure_hPa` fully materialized on the lidar altitude grid;
- readable atmosphere-source metadata and fallback fraction.

### Thermodynamic boundary

Level 2 does not rediscover meteorology. It consumes the canonical Level 1 pressure/temperature profiles and their provenance. Source cache filenames are operational details, not scientific source identities.

### Photon-counting boundary

The current correction path and provisional guard are not equivalent to a characterized detector saturation limit. Physical saturation and dark-current/dead-time ordering remain P4 instrument-evidence items.

## Level 2 — LEBEAR

### Productive method identity

The current productive Level 2 method is backward elastic Klett–Fernald–Sasano retrieval. Research kernels may support other integration directions, but the published productive contract remains explicitly backward.

The current schema and retrieval method versions are separate identities. See `docs/level2_schema.md` for the exact current version and product metadata.

### Inputs

- validated Level 1 corrected signals/RCS and their uncertainties;
- complete Level 1 pressure/temperature profiles;
- explicit Level 2 processing recipe from `config.yaml`;
- station-derived assumptions such as the resolved lidar-ratio climatology where applicable.

### Productive sequence

For each configured elastic wavelength, Level 2 currently performs:

1. channel discovery and temporal block reduction;
2. analog/photon-counting gluing or explicitly allowed single-channel fallback;
3. retrieval-input QA;
4. molecular Rayleigh calculation from the Level 1 thermodynamic profile;
5. Rayleigh reference search/calibration and explicit QA;
6. backward KFS Monte Carlo retrieval;
7. block acceptance and aggregate product construction;
8. complete/partial/failed multispectral product accounting;
9. FAIR provenance and optional QA visualization.

The canonical ownership map for these steps is maintained in `docs/code_inventory.md`.

### Support semantics

- Missing scientific support remains `NaN`; it is not filled to extend vertical coverage.
- Signal means and reported signal uncertainties use common sample support.
- Missing signal uncertainty is not interpreted as zero uncertainty.
- A finite scattering ratio is a diagnostic and does not by itself establish aerosol-retrieval support.
- Backward retrieval support does not extend above its accepted physical reference boundary.
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
