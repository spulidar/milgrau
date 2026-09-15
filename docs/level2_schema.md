# MILGRAU Level 2 product schema

This document describes the current versioned LEBEAR NetCDF contract on `new-architecture`. It is descriptive, not a second source of scientific constants: equations remain owned by the numerical modules, variable metadata by `milgrau.level2.metadata`, and schema assembly by `milgrau.level2.dataset`.

The validated productive numerical baseline remains backward Klett–Fernald–Sasano retrieval. Schema/metadata work in P3 does not authorize extrapolation, filling unsupported bins, changing reference QA, changing lidar-ratio assumptions, or enabling cloud/SNR gates.

## Product identities

The product intentionally separates three kinds of identity:

- package/software version: normal MILGRAU release identity;
- `level2_product_schema_version = "1"`: storage names/dimensions/metadata contract;
- `level2_retrieval_method_version = "2"`: productive L2 method identity independent of package CalVer and schema-only changes. Method v2 requires common value/uncertainty support during averaging and treats missing signal uncertainty as unsupported rather than as zero Monte Carlo noise.

Incremental reuse rejects a product whose schema version, retrieval-method version, productive KFS identity, Fernald scientific identity, or versioned gluing-selection score does not match the running code. Partial products are never incrementally reusable.

## Coordinates and time model

| Coordinate | Meaning | Units/semantics |
| --- | --- | --- |
| `time` | original Level 1 profile timestamps used for time-expanded diagnostics | datetime; productive inversion is not performed separately at every expanded timestamp |
| `block_time` | start/floored timestamp of each configured retrieval block | datetime |
| `wavelength` | successfully processed elastic wavelength | `nm` |
| `altitude` | altitude above station | `m`, positive upward |

Dedicated `requested_wavelength`, `processed_wavelength`, and `failed_wavelength` dimensions belong to the multispectral completeness contract, not to the scientific wavelength coordinate. `wavelength` equals `processed_wavelengths` exactly.

## Canonical aggregate/block naming

Schema v1 stores exactly one aggregate aerosol optical product per quantity:

- `aerosol_backscatter_mean(wavelength, altitude)`
- `aerosol_backscatter_mean_error(wavelength, altitude)`
- `aerosol_extinction_mean(wavelength, altitude)`
- `aerosol_extinction_mean_error(wavelength, altitude)`

The former unsuffixed NetCDF aliases `aerosol_backscatter`, `aerosol_backscatter_error`, `aerosol_extinction`, and `aerosol_extinction_error` were exact duplicate arrays and are not part of schema v1. Older/unversioned products are reprocessed instead of retaining permanent duplicate compatibility variables.

Block products remain explicit:

- `aerosol_backscatter_block(block_time, wavelength, altitude)`
- `aerosol_backscatter_error_block(block_time, wavelength, altitude)`
- `aerosol_extinction_block(block_time, wavelength, altitude)`
- `aerosol_extinction_error_block(block_time, wavelength, altitude)`

Internal Python field names in `OpticalProducts` are runtime implementation details and do not define NetCDF aliases.

## Units and variable metadata

`milgrau.level2.metadata` is the canonical variable/coordinate metadata registry. Dataset construction fails if the emitted data-variable inventory differs from that registry, so a newly added stored field cannot silently appear without metadata. Every stored data variable has a `long_name`.

Physical units are explicit where the quantity is physically calibrated:

| Family | Units |
| --- | --- |
| molecular backscatter | `m-1 sr-1` |
| molecular extinction | `m-1` |
| molecular two-way transmission | `1` |
| unscaled simulated molecular signal `beta*T/r²` | `m-3 sr-1` |
| unscaled simulated molecular RCS | `m-1 sr-1` |
| aerosol backscatter and one-sigma uncertainty | `m-1 sr-1` |
| aerosol extinction and one-sigma uncertainty | `m-1` |
| aerosol lidar ratio and its configured standard deviation | `sr` |
| scattering ratio, correlations, relative errors/fractions/SNR diagnostics | `1` |
| reference/gluing altitudes | `m` |

Selected lidar signals are deliberately **not** assigned invented SI units. Analog and photon-counting channels enter the selected/glued signal in source-dependent instrumental spaces. These fields therefore carry explicit `unit_status` metadata such as `source_dependent_channel_native_corrected` or `source_dependent_relative_range_squared` instead of a false absolute radiometric unit. Rayleigh calibration and gluing coefficients are documented the same way when their units depend on the selected source space.

## Missing values and support

NaN is never filled or interpolated merely to extend retrieval coverage.

For productive temporal/block averaging in method v2, a sample contributes to a reported signal mean only when both the signal and its one-sigma uncertainty are finite and the uncertainty is non-negative. The signal mean and propagated uncertainty therefore use one common mask and one common effective sample count. A finite signal with missing uncertainty is unsupported for that reduction; missing uncertainty is never interpreted as zero uncertainty.

For KFS Monte Carlo retrieval in method v2, non-finite `rcs_error` is preserved as unsupported. It is not replaced with zero before perturbation. A missing uncertainty sample on the requested backward integration support invalidates that productive branch, and invalid KFS blocks do not publish partial aerosol optical arrays as accepted block products.

For aerosol backscatter/extinction products, NaN means no accepted productive backward-retrieval support at that altitude. Internal unsupported gaps are not bridged. Aggregate products use only accepted retrieval blocks, and aggregate optical means/errors use the same finite value/uncertainty support at each altitude.

`scattering_ratio_mean` and `scattering_ratio_block` are measured-to-molecular diagnostics. They can remain finite above the productive backward KFS boundary; a finite scattering ratio is **not** evidence of supported aerosol retrieval.

`retrieval_success_flag(block_time, wavelength)` is a block-level acceptance flag. It is not altitude-resolved support. A zero can include a block that was not attempted or that was rejected at input, Rayleigh QA, or KFS; dedicated source/input/reference/KFS diagnostics provide the stage information.

The future variables `retrieval_support_flag(..., altitude)` and `retrieval_top_altitude_m` remain intentionally absent until their frozen semantics have dedicated synthetic tests.

## Flags

Numeric state fields use integer `flag_values` arrays and stable `flag_meanings`, rather than comma-separated textual value lists. Aggregate and block variants share the same mapping.

Important interpretations:

- `signal_source_flag`: invalid / glued / photon-counting / analog;
- `retrieval_input_invalid_reason`: stable reason enum; zero means valid input;
- `gluing_merge_source_flag`: photon-counting / blend / analog / invalid per altitude bin;
- `retrieval_success_flag`: not successful / successful at block level;
- `rayleigh_reference_success_flag[_block]`: reference QA not passed / passed; zero can include not attempted;
- `kfs_backward_valid_flag*` and `kfs_forward_valid_flag*`: branch diagnostic only; zero can mean unrequested or invalid, so `KFS_Mode` / `integration_mode` identifies the productive branch;
- `kfs_branch*`: location relative to the reference bin, not a replacement for future retrieval-support semantics;
- failed-wavelength stage/code: stable numeric failure identity independent of Python exception strings.

Human-readable failure messages/cause summaries remain supplementary; program logic uses stable numeric stage/code fields.

## Main variable families

The full exact variable-name registry lives in `milgrau.level2.metadata`. The storage model is:

- molecular fields: `(wavelength, altitude)` plus block-scaled molecular RCS;
- selected/glued signal: time-expanded `(time, wavelength, altitude)`, block `(block_time, wavelength, altitude)`, and aggregate `(wavelength, altitude)` forms where appropriate;
- optical products: aggregate `(wavelength, altitude)` and block `(block_time, wavelength, altitude)`;
- Rayleigh calibration/reference diagnostics: aggregate `(wavelength)` plus block `(block_time, wavelength)`;
- KFS branch diagnostics: aggregate `(wavelength, altitude)` plus block `(block_time, wavelength, altitude)` and branch-validity vectors;
- gluing/source/input diagnostics: time-expanded plus block forms;
- completeness/failure variables: requested/processed/failed wavelength dimensions.

Time-expanded gluing/source diagnostics repeat block decisions on the original Level 1 profile-time coordinate for traceability. They are not independent retrievals.

## Productive method provenance

The final NetCDF records stable machine-readable productive identity including:

- `level2_retrieval_method_version`;
- `level2_retrieval_method_change`;
- `elastic_backscatter_inversion_method`;
- `integration_mode = backward` and matching `KFS_Mode`;
- Fernald implementation/scientific-change identity;
- `uncertainty_method = Monte Carlo` and `uncertainty_scope = partial Monte Carlo dispersion; not a total uncertainty budget`;
- Monte Carlo iteration count and random seed;
- KFS reference-boundary model `beta_total_ref=beta_mol_ref*(1+aerosol_ref_fraction)`;
- aerosol reference fraction and its relative uncertainty;
- minimum lidar ratio / negative-aerosol policy;
- `lidar_ratio_assumed_sr`, `lidar_ratio_std_sr`, and readable `lidar_ratio_source`;
- versioned gluing-selection score formula and its four exact weights.

Gluing score v1 preserves the existing numerical rule exactly:

`relative_rmse + abs(relative_bias) + 0.001*intercept_percent + 0.01*saturation_fraction`

The weights are named constants in `milgrau.level2.gluing`; changing the formula/version makes existing products stale instead of silently reusing them.

## Completeness contract

Every publishable Level 2 product stores:

- `requested_wavelengths`;
- `processed_wavelengths`;
- `failed_wavelengths`;
- stable failure stage/code and human-readable message/cause for failed wavelengths;
- `product_completeness` and `product_status`.

A wavelength is in the scientific `wavelength` coordinate only when it produced a usable scientific result. A file may be written as partial for diagnosis, but partial products are deliberately not reused incrementally.

## Provenance boundary

The product currently favors readable, portable provenance over host-specific paths or opaque hashes:

- source Level 1 **filename**, not an absolute local path;
- stable station profile and instrument calibration IDs when available;
- processing/station configuration filenames;
- exact processing/station YAML snapshots stored in the product;
- thermodynamic source identity inherited/materialized from Level 1;
- software/method/schema identities described above.

Full local paths, secrets, transient cache paths, and operational tracebacks do not belong in the scientific product. Current provenance intentionally does not publish configuration/Git SHA attributes merely for appearance of reproducibility; exact YAML plus stable IDs are the readable configuration record. A Level 1 source-content identity is tracked separately in the roadmap because cache correctness and scientific lineage are now named consumers; it is not claimed as implemented until the provenance/currentness code and tests land.

## Known limitations

- Productive elastic inversion is backward KFS from one accepted Rayleigh reference toward lower altitude.
- Aerosol extinction is conditional on assumed aerosol lidar ratio.
- Current optical uncertainty remains a partial budget: signal Monte Carlo, scalar lidar-ratio perturbation and reference-boundary perturbation are represented, while correlation/systematic semantics and fitted gluing-coefficient uncertainty still require explicit characterization.
- Physical photon-counting saturation is not characterized; the current dead-time occupancy guard is provisional and must not be described as a detector saturation limit.
- Cloud screening is not yet a productive Rayleigh-reference rejection gate.
- No hard propagated-error SNR gate is enabled without SPU evidence.
- Current NumPy 2.5/netCDF4 1.7.4 write-time deprecation warnings are tracked as an upstream dependency interaction; they are not hidden by pinning NumPy backwards.

## Deferred high-column schema

The high-column redesign remains outside schema v1. Before adding altitude-resolved support/top, ensemble members, cascade handoffs or backbone products, P5 must supply their synthetic acceptance tests and preserve the frozen rule that the target altitude is never permission to extrapolate or fabricate support.
