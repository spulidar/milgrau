# MILGRAU Level 2 product schema

This document is the P3 schema audit for the current LEBEAR NetCDF product. It describes what the writer actually emits before schema cleanup so migration decisions are explicit and testable.

Audit baseline: branch `new-architecture`, after P2 closure. The numerical baseline remains the validated productive backward KFS retrieval; this document does not authorize changes to retrieval equations or support.

## Coordinate model

| Coordinate | Meaning | Current units |
| --- | --- | --- |
| `time` | Level 1 profile times expanded from accepted Level 2 blocks | inherited datetime coordinate |
| `block_time` | representative time for each Level 2 averaging/retrieval block | datetime coordinate |
| `wavelength` | successfully processed elastic wavelength | `nm` |
| `altitude` | altitude above station | `m` |

Completeness arrays use dedicated dimensions `requested_wavelength`, `processed_wavelength`, and `failed_wavelength`; these are contract dimensions, not scientific coordinates.

## Current data-variable inventory

The role column distinguishes source/diagnostic state from scientific optical products. `aggregate` means one profile per wavelength produced from accepted blocks; `block` means one profile/state per `block_time`; `time-expanded` means block state repeated onto the original Level 1 `time` grid for traceability.

### Molecular and selected-signal variables

| Variable | Dimensions | Role | Current metadata status |
| --- | --- | --- | --- |
| `molecular_backscatter` | `(wavelength, altitude)` | molecular scientific input | units/long name need P3 audit |
| `molecular_extinction` | `(wavelength, altitude)` | molecular scientific input | units/long name need P3 audit |
| `molecular_transmission` | `(wavelength, altitude)` | molecular scientific input | units/long name need P3 audit |
| `simulated_molecular_signal` | `(wavelength, altitude)` | molecular diagnostic | units/description need P3 audit |
| `simulated_molecular_range_corrected_signal` | `(wavelength, altitude)` | molecular diagnostic | units/description need P3 audit |
| `scaled_molecular_range_corrected_signal` | `(wavelength, altitude)` | aggregate calibrated molecular diagnostic | units/description need P3 audit |
| `scaled_molecular_range_corrected_signal_block` | `(block_time, wavelength, altitude)` | block calibrated molecular diagnostic | units/description need P3 audit |
| `glued_corrected_signal` | `(time, wavelength, altitude)` | time-expanded selected signal | description present; units missing |
| `glued_corrected_signal_error` | `(time, wavelength, altitude)` | time-expanded one-sigma selected-signal uncertainty | units/description incomplete |
| `glued_corrected_signal_block` | `(block_time, wavelength, altitude)` | block selected signal | units/description incomplete |
| `glued_corrected_signal_error_block` | `(block_time, wavelength, altitude)` | block selected-signal uncertainty | units/description incomplete |
| `glued_corrected_signal_mean` | `(wavelength, altitude)` | aggregate selected signal | units/description incomplete |
| `glued_corrected_signal_error_mean` | `(wavelength, altitude)` | aggregate selected-signal uncertainty | units/description incomplete |
| `glued_range_corrected_signal` | `(time, wavelength, altitude)` | time-expanded selected RCS | description present; units missing |
| `glued_range_corrected_signal_error` | `(time, wavelength, altitude)` | time-expanded RCS uncertainty | units/description incomplete |
| `glued_range_corrected_signal_block` | `(block_time, wavelength, altitude)` | block selected RCS | units/description incomplete |
| `glued_range_corrected_signal_error_block` | `(block_time, wavelength, altitude)` | block RCS uncertainty | units/description incomplete |
| `glued_range_corrected_signal_mean` | `(wavelength, altitude)` | aggregate selected RCS | units/description incomplete |
| `glued_range_corrected_signal_error_mean` | `(wavelength, altitude)` | aggregate RCS uncertainty | units/description incomplete |
| `gluing_merge_source_flag` | `(time, wavelength, altitude)` | time-expanded per-bin source flag | flag metadata present |
| `gluing_merge_source_flag_block` | `(block_time, wavelength, altitude)` | block per-bin source flag | should share canonical flag metadata with time-expanded form |

### Optical products

| Variable | Dimensions | Role | Current metadata status |
| --- | --- | --- | --- |
| `scattering_ratio_mean` | `(wavelength, altitude)` | aggregate optical product | `units=1`; description present |
| `scattering_ratio_block` | `(block_time, wavelength, altitude)` | block optical product | `units=1`; description present |
| `aerosol_backscatter_mean` | `(wavelength, altitude)` | **canonical aggregate aerosol backscatter** | units/long name need P3 audit |
| `aerosol_backscatter_mean_error` | `(wavelength, altitude)` | **canonical aggregate one-sigma backscatter uncertainty** | units/uncertainty description need P3 audit |
| `aerosol_extinction_mean` | `(wavelength, altitude)` | **canonical aggregate aerosol extinction** | units/long name need P3 audit |
| `aerosol_extinction_mean_error` | `(wavelength, altitude)` | **canonical aggregate one-sigma extinction uncertainty** | units/uncertainty description need P3 audit |
| `aerosol_backscatter` | `(wavelength, altitude)` | legacy duplicate alias of `aerosol_backscatter_mean` | remove through explicit schema migration |
| `aerosol_backscatter_error` | `(wavelength, altitude)` | legacy duplicate alias of `aerosol_backscatter_mean_error` | remove through explicit schema migration |
| `aerosol_extinction` | `(wavelength, altitude)` | legacy duplicate alias of `aerosol_extinction_mean` | remove through explicit schema migration |
| `aerosol_extinction_error` | `(wavelength, altitude)` | legacy duplicate alias of `aerosol_extinction_mean_error` | remove through explicit schema migration |
| `aerosol_backscatter_block` | `(block_time, wavelength, altitude)` | block aerosol backscatter | units/long name need P3 audit |
| `aerosol_backscatter_error_block` | `(block_time, wavelength, altitude)` | block one-sigma backscatter uncertainty | units/description need P3 audit |
| `aerosol_extinction_block` | `(block_time, wavelength, altitude)` | block aerosol extinction | units/long name need P3 audit |
| `aerosol_extinction_error_block` | `(block_time, wavelength, altitude)` | block one-sigma extinction uncertainty | units/description need P3 audit |
| `retrieval_success_flag` | `(block_time, wavelength)` | block productive-retrieval validity | flag metadata present |
| `retrieval_success_fraction` | `(wavelength)` | aggregate success diagnostic | `units=1`; description present |

The `OpticalProducts` runtime contract stores the aggregate arrays as `aerosol_backscatter`, `aerosol_backscatter_error`, `aerosol_extinction`, and `aerosol_extinction_error`. Those Python field names are internal runtime names. The NetCDF schema deliberately distinguishes aggregate (`*_mean`) from block (`*_block`) products; the unsuffixed NetCDF variables do not encode a distinct scientific quantity.

### Rayleigh-reference and calibration diagnostics

Aggregate `(wavelength)` variables:

- `rayleigh_reference_altitude_m`
- `rayleigh_reference_start_altitude_m`
- `rayleigh_reference_stop_altitude_m`
- `rayleigh_reference_valid_bins`
- `rayleigh_reference_success_flag`
- `rayleigh_reference_relative_slope`
- `rayleigh_reference_relative_variance`
- `rayleigh_reference_valid_fraction`
- `rayleigh_calibration_factor`
- `rayleigh_calibration_intercept`

Block `(block_time, wavelength)` counterparts:

- `rayleigh_reference_altitude_m_block`
- `rayleigh_reference_start_altitude_m_block`
- `rayleigh_reference_stop_altitude_m_block`
- `rayleigh_reference_valid_bins_block`
- `rayleigh_reference_success_flag_block`
- `rayleigh_reference_relative_slope_block`
- `rayleigh_reference_relative_variance_block`
- `rayleigh_reference_valid_fraction_block`
- `rayleigh_calibration_factor_block`
- `rayleigh_calibration_intercept_block`

Current gaps: only part of this family has units/descriptions attached. P3 must make aggregate and block metadata symmetric. Reference altitudes are physical meters; slope/variance/valid-fraction diagnostics are dimensionless; calibration-factor/intercept units must describe the actual signal/RCS spaces rather than be guessed.

### Lidar-ratio and KFS diagnostics

| Variable | Dimensions | Role |
| --- | --- | --- |
| `lidar_ratio_assumed_sr` | `(wavelength)` | assumed aerosol lidar ratio used by elastic retrieval |
| `lidar_ratio_std_sr` | `(wavelength)` | configured lidar-ratio uncertainty |
| `kfs_backward_valid_flag` | `(wavelength)` | aggregate backward-branch validity diagnostic |
| `kfs_forward_valid_flag` | `(wavelength)` | aggregate forward research-branch validity diagnostic |
| `kfs_backward_valid_flag_block` | `(block_time, wavelength)` | block backward validity |
| `kfs_forward_valid_flag_block` | `(block_time, wavelength)` | block forward research validity |
| `kfs_branch` | `(wavelength, altitude)` | aggregate branch/support-origin diagnostic |
| `kfs_branch_block` | `(block_time, wavelength, altitude)` | block branch diagnostic |

The productive success criterion remains backward KFS. Forward fields are retained as research/diagnostic state and must not be described as productive support.

### Gluing diagnostics

Time-expanded `(time, wavelength)` variables:

- `gluing_attempted_flag`
- `gluing_success_flag`
- `single_channel_fallback_flag`
- `gluing_split_altitude_m`
- `gluing_start_altitude_m`
- `gluing_stop_altitude_m`
- `gluing_slope`
- `gluing_intercept`
- `gluing_correlation`
- `gluing_relative_rmse`
- `gluing_relative_bias`

Block `(block_time, wavelength)` counterparts append `_block` to every name above.

The altitude diagnostics require meter units on both time-expanded and block forms. Correlation/RMSE/bias are dimensionless. Slope/intercept metadata must state the mapping direction (analog into virtual photon-counting signal space) rather than rely on variable names alone.

### Source-selection and retrieval-input QA

Time-expanded `(time, wavelength)` variables:

- `signal_source_flag`
- `retrieval_input_valid_flag`
- `retrieval_input_invalid_reason`
- `retrieval_input_snr_median`

Block `(block_time, wavelength)` counterparts:

- `signal_source_flag_block`
- `retrieval_input_valid_flag_block`
- `retrieval_input_invalid_reason_block`
- `retrieval_input_snr_median_block`

These are diagnostics, not replacements for future `retrieval_support_flag[..., altitude]`. A valid input or successful block does not mean every altitude bin is scientifically supported.

### Multispectral completeness contract

| Variable | Dimensions | Meaning |
| --- | --- | --- |
| `requested_wavelengths` | `(requested_wavelength)` | requested wavelengths |
| `processed_wavelengths` | `(processed_wavelength)` | wavelengths with published scientific result |
| `failed_wavelengths` | `(failed_wavelength)` | requested wavelengths that failed |
| `failed_wavelength_stage` | `(failed_wavelength)` | stable numeric failure stage |
| `failed_wavelength_code` | `(failed_wavelength)` | stable numeric failure code |
| `failed_wavelength_message` | `(failed_wavelength)` | human-readable diagnosis |
| `failed_wavelength_cause` | `(failed_wavelength)` | compact cause summary |

The scientific `wavelength` coordinate must equal `processed_wavelengths` exactly. Partial products are intentionally not incrementally reusable.

## Flag metadata audit finding

The current writer supplies `flag_values` and `flag_meanings` to many state variables, but several `flag_values` attributes are serialized as comma-separated strings. P3 must migrate flag metadata to a representation that is unambiguous and CF-friendly while preserving integer data values and stable meanings. Aggregate/block forms of the same flag must use the same mapping.

No flag cleanup may convert scientific support into a simple finite-value test. In particular, `retrieval_success_flag` is block-level success; future altitude-resolved support remains a separate frozen concept.

## Aggregate optical alias decision and migration policy

The writer currently materializes each aggregate aerosol optical array twice from the same runtime array:

- `aerosol_backscatter_mean` == `aerosol_backscatter`
- `aerosol_backscatter_mean_error` == `aerosol_backscatter_error`
- `aerosol_extinction_mean` == `aerosol_extinction`
- `aerosol_extinction_mean_error` == `aerosol_extinction_error`

This is schema duplication, not two scientific products.

**Decision:** the `_mean` names are canonical because they make the aggregate/block distinction explicit and they are already part of the minimum Level 2 contract. The unsuffixed NetCDF aliases are legacy schema residue.

**Migration policy:**

1. introduce an explicit Level 2 product-schema version before removing aliases;
2. update internal QA/explorer consumers to canonical `_mean` names;
3. remove the four unsuffixed aliases in the same schema-version change rather than retaining indefinite duplicate arrays;
4. make pre-version/older-schema Level 2 products stale for incremental reuse when the schema version becomes mandatory;
5. do not rename the internal `OpticalProducts` dataclass fields merely to mimic NetCDF names; runtime and storage contracts are separate concerns;
6. do not add future support/top variables as part of this alias migration.

This policy intentionally favors one truthful stored quantity over permanent compatibility duplication. Users with old Level 2 files can reprocess them with the versioned schema.

## Global attributes: current identity groups

The writer currently carries Level 1 attributes forward, then adds Level 2 identity/provenance including:

- processing identity: `Processing_level`, `Pipeline`, `Input_Level1_File`, `LEBEAR_Mode`, `LEBEAR_Block_Average_Minutes`;
- productive inversion identity: `KFS_Mode`, `KFS_Mode_Description`, `elastic_backscatter_inversion_method`, `integration_mode`, `fernald_implementation_version`, `fernald_scientific_change`;
- method descriptions: `Molecular_Rayleigh_Method`, `Rayleigh_Calibration_Method`, `Gluing_Method`, `Gluing_Error_Propagation`, `Signal_Selection_Policy`, `Single_Channel_QA`;
- recipe/threshold provenance: `Single_Channel_Priority`, Rayleigh QA limits, atmosphere/gluing source/channel summaries;
- completeness identity: `product_completeness`, `product_status`, `Wavelength_Order`, `Partial_Product_Reuse`;
- uncertainty scope: `uncertainty_scope`, `uncertainty_method`;
- molecular-atmosphere implementation identity inherited from shared scientific metadata.

P3 must separate package/software version, scientific-method version and storage-schema version so a schema-only migration does not pretend an equation changed, and a future scientific-method change cannot masquerade as a metadata-only rewrite.

## Known semantic constraints carried into P3

- Productive elastic inversion is backward KFS from one accepted high-altitude Rayleigh reference toward lower altitude.
- Optical bins outside valid backward support remain unsupported/NaN; finite-value status is not a substitute for support semantics.
- Elastic extinction is conditional on assumed aerosol lidar ratio.
- `retrieval_success_flag(block_time,wavelength)` is block-level productive success, not altitude-resolved support.
- Future `retrieval_support_flag` / `retrieval_top_altitude_m` are not added until their synthetic support tests exist.
- Cloud screening remains nonproductive until validated on SPU observations.
- Physical PC saturation remains uncharacterized; the current guard is provisional and must not be relabeled as detector characterization.

## P3 implementation order from this audit

1. Add explicit schema identity and tests for currentness/staleness.
2. Atomically migrate internal consumers to canonical `*_mean` aerosol fields and remove the four duplicate stored aliases.
3. Complete units, long names/descriptions and uncertainty metadata for every physical variable family.
4. Normalize flag metadata representation and add contract tests.
5. Version/document gluing selection score and other method provenance without changing numerical weights.
6. Define the immutable input-manifest/provenance policy and focused product/method documentation.
