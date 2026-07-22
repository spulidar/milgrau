# Typed Level 2 retrieval contract

`process_wavelength` returns one frozen, slotted
`WavelengthRetrievalResult` instead of an unstructured dictionary. The top-level
result composes six domain-specific records so field ownership and dimensions
remain visible:

| Record | Responsibility |
| --- | --- |
| `MolecularProfiles` | molecular source, backscatter, extinction, transmission, simulated signal, and scaled molecular RCS |
| `GluedSignals` | source channels plus time, block, and mean corrected/RCS signals and merge-source flags |
| `OpticalProducts` | scattering ratio, aerosol backscatter/extinction, uncertainties, and valid-block flags |
| `RayleighDiagnostics` | aggregate and per-block reference window, QA, and calibration diagnostics |
| `KfsDiagnostics` | lidar-ratio assumptions and aggregate/per-block branch flags |
| `GluingDiagnostics` | aggregate-time and per-block gluing success, fallback, window, fit, and residual diagnostics |

The only optional fields are `analog_channel` and `photon_channel`, because a
configured fallback can process a wavelength with one source channel. At least
one must be present. Every other constructor field is required.

## Runtime dimensions and dtypes

For `T` Level 1 profiles, `B` averaging blocks, and `Z` altitude bins:

- `block_time` is `(B,)` with dtype `datetime64[ns]`;
- physical altitude profiles are `(Z,)` with dtype `float64`;
- time-expanded profiles are `(T, Z)` with dtype `float64`;
- block profiles are `(B, Z)` with dtype `float64`;
- time diagnostics are `(T,)`, block diagnostics are `(B,)`;
- semantic flags use `int8` in the retrieval contract;
- Rayleigh valid-bin counts use `int32` internally;
- wavelength, scalar counts/flags, sources, and scalar diagnostics have explicit
  integer, string, optional-string, or floating-point types.

`WavelengthRetrievalResult.validate` checks these invariants immediately after
retrieval. `validate_retrieval_results` repeats validation at the dataset
boundary, rejects empty or non-contract collections, duplicate wavelengths, and
different block-time coordinates. This happens before xarray assembly and
before NetCDF writing.

## Retrieval stages

`process_wavelength` is a sequencer over five explicit boundaries:

1. `prepare_wavelength_blocks` selects channels and returns
   `WavelengthBlockInputs` with grouped signals, errors, masks, times, and
   configuration;
2. `glue_signal_blocks` returns `BlockGluingResult` with block signals,
   uncertainties, source flags, and diagnostics;
3. `build_molecular_model` returns `MolecularModel` with atmospheric profiles
   and the retrieval assumptions consumed downstream;
4. `retrieve_optical_blocks` performs Rayleigh QA and KFS and returns the four
   public molecular/optical/Rayleigh/KFS records;
5. `assemble_wavelength_result` expands block results to time and constructs the
   validated `WavelengthRetrievalResult`.

Failures are wrapped as `RetrievalStageError` with one of the stable stage names
`selection_and_blocking`, `gluing`, `molecular_model`, `rayleigh_kfs`, or
`result_assembly`, while preserving the original exception as `__cause__`.

## Dataset compatibility

`milgrau.level2.dataset` consumes named attributes from the typed records; it no
longer synchronizes arbitrary dictionary keys with NetCDF variables. Dataset
names, dimensions, dtypes, attributes, and numerical values remain unchanged.
In particular, Rayleigh valid-bin counts remain converted to the historical
NetCDF `float64` representation even though their retrieval-contract dtype is
`int32`.

The frozen synthetic 532 nm fixture hashes every coordinate and data variable,
including name, dimensions, dtype, shape, attributes, and contiguous value
bytes, plus global attributes. Its digest before and after migration is:

```text
3a5cfa6aff51948afe8e5a2c889988cf50763f3ef07de14bc7b9261695127299
```

This change defines an internal engineering boundary only. It does not alter
gluing, molecular calculations, Rayleigh selection, KFS inversion, aggregation,
scientific parameters, or the public Level 2 NetCDF schema.
