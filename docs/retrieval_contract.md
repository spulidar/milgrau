# Typed Level 2 retrieval contract

`process_wavelength` returns one frozen, slotted
`WavelengthRetrievalResult` instead of an unstructured dictionary. The top-level
result composes seven domain-specific records so field ownership and dimensions
remain visible:

| Record | Responsibility |
| --- | --- |
| `MolecularProfiles` | molecular source, backscatter, extinction, transmission, simulated signal, and scaled molecular RCS |
| `GluedSignals` | available channels plus selected time, block, and mean corrected/RCS signals and per-bin merge-source flags |
| `OpticalProducts` | scattering ratio, aerosol backscatter/extinction, uncertainties, and retrieval-success flags |
| `RayleighDiagnostics` | aggregate and per-block reference window, QA, and calibration diagnostics |
| `KfsDiagnostics` | lidar-ratio assumptions, aggregate/per-block branch flags, and independent backward/forward validity flags |
| `GluingDiagnostics` | time and block gluing-attempt, gluing-approval, single-channel fallback, window, fit, and residual diagnostics |
| `SignalSelectionDiagnostics` | per-time/per-block selected source, retrieval-input validity, invalid-reason code, and median SNR diagnostic |

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
2. `glue_signal_blocks` attempts gluing when both channels exist, assesses
   PC/AN candidates independently when needed, and returns `BlockGluingResult`
   with selected block signals, source/validity states, uncertainties, and
   diagnostics;
3. `build_molecular_model` returns `MolecularModel` with atmospheric profiles
   and the retrieval assumptions consumed downstream;
4. `retrieve_optical_blocks` performs Rayleigh QA and KFS and returns the four
   public molecular/optical/Rayleigh/KFS records;
5. `assemble_wavelength_result` expands block results to time and constructs the
   validated `WavelengthRetrievalResult`.

Failures are wrapped as `RetrievalStageError` with one of the stable stage names
`selection_and_blocking`, `gluing`, `molecular_model`, `rayleigh_kfs`, or
`result_assembly`, while preserving the original exception as `__cause__`.

## SCI-001 scientific versioning

`milgrau.level2.dataset` consumes named attributes from the typed records; it no
longer synchronizes arbitrary dictionary keys with NetCDF variables. SCI-001
deliberately changes aerosol optical values, adds independent backward/forward
validity flags, marks only the exact boundary bin as the reference branch, and
writes the Fernald implementation identity. Rayleigh valid-bin counts remain
converted to the existing NetCDF `float64` representation even though their
retrieval-contract dtype is `int32`.

The synthetic 532 nm fixture hashes every coordinate and data variable,
including name, dimensions, dtype, shape, attributes, and contiguous value
bytes, plus global attributes. The SCI-001 checkpoint digest before the
SCI-002 state-schema change was:

```text
b6edc57dde2750550078f8b08011534b26171e146d1fe9127033df78d75c5ccb
```

The pre-SCI-001 digest remains intentionally obsolete because it encoded the
incorrect backward molecular sign and is not scientific truth. The checkpoint
digest above is retained only as audit evidence for the approved glued path;
SCI-002 necessarily changes the contract hash by adding independent state
variables. Level 2 products with Fernald implementation versions before 2 must
still be reprocessed.

## SCI-002 signal selection and retrieval validity

`SignalSource` is serialized as `invalid=0`, `glued=1`,
`photon_counting=2`, and `analog=3`. Channel existence, gluing attempt,
gluing approval, single-channel selection, retrieval-input validity, and final
retrieval success are independent states. In particular, rejected or
unavailable gluing does not prevent a QA-valid PC-only or AN-only profile from
reaching Rayleigh calibration and the two-sided Fernald-v2 kernel.

Minimum single-channel QA uses no new empirical SNR threshold. The complete
positive-altitude domain and configured Rayleigh interval must be covered by a
finite positive corrected signal and finite non-negative channel uncertainty;
at least one positive uncertainty bin must make median absolute SNR calculable.
An explicit successful Level 1 channel-correction flag is required. PC also
requires an explicit saturation mask with no flagged bin in the input domain.
Together with the corrected-signal contract, that flag records completion of
background, dead-time (when configured), bin-shift, and dark-current processing;
the current Level 1 contract does not expose a separate block background flag.
Analog saturation thresholds are not invented by SCI-002 and remain an
instrument-validation limitation for SCI-007.

When gluing fails and both channels pass the same coverage/finite-uncertainty
requirements, `inversion.gluing.single_channel_priority` makes the choice
deterministic. Its repository default is `photon_counting`, preserving the
previous PC-first fallback order explicitly; `analog` is also valid. The old
`fallback_to_photon_counting` key was removed because it could not truthfully
describe AN-only selection.

Contracts reject contradictory combinations before dataset assembly. An
invalid input leaves all block optical products `NaN`; a valid input may still
have `retrieval_success_flag=0` if Rayleigh QA or either KFS side fails later.
The current schema duplicates several block states on `time`; ENG-022/035 will
centralize temporal representation later without changing these semantics.

The SCI-002 schema-only golden digest (the approved glued optical values remain
covered independently by numeric assertions) is:

```text
4aa7f2102e5a2d4060d32311539780196a4d0b2e794770b96782ba715457d002
```
