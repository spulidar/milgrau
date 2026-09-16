# MILGRAU Level 2 product schema

This document describes the current versioned LEBEAR NetCDF contract on `new-architecture`. It is descriptive, not a second source of scientific constants: numerical equations remain owned by the scientific modules, configuration values by `config.yaml` / station state where appropriate, and storage assembly by the Level 2 dataset modules.

## Current identities

MILGRAU keeps software, storage and scientific-method identities separate:

- package/software version: normal MILGRAU release identity;
- `level2_product_schema_version = "3"`;
- `level2_product_schema_change = "auditable_rayleigh_candidate_catalogue"`;
- `level2_retrieval_method_version = "4"`;
- productive inversion: backward Klett–Fernald–Sasano.

Schema 2 introduced altitude-resolved backward-inversion support, top/bottom bounds and effective successful-block count. Schema 3 retains those semantics and adds the complete auditable Rayleigh candidate catalogue. Schema 3 is a storage/traceability change relative to method 4; it does not itself change the KFS equations, lidar-ratio assumption, uncertainty propagation or candidate ranking.

Method 4 changes the productive Rayleigh selection order to:

`enumerate all complete candidates -> diagnose -> minimum QA -> rank accepted candidates only`.

Among candidates that pass the configured minimum shape/calibration QA, the current productive ranking remains the historical diagnostic cost

`relative_slope + relative_variance`,

with deterministic lower-grid-index tie breaking. There is no productive altitude preference and no hard propagated-SNR threshold. Candidate SNR is persisted as a diagnostic only.

Method 4 retains the common-support and conservative block-uncertainty semantics established by method 3.

Incremental reuse rejects products whose schema version, retrieval-method version, productive KFS identity, Fernald scientific identity, gluing-score identity or exact Level 1 content identity no longer matches the running method. Partial products are never incrementally reused.

## Main coordinates

| Coordinate | Meaning | Semantics |
| --- | --- | --- |
| `time` | original Level 1 timestamps used for time-expanded diagnostics | not an independent KFS retrieval axis |
| `block_time` | start/floored timestamp of each retrieval block | productive block axis |
| `wavelength` | successfully processed elastic wavelength | `nm` |
| `altitude` | altitude above station | `m`, positive upward |
| `rayleigh_candidate` | ordered complete candidate windows on the shared altitude/search grid | schema-3 audit axis |

`requested_wavelength`, `processed_wavelength` and `failed_wavelength` belong to the multispectral completeness contract. The scientific `wavelength` coordinate equals `processed_wavelengths` exactly.

## Canonical optical products

There is one aggregate stored variable per aerosol optical quantity:

- `aerosol_backscatter_mean(wavelength, altitude)`;
- `aerosol_backscatter_mean_error(wavelength, altitude)`;
- `aerosol_extinction_mean(wavelength, altitude)`;
- `aerosol_extinction_mean_error(wavelength, altitude)`.

Block products are explicit:

- `aerosol_backscatter_block(block_time, wavelength, altitude)`;
- `aerosol_backscatter_error_block(block_time, wavelength, altitude)`;
- `aerosol_extinction_block(block_time, wavelength, altitude)`;
- `aerosol_extinction_error_block(block_time, wavelength, altitude)`.

Legacy unsuffixed duplicate aggregate aliases are not part of the versioned schema.

Elastic extinction is conditional on the configured aerosol lidar ratio. It is not presented as an independently measured Raman extinction profile.

## Units and metadata ownership

Base Level 2 variable metadata is owned by `milgrau.level2.metadata`. Schema-specific candidate metadata is owned by `milgrau.level2.rayleigh_catalogue_dataset`. The NetCDF contract tests require the emitted inventory to be covered by these registries, so public variables cannot silently appear without readable metadata.

Physically calibrated quantities use physical units where defined, including:

- molecular/aerosol backscatter: `m-1 sr-1`;
- molecular/aerosol extinction: `m-1`;
- aerosol lidar ratio and configured standard deviation: `sr`;
- altitude diagnostics: `m`;
- fractions, correlations, SNR, flags, counts and dimensionless QA metrics: `1`.

Selected lidar signals are deliberately not given invented SI radiometric units. Source-dependent instrumental quantities and their calibration coefficients use explicit `unit_status` metadata instead.

MILGRAU currently makes no formal external metadata-convention conformance claim. No global `Conventions` attribute is added merely for appearance.

## Missing values and uncertainty support

NaN is never filled/interpolated merely to extend retrieval coverage.

A signal sample contributes to productive averaging only when its value is finite and its one-sigma uncertainty is finite and non-negative. Missing uncertainty is not converted to zero uncertainty.

The backward Monte Carlo path preserves missing `rcs_error` as unsupported. Unsupported internal bins are not bridged to create a longer accepted optical profile.

Method-4 aggregate optical uncertainty retains the method-3 conservative block dependence policy:

`sigma_mean = sum(sigma_block) / n_effective`

on common value/error support. This deliberately provides no automatic `1/sqrt(N)` gain for the current mixed block-level uncertainty, because aerosol-lidar-ratio nuisance is shared and reference-boundary dependence has not been decomposed sufficiently to justify block independence.

Gluing uncertainty remains partial measurement-noise propagation. Fitted gluing slope/intercept uncertainty and covariance are not included and are not claimed negligible.

## Altitude-resolved backward inversion support

Schema 2 introduced, and schema 3 retains:

- `retrieval_inversion_support_flag(wavelength, altitude)`;
- `retrieval_inversion_support_flag_block(block_time, wavelength, altitude)`;
- `retrieval_inversion_effective_block_count(wavelength, altitude)`;
- aggregate `retrieval_bottom_altitude_m` / `retrieval_top_altitude_m`;
- block `retrieval_bottom_altitude_m_block` / `retrieval_top_altitude_m_block`.

Block inversion support is the contiguous common optical value/error path ending at that block's accepted exact Rayleigh boundary. An invalid internal bin breaks the backward path; lower bins are not relabeled supported by jumping across the gap.

Aggregate inversion support is also contiguous. The separate effective-block count exposes whether an altitude is supported by, for example, 5/5 blocks or only 1/5.

These fields describe **algorithmic inversion support**, not validated full instrument support. No evidence-backed lower overlap/instrument mask is currently applied. Therefore a very low algorithmic bottom altitude must not be interpreted as quantitative near-field validation. The product records this limitation through `retrieval_inversion_support_scope` and `retrieval_inversion_support_instrument_mask = not_applied_uncharacterized`.

A future stricter `retrieval_support_flag` remains reserved for the conjunction of inversion support and validated instrument support.

## Schema-3 Rayleigh candidate catalogue

Every complete candidate window inside the configured physical Rayleigh search interval is persisted on

`(block_time, wavelength, rayleigh_candidate)`.

Common candidate geometry is stored once on `rayleigh_candidate`:

- center/start/stop altitude;
- center altitude-grid index.

Per block/wavelength/candidate the product stores:

- evaluated flag;
- valid-bin count and valid fraction;
- relative slope and relative variance;
- origin-constrained calibration factor;
- free-intercept diagnostic;
- propagated-uncertainty SNR median and contributing-bin count;
- historical diagnostic cost;
- minimum-QA rejection bit mask;
- accepted flag;
- **unfiltered minimum-cost flag** before QA;
- **productive selected flag** after QA-first filtering.

The rejection bit mask currently represents the enabled minimum gates:

- bit 1: insufficient valid fraction;
- bit 2: invalid calibration;
- bit 4: excess relative slope;
- bit 8: excess relative variance.

A zero rejection mask means the candidate passed every currently enabled minimum gate. Propagated SNR remains diagnostic-only and therefore does not set a rejection bit.

Keeping both `rayleigh_candidate_unfiltered_min_cost_flag` and `rayleigh_candidate_selected_flag` makes the method-v4 change auditable: when the historical raw minimum fails QA, the file can show that it was not selected and which QA-passing candidate was used instead.

The schema-3 validator requires:

- finite, ordered common candidate geometry;
- accepted flag exactly equivalent to evaluated + zero rejection mask;
- productive selection only among accepted candidates;
- exactly one unfiltered minimum-cost candidate per evaluated block/wavelength;
- exactly one productive selected candidate for each successful Rayleigh block/wavelength;
- persisted selected candidate altitude exactly equal to the Rayleigh reference actually used by the retrieval.

The generic Level 2 contract dispatches to this validator for every product declaring schema 3 or later. A file cannot satisfy schema 3 merely by carrying the version attribute while omitting/corrupting the catalogue.

## Scattering ratio is diagnostic, not retrieval support

`scattering_ratio_mean` and block equivalents may remain finite above the accepted KFS boundary. Finite measured-to-molecular scattering ratio at high altitude is not evidence of supported aerosol backscatter/extinction there.

This distinction is especially important in weak-signal upper tails, where finite or visually structured ratios can coexist with large propagated uncertainty and no accepted backward optical support.

## Productive provenance

The final product records machine-readable identity including:

- schema version/change;
- retrieval method version/change;
- productive KFS method/integration mode;
- Rayleigh selection policy and SNR policy;
- Fernald implementation/scientific-change identity;
- molecular-atmosphere identity;
- Monte Carlo method/iterations/seed;
- uncertainty scope and block-correlation policy;
- KFS boundary model and aerosol-reference assumption;
- assumed aerosol lidar ratio and uncertainty;
- versioned gluing-selection score;
- exact Level 1 content hash;
- normalized MILGRAU Python-source hash and repository revision where available;
- exact configuration snapshots used by the run.

These identities are deliberately separate from package CalVer so scientific/storage changes make older products stale even when ordinary software versioning would be insufficient.

## Future high-column state

Schema 3 does **not** introduce a high-column backbone, ensemble or cascade. Those remain separate R&D gates.

Before a long-mean backbone becomes productive, MILGRAU requires temporal-support/stability evidence showing which blocks/time intervals actually contribute at each altitude. Longer averaging must not allow a transient early interval to create an apparently representative full-measurement high-altitude claim.

Future backbone/ensemble/cascade fields will be added only when their scientific contracts are validated and will trigger deliberate schema/method version changes as appropriate.
