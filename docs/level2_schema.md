# Current Level 2 product

LEBEAR produces one canonical block-resolved elastic Level 2 dataset. It uses
two-sided Klett–Fernald–Sasano integration, a progressive vertical grid,
prioritized molecular-reference ranges and selection-aware Monte Carlo.

The product is conditional on the declared aerosol lidar ratio and reference
boundary. It does not assert that either quantity was independently observed.

## Coordinates

| Coordinate | Meaning |
| --- | --- |
| `block_time` | representative UTC time of each averaging block |
| `block_start_utc`, `block_end_utc` | measured UTC extent of each block |
| `segment_id` | contiguous scientific segment supplying each block |
| `solar_regime` | `day` or `night` regime supplying each block |
| `wavelength` | processed elastic wavelength in nm |
| `altitude` | progressive-grid altitude above the lidar in m |
| `residual_fraction` | declared aerosol/molecular backscatter ratio at the reference |
| `mc_iteration` | Monte Carlo realization index |

The `residual_fraction` dimension represents explicit conditional scenarios;
it is not a sampled probability distribution. Routine processing uses `f=0`.

Temporal membership remains anchored to the configured wall-clock block size,
but a block is never allowed to cross a scientific `segment_id` boundary.
`block_time` is the mean timestamp of the profiles actually contributing to
that segment-homogeneous block. `solar_elevation_deg(block_time)` stores the
mean geometric solar-center elevation of those source profiles. The Level 2
product also carries the session segment table and the exact solar threshold /
algorithm provenance inherited from Level 1.

## Vertical representation

- Native bins are preserved below the first aggregation threshold.
- Higher cells contain strict contiguous groups of native bins.
- A missing or invalid source bin invalidates its cell; gaps are not bridged.
- `effective_vertical_resolution_m(wavelength, altitude)` reports the achieved
  cell width.
- `source_bin_count(wavelength, altitude)` reports how many native bins formed
  each cell.

Signed finite background-corrected signal can be averaged into a cell. KFS
still requires finite positive signal along the oriented integration path.

## Molecular state and assumptions

- `molecular_backscatter(block_time, wavelength, altitude)`
- `molecular_extinction(block_time, wavelength, altitude)`
- `lidar_ratio_assumed_sr(block_time, wavelength)`
- `lidar_ratio_std_sr(block_time, wavelength)`

Level 1 stores the canonical thermodynamic state as an hourly
`atmosphere_time x altitude` field. For every Level 2 block, temperature is
interpolated linearly in time and pressure in `log(P)`; molecular
backscatter/extinction are then calculated at that block time and aggregated
onto the progressive grid. Molecular state is stored even for blocks whose
lidar signal is not retrieval-valid. Level 2 performs no external meteorology
IO.

## Solar regime and segment provenance

Level 2 does not recompute solar geometry. It inherits the Level 1/L0
classification and stores:

- `segment_id(block_time)`;
- `solar_regime(block_time)`;
- `solar_elevation_deg(block_time)`;
- `Segment_Label(segments)`;
- `Segment_Regime(segments)`;
- segment start/end times;
- `Solar_Day_Night_Threshold_deg`;
- `Solar_Position_Algorithm`.

LEBEAR may subset the same scientific session with `--regime day|night`,
`--segment segXX`, and/or `--time-window-utc`. These are derived selections,
not new session identities.

## Signal and gluing

- `range_corrected_signal_block(block_time, wavelength, altitude)`
- `range_corrected_signal_error_block(block_time, wavelength, altitude)`
- `signal_source_flag(block_time, wavelength)`
- `retrieval_input_valid_flag(block_time, wavelength)`
- `retrieval_input_invalid_reason(block_time, wavelength)`
- gluing attempted/success/fallback flags;
- gluing start, split and stop altitudes;
- fitted slope/intercept, correlation, relative RMSE and relative bias.

The saved block RCS is the signal used by the inversion after subtraction of
the nominal fitted residual `B z²` term.

## Molecular reference and background

For every block/wavelength the product stores:

- selected reference altitude;
- lower and upper bounds of the search range that supplied it;
- search-range index and fallback flag;
- relative slope, relative variance, valid fraction and diagnostic cost;
- median uncertainty-based SNR diagnostic;
- selected cell resolution and native source-bin count;
- highest contiguous path altitude available before inversion;
- fitted residual background, formal standard error and
  calibration/background correlation.

The primary range is tried first. Only if it has no Rayleigh-accepted,
path-admissible candidate is the fallback range considered. Altitude is not
added to the ranking cost and does not certify molecular purity.

## Optical products

Nominal block solutions:

- `aerosol_backscatter_nominal_block`
- `aerosol_extinction_nominal_block`

Monte Carlo summaries:

- aerosol backscatter/extinction mean;
- random standard deviation;
- 2.5% and 97.5% quantiles;
- `mc_valid_fraction` at every block, wavelength, boundary scenario and
  altitude.

Period summaries:

- `aerosol_backscatter_mean`
- `aerosol_extinction_mean`
- `temporal_support_count`
- `temporal_support_fraction`

Temporal means are finite-only at each altitude. They must be interpreted with
the corresponding temporal support.

## Two-sided branch diagnostics

- `kfs_backward_valid_flag(block_time, wavelength)`
- `kfs_forward_valid_flag(block_time, wavelength)`
- `kfs_backward_endpoint_altitude_m(block_time, wavelength)`
- `kfs_forward_endpoint_altitude_m(block_time, wavelength)`
- matching endpoint variables for every residual-fraction/Monte Carlo draw.

An endpoint is the last finite cell reached continuously from the reference in
that branch. A forward branch may be useful without reaching the grid top.
Users must not infer support to 20 or 25 km merely because the file contains
those altitude coordinates.

`retrieval_success_flag` currently indicates that the nominal backward branch
is valid. Explicit connected-column, 20 km and 25 km status fields are a
pre-release task and will replace any temptation to treat this flag as a
full-column certificate.

## Selection-aware Monte Carlo

For each realization MILGRAU:

1. perturbs native RCS with its propagated one-sigma measurement uncertainty;
2. refits the robust residual background and molecular scaling;
3. subtracts the fitted `B z²` term;
4. rebuilds the progressive grid;
5. reruns native-grid Rayleigh QA and prioritized range selection;
6. samples aerosol lidar ratio from its configured uncertainty;
7. runs the two-sided inversion.

The product records selected-reference samples, selected search-range samples,
background samples, branch endpoints and altitude-resolved finite-realization
fractions. Selection success is diagnostic and is not converted into a hidden
acceptance threshold.

## Missing values and support

- `NaN` means unavailable or unsupported, not zero aerosol.
- Missing one-sigma signal uncertainty is unsupported and is never converted to
  zero noise.
- Internal gaps are not interpolated for retrieval.
- A high reference or endpoint is not automatically a scientifically better
  result.
- Physical near-field/overlap validity and characterized detector saturation
  remain separate instrument-validation questions.

## Completeness and provenance

The dataset records requested, processed and failed wavelengths plus product
completeness/status. At least one valid block is required for a wavelength to
be listed as processed.

Scientific lineage includes:

- exact source Level 1 SHA-256;
- exact processing and station YAML snapshots;
- software version;
- installed Python source-content hash;
- upstream thermodynamic and station/calibration provenance;
- Monte Carlo seed and iteration count;
- integration, reference-selection, boundary and uncertainty policies.

The exact source hash and software release identify the implemented retrieval;
no separate selectable retrieval generation is exposed.
