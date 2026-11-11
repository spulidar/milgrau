# Scientific configuration reference for operators

This guide explains settings that matter to a researcher. Use the [operating guide](operating_milgrau.md) for commands. The values below describe the current distributed `config.yaml`; a locally edited recipe takes precedence and must be recorded in scientific provenance.

## Files and responsibilities

| File | Scientific purpose |
| --- | --- |
| `config.yaml` | Processing policy, selection thresholds, atmosphere, inversion and figure configuration |
| `station.yaml` | Observatory coordinates, instrument history, dated channel calibrations, geometry and station metadata |
| `scc.yaml` | Optional SCC product IDs, channel mappings and ancillary settings; not required when SCC is disabled |

Do not store network credentials in these files. `scc.enabled: false` is the standard configuration.

## Data locations and operation

| Setting | Distributed value | Meaning |
| --- | --- | --- |
| `directories.raw_data` | `01-data` | Original Licel data tree |
| `directories.processed_data` | `02-processed_data` | Scientific output root |
| `directories.log_dir` | `logs` | Operational logs |
| `processing.incremental` | `true` | Reuse valid/current outputs where supported |
| `processing.console_level` | `INFO` | Operator-facing progress and warnings |
| `processing.file_level` | `DEBUG` | Detailed trace and diagnostic log |
| `processing.spurious_extensions` | `.dpp,.zip,.txt,.json,.7z,.tar,.dat` | File extensions not treated as Licel candidates |
| `processing.quarantine_dir` | `quarantine` | Directory for *explicit* audited quarantine operations, not automatic raw-scan mutation |
| `processing.raw_scan_ignore_dirs` | `[]` | Additional directory basenames excluded from raw discovery |

Discovery does **not** automatically move or delete rejected files. A corrupt header can be skipped and recorded in the log without modifying the source.

## Level 0 — acquisition and session rules

| Setting | Distributed value | Interpretation |
| --- | --- | --- |
| `level0.acquisition_qa.laser_shot_tolerance_fraction` | `0.002` | Deviation tolerance around the homogeneous-group modal shot count |
| `level0.acquisition_qa.licel_header_time_jitter_s` | `1.0` | Allowable header duration deviation from nominal duration |
| `level0.session.max_gap_seconds` | `1800` | Maximum acquisition pause compatible with one continuous session |
| `level0.solar_regime.day_night_threshold_deg` | `-3.0` | Threshold for geometric solar-center elevation |
| `level0.dark_current.max_association_hours` | `12.0` | Maximum distance between a dark acquisition and nearest session |
| `level0.surface_weather.missing_policy` | `nan` | Keep missing Open-Meteo hourly fields missing; `fail` rejects unavailable required hours |

These are selection and QA policies, not measured properties of the lidar. `station.yaml` supplies the date-resolved instrument facts.

## Level 1 — signal and molecular state

| Setting | Distributed value | Meaning |
| --- | --- | --- |
| `level1.background.start_altitude_m / stop_altitude_m` | `29000 / 29999` | Search interval used for background estimation |
| `level1.background.estimator` | `median_mad` | Robust background estimator |
| `level1.background.minimum_valid_bins` | `8` | Required valid input bins |
| `level1.background.outlier_sigma` | `4.5` | Robust outlier rejection threshold |
| `level1.photon_counting.deadtime_min_denominator` | `0.05` | Numerical guard during dead-time correction, **not** independent detector calibration |
| `level1.pbl.reference_channel` | `532.AN` | PBL diagnostic channel |
| `level1.pbl.min_search_altitude_m / max_search_altitude_m` | `500 / 4000` | PBL search interval |
| `level1.pbl.smooth_bins` | `15` | PBL smoothing width |
| `level1.missing_channel_calibration.policy` | `neutral_with_warning` | Keep historical processing possible with warning instead of inventing a measured calibration |

The neutral calibration values are `deadtime_us: 0`, `bin_shift_bins: 0`, `background_offset: 0`. They are not scientific measurements.

### Atmosphere

- `time_resolution_minutes: 60`.
- `source_priority: ["era5", "radiosonde", "ussa76"]`.
- `external_profile_outside_coverage: "ussa76"`, identifying vertical extension.
- `radiosonde.synoptic_hours_utc: [0, 12]`, `selection: nearest`, normal `max_time_delta_hours: 6.0`.
- ERA5 pressure-level dataset, grid and pressure levels are explicit under `level1.atmosphere.era5`.
- The operational queue threshold is controlled by the CLI; it is **not** a scientific-quality score.
- Explicit radio-only and congested modes use a 12-hour observation selection tolerance.

See [Atmospheric profiles](atmospheric_profiles.md).

## Level 2 — elastic aerosol retrieval

| Setting | Distributed value | Scientific effect |
| --- | --- | --- |
| `inversion.wavelengths_to_process` | `[355,532]` | Elastic channels requested, nm |
| `inversion.block_average_minutes` | `20` | Temporal block averaging |
| `inversion.monte_carlo_iterations` | `150` | Number of stochastic retrieval realizations |
| `inversion.random_seed` | `143` | Reproducibility of stochastic sampling |
| `inversion.beta_ref_relative_std` | `0.0` | Additional reference-β perturbation only if independently supported |
| `inversion.aerosol_ref_fraction` | `0.0` | Initial reference aerosol fraction assumption |
| `inversion.min_lidar_ratio_sr` | `10.0` | Numerical lower bound, sr |
| `inversion.allow_negative_aerosol` | `false` | Treatment of negative retrieved aerosol |
| `inversion.molecular_fit` | Explicit altitude/range/slope/variance/fraction settings | Selection of the molecular reference |
| `inversion.retrieval.reference_search_ranges_m` | `[10000,15000]` then `[5000,20000]` | Ordered reference search ranges |
| `inversion.retrieval.path_start_altitude_m` | `600` | Lower intended path limit (not proof of valid overlap) |
| `inversion.retrieval.progressive_grid_schedule` | Altitude-dependent bins | Retrieval vertical sampling |
| `inversion.gluing` | Explicit correlation/saturation/fit criteria | Analog/PC joining or permitted single-channel fallback |
| `inversion.products` | Individual save flags | Which Level 2 fields are stored |

The SPU lidar-ratio climatology is in `station.yaml`, not duplicated in the normal `config.yaml`. Its original derivation is an open provenance question; see [lidar-ratio evidence](lidar_ratio_climatology_provenance.md).

**Cloud screening:** `inversion.cloud_screening.enabled: false` in the distributed recipe. A preliminary utility module exists, but it should **not** be described as an active routine cloud classifier. Its future status is under cleanup review.

## Figures and SCC

`visualization` selects image formats, logos, plotted channels, altitude ranges, atmospheric comparison bands and display downsampling. None of these settings should change the scientific NetCDF resolution.

`scc.enabled: false` means no SCC package staging by default. When enabled, the `scc.yaml` catalog provides SCC instrument/product mapping. Actual remote upload requires its own explicit command.

## Before changing a scientific setting

Record the setting's original and proposed values, physical justification, affected sessions, expected changes in uncertainty/valid range, the NetCDF/provenance used to compare results, and whether L1/L2 must be regenerated. A numerical setting is part of the scientific method, not only a user preference.


## Detailed Level 2 options in the distributed recipe

The following settings are *scientific analysis choices*. Their configured values are not inferred from the current aerosol observation.

| Key suffix under `inversion` | Distributed value | Purpose |
| --- | --- | --- |
| `products.save_glued_signal` | `true` | Preserve selected/glued input for retrieval review |
| `products.save_molecular_profile` | `true` | Save Rayleigh/molecular terms |
| `products.save_scattering_ratio` | `false` | Save this optional diagnostic only if requested |
| `products.save_aerosol_backscatter` | `true` | Write aerosol β |
| `products.save_aerosol_extinction` | `true` | Write conditional aerosol α |
| `products.save_uncertainty` | `true` | Preserve uncertainty products |
| `products.save_quality_flags` | `true` | Retain scientific masks and retrieval status |
| `retrieval.residual_aerosol_fractions` | `[0.0]` | Tested residual-aerosol fractions at the reference |
| `retrieval.uncertainty_mode` | `independent` | Explicit uncertainty aggregation semantics |
| `molecular_fit.ref_alt_min_m / ref_alt_max_m` | `5000 / 25000` | Molecular-reference search support |
| `molecular_fit.ref_window_m` | `1000` | Width of reference candidate |
| `molecular_fit.max_relative_slope` | `0.25` | Screening criterion for trend |
| `molecular_fit.max_relative_variance` | `0.50` | Screening criterion for fluctuations |
| `molecular_fit.min_valid_fraction` | `0.50` | Minimum candidate support |
| `gluing.window_length_bins` | `120` | Local analog/PC fit window |
| `gluing.correlation_threshold` | `0.95` | Minimum correlation |
| `gluing.intercept_threshold` | `5.0` | Fit intercept screening |
| `gluing.minmax_threshold` | `0.05` | Ratio of dynamic signal extrema |
| `gluing.max_relative_rmse` | `0.08` | Maximum fit error |
| `gluing.max_relative_bias` | `0.05` | Maximum relative bias |
| `gluing.min_valid_fraction` | `0.80` | Minimum usable gluing samples |
| `gluing.max_saturation_fraction` | `0.20` | Limit for candidate gluing |
| `gluing.invalid_saturation_fraction` | `1.0` | Explicit invalid-saturation behavior |
| `gluing.search_min_idx / search_max_idx` | `150 / 1200` | Candidate bin search interval |
| `gluing.allow_single_channel_fallback` | `true` | Use eligible single channel when gluing fails |
| `gluing.single_channel_priority` | `photon_counting` | Preference for fallback channel |
| `cloud_screening.enabled` | `false` | Preliminary method is disabled in the standard recipe |

The progressive altitude grid from the distributed `inversion.retrieval.progressive_grid_schedule` has bin widths 7.5 m (at low altitude), 15 m from 6 km, 30 m from 10 km, 60 m from 15 km and 100 m from 25 km. These are configured **numerical sampling** steps, not claims of independent instrument-resolved information at those heights.

### Plot settings

The distributed `visualization` section uses `output_format: webp`, `dpi: 120`, plotted altitude ranges `[5, 15, 30]` km and selected channels (1064, 532 and 355 nm) as applicable. The `quicklook` presentation controls include `show_pbl`, `show_tropopause`, `mean_profile_smooth_bins: 20`, maximum time gap of 10 minutes for display, 720 time samples and 1,200 altitude bins.

Atmospheric plots have a maximum plotted height of 30 km and use the configured comparison altitude bands (0–3, 3–10 and 10–20 km). Level 2 plots additionally control optional gluing, reference, Monte Carlo and optical-profile panels. Display downsampling and palettes **do not** change the values in the saved scientific NetCDF.

### Station history and valid scientific context

The distributed SPU station file currently identifies the following *dated catalog periods*. They are descriptive catalog entries, **not independent proof that every historical calibration has been experimentally verified**.

| Station profile | Catalog period |
| --- | --- |
| `spu-legacy` | 2000-01-01 through 2017-03-07 |
| `spu-apel-2017` | 2017-03-08 through 2018-03-18 |
| `spu-raman-2018` | 2018-03-19 through 2024-09-09 |
| `spu-merionc-2024` | From 2024-09-10 |

Profile selection uses the station's local civil dates. `station.yaml` also distinguishes lidar site coordinates, separate radiosonde-launch coordinates, calibration IDs, nominal laser configuration, and overlap-model uncertainty/status. Historical calibration fields may be estimated, provisional or not characterized; the corresponding scientific limitation should remain visible.

### SCC operator policy (optional)

The distributed `scc.enabled` default is `false`; `scc.config_file` points to `scc.yaml`, and `scc.state_dir` to `.state/scc` (a **persistent remote-submission journal**, not disposable cache). HTTP settings specify a 60-second timeout and two safe GET retries; upload/delete POST operations are **not** automatically retried.

With SCC enabled, `scc.yaml` defines the endpoint, SCC configuration/product IDs and channel maps by historical station profile and solar regime. Its `scc_policy.sounding.source: cached_observation` permits **verified Wyoming or IGRA2 only**, with `missing_policy: skip` for absent observations. Standard USSA76 or ERA5 atmospheric profiles are not fabricated into a measured SCC sounding.

The SCC catalog and the scientific instrument `station.yaml` have different meanings; a server product ID is not evidence of detector calibration.

See [SCC operator guide](scc_batch_operator_guide.md) and [Lidar-ratio provenance](lidar_ratio_climatology_provenance.md).

## Valid ranges and invalid settings

MILGRAU's Level 0 configuration resolver explicitly validates required sections, type, finite/positive numeric values, station fields and supported enum options. Level 1 and Level 2 have their own strict validators. **Do not assume that arbitrary percentages, new mode names or missing fields will be silently accepted.** Use the documented default keys, `--help` and program validation output; do not mix values from an older experimental YAML branch into the current recipe.

For a scientifically meaningful parameter-sensitivity study, save the original YAML, record a specific hypothesis for each change and review its consequences in the Level 1/Level 2 uncertainties and validity masks.
