# MILGRAU scientific and user documentation

MILGRAU turns atmospheric lidar observations into progressively more interpreted scientific products. This guide is organized **by what a researcher needs to do or understand**, from a raw Licel record to aerosol optical retrieval.

## Begin here

1. [Getting started](getting_started.md) — install MILGRAU, supply scientific settings, process one session and inspect results.
2. [Processing levels](processing_levels.md) — understand what Level 0, Level 1 and Level 2 mean.
3. [Scientific quality and interpretation](scientific_quality.md) — check whether a result is scientifically supported before using it.

## Processing and interpretation

| Guide | What it answers |
| --- | --- |
| [LIBIDS / Level 0](libids.md) | Which measurements were included? Which instrument and dark-current data were associated? |
| [LIPANCORA / Level 1](lipancora.md) | How were signals corrected, uncertainties estimated and molecular atmosphere represented? |
| [Atmospheric profiles](atmospheric_profiles.md) | When does MILGRAU use ERA5, Wyoming, IGRA2 or USSA76, and what was actually used? |
| [LEBEAR / Level 2](lebear.md) | How are aerosol optical coefficients retrieved, with what assumptions and valid support? |
| [Level 2 variables](level2_schema.md) | What does each dimension, diagnostic and product field mean? |
| [SCC workflow](scc_batch_operator_guide.md) | How are SCC packages prepared, checked, sent and downloaded? |
| [Explorer](explorer.md) | How do I inspect scientific NetCDF results without changing them? |

## Scientific references and specialist notes

- [Scientific traceability and literature](scientific_traceability.md): method interpretation, principal physical references and open validation questions.
- [Lidar-ratio climatology provenance](lidar_ratio_climatology_provenance.md): limits of the legacy empirical monthly values.
- [Overlap model](overlap_model.md): current geometric assumptions and evidence limits.
- [Raman-channel evidence](spu_raman_channel_provenance.md): channel identification and outstanding instrument facts.
- [NetCDF metadata policy](netcdf_metadata_policy.md): provenance and metadata conventions.

## Supporting development, operations and historical records

The following are useful to maintainers or for rare troubleshooting, but are **not required to follow the normal scientific workflow**: [configuration ownership](configuration_ownership.md), [code inventory](code_inventory.md), [SCC protocol details](scc_native_client.md), [SCC Level 0 interoperability](scc_level0_interoperability.md), [SCC validation evidence](scc_phase5_validation.md), [integration gates](scc_integration_merge_plan.md), [archived SCC development history](history/scc_integration_development_tracker.md) and [publication integration](publication_integration.md).

The [project tracker](../.github/tracker.md) records open scientific validation tasks and future documentation work. A checked item indicates a documented/reviewed task, not proof that a physical or scientific uncertainty is resolved.
